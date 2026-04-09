"""
Sahteïn Bot (v2.1)
=================

The core application logic for Sahteïn.
Implements the durable RAG pipeline:
  1) QueryAnalyzer (LLM): safety + intent + filters
  2) HybridRetriever: retrieve (hybrid) -> rerank (LLM) -> select
  3) ResponseGenerator (LLM): narrative generation

Supports flexible model selection via:
  - Environment variable (OPENAI_MODEL)
  - API request parameter
  - A/B testing

Session Memory:
  - Short-term memory via SessionManager (TTL 30 min)
  - Avoids re-proposing same recipes
  - Tracks conversation context for continuations
"""

from __future__ import annotations

import logging
from functools import lru_cache
from dataclasses import dataclass
from typing import Optional, List

from .core.config import get_settings
from .core.model_selector import get_model_for_request
from .llm.query_analyzer import QueryAnalyzer
from .llm.response_generator import ResponseGenerator
from .rag.retriever import HybridRetriever
from .rag.session_manager import SessionManager, is_continuation
from .schemas.responses import RecipeNarrative, SahtenResponse, RecipeCard

logger = logging.getLogger(__name__)


@dataclass
class SahtenConfig:
    enable_safety_check: bool = True
    enable_narrative_generation: bool = True


class SahtenBot:
    """
    Main Sahteïn chatbot.
    
    Supports dynamic model selection per-request.
    Includes session memory to avoid repeating recipes.
    """
    
    def __init__(self, config: Optional[SahtenConfig] = None):
        self.config = config or SahtenConfig()
        settings = get_settings()
        
        # Default model from settings
        self.default_model = settings.openai_model
        
        # Initialize components with default model
        self.analyzer = QueryAnalyzer(model=self.default_model)
        self.retriever = HybridRetriever()
        self.generator = ResponseGenerator(model=self.default_model)
        
        # Session memory (TTL 30 min, max 2000 sessions)
        self.session_manager = SessionManager(max_sessions=2000, ttl_minutes=30)

        logger.info("SahtenBot initialized (default_model=%s, session_memory=enabled)", self.default_model)

    def _get_analyzer(self, model: str) -> QueryAnalyzer:
        """Get analyzer for specific model (cached or new)."""
        if model == self.default_model:
            return self.analyzer
        return QueryAnalyzer(model=model)
    
    def _get_generator(self, model: str) -> ResponseGenerator:
        """Get generator for specific model (cached or new)."""
        if model == self.default_model:
            return self.generator
        return ResponseGenerator(model=model)

    def _mood_description(self, analysis) -> str:
        """Return a short human-readable description for the detected mood tags."""
        tag_labels = {
            "leger":        "légère et fraîche ",
            "rapide":       "rapide et simple ",
            "reconfortant": "réconfortante et chaleureuse ",
            "frais":        "fraîche et estivale ",
            "hiver":        "chaleureuse pour l'hiver ",
            "ete":          "estivale et légère ",
            "festif":       "festive et généreuse ",
            "traditionnel": "traditionnelle et authentique ",
            "convivial":    "conviviale à partager ",
            "facile":       "facile à préparer ",
            "copieux":      "copieuse et généreuse ",
        }
        tags = getattr(analysis, "mood_tags", []) or []
        labels = [tag_labels[t] for t in tags if t in tag_labels]
        return labels[0] if labels else ""

    async def chat(
        self,
        message: str,
        *,
        debug: bool = False,
        model: Optional[str] = None,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> tuple[SahtenResponse, Optional[dict], dict]:
        """
        Process a chat message.
        
        Args:
            message: User's message
            debug: Include debug info in response
            model: Override model selection (None = use default/A/B)
            request_id: Request ID for A/B testing
            session_id: Session ID for conversation memory
        
        Returns:
            (SahtenResponse, debug_info, trace_meta)
        """
        # Determine model to use
        effective_model = get_model_for_request(
            request_id or "default",
            model
        )
        
        # Get components for this model
        analyzer = self._get_analyzer(effective_model)
        generator = self._get_generator(effective_model)
        
        # Get or create session for memory
        session = None
        if session_id:
            session = self.session_manager.get_or_create(session_id)
            logger.debug("Session %s: %d turns, %d recipes proposed", 
                        session_id, len(session.conversation_history), len(session.recipes_proposed))
        
        # 1) Analyze query
        analysis = await analyzer.analyze(message)

        if not analysis.safety.is_safe:
            narrative = generator.generate_redirect(analysis.redirect_suggestion)
            return (
                SahtenResponse(
                    response_type="redirect",
                    narrative=narrative,
                    recipes=[],
                    recipe_count=0,
                    intent_detected=analysis.intent,
                    confidence=analysis.intent_confidence,
                    model_used=effective_model,
                ),
                {"analysis": analysis.model_dump(), "model": effective_model} if debug else None,
                {},
            )

        if analysis.intent == "greeting":
            return (
                SahtenResponse(
                    response_type="greeting",
                    narrative=generator.generate_greeting(),
                    recipes=[],
                    recipe_count=0,
                    intent_detected=analysis.intent,
                    confidence=analysis.intent_confidence,
                    model_used=effective_model,
                ),
                {"analysis": analysis.model_dump(), "model": effective_model} if debug else None,
                {},
            )

        if analysis.intent == "off_topic" or not analysis.is_culinary:
            narrative = generator.generate_redirect(analysis.redirect_suggestion)
            return (
                SahtenResponse(
                    response_type="redirect",
                    narrative=narrative,
                    recipes=[],
                    recipe_count=0,
                    intent_detected=analysis.intent,
                    confidence=analysis.intent_confidence,
                    model_used=effective_model,
                ),
                {"analysis": analysis.model_dump(), "model": effective_model} if debug else None,
                {},
            )

        if analysis.intent == "clarification":
            term = analysis.dish_name or message
            return (
                SahtenResponse(
                    response_type="clarification",
                    narrative=generator.generate_clarification(term),
                    recipes=[],
                    recipe_count=0,
                    intent_detected=analysis.intent,
                    confidence=analysis.intent_confidence,
                    model_used=effective_model,
                ),
                {"analysis": analysis.model_dump(), "model": effective_model} if debug else None,
                {},
            )

        # 2) Retrieve + rerank
        # Exclude recipes already proposed in this session
        exclude_urls = session.recipes_proposed if session else []
        
        recipes, is_base2, retrieval_debug = await self.retriever.search_with_rerank(
            analysis, raw_query=message, debug=debug, exclude_urls=exclude_urls
        )

        if not recipes:
            exclude_set = set(session.recipes_proposed) if session else None

            # For mood/vague queries: try a category-based fallback pick
            if analysis.intent in ("recipe_by_mood", "recipe_by_diet", "multi_recipe"):
                recipe_card = self.retriever.get_mood_fallback_recipe(
                    analysis, exclude_urls=exclude_set
                )
                if recipe_card:
                    mood_desc = self._mood_description(analysis)
                    narrative = RecipeNarrative(
                        hook=f"Je n'ai pas trouvé exactement ce que vous cherchez, mais voici une idée {mood_desc}qui pourrait vous plaire.",
                        cultural_context=(
                            f"{recipe_card.title} — une recette libanaise du corpus de L'Orient-Le Jour. "
                            "La cuisine du Levant a toujours une belle surprise en réserve !"
                        ),
                        teaser="Cliquez pour découvrir la recette complète.",
                        cta="Explorez nos recettes sur L'Orient-Le Jour",
                        closing="Sahteïn !",
                    )
                    recipes = [recipe_card]
                    response_type = "recipe_olj"
                    if session and recipe_card.url:
                        session.add_turn(
                            user_message=message,
                            intent=analysis.intent,
                            primary_dish=recipe_card.title,
                            ingredients=analysis.ingredients,
                            recipe_url=recipe_card.url,
                            response_summary="1 recette mood proposée",
                        )
                        self.session_manager.save(session)

            if not recipes:
                result = self.retriever.get_olj_recommendation_by_ingredient(
                    analysis, message, exclude_urls=exclude_set
                )
                if result:
                    recipe_card, matched_ingredient = result
                    if matched_ingredient:
                        cultural_context = (
                            f"une recette libanaise qui partage {matched_ingredient} avec votre demande : "
                            f"{recipe_card.title}. Un plat typique du Levant !"
                        )
                    else:
                        cultural_context = (
                            f"une recette libanaise qui pourrait vous plaire : {recipe_card.title}. "
                            "La cuisine du Levant offre plein de surprises !"
                        )
                    narrative = RecipeNarrative(
                        hook="Je n'ai pas cette recette exacte dans mes carnets, mais voici ce que je vous propose :",
                        cultural_context=cultural_context,
                        teaser="Cliquez pour découvrir la recette complète.",
                        cta="Explorez nos recettes sur L'Orient-Le Jour",
                        closing="Sahteïn !",
                    )
                    recipes = [recipe_card]
                    response_type = "recipe_base2" if recipe_card.source == "base2" else "recipe_olj"
                    if session and recipe_card.url:
                        session.add_turn(
                            user_message=message,
                            intent=analysis.intent,
                            primary_dish=analysis.dish_name,
                            ingredients=analysis.ingredients,
                            recipe_url=recipe_card.url,
                            response_summary="1 recette alternative proposée",
                        )
                        self.session_manager.save(session)

            if not recipes:
                narrative = RecipeNarrative(
                    hook="Je n'ai pas trouvé de recette correspondant exactement à votre demande.",
                    cultural_context=(
                        "La cuisine libanaise est très riche ! Précisez un ingrédient, un plat ou une envie "
                        "(léger, réconfortant, rapide…) et je vous proposerai une recette adaptée."
                    ),
                    teaser=None,
                    cta="Explorez nos recettes sur L'Orient-Le Jour",
                    closing="Sahteïn !",
                )
                recipes = []
                response_type = "redirect"
            turn_count = len(session.conversation_history) if session else 0
            return (
                SahtenResponse(
                    response_type=response_type,
                    narrative=narrative,
                    recipes=recipes,
                    olj_recommendation=None,
                    recipe_count=len(recipes),
                    intent_detected=analysis.intent,
                    confidence=analysis.intent_confidence,
                    model_used=effective_model,
                ),
                (
                    {"analysis": analysis.model_dump(), "retrieval": retrieval_debug, "model": effective_model}
                    if debug
                    else None
                ),
                {"session_turn_count": turn_count},
            )

        # 3) Generate narrative
        if self.config.enable_narrative_generation:
            narrative = await generator.generate_narrative(
                user_query=message,
                analysis=analysis,
                recipes=recipes,
                is_base2_fallback=is_base2,
            )
        else:
            narrative = generator.generate_redirect(
                "Mode sans génération narrative activé."
            )

        olj_reco = None
        if is_base2:
            olj_reco = self.retriever.get_olj_recommendation(
                analysis, exclude_titles=[r.title for r in recipes]
            )

        if analysis.intent == "menu_composition":
            response_type = "menu"
        elif is_base2:
            response_type = "recipe_base2"
        else:
            response_type = "recipe_olj"

        resp = SahtenResponse(
            response_type=response_type,
            narrative=narrative,
            recipes=recipes,
            olj_recommendation=olj_reco,
            recipe_count=len(recipes),
            intent_detected=analysis.intent,
            confidence=analysis.intent_confidence,
            model_used=effective_model,
        )
        
        # Save session with new recipes
        if session and recipes:
            primary_url = recipes[0].url if recipes else None
            session.add_turn(
                user_message=message,
                intent=analysis.intent,
                primary_dish=analysis.dish_name,
                ingredients=analysis.ingredients,
                recipe_url=primary_url,
                response_summary=f"{len(recipes)} recettes proposées",
            )
            self.session_manager.save(session)
        
        dbg = (
            {"analysis": analysis.model_dump(), "retrieval": retrieval_debug, "model": effective_model}
            if debug
            else None
        )
        turn_count = len(session.conversation_history) if session else 0
        trace_meta = {"session_turn_count": turn_count}
        return resp, dbg, trace_meta


# Singleton instance
_bot_instance: Optional[SahtenBot] = None


def get_bot() -> SahtenBot:
    """Get singleton bot instance."""
    global _bot_instance
    if _bot_instance is None:
        _bot_instance = SahtenBot()
    return _bot_instance


def reload_bot() -> SahtenBot:
    """Force reload bot instance (after config change)."""
    global _bot_instance
    _bot_instance = SahtenBot()
    return _bot_instance
