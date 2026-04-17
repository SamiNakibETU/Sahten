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
import unicodedata
from functools import lru_cache
from dataclasses import dataclass
from typing import Optional, List

from .core.config import get_settings
from .core.model_selector import get_model_for_request
from .llm.query_analyzer import QueryAnalyzer
from .llm.response_generator import ResponseGenerator, EXACT_ALTERNATIVE_HOOK
from .rag.retriever import HybridRetriever
from .rag.session_manager import SessionManager, is_continuation, format_session_hints_for_analyzer
from .schemas.responses import RecipeNarrative, SahtenResponse, RecipeCard


def _normalize_for_compare(text: str) -> str:
    """Normalize text for dish name matching: lowercase, no accents, no punctuation."""
    nfd = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn").strip()


def _dish_found_in_results(
    dish_name: str,
    dish_name_variants: Optional[List[str]],
    recipes: List[RecipeCard],
) -> bool:
    """Return True if at least one recipe clearly matches the queried dish name."""
    if not dish_name or not recipes:
        return True  # Cannot determine → assume match, don't show alternative hook
    norm_dish = _normalize_for_compare(dish_name)
    variants = [norm_dish] + [
        _normalize_for_compare(v) for v in (dish_name_variants or []) if v
    ]
    variants = [v for v in variants if len(v) >= 3]
    if not variants:
        return True
    for recipe in recipes:
        title_norm = _normalize_for_compare(recipe.title or "")
        if any(v in title_norm for v in variants):
            return True
    return False

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
        
        # 1) Analyze query — pass session context for continuation resolution
        # (e.g. "encore une autre" understood as "encore une autre recette à la tomate")
        session_hint_for_analyzer: Optional[str] = None
        if session and session.has_history():
            ctx = session.get_context_for_continuation()
            session_hint_for_analyzer = format_session_hints_for_analyzer(ctx)

        analysis = await analyzer.analyze(message, session_hint=session_hint_for_analyzer)

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
        # Exclude recipes already proposed in this session (by URL and by title)
        exclude_urls = session.recipes_proposed if session else []
        exclude_titles: set[str] = set()
        if session:
            for turn in session.conversation_history:
                for t in turn.recipe_titles:
                    if t:
                        exclude_titles.add(_normalize_for_compare(t))

        recipes, is_base2, retrieval_debug = await self.retriever.search_with_rerank(
            analysis,
            raw_query=message,
            debug=debug,
            exclude_urls=exclude_urls,
            exclude_titles=exclude_titles if exclude_titles else None,
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
                            recipe_titles=[recipe_card.title] if recipe_card.title else [],
                        )
                        self.session_manager.save(session)

            if not recipes:
                result = self.retriever.get_olj_recommendation_by_ingredient(
                    analysis,
                    message,
                    exclude_urls=exclude_set,
                    exclude_titles=exclude_titles,
                )
                if result:
                    recipe_card, matched_ingredient = result

                    # For ingredient queries, only accept the fallback if there is
                    # a proven shared ingredient — otherwise we'd return an unrelated
                    # recipe (e.g. lentil soup for a potato query).
                    if analysis.intent == "recipe_by_ingredient" and matched_ingredient is None:
                        result = None  # discard irrelevant fallback

                if result:
                    recipe_card, matched_ingredient = result
                    is_specific = analysis.intent == "recipe_specific"
                    if is_specific:
                        hook = EXACT_ALTERNATIVE_HOOK
                        cultural_context = (
                            f"{recipe_card.title} est la recette la plus proche que j'ai trouvée. "
                            "Elle partage des saveurs libanaises avec votre demande."
                        )
                    elif matched_ingredient:
                        ing_label = matched_ingredient
                        hook = (
                            f"Je n'ai plus d'autre recette à la {ing_label} dans mon répertoire "
                            "pour l'instant, mais voici une suggestion qui pourrait vous plaire :"
                        )
                        cultural_context = (
                            f"{recipe_card.title} utilise également {ing_label} et s'inscrit dans "
                            "la belle tradition de la cuisine du Levant."
                        )
                    else:
                        hook = (
                            "Je n'ai pas d'autre recette correspondant exactement à votre demande, "
                            "mais voici une suggestion proche :"
                        )
                        cultural_context = (
                            f"{recipe_card.title} — une recette libanaise qui pourrait vous plaire."
                        )
                    narrative = RecipeNarrative(
                        hook=hook,
                        cultural_context=cultural_context,
                        teaser=None,
                        cta="Retrouvez la recette complète sur L'Orient-Le Jour.",
                        closing="Sahteïn !",
                    )
                    recipes = [recipe_card]
                    response_type = "recipe_base2" if recipe_card.source == "base2" else "recipe_olj"
                    if session:
                        session.add_turn(
                            user_message=message,
                            intent=analysis.intent,
                            primary_dish=analysis.dish_name,
                            ingredients=analysis.ingredients,
                            recipe_url=recipe_card.url or None,
                            response_summary="1 recette alternative proposée",
                            recipe_titles=[recipe_card.title] if recipe_card.title else [],
                        )
                        self.session_manager.save(session)

            if not recipes:
                # Build a contextual "no results" message based on what was asked
                ing_list = analysis.ingredients or []
                if analysis.intent == "recipe_by_ingredient" and ing_list:
                    ing_str = ", ".join(ing_list[:3])
                    no_result_hook = f"Je n'ai plus d'autres recettes à base de {ing_str} dans mon répertoire pour l'instant."
                    no_result_body = "N'hésitez pas à me demander un autre ingrédient ou une envie du moment — légère, réconfortante, rapide — et je vous trouverai quelque chose."
                else:
                    no_result_hook = "Je n'ai pas trouvé de recette correspondant exactement à votre demande."
                    no_result_body = (
                        "Précisez un ingrédient, un plat ou une envie "
                        "(léger, réconfortant, rapide…) et je vous proposerai une recette adaptée."
                    )
                narrative = RecipeNarrative(
                    hook=no_result_hook,
                    cultural_context=no_result_body,
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
        # Detect if the result is an alternative (not an exact match for the queried dish)
        is_alternative_response = (
            analysis.intent == "recipe_specific"
            and bool(analysis.dish_name)
            and bool(recipes)
            and not is_base2
            and not _dish_found_in_results(
                analysis.dish_name, analysis.dish_name_variants, recipes
            )
        )

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

        # Force the contractual alternative hook if the recipe is not an exact match
        if is_alternative_response and isinstance(narrative, RecipeNarrative):
            narrative = RecipeNarrative(
                hook=EXACT_ALTERNATIVE_HOOK,
                cultural_context=narrative.cultural_context,
                teaser=narrative.teaser,
                cta=narrative.cta,
                closing=narrative.closing,
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
        
        # Save session with new recipes (URL + titles for deduplication)
        if session and recipes:
            primary_url = recipes[0].url if recipes else None
            session.add_turn(
                user_message=message,
                intent=analysis.intent,
                primary_dish=analysis.dish_name,
                ingredients=analysis.ingredients,
                recipe_url=primary_url,
                response_summary=f"{len(recipes)} recettes proposées",
                recipe_titles=[r.title for r in recipes if r.title],
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
