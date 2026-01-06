"""
Sahten Bot (MVP)
================

The core application logic for Sahten.
Implements the durable RAG pipeline:
  1) QueryAnalyzer (LLM): safety + intent + filters
  2) HybridRetriever: retrieve (hybrid) -> rerank (LLM) -> select
  3) ResponseGenerator (LLM): narrative generation

Supports flexible model selection via:
  - Environment variable (OPENAI_MODEL)
  - API request parameter
  - A/B testing
"""

from __future__ import annotations

import logging
from functools import lru_cache
from dataclasses import dataclass
from typing import Optional

from .core.config import get_settings
from .core.model_selector import get_model_for_request
from .llm.query_analyzer import QueryAnalyzer
from .llm.response_generator import ResponseGenerator
from .rag.retriever import HybridRetriever
from .schemas.responses import RecipeNarrative, SahtenResponse

logger = logging.getLogger(__name__)


@dataclass
class SahtenConfig:
    enable_safety_check: bool = True
    enable_narrative_generation: bool = True


class SahtenBot:
    """
    Main Sahten chatbot.
    
    Supports dynamic model selection per-request.
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

        logger.info("SahtenBot initialized (default_model=%s)", self.default_model)

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

    async def chat(
        self,
        message: str,
        *,
        debug: bool = False,
        model: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> tuple[SahtenResponse, Optional[dict]]:
        """
        Process a chat message.
        
        Args:
            message: User's message
            debug: Include debug info in response
            model: Override model selection (None = use default/A/B)
            request_id: Request ID for A/B testing
        
        Returns:
            (SahtenResponse, debug_info)
        """
        # Determine model to use
        effective_model = get_model_for_request(
            request_id or "default",
            model
        )
        
        # Get components for this model
        analyzer = self._get_analyzer(effective_model)
        generator = self._get_generator(effective_model)
        
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
            )

        # 2) Retrieve + rerank
        recipes, is_base2, retrieval_debug = await self.retriever.search_with_rerank(
            analysis, raw_query=message, debug=debug
        )

        if not recipes:
            olj_reco = self.retriever.get_olj_recommendation(analysis)
            narrative = RecipeNarrative(
                hook="Je n'ai pas trouvé exactement ce que tu cherches.",
                cultural_context="Mais la cuisine libanaise est très riche: on peut trouver une alternative qui te plaira.",
                teaser="Dis-moi un ingrédient ou une envie (frais, réconfortant, rapide) et je te propose mieux.",
                cta="Explore nos recettes sur L'Orient-Le Jour",
                closing="Sahten !",
            )
            return (
                SahtenResponse(
                    response_type="redirect",
                    narrative=narrative,
                    recipes=[],
                    olj_recommendation=olj_reco,
                    recipe_count=0,
                    intent_detected=analysis.intent,
                    confidence=analysis.intent_confidence,
                    model_used=effective_model,
                ),
                (
                    {"analysis": analysis.model_dump(), "retrieval": retrieval_debug, "model": effective_model}
                    if debug
                    else None
                ),
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
        dbg = (
            {"analysis": analysis.model_dump(), "retrieval": retrieval_debug, "model": effective_model}
            if debug
            else None
        )
        return resp, dbg


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
