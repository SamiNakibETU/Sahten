"""
Sahten Bot (Main Application)
==============================

The core application logic for Sahten.
Implements the durable RAG pipeline:
  1) QueryAnalyzer (LLM): safety + intent + filters
  2) HybridRetriever: retrieve (hybrid) -> rerank (LLM) -> select (dedup/diversity)
  3) ResponseGenerator (LLM): narrative generation

"""

from __future__ import annotations

import logging
from functools import lru_cache
from dataclasses import dataclass
from typing import Optional

from .llm.query_analyzer import QueryAnalyzer
from .llm.response_generator import ResponseGenerator
from .rag.retriever import HybridRetriever
from .schemas.responses import RecipeNarrative, SahtenResponse

logger = logging.getLogger(__name__)


@dataclass
class SahtenConfig:
    enable_safety_check: bool = True
    enable_narrative_generation: bool = True
    model: str = "gpt-4o-mini"


class SahtenBot:
    def __init__(self, config: Optional[SahtenConfig] = None):
        self.config = config or SahtenConfig()

        self.analyzer = QueryAnalyzer(model=self.config.model)
        self.retriever = HybridRetriever()
        self.generator = ResponseGenerator(model=self.config.model)

        logger.info("SahtenBot initialized (model=%s)", self.config.model)

    async def chat(
        self, message: str, *, debug: bool = False
    ) -> tuple[SahtenResponse, Optional[dict]]:
        # 1) Analyze
        analysis = await self.analyzer.analyze(message)

        if not analysis.safety.is_safe:
            narrative = self.generator.generate_redirect(analysis.redirect_suggestion)
            return (
                SahtenResponse(
                    response_type="redirect",
                    narrative=narrative,
                    recipes=[],
                    recipe_count=0,
                    intent_detected=analysis.intent,
                    confidence=analysis.intent_confidence,
                ),
                {"analysis": analysis.model_dump()} if debug else None,
            )

        if analysis.intent == "greeting":
            return (
                SahtenResponse(
                    response_type="greeting",
                    narrative=self.generator.generate_greeting(),
                    recipes=[],
                    recipe_count=0,
                    intent_detected=analysis.intent,
                    confidence=analysis.intent_confidence,
                ),
                {"analysis": analysis.model_dump()} if debug else None,
            )

        if analysis.intent == "off_topic" or not analysis.is_culinary:
            narrative = self.generator.generate_redirect(analysis.redirect_suggestion)
            return (
                SahtenResponse(
                    response_type="redirect",
                    narrative=narrative,
                    recipes=[],
                    recipe_count=0,
                    intent_detected=analysis.intent,
                    confidence=analysis.intent_confidence,
                ),
                {"analysis": analysis.model_dump()} if debug else None,
            )

        if analysis.intent == "clarification":
            term = analysis.dish_name or message
            return (
                SahtenResponse(
                    response_type="clarification",
                    narrative=self.generator.generate_clarification(term),
                    recipes=[],
                    recipe_count=0,
                    intent_detected=analysis.intent,
                    confidence=analysis.intent_confidence,
                ),
                {"analysis": analysis.model_dump()} if debug else None,
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
                ),
                (
                    {"analysis": analysis.model_dump(), "retrieval": retrieval_debug}
                    if debug
                    else None
                ),
            )

        # 3) Narrative
        if self.config.enable_narrative_generation:
            narrative = await self.generator.generate_narrative(
                user_query=message,
                analysis=analysis,
                recipes=recipes,
                is_base2_fallback=is_base2,
            )
        else:
            narrative = self.generator.generate_redirect(
                "Mode sans génération narrative activé. Dis-moi ce que tu cherches et je te propose une recette."
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
        )
        dbg = (
            {"analysis": analysis.model_dump(), "retrieval": retrieval_debug}
            if debug
            else None
        )
        return resp, dbg


@lru_cache(maxsize=1)
def get_bot() -> SahtenBot:
    """Singleton bot instance."""
    return SahtenBot()
