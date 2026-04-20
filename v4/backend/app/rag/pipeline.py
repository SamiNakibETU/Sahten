"""Pipeline RAG complet de bout-en-bout.

    user_query
        -> QueryAnalyzer (LLM, JSON schema)        # filtres + reformulation
        -> HybridRetriever (pgvector + tsvector)   # candidats fusionnés RRF
        -> Reranker (Cohere ou BGE local)          # cross-encoder
        -> ResponseGenerator (LLM, JSON schema)    # réponse + grounding
        -> validate_grounding                      # filtre des phrases non sourcées
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from ..llm.query_understanding import QueryAnalyzer, QueryPlan
from ..llm.response_generator import GroundedAnswer, ResponseGenerator
from ..settings import get_settings
from .embeddings import OpenAIEmbeddings
from .reranker import Reranker, build_default_reranker
from .retriever import HybridRetriever, Hit
from .reranker import RerankedHit

log = structlog.get_logger(__name__)


@dataclass
class PipelineResult:
    plan: QueryPlan
    hits: list[Hit]
    reranked: list[RerankedHit]
    answer: GroundedAnswer
    timings_ms: dict[str, int]


class RagPipeline:
    def __init__(
        self,
        analyzer: QueryAnalyzer | None = None,
        retriever: HybridRetriever | None = None,
        reranker: Reranker | None = None,
        generator: ResponseGenerator | None = None,
    ) -> None:
        self.analyzer = analyzer or QueryAnalyzer()
        self.retriever = retriever or HybridRetriever(OpenAIEmbeddings())
        self.reranker = reranker or build_default_reranker()
        self.generator = generator or ResponseGenerator()
        self.settings = get_settings()

    async def answer(
        self,
        session: AsyncSession,
        user_query: str,
        *,
        rerank_top_n: int | None = None,
    ) -> PipelineResult:
        timings: dict[str, int] = {}
        rerank_top_n = rerank_top_n or self.settings.rag_rerank_top_k

        t0 = time.perf_counter()
        plan = await self.analyzer.analyze(user_query)
        timings["query_understanding_ms"] = int((time.perf_counter() - t0) * 1000)

        t1 = time.perf_counter()
        q = plan.rewritten_query or user_query
        hits = await self.retriever.search(
            session,
            q,
            chef_slugs=plan.chef_slugs,
            ingredient_slugs=plan.ingredient_slugs,
            category_slugs=plan.category_slugs,
            keyword_slugs=plan.keyword_slugs,
            final_limit=max(30, rerank_top_n * 4),
        )
        # Les tables de liaison (article_ingredients, etc.) peuvent être vides même
        # si le texte des chunks mentionne l'ingrédient → filtres structurés = 0
        # candidat. On retombe sur recherche hybride plein texte + vecteur sans filtre.
        if not hits and (
            plan.chef_slugs
            or plan.ingredient_slugs
            or plan.category_slugs
            or plan.keyword_slugs
        ):
            log.warning(
                "rag.pipeline.retrieval_empty_with_filters_retrying_broad",
                ingredient_slugs=plan.ingredient_slugs,
                chef_slugs=plan.chef_slugs,
            )
            hits = await self.retriever.search(
                session,
                q,
                chef_slugs=[],
                ingredient_slugs=[],
                category_slugs=[],
                keyword_slugs=[],
                final_limit=max(30, rerank_top_n * 4),
            )
        timings["retrieval_ms"] = int((time.perf_counter() - t1) * 1000)

        t2 = time.perf_counter()
        reranked = await self.reranker.rerank(
            plan.rewritten_query or user_query, hits, top_n=rerank_top_n
        )
        reranked = [
            r for r in reranked if r.rerank_score >= self.settings.rag_min_rerank_score
        ] or reranked[: max(1, rerank_top_n // 2)]
        timings["rerank_ms"] = int((time.perf_counter() - t2) * 1000)

        t3 = time.perf_counter()
        answer = await self.generator.generate(user_query, reranked)
        timings["generation_ms"] = int((time.perf_counter() - t3) * 1000)
        timings["total_ms"] = sum(timings.values())

        log.info(
            "rag.pipeline.completed",
            query=user_query[:80],
            intent=plan.intent,
            n_hits=len(hits),
            n_reranked=len(reranked),
            timings_ms=timings,
        )
        return PipelineResult(
            plan=plan, hits=hits, reranked=reranked,
            answer=answer, timings_ms=timings,
        )
