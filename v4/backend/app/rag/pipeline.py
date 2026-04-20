"""Pipeline RAG complet de bout-en-bout.

    user_query
        -> QueryAnalyzer (LLM, JSON schema)        # filtres + reformulation
        -> HybridRetriever (pgvector + tsvector)   # candidats fusionnés RRF
        -> Reranker (Cohere ou BGE local)          # cross-encoder
        -> ResponseGenerator (LLM, JSON schema)    # réponse + grounding
        -> validate_grounding                      # filtre des phrases non sourcées
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from .. import sessions as sessions_store
from ..llm.query_understanding import QueryAnalyzer, QueryPlan
from ..llm.response_generator import GroundedAnswer, ResponseGenerator
from ..settings import get_settings
from .embeddings import OpenAIEmbeddings
from .reranker import Reranker, build_default_reranker
from .retriever import HybridRetriever, Hit
from .reranker import RerankedHit

log = structlog.get_logger(__name__)

_STOPWORDS_FR = {
    "avec",
    "pour",
    "dans",
    "une",
    "des",
    "les",
    "du",
    "de",
    "la",
    "le",
    "un",
    "est",
    "sont",
    "que",
    "qui",
    "recette",
    "recettes",
    "comment",
    "fait",
    "faire",
    "donne",
    "donne-moi",
    "moi",
    "sur",
    "plus",
}


def _retrieval_fallback_queries(user_query: str, plan: QueryPlan) -> list[str]:
    """Requêtes plus courtes / centrées ingrédient si la recherche hybride ne renvoie rien."""
    seen: set[str] = set()
    out: list[str] = []
    uq = (user_query or "").strip()

    def add(s: str) -> None:
        s = " ".join(s.split())
        if len(s) < 2 or s in seen:
            return
        seen.add(s)
        out.append(s)

    for slug in plan.ingredient_slugs:
        if not slug:
            continue
        w = slug.replace("-", " ").strip()
        add(w)
        if w:
            add(f"{w} recette")
    for w in re.findall(r"\b[\wàâäéèêëïîôùûç]+\b", uq.lower()):
        if w not in _STOPWORDS_FR and len(w) > 2:
            add(w)
            if len(w) > 4:
                add(f"{w} recette")
    rw = (plan.rewritten_query or "").strip()
    if rw and rw.casefold() != uq.casefold():
        add(rw)
    return out[:10]


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

    async def _retrieve_hits(
        self,
        session: AsyncSession,
        user_query: str,
        plan: QueryPlan,
        q: str,
        rerank_top_n: int,
        exclude_article_external_ids: list[int],
    ) -> list[Hit]:
        """Recherche hybride + retries (filtres, fallbacks)."""
        final_limit = max(30, rerank_top_n * 4)
        excl = exclude_article_external_ids

        hits = await self.retriever.search(
            session,
            q,
            chef_slugs=plan.chef_slugs,
            ingredient_slugs=plan.ingredient_slugs,
            category_slugs=plan.category_slugs,
            keyword_slugs=plan.keyword_slugs,
            final_limit=final_limit,
            exclude_article_external_ids=excl,
        )
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
                final_limit=final_limit,
                exclude_article_external_ids=excl,
            )
        if not hits:
            base = q.strip()
            for alt in _retrieval_fallback_queries(user_query, plan):
                if alt.strip() == base:
                    continue
                try:
                    hits = await self.retriever.search(
                        session,
                        alt,
                        chef_slugs=[],
                        ingredient_slugs=[],
                        category_slugs=[],
                        keyword_slugs=[],
                        final_limit=final_limit,
                        exclude_article_external_ids=excl,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "rag.pipeline.retrieval_fallback_failed",
                        alt=alt[:80],
                        error=str(exc),
                    )
                    continue
                if hits:
                    log.info(
                        "rag.pipeline.retrieval_fallback_hit",
                        alt_preview=alt[:80],
                        n_hits=len(hits),
                    )
                    break
        return hits

    async def answer(
        self,
        session: AsyncSession,
        user_query: str,
        *,
        rerank_top_n: int | None = None,
        session_id: str | None = None,
        conversation_history: str | None = None,
    ) -> PipelineResult:
        timings: dict[str, int] = {}
        rerank_top_n = rerank_top_n or self.settings.rag_rerank_top_k

        t0 = time.perf_counter()
        try:
            plan = await self.analyzer.analyze(user_query)
        except Exception as exc:  # noqa: BLE001
            log.warning("rag.pipeline.query_analyze_failed_fallback", error=str(exc))
            plan = QueryPlan(
                rewritten_query=(user_query or "").strip(),
                intent="mixed",
                chef_slugs=[],
                ingredient_slugs=[],
                category_slugs=[],
                keyword_slugs=[],
                focus_section_kinds=[],
                needs_context_after=False,
            )
        timings["query_understanding_ms"] = int((time.perf_counter() - t0) * 1000)

        t1 = time.perf_counter()
        q = plan.rewritten_query or user_query
        excluded: list[int] = []
        if session_id:
            try:
                excluded = await sessions_store.recent_article_external_ids(session_id)
            except Exception as exc:  # noqa: BLE001
                log.warning("rag.pipeline.session_exclusions_failed", error=str(exc))
        try:
            hits = await self._retrieve_hits(
                session, user_query, plan, q, rerank_top_n, excluded
            )
            if not hits and excluded:
                log.info(
                    "rag.pipeline.retrieval_retry_no_session_article_exclusions",
                    n_excluded=len(excluded),
                )
                hits = await self._retrieve_hits(
                    session, user_query, plan, q, rerank_top_n, []
                )
        except Exception as exc:  # noqa: BLE001
            log.exception("rag.pipeline.retrieval_failed")
            hits = []
        timings["retrieval_ms"] = int((time.perf_counter() - t1) * 1000)

        t2 = time.perf_counter()
        rq = plan.rewritten_query or user_query
        try:
            reranked = await self.reranker.rerank(rq, hits, top_n=rerank_top_n)
        except Exception as exc:  # noqa: BLE001
            # Cohere indispo, timeout, quota, etc. — on continue sans rerank.
            log.warning("rag.pipeline.rerank_failed_fallback", error=str(exc))
            reranked = [
                RerankedHit(hit=h, rerank_score=float(h.score_rrf or 0.0))
                for h in hits[:rerank_top_n]
            ]
        reranked = [
            r for r in reranked if r.rerank_score >= self.settings.rag_min_rerank_score
        ] or reranked[: max(1, rerank_top_n // 2)]
        timings["rerank_ms"] = int((time.perf_counter() - t2) * 1000)

        t3 = time.perf_counter()
        answer = await self.generator.generate(
            user_query,
            reranked,
            conversation_history=conversation_history,
        )
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
