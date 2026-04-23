"""Pipeline RAG complet de bout-en-bout.

    user_query
        -> QueryAnalyzer (LLM) + SessionFocus (LLM, si historique)  # plan + fil
        -> HybridRetriever (pgvector + tsvector)   # RRF, élargissement corpus
        -> Reranker (Cohere ou BGE local)          # cross-encoder
        -> ResponseGenerator (LLM)                  # réponse + grounding
        -> validate_grounding                      # filtre des phrases non sourcées
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import defaultdict
from dataclasses import dataclass

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from .. import sessions as sessions_store
from ..llm.query_understanding import QueryAnalyzer, QueryPlan
from ..llm.session_focus import SessionFocus, SessionFocusAnalyzer
from ..llm.response_generator import GroundedAnswer, ResponseGenerator
from ..settings import get_settings
from .embeddings import OpenAIEmbeddings
from .reranker import Reranker, build_default_reranker
from .retriever import HybridRetriever, Hit
from .reranker import RerankedHit

log = structlog.get_logger(__name__)


def _interleave_hits_by_article(hits: list[Hit]) -> list[Hit]:
    """Mélange les extraits : un chunk par article en tour à tour, ordre d’abord
    d’apparition dans `hits` (souvent score RRF décroissant). Aide le reranker
    à voir plusieurs fiches, pas 10 chunks d’une seule (ex. seulement « sauce »)."""
    if not hits or len(hits) < 2:
        return hits

    by_art: dict[int, list[Hit]] = defaultdict(list)
    for h in hits:
        by_art[h.article_external_id].append(h)
    order: list[int] = []
    seen: set[int] = set()
    for h in hits:
        a = h.article_external_id
        if a not in seen:
            seen.add(a)
            order.append(a)
    buckets = [by_art[aid] for aid in order]
    out: list[Hit] = []
    i = 0
    while any(i < len(b) for b in buckets):
        for b in buckets:
            if i < len(b):
                out.append(b[i])
        i += 1
    return out


def _dedupe_merge_hits(primary: list[Hit], secondary: list[Hit], *, cap: int = 100) -> list[Hit]:
    """Préserve l’ordre (fusionner sans doublon de chunk) pour alimenter le reranker."""
    seen: set[int] = {h.chunk_id for h in primary}
    out = list(primary)
    for h in secondary:
        if h.chunk_id in seen:
            continue
        seen.add(h.chunk_id)
        out.append(h)
        if len(out) >= cap:
            break
    return out


def _expand_search_q_with_ingredients(base_q: str, plan: QueryPlan) -> str:
    """Aligne requête BM25/embedding sur les mots d’ingrédient s’ils ne sont pas déjà dans q."""
    qn = (base_q or "").strip()
    qlow = qn.lower()
    extra: list[str] = []
    for s in plan.ingredient_slugs or []:
        w = s.replace("-", " ").strip()
        if not w:
            continue
        if w.lower() in qlow:
            continue
        extra.append(w)
    if not extra:
        return qn
    return f"{qn} {' '.join(extra)}".strip()


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
        self._session_focus = SessionFocusAnalyzer()
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
        *,
        force_corpus_broaden: bool = False,
    ) -> list[Hit]:
        """Recherche hybride + retries (filtres, fallbacks, élargissement corpus)."""
        s = self.settings
        excl = exclude_article_external_ids
        ex_boost = min(80, len(excl) * s.rag_retrieval_extra_limit_per_excluded)
        final_limit = max(30, rerank_top_n * 4) + ex_boost

        has_sql_filters = bool(
            plan.chef_slugs
            or plan.ingredient_slugs
            or plan.category_slugs
            or plan.keyword_slugs
        )

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
        if not hits and has_sql_filters:
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
        if (
            s.rag_retrieval_widen_enabled
            and has_sql_filters
            and hits
            and (
                force_corpus_broaden
                or len(hits) < s.rag_retrieval_widen_min_hits
            )
        ):
            try:
                broad = await self.retriever.search(
                    session,
                    q,
                    chef_slugs=[],
                    ingredient_slugs=[],
                    category_slugs=[],
                    keyword_slugs=[],
                    final_limit=final_limit + 24,
                    exclude_article_external_ids=excl,
                )
                n0 = len(hits)
                hits = _dedupe_merge_hits(hits, broad, cap=120)
                if len(hits) > n0:
                    log.info(
                        "rag.pipeline.retrieval_merged_corpus_broaden",
                        n_before=n0,
                        n_after=len(hits),
                        force=force_corpus_broaden,
                    )
            except Exception as exc:  # noqa: BLE001
                log.warning("rag.pipeline.retrieval_widen_merge_failed", error=str(exc))
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
        focus: SessionFocus | None = None
        try:
            if conversation_history and conversation_history.strip():
                plan, focus = await asyncio.gather(
                    self.analyzer.analyze(
                        user_query, conversation_history=conversation_history
                    ),
                    self._session_focus.infer(
                        user_query, conversation_history
                    ),
                )
            else:
                plan = await self.analyzer.analyze(
                    user_query, conversation_history=None
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("rag.pipeline.query_focus_failed_fallback", error=str(exc))
            try:
                plan = await self.analyzer.analyze(
                    user_query, conversation_history=conversation_history
                )
            except Exception as exc2:  # noqa: BLE001
                log.warning("rag.pipeline.query_analyze_failed_fallback", error=str(exc2))
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
            focus = None
        timings["query_understanding_ms"] = int((time.perf_counter() - t0) * 1000)

        t1 = time.perf_counter()
        base_q = (plan.rewritten_query or user_query or "").strip()
        q = _expand_search_q_with_ingredients(base_q, plan)
        if focus and (focus.search_boost_phrase or "").strip():
            q = f"{q} {(focus.search_boost_phrase or '').strip()}".strip()
        force_broaden = bool(focus and focus.suggest_broaden_corpus_search)
        excluded: list[int] = []
        if session_id:
            try:
                excluded = await sessions_store.recent_article_external_ids(session_id)
            except Exception as exc:  # noqa: BLE001
                log.warning("rag.pipeline.session_exclusions_failed", error=str(exc))
        try:
            hits = await self._retrieve_hits(
                session,
                user_query,
                plan,
                q,
                rerank_top_n,
                excluded,
                force_corpus_broaden=force_broaden,
            )
            if not hits and excluded:
                log.info(
                    "rag.pipeline.retrieval_retry_no_session_article_exclusions",
                    n_excluded=len(excluded),
                )
                hits = await self._retrieve_hits(
                    session,
                    user_query,
                    plan,
                    q,
                    rerank_top_n,
                    [],
                    force_corpus_broaden=force_broaden,
                )
        except Exception as exc:  # noqa: BLE001
            log.exception("rag.pipeline.retrieval_failed")
            hits = []
        timings["retrieval_ms"] = int((time.perf_counter() - t1) * 1000)

        t2 = time.perf_counter()
        rq = q
        hits_for_rerank = hits
        if self.settings.rag_prererank_interleave and len(hits) > 1:
            hits_for_rerank = _interleave_hits_by_article(hits)
            n_art = len({h.article_external_id for h in hits})
            if n_art > 1 and len(hits_for_rerank) == len(hits):
                log.info(
                    "rag.pipeline.prererank_interleave",
                    n_hits=len(hits),
                    n_articles=n_art,
                )
        try:
            reranked = await self.reranker.rerank(
                rq, hits_for_rerank, top_n=rerank_top_n
            )
        except Exception as exc:  # noqa: BLE001
            # Cohere indispo, timeout, quota, etc. — on continue sans rerank.
            log.warning("rag.pipeline.rerank_failed_fallback", error=str(exc))
            reranked = [
                RerankedHit(hit=h, rerank_score=float(h.score_rrf or 0.0))
                for h in hits_for_rerank[:rerank_top_n]
            ]
        reranked = [
            r for r in reranked if r.rerank_score >= self.settings.rag_min_rerank_score
        ] or reranked[: max(1, rerank_top_n // 2)]
        timings["rerank_ms"] = int((time.perf_counter() - t2) * 1000)

        t3 = time.perf_counter()
        thread_summary = (
            (focus.thread_summary or "").strip()
            if focus
            else ""
        )
        answer = await self.generator.generate(
            user_query,
            reranked,
            conversation_history=conversation_history,
            session_thread_summary=thread_summary or None,
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
