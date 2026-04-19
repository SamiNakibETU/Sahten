"""Reranking cross-encoder. Cohere principal, fallback local optionnel."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol

import cohere
import structlog

from ..settings import get_settings
from .retriever import Hit

log = structlog.get_logger(__name__)


@dataclass
class RerankedHit:
    hit: Hit
    rerank_score: float


class Reranker(Protocol):
    name: str

    async def rerank(
        self, query: str, hits: list[Hit], top_n: int
    ) -> list[RerankedHit]: ...


class CohereReranker:
    name = "cohere"

    def __init__(self, model: str | None = None) -> None:
        s = get_settings()
        if not s.cohere_api_key:
            raise RuntimeError("COHERE_API_KEY manquant pour le reranker Cohere.")
        self._client = cohere.AsyncClientV2(api_key=s.cohere_api_key)
        self.model = model or s.cohere_rerank_model

    async def rerank(
        self, query: str, hits: list[Hit], top_n: int
    ) -> list[RerankedHit]:
        if not hits:
            return []
        docs = [h.chunk_text for h in hits]
        response = await self._client.rerank(
            model=self.model,
            query=query,
            documents=docs,
            top_n=min(top_n, len(docs)),
        )
        return [
            RerankedHit(hit=hits[r.index], rerank_score=float(r.relevance_score))
            for r in response.results
        ]


class LocalBgeReranker:
    name = "bge-local"

    def __init__(self, model_name: str | None = None) -> None:
        s = get_settings()
        # Import lazy: torch est lourd
        from sentence_transformers import CrossEncoder  # type: ignore

        self._model = CrossEncoder(model_name or s.local_rerank_model)

    async def rerank(
        self, query: str, hits: list[Hit], top_n: int
    ) -> list[RerankedHit]:
        if not hits:
            return []
        pairs = [[query, h.chunk_text] for h in hits]
        scores: list[float] = await asyncio.to_thread(
            lambda: list(map(float, self._model.predict(pairs)))
        )
        ranked = sorted(zip(hits, scores), key=lambda t: t[1], reverse=True)
        return [RerankedHit(hit=h, rerank_score=s) for h, s in ranked[:top_n]]


def build_default_reranker() -> Reranker:
    """Cohere si dispo, sinon fallback local si activé, sinon no-op."""
    s = get_settings()
    if s.cohere_api_key:
        return CohereReranker()
    if s.enable_local_rerank_fallback:
        try:
            return LocalBgeReranker()
        except Exception as e:  # noqa: BLE001
            log.warning("local_reranker.unavailable", error=str(e))
    return _NoOpReranker()


class _NoOpReranker:
    name = "noop"

    async def rerank(
        self, query: str, hits: list[Hit], top_n: int
    ) -> list[RerankedHit]:
        return [RerankedHit(hit=h, rerank_score=h.score_rrf) for h in hits[:top_n]]
