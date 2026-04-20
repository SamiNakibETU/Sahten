"""Provider d'embeddings OpenAI.

Modèle et dimension lus depuis `settings.embedding_model` /
`settings.embedding_dim` (par défaut text-embedding-3-small / 1536).

Interface stricte + retry + batching automatique. Aucun fallback silencieux.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from openai import AsyncOpenAI
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..settings import get_settings


class EmbeddingProvider(Protocol):
    model: str
    dim: int

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class OpenAIEmbeddings:
    """Client OpenAI embeddings, modèle/dim configurables via settings."""

    BATCH = 96

    def __init__(self, model: str | None = None) -> None:
        s = get_settings()
        if not s.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY manquant pour les embeddings.")
        self.model = model or s.embedding_model
        self.dim = s.embedding_dim
        self._client = AsyncOpenAI(api_key=s.openai_api_key)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for start in range(0, len(texts), self.BATCH):
            batch = list(texts[start : start + self.BATCH])
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(5),
                wait=wait_exponential(multiplier=0.5, max=10),
                retry=retry_if_exception_type(Exception),
                reraise=True,
            ):
                with attempt:
                    resp = await self._client.embeddings.create(
                        model=self.model, input=batch
                    )
            out.extend(d.embedding for d in resp.data)
        return out
