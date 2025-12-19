"""
Embedding client abstraction for lexical/vector hybrid retrieval.
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
from abc import ABC, abstractmethod
from typing import Iterable

from app.models.config import settings

logger = logging.getLogger(__name__)


class EmbeddingClient(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        """Return embeddings for the provided texts."""
        raise NotImplementedError


class MockEmbeddingClient(EmbeddingClient):
    """
    Deterministic mock embeddings for tests/offline environments.

    Uses a text-seeded PRNG so scores remain stable between runs.
    """

    def __init__(self, dimension: int = 128):
        self.dimension = dimension

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        vectors: list[list[float]] = []

        for text in texts:
            seed = hashlib.sha256((text or "").encode("utf-8")).hexdigest()
            rng = random.Random(int(seed, 16))
            raw = [rng.uniform(-1.0, 1.0) for _ in range(self.dimension)]
            norm = math.sqrt(sum(v * v for v in raw)) or 1.0
            vectors.append([v / norm for v in raw])

        return vectors


class OpenAIEmbeddingClient(EmbeddingClient):
    """Embedding client backed by OpenAI's embeddings endpoint."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "OpenAI package not installed. Run `pip install openai` to enable embeddings."
            ) from exc

        if not (api_key or settings.openai_api_key):
            raise ValueError("OpenAI API key missing for embedding client")

        self.client = OpenAI(api_key=api_key or settings.openai_api_key)
        self.model = model or settings.embedding_model

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        batched_texts = list(texts)
        if not batched_texts:
            return []

        response = self.client.embeddings.create(
            model=self.model,
            input=batched_texts,
        )
        return [item.embedding for item in response.data]


def get_embedding_client(provider: str | None = None) -> EmbeddingClient:
    """Factory returning the configured embedding client."""
    provider = provider or settings.embedding_provider

    if provider == "openai":
        try:
            return OpenAIEmbeddingClient()
        except Exception as exc:  # pragma: no cover - falls back to mock
            logger.warning("Falling back to mock embeddings: %s", exc)
            return MockEmbeddingClient()

    # Default to mock embeddings (deterministic, offline-friendly)
    return MockEmbeddingClient()

