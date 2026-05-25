"""Suivi des coûts API par requête (OpenAI chat/embeddings, Cohere rerank).

Tarifs USD / 1M tokens (tier Standard OpenAI, mai 2026) :
  https://platform.openai.com/docs/pricing
Cohere rerank : $2.00 / 1 000 search units (hypothèse v3 ≈ v4 fast) :
  https://cohere.com/pricing
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Iterator

# --- Tarifs chat OpenAI ($ / 1M tokens) ------------------------------------

_OPENAI_CHAT: dict[str, tuple[float, float]] = {
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
}

# --- Tarifs embeddings OpenAI ($ / 1M tokens input) ------------------------

_OPENAI_EMBED: dict[str, float] = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
}

# --- Cohere rerank ($ / search unit) ---------------------------------------

_COHERE_RERANK_PER_SEARCH: dict[str, float] = {
    "rerank-v4.0-fast": 2.0 / 1000.0,
    "rerank-v4.0-pro": 2.5 / 1000.0,
    "rerank-multilingual-v3.0": 2.0 / 1000.0,
    "rerank-english-v3.0": 2.0 / 1000.0,
    "rerank-v3.5": 2.0 / 1000.0,
}
_DEFAULT_COHERE_RERANK_PER_SEARCH = 2.0 / 1000.0

_COHERE_DOCS_PER_SEARCH = 100
_COHERE_MAX_TOKENS_PER_DOC = 500
_CHARS_PER_TOKEN_EST = 4

_current: ContextVar[RequestCostAccumulator | None] = ContextVar(
    "request_cost_accumulator", default=None
)


def _chat_rates(model: str) -> tuple[float, float]:
    key = (model or "").strip()
    if key in _OPENAI_CHAT:
        return _OPENAI_CHAT[key]
    for prefix, rates in _OPENAI_CHAT.items():
        if key.startswith(prefix):
            return rates
    return _OPENAI_CHAT["gpt-4.1-mini"]


def _embed_rate(model: str) -> float:
    key = (model or "").strip()
    return _OPENAI_EMBED.get(key, _OPENAI_EMBED["text-embedding-3-small"])


def _rerank_rate(model: str) -> float:
    key = (model or "").strip()
    return _COHERE_RERANK_PER_SEARCH.get(key, _DEFAULT_COHERE_RERANK_PER_SEARCH)


def estimate_cohere_search_units(documents: list[str]) -> int:
    """Estime les search units Cohere (règle : 100 docs/search, split >500 tok)."""
    if not documents:
        return 0
    counted = 0
    for doc in documents:
        text = (doc or "").strip()
        if not text:
            counted += 1
            continue
        tokens = max(1, len(text) // _CHARS_PER_TOKEN_EST)
        if tokens <= _COHERE_MAX_TOKENS_PER_DOC:
            counted += 1
        else:
            counted += (tokens + _COHERE_MAX_TOKENS_PER_DOC - 1) // _COHERE_MAX_TOKENS_PER_DOC
    return max(1, (counted + _COHERE_DOCS_PER_SEARCH - 1) // _COHERE_DOCS_PER_SEARCH)


@dataclass
class RequestCostAccumulator:
    """Accumule usage + coût estimé pour une requête chat."""

    openai_chat: dict[str, dict[str, Any]] = field(default_factory=dict)
    openai_embeddings: dict[str, Any] = field(default_factory=lambda: {
        "tokens": 0,
        "calls": 0,
        "models": {},
        "usd": 0.0,
    })
    cohere_rerank: dict[str, dict[str, Any]] = field(default_factory=dict)

    def record_openai_chat(
        self,
        step: str,
        *,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        pin, pout = _chat_rates(model)
        usd = (prompt_tokens * pin + completion_tokens * pout) / 1_000_000.0
        self.openai_chat[step] = {
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "usd": round(usd, 6),
        }

    def record_openai_chat_usage(self, step: str, *, model: str, usage: Any) -> None:
        if usage is None:
            return
        self.record_openai_chat(
            step,
            model=model,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        )

    def record_openai_embedding(
        self,
        *,
        model: str,
        tokens: int,
    ) -> None:
        tok = max(0, int(tokens))
        rate = _embed_rate(model)
        usd = tok * rate / 1_000_000.0
        emb = self.openai_embeddings
        emb["tokens"] = int(emb.get("tokens", 0)) + tok
        emb["calls"] = int(emb.get("calls", 0)) + 1
        models: dict[str, int] = emb.setdefault("models", {})
        models[model] = int(models.get(model, 0)) + tok
        emb["usd"] = round(float(emb.get("usd", 0.0)) + usd, 6)

    def record_cohere_rerank(
        self,
        step: str,
        *,
        model: str,
        documents: list[str],
        search_units: int | None = None,
    ) -> None:
        units = int(search_units) if search_units is not None else estimate_cohere_search_units(documents)
        usd = units * _rerank_rate(model)
        self.cohere_rerank[step] = {
            "model": model,
            "documents": len(documents),
            "search_units": units,
            "usd": round(usd, 6),
        }

    @property
    def estimated_usd(self) -> float:
        total = float(self.openai_embeddings.get("usd", 0.0))
        for row in self.openai_chat.values():
            total += float(row.get("usd", 0.0))
        for row in self.cohere_rerank.values():
            total += float(row.get("usd", 0.0))
        return round(total, 6)

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimated_usd": self.estimated_usd,
            "openai_chat": self.openai_chat,
            "openai_embeddings": self.openai_embeddings,
            "cohere_rerank": self.cohere_rerank,
        }


def get_request_cost() -> RequestCostAccumulator | None:
    return _current.get()


@contextmanager
def cost_tracker_scope() -> Iterator[RequestCostAccumulator]:
    acc = RequestCostAccumulator()
    token: Token = _current.set(acc)
    try:
        yield acc
    finally:
        _current.reset(token)


def cohere_billed_search_units(response: Any, documents: list[str]) -> int:
    """Lit billed_units Cohere si présent, sinon estime."""
    meta = getattr(response, "meta", None)
    billed = getattr(meta, "billed_units", None) if meta is not None else None
    if billed is not None:
        units = getattr(billed, "search_units", None)
        if units is not None:
            return max(0, int(units))
    return estimate_cohere_search_units(documents)
