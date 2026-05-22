"""Modèles LLM autorisés pour Sahteïn (chat / RAG)."""

from __future__ import annotations

from ..settings import get_settings

ALLOWED_LLM_MODELS: tuple[str, ...] = (
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4.1",
)

DEFAULT_LLM_MODEL = "gpt-4.1-mini"


def resolve_llm_model(request_override: str | None = None) -> str:
    """Modèle effectif : override client (dev) ou `LLM_MODEL` serveur."""
    settings_default = (get_settings().llm_model or DEFAULT_LLM_MODEL).strip()
    default = (
        settings_default
        if settings_default in ALLOWED_LLM_MODELS
        else DEFAULT_LLM_MODEL
    )
    raw = (request_override or "").strip()
    if not raw or raw.lower() == "auto":
        return default
    if raw in ALLOWED_LLM_MODELS:
        return raw
    return default


def list_llm_models() -> list[dict[str, str]]:
    labels = {
        "gpt-4.1-mini": "GPT-4.1 mini",
        "gpt-4.1-nano": "GPT-4.1 nano",
        "gpt-4.1": "GPT-4.1",
    }
    return [
        {"id": mid, "label": labels[mid], "provider": "openai"}
        for mid in ALLOWED_LLM_MODELS
    ]
