"""
Routage LLM : OpenAI (défaut) vs Groq (OpenAI-compatible).

Groq : base_url https://api.groq.com/openai/v1 + GROQ_API_KEY.
Modèles : https://console.groq.com/docs/models
"""

from __future__ import annotations

from openai import AsyncOpenAI

# IDs production GroqCloud (table « Production models »)
GROQ_MODEL_IDS: frozenset[str] = frozenset(
    {
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
    }
)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def is_groq_model(model: str | None) -> bool:
    if not model:
        return False
    return model.strip() in GROQ_MODEL_IDS


def provider_credentials_ok(model: str | None) -> bool:
    """True si la clé attendue pour ce modèle est définie."""
    from .config import get_settings

    s = get_settings()
    if not model:
        return bool(s.openai_api_key)
    if is_groq_model(model):
        return bool(s.groq_api_key)
    return bool(s.openai_api_key)


def async_openai_client_for_model(model: str) -> AsyncOpenAI:
    """Client AsyncOpenAI pointant vers OpenAI ou Groq selon l’ID de modèle."""
    from .config import get_settings

    s = get_settings()
    if is_groq_model(model):
        if not s.groq_api_key:
            raise ValueError("GROQ_API_KEY manquant pour les modèles Groq")
        return AsyncOpenAI(
            api_key=s.groq_api_key,
            base_url=GROQ_BASE_URL,
            max_retries=2,
            timeout=90.0,
        )
    if not s.openai_api_key:
        raise ValueError("OPENAI_API_KEY manquant")
    return AsyncOpenAI(api_key=s.openai_api_key, max_retries=2, timeout=12.0)


def uses_openai_json_schema(model: str) -> bool:
    """Groq ne prend pas en charge response_format json_schema comme OpenAI ; on utilise json_object."""
    return not is_groq_model(model)
