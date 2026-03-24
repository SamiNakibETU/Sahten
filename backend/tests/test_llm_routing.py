"""Routage OpenAI vs Groq."""

from app.core.llm_routing import GROQ_MODEL_IDS, is_groq_model, uses_openai_json_schema


def test_is_groq_model_known_ids():
    assert is_groq_model("llama-3.3-70b-versatile")
    assert is_groq_model("openai/gpt-oss-20b")
    assert not is_groq_model("gpt-4.1-nano")
    assert not is_groq_model(None)


def test_groq_model_ids_nonempty():
    assert "llama-3.1-8b-instant" in GROQ_MODEL_IDS


def test_json_schema_only_openai():
    assert uses_openai_json_schema("gpt-4.1-mini")
    assert not uses_openai_json_schema("llama-3.3-70b-versatile")
