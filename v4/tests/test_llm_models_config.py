from backend.app.llm.models_config import resolve_llm_model


def test_resolve_llm_model_prefers_server_default(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "gpt-4.1-mini")
    from backend.app.settings import get_settings

    get_settings.cache_clear()
    assert resolve_llm_model(None) == "gpt-4.1-mini"
    assert resolve_llm_model("auto") == "gpt-4.1-mini"
    get_settings.cache_clear()


def test_resolve_llm_model_allows_client_override(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "gpt-4.1-mini")
    from backend.app.settings import get_settings

    get_settings.cache_clear()
    assert resolve_llm_model("gpt-4.1-nano") == "gpt-4.1-nano"
    get_settings.cache_clear()


def test_resolve_llm_model_rejects_unknown(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "gpt-4.1-mini")
    from backend.app.settings import get_settings

    get_settings.cache_clear()
    assert resolve_llm_model("gpt-5") == "gpt-4.1-mini"
    get_settings.cache_clear()
