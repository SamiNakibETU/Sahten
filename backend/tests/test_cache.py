"""Response cache LRU + settings."""

from __future__ import annotations

import pytest

from app.core.config import get_settings
from app.core.response_cache import ResponseCache
from app.schemas.responses import RecipeNarrative, SahtenResponse


@pytest.fixture
def isolated_cache(monkeypatch):
    monkeypatch.setenv("ENABLE_RESPONSE_CACHE", "true")
    get_settings.cache_clear()
    import app.core.response_cache as rcmod

    prev = rcmod._response_cache
    rcmod._response_cache = ResponseCache(max_size=10, ttl_seconds=3600)
    yield rcmod._response_cache
    rcmod._response_cache = prev
    get_settings.cache_clear()


def test_settings_rerank_model_is_nano():
    get_settings.cache_clear()
    assert get_settings().rerank_model == "gpt-4.1-nano"


def test_response_cache_put_get(isolated_cache):
    c = isolated_cache
    narrative = RecipeNarrative(
        hook="Test hook long enough",
        cultural_context="Contexte assez long pour valider",
        teaser=None,
        cta="CTA",
        closing="Sahten !",
    )
    resp = SahtenResponse(
        response_type="greeting",
        narrative=narrative,
        recipes=[],
        recipe_count=0,
        intent_detected="greeting",
        confidence=1.0,
        model_used="gpt-4.1-nano",
    )
    meta = {"timings_ms": {"total": 12}}
    c.put("bonjour", "gpt-4.1-nano", (resp, None, meta))
    hit = c.get("bonjour", "gpt-4.1-nano")
    assert hit is not None
    r2, dbg2, m2 = hit
    assert r2.response_type == "greeting"
    assert m2["timings_ms"]["total"] == 12
    assert dbg2 is None


def test_response_cache_miss_different_model(isolated_cache):
    c = isolated_cache
    assert c.get("hello", "gpt-4.1-mini") is None
