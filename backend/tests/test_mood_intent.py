"""Tests intent saison / humeur / moment (recipe_by_mood vs recipe_specific)."""

from __future__ import annotations

import pytest

from app.core.intent_router import route_intent_deterministic
from app.llm.query_analyzer import QueryAnalyzer


@pytest.mark.parametrize(
    "query,expected_intent,key_tags",
    [
        ("recette pour l'hiver", "recipe_by_mood", ["hiver", "reconfortant"]),
        ("recette pour l'hiver ", "recipe_by_mood", ["hiver"]),
        ("idee plat automne", "recipe_by_mood", ["reconfortant"]),
        ("plat pour le printemps", "recipe_by_mood", ["frais", "leger"]),
        ("recette legere pour ce soir", "recipe_by_mood", ["leger", "frais"]),
        ("recette pour ce midi", "recipe_by_mood", ["rapide", "frais"]),
        ("plat pour dimanche", "recipe_by_mood", ["convivial", "festif"]),
    ],
)
def test_route_mood_or_season(query: str, expected_intent: str, key_tags: list[str]) -> None:
    r = route_intent_deterministic(query)
    assert r is not None
    assert r.intent == expected_intent
    tags = r.mood_tags or []
    for t in key_tags:
        assert t in tags, f"missing tag {t!r} in {tags!r} for {query!r}"


def test_route_recette_with_named_dish_stays_specific() -> None:
    r = route_intent_deterministic("recette taboulé pour ce soir")
    assert r is not None
    assert r.intent == "recipe_specific"
    assert r.dish_name


def test_route_recette_poulet_hiver_not_pure_season_mood() -> None:
    """Saison seule après un plat nommé : ne pas classer toute la requête en recipe_by_mood."""
    r = route_intent_deterministic("recette de poulet pour l'hiver")
    assert r is not None
    assert r.intent == "recipe_specific"
    assert "poulet" in (r.dish_name or "").lower()


def test_fallback_offline_mood_hiver() -> None:
    a = QueryAnalyzer(api_key="")
    r = a._fallback_analysis("recette pour l'hiver")
    assert r.intent == "recipe_by_mood"
    assert r.mood_tags
    assert "hiver" in r.mood_tags
