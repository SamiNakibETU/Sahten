"""Tests routeur minimal + fallback offline + mapper QueryPlan."""

from __future__ import annotations

import pytest

from app.core.intent_router import route_intent_deterministic
from app.llm.query_analyzer import QueryAnalyzer
from app.schemas.query_plan import QueryPlan
from app.schemas.query_plan_mapper import query_plan_to_analysis


def test_route_menu_and_greeting_still_deterministic() -> None:
    r = route_intent_deterministic("menu entrée plat dessert")
    assert r is not None
    assert r.intent == "menu_composition"
    g = route_intent_deterministic("bonjour")
    assert g is not None
    assert g.intent == "greeting"


def test_route_culinary_ambiguous_returns_none() -> None:
    """Saisons / plats : désormais résolus par QueryPlan (LLM)."""
    assert route_intent_deterministic("recette pour l'hiver") is None
    assert route_intent_deterministic("recette libanaise") is None
    assert route_intent_deterministic("recette taboulé pour ce soir") is None


def test_query_plan_browse_to_analysis() -> None:
    plan = QueryPlan(
        task="browse_corpus",
        cuisine_scope="lebanese_olj",
        course="plat",
        retrieval_focus="plat libanais familial mezze grillade",
        mood_tags=["convivial"],
    )
    a = query_plan_to_analysis(plan)
    assert a.intent == "recipe_by_mood"
    assert a.plan is not None
    assert "liban" in (a.mood_tags or [])


def test_query_plan_named_dish() -> None:
    plan = QueryPlan(
        task="named_dish",
        cuisine_scope="non_lebanese_named",
        course="any",
        dish_name="fajitas",
        dish_variants=["fajitas"],
        inferred_main_ingredients=["poulet", "viande", "poivron"],
        retrieval_focus="fajitas poulet poivron",
    )
    a = query_plan_to_analysis(plan)
    assert a.intent == "recipe_specific"
    assert a.dish_name == "fajitas"


def test_fallback_offline_mood_hiver() -> None:
    a = QueryAnalyzer(api_key="")
    r = a._fallback_analysis("recette pour l'hiver")
    assert r.intent == "recipe_by_mood"
    assert r.mood_tags
    assert "hiver" in r.mood_tags
