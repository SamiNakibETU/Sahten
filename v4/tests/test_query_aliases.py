from backend.app.llm.query_understanding import QueryPlan
from backend.app.rag.pipeline import (
    _expand_query_with_aliases,
    _expand_search_q_with_ingredients,
    _retrieval_fallback_queries,
)


def test_expand_query_with_aliases_for_houmous() -> None:
    out = _expand_query_with_aliases("recette houmous")
    low = out.lower()
    assert "houmous" in low
    assert "hommos" in low
    assert "hummus" in low


def test_retrieval_fallback_queries_include_hommos_variant() -> None:
    plan = QueryPlan(
        rewritten_query="recette houmous",
        intent="recipe",
        ingredient_slugs=["houmous"],
    )
    queries = _retrieval_fallback_queries("recette hoummous", plan)
    blob = " | ".join(queries).lower()
    assert "hommos" in blob
    assert "hummus" in blob


def test_expand_search_query_with_ingredient_aliases() -> None:
    plan = QueryPlan(
        rewritten_query="recette hoummous",
        intent="recipe",
        ingredient_slugs=["houmous"],
    )
    out = _expand_search_q_with_ingredients("recette hoummous", plan).lower()
    assert "hommos" in out
