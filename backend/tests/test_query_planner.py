"""
Tests for QueryPlannerAgent
"""

import pytest
from app.rag.query_planner_agent import QueryPlannerAgent
from app.rag.classifier_agent import ClassifierAgent
from app.models.llm_client import MockLLMClient


@pytest.fixture
def classifier():
    """Create classifier"""
    return ClassifierAgent(llm_client=MockLLMClient())


@pytest.fixture
def planner():
    """Create query planner"""
    return QueryPlannerAgent()


def test_plan_recipe_by_name(classifier, planner):
    """Test planning for recipe by name query"""
    query = "Je veux la recette du taboulé"
    classification = classifier.classify(query)
    plan = planner.plan(classification, query)

    assert plan.need_type == "recipe_by_name"
    assert plan.language == "fr"
    assert plan.link_query is not None
    assert "taboule" in plan.link_query.lower() or "tabbouleh" in plan.link_query.lower()


def test_plan_recipe_by_ingredients(classifier, planner):
    """Test planning for recipe by ingredients query"""
    query = "J'ai du poulet et des tomates, que puis-je faire?"
    classification = classifier.classify(query)
    plan = planner.plan(classification, query)

    assert plan.need_type == "recipe_by_ingredients"
    assert len(plan.ingredients) > 0
    assert plan.retrieval_query is not None


def test_plan_greeting(classifier, planner):
    """Test planning for greeting"""
    query = "Bonjour"
    classification = classifier.classify(query)
    plan = planner.plan(classification, query)

    assert plan.need_type == "greeting"
    assert plan.link_query is None  # No link needed for greeting


def test_plan_about_bot(classifier, planner):
    """Test planning for about_bot query"""
    query = "Qui es-tu?"
    classification = classifier.classify(query)
    plan = planner.plan(classification, query)

    assert plan.need_type == "about_bot"
    assert plan.link_query is None


def test_plan_off_topic(classifier, planner):
    """Test planning for off-topic query"""
    query = "Quelle heure est-il?"
    classification = classifier.classify(query)
    plan = planner.plan(classification, query)

    assert plan.need_type == "off_topic"
    assert plan.link_query is None


def test_retrieval_query_contains_relevant_terms(classifier, planner):
    """Test that retrieval query contains relevant terms"""
    queries = [
        ("recette de hummus", ["hummus"]),
        ("J'ai des pois chiches et du tahini", ["pois chiches", "tahini"]),
        ("Comment faire un mezze rapide?", ["mezze", "rapide"]),
    ]

    for query, expected_terms in queries:
        classification = classifier.classify(query)
        plan = planner.plan(classification, query)

        retrieval_lower = plan.retrieval_query.lower()
        # At least one expected term should be in retrieval query
        assert any(term.lower() in retrieval_lower for term in expected_terms), \
            f"Expected terms {expected_terms} in retrieval query: {plan.retrieval_query}"


def test_constraints_extraction(classifier, planner):
    """Test that constraints are properly extracted"""
    query = "Je veux un plat végétarien au four"
    classification = classifier.classify(query)
    plan = planner.plan(classification, query)

    constraints_text = " ".join(plan.constraints).lower()
    assert "four" in constraints_text or "végétarien" in constraints_text


def test_primary_dish_extraction(classifier, planner):
    """Test primary dish extraction"""
    queries_dishes = [
        ("recette de taboulé", ["taboule", "tabbouleh"]),  # Accept both normalized forms
        ("comment faire du hummus", ["hummus"]),
        ("kebbeh libanais", ["kebb", "kibb"]),  # Partial match ok
    ]

    for query, expected_dish_parts in queries_dishes:
        classification = classifier.classify(query)
        plan = planner.plan(classification, query)

        if plan.primary_dish:
            primary_lower = plan.primary_dish.lower()
            assert any(expected in primary_lower for expected in expected_dish_parts), \
                f"Expected one of {expected_dish_parts} in primary_dish: {plan.primary_dish}"


def test_language_preserved(classifier, planner):
    """Test that language detection is preserved in plan"""
    queries_langs = [
        ("Je veux du taboulé", "fr"),
        ("I want tabbouleh", "non_fr"),
    ]

    for query, expected_lang in queries_langs:
        classification = classifier.classify(query)
        plan = planner.plan(classification, query)

        assert plan.language == expected_lang


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
