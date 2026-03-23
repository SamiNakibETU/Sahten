"""Non-regression: unified narrative generator + validator (no live API)."""

from __future__ import annotations

import pytest

from app.core.config import get_settings
from app.llm.response_generator import EXACT_ALTERNATIVE_HOOK, ResponseGenerator
from app.schemas.responses import EvidenceBundle, RecipeCard, RecipeNarrative, SharedIngredientProof


@pytest.fixture
def clear_settings(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_response_generator_uses_fallback_model_from_settings(clear_settings):
    g = ResponseGenerator()
    assert g.fallback_model == "gpt-4.1-mini"
    assert g.model == get_settings().openai_model


def test_validate_accepts_alternative_with_exact_contractual_hook(clear_settings):
    """not_found_with_alternative : le hook doit être exactement EXACT_ALTERNATIVE_HOOK."""
    g = ResponseGenerator(api_key="")
    evidence = EvidenceBundle(
        response_type="not_found_with_alternative",
        intent_detected="recipe_specific",
        user_query="recette fajitas",
        selected_recipe_cards=[
            RecipeCard(
                source="olj",
                title="Légumes farcis au quinoa",
                url="https://www.lorientlejour.com/article/1234567",
                category="plat_principal",
            )
        ],
        shared_ingredient_proof=SharedIngredientProof(
            query_ingredient="poulet",
            normalized_ingredient="poulet",
            shared_ingredients=["poulet", "légumes"],
            recipe_title="Légumes farcis au quinoa",
            recipe_url="https://www.lorientlejour.com/article/1234567",
            proof_score=2,
        ),
    )
    nar = RecipeNarrative(
        hook=EXACT_ALTERNATIVE_HOOK,
        cultural_context="Les légumes farcis au quinoa de Joanna Kassem utilisent aussi du poulet.",
        teaser=None,
        cta="Découvre sur L'Orient-Le Jour",
        closing="Sahten !",
    )
    assert g._validate(nar, evidence)


def test_validate_rejects_banned_patterns(clear_settings):
    g = ResponseGenerator(api_key="")
    evidence = EvidenceBundle(
        response_type="recipe_olj",
        intent_detected="recipe_specific",
        user_query="recette fajitas",
        selected_recipe_cards=[
            RecipeCard(
                source="olj",
                title="Légumes farcis au quinoa",
                url="https://www.lorientlejour.com/article/1234567",
                category="plat_principal",
            )
        ],
    )
    nar = RecipeNarrative(
        hook="La cuisine libanaise regorge de plats simples.",
        cultural_context="Les légumes farcis au quinoa sont un bon choix.",
        teaser=None,
        cta="OLJ",
        closing="Sahten !",
    )
    assert not g._validate(nar, evidence)


def test_fallback_narrative_without_recipes(clear_settings):
    g = ResponseGenerator(api_key="")
    result = g._fallback_narrative("blanquette de veau", [])
    assert "pas" in result.hook.lower()


def test_fallback_alternative_with_proof(clear_settings):
    g = ResponseGenerator(api_key="")
    alt = RecipeCard(
        source="olj",
        title="Mouloukhiye de Tara Khattar",
        url="https://www.lorientlejour.com/article/1",
        category="plat_principal",
        chef="Tara Khattar",
    )
    proof = SharedIngredientProof(
        query_ingredient="veau",
        normalized_ingredient="viande",
        shared_ingredients=["viande"],
        recipe_title="Mouloukhiye",
        recipe_url="https://www.lorientlejour.com/article/1",
        proof_score=1,
    )
    result = g._fallback_alternative("blanquette de veau", alt, proof)
    assert "Mouloukhiye" in result.cultural_context
    assert "viande" in result.cultural_context
