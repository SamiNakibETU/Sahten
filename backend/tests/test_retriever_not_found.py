"""
Tests for get_olj_recommendation_by_ingredient (recette non trouvée).
"""

from __future__ import annotations

import pytest

from app.rag.retriever import HybridRetriever
from app.schemas.query_analysis import QueryAnalysis


@pytest.fixture(scope="module")
def retriever() -> HybridRetriever:
    """Retriever avec données réelles."""
    return HybridRetriever()


def test_filter_main_ingredients():
    """Filtrage du bruit dans main_ingredients."""
    filtered = HybridRetriever._filter_main_ingredients(
        ["persil", "2", "verts", "menthe", "dolive", "bourghol"]
    )
    assert "persil" in filtered
    assert "menthe" in filtered
    assert "bourghol" in filtered
    assert "olive" in filtered  # dolive -> olive
    assert "2" not in filtered
    assert "verts" not in filtered


def test_extract_ingredients_from_dish_name():
    """Heuristique extraction ingrédients depuis dish_name."""
    assert "citron" in HybridRetriever._extract_ingredients_from_dish_name("tarte au citron")
    assert "pommes" in HybridRetriever._extract_ingredients_from_dish_name("tarte aux pommes")
    ings = HybridRetriever._extract_ingredients_from_dish_name("poulet curry")
    assert "poulet" in ings
    assert "curry" in ings
    assert HybridRetriever._extract_ingredients_from_dish_name("taboulé") == []
    assert HybridRetriever._extract_ingredients_from_dish_name(None) == []


def test_get_reco_by_ingredient_poulet(retriever: HybridRetriever):
    """ingredients=['poulet'] doit retourner une recette avec poulet."""
    analysis = QueryAnalysis(
        intent="recipe_by_ingredient",
        ingredients=["poulet"],
        recipe_count=1,
    )
    result = retriever.get_olj_recommendation_by_ingredient(analysis, "recette poulet")
    assert result is not None
    card, matched = result
    assert card.title
    assert card.url and "lorientlejour" in card.url
    assert card.source in ("olj", "base2")
    # Si match par ingrédient, matched devrait être "poulet"
    if matched:
        assert matched.lower() == "poulet"


def test_get_reco_by_ingredient_fallback_category(retriever: HybridRetriever):
    """Sans ingrédients, category='dessert' doit retourner un dessert."""
    analysis = QueryAnalysis(
        intent="recipe_by_category",
        category="dessert",
        ingredients=[],
        recipe_count=1,
    )
    result = retriever.get_olj_recommendation_by_ingredient(analysis, "un dessert")
    assert result is not None
    card, matched = result
    assert card.title
    assert card.category == "dessert" or card.source == "base2"
    assert matched is None  # fallback, pas de match ingrédient


def test_get_reco_exclude_urls(retriever: HybridRetriever):
    """exclude_urls doit exclure les recettes déjà proposées."""
    analysis = QueryAnalysis(
        intent="recipe_by_ingredient",
        ingredients=["poulet"],
        recipe_count=1,
    )
    result1 = retriever.get_olj_recommendation_by_ingredient(analysis, "poulet")
    assert result1 is not None
    card1, _ = result1
    exclude = {card1.url}
    result2 = retriever.get_olj_recommendation_by_ingredient(
        analysis, "poulet", exclude_urls=exclude
    )
    assert result2 is not None
    card2, _ = result2
    assert card2.url != card1.url


def test_get_reco_inferred_ingredients(retriever: HybridRetriever):
    """inferred_ingredients (ex: figues) doit matcher une recette."""
    analysis = QueryAnalysis(
        intent="recipe_specific",
        dish_name="tiramisu",
        ingredients=[],
        inferred_ingredients=["figues", "miel"],  # simule tiramisu -> on cherche figues
        recipe_count=1,
    )
    result = retriever.get_olj_recommendation_by_ingredient(analysis, "recette tiramisu")
    assert result is not None
    card, matched = result
    assert card.title
    # Soit match par figues/miel, soit fallback
    assert card.url or card.title


@pytest.mark.asyncio
async def test_bot_not_found_returns_olj_narrative():
    """Quand aucune recette trouvée, le bot doit retourner la narrative OLJ."""
    from unittest.mock import AsyncMock, patch
    from app.bot import get_bot
    from app.schemas.responses import RecipeNarrative

    bot = get_bot()
    # Mock search_with_rerank pour retourner [] (aucune recette)
    with patch.object(
        bot.retriever,
        "search_with_rerank",
        new_callable=AsyncMock,
        return_value=([], False, None),
    ):
        resp, _ = await bot.chat("recette de xyz inexistante 123", debug=False)
        # Le flux "not found" doit être déclenché
        assert resp.recipe_count == 0 or resp.recipe_count == 1
        if resp.recipe_count == 1:
            # On a une alternative proposée
            assert resp.recipes
            assert resp.recipes[0].title
        assert resp.narrative is not None
        hook = resp.narrative.hook if hasattr(resp.narrative, "hook") else str(resp.narrative)
        assert "désolé" in hook.lower() or "pardonner" in hook.lower()
