"""
Tests for IngredientNormalizer typo handling.
"""

from app.data.ingredient_normalizer import ingredient_normalizer


def test_poullet_typo_maps_to_poulet():
    normalized = ingredient_normalizer.normalize_ingredient_list(["poullet"])
    assert any("poulet" in entry for entry in normalized)


def test_youghourt_typo_maps_to_yaourt():
    normalized = ingredient_normalizer.normalize_ingredient_list(["youghourt"])
    assert any("yaourt" in entry for entry in normalized)
