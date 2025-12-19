"""
Unit tests for DishNormalizer to ensure critical mappings stay intact.
"""

from app.data.dish_normalizer import dish_normalizer


def test_taboule_variations_normalize_to_canonical():
    assert dish_normalizer.normalize("taboulé") == "tabbouleh"
    assert dish_normalizer.normalize("tabbouli") == "tabbouleh"


def test_taboule_expansion_contains_variants():
    variants = dish_normalizer.expand("tabbouleh")
    assert "tabbouleh" in variants
    assert "taboule" in variants
    assert "tabouleh" in variants


def test_taboule_typo_is_normalized():
    assert dish_normalizer.normalize("taboul") == "tabbouleh"
    assert dish_normalizer.normalize("tabouleh") == "tabbouleh"
