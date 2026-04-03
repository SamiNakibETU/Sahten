"""Tests filtrage citations / titres non-recette."""

from app.rag.citation_quality import (
    passage_suggests_editorial_noise,
    sanitize_cited_passage,
    title_suggests_non_recipe_article,
)


def test_title_interview_pattern():
    assert title_suggests_non_recipe_article("8 questions gourmandes à John")
    assert not title_suggests_non_recipe_article("Taboulé libanais")


def test_passage_noise():
    assert passage_suggests_editorial_noise("Quand avez-vous commencé à cuisiner ?")
    assert not passage_suggests_editorial_noise("Mélangez le persil haché avec le bourghol.")


def test_sanitize_cited_passage():
    bad = "Quand avez-vous commencé ? Carla raconte son parcours."
    out = sanitize_cited_passage(bad, title="Velouté de potiron")
    assert out
    assert "commenc" not in out.lower() or "parcours" not in out.lower()
