"""Tests du sectionizer HTML.

On valide :
1. La reconnaissance des balises ``<section data-kind="…">`` produites par
   ``mapper._assemble_body_html`` (chemin moderne, vrai payload).
2. La rétro-compat : du HTML CMS "à plat" (sans nos balises) doit
   continuer à donner des sections via heuristiques heading/classes.
"""

from __future__ import annotations

from backend.app.ingestion.html_sectionizer import sectionize
from backend.app.ingestion.mapper import map_article


def test_sectionize_recognizes_explicit_data_kind(taboule_payload):
    m = map_article(taboule_payload)
    assert m.body_html
    sections = sectionize(m.body_html)
    kinds = {s.kind for s in sections}
    expected = {
        "recipe_meta",
        "recipe_summary",
        "recipe_history",
        "ingredients_list",
        "recipe_steps",
        "chef_astuce",
        "chef_bio",
    }
    missing = expected - kinds
    assert not missing, f"sections explicites manquantes : {missing}"


def test_sectionize_history_contains_commandments(taboule_payload):
    m = map_article(taboule_payload)
    sections = sectionize(m.body_html or "")
    history = [s for s in sections if s.kind == "recipe_history"]
    assert history, "recipe_history manquante"
    text = history[0].text.lower()
    assert "essentiellement du persil" in text  # 1er commandement


def test_sectionize_ingredients_keeps_persil(taboule_payload):
    m = map_article(taboule_payload)
    sections = sectionize(m.body_html or "")
    ing = [s for s in sections if s.kind == "ingredients_list"]
    assert ing
    assert "persil" in ing[0].text.lower()


def test_sectionize_steps_mention_persil_or_bourghol(taboule_payload):
    m = map_article(taboule_payload)
    sections = sectionize(m.body_html or "")
    steps = [s for s in sections if s.kind == "recipe_steps"]
    assert steps
    text = steps[0].text.lower()
    assert "persil" in text or "bourghol" in text


def test_sectionize_chef_bio_mentions_souk_el_tayeb(taboule_payload):
    m = map_article(taboule_payload)
    sections = sectionize(m.body_html or "")
    bio = [s for s in sections if s.kind == "chef_bio"]
    assert bio
    assert "souk el-tayeb" in bio[0].text.lower()


def test_sectionize_legacy_html_still_works():
    """Rétro-compat : du HTML brut sans data-kind doit produire des sections."""
    legacy = """
    <h2>Ingrédients</h2><ul><li>persil</li><li>citron</li></ul>
    <h2>Préparation</h2><ol><li>Hacher</li><li>Mélanger</li></ol>
    <blockquote>Le taboulé, c'est la vie.</blockquote>
    """
    sections = sectionize(legacy)
    kinds = {s.kind for s in sections}
    assert "ingredients_list" in kinds
    assert "recipe_steps" in kinds
    assert "quote" in kinds
