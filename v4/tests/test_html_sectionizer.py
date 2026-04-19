"""Tests du sectionizer HTML : on doit reconnaître bio + ingrédients + steps."""

from __future__ import annotations

import json
from pathlib import Path

from backend.app.ingestion.html_sectionizer import sectionize

FIXTURE = (
    Path(__file__).parent / "fixtures" / "whitebeard_1227694_mock.json"
)


def _load_html() -> str:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return payload["data"][0]["contents"]["html"]


def test_sectionize_extracts_bio_section():
    sections = sectionize(_load_html())
    bios = [s for s in sections if s.kind == "bio"]
    assert bios, "section bio manquante"
    joined = " ".join(s.text for s in bios)
    assert "Souk el-Tayeb" in joined
    assert "Tawlet" in joined or "Beit" in joined


def test_sectionize_recognizes_ingredients_list():
    sections = sectionize(_load_html())
    ingr = [s for s in sections if s.kind == "ingredients_list"]
    assert ingr, "ingredients_list manquante"
    text = " ".join(s.text for s in ingr).lower()
    assert "persil" in text and "bourghol" in text and "citron" in text


def test_sectionize_recognizes_recipe_steps():
    sections = sectionize(_load_html())
    steps = [s for s in sections if s.kind == "recipe_steps"]
    assert steps, "recipe_steps manquante"
    text = " ".join(s.text for s in steps).lower()
    assert "trancher" in text or "tremper" in text


def test_sectionize_keeps_quote():
    sections = sectionize(_load_html())
    quotes = [s for s in sections if s.kind == "quote"]
    assert quotes
    assert "Liban" in " ".join(q.text for q in quotes)


def test_sectionize_includes_9_commandments_list():
    """Les 9 commandements sont dans une <ol> qui suit un <h2>."""
    sections = sectionize(_load_html())
    list_sections = [
        s for s in sections
        if s.kind in {"list", "ingredients_list", "recipe_steps"}
        and (s.metadata.get("ordered") if s.metadata else False)
    ]
    assert any("commandements" in (s.heading or "").lower() or
               "essentiellement du persil" in s.text.lower()
               for s in list_sections), \
        "Les 9 commandements non capturés"
