"""Tests du mapping payload WhiteBeard -> MappedArticle."""

from __future__ import annotations

import json
from pathlib import Path

from backend.app.ingestion.mapper import map_article

FIXTURE = (
    Path(__file__).parent / "fixtures" / "whitebeard_1227694_mock.json"
)


def _load() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_map_article_basic_fields():
    m = map_article(_load())
    assert m.external_id == 1227694
    assert m.url.startswith("https://www.lorientlejour.com/")
    assert "Mouzawak" in m.title
    assert m.summary and "Souk el-Tayeb" in m.summary
    assert m.introduction and "Libanais" in m.introduction
    assert m.body_html and "commandements" in m.body_html.lower()
    assert m.body_text and "persil" in m.body_text.lower()
    assert m.cover_image_url and m.cover_image_url.endswith(".jpg")
    assert m.cover_image_caption and "Marc Fayad" in m.cover_image_caption
    assert m.first_published_at is not None
    assert m.is_premium is False


def test_map_article_authors_with_bio():
    m = map_article(_load())
    names = {a.name for a in m.authors}
    assert "Kamal Mouzawak" in names
    chef = next(a for a in m.authors if a.name == "Kamal Mouzawak")
    assert chef.role == "featured_chef"
    assert chef.biography_text and "Souk el-Tayeb" in chef.biography_text
    assert chef.department == "Cuisine"


def test_map_article_keywords_and_categories():
    m = map_article(_load())
    kw_names = {k.name for k in m.keywords}
    assert {"Taboulé", "Kamal Mouzawak", "Souk el-Tayeb", "Liban"} <= kw_names
    cat_names = {c.name for c in m.categories}
    assert "Cuisine" in cat_names


def test_map_article_sections_present():
    m = map_article(_load())
    kinds = {s.kind for s in m.sections}
    assert {"bio", "ingredients_list", "recipe_steps", "quote"} <= kinds


def test_map_article_status_ok_when_full_payload():
    m = map_article(_load())
    assert m.ingestion_status == "ok"
    assert m.ingestion_notes is None
