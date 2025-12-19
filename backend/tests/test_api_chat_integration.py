"""
Integration tests hitting the FastAPI /api/chat endpoint.
They verify that the full pipeline (routes -> pipeline -> response) behaves as expected
for high-priority demo scenarios.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from main import app

TABOULE_URL = (
    "https://www.lorientlejour.com/cuisine-liban-a-table/1227694/"
    "le-vrai-taboule-de-kamal-mouzawak.html"
)


@pytest.fixture(scope="module")
def api_client() -> TestClient:
    """Spin up the FastAPI TestClient once for the module."""
    with TestClient(app) as client:
        yield client


def _chat(api_client: TestClient, message: str) -> dict:
    """Helper to send a chat request with debug info enabled."""
    response = api_client.post(
        "/api/chat",
        json={"message": message, "debug": True},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["html"], "Expected HTML payload in response"
    html_lower = data["html"].lower()
    # Basic safety: HTML payload must be present and contain no obvious English keywords.
    assert "<p" in data["html"]
    forbidden = (" thank you", "thanks", " the ", " you may ", "english")
    assert not any(token in html_lower for token in forbidden), (
        f"Unexpected English content for '{message}'"
    )
    return data


def test_taboule_exact_match(api_client: TestClient):
    """`recette du taboulé` must always return the Kamal Mouzawak article with high confidence."""
    data = _chat(api_client, "recette du taboule")

    assert data["scenario_id"] == 1
    assert data["primary_url"] == TABOULE_URL

    link_debug = data["debug_info"]["link_resolution"]
    assert link_debug["confidence"] >= 0.85
    assert link_debug["has_primary"]


def test_yaourt_recipe_flow(api_client: TestClient):
    """Ingredient queries like `recette avec du yaourt` should pick scenario 2/8 and expose an OLJ link."""
    data = _chat(api_client, "recette avec du yaourt")

    assert data["scenario_id"] in (2, 8)
    assert data["primary_url"].startswith("https://www.lorientlejour.com/")
    assert (
        "yaourt" in data["html"].lower()
        or "labn" in data["html"].lower()
        or "yaourt" in data["primary_url"]
        or "labne" in data["primary_url"]
    ), "Expected to reference yaourt/labné in response or URL"


def test_japanese_ramen_is_flagged_off_scope(api_client: TestClient):
    """Foreign cuisine should trigger scenario 6 (hors cuisine libanaise) with a safe OLJ fallback."""
    data = _chat(api_client, "recette de ramen japonais")

    assert data["scenario_id"] == 6
    assert data["primary_url"].startswith("https://www.lorientlejour.com/")
    assert "libanais" in data["html"].lower() or "libanaise" in data["html"].lower()
    assert (
        data["debug_info"]["scenario"]["scenario_name"] == "hors_cuisine_libanaise"
    )


def test_about_bot_presentation(api_client: TestClient):
    """`Qui es-tu ?` should return the bot presentation scenario."""
    data = _chat(api_client, "Qui es-tu ?")

    assert data["scenario_id"] == 5
    assert "sahtein" in data["html"].lower()
    assert data["primary_url"].startswith("https://www.lorientlejour.com/")
    assert data["debug_info"]["scenario"]["scenario_name"] == "about_bot"
