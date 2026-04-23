"""Fixtures partagées pour les tests d'ingestion / chunking.

On utilise les **vrais** payloads WhiteBeard téléchargés par
``scripts/audit_whitebeard.py`` (rangés sous ``tests/fixtures/audit/``).
Ces fixtures sont la vérité terrain : si le test passe, le code marche
sur staging.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "audit"


def _load_json(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))


def _enrich_chef_bio(payload: dict[str, Any]) -> dict[str, Any]:
    """Reproduit en local ce que ``service._enrich_chef_bio`` fait en prod :
    si le chef est référencé sans bio, on remplace par sa fiche complète."""
    data = payload.get("data") or []
    if not isinstance(data, list) or not data:
        return payload
    item = data[0]
    chef = item.get("chef") or {}
    if not isinstance(chef, dict):
        return payload
    chef_id = chef.get("id")
    chef_contents = (chef.get("contents") or "").strip() if isinstance(chef.get("contents"), str) else ""
    if chef_contents:
        return payload
    if not chef_id:
        return payload
    chef_path = FIXTURES_DIR / f"chef_{chef_id}.json"
    if not chef_path.is_file():
        return payload
    chef_full_payload = json.loads(chef_path.read_text(encoding="utf-8"))
    chef_data = chef_full_payload.get("data") or []
    if isinstance(chef_data, list) and chef_data and isinstance(chef_data[0], dict):
        item["chef"] = chef_data[0]
    return payload


@pytest.fixture
def taboule_payload() -> dict[str, Any]:
    """Payload taboulé (article 1227694) enrichi de la fiche chef Kamal Mouzawak."""
    return _enrich_chef_bio(_load_json("1227694.json"))


@pytest.fixture
def spaghettis_payload() -> dict[str, Any]:
    """Payload simple (recette spaghettis labné de Carla Rebeiz)."""
    return _enrich_chef_bio(_load_json("1501133.json"))
