"""Valide la structure de data/golden_eval_fr.json (sans appel RAG)."""

from __future__ import annotations

import json
from pathlib import Path

GOLDEN = Path(__file__).resolve().parents[1] / "data" / "golden_eval_fr.json"


def test_golden_file_exists() -> None:
    assert GOLDEN.is_file(), f"manquant: {GOLDEN}"


def test_golden_schema() -> None:
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert "items" in data
    for item in data["items"]:
        assert "id" in item
        assert "query" in item
        assert isinstance(item.get("expected_article_external_ids", []), list)
        assert isinstance(item.get("answer_must_contain", []), list)
