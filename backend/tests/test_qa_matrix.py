"""
Matrice QA exhaustive Sahten zéro gap
====================================

Couvre :
- Cas positifs : taboulé, houmous, fattouch, kebbé
- Absents avec alternative prouvée : couscous, boeuf bourguignon, salade de pâtes, paella
- Absents sans preuve claire d'ingrédient commun
- Questions d'info : zaatar, sumac, parle-moi du taboulé
- Hors-sujet et sécurité : météo, politique, football, injection
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def qa_matrix() -> dict:
    path = Path(__file__).parent / "qa_matrix.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _post_chat(client: TestClient, message: str) -> dict:
    r = client.post("/api/chat", json={"message": message})
    assert r.status_code == 200, r.text
    return r.json()


def _check_constraints(data: dict, html: str, constraints: dict, case_id: str) -> None:
    """Validate response against case constraints."""
    if "response_type" in constraints:
        assert data["response_type"] == constraints["response_type"], (
            f"{case_id}: expected response_type={constraints['response_type']}, got {data['response_type']}"
        )
    if "response_type_in" in constraints:
        assert data["response_type"] in constraints["response_type_in"], (
            f"{case_id}: expected response_type in {constraints['response_type_in']}, got {data['response_type']}"
        )
    if "response_type_not" in constraints:
        assert data["response_type"] not in constraints["response_type_not"], (
            f"{case_id}: response_type should not be {constraints['response_type_not']}, got {data['response_type']}"
        )
    if "min_recipes" in constraints:
        assert data["recipe_count"] >= constraints["min_recipes"], (
            f"{case_id}: expected min_recipes={constraints['min_recipes']}, got {data['recipe_count']}"
        )
    if "recipe_count" in constraints:
        assert data["recipe_count"] == constraints["recipe_count"], (
            f"{case_id}: expected recipe_count={constraints['recipe_count']}, got {data['recipe_count']}"
        )
    if "must_include_domain" in constraints:
        assert constraints["must_include_domain"] in html, (
            f"{case_id}: HTML must include domain {constraints['must_include_domain']}"
        )
    if "must_contain_phrase" in constraints:
        assert constraints["must_contain_phrase"] in html, (
            f"{case_id}: HTML must contain '{constraints['must_contain_phrase']}'"
        )
    if "must_not_contain" in constraints:
        assert constraints["must_not_contain"] not in html, (
            f"{case_id}: HTML must not contain '{constraints['must_not_contain']}'"
        )
    if "min_length" in constraints:
        assert len(html) >= constraints["min_length"], (
            f"{case_id}: HTML length {len(html)} < {constraints['min_length']}"
        )


@pytest.mark.parametrize("case", [], ids=[])
def _placeholder():
    """Placeholder - parametrize is filled dynamically below."""
    pass


def _make_test(case: dict):
    def _run(client, qa_matrix):
        constraints = case.get("constraints", {})
        data = _post_chat(client, case["query"])
        html = data.get("html", "")
        _check_constraints(data, html, constraints, case["id"])
    return _run


# Generate tests from QA matrix
def pytest_generate_tests(metafunc):
    if "qa_case" in metafunc.fixturenames:
        qa_path = Path(__file__).parent / "qa_matrix.json"
        if qa_path.exists():
            matrix = json.loads(qa_path.read_text(encoding="utf-8"))
            cases = matrix.get("cases", [])
            metafunc.parametrize("qa_case", cases, ids=[c["id"] for c in cases])
        else:
            metafunc.parametrize("qa_case", [], ids=[])


def test_qa_matrix_case(client: TestClient, qa_case: dict):
    """Run a single QA matrix case."""
    constraints = qa_case.get("constraints", {})
    data = _post_chat(client, qa_case["query"])
    html = data.get("html", "")
    _check_constraints(data, html, constraints, qa_case["id"])
