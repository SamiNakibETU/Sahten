"""
V7.2 Evaluation Harness
======================

Runs a small matrix of queries and validates constraints:
- min recipe count
- response type
- no duplicate urls
- OLJ domain constraints

Note: This test uses the real V7.2 pipeline (LLM calls). It is intended for
local/CI environments where OPENAI_API_KEY is configured.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest


def _load_matrix() -> dict:
    p = Path(__file__).parent / "test_matrix.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _urls_from_response(resp) -> list[str]:
    """URLs structurées depuis SahtenResponse (prioritaire, stable)."""
    urls: list[str] = []
    for r in getattr(resp, "recipes", None) or []:
        u = getattr(r, "url", None) or ""
        if u:
            urls.append(u)
    olj = getattr(resp, "olj_recommendation", None)
    if olj and getattr(olj, "url", None):
        urls.append(olj.url)
    return urls


def _extract_urls(html: str) -> list[str]:
    """
    Extract *recipe card* URLs from HTML (fallback si pas de champ structuré).

    Composer actuel : recipe-card-link-wrapper (OLJ), olj-cta-link (CTA).
    """
    import re

    html = html or ""
    # Cartes recette OLJ
    urls = re.findall(
        r'<a\s+href="([^"]+)"[^>]*class="recipe-card-link-wrapper"',
        html,
    )
    if urls:
        return urls
    # Ancien marquage (rétrocompat tests)
    urls = re.findall(r'<a href="([^"]+)"[^>]*class="sahtein-card-title"', html)
    if urls:
        return urls
    # CTA secondaire
    urls = re.findall(r'<a href="([^"]+)"[^>]*class="olj-cta-link"', html)
    if urls:
        return urls
    return re.findall(r'href="([^"]+)"', html)


def _extract_categories(html: str) -> list[str]:
    import re

    return [c.strip().lower() for c in re.findall(r'class=\"recipe-category\">([^<]+)<', html or "")]


@pytest.mark.llm
@pytest.mark.asyncio
async def test_matrix():
    from app.bot import get_bot
    from app.api.response_composer import compose_html_response

    matrix = _load_matrix()
    bot = get_bot()

    for case in matrix["cases"]:
        query = case["query"]
        constraints: Dict[str, Any] = case.get("constraints", {})

        resp, debug, _ = await bot.chat(query, debug=True)
        html = compose_html_response(resp)

        structured_urls = _urls_from_response(resp)

        # Response type constraint
        if "response_type" in constraints:
            assert resp.response_type == constraints["response_type"], case["id"]

        # Min recipes
        if "min_recipes" in constraints:
            assert resp.recipe_count >= int(constraints["min_recipes"]), case["id"]

        urls = structured_urls if structured_urls else _extract_urls(html)
        categories = _extract_categories(html)

        # Domain constraint
        if "must_include_domain" in constraints:
            dom = constraints["must_include_domain"]
            assert any(dom in u for u in urls), case["id"]

        # Duplicates
        if constraints.get("no_duplicate_urls"):
            seen = [u for u in urls if "lorientlejour.com" in u]
            assert len(seen) == len(set(seen)), case["id"]

        # Category constraints (simple substring)
        if "all_categories_include" in constraints:
            needle = str(constraints["all_categories_include"]).lower()
            # Only validate if we have categories rendered
            assert categories, case["id"]
            assert all(needle in c for c in categories[: int(constraints.get("min_recipes", len(categories)))]), case["id"]


