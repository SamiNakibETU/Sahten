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


def _extract_urls(html: str) -> list[str]:
    """
    Extract *recipe card* URLs only.

    Note: The HTML composer includes the same URL multiple times (card title + button).
    For evaluation we care about duplicate *recipes*, not duplicate href occurrences.
    """
    import re

    # Prefer card-title links (one per recipe card)
    urls = re.findall(r'<a href=\"([^\"]+)\"[^>]*class=\"sahtein-card-title\"', html or "")
    if urls:
        return urls

    # Fallback: any href
    return re.findall(r'href=\"([^\"]+)\"', html or "")


def _extract_categories(html: str) -> list[str]:
    import re

    return [c.strip().lower() for c in re.findall(r'class=\"sahtein-category\">([^<]+)<', html or "")]


@pytest.mark.asyncio
async def test_matrix():
    from app.bot import get_bot
    from app.api.response_composer import compose_html_response

    matrix = _load_matrix()
    bot = get_bot()

    for case in matrix["cases"]:
        query = case["query"]
        constraints: Dict[str, Any] = case.get("constraints", {})

        resp, debug = await bot.chat(query, debug=True)
        html = compose_html_response(resp)

        # Response type constraint
        if "response_type" in constraints:
            assert resp.response_type == constraints["response_type"], case["id"]

        # Min recipes
        if "min_recipes" in constraints:
            assert resp.recipe_count >= int(constraints["min_recipes"]), case["id"]

        urls = _extract_urls(html)
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


