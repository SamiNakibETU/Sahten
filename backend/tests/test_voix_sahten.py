"""
Tests voix éditoriale OLJ.

Évalue la qualité rédactionnelle des réponses alternatives:
- clarté et brièveté
- précision (requête, recette, ingrédient partagé)
- absence de formulations génériques
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Bonus « ton pro » : vouvoiement serviable (plus de marqueurs tutoyés).
SERVICE_MARKERS = (
    "voici",
    "pour vous",
    "vous trouverez",
    "je vous",
    "consultez",
    "sur l'orient-le jour",
    "nos fiches",
    "cette recette",
)

BANNED_PHRASES = [
    "partage au moins un ingrédient",
    "expérience gustative similaire",
    "savoureux et convivial",
    "incarne cette",
    "raviver",
    "plats colorés",
    "plats colores",
    "au liban, on aime",
    "explosion de saveurs",
]
MAX_NARRATIVE_CHARS = 500


def _load_cases() -> list[dict]:
    p = Path(__file__).parent / "voix_sahten_cases.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    return data.get("cases", [])


def _extract_narrative_text(response) -> str:
    """Extract full narrative text from SahtenResponse."""
    parts = []
    if response.conversation_blocks:
        for b in response.conversation_blocks:
            if b.block_type in ("assistant_message", "grounded_snippet"):
                parts.append(b.text)
    elif response.narrative:
        if isinstance(response.narrative, str):
            parts.append(response.narrative)
        else:
            nar = response.narrative
            for attr in ("hook", "cultural_context", "teaser", "cta"):
                val = getattr(nar, attr, None)
                if val:
                    parts.append(val)
    return " ".join(parts).strip()


def _score_response(
    response, case: dict, text: str, debug_info: dict | None = None
) -> tuple[int, list[str]]:
    """
    Score 0-8 (7 critères + bonus oral pour alternatives).
    """
    score = 0
    failures = []
    text_lower = text.lower()

    # 1. Brevity (1 pt)
    if len(text) <= MAX_NARRATIVE_CHARS:
        score += 1
    else:
        failures.append(f"brevity: {len(text)} > {MAX_NARRATIVE_CHARS} chars")

    # 2. Mentions query (1 pt) — relâché pour recipe_not_found
    query = (case.get("query") or "").lower()
    query_terms = [w for w in query.split() if len(w) >= 4 and w not in ("recette", "recettes")]
    if query_terms and any(term in text_lower for term in query_terms):
        score += 1
    elif not query_terms or response.response_type == "recipe_not_found":
        score += 1
    else:
        failures.append(f"mentions_query: {query_terms[:3]} not in text")

    # 3. Mentions recipe (1 pt)
    if response.recipes:
        title_tokens = [t for t in (response.recipes[0].title or "").lower().split() if len(t) >= 4]
        if title_tokens and any(t in text_lower for t in title_tokens[:3]):
            score += 1
        elif not title_tokens:
            score += 1
        else:
            failures.append("mentions_recipe: title not clearly in text")
    else:
        score += 1

    # 4. Mentions ingredient when proof exists (1 pt)
    shared: list = []
    if response.response_type == "not_found_with_alternative" and debug_info:
        proof = debug_info.get("shared_ingredient_proof") or {}
        shared = proof.get("shared_ingredients") or []
    if response.response_type == "not_found_with_alternative" and shared:
        if any(ing.lower() in text_lower for ing in shared[:5]):
            score += 1
        else:
            failures.append(f"mentions_ingredient: shared {shared[:3]} not in text")
    else:
        score += 1

    # 5. No banned phrases (2 pts)
    found_banned = [p for p in BANNED_PHRASES if p in text_lower]
    if not found_banned:
        score += 2
    else:
        failures.append(f"no_banned: found {found_banned}")

    # 6. Editorial style (1 pt): éviter les marqueurs trop assistants
    assistant_markers = [
        "ton guide culinaire",
        "chef dévoué",
        "quelle saveur",
        "raviver aujourd'hui",
    ]
    found_assistant = [p for p in assistant_markers if p in text_lower]
    if not found_assistant:
        score += 1
    else:
        failures.append(f"editorial_style: found {found_assistant}")

    # 7. Ton pro / serviable (bonus +1 pour alternative avec marqueurs vouvoiement)
    if response.response_type == "not_found_with_alternative" and any(
        m in text_lower for m in SERVICE_MARKERS
    ):
        score += 1

    return score, failures


@pytest.fixture(scope="module")
def voix_cases():
    return _load_cases()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _load_cases(), ids=[c["id"] for c in _load_cases()])
async def test_voix_sahten_case_scores_above_threshold(case: dict, monkeypatch):
    """Chaque cas voix OLJ doit passer la grille qualité."""
    monkeypatch.setenv("OPENAI_API_KEY", "")  # Use fallback for CI; set key for full LLM run
    from app.core.config import get_settings
    get_settings.cache_clear()

    from app.bot import reload_bot

    bot = reload_bot()
    response, debug_info, trace_meta = await bot.chat(case["query"], debug=True, session_id=f"voix-{case['id']}")

    expected_type = case.get("expected_type")
    if expected_type and response.response_type != expected_type:
        # Skip scoring if we didn't get the expected flow (e.g. recipe_olj for pizza if in base)
        if response.response_type == "recipe_olj":
            pytest.skip(f"Recipe found in OLJ for {case['id']}, skipping alternative scoring")
        # Still score not_found_with_alternative and recipe_not_found
        pass

    text = _extract_narrative_text(response)
    assert text, f"Narrative should not be empty for {case['id']}"

    score, failures = _score_response(response, case, text, debug_info)
    # Alternative : 6/7 ou 6/8 (bonus oral) ; autres flux : 5/7
    threshold = 6 if response.response_type == "not_found_with_alternative" else 5
    assert score >= threshold, (
        f"{case['id']}: score {score}/8 < {threshold}. Failures: {failures}. "
        f"Text (len={len(text)}): {text[:200]}..."
    )
    assert "timings_ms" in trace_meta
    assert "total" in trace_meta["timings_ms"]


def test_voix_sahten_no_banned_phrases_in_fallback():
    """Fallback narrative must never contain banned phrases."""
    from app.llm.response_generator import ResponseGenerator
    from app.schemas.responses import RecipeCard, SharedIngredientProof

    gen = ResponseGenerator(api_key="")
    alt = RecipeCard(
        source="olj",
        title="Légumes farcis au quinoa",
        url="https://www.lorientlejour.com/article/1234567",
        category="plat_principal",
    )
    proof = SharedIngredientProof(
        query_ingredient="poulet",
        normalized_ingredient="poulet",
        shared_ingredients=["poulet", "légumes"],
        recipe_title=alt.title,
        recipe_url=alt.url,
        proof_score=2,
    )
    nar = gen._fallback_alternative(
        user_query="recette fajitas",
        alternative=alt,
        proof=proof,
    )
    text = " ".join(
        p for p in [nar.hook, nar.cultural_context, nar.teaser, nar.cta] if p
    ).lower()
    for banned in BANNED_PHRASES:
        assert banned not in text, f"Fallback must not contain: {banned}"
