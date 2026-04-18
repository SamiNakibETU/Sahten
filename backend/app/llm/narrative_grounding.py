"""
Contrôles légers d'ancrage narrative (post-LLM).

La validation principale reste dans ResponseGenerator._validate ; ce module
expose des helpers réutilisables pour les tests ou le logging.
"""

from __future__ import annotations

from typing import List

from ..schemas.responses import RecipeCard, RecipeNarrative


def narrative_contains_raw_url(text: str) -> bool:
    """True si le texte contient une URL brute (à éviter dans hook/detail/cta)."""
    if not text:
        return False
    return "http://" in text or "https://" in text


def collect_allowed_tokens_from_cards(cards: List[RecipeCard]) -> set[str]:
    """Tokens normalisés (longueur >= 4) issus des titres et chefs des cartes."""
    from unidecode import unidecode

    out: set[str] = set()
    for c in cards:
        for part in (c.title or "", c.chef or ""):
            for w in unidecode(part.lower()).split():
                w = "".join(ch for ch in w if ch.isalnum())
                if len(w) >= 4:
                    out.add(w)
    return out


def narrative_has_grounding_hint(
    narrative: RecipeNarrative,
    cards: List[RecipeCard],
) -> bool:
    """
    True si au moins un token significatif du titre ou du chef apparaît dans le texte combiné.
    Utilisable pour des métriques ou des tests, pas comme garde-fou strict (faux négatifs possibles).
    """
    if not cards:
        return True
    allowed = collect_allowed_tokens_from_cards(cards)
    if not allowed:
        return True
    combined = " ".join(
        p
        for p in [
            narrative.hook or "",
            narrative.cultural_context or "",
            narrative.cta or "",
        ]
        if p
    )
    from unidecode import unidecode

    words = unidecode(combined.lower()).split()
    blob = " ".join(words)
    return any(t in blob for t in list(allowed)[:40])
