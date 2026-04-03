"""
Filtre passages / titres non recette (interviews, rubriques) pour le grounding RAG.
"""

from __future__ import annotations

import re
from typing import Optional

# Phrases typiques d’articles non-fiche recette mélangés au corpus
_NON_RECIPE_TITLE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b\d+\s+questions?\s+gourmandes?\b", re.I),
    re.compile(r"\bquestions?\s+gourmandes?\b", re.I),
    re.compile(r"\binterview\b", re.I),
    re.compile(r"\bportrait\b", re.I),
    re.compile(r"\bchez\s+(lui|elle)\b", re.I),
    re.compile(r"\brencontre\s+avec\b", re.I),
)

_NON_RECIPE_PASSAGE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bquand\s+avez-?vous\s+commenc", re.I),
    re.compile(r"\b8\s+questions\b", re.I),
    re.compile(r"\bquestions?\s+gourmandes?\b", re.I),
    re.compile(r"\bson\s+parcours\b", re.I),
    re.compile(r"\bchez\s+lui\b", re.I),
    re.compile(r"\bchez\s+elle\b", re.I),
)


def title_suggests_non_recipe_article(title: str) -> bool:
    if not (title or "").strip():
        return True
    t = title.strip()
    for rx in _NON_RECIPE_TITLE_PATTERNS:
        if rx.search(t):
            return True
    return False


def passage_suggests_editorial_noise(text: str) -> bool:
    if not (text or "").strip():
        return False
    for rx in _NON_RECIPE_PASSAGE_PATTERNS:
        if rx.search(text):
            return True
    return False


def sanitize_cited_passage(
    passage: Optional[str],
    *,
    title: str,
    fallback_excerpt_chars: int = 220,
) -> Optional[str]:
    """
    Retourne None si le passage est bruit éditorial ; sinon passage nettoyé ou extrait titre+début safe.
    """
    if not passage or not passage.strip():
        return None
    p = passage.strip()
    if passage_suggests_editorial_noise(p):
        # Tomber sur un extrait « recette » minimal : début du texte sans la phrase d’interview
        return _first_recipe_like_sentence(title, p) or (title[:120] + "…" if title else None)
    return p


def _first_recipe_like_sentence(title: str, passage: str) -> Optional[str]:
    """Prend la première phrase du passage qui ne matche pas le bruit interview."""
    blob = f"{title}. {passage}"
    parts = re.split(r"(?<=[.!?])\s+", blob)
    for sent in parts:
        s = sent.strip()
        if len(s) < 20:
            continue
        if passage_suggests_editorial_noise(s):
            continue
        return s[:400] + ("…" if len(s) > 400 else "")
    return None
