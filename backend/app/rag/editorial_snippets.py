"""
Extraits éditoriaux déterministes depuis search_text (chapô + anecdote) pour le LLM générateur.
"""

from __future__ import annotations

import re
from typing import Optional

# Début probable de liste d'ingrédients / sections techniques
_RE_INGRED_OR_SECTION = re.compile(
    r"(?:^|[.!?\n])\s*(?:"
    r"\d+[\d,\s]*(?:g|kg|ml|cl|l|c\.|cuill)\b|"
    r"c\.\s*à\s*soupe|c\.\s*à\s*café|"
    r"\bIngrédients\b|\bPour la (?:pâte|crème|sauce|garniture)\b|\bPour le montage\b)",
    re.IGNORECASE | re.MULTILINE,
)

# Signaux d'anecdote / voix auteur (corpus OLJ)
_RE_STORY = re.compile(
    r"«[^»]{10,220}»|"
    r"ma mère|teta|téta|grand-?mère|à mes yeux|"
    r"ultime plaisir|aimait (?:le )?faire|racont|souvenir|"
    r"en est très fière|Ma famille|je suis|nous sommes",
    re.IGNORECASE,
)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _truncate_at_sentence(text: str, max_len: int) -> str:
    text = text.strip()
    if len(text) <= max_len:
        return text
    cut = text[:max_len]
    last_period = cut.rfind(".")
    if last_period > max_len // 3:
        return cut[: last_period + 1].strip()
    sp = cut.rfind(" ")
    return (cut[:sp] + "…").strip() if sp > 40 else cut + "…"


def extract_recipe_lead(search_text: str, *, max_chars: int = 480) -> str:
    """Chapô / début narratif avant zone ingrédients."""
    if not search_text or not search_text.strip():
        return ""
    text = search_text.strip()
    # Ignorer un préfixe très court (titre seul)
    search_from = min(120, len(text) // 4)
    m = _RE_INGRED_OR_SECTION.search(text, pos=max(0, search_from))
    if m:
        chunk = text[: m.start()].strip()
    else:
        chunk = text[: max_chars + 120]
    if not chunk:
        chunk = text[:max_chars]
    return _truncate_at_sentence(chunk, max_chars)


def extract_story_snippet(
    search_text: str,
    cited_passage: Optional[str],
    *,
    max_chars: int = 300,
) -> str:
    """Une ou deux phrases avec signal d'anecdote ou citation."""
    if not search_text or not search_text.strip():
        return ""
    text = search_text.strip()
    cited = (cited_passage or "").strip()

    # Découpage grossier en phrases
    parts = _SENTENCE_END.split(text)
    picked: list[str] = []
    for p in parts:
        p = p.strip()
        if len(p) < 25:
            continue
        if cited and p in cited:
            continue
        if _RE_STORY.search(p):
            picked.append(p)
            if len(" ".join(picked)) >= max_chars - 20:
                break
        if len(picked) >= 2:
            break

    if not picked:
        return ""

    out = " ".join(picked).strip()
    if cited:
        # Éviter doublon quasi exact avec cited_passage
        cnorm = cited[:80].lower()
        if cnorm and cnorm in out.lower():
            return ""
    return _truncate_at_sentence(out, max_chars)


def extract_editorial_snippets(
    search_text: str,
    cited_passage: Optional[str],
    *,
    lead_max: int = 480,
    story_max: int = 300,
    combined_max: int = 1100,
) -> tuple[str, str]:
    """
    Retourne (recipe_lead, story_snippet) avec plafond global.
    """
    lead = extract_recipe_lead(search_text, max_chars=lead_max)
    story = extract_story_snippet(search_text, cited_passage, max_chars=story_max)
    total = len(lead) + len(story)
    if total <= combined_max:
        return lead, story
    # Réduire le lead en priorité
    budget = combined_max - len(story) - 2
    if budget < 80:
        story = _truncate_at_sentence(story, max(120, combined_max // 3))
        budget = combined_max - len(story) - 2
    lead = _truncate_at_sentence(lead, max(80, budget))
    return lead, story
