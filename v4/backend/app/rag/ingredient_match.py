"""Correspondance ingrédient demandé ↔ chunks indexés (ingredients_list)."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Sequence

from ..llm.query_understanding import QueryPlan
from .reranker import RerankedHit
from .retriever import Hit

INGREDIENT_SLUG_ALIASES: dict[str, tuple[str, ...]] = {
    "concombre": ("concombre", "concombres"),
    "tomate": ("tomate", "tomates"),
    "poulet": ("poulet", "poulets"),
    "citron": ("citron", "citrons"),
    "aubergine": ("aubergine", "aubergines"),
    "courgette": ("courgette", "courgettes"),
    "pois-chiche": ("pois chiche", "pois chiches", "pois-chiche", "pois-chiches"),
    "pois-chiches": ("pois chiche", "pois chiches", "pois-chiche", "pois-chiches"),
}

_INGREDIENT_EXTRACT_RE = re.compile(
    r"(?i)\b(?:avec|au|aux|à base de)\s+"
    r"(?:du|de la|de l'|des|d'|de)\s*"
    r"([\wàâäéèêëïîôùûç\-]+)"
)
_INGREDIENT_RECETTE_RE = re.compile(
    r"(?i)\brecette(?:s)?\s+(?:au|aux|à base de|avec)\s+"
    r"(?:du|de la|de l'|des|d'|de)?\s*"
    r"([\wàâäéèêëïîôùûç\-]+)"
)
_KNOWN_ING_WORDS: dict[str, str] = {
    "concombre": "concombre",
    "concombres": "concombre",
    "tomate": "tomate",
    "tomates": "tomate",
    "poulet": "poulet",
    "poulets": "poulet",
    "citron": "citron",
    "citrons": "citron",
    "aubergine": "aubergine",
    "aubergines": "aubergine",
    "courgette": "courgette",
    "courgettes": "courgette",
    "halloumi": "halloumi",
    "boulgour": "boulgour",
    "menthe": "menthe",
    "persil": "persil",
    "yaourt": "yaourt",
}


def _ascii_slug(word: str) -> str:
    s = unicodedata.normalize("NFKD", word.strip().lower())
    s = s.encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or word.strip().lower()


def slug_search_terms(slug: str) -> tuple[str, ...]:
    raw = (slug or "").strip().lower()
    if raw in INGREDIENT_SLUG_ALIASES:
        return INGREDIENT_SLUG_ALIASES[raw]
    base = raw.replace("-", " ")
    terms = {raw, base}
    if base.endswith("s") and len(base) > 3:
        terms.add(base[:-1])
    else:
        terms.add(f"{base}s")
    return tuple(sorted(terms, key=len, reverse=True))


def text_contains_ingredient(text: str, terms: tuple[str, ...]) -> bool:
    low = (text or "").lower().replace("’", "'")
    for term in terms:
        t = term.lower().replace("-", " ")
        if re.search(rf"(?<![\wàâäéèêëïîôùûç-]){re.escape(t)}(?![\wàâäéèêëïîôùûç-])", low):
            return True
    return False


def chunk_confirms_ingredient(
    section_kind: str | None,
    chunk_text: str,
    slug: str,
    *,
    strict: bool = True,
) -> bool:
    sk = (section_kind or "").lower()
    allowed = ("ingredients_list",) if strict else ("ingredients_list", "recipe_summary", "anchor")
    if sk not in allowed:
        return False
    return text_contains_ingredient(chunk_text, slug_search_terms(slug))


def _word_to_slug(word: str) -> str | None:
    w = word.strip().lower().replace("’", "'")
    if w in _KNOWN_ING_WORDS:
        return _KNOWN_ING_WORDS[w]
    if len(w) < 3:
        return None
    return _ascii_slug(w)


def extract_ingredient_slugs_from_text(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    found: list[str] = []
    seen: set[str] = set()
    for pattern in (_INGREDIENT_EXTRACT_RE, _INGREDIENT_RECETTE_RE):
        for m in pattern.finditer(text):
            slug = _word_to_slug(m.group(1))
            if slug and slug not in seen:
                seen.add(slug)
                found.append(slug)
    for m in re.finditer(r"(?i)\b([\wàâäéèêëïîôùûç-]+)\b", text):
        slug = _word_to_slug(m.group(1))
        if slug and slug not in seen and slug in INGREDIENT_SLUG_ALIASES:
            seen.add(slug)
            found.append(slug)
    return found


def supplement_ingredient_slugs(
    plan: QueryPlan,
    user_query: str,
    conversation_history: str | None,
) -> QueryPlan:
    slugs = list(plan.ingredient_slugs or [])
    seen = set(slugs)
    blob = "\n".join(
        part for part in (conversation_history or "", user_query or "") if part.strip()
    )
    for slug in extract_ingredient_slugs_from_text(blob):
        if slug not in seen:
            seen.add(slug)
            slugs.append(slug)
    if slugs == (plan.ingredient_slugs or []):
        return plan
    return plan.model_copy(update={"ingredient_slugs": slugs})


def _article_ids_matching_slugs(
    items: Sequence[Hit | RerankedHit],
    slugs: list[str],
    *,
    strict: bool,
) -> set[int]:
    by_article: dict[int, list[Hit]] = defaultdict(list)
    for item in items:
        hit = item.hit if isinstance(item, RerankedHit) else item
        by_article[int(hit.article_external_id)].append(hit)

    valid: set[int] = set()
    for aid, article_hits in by_article.items():
        ok = True
        for slug in slugs:
            if not any(
                chunk_confirms_ingredient(h.section_kind, h.chunk_text, slug, strict=strict)
                for h in article_hits
            ):
                ok = False
                break
        if ok:
            valid.add(aid)
    return valid


def filter_hits_by_ingredient_slugs(
    hits: list[Hit],
    slugs: list[str] | None,
    *,
    strict: bool = True,
) -> list[Hit]:
    if not slugs:
        return hits
    valid = _article_ids_matching_slugs(hits, slugs, strict=strict)
    if not valid and strict:
        valid = _article_ids_matching_slugs(hits, slugs, strict=False)
    if not valid:
        return []
    return [h for h in hits if int(h.article_external_id) in valid]


def filter_reranked_by_ingredient_slugs(
    reranked: list[RerankedHit],
    slugs: list[str] | None,
    *,
    strict: bool = True,
) -> list[RerankedHit]:
    if not slugs:
        return reranked
    valid = _article_ids_matching_slugs(reranked, slugs, strict=strict)
    if not valid and strict:
        valid = _article_ids_matching_slugs(reranked, slugs, strict=False)
    if not valid:
        return []
    return [r for r in reranked if int(r.hit.article_external_id) in valid]


def article_external_id_for_recipe_card(
    recipe_card,
    hits: list[RerankedHit],
) -> int | None:
    if recipe_card is None:
        return None
    want = set(recipe_card.source_chunk_ids or [])
    for h in hits:
        if h.hit.chunk_id in want:
            return int(h.hit.article_external_id)
    return None


def recipe_card_matches_required_ingredients(
    recipe_card,
    hits: list[RerankedHit],
    slugs: list[str] | None,
    *,
    strict: bool = True,
) -> bool:
    if not slugs or recipe_card is None:
        return True
    aid = article_external_id_for_recipe_card(recipe_card, hits)
    if aid is None:
        return False
    article_hits = [h.hit for h in hits if int(h.hit.article_external_id) == aid]
    for slug in slugs:
        if not any(
            chunk_confirms_ingredient(h.section_kind, h.chunk_text, slug, strict=strict)
            for h in article_hits
        ):
            if strict and any(
                chunk_confirms_ingredient(
                    h.section_kind, h.chunk_text, slug, strict=False
                )
                for h in article_hits
            ):
                continue
            return False
    return True
