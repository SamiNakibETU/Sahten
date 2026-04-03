"""
Constantes et petits helpers partagés par le retrieval (TF-IDF, alternatives, rerank).
Extrait de retriever.py pour lisibilité (phase 5 plan SOTA).
"""

from __future__ import annotations

from typing import List

from ..data.normalizers import normalize_text
from ..schemas.canonical import CanonicalCategory

GENERIC_QUERY_TOKENS = {
    "recette",
    "recettes",
    "comment",
    "faire",
    "avec",
    "plat",
    "plats",
    "cuisine",
    "libanais",
    "libanaise",
    "libanaises",
    "bonjour",
    "salut",
    "famille",
    "familial",
    "francais",
    "francaise",
    "français",
    "française",
    "typique",
    "typiques",
    "votre",
    "notre",
    "leurs",
    "idee",
    "idees",
    "envie",
    "quelque",
    "chose",
    "pour",
}

MATCH_REASON_KEYWORDS = {
    "shared_stewed_beef": {"boeuf", "beef", "viande", "viande hachee"},
    "shared_wrap_like_format": {"pain", "bread", "galette"},
    "shared_grilled_spiced_meat": {"poulet", "chicken", "agneau", "lamb", "boeuf"},
    "shared_rice_base": {"riz", "rice", "semoule", "moghrabieh"},
}

WEAK_SHARED_INGREDIENTS: frozenset[str] = frozenset(
    normalize_text(x)
    for x in (
        "oeuf",
        "oeufs",
        "lait",
        "beurre",
        "farine",
        "sucre",
        "sel",
        "poivre",
        "huile",
        "huile dolive",
        "ail",
        "oignon",
        "oignons",
        "eau",
        "creme",
        "crème",
        "citron",
        "vanille",
    )
)

_STRUCTURAL_INGREDIENTS: frozenset[str] = frozenset(
    normalize_text(x)
    for x in (
        "moghrabieh",
        "riz",
        "semoule",
        "poulet",
        "agneau",
        "boeuf",
        "viande",
        "viande hachee",
        "poisson",
        "halloumi",
        "akkawi",
        "labneh",
        "houmous",
        "taboule",
        "falafel",
        "freekeh",
        "lentille",
        "lentilles",
        "kafta",
        "kebbe",
        "warak enab",
        "manakish",
        "pain",
        "vermicelles",
        "fromage",
    )
)

SAVORY_INFERRED: frozenset[CanonicalCategory] = frozenset(
    {
        "plat_principal",
        "mezze_froid",
        "mezze_chaud",
        "entree",
        "salade",
        "soupe",
        "sauces",
    }
)


def ingredient_overlap_is_meaningful(shared_terms: List[str]) -> bool:
    norms = {normalize_text(t) for t in (shared_terms or []) if t}
    if not norms:
        return False
    if norms & _STRUCTURAL_INGREDIENTS:
        return True
    return bool(norms - WEAK_SHARED_INGREDIENTS)


def doc_category_matches_inferred(
    doc_cat: CanonicalCategory, inferred: CanonicalCategory
) -> bool:
    if inferred == "dessert":
        return doc_cat in ("dessert", "autre", "boisson")
    if inferred in SAVORY_INFERRED:
        return doc_cat != "dessert"
    return True
