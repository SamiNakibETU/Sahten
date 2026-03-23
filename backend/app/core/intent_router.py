"""
Intent Router (deterministic-first)
===================================

Classifies critical intents by rules before any LLM call.
LLM is used only for ambiguous cases.

Critical intents: recipe_specific, recipe_by_ingredient, menu, multi_recipe,
about_bot, off_topic, greeting, clarification.
"""

from __future__ import annotations

import re
from typing import List, Optional

from unidecode import unidecode

from .safety import (
    FOOD_KEYWORDS,
    OFF_TOPIC_STRONG,
    is_culinary_by_rules,
    is_off_topic_by_rules,
)
from ..data.dish_normalizer import dish_normalizer
from ..schemas.query_analysis import QueryAnalysis, SafetyCheck


# Greeting patterns
GREETING_WORDS = frozenset({
    "bonjour", "salut", "hello", "hi", "marhaba", "marhaba", "salam",
    "bonsoir", "coucou", "hey",
})

# Clarification / recipe_info patterns ("c'est quoi X", "qu'est-ce que X")
CLARIFICATION_STARTERS = (
    "c'est quoi", "quest ce que", "qu'est-ce que", "keske", "keskese",
    "parle-moi", "explique-moi", "dis-moi ce que", "quest-ce que",
)

# Synonymes de plats pour le résolveur exact (variantes fréquentes LLM / utilisateur)
DISH_ALIAS_VARIANTS: dict[str, List[str]] = {
    "houmous": ["houmous", "hummus", "hommos", "hoummos"],
    "hummus": ["hummus", "houmous"],
    "fattouch": ["fattouch", "fattoush", "fatouch", "fattouche"],
    "fattoush": ["fattoush", "fattouch"],
    "taboule": ["taboule", "taboulé", "tabbouleh", "taboulé"],
    "taboulé": ["taboulé", "taboule", "tabbouleh"],
}


def _strip_leading_articles(fragment: str) -> str:
    s = fragment.strip()
    lowered = s.lower()
    for prefix in (
        "du ", "de la ", "de l'", "de l’", "de ", "d'", "d’",
        "la ", "le ", "les ", "un ", "une ", "l'", "l’", "des ",
    ):
        if lowered.startswith(prefix):
            s = s[len(prefix) :].strip()
            lowered = s.lower()
    return s


def _dish_tail_after_recette(q_ascii: str) -> Optional[str]:
    if "recette" not in q_ascii:
        return None
    tail = q_ascii.split("recette", 1)[1].strip()
    tail = _strip_leading_articles(tail)
    if not tail:
        return None
    return " ".join(tail.split()[:6]).strip() or None


def _expand_dish_variants(dish: str) -> List[str]:
    key = unidecode(dish).lower().strip()
    out = {dish.strip(), key}
    for canon, variants in DISH_ALIAS_VARIANTS.items():
        if key == canon or canon in key or key in canon:
            out.update(variants)
    return [x for x in out if x]


def _recipe_specific_from_dish(dish: str, *, reasoning: str) -> QueryAnalysis:
    dish = dish.strip()
    canonical = dish_normalizer.normalize(dish) or dish
    alias_forms = dish_normalizer.get_all_aliases(canonical)
    variants = list(
        dict.fromkeys(
            [dish, canonical, *alias_forms, *_expand_dish_variants(dish), *_expand_dish_variants(canonical)]
        )
    )
    return QueryAnalysis(
        safety=SafetyCheck(is_safe=True, threat_type="none"),
        intent="recipe_specific",
        intent_confidence=0.92,
        is_culinary=True,
        dish_name=canonical,
        dish_name_variants=variants,
        reasoning=reasoning,
    )


def route_intent_deterministic(query: str) -> Optional[QueryAnalysis]:
    """
    Classify intent by deterministic rules. Returns QueryAnalysis when
    we can classify with high confidence, else None (fall back to LLM).
    """
    if not query or not isinstance(query, str):
        return None
    q_raw = query.strip()
    q = unidecode(q_raw).lower().strip()
    words = {unidecode(w).lower() for w in q.split()}

    # 0. Présentation du bot
    about_patterns = (
        "qui es-tu", "qui es tu", "qui etes-vous", "qui êtes-vous", "qui etes vous",
        "presente-toi", "presente toi", "presentes-toi",
        "c'est quoi sahten", "quest ce que sahten", "qu'est-ce que sahten",
        "tu es qui", "c'est qui sahten",
    )
    if any(p in q for p in about_patterns):
        return QueryAnalysis(
            safety=SafetyCheck(is_safe=True, threat_type="none"),
            intent="about_bot",
            intent_confidence=1.0,
            is_culinary=True,
            reasoning="IntentRouter: about_bot pattern",
        )

    # 1. Menu entrée / plat / dessert (avant les branches « requête courte »)
    if ("entree" in q or "entrée" in q_raw.lower()) and "plat" in q and "dessert" in q:
        return QueryAnalysis(
            safety=SafetyCheck(is_safe=True, threat_type="none"),
            intent="menu_composition",
            intent_confidence=0.95,
            is_culinary=True,
            recipe_count=3,
            reasoning="IntentRouter: menu composition keywords",
        )

    # 1b. « menu libanais », « repas libanais », « menu complet » (sans liste entrée/plat/dessert)
    if ("menu" in q or "repas" in q) and (
        "libanais" in q
        or "libanaise" in q
        or "libanaises" in q
        or "complet" in q
        or "type libanais" in q
    ):
        return QueryAnalysis(
            safety=SafetyCheck(is_safe=True, threat_type="none"),
            intent="menu_composition",
            intent_confidence=0.93,
            is_culinary=True,
            recipe_count=3,
            reasoning="IntentRouter: menu / repas libanais (3 temps)",
        )

    # 1c. Faim + contrainte temps / simplicité (évite « plat » comme nom de recette)
    if "faim" in q and any(
        w in q for w in ("rapide", "vite", "express", "facile", "simple", "pas long")
    ):
        return QueryAnalysis(
            safety=SafetyCheck(is_safe=True, threat_type="none"),
            intent="recipe_by_mood",
            intent_confidence=0.9,
            is_culinary=True,
            recipe_count=1,
            mood_tags=["rapide", "facile", "copieux"],
            reasoning="IntentRouter: faim + rapide/facile",
        )

    # 1d. Plat d'été / envie estivale
    if "plat" in q and (
        "ete" in q
        or "été" in q_raw.lower()
        or "estiv" in q
        or "summer" in q
    ):
        return QueryAnalysis(
            safety=SafetyCheck(is_safe=True, threat_type="none"),
            intent="recipe_by_mood",
            intent_confidence=0.88,
            is_culinary=True,
            recipe_count=1,
            mood_tags=["ete", "frais", "leger"],
            reasoning="IntentRouter: plat d'été",
        )

    # 2. Plusieurs mezze / idées chiffrées
    if ("mezze" in q or "mezzé" in q_raw.lower()) and any(
        w in q for w in ("plusieurs", "quelques", "divers", "varies", "variees", "plusieur")
    ):
        return QueryAnalysis(
            safety=SafetyCheck(is_safe=True, threat_type="none"),
            intent="multi_recipe",
            intent_confidence=0.88,
            is_culinary=True,
            recipe_count=4,
            category="mezze_froid",
            reasoning="IntentRouter: plusieurs mezze (sans chiffre)",
        )

    m_mezze = re.search(r"\b(\d+)\b", q)
    if m_mezze and "mezze" in q:
        n = int(m_mezze.group(1))
        n = min(10, max(2, n))
        return QueryAnalysis(
            safety=SafetyCheck(is_safe=True, threat_type="none"),
            intent="multi_recipe",
            intent_confidence=0.9,
            is_culinary=True,
            recipe_count=n,
            category="mezze_froid",
            reasoning="IntentRouter: N idées mezze",
        )

    # 3. Salutation simple (peu de mots)
    if len(words) <= 2 and words & GREETING_WORDS:
        return QueryAnalysis(
            safety=SafetyCheck(is_safe=True, threat_type="none"),
            intent="greeting",
            intent_confidence=1.0,
            is_culinary=True,
            reasoning="IntentRouter: greeting detected",
        )

    # 4. Hors-sujet fort
    if is_off_topic_by_rules(q_raw):
        return QueryAnalysis(
            safety=SafetyCheck(is_safe=True, threat_type="none"),
            intent="off_topic",
            intent_confidence=1.0,
            is_culinary=False,
            redirect_suggestion="Le vrai match c'est en cuisine ! Un kafta ?",
            reasoning="IntentRouter: off_topic by rules",
        )

    # 5. Clarification / définition
    for start in CLARIFICATION_STARTERS:
        if q.startswith(start) or f" {start} " in f" {q} ":
            tail = q.split(start, 1)[-1].strip()
            term = tail.split(".")[0].split("?")[0].strip()
            for prefix in ("le ", "la ", "du ", "de ", "un ", "une "):
                if term.startswith(prefix):
                    term = term[len(prefix) :].strip()
                    break
            if len(term) >= 2:
                return QueryAnalysis(
                    safety=SafetyCheck(is_safe=True, threat_type="none"),
                    intent="clarification",
                    intent_confidence=0.95,
                    is_culinary=True,
                    dish_name=term or q_raw,
                    reasoning="IntentRouter: clarification pattern",
                )

    # 6. Par ingrédient : « … avec … », « un plat avec du yaourt »
    if is_culinary_by_rules(q_raw) and "avec" in q:
        tail = q.split("avec", 1)[1].strip()
        tail = _strip_leading_articles(tail)
        if tail:
            first_tokens = [t.strip(" ,.;:!?") for t in tail.split()[:4] if t.strip(" ,.;:!?")]
            if first_tokens:
                return QueryAnalysis(
                    safety=SafetyCheck(is_safe=True, threat_type="none"),
                    intent="recipe_by_ingredient",
                    intent_confidence=0.88,
                    is_culinary=True,
                    ingredients=first_tokens,
                    reasoning="IntentRouter: avec [ingrédient]",
                )

    # 7. Recette précise : « recette (du) X », avant requête courte
    if is_culinary_by_rules(q_raw):
        dish_r = _dish_tail_after_recette(q)
        if dish_r:
            return _recipe_specific_from_dish(
                dish_r, reasoning="IntentRouter: recette X pattern"
            )

        if "comment faire" in q:
            tail = q.split("faire", 1)[1].strip()
            tail = _strip_leading_articles(tail)
            if tail:
                d = " ".join(tail.split()[:6]).strip()
                if d:
                    return _recipe_specific_from_dish(
                        d, reasoning="IntentRouter: comment faire X"
                    )
        if "comment preparer" in q:
            tail = q.split("preparer", 1)[1].strip()
            tail = _strip_leading_articles(tail)
            if tail:
                d = " ".join(tail.split()[:6]).strip()
                if d:
                    return _recipe_specific_from_dish(
                        d, reasoning="IntentRouter: comment preparer X"
                    )

        # 8. Requête courte = nom de plat probable
        tokens = q.strip().split()
        if len(tokens) <= 4:
            return _recipe_specific_from_dish(
                q_raw.strip(), reasoning="IntentRouter: short culinary query as recipe_specific"
            )

    return None
