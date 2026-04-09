"""
Intent Router (déterministe minimal)
===================================

Seuls les cas triviaux / sécurité à fort signal : le reste passe par QueryPlan (LLM).
"""

from __future__ import annotations

from typing import Optional

from unidecode import unidecode

from .query_plan_patterns import pattern_override_plan
from .safety import is_off_topic_by_rules
from ..schemas.query_analysis import QueryAnalysis, SafetyCheck
from ..schemas.query_plan_mapper import query_plan_to_analysis


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

def route_intent_deterministic(query: str) -> Optional[QueryAnalysis]:
    """
    Cas triviaux uniquement ; le culinaire ambigu → None (QueryPlan LLM).
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

    po = pattern_override_plan(q_raw)
    if po is not None:
        return query_plan_to_analysis(po)

    # 2b. Requête ultra-vague contenant "recette" sans nom de plat ni ingrédient
    # Ex : "recette", "une recette", "donne-moi une recette"
    # → recipe_by_mood (browse_corpus) immédiatement, jamais off_topic
    VAGUE_RECIPE_TRIGGERS = frozenset({"recette", "recettes"})
    STOP_WORDS = frozenset({"une", "un", "de", "la", "le", "du", "donne", "moi", "je", "veux", "voudrais", "avoir", "avoir"})
    if words & VAGUE_RECIPE_TRIGGERS and len(words) <= 5:
        non_recipe_words = words - VAGUE_RECIPE_TRIGGERS - STOP_WORDS
        if not non_recipe_words:
            return QueryAnalysis(
                safety=SafetyCheck(is_safe=True, threat_type="none"),
                intent="recipe_by_mood",
                intent_confidence=1.0,
                is_culinary=True,
                recipe_count=1,
                reasoning="IntentRouter: vague 'recette' query → recipe_by_mood with clarification",
            )

    # 2. Salutation simple (peu de mots)
    if len(words) <= 2 and words & GREETING_WORDS:
        return QueryAnalysis(
            safety=SafetyCheck(is_safe=True, threat_type="none"),
            intent="greeting",
            intent_confidence=1.0,
            is_culinary=True,
            reasoning="IntentRouter: greeting detected",
        )

    # 3. Hors-sujet fort
    if is_off_topic_by_rules(q_raw):
        return QueryAnalysis(
            safety=SafetyCheck(is_safe=True, threat_type="none"),
            intent="off_topic",
            intent_confidence=1.0,
            is_culinary=False,
            redirect_suggestion="Le vrai match c'est en cuisine ! Un kafta ?",
            reasoning="IntentRouter: off_topic by rules",
        )

    # 4. Clarification / définition
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

    return None
