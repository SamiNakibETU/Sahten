"""
Safety Gate & Intent Router (deterministic)
==========================================

Runs BEFORE the LLM to guarantee:
- Jailbreak/injection blocks
- Culinary vs off-topic routing

LLM is auxiliary; critical routing decisions are deterministic.
"""

from dataclasses import dataclass
from typing import Optional, Set
from unidecode import unidecode


@dataclass
class SafetyResult:
    """Result of deterministic safety check."""
    blocked: bool
    threat_type: str = "none"
    redirect_suggestion: Optional[str] = None


# Jailbreak / injection markers (blocked before any LLM call)
INJECTION_MARKERS = frozenset({
    "ignore previous instructions", "ignore tes instructions",
    "oublie tes instructions", "forget your instructions",
    "you are now", "tu es maintenant", "tu es desormais",
    "system prompt", "montre ton prompt", "show me your prompt",
    "prompt systeme", "dan mode", "jailbreak", "bypass",
    "[system]", "[inst]", "[INST]", "###", "%%",
    "disregard", "override", "new instructions",
})


# Food-related keywords: if present, query is NEVER off_topic
FOOD_KEYWORDS = frozenset({
    "couscous", "paella", "pizza", "risotto", "sushi", "burger", "hamburger",
    "curry", "ramen", "lasagne", "lasagna", "pasta", "tacos", "pates", "pâtes",
    "boeuf", "bœuf", "beef", "poulet", "viande", "poisson", "agneau", "lamb",
    "salade", "soupe", "recette", "plat", "cuisiner", "manger", "diner", "dejeuner",
    "gateau", "gâteau", "crepe", "crêpe", "quiche", "carbonara", "bolognaise",
    "tiramisu", "fondant", "tarte", "ravioli", "gnocchi", "spaghetti",
    "nouilles", "noodle", "houmous", "hummus", "taboulé", "taboule", "tabbouleh",
})


# Non-culinary topics: only these can be off_topic
OFF_TOPIC_STRONG = frozenset({
    "météo", "meteo", "weather", "heure", "politique", "football", "sport",
    "news", "actualite", "actualité", "president", "président", "election",
    "élection", "match", "score", "resultat", "résultat", "classement",
})


def safety_gate_check(query: str) -> SafetyResult:
    """
    Deterministic safety check before any LLM call.
    Blocks obvious injection/jailbreak attempts.
    """
    if not query or not isinstance(query, str):
        return SafetyResult(blocked=False)
    q = unidecode(query).lower().strip()
    for marker in INJECTION_MARKERS:
        if marker in q:
            return SafetyResult(
                blocked=True,
                threat_type="injection",
                redirect_suggestion="On reste en cuisine ! Tu veux une recette libanaise ?",
            )
    return SafetyResult(blocked=False)


def is_culinary_by_rules(query: str) -> bool:
    """
    Deterministic: query contains food-related content.
    Used to override LLM off_topic when False.
    """
    if not query:
        return False
    words = {unidecode(w).lower() for w in query.split()}
    return bool(words & FOOD_KEYWORDS)


def is_off_topic_by_rules(query: str) -> bool:
    """
    Deterministic: query is clearly non-culinary (météo, sport, etc).
    Only returns True for strong off-topic signals.
    """
    if not query:
        return False
    q_low = unidecode(query).lower()
    # Phrases claires (évite les faux positifs type « matcha »)
    off_phrases = (
        "score du match",
        "resultat du match",
        "résultat du match",
        "match de football",
        "qui est le president",
        "qui est le président",
    )
    if any(p in q_low for p in off_phrases):
        return not is_culinary_by_rules(query)

    words = {unidecode(w).lower() for w in query.split()}
    # Must have off-topic word AND no food word
    return bool(words & OFF_TOPIC_STRONG) and not (words & FOOD_KEYWORDS)


def should_override_to_recipe_specific(intent: str, is_culinary: bool, query: str) -> bool:
    """
    If LLM said off_topic but query has food keyword, override to recipe_specific.
    """
    if intent != "off_topic" and is_culinary:
        return False
    return is_culinary_by_rules(query)
