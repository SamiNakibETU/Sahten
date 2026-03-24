"""
Détection déterministe : demandes de recettes par saison / moment / humeur.

Évite de classer « recette pour l'hiver » en recipe_specific (faux plat « pour l'hiver »).
"""

from __future__ import annotations

import re
from typing import Optional

from unidecode import unidecode

from .safety import is_culinary_by_rules
from ..schemas.query_analysis import QueryAnalysis, SafetyCheck

# Mots seuls après « recette » = contrainte humeur, pas nom de plat
# Après un mot-contrainte (léger, rapide…), ces jetons seuls ne forment pas un plat
_CONTEXT_ONLY_AFTER_MOOD: frozenset[str] = frozenset(
    {
        "pour",
        "ce",
        "cette",
        "ca",
        "cela",
        "le",
        "la",
        "les",
        "un",
        "une",
        "du",
        "de",
        "des",
        "l",
        "a",
        "au",
        "aux",
        "en",
        "et",
        "ou",
        "d",
        "soir",
        "midi",
        "matin",
        "demain",
        "hier",
        "maintenant",
        "plus",
        "tard",
    }
)

_SEASON_SUBSTRINGS: tuple[str, ...] = (
    "hiver",
    "ete",
    "automne",
    "printemps",
    "hivernal",
    "estival",
)

_MOOD_CONSTRAINT_WORDS: frozenset[str] = frozenset(
    {
        "rapide",
        "vite",
        "facile",
        "faciles",
        "simple",
        "express",
        "leger",
        "léger",
        "legere",
        "légère",
        "reconfortant",
        "reconfortante",
        "festif",
        "festive",
        "frais",
        "fraiche",
        "fraîche",
    }
)


def _strip_leading_articles(fragment: str) -> str:
    s = fragment.strip()
    lowered = s.lower()
    for prefix in (
        "du ",
        "de la ",
        "de l'",
        "de l’",
        "de ",
        "d'",
        "d’",
        "la ",
        "le ",
        "les ",
        "un ",
        "une ",
        "l'",
        "l’",
        "des ",
    ):
        if lowered.startswith(prefix):
            s = s[len(prefix) :].strip()
            lowered = s.lower()
    return s


def has_substantive_dish_after_recette(q_ascii: str) -> bool:
    """
    True si la queue après « recette » ressemble à un plat (ex. taboulé, pas seulement « rapide »).
    Utilisé pour ne pas classer « recette taboulé pour ce soir » en simple mood « ce soir ».
    """
    q = unidecode(q_ascii).lower()
    if "recette" not in q:
        return False
    tail = q.split("recette", 1)[1].strip()
    tail = _strip_leading_articles(tail)
    if not tail:
        return False
    if tail_is_mood_or_season_context_only(tail):
        return False
    tokens = [re.sub(r"^[^\w]+|[^\w]+$", "", unidecode(w).lower()) for w in tail.split() if w.strip()]
    if not tokens:
        return False

    def _ctx_or_season_token(t: str) -> bool:
        if t in _CONTEXT_ONLY_AFTER_MOOD:
            return True
        return any(s in t for s in _SEASON_SUBSTRINGS)

    if len(tokens) >= 2 and tokens[0] in _MOOD_CONSTRAINT_WORDS:
        if all(_ctx_or_season_token(t) for t in tokens[1:]):
            return False

    # Un seul token : soit contrainte (rapide), soit nom de plat (taboulé)
    if len(tokens) == 1:
        return tokens[0] not in _MOOD_CONSTRAINT_WORDS
    # « taboulé pour ce soir », « poulet au citron » : le premier mot n’est typiquement pas une contrainte seule
    if tokens[0] in _MOOD_CONSTRAINT_WORDS:
        return len(tokens) > 2
    return True


# Queue après « recette » / « comment faire » qui n’est pas un nom de plat
_MOOD_TAIL_RES = (
    re.compile(r"^pour\s+l['\u2019]?hiver\b", re.I),
    re.compile(r"^pour\s+l['\u2019]?ete\b", re.I),
    re.compile(r"^pour\s+l['\u2019]?automne\b", re.I),
    re.compile(r"^pour\s+le\s+printemps\b", re.I),
    re.compile(r"^pour\s+ce\s+soir\b", re.I),
    re.compile(r"^pour\s+ce\s+midi\b", re.I),
    re.compile(r"^pour\s+le\s+midi\b", re.I),
    re.compile(r"^pour\s+le\s+dejeuner\b", re.I),
    re.compile(r"^pour\s+le\s+petit[-\s]?dejeuner\b", re.I),
    re.compile(r"^pour\s+le\s+brunch\b", re.I),
    re.compile(r"^pour\s+le\s+soir\b", re.I),
    re.compile(r"^pour\s+un\s+soir\b", re.I),
    re.compile(r"^pour\s+le\s+week[-\s]?end\b", re.I),
    re.compile(r"^pour\s+dimanche\b", re.I),
    re.compile(r"^en\s+hiver\b", re.I),
    re.compile(r"^en\s+ete\b", re.I),
    re.compile(r"^en\s+automne\b", re.I),
    re.compile(r"^au\s+printemps\b", re.I),
    re.compile(r"^d['\u2019]?hiver\b", re.I),
    re.compile(r"^d['\u2019]?ete\b", re.I),
)


def tail_is_mood_or_season_context_only(fragment: str) -> bool:
    """True si le fragment (souvent la queue après « recette ») décrit un contexte, pas un plat."""
    if not fragment or not fragment.strip():
        return True
    t = unidecode(fragment).lower().strip()
    t = re.sub(r"[?!.,;:]+$", "", t).strip()
    if not t:
        return True
    for rx in _MOOD_TAIL_RES:
        if rx.match(t):
            return True
    # Seulement saison / moment (peu de tokens)
    tokens = t.split()
    if len(tokens) <= 3:
        season_words = {"hiver", "ete", "automne", "printemps", "hivernal", "hivernale", "estival", "estivale"}
        if tokens and all(unidecode(w).lower().strip("?!.,'") in season_words for w in tokens):
            return True
    return False


def _recipe_discovery_signal(q: str) -> bool:
    """L’utilisateur cherche une ou des recettes (vs définition d’un ingrédient)."""
    return any(
        x in q
        for x in (
            "recette",
            "recettes",
            "plat",
            "plats",
            "idee",
            "idée",
            "idees",
            "idées",
            "suggestion",
            "suggestions",
            "envie",
            "propose",
            "on mange",
            "que cuisiner",
            "quoi cuisiner",
            "comment faire",
            "comment preparer",
            "que faire manger",
        )
    )


def try_recipe_by_mood_or_season(q_ascii: str, q_raw: str) -> Optional[QueryAnalysis]:
    """
    recipe_by_mood quand la requête ancre saison / moment / humeur + intention recette.

    Retourne None si aucune ancre détectée (le routeur plat / recette précise peut s’appliquer).
    """
    if not is_culinary_by_rules(q_raw):
        return None
    q = q_ascii.strip().lower()
    if not _recipe_discovery_signal(q):
        return None

    # Ne pas prendre « hiver / automne … » pour une recette déjà centrée sur un plat nommé
    if "recette" in q and has_substantive_dish_after_recette(q_ascii):
        return None

    safe = SafetyCheck(is_safe=True, threat_type="none")

    if any(w in q for w in ("hiver", "hivernal", "hivernale", "hivernaux")):
        return QueryAnalysis(
            safety=safe,
            intent="recipe_by_mood",
            intent_confidence=0.91,
            is_culinary=True,
            recipe_count=1,
            mood_tags=["hiver", "reconfortant", "chaud"],
            reasoning="IntentRouter: recette / plat + hiver",
        )
    if "automne" in q:
        return QueryAnalysis(
            safety=safe,
            intent="recipe_by_mood",
            intent_confidence=0.88,
            is_culinary=True,
            recipe_count=1,
            mood_tags=["reconfortant", "traditionnel", "chaud"],
            reasoning="IntentRouter: recette + automne",
        )
    if "printemps" in q:
        return QueryAnalysis(
            safety=safe,
            intent="recipe_by_mood",
            intent_confidence=0.88,
            is_culinary=True,
            recipe_count=1,
            mood_tags=["frais", "leger", "traditionnel"],
            reasoning="IntentRouter: recette + printemps",
        )
    if any(w in q for w in ("ete", "estiv", "summer")) or "été" in q_raw.lower():
        return QueryAnalysis(
            safety=safe,
            intent="recipe_by_mood",
            intent_confidence=0.9,
            is_culinary=True,
            recipe_count=1,
            mood_tags=["ete", "frais", "leger"],
            reasoning="IntentRouter: recette + été / estival",
        )

    # Ci-dessous : humeur / moment — éviter d’écraser « recette [plat] … »
    if has_substantive_dish_after_recette(q_ascii):
        return None

    # Léger / frais avant « ce soir » pour « idée légère pour ce soir »
    if any(w in q for w in ("leger", "léger", "pas lourd", "fraicheur", "rafraichissant")):
        return QueryAnalysis(
            safety=safe,
            intent="recipe_by_mood",
            intent_confidence=0.84,
            is_culinary=True,
            recipe_count=1,
            mood_tags=["leger", "frais", "rapide"],
            reasoning="IntentRouter: envie légère / fraîche",
        )
    if any(w in q for w in ("reconfortant", "reconfort", "cocooning", "chaleureux", "chaleureuse")):
        return QueryAnalysis(
            safety=safe,
            intent="recipe_by_mood",
            intent_confidence=0.85,
            is_culinary=True,
            recipe_count=1,
            mood_tags=["reconfortant", "chaud", "copieux"],
            reasoning="IntentRouter: envie réconfortante",
        )
    if any(w in q for w in ("festif", "fete", "fête", "convivial", "partager", "buffet")):
        return QueryAnalysis(
            safety=safe,
            intent="recipe_by_mood",
            intent_confidence=0.84,
            is_culinary=True,
            recipe_count=1,
            mood_tags=["convivial", "festif", "copieux", "traditionnel"],
            reasoning="IntentRouter: recette festive / conviviale",
        )
    if any(
        w in q
        for w in (
            "ce midi",
            "pour ce midi",
            "pour le midi",
            "dejeuner",
            "déjeuner",
            "petit dejeuner",
            "petit-dejeuner",
            "brunch",
        )
    ):
        return QueryAnalysis(
            safety=safe,
            intent="recipe_by_mood",
            intent_confidence=0.85,
            is_culinary=True,
            recipe_count=1,
            mood_tags=["rapide", "frais", "facile"],
            reasoning="IntentRouter: recette + midi / déjeuner / brunch",
        )
    if "ce soir" in q or "pour ce soir" in q or "pour le soir" in q:
        return QueryAnalysis(
            safety=safe,
            intent="recipe_by_mood",
            intent_confidence=0.86,
            is_culinary=True,
            recipe_count=1,
            mood_tags=["rapide", "copieux", "facile"],
            reasoning="IntentRouter: recette + ce soir / soir",
        )
    if any(w in q for w in ("week-end", "weekend", "dimanche", "samedi")):
        return QueryAnalysis(
            safety=safe,
            intent="recipe_by_mood",
            intent_confidence=0.83,
            is_culinary=True,
            recipe_count=1,
            mood_tags=["convivial", "festif", "traditionnel", "copieux"],
            reasoning="IntentRouter: recette + week-end / dimanche",
        )
    if any(w in q for w in ("rapide", "vite", "express", "pas long", "15 min", "30 min", "trente min")):
        return QueryAnalysis(
            safety=safe,
            intent="recipe_by_mood",
            intent_confidence=0.87,
            is_culinary=True,
            recipe_count=1,
            mood_tags=["rapide", "facile"],
            reasoning="IntentRouter: recette rapide",
        )
    if any(w in q for w in ("pas cher", "economique", "economiques", "bon marche", "petit budget")):
        return QueryAnalysis(
            safety=safe,
            intent="recipe_by_mood",
            intent_confidence=0.82,
            is_culinary=True,
            recipe_count=1,
            mood_tags=["facile", "rapide", "traditionnel"],
            reasoning="IntentRouter: recette économique / petit budget",
        )
    return None
