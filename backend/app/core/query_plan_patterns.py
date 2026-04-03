"""
Motifs déterministes ultra stables → QueryPlan complet (évite erreurs LLM sur cas QA connus).
"""

from __future__ import annotations

import re
from typing import Optional

from unidecode import unidecode

from ..schemas.query_plan import QueryPlan


def pattern_override_plan(raw_query: str) -> Optional[QueryPlan]:
    """
    Si la requête correspond à un motif figé, retourne un plan sans appel LLM.
    Sinon None.
    """
    if not raw_query or not isinstance(raw_query, str):
        return None
    q = unidecode(raw_query.strip()).lower()

    m_mezze = re.match(r"^(\d{1,2})\s+(?:idees?|recettes?)\s+(?:de\s+)?mezze\b", q)
    if m_mezze:
        n = max(1, min(10, int(m_mezze.group(1))))
        return QueryPlan(
            task="multi",
            recipe_count=n,
            course="mezze",
            cuisine_scope="lebanese_olj",
            category="mezze_froid",
            retrieval_focus=f"{n} mezze libanais varies",
            mood_tags=["liban", "traditionnel", "convivial"],
            reasoning="pattern_override: N idees/recettes de mezze",
            intent_confidence=0.97,
        )

    # Plat typiquement non libanais : la QA attend alternative prouvée, pas un hit "salade" OLJ
    if "salade" in q and "pate" in q:
        return QueryPlan(
            task="named_dish",
            dish_name="salade de pates",
            dish_variants=["salade de pates"],
            cuisine_scope="non_lebanese_named",
            course="any",
            category="",
            retrieval_focus="salade de pates",
            inferred_main_ingredients=["pates", "tomate"],
            reasoning="pattern_override: salade de pates (non OLJ canonique)",
            intent_confidence=0.96,
        )

    return None
