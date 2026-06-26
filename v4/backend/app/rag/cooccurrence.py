"""Voisinage culinaire ingrédient↔ingrédient (lecture du graphe offline).

Lit data/ingredient_cooccurrence.json (généré par scripts/build_cooccurrence.py).
FONDATION / diagnostic — volontairement **non branché** au retrieval pour
l'instant (cf. docs/metadata-graph-sota.md : évaluer la géométrie avant
d'intégrer, recommandation de l'audit Epicure). Usage prévu plus tard :
« avec quoi cuisiner X », « même ingrédient, autre registre », substitutions.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_PATH = Path(__file__).resolve().parents[3] / "data" / "ingredient_cooccurrence.json"


@lru_cache(maxsize=1)
def _graph() -> dict[str, Any]:
    try:
        return json.loads(_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {"neighbors": {}}


def related_ingredients(slug: str, k: int = 5) -> list[str]:
    """Voisins culinaires d'un ingrédient (NPMI décroissant), [] si inconnu."""
    nb = _graph().get("neighbors", {}).get((slug or "").strip().lower(), [])
    return [d["ingredient"] for d in nb[:k] if d.get("ingredient")]


def related_with_scores(slug: str, k: int = 5) -> list[tuple[str, float]]:
    nb = _graph().get("neighbors", {}).get((slug or "").strip().lower(), [])
    return [(d["ingredient"], float(d.get("npmi", 0.0))) for d in nb[:k]]
