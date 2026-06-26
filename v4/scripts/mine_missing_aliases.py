"""Squelette — mineur d'alias manquants (agent auto-améliorant, PARTIE 2).

Lit les requêtes en échec en prod (traces 'recipe_not_found' + avis négatifs),
propose des alias via LLM, les VALIDE contre l'API live, et n'écrit que les alias
validés sans régression. Voir docs/alias-self-improving-agent.md.

Statut : SQUELETTE (les TODO marquent les points d'intégration). Volontairement
conservateur : propose, valide, met en file de revue ; n'auto-merge rien sans
golden set vert.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
DISHES = _ROOT / "data" / "aliases_dishes.json"
REVIEW_QUEUE = _ROOT / "data" / "aliases_review_queue.json"


async def collect_failed_queries(limit: int) -> list[str]:
    """Requêtes réelles en échec : recipe_not_found + avis négatifs.

    TODO: brancher analytics_store.get_traces() et la liste Redis
    'sahten:feedback:negative_reasons' (get_feedback_stats()).
    """
    # from backend.app.analytics_store import get_traces, get_feedback_stats
    # traces = await get_traces(limit) ; fb = await get_feedback_stats()
    # -> extraire les `query` où recipe_not_found / confidence basse / "Non"
    raise NotImplementedError("brancher analytics_store")


def propose_alias(term: str, model: str = "gpt-4.1-nano") -> dict[str, Any]:
    """LLM propose forme canonique + injection candidate pour un terme orphelin.

    TODO: appel structured-output (réutiliser le client de scripts/benchmark_models).
    Retourne {canonical, inject:[...], rationale}.
    """
    raise NotImplementedError("brancher un appel LLM structured-output")


def validate_against_api(term: str, inject: list[str], base_url: str) -> dict[str, Any]:
    """Oracle = API live. Réutilise la logique de scripts/validate_aliases.

    Retourne {target, rank_before, rank_after, accepted: bool}.
    """
    # from validate_aliases import _post, _sources, _rank
    raise NotImplementedError("réutiliser validate_aliases._post/_rank")


def is_regression_free() -> bool:
    """Golden set + tests alias verts ?

    TODO: subprocess scripts/run_rag_eval.py + pytest tests/test_query_aliases.py.
    """
    raise NotImplementedError("brancher golden runner + pytest")


def enqueue_for_review(entry: dict[str, Any]) -> None:
    queue = []
    if REVIEW_QUEUE.is_file():
        queue = json.loads(REVIEW_QUEUE.read_text(encoding="utf-8"))
    queue.append(entry)
    REVIEW_QUEUE.write_text(json.dumps(queue, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Mineur d'alias manquants (squelette).")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--base-url", default="https://web-sahtein-19-04-staging.up.railway.app")
    p.add_argument("--confidence-threshold", type=float, default=0.8)
    p.parse_args()
    print(
        "Squelette. Étapes : collect_failed_queries -> propose_alias -> "
        "validate_against_api -> (accepted & is_regression_free ? écrire dans "
        "aliases_dishes.json : enqueue_for_review). Voir "
        "docs/alias-self-improving-agent.md."
    )


if __name__ == "__main__":
    main()
