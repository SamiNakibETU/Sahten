"""Évaluation offline du pipeline RAG sur un jeu d'or (golden set).

Usage (depuis la racine `v4/` avec dépendances + base peuplée)::

    set PYTHONPATH=backend
    python scripts/run_rag_eval.py --golden data/golden_eval_fr.json

Contrôles :
  * **retrieval** : au moins un ``expected_article_external_id`` apparaît dans
    les ``top_k`` premiers résultats rerankés (si la liste est non vide).
  * **réponse** : chaque chaîne de ``answer_must_contain`` apparaît dans le
    texte joint des phrases sourcées (insensible à la casse). Si la liste est
    vide, le test est ignoré pour ce critère.

Sortie : JSON sur stdout, code 0 si tout passe, 1 sinon.

Pour RAGAS complet (faithfulness, etc.), installer ``pip install -e .[eval]``
et brancher un appel ``ragas`` séparé — ce script reste volontairement léger.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

# Permet ``python scripts/run_rag_eval.py`` depuis v4/
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


async def _run_one(
    pipeline: Any,
    session: Any,
    item: dict[str, Any],
    *,
    top_k: int,
) -> dict[str, Any]:
    from app.rag.pipeline import RagPipeline

    if not isinstance(pipeline, RagPipeline):
        raise TypeError("pipeline")

    q = (item.get("query") or "").strip()
    expected = [int(x) for x in (item.get("expected_article_external_ids") or [])]
    must = [str(x) for x in (item.get("answer_must_contain") or [])]

    result = await pipeline.answer(session, q, session_id=None, conversation_history=None)
    reranked = result.reranked[:top_k]
    top_ids = [r.hit.article_external_id for r in reranked]

    retrieval_ok = True
    if expected:
        exp_set = set(expected)
        retrieval_ok = any(int(x) in exp_set for x in top_ids)

    text = " ".join(s.text for s in result.answer.answer_sentences)
    low = text.lower()
    missing = [m for m in must if m.lower() not in low]
    answer_ok = len(missing) == 0 if must else True

    return {
        "id": item.get("id"),
        "query": q,
        "retrieval_ok": retrieval_ok,
        "expected_article_external_ids": expected,
        "top_article_ids_sample": [int(x) for x in top_ids[:8]],
        "answer_ok": answer_ok,
        "answer_must_contain_missing": missing,
        "n_reranked": len(result.reranked),
        "confidence": result.answer.confidence,
        "timings_ms": result.timings_ms,
        "passed": retrieval_ok and answer_ok,
    }


async def main_async(args: argparse.Namespace) -> int:
    from app.db.base import get_sessionmaker
    from app.rag.pipeline import RagPipeline

    golden_path = Path(args.golden)
    if not golden_path.is_file():
        print(json.dumps({"error": f"fichier introuvable: {golden_path}"}), file=sys.stderr)
        return 1

    spec = json.loads(golden_path.read_text(encoding="utf-8"))
    items = spec.get("items") or []
    if not items:
        print(json.dumps({"error": "aucun item dans le golden set"}), file=sys.stderr)
        return 1

    pipeline = RagPipeline()
    sm = get_sessionmaker()
    rows: list[dict[str, Any]] = []
    async with sm() as session:
        for item in items:
            row = await _run_one(pipeline, session, item, top_k=args.top_k)
            rows.append(row)

    summary = {
        "golden": str(golden_path),
        "version": spec.get("version"),
        "top_k": args.top_k,
        "n_items": len(rows),
        "n_passed": sum(1 for r in rows if r.get("passed")),
        "results": rows,
    }
    summary["all_passed"] = summary["n_passed"] == summary["n_items"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["all_passed"] else 1


def main() -> None:
    p = argparse.ArgumentParser(description="Évalue le RAG sur un golden set JSON.")
    p.add_argument(
        "--golden",
        default="data/golden_eval_fr.json",
        help="Chemin vers le JSON (défaut: data/golden_eval_fr.json)",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Profondeur retrieval à considérer pour le hit attendu",
    )
    args = p.parse_args()
    try:
        raise SystemExit(asyncio.run(main_async(args)))
    except KeyboardInterrupt:
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
