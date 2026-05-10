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


def _evaluate_item(
    item: dict[str, Any],
    *,
    top_ids: list[int],
    answer_sentences: list[dict[str, Any]],
    follow_up: str,
    recipe_card: dict[str, Any] | None,
    recipe_card_secondary: dict[str, Any] | None,
    confidence: float,
    n_reranked: int,
    timings_ms: dict[str, int] | None,
) -> dict[str, Any]:
    q = (item.get("query") or "").strip()
    expected = [int(x) for x in (item.get("expected_article_external_ids") or [])]
    must = [str(x) for x in (item.get("answer_must_contain") or [])]
    must_not = [str(x) for x in (item.get("answer_must_not_contain") or [])]
    any_contains = [str(x) for x in (item.get("answer_any_contains") or [])]
    answer_prefix = str(item.get("answer_prefix") or "").strip()
    follow_up_must = [str(x) for x in (item.get("follow_up_must_contain") or [])]
    max_recipe_cards = item.get("max_recipe_cards")
    require_recipe_card = bool(item.get("require_recipe_card", False))
    require_chef_card = bool(item.get("require_chef_card", False))

    retrieval_ok = True
    if expected:
        exp_set = set(expected)
        retrieval_ok = any(int(x) in exp_set for x in top_ids)

    sentence_texts = [str(s.get("text") or "") for s in answer_sentences]
    text = " ".join(sentence_texts)
    low = text.lower()
    missing = [m for m in must if m.lower() not in low]
    forbidden = [m for m in must_not if m.lower() in low]
    any_contains_ok = True
    if any_contains:
        any_contains_ok = any(m.lower() in low for m in any_contains)
    prefix_ok = True
    if answer_prefix:
        prefix_ok = low.startswith(answer_prefix.lower())
    follow_up_low = (follow_up or "").strip().lower()
    follow_up_missing = [m for m in follow_up_must if m.lower() not in follow_up_low]
    recipe_cards_count = int(recipe_card is not None) + int(recipe_card_secondary is not None)
    recipe_cards_ok = True
    if max_recipe_cards is not None:
        recipe_cards_ok = recipe_cards_count <= int(max_recipe_cards)
    require_recipe_card_ok = (
        (recipe_card is not None) if require_recipe_card else True
    )
    require_chef_card_ok = (
        bool(item.get("chef_card_present", False)) if require_chef_card else True
    )
    answer_ok = (
        (len(missing) == 0 if must else True)
        and not forbidden
        and any_contains_ok
        and prefix_ok
        and not follow_up_missing
        and recipe_cards_ok
        and require_recipe_card_ok
        and require_chef_card_ok
    )

    return {
        "id": item.get("id"),
        "query": q,
        "retrieval_ok": retrieval_ok,
        "expected_article_external_ids": expected,
        "top_article_ids_sample": [int(x) for x in top_ids[:8]],
        "answer_ok": answer_ok,
        "answer_must_contain_missing": missing,
        "answer_must_not_contain_found": forbidden,
        "answer_any_contains_ok": any_contains_ok,
        "answer_prefix_ok": prefix_ok,
        "follow_up_must_contain_missing": follow_up_missing,
        "recipe_cards_count": recipe_cards_count,
        "max_recipe_cards_ok": recipe_cards_ok,
        "require_recipe_card_ok": require_recipe_card_ok,
        "require_chef_card_ok": require_chef_card_ok,
        "n_reranked": n_reranked,
        "confidence": confidence,
        "timings_ms": timings_ms or {},
        "passed": retrieval_ok and answer_ok,
    }


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
    result = await pipeline.answer(session, q, session_id=None, conversation_history=None)
    reranked = result.reranked[:top_k]
    top_ids = [int(r.hit.article_external_id) for r in reranked]
    answer_sentences = [
        {"text": s.text, "source_chunk_ids": s.source_chunk_ids}
        for s in result.answer.answer_sentences
    ]
    item2 = dict(item)
    item2["chef_card_present"] = result.answer.chef_card is not None
    return _evaluate_item(
        item2,
        top_ids=top_ids,
        answer_sentences=answer_sentences,
        follow_up=result.answer.follow_up or "",
        recipe_card=result.answer.recipe_card.model_dump(mode="json") if result.answer.recipe_card else None,
        recipe_card_secondary=(
            result.answer.recipe_card_secondary.model_dump(mode="json")
            if result.answer.recipe_card_secondary
            else None
        ),
        confidence=float(result.answer.confidence or 0.0),
        n_reranked=len(result.reranked),
        timings_ms=result.timings_ms,
    )


async def _run_one_http(
    base_url: str,
    item: dict[str, Any],
    *,
    top_k: int,
    timeout_s: int,
    session_prefix: str,
) -> dict[str, Any]:
    import requests

    q = (item.get("query") or "").strip()
    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "query": q,
        "session_id": f"{session_prefix}-{(item.get('id') or 'case')}",
    }

    def _post() -> requests.Response:
        return requests.post(url, json=payload, timeout=timeout_s)

    try:
        resp = await asyncio.to_thread(_post)
    except Exception as exc:  # noqa: BLE001
        return {
            "id": item.get("id"),
            "query": q,
            "passed": False,
            "error": f"http_error: {exc}",
        }
    if resp.status_code >= 400:
        return {
            "id": item.get("id"),
            "query": q,
            "passed": False,
            "error": f"http_status_{resp.status_code}",
            "body_sample": resp.text[:300],
        }
    try:
        body = resp.json()
    except Exception as exc:  # noqa: BLE001
        return {
            "id": item.get("id"),
            "query": q,
            "passed": False,
            "error": f"invalid_json: {exc}",
            "body_sample": resp.text[:300],
        }

    sources = body.get("sources") or []
    top_ids = [int(s.get("article_external_id")) for s in sources[:top_k] if s.get("article_external_id") is not None]
    item2 = dict(item)
    item2["chef_card_present"] = body.get("chef_card") is not None
    return _evaluate_item(
        item2,
        top_ids=top_ids,
        answer_sentences=body.get("answer_sentences") or [],
        follow_up=str(body.get("follow_up") or ""),
        recipe_card=body.get("recipe_card"),
        recipe_card_secondary=body.get("recipe_card_secondary"),
        confidence=float(body.get("confidence") or 0.0),
        n_reranked=len(sources),
        timings_ms=body.get("timings_ms") or {},
    )


async def main_async(args: argparse.Namespace) -> int:
    golden_path = Path(args.golden)
    if not golden_path.is_file():
        print(json.dumps({"error": f"fichier introuvable: {golden_path}"}), file=sys.stderr)
        return 1

    spec = json.loads(golden_path.read_text(encoding="utf-8"))
    items = spec.get("items") or []
    if not items:
        print(json.dumps({"error": "aucun item dans le golden set"}), file=sys.stderr)
        return 1

    rows: list[dict[str, Any]] = []
    if args.base_url:
        for item in items:
            row = await _run_one_http(
                args.base_url,
                item,
                top_k=args.top_k,
                timeout_s=args.timeout,
                session_prefix=args.session_prefix,
            )
            rows.append(row)
    else:
        from app.db.base import get_sessionmaker
        from app.rag.pipeline import RagPipeline

        pipeline = RagPipeline()
        sm = get_sessionmaker()
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
    p.add_argument(
        "--base-url",
        default="",
        help="URL de base API pour évaluer en live (ex: https://...up.railway.app). Vide = mode DB locale.",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=40,
        help="Timeout HTTP par requête en secondes (mode --base-url).",
    )
    p.add_argument(
        "--session-prefix",
        default="eval",
        help="Préfixe session_id pour les appels live (mode --base-url).",
    )
    args = p.parse_args()
    try:
        raise SystemExit(asyncio.run(main_async(args)))
    except KeyboardInterrupt:
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
