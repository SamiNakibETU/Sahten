#!/usr/bin/env python3
"""
Batch RAG / intent diagnostic sur le jeu golden (Phase 0).

Usage (depuis backend/) :
  python scripts/run_rag_golden.py
  python scripts/run_rag_golden.py --yaml ../data/rag_golden_queries.yaml --out report.jsonl
  python scripts/run_rag_golden.py --offline   # sans appels LLM (analyse fallback)

Sans clés API, l’analyse est en fallback ; le retrieval lexical fonctionne si olj_canonical.json est présent.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

# Racine backend sur sys.path
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


async def run_one(bot, text: str, *, debug: bool) -> dict:
    from app.bot import SahtenBot

    assert isinstance(bot, SahtenBot)
    response, dbg, trace = await bot.chat(text, debug=debug, session_id=None)
    row: dict = {
        "user": text,
        "response_type": response.response_type,
        "intent": response.intent_detected,
        "recipe_count": response.recipe_count,
        "primary_titles": [r.title for r in (response.recipes or [])][:5],
        "primary_urls": [r.url for r in (response.recipes or [])][:5],
        "routing_source": trace.get("routing_source"),
        "timings_ms": trace.get("timings_ms"),
        "cache_hit": trace.get("cache_hit"),
    }
    if dbg:
        row["analysis"] = dbg.get("analysis")
        an = dbg.get("analysis")
        if isinstance(an, dict):
            pl = an.get("plan")
            if isinstance(pl, dict):
                row["plan_task"] = pl.get("task")
                row["plan_course"] = pl.get("course")
                row["plan_cuisine_scope"] = pl.get("cuisine_scope")
        r = dbg.get("retrieval")
        if isinstance(r, dict):
            row["retrieval_query"] = r.get("retrieval_query")
            row["lexical_top_urls"] = [u for u, _ in (r.get("lexical_top") or [])[:5]]
            row["selected_urls"] = r.get("selected")
            row["rerank_shortcircuit"] = r.get("rerank_shortcircuit")
    return row


def load_golden(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return list(data.get("queries") or [])


def score_row(item: dict, result: dict) -> dict:
    """Compare résultat aux attentes du golden."""
    checks: dict = {}
    exp_url = item.get("expected_url_substring")
    if exp_url:
        urls = " ".join(str(u) for u in (result.get("primary_urls") or []) if u)
        checks["url_match"] = exp_url in urls
    exp_intent = item.get("expected_intent")
    if exp_intent:
        checks["intent_match"] = result.get("intent") == exp_intent
    exp_task = item.get("expected_task")
    if exp_task:
        pt = result.get("plan_task")
        if pt is not None:
            checks["task_match"] = pt == exp_task
    exp_course = item.get("expected_course")
    if exp_course:
        pc = result.get("plan_course")
        if pc is not None:
            checks["course_match"] = pc == exp_course
    exp_cs = item.get("expected_cuisine_scope")
    if exp_cs:
        pcs = result.get("plan_cuisine_scope")
        if pcs is not None:
            checks["cuisine_scope_match"] = pcs == exp_cs
    return checks


async def main_async(args: argparse.Namespace) -> int:
    from app.bot import SahtenBot, SahtenConfig

    yaml_path = Path(args.yaml).resolve()
    if not yaml_path.is_file():
        print(f"Fichier introuvable: {yaml_path}", file=sys.stderr)
        return 1

    items = load_golden(yaml_path)
    config = SahtenConfig(enable_safety_check=True, enable_narrative_generation=not args.skip_generation)
    bot = SahtenBot(config=config)

    out_path = Path(args.out).resolve() if args.out else None
    out_f = out_path.open("w", encoding="utf-8") if out_path else None

    summary = {"ok": 0, "fail": 0, "total": len(items), "at": datetime.now(timezone.utc).isoformat()}

    try:
        for item in items:
            qid = item.get("id", "?")
            text = item.get("text") or ""
            try:
                result = await run_one(bot, text, debug=True)
                checks = score_row(item, result)
                record = {
                    "id": qid,
                    "tags": item.get("tags"),
                    "expected": {
                        "url_substring": item.get("expected_url_substring"),
                        "intent": item.get("expected_intent"),
                        "task": item.get("expected_task"),
                        "course": item.get("expected_course"),
                        "cuisine_scope": item.get("expected_cuisine_scope"),
                    },
                    "checks": checks,
                    "result": result,
                }
                if checks:
                    if all(checks.values()):
                        summary["ok"] += 1
                    else:
                        summary["fail"] += 1
                line = json.dumps(record, ensure_ascii=False)
                print(line)
                if out_f:
                    out_f.write(line + "\n")
            except Exception as e:
                err = {"id": qid, "error": str(e), "user": text}
                summary["fail"] += 1
                print(json.dumps(err, ensure_ascii=False), file=sys.stderr)
                if out_f:
                    out_f.write(json.dumps(err, ensure_ascii=False) + "\n")
    finally:
        if out_f:
            out_f.close()

    print(json.dumps({"summary": summary}, ensure_ascii=False), file=sys.stderr)
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Run golden RAG batch diagnostic")
    p.add_argument(
        "--yaml",
        default=str(_BACKEND_ROOT.parent / "data" / "rag_golden_queries.yaml"),
        help="Chemin vers rag_golden_queries.yaml",
    )
    p.add_argument("--out", default="", help="Écrire JSONL (une ligne par requête)")
    p.add_argument(
        "--skip-generation",
        action="store_true",
        help="Désactive la génération narrative (plus rapide ; non implémenté ici — réservé)",
    )
    p.add_argument("--offline", action="store_true", help="Réservé (utiliser OPENAI_API_KEY vide via .env)")
    args = p.parse_args()
    if not args.out:
        args.out = None
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
