#!/usr/bin/env python3
"""
Mesure p50/p95 des timings (trace_meta) sur un jeu de requêtes — sans serveur HTTP.

Usage (depuis le dossier backend, venv activé) :
  python scripts/benchmark_latency.py

Nécessite OPENAI_API_KEY pour les appels LLM réels ; sans clé, le bot utilise les fallbacks (temps court).
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

# Ajouter le dossier backend au path
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

os.environ.setdefault("ENABLE_RESPONSE_CACHE", "false")

from app.bot import reload_bot  # noqa: E402


def _pctl(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


async def main() -> None:
    qa_path = BACKEND_ROOT / "tests" / "qa_matrix.json"
    extra = [
        "bonjour",
        "recette taboulé",
        "recette pizza",
        "météo demain",
    ]
    queries: list[str] = []
    if qa_path.exists():
        data = json.loads(qa_path.read_text(encoding="utf-8"))
        for c in data.get("cases", [])[:40]:
            q = c.get("query")
            if q:
                queries.append(q)
    queries.extend(extra)
    queries = queries[:50]

    bot = reload_bot()
    totals: list[float] = []
    analyses: list[float] = []
    retrievals: list[float] = []
    generations: list[float] = []

    for q in queries:
        t0 = time.perf_counter()
        _, _, meta = await bot.chat(q, debug=False, request_id="bench", session_id=None)
        totals.append((time.perf_counter() - t0) * 1000)
        tm = (meta or {}).get("timings_ms") or {}
        if "analysis" in tm:
            analyses.append(float(tm["analysis"]))
        if "retrieval" in tm:
            retrievals.append(float(tm["retrieval"]))
        if "generation" in tm:
            generations.append(float(tm["generation"]))

    print(f"Requêtes: {len(queries)}")
    print(f"total_ms  p50={_pctl(totals, 50):.1f}  p95={_pctl(totals, 95):.1f}")
    if analyses:
        print(f"analysis_ms  p50={_pctl(analyses, 50):.1f}  p95={_pctl(analyses, 95):.1f}")
    if retrievals:
        print(f"retrieval_ms p50={_pctl(retrievals, 50):.1f}  p95={_pctl(retrievals, 95):.1f}")
    if generations:
        print(f"generation_ms p50={_pctl(generations, 50):.1f} p95={_pctl(generations, 95):.1f}")
    if totals:
        print(f"mean_total_ms={statistics.mean(totals):.1f}")


if __name__ == "__main__":
    asyncio.run(main())
