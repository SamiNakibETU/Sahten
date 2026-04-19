"""Évaluation RAGAS du pipeline complet sur le golden set.

Métriques produites (et stockées en DB dans `eval_runs`) :
  - faithfulness         : % phrases de la réponse soutenues par le contexte
  - answer_relevancy     : pertinence sémantique de la réponse / question
  - context_precision    : qualité du tri du contexte récupéré
  - context_recall       : couverture par rapport à la réponse attendue

Usage :
    python v4/scripts/eval_rag.py --golden v4/tests/golden/golden_set.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

from app.db.base import get_sessionmaker
from app.db.models import EvalRun
from app.rag.pipeline import RagPipeline


async def run_eval(golden_path: Path, write_db: bool = True) -> dict:
    try:
        from datasets import Dataset  # type: ignore
        from ragas import evaluate  # type: ignore
        from ragas.metrics import (  # type: ignore
            answer_relevancy, context_precision, context_recall, faithfulness,
        )
    except Exception as e:  # noqa: BLE001
        print(f"RAGAS non installé : {e}. `pip install -e .[eval]`", file=sys.stderr)
        sys.exit(2)

    pipeline = RagPipeline()
    sm = get_sessionmaker()

    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []

    with golden_path.open(encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    async with sm() as session:
        for item in items:
            r = await pipeline.answer(session, item["query"])
            answer = " ".join(s.text for s in r.answer.answer_sentences)
            ctx = [h.hit.chunk_text for h in r.reranked]
            questions.append(item["query"])
            answers.append(answer)
            contexts.append(ctx)
            ground_truths.append(" ".join(item.get("must_contain", [])))

    ds = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })
    scores = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    summary = {k: float(v) for k, v in dict(scores).items()}

    if write_db:
        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=Path.cwd()
            ).decode().strip()
        except Exception:  # noqa: BLE001
            sha = None
        async with sm() as session:
            session.add(EvalRun(
                git_sha=sha,
                config_json={"model": os.environ.get("LLM_MODEL", "")},
                faithfulness=summary.get("faithfulness"),
                answer_relevancy=summary.get("answer_relevancy"),
                context_precision=summary.get("context_precision"),
                context_recall=summary.get("context_recall"),
            ))
            await session.commit()
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--golden", default="v4/tests/golden/golden_set.jsonl")
    p.add_argument("--no-db", action="store_true")
    args = p.parse_args()
    summary = asyncio.run(
        run_eval(Path(args.golden), write_db=not args.no_db)
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
