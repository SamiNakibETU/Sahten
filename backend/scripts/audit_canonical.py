"""
Audit Canonical Dataset
=======================

Checks for:
- Empty critical fields
- Unknown categories/difficulties
- Duplicates
- Non-OLJ URLs

This is used for repeatable quality checks.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="v2/data/olj_canonical.json")
    args = parser.parse_args()

    # Project root = .../V3 (scripts live under v2/backend/scripts/)
    project_root = Path(__file__).resolve().parents[3]
    path = Path(args.input)
    inp = (path if path.is_absolute() else (project_root / path)).resolve()

    docs = json.loads(inp.read_text(encoding="utf-8"))
    if not isinstance(docs, list):
        raise ValueError("Expected list JSON")

    missing_title = 0
    missing_search_text = 0
    missing_category = 0
    missing_difficulty = 0
    non_olj = 0

    urls = []
    categories = Counter()
    difficulties = Counter()

    for d in docs:
        url = str(d.get("url") or "")
        urls.append(url)

        title = (d.get("title") or "").strip()
        if not title:
            missing_title += 1

        st = (d.get("search_text") or "").strip()
        if not st:
            missing_search_text += 1

        cat = (d.get("category_canonical") or "").strip()
        if not cat:
            missing_category += 1
        categories[cat or ""] += 1

        diff = (d.get("difficulty_canonical") or "").strip()
        if not diff:
            missing_difficulty += 1
        difficulties[diff or ""] += 1

        if url and "lorientlejour.com" not in url:
            non_olj += 1

    dup_urls = [u for u, c in Counter(urls).items() if u and c > 1]

    report = {
        "total_docs": len(docs),
        "missing_title": missing_title,
        "missing_search_text": missing_search_text,
        "missing_category": missing_category,
        "missing_difficulty": missing_difficulty,
        "duplicate_urls": len(dup_urls),
        "non_olj_urls": non_olj,
        "top_categories": categories.most_common(15),
        "top_difficulties": difficulties.most_common(15),
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    # Fail fast if critical issues remain
    if missing_title or missing_search_text or len(dup_urls):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
