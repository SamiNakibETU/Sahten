#!/usr/bin/env python3
"""
Audit du fichier olj_canonical.json : couverture main_ingredients et search_text.

Usage (depuis la racine du repo):
  python backend/scripts/audit_canonical_data.py
  python backend/scripts/audit_canonical_data.py --json

Sortie : statistiques sur le nombre de fiches sans ingrédients ou avec search_text court.
Les enrichissements éditoriaux doivent être complétés en amont (pipeline de données), pas ici.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit canonical OLJ JSON")
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Chemin vers olj_canonical.json (defaut: data/olj_canonical.json depuis la racine repo)",
    )
    parser.add_argument("--json", action="store_true", help="Sortie JSON sur stdout")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    data_path = args.data or (repo_root / "data" / "olj_canonical.json")
    if not data_path.is_file():
        print(f"Fichier introuvable: {data_path}", file=sys.stderr)
        return 1

    raw = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        print("Format attendu: liste de documents", file=sys.stderr)
        return 1

    n = len(raw)
    no_ingredients = 0
    short_search = 0
    short_threshold = 80
    not_recipe = 0

    for doc in raw:
        if not doc.get("is_recipe", True):
            not_recipe += 1
            continue
        mains = doc.get("main_ingredients") or []
        if not mains:
            no_ingredients += 1
        st = (doc.get("search_text") or "").strip()
        if len(st) < short_threshold:
            short_search += 1

    stats = {
        "path": str(data_path),
        "total_documents": n,
        "recipe_documents": n - not_recipe,
        "non_recipe_skipped": not_recipe,
        "recipes_without_main_ingredients": no_ingredients,
        "recipes_with_search_text_under_chars": short_search,
        "search_text_threshold_chars": short_threshold,
    }

    if args.json:
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    else:
        print("Audit canonical OLJ")
        print(f"  Fichier: {data_path}")
        print(f"  Documents total: {n}")
        print(f"  Recettes (is_recipe): {stats['recipe_documents']}")
        print(f"  Sans main_ingredients: {no_ingredients} (priorité enrichissement)")
        print(f"  search_text < {short_threshold} caractères: {short_search}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
