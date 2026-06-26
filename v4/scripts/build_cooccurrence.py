"""Construit un graphe de cooccurrence ingrédient-recette (offline, Epicure P2).

Sortie : data/ingredient_cooccurrence.json — pour chaque ingrédient, ses voisins
les plus liés (par NPMI), calculés sur les recettes du corpus. Étape de
DIAGNOSTIC : à évaluer avant toute intégration runtime (cf.
docs/metadata-graph-sota.md). Ne touche pas au pipeline.

Sources d'ingrédients par recette (par ordre de préférence) :
  1. --recipes <json> : liste d'objets {title, ingredient_slugs:[...]} (idéal,
     ex. export Postgres ArticleIngredient — slugs canoniques propres) ;
  2. sinon, format "enriché" {title, ingredients:[lignes brutes]} : extraction
     heuristique (quantités/unités retirées) — PROTOTYPE, qualité limitée.

Usage ::
    python scripts/build_cooccurrence.py --recipes data/recipes_ingredients.json
    python scripts/build_cooccurrence.py --recipes /tmp/enriched.json --min-recipes 4
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]

# Unités / quantités à retirer en tête de ligne d'ingrédient.
_UNITS = (
    "g", "kg", "mg", "ml", "cl", "dl", "l", "litre", "litres", "gr", "grammes",
    "cuillere", "cuilleres", "cuillère", "cuillères", "c", "cs", "cc", "càs", "càc",
    "pincee", "pincée", "gousse", "gousses", "botte", "bottes", "tranche", "tranches",
    "sachet", "sachets", "boite", "boîte", "verre", "verres", "tasse", "tasses",
    "poignee", "poignée", "filet", "brin", "brins", "feuille", "feuilles", "pot",
)
_STOP = {
    "de", "du", "des", "d", "la", "le", "les", "l", "a", "à", "au", "aux", "et",
    "ou", "en", "pour", "the", "of", "fine", "fines", "fin", "gros", "grosse",
    "grosses", "frais", "fraiche", "fraîche", "fraiches", "fraîches", "petit",
    "petite", "petits", "petites", "grand", "grande", "moulu", "moulue", "rape",
    "rapee", "râpé", "râpée", "haché", "hachee", "hachée", "clarifie", "clarifié",
    "doux", "sale", "salé", "sel", "poivre", "sucre", "sucré", "eau", "huile",
    "olive", "soupe", "cafe", "café", "pate", "pâte", "garniture", "sirop",
    "environ", "selon", "gout", "goût", "facultatif", "qs", "quelques", "morceaux",
}
_QTY_RE = re.compile(r"^[\d\s./,½¼¾()-]+")


def _norm(s: str) -> str:
    t = unicodedata.normalize("NFKD", (s or "").lower())
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return re.sub(r"[^a-z\s'-]+", " ", t)


def _singular(w: str) -> str:
    return w[:-1] if (len(w) > 3 and w.endswith("s")) else w


def _extract_slugs_from_lines(lines: list[str]) -> set[str]:
    """Heuristique : 1-2 mots significatifs par ligne d'ingrédient (prototype)."""
    slugs: set[str] = set()
    for raw in lines:
        if not isinstance(raw, str):
            continue
        line = raw.strip().rstrip(":")
        if not line or line.lower().startswith(("pour ", "pour la", "pour le")):
            continue
        s = _QTY_RE.sub("", _norm(line))
        words = [
            _singular(w)
            for w in s.split()
            if w not in _UNITS and w not in _STOP and len(w) >= 3
        ]
        if not words:
            continue
        # premier mot significatif = tête d'ingrédient (ex. "semoule fine" -> "semoule")
        slugs.add(words[0])
    return slugs


def _recipes_to_slugsets(recipes: list[dict[str, Any]]) -> list[set[str]]:
    out: list[set[str]] = []
    for r in recipes:
        if not isinstance(r, dict):
            continue
        if r.get("ingredient_slugs"):
            out.append({str(x).strip().lower() for x in r["ingredient_slugs"] if x})
        elif r.get("ingredients"):
            sl = _extract_slugs_from_lines(r["ingredients"])
            if sl:
                out.append(sl)
    return out


def build(slugsets: list[set[str]], *, min_recipes: int, top_k: int) -> dict[str, Any]:
    n = len(slugsets)
    df: Counter[str] = Counter()
    for s in slugsets:
        df.update(s)
    keep = {ing for ing, c in df.items() if c >= min_recipes}
    pair: Counter[tuple[str, str]] = Counter()
    for s in slugsets:
        items = sorted(s & keep)
        for a, b in combinations(items, 2):
            pair[(a, b)] += 1

    def npmi(a: str, b: str, co: int) -> float:
        # NPMI = ln(p(a,b)/(p(a)p(b))) / -ln(p(a,b))
        p_ab = co / n
        p_a = df[a] / n
        p_b = df[b] / n
        if p_ab <= 0:
            return 0.0
        return math.log(p_ab / (p_a * p_b)) / (-math.log(p_ab))

    neighbors: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (a, b), co in pair.items():
        if co < 2:
            continue
        score = round(npmi(a, b, co), 3)
        neighbors[a].append({"ingredient": b, "cooccur": co, "npmi": score})
        neighbors[b].append({"ingredient": a, "cooccur": co, "npmi": score})
    for ing in neighbors:
        neighbors[ing].sort(key=lambda d: (d["npmi"], d["cooccur"]), reverse=True)
        neighbors[ing] = neighbors[ing][:top_k]

    return {
        "description": "Graphe de cooccurrence ingrédient-recette (NPMI). PROTOTYPE "
                       "offline pour diagnostic (Epicure P2) — non branché au runtime.",
        "n_recipes": n,
        "n_ingredients_kept": len(keep),
        "min_recipes": min_recipes,
        "neighbors": dict(sorted(neighbors.items())),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Graphe cooccurrence ingrédients (offline).")
    p.add_argument("--recipes", required=True, help="JSON: liste {title, ingredient_slugs|ingredients}")
    p.add_argument("--out", default=str(_ROOT / "data" / "ingredient_cooccurrence.json"))
    p.add_argument("--min-recipes", type=int, default=4)
    p.add_argument("--top-k", type=int, default=8)
    args = p.parse_args()

    data = json.loads(Path(args.recipes).read_text(encoding="utf-8"))
    recipes = data if isinstance(data, list) else list(data.values())
    slugsets = _recipes_to_slugsets([r for r in recipes if isinstance(r, dict)])
    graph = build(slugsets, min_recipes=args.min_recipes, top_k=args.top_k)
    Path(args.out).write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"recipes={graph['n_recipes']} ingredients_kept={graph['n_ingredients_kept']} -> {args.out}")
    # aperçu
    for ing in list(graph["neighbors"])[:12]:
        nb = ", ".join(f"{d['ingredient']}({d['npmi']})" for d in graph["neighbors"][ing][:5])
        print(f"  {ing:16} -> {nb}")


if __name__ == "__main__":
    main()
