"""Éval live (smoke) de l'assistant déployé : 20 questions couvrant toutes les
catégories attendues. Capture la réponse complète pour jugement qualitatif.

Usage : python scripts/live_smoke_eval.py [--base-url ...] [--out ...]
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request

DEFAULT_BASE = "https://web-sahtein-19-04-staging.up.railway.app"

# (catégorie, question, attendu)
QUESTIONS: list[tuple[str, str, str]] = [
    ("plat-precis", "la recette du taboulé de Kamal Mouzawak", "trouve le taboulé + carte"),
    ("plat-precis", "les manaïichs du Chouf de Salim Azzam", "trouve 1474718 + carte"),
    ("ortho", "recette manouche", "doit trouver les manaïichs (1474718)"),
    ("ortho", "manakish", "doit trouver les manaïichs (1474718)"),
    ("ortho", "recette de tabbouleh", "doit trouver le taboulé"),
    ("ortho", "hommos", "doit trouver le houmous"),
    ("indispo", "donne-moi une recette de sushi", "abstention PRO en FR, pas de carte hors-sujet"),
    ("indispo", "comment faire une pizza margherita ?", "abstention PRO en FR"),
    ("insulte", "t'es nul et inutile", "réponse calme et pro, recentre sur la cuisine"),
    ("hors-sujet", "quelle est la capitale du Japon ?", "recentre poliment sur la cuisine libanaise"),
    ("hors-sujet", "raconte-moi une blague", "recentre poliment, ton chaleureux"),
    ("ingredient", "une recette avec de l'aubergine", "plat à l'aubergine + carte"),
    ("ingredient", "un plat au poulet pour recevoir", "plat au poulet + carte"),
    ("saison", "une recette légère pour l'été", "suggestion estivale cohérente"),
    ("saison", "un plat réconfortant d'hiver au potiron", "plat d'hiver/potiron"),
    ("facilite", "une recette facile et rapide pour ce soir", "recette simple, pas de dessert imposé"),
    ("facilite", "quelque chose de très simple à préparer", "recette simple"),
    ("entree", "une entrée libanaise pour commencer un repas", "entrée/mezzé"),
    ("plat", "un plat principal libanais consistant", "plat principal"),
    ("dessert", "un dessert libanais traditionnel", "dessert (maamoul, sfouf, ouayamat...)"),
]


def chat(base: str, q: str, sid: str) -> dict:
    body = json.dumps({"query": q, "session_id": sid}).encode()
    req = urllib.request.Request(base + "/api/chat", body, {"Content-Type": "application/json"})
    try:
        d = json.loads(urllib.request.urlopen(req, timeout=45).read())
        return {
            "answer": " ".join(s.get("text", "") for s in (d.get("answer_sentences") or [])),
            "card": (d.get("recipe_card") or {}).get("title") if d.get("recipe_card") else None,
            "follow_up": d.get("follow_up"),
            "confidence": d.get("confidence"),
            "sources": [
                (s.get("article_external_id"), (s.get("article_title") or "")[:40])
                for s in (d.get("sources") or [])[:3]
            ],
        }
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:120]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=DEFAULT_BASE)
    p.add_argument("--out", default="")
    p.add_argument("--delay", type=float, default=4.0)
    args = p.parse_args()
    base = args.base_url.rstrip("/")
    out = args.out or "live_smoke_results.json"
    rows = []
    for i, (cat, q, exp) in enumerate(QUESTIONS, 1):
        r = chat(base, q, f"smoke-{i}")
        r.update({"n": i, "category": cat, "query": q, "expected": exp})
        rows.append(r)
        json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        if "error" in r:
            print(f"[{i:02d}|{cat}] ERROR {r['error']}", flush=True)
        else:
            print(f"[{i:02d}|{cat}] q={q!r}", flush=True)
            print(f"      card={r['card']} conf={r['confidence']} src={r['sources']}", flush=True)
            print(f"      ANSWER: {r['answer'][:300]}", flush=True)
            print(f"      FOLLOW: {r['follow_up']}", flush=True)
        time.sleep(args.delay)
    print(f"\nDONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
