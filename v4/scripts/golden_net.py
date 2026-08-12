"""Filet de régression déterministe sur le golden set (live, sans LLM-juge).

Contrairement à qa_grid.py (LLM-juge, coûteux, non déterministe), ce runner
n'applique QUE des vérifications mécaniques : article attendu dans les sources,
sous-chaînes obligatoires/interdites (insensibles casse+accents), présence et
nombre de cartes. Reproductible et gratuit → exécutable avant CHAQUE
changement de retrieval/classement.

Usage :
    python scripts/golden_net.py                          # localhost:80
    python scripts/golden_net.py --base http://localhost:8000
    python scripts/golden_net.py --out baseline.json
    python scripts/golden_net.py --compare baseline.json  # diff vs référence

Codes retour : 0 = pas de régression, 1 = régression vs --compare (ou échec
d'un cas hors known_gap sans --compare).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path

# Articles disparus côté OLJ (API 404, page publique 500) : les cas qui en
# dépendent sont mesurés mais ne comptent pas comme régression.
KNOWN_MISSING_ARTICLES = {1227694}  # taboulé de Kamal Mouzawak, constaté 2026-08-09


def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s.lower()).strip()


def chat(base: str, query: str, sid: str) -> dict:
    body = json.dumps({"query": query, "session_id": sid, "debug": True}).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/api/chat", body, {"Content-Type": "application/json"}
    )
    d = json.loads(urllib.request.urlopen(req, timeout=150).read())
    answer = " ".join(s.get("text", "") for s in (d.get("answer_sentences") or []))
    cards = [c for c in (d.get("recipe_card"), d.get("recipe_card_secondary")) if c]
    return {
        "answer": answer,
        "cards": [c.get("title") for c in cards],
        "n_cards": len(cards),
        "chef_card": bool(d.get("chef_card")),
        "sources": [s.get("article_external_id") for s in (d.get("sources") or [])],
        "strategy": d.get("answer_strategy"),
        "confidence": d.get("confidence"),
    }


def check(item: dict, r: dict) -> list[str]:
    """Retourne la liste des vérifications échouées (vide = cas OK)."""
    fails: list[str] = []
    na = norm(r["answer"])

    for want in item.get("answer_must_contain") or []:
        if norm(want) not in na:
            fails.append(f"manque:{want}")
    any_of = item.get("answer_any_contains") or []
    if any_of and not any(norm(w) in na for w in any_of):
        fails.append(f"aucun-de:{any_of}")
    for bad in item.get("answer_must_not_contain") or []:
        if norm(bad) in na:
            fails.append(f"interdit:{bad}")

    expected = [int(x) for x in item.get("expected_article_external_ids") or []]
    if expected and not (set(expected) & set(r["sources"])):
        fails.append(f"article-absent:{expected}")

    if item.get("require_recipe_card") and r["n_cards"] == 0:
        fails.append("carte-recette-absente")
    if item.get("require_chef_card") and not r["chef_card"]:
        fails.append("carte-chef-absente")
    mx = item.get("max_recipe_cards")
    if mx is not None and r["n_cards"] > mx:
        fails.append(f"trop-de-cartes:{r['n_cards']}>{mx}")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost")
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data" / "golden_eval_fr.json"))
    ap.add_argument("--out", default=None, help="écrire les résultats JSON ici")
    ap.add_argument("--compare", default=None, help="baseline JSON : échoue sur toute régression")
    args = ap.parse_args()

    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    results: dict[str, dict] = {}
    t_start = time.time()

    for it in data["items"]:
        cid = it["id"]
        gap = bool(set(int(x) for x in it.get("expected_article_external_ids") or [])
                   & KNOWN_MISSING_ARTICLES)
        try:
            # Cas MULTI-TOURS : plusieurs questions dans la MÊME session, vérifiées
            # sur le dernier tour. Indispensable — le bug de contamination par
            # l'historique (11/08 : « recette avec du poulet » -> « pas dans mes
            # carnets » dès le 2e tour) était invisible aux cas mono-tour, qui
            # ouvrent tous une session neuve.
            if it.get("turns"):
                sid = f"gmt-{cid}"[:40]
                for prev in it["turns"][:-1]:
                    chat(args.base, prev, sid)
                r = chat(args.base, it["turns"][-1], sid)
                fails = check(it, r)
                if fails:
                    sid2 = f"gmt2-{cid}"[:40]
                    for prev in it["turns"][:-1]:
                        chat(args.base, prev, sid2)
                    r2 = chat(args.base, it["turns"][-1], sid2)
                    fails2 = check(it, r2)
                    if not fails2:
                        r, fails = r2, []
            else:
                r = chat(args.base, it["query"], f"golden-{cid}"[:40])
                fails = check(it, r)
                if fails:
                    # Anti-flake : la génération LLM (temp 0.2) rend certains cas
                    # instables — un échec ne compte qu'une fois CONFIRMÉ par un
                    # second essai (session neuve pour éviter tout effet mémoire).
                    r2 = chat(args.base, it["query"], f"golden2-{cid}"[:40])
                    fails2 = check(it, r2)
                    if not fails2:
                        r, fails = r2, []
        except Exception as e:  # noqa: BLE001
            r, fails = {"strategy": None, "confidence": None}, [f"exception:{e}"[:120]]
        status = "PASS" if not fails else ("KNOWN_GAP" if gap else "FAIL")
        results[cid] = {
            "status": status,
            "category": it["category"],
            "fails": fails,
            "strategy": r.get("strategy"),
            "confidence": r.get("confidence"),
        }
        mark = {"PASS": "ok ", "FAIL": "ECH", "KNOWN_GAP": "gap"}[status]
        print(f"  [{mark}] {cid:<36} {';'.join(fails)[:70]}")

    n = len(results)
    n_pass = sum(1 for v in results.values() if v["status"] == "PASS")
    n_gap = sum(1 for v in results.values() if v["status"] == "KNOWN_GAP")
    n_fail = n - n_pass - n_gap
    print(f"\n{n_pass}/{n} PASS, {n_fail} FAIL, {n_gap} known_gap "
          f"({time.time()-t_start:.0f}s)")

    payload = {"when": time.strftime("%Y-%m-%dT%H:%M:%S"), "base": args.base,
               "summary": {"pass": n_pass, "fail": n_fail, "known_gap": n_gap},
               "results": results}
    if args.out:
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
        print(f"référence écrite : {args.out}")

    if args.compare:
        base = json.loads(Path(args.compare).read_text(encoding="utf-8"))["results"]
        regressions = [cid for cid, v in results.items()
                       if v["status"] == "FAIL" and base.get(cid, {}).get("status") == "PASS"]
        fixed = [cid for cid, v in results.items()
                 if v["status"] == "PASS" and base.get(cid, {}).get("status") in ("FAIL", "KNOWN_GAP")]
        if fixed:
            print("réparés :", ", ".join(fixed))
        if regressions:
            print("REGRESSIONS :", ", ".join(regressions))
            return 1
        print("aucune régression vs", args.compare)
        return 0

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
