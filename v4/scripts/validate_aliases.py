"""Validation empirique des alias/translittérations contre l'API live Sahteïn.

Oracle auto-référent : pour chaque plat, la **forme de référence** (telle
qu'indexée) définit l'article cible (top source). On teste ensuite chaque graphie
utilisateur : « marche » = l'article cible revient dans les sources (rang ≤ 3).
Pour les graphies qui échouent, on teste l'**injection minimale** (variante +
forme indexée) jusqu'à ce que ça remonte.

Sortie : v4/data/aliases_dishes.json (donnée auditable, validation embarquée).

Usage (depuis v4/) ::
    python scripts/validate_aliases.py
    python scripts/validate_aliases.py --base-url https://...up.railway.app
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = "https://web-sahtein-19-04-staging.up.railway.app"

# Panel ancré sur le corpus OLJ réel (titres vérifiés dans data_base_OLJ_enriched).
# reference = graphie proche de l'index (définit la cible). inject = formes à
# ajouter pour rattraper les graphies qui échouent.
PANEL_DISHES: list[dict[str, Any]] = [
    {"canonical": "manouche", "reference": "manaiche",
     "variants": ["manouche", "manouché", "man'ouché", "manakish", "manaeesh", "mankoushe", "manouchi"],
     "inject": ["manaiche", "manaiches"]},
    {"canonical": "taboule", "reference": "taboulé",
     "variants": ["taboule", "tabboule", "tabbouleh", "tabouli", "taboulah", "tabbouli"],
     "inject": ["taboulé"]},
    {"canonical": "mouloukhieh", "reference": "mouloukhiyé",
     "variants": ["mouloukhia", "molokhia", "mloukhiyeh", "moloukhieh", "mloukhieh", "molokheya"],
     "inject": ["mouloukhiyé", "mouloukhiya"]},
    {"canonical": "kafta", "reference": "kafta bi batata",
     "variants": ["kafta", "kefta", "kofta", "köfte", "kafta batata"],
     "inject": ["kafta"]},
    {"canonical": "maamoul", "reference": "maamoul",
     "variants": ["maamoul", "ma'moul", "mamoul", "maamul", "maamoel"],
     "inject": ["maamoul", "maamouls"]},
    {"canonical": "kebbe", "reference": "kebbé",
     "variants": ["kebbe", "kibbeh", "kebbeh", "kibbe", "kubba", "kébbé"],
     "inject": ["kebbé", "kebbe"]},
    {"canonical": "fatteh", "reference": "fatteh",
     "variants": ["fatteh", "fatte", "fattah", "fattit"],
     "inject": ["fatteh", "fattit"]},
    {"canonical": "su-beureg", "reference": "su beureg",
     "variants": ["su beureg", "su böreg", "soubereg", "su borek", "beoreg"],
     "inject": ["su beureg"]},
    {"canonical": "ourfa-kebab", "reference": "ourfa kebab",
     "variants": ["ourfa kebab", "urfa kebab", "ourfa", "urfa kebabi"],
     "inject": ["ourfa kebab"]},
    {"canonical": "samboussek", "reference": "samboussek",
     "variants": ["samboussek", "sambousek", "sambusak", "samboosa", "sambousa"],
     "inject": ["samboussek"]},
    {"canonical": "ouayamat", "reference": "ouayamat",
     "variants": ["ouayamat", "awamat", "awameh", "awwama", "loukoumades"],
     "inject": ["ouayamat"]},
    {"canonical": "fattouche", "reference": "fattouche",
     "variants": ["fattouche", "fattoush", "fattouch", "fattush"],
     "inject": ["fattouche"]},
    {"canonical": "houmous", "reference": "houmous",
     "variants": ["hommos", "hummus", "hoummous", "homos"],
     "inject": ["houmous", "hommos"]},
    {"canonical": "kafta-batata", "reference": "kafta bi batata",
     "variants": ["kafta bi batata", "kafta batata", "kefta batata"],
     "inject": ["kafta bi batata"]},
]


def _post(base: str, q: str, sid: str, timeout: int = 30) -> dict[str, Any] | None:
    body = json.dumps({"query": q, "session_id": sid}).encode()
    req = urllib.request.Request(base + "/api/chat", body, {"Content-Type": "application/json"})
    for attempt in (1, 2):
        try:
            return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        except Exception:
            if attempt == 2:
                return None
            time.sleep(1.0)
    return None


def _sources(d: dict[str, Any] | None) -> list[int]:
    if not d:
        return []
    return [int(s["article_external_id"]) for s in (d.get("sources") or [])
            if s.get("article_external_id") is not None]


def _rank(target: int, ids: list[int]) -> int | None:
    seen: list[int] = []
    for i in ids:
        if i not in seen:
            seen.append(i)
        if i == target:
            return seen.index(i) + 1
    return None


def run(base: str, out_path: Path) -> None:
    results: list[dict[str, Any]] = []
    n = 0
    for dish in PANEL_DISHES:
        can = dish["canonical"]
        ref = _post(base, dish["reference"], f"alias-ref-{can}")
        ref_ids = _sources(ref)
        target = ref_ids[0] if ref_ids else None
        ref_card = (ref.get("recipe_card") or {}).get("title") if ref and ref.get("recipe_card") else None
        entry: dict[str, Any] = {
            "canonical": can,
            "reference": dish["reference"],
            "target_article_id": target,
            "target_title": ref_card,
            "user_variants": dish["variants"],
            "inject": [],
            "validation": [],
            "still_failing": [],
        }
        if target is None:
            entry["note"] = "reference n'a rien retourné — cible inconnue, à investiguer"
            results.append(entry)
            _save(out_path, results)
            print(f"[warn] {can}: reference sans cible", flush=True)
            continue

        needed_inject = False
        for v in dish["variants"]:
            r = _rank(target, _sources(_post(base, v, f"alias-{can}-{abs(hash(v))%9999}")))
            ok = r is not None and r <= 3
            row = {"query": v, "found": ok, "rank": r}
            if not ok:
                # tenter l'injection
                vi = v + " " + " ".join(dish["inject"])
                r2 = _rank(target, _sources(_post(base, vi, f"alias-{can}-inj-{abs(hash(v))%9999}")))
                ok2 = r2 is not None and r2 <= 3
                row["with_inject"] = {"query": vi, "found": ok2, "rank": r2}
                if ok2:
                    needed_inject = True
                else:
                    entry["still_failing"].append(v)
            entry["validation"].append(row)
            n += 1
            time.sleep(0.3)
        entry["inject"] = dish["inject"] if needed_inject else []
        # groupe final pour _ALIAS_GROUPS : variantes + injection (même concept)
        entry["alias_group"] = sorted(set(dish["variants"] + dish["inject"] + [dish["reference"]]))
        results.append(entry)
        _save(out_path, results)
        cov = sum(1 for x in entry["validation"]
                  if x["found"] or (x.get("with_inject") or {}).get("found"))
        print(f"[done] {can}: target={target} couverture={cov}/{len(dish['variants'])} "
              f"inject={'oui' if needed_inject else 'non'} fails={entry['still_failing']}", flush=True)
    print(f"\nTOTAL probes≈{n}. Écrit: {out_path}", flush=True)


def _save(path: Path, results: list[dict[str, Any]]) -> None:
    payload = {
        "description": "Alias/translittérations de plats validés empiriquement contre l'API live. "
                       "'inject' = formes à ajouter à la requête pour rattraper les graphies en échec.",
        "generated_against": DEFAULT_BASE,
        "dishes": results,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=DEFAULT_BASE)
    p.add_argument("--out", default=str(_ROOT / "data" / "aliases_dishes.json"))
    args = p.parse_args()
    run(args.base_url.rstrip("/"), Path(args.out))


if __name__ == "__main__":
    main()
