"""Enrichissement d'alias à l'INDEXATION (solution systémique).

Au lieu de deviner la bonne graphie à chaque requête (fragile, non déterministe),
on attache à chaque article — une fois, à l'ingestion — TOUTES les graphies
plausibles de son plat (translittérations + synonymes). Ce texte alimente le
`search_tsv` du chunk d'ancrage, donc n'importe quelle graphie utilisateur matche
l'article par recherche lexicale, de façon déterministe.

Source des groupes de synonymes : data/aliases_dishes.json + aliases_ingredients.json
(donnée auditable, maintenue manuellement + par l'agent auto-améliorant). Le lien
groupe→article se fait par **contenu** : un terme du groupe présent dans le titre.
"""

from __future__ import annotations

import json
import unicodedata
from functools import lru_cache
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parents[3] / "data"


def _norm(s: str) -> str:
    t = unicodedata.normalize("NFKD", (s or "").lower()).replace("’", "'")
    return "".join(c for c in t if not unicodedata.combining(c))


def transliterate_variants(term: str) -> set[str]:
    """Graphies dérivées PROPRES d'un terme curé : forme brute, sans accent, sans
    apostrophe, et singulier/pluriel. On ne génère PAS de substitutions hasardeuses
    (ou/u, ch/sh…) — les vraies graphies vivent dans la donnée (user_variants),
    ce qui évite le bruit et reste maintenable."""
    raw = (term or "").strip().lower().replace("’", "'")
    if not raw:
        return set()
    out: set[str] = {raw, _norm(raw), raw.replace("'", ""), _norm(raw).replace("'", "")}
    for w in list(out):  # singulier/pluriel
        if w.endswith("s") and len(w) > 3:
            out.add(w[:-1])
        else:
            out.add(w + "s")
    return {w.strip() for w in out if len(w.strip()) >= 3}


@lru_cache(maxsize=1)
def _synonym_groups() -> list[tuple[frozenset[str], frozenset[str]]]:
    """[(match_terms normalisés, toutes les graphies)] depuis les datasets."""
    sources = [
        ("aliases_dishes.json", "dishes",
         ("canonical_indexed", "indexed_forms", "user_variants")),
        ("aliases_ingredients.json", "ingredients",
         ("canonical", "variants", "inject")),
    ]
    groups: list[tuple[frozenset[str], frozenset[str]]] = []
    for fn, key, fields in sources:
        try:
            data = json.loads((_DATA_DIR / fn).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        for entry in data.get(key, []):
            terms: set[str] = set()
            for f in fields:
                v = entry.get(f)
                if isinstance(v, str):
                    terms.add(v)
                elif isinstance(v, list):
                    terms |= {x for x in v if isinstance(x, str)}
            if not terms:
                continue
            spellings: set[str] = set()
            for t in terms:
                spellings |= transliterate_variants(t)
            match = {_norm(t) for t in terms if len(_norm(t)) >= 4}
            if match and spellings:
                groups.append((frozenset(match), frozenset(spellings)))
    return groups


def article_alias_text(title: str, extra: str = "") -> str:
    """Texte d'alias searchable d'un article : union des graphies des groupes de
    synonymes dont un terme apparaît dans le titre (ou `extra`). '' si aucun."""
    hay = _norm(title) + " " + _norm(extra)
    spellings: set[str] = set()
    for match_terms, group_spellings in _synonym_groups():
        if any(m in hay for m in match_terms):
            spellings |= set(group_spellings)
    return " ".join(sorted(spellings))
