"""Phase 0 — Sonde API WhiteBeard.

Appelle GET {OLJ_API_BASE}/content/{article_id} et sauvegarde la réponse brute
dans tests/fixtures/whitebeard_<id>.json. Affiche un audit champ-par-champ
(longueur, présence) pour valider qu'on a bien tout ce qu'il faut côté API
avant de coder l'ingestion exhaustive.

Usage:
    python v4/scripts/fetch_whitebeard.py 1227694 [1233126 ...]

Variables d'env requises (lues depuis sahten_github/backend/.env si présent):
    OLJ_API_KEY, OLJ_API_BASE
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

# Forcer UTF-8 sur stdout/stderr (Windows cp1252 sinon)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / "backend" / ".env"
FIXTURES_DIR = REPO_ROOT / "v4" / "tests" / "fixtures"


def _load_env_file(path: Path) -> dict[str, str]:
    """Mini parseur .env (KEY=VALUE, ignore commentaires et lignes vides)."""
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def _audit_field(payload: dict[str, Any], path: str) -> str:
    """Retourne un résumé court (présence, type, taille) pour un champ donné."""
    parts = path.split(".")
    cur: Any = payload
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return f"  - {path}: ABSENT"
    if cur is None:
        return f"  - {path}: null"
    if isinstance(cur, str):
        preview = cur[:80].replace("\n", " ")
        return f"  - {path}: str(len={len(cur)}) '{preview}…'"
    if isinstance(cur, list):
        return f"  - {path}: list(len={len(cur)})"
    if isinstance(cur, dict):
        return f"  - {path}: dict(keys={list(cur.keys())[:6]})"
    return f"  - {path}: {type(cur).__name__}={cur}"


def fetch(article_id: str, base: str, key: str) -> dict[str, Any]:
    url = f"{base.rstrip('/')}/content/{article_id}"
    headers = {"API-Key": key, "Accept": "application/json"}
    with httpx.Client(timeout=30.0) as c:
        r = c.get(url, headers=headers)
    print(f"\n=== {article_id} → HTTP {r.status_code} ({len(r.content)} bytes) ===")
    r.raise_for_status()
    return r.json()


def audit(payload: dict[str, Any], article_id: str) -> None:
    data = payload.get("data", [])
    if not data:
        print("  /!\\ data array vide")
        return
    item = data[0] if isinstance(data, list) else data
    print(f"  Top-level keys ({len(item)}):", sorted(item.keys()))
    print("\n  Champs critiques pour le RAG :")
    fields = [
        "id", "url", "slug", "title", "firstPublished", "lastUpdate",
        "time_to_read", "signature", "summary", "introduction", "contents",
        "content_length", "premium",
    ]
    for f in fields:
        print(_audit_field(item, f))

    authors = item.get("authors") or []
    print(f"\n  authors[]: len={len(authors)}")
    for i, a in enumerate(authors[:3]):
        if not isinstance(a, dict):
            continue
        print(f"    [{i}] keys={sorted(a.keys())}")
        for k in ("id", "name", "biography", "description", "image", "department"):
            if k in a:
                v = a[k]
                if isinstance(v, str):
                    print(f"      {k}: str(len={len(v)}) '{v[:60]}…'")
                else:
                    print(f"      {k}: {type(v).__name__}={v}")

    keywords = item.get("keywords") or []
    print(f"\n  keywords[]: len={len(keywords)}")
    for kw in keywords[:5]:
        if isinstance(kw, dict):
            name = kw.get("name", "?")
            desc = kw.get("description") or ""
            print(f"    - {name!r} desc(len={len(desc)})")

    cats = item.get("categories") or []
    print(f"\n  categories[]: len={len(cats)}")
    for c in cats[:3]:
        if isinstance(c, dict):
            print(f"    - {c.get('name')!r} desc(len={len(c.get('description') or '')})")

    atts = item.get("attachments") or []
    print(f"\n  attachments[]: len={len(atts)}")
    inline = item.get("inline_attachments") or []
    print(f"  inline_attachments[]: len={len(inline)}")

    contents = item.get("contents")
    if isinstance(contents, dict):
        html = contents.get("html") or ""
        print(f"\n  contents.html: len={len(html)}")
    elif isinstance(contents, str):
        print(f"\n  contents (string): len={len(contents)}")

    seo = item.get("seo") or {}
    if isinstance(seo, dict):
        print(f"\n  seo keys: {sorted(seo.keys())}")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2

    env = _load_env_file(ENV_PATH)
    api_key = os.environ.get("OLJ_API_KEY") or env.get("OLJ_API_KEY", "")
    api_base = (
        os.environ.get("OLJ_API_BASE")
        or env.get("OLJ_API_BASE")
        or "https://api.lorientlejour.com/cms"
    )

    if not api_key:
        print("ERREUR: OLJ_API_KEY non configurée (ni env, ni .env)", file=sys.stderr)
        return 1

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Base API : {api_base}")
    print(f"Clé      : len={len(api_key)} (masquée)")
    print(f"Fixtures : {FIXTURES_DIR}")

    rc = 0
    for aid in argv[1:]:
        try:
            payload = fetch(aid, api_base, api_key)
        except httpx.HTTPStatusError as e:
            print(f"  ECHEC HTTP : {e}", file=sys.stderr)
            rc = 1
            continue
        except Exception as e:
            print(f"  ECHEC : {e}", file=sys.stderr)
            rc = 1
            continue

        out = FIXTURES_DIR / f"whitebeard_{aid}.json"
        out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  -> fixture sauvegardee : {out.relative_to(REPO_ROOT)}")
        audit(payload, aid)

    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
