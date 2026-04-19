"""Diagnostic auth WhiteBeard : essaie plusieurs schémas d'authentification."""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import httpx

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / "backend" / ".env"


def load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def main() -> int:
    env = load_env(ENV_PATH)
    key = os.environ.get("OLJ_API_KEY") or env.get("OLJ_API_KEY", "")
    base = (
        os.environ.get("OLJ_API_BASE")
        or env.get("OLJ_API_BASE")
        or "https://api.lorientlejour.com/cms"
    )
    if not key:
        print("OLJ_API_KEY absente", file=sys.stderr)
        return 1

    aid = "1227694"
    url = f"{base.rstrip('/')}/content/{aid}"
    print(f"URL: {url}")
    print(f"Key: len={len(key)}, first2='{key[:2]}', last2='{key[-2:]}'")

    attempts = [
        ("API-Key header", {"API-Key": key}),
        ("Token header", {"Token": key}),
        ("Authorization Bearer", {"Authorization": f"Bearer {key}"}),
        ("Authorization API-Key", {"Authorization": f"API-Key {key}"}),
        ("X-API-Key header", {"X-API-Key": key}),
        ("api_key query", {}),
    ]
    for label, headers in attempts:
        params = {"api_key": key} if "query" in label else None
        try:
            with httpx.Client(timeout=15.0) as c:
                r = c.get(url, headers=headers, params=params)
            body = r.text[:200].replace("\n", " ")
            print(f"  [{r.status_code}] {label} -> {body!r}")
        except Exception as e:
            print(f"  [ERR] {label} -> {e!r}")

    # Probe basique de l'API root pour vérifier que la base est joignable
    print("\nProbe racine:")
    for path in ("", "/health", "/status", "/openapi.json"):
        try:
            with httpx.Client(timeout=10.0) as c:
                r = c.get(base.rstrip("/") + path)
            print(f"  GET {base.rstrip('/') + path} -> {r.status_code}")
        except Exception as e:
            print(f"  GET {base.rstrip('/') + path} -> ERR {e!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
