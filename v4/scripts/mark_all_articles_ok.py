"""Aligne l’historique DB sur le mapper actuel : tous les articles en statut « ok ».

Usage (depuis le dossier ``v4`` du dépôt) :

  cd sahten_github/v4
  python scripts/mark_all_articles_ok.py

Variables :
  - ``DATABASE_URL`` (recommandé) : DSN async SQLAlchemy, ex.
    ``postgresql+asyncpg://USER:PASS@HOST:5432/DB``
  - Sinon : fichier ``.env`` à la racine de ``v4`` (chargé automatiquement).

Ne modifie pas les lignes en ``failed`` (échec d’ingestion avéré).

Railway (terminal) : ``railway run --service <nom> python scripts/mark_all_articles_ok.py``
depuis ``v4`` si le service a ``DATABASE_URL`` déjà injectée.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path


def _load_dotenv_v4() -> None:
    """Charge ``v4/.env`` sans dépendance (les variables déjà exportées gagnent)."""
    v4 = Path(__file__).resolve().parents[1]
    env_path = v4 / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv_v4()

# Permet ``python scripts/mark_all_articles_ok.py`` depuis ``v4/``
_backend_dir = Path(__file__).resolve().parents[1] / "backend"
if _backend_dir.is_dir():
    sys.path.insert(0, str(_backend_dir))

from sqlalchemy import text  # noqa: E402

from app.db.base import get_sessionmaker  # noqa: E402


async def main() -> None:
    sm = get_sessionmaker()
    async with sm() as session:
        r = await session.execute(
            text(
                "UPDATE articles SET ingestion_status = 'ok' "
                "WHERE ingestion_status IS DISTINCT FROM 'failed'"
            )
        )
        await session.commit()
        n = getattr(r, "rowcount", None)
        print(f"ingestion_status mis à « ok » (hors failed). Lignes : {n}")


if __name__ == "__main__":
    asyncio.run(main())
