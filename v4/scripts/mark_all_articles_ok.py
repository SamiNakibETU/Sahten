"""Aligne l’historique DB sur le mapper actuel : tous les articles en statut « ok ».

Usage (répertoire v4 du projet, ``DATABASE_URL`` dans l’environnement) :

  cd sahten_github/v4
  python scripts/mark_all_articles_ok.py

Ne modifie pas les lignes en ``failed`` (échec d’ingestion avéré).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Permet ``python -m scripts.mark_all_articles_ok`` depuis le répertoire backend/
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
