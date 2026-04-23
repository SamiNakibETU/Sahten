"""Debug helper : interroge plusieurs pages de /publication/17/content pour
comprendre comment paginer les recettes."""

from __future__ import annotations

import asyncio
import json

from app.ingestion.whitebeard_client import WhiteBeardClient


async def main() -> None:
    async with WhiteBeardClient() as cli:
        for ct in (4, None):
            print(f"\n=== content_type={ct} ===")
            for p in (1, 2, 3, 4, 5):
                payload = await cli.list_publication_content(
                    publication_id=17, content_type=ct, limit=50, page=p
                )
                data = payload.get("data") or []
                first_id = data[0].get("id") if data else None
                last_id = data[-1].get("id") if data else None
                print(
                    f"  page={p:>2}  total={payload.get('total')}  "
                    f"page_returned={payload.get('page')}  "
                    f"len={len(data)}  first={first_id}  last={last_id}"
                )


if __name__ == "__main__":
    asyncio.run(main())
