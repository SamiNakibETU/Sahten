"""Client WhiteBeard exhaustif.

Récupère un article par son external_id et retourne le **payload brut complet**
(aucune perte d'information). Le mapping vers le modèle ORM se fait dans
`mapper.py` à partir du payload brut.

Endpoints (cf. https://docs.whitebeard.net/api/) :
    GET /content/{id}
    GET /content?limit=&offset=&category=...

Auth : header `API-Key`.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..settings import get_settings


class WhiteBeardError(Exception):
    """Erreur WhiteBeard non récupérable (4xx hors 429)."""


class WhiteBeardClient:
    """Client async réutilisable. Pool de connexions partagé."""

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        s = get_settings()
        self.api_base = (api_base or s.olj_api_base).rstrip("/")
        self.api_key = api_key or s.olj_api_key
        if not self.api_key:
            raise WhiteBeardError("OLJ_API_KEY non configurée")
        self._client = httpx.AsyncClient(
            base_url=self.api_base,
            headers={"API-Key": self.api_key, "Accept": "application/json"},
            timeout=timeout,
        )

    async def __aenter__(self) -> "WhiteBeardClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------
    async def fetch_content(self, article_id: int | str) -> dict[str, Any]:
        """GET /content/{id} -> payload brut complet."""
        return await self._get(f"/content/{article_id}")

    async def list_content(
        self,
        *,
        category: str | None = None,
        limit: int = 50,
        offset: int = 0,
        keyword: str | None = None,
    ) -> dict[str, Any]:
        """GET /content?limit&offset&category&keyword."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if category:
            params["category"] = category
        if keyword:
            params["keyword"] = keyword
        return await self._get("/content", params=params)

    async def iter_all_articles(
        self,
        *,
        category: str | None = None,
        page_size: int = 50,
        max_pages: int | None = None,
        delay_s: float = 0.2,
    ):
        """Itère sur toutes les pages de /content (génère des dicts d'article)."""
        offset = 0
        page = 0
        while True:
            page += 1
            payload = await self.list_content(
                category=category, limit=page_size, offset=offset
            )
            data = payload.get("data") or []
            if not data:
                return
            for item in data:
                yield item
            if len(data) < page_size:
                return
            if max_pages is not None and page >= max_pages:
                return
            offset += page_size
            await asyncio.sleep(delay_s)

    # ------------------------------------------------------------------
    async def _get(self, path: str, *, params: dict | None = None) -> dict[str, Any]:
        last_error: Exception | None = None
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(4),
            wait=wait_exponential(multiplier=0.5, max=8.0),
            retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
            reraise=True,
        ):
            with attempt:
                r = await self._client.get(path, params=params)
                if r.status_code == 429:
                    raise httpx.HTTPStatusError(
                        "rate limited", request=r.request, response=r
                    )
                if r.status_code >= 400:
                    body = r.text[:500]
                    raise WhiteBeardError(
                        f"HTTP {r.status_code} on {path}: {body}"
                    )
                return r.json()
        if last_error:
            raise last_error
        raise WhiteBeardError("Échec inattendu sans exception capturée")
