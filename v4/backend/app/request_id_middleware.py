"""Middleware : corrélation des requêtes via X-Request-ID.

Chaque requête reçoit (ou génère) un identifiant unique propagé :
  - Dans le header de réponse : X-Request-ID
  - Dans le contexte structlog : toutes les logs de la requête le portent

Permet de retrouver toutes les logs d'un appel spécifique en production.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

log = structlog.get_logger(__name__)

_REQUEST_ID_HEADER = "X-Request-ID"


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Injecte/propage X-Request-ID et lie l'ID au contexte structlog."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Réutilise l'ID fourni par le client (ex. WhiteBeard) ou en génère un.
        raw = (request.headers.get(_REQUEST_ID_HEADER) or "").strip()
        # Valider : alphanumérique + tirets, max 64 chars — sinon générer.
        if raw and len(raw) <= 64 and raw.replace("-", "").replace("_", "").isalnum():
            request_id = raw
        else:
            request_id = "rid-" + uuid.uuid4().hex[:16]

        # Lier l'ID au contexte structlog pour cette coroutine.
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        response = await call_next(request)

        # Propager dans la réponse pour corrélation côté client/infra.
        response.headers[_REQUEST_ID_HEADER] = request_id
        return response
