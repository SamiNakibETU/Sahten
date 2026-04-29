"""En-têtes HTTP de durcissement (sans CSP strict qui casserait les pages inline)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        # Pas de X-Frame-Options global : l'intégration site peut utiliser iframe cross-origin.
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault(
            "Permissions-Policy",
            "camera=(), microphone=(), geolocation=(), payment=()",
        )
        # API-only hardening : les navigateurs n'exécutent pas de JS sur le JSON.
        if request.url.path.startswith("/api/"):
            response.headers.setdefault(
                "Cache-Control",
                "no-store, no-cache, must-revalidate, private",
            )
        return response
