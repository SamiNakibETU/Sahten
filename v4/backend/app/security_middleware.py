"""En-têtes HTTP de durcissement conforme aux standards OWASP Secure Headers.

Architecture :
- /api/*       → JSON, pas de cache, X-Frame-Options: DENY
- /dashboard   → pages internes, X-Frame-Options: SAMEORIGIN
- /widget/*    → iframe cross-origin (iframe embeddable sur lorientlejour.com),
                 pas de X-Frame-Options (allow-from n'est plus supporté)
- Toutes routes → X-Content-Type-Options, Referrer-Policy, Permissions-Policy
- staging/prod  → HSTS (maxAge 1 an, includeSubDomains)
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Détection env sans dépendance circulaire (settings non chargé ici).
_APP_ENV = os.getenv("APP_ENV", "local").lower()
_IS_PROD_LIKE = _APP_ENV in ("staging", "production")

# Pages internes (dashboard, admin) : on peut y appliquer X-Frame-Options: DENY.
_ADMIN_PATHS = ("/dashboard", "/admin", "/api/admin", "/api/traces",
                "/api/analytics", "/api/metrics", "/api/health/deep")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        path = request.url.path

        # ── En-têtes universels ────────────────────────────────────────────────
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault(
            "Permissions-Policy",
            "camera=(), microphone=(), geolocation=(), payment=()",
        )

        # ── HSTS : uniquement en staging/production (HTTP→HTTPS forcé) ────────
        if _IS_PROD_LIKE:
            response.headers.setdefault(
                "Strict-Transport-Security",
                "max-age=31536000; includeSubDomains",
            )

        # ── Endpoints API JSON ─────────────────────────────────────────────────
        if path.startswith("/api/"):
            response.headers.setdefault(
                "Cache-Control",
                "no-store, no-cache, must-revalidate, private",
            )
            response.headers.setdefault("X-Frame-Options", "DENY")
            response.headers.setdefault(
                "Content-Security-Policy",
                "default-src 'none'; frame-ancestors 'none'",
            )
            return response

        # ── Pages internes (dashboard, admin HTML) ────────────────────────────
        is_admin_page = any(path.startswith(p) for p in _ADMIN_PATHS)
        if is_admin_page:
            response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
            response.headers.setdefault(
                "Content-Security-Policy",
                (
                    "default-src 'self'; "
                    "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                    "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                    "img-src 'self' data: https:; "
                    "frame-ancestors 'self'"
                ),
            )
            return response

        # ── Widget (iframe cross-origin) : pas de X-Frame-Options ─────────────
        # lorientlejour.com intègre le widget en iframe ; X-Frame-Options DENY
        # ou SAMEORIGIN casserait l'embed. CSP frame-ancestors plus précis.
        if path.startswith("/widget"):
            response.headers.setdefault(
                "Content-Security-Policy",
                (
                    "default-src 'self'; "
                    # DOMPurify (défense XSS n°1) chargé depuis jsdelivr, secours unpkg.
                    # SANS ces origines, la CSP bloquait DOMPurify -> repli sur un
                    # sanitizer plus faible. On liste explicitement (pas de wildcard).
                    "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://unpkg.com; "
                    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://fonts.cdnfonts.com; "
                    "font-src 'self' https://fonts.gstatic.com https://fonts.cdnfonts.com data:; "
                    "img-src 'self' data: https:; "
                    # fetch /api/chat + SSE /api/chat/stream sur la même origine
                    "connect-src 'self'; "
                    "base-uri 'self'; "
                    "form-action 'self'; "
                    "object-src 'none'; "
                    "frame-ancestors https://www.lorientlejour.com https://*.lorientlejour.com 'self'"
                ),
            )

        return response
