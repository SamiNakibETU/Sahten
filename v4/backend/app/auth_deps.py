"""Contrôle d'accès aux routes opérateur (analytics, admin, sessions, santé détaillaude)."""

from __future__ import annotations

import hmac
import secrets

import structlog
from fastapi import Header, HTTPException

from .settings import get_settings

log = structlog.get_logger(__name__)

ADMIN_HEADER = "X-Sahten-Admin-Token"


def _compare_token(provided: str, expected: str) -> bool:
    if not provided or not expected:
        return False
    if len(provided) != len(expected):
        return False
    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


async def require_admin_token(
    x_sahten_admin_token: str | None = Header(default=None, alias=ADMIN_HEADER),
    authorization: str | None = Header(default=None),
) -> None:
    """Exige le jeton `SAHTEN_ADMIN_API_TOKEN` sauf en `local` si le jeton n'est pas configuré."""
    settings = get_settings()
    expected = (settings.admin_api_token or "").strip()

    if not expected:
        if settings.app_env == "local":
            log.warning("auth.admin_skipped", reason="no SAHTEN_ADMIN_API_TOKEN in local")
            return
        log.error("auth.admin_misconfigured", env=settings.app_env)
        raise HTTPException(
            status_code=503,
            detail="Authentification administrateur non configurée sur le serveur.",
        )

    provided = (x_sahten_admin_token or "").strip()
    if authorization and authorization.lower().startswith("bearer "):
        provided = authorization[7:].strip()

    if not provided or not _compare_token(provided, expected):
        # Ne pas divulguer si le jeton manque ou est invalide
        raise HTTPException(status_code=401, detail="Non autorisé")


def generate_admin_token_suggestion() -> str:
    """Utile en CLI / doc : jeton aléatoire suffisamment long."""
    return secrets.token_urlsafe(24)
