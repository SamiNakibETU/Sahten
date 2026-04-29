"""Sahteïn v4 — point d'entrée FastAPI."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api import admin, chat, health, misc, sessions, web, webhook
from app.limiter_support import limiter
from app.security_middleware import SecurityHeadersMiddleware
from app.settings import Settings, get_settings


def configure_logging(level: str) -> None:
    logging.basicConfig(level=level)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
    )


def _warn_if_railway_without_strict_env(settings: Settings) -> None:
    """Évite un déploiement Railway accidentel en mode local (pas de jeton admin)."""
    import os

    if settings.app_env != "local":
        return
    if not (os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("RAILWAY_PUBLIC_DOMAIN")):
        return
    logging.getLogger("sahten.security").warning(
        "APP_ENV=local sur Railway : utiliser APP_ENV=staging|production avec "
        "SAHTEN_ADMIN_API_TOKEN et SAHTEN_CORS_ORIGINS pour durcir l'API."
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    s = get_settings()
    s.validate_security_at_startup()
    _warn_if_railway_without_strict_env(s)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    show_docs = settings.should_expose_openapi()
    app = FastAPI(
        title="Sahteïn v4",
        version="4.0.0a1",
        description="RAG SOTA — Postgres + pgvector + hybrid retrieval + grounding",
        lifespan=lifespan,
        docs_url="/docs" if show_docs else None,
        redoc_url="/redoc" if show_docs else None,
        openapi_url="/openapi.json" if show_docs else None,
    )
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    # Routers API d'abord (priorité de routing FastAPI : premier matché gagne).
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(webhook.router)
    app.include_router(sessions.router)
    app.include_router(admin.router)
    app.include_router(misc.router)
    # Pages HTML / statiques en dernier (catch-all sur /assets, /css, /js).
    app.include_router(web.router)
    return app


app = create_app()
