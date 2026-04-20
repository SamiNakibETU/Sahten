"""Sahteïn v4 — point d'entrée FastAPI."""

from __future__ import annotations

import logging

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import admin, chat, health, misc, sessions, web, webhook
from app.settings import get_settings


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


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(
        title="Sahteïn v4",
        version="4.0.0a1",
        description="RAG SOTA — Postgres + pgvector + hybrid retrieval + grounding",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
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
