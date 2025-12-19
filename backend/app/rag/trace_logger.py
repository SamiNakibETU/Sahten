"""
Trace logging utilities for Sahtein.

Writes chat interactions to JSONL so we can replay/debug conversations.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from app.models.config import settings

logger = logging.getLogger(__name__)


class TraceLogger:
    """Append-only JSONL logger for chat interactions."""

    def __init__(self, log_path: Path | None = None):
        self.log_path = log_path or settings.trace_log_path
        self._lock = Lock()

    def log_chat_interaction(
        self,
        request_id: str,
        user_message: str,
        response_html: str,
        scenario_id: int | None,
        scenario_name: str | None,
        primary_url: str | None,
        debug_info: dict | None,
        metadata: dict | None = None,
    ) -> None:
        """Persist a single chat trace as a JSON line."""

        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "user_message": user_message,
            "response_html": response_html,
            "scenario_id": scenario_id,
            "scenario_name": scenario_name,
            "primary_url": primary_url,
            "debug_info": debug_info or {},
            "metadata": metadata or {},
        }

        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            serialized = json.dumps(entry, ensure_ascii=False)
            with self._lock:
                with self.log_path.open("a", encoding="utf-8") as log_file:
                    log_file.write(serialized + "\n")
        except Exception as exc:  # pragma: no cover - logging failure shouldn't break requests
            logger.warning("Failed to persist chat trace: %s", exc)


_TRACE_LOGGER: TraceLogger | None = None


def get_trace_logger() -> TraceLogger:
    """Return a singleton TraceLogger instance."""
    global _TRACE_LOGGER
    if _TRACE_LOGGER is None:
        _TRACE_LOGGER = TraceLogger()
    return _TRACE_LOGGER


def set_trace_logger(logger_instance: TraceLogger | None) -> None:
    """Override the global trace logger (primarily for tests)."""
    global _TRACE_LOGGER
    _TRACE_LOGGER = logger_instance

