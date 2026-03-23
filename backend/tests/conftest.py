"""Pytest configuration: disable response cache by default for deterministic tests."""

from __future__ import annotations

import os

os.environ["ENABLE_RESPONSE_CACHE"] = "false"

import pytest

from app.core.config import get_settings


@pytest.fixture(scope="session", autouse=True)
def _reset_settings_cache_session() -> None:
    get_settings.cache_clear()
    yield
