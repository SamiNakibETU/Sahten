"""
LRU + TTL in-memory cache for chat responses (same query + model).
Skipped when session has conversation history (recipe exclusion / continuity).
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Tuple

from ..data.normalizers import normalize_text
from .config import get_settings

CacheEntry = Tuple[Any, Optional[dict], dict]  # SahtenResponse, debug_info, trace_meta


class ResponseCache:
    """Thread-safe LRU cache with per-entry TTL."""

    def __init__(self, max_size: int = 500, ttl_seconds: int = 3600) -> None:
        self._data: OrderedDict[str, tuple[CacheEntry, datetime]] = OrderedDict()
        self._lock = threading.Lock()
        self.max_size = max_size
        self.ttl = timedelta(seconds=ttl_seconds)

    def _make_key(self, message: str, model: str) -> str:
        norm = normalize_text((message or "").strip().lower())
        raw = f"{norm}|{model or ''}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, message: str, model: str) -> Optional[CacheEntry]:
        key = self._make_key(message, model)
        now = datetime.now(timezone.utc)
        with self._lock:
            if key not in self._data:
                return None
            entry, expires_at = self._data.pop(key)
            if expires_at < now:
                return None
            self._data[key] = (entry, expires_at)
            resp, dbg, meta = entry
            meta_copy = json.loads(json.dumps(meta))
            try:
                resp_copy = resp.model_copy(deep=True)
            except AttributeError:
                import copy

                resp_copy = copy.deepcopy(resp)
            return resp_copy, dbg, meta_copy

    def put(self, message: str, model: str, value: CacheEntry) -> None:
        key = self._make_key(message, model)
        expires_at = datetime.now(timezone.utc) + self.ttl
        with self._lock:
            if key in self._data:
                self._data.pop(key, None)
            self._data[key] = (value, expires_at)
            while len(self._data) > self.max_size:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


_response_cache: Optional[ResponseCache] = None


def get_response_cache() -> ResponseCache:
    global _response_cache
    if _response_cache is None:
        s = get_settings()
        _response_cache = ResponseCache(
            max_size=s.response_cache_max_size,
            ttl_seconds=s.response_cache_ttl_seconds,
        )
    return _response_cache
