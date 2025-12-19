"""
Session manager with TTL + LRU and conversation memory.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, UTC
from typing import Any, Optional
import re
import threading


@dataclass
class ConversationTurn:
    user_message: str
    bot_response_summary: str
    intent: str
    primary_dish: str | None
    ingredients: list[str]
    recipe_url: str | None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class SessionState:
    session_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_activity: datetime = field(default_factory=lambda: datetime.now(UTC))
    conversation_history: list[ConversationTurn] = field(default_factory=list)
    recipes_proposed: list[str] = field(default_factory=list)
    ingredients_mentioned: set[str] = field(default_factory=set)
    preferences: dict[str, Any] = field(default_factory=dict)

    def add_turn(
        self,
        user_message: str,
        intent: str,
        primary_dish: str | None = None,
        ingredients: list[str] | None = None,
        recipe_url: str | None = None,
        response_summary: str = "",
    ) -> None:
        turn = ConversationTurn(
            user_message=user_message,
            bot_response_summary=response_summary,
            intent=intent,
            primary_dish=primary_dish,
            ingredients=ingredients or [],
            recipe_url=recipe_url,
        )
        self.conversation_history.append(turn)
        if recipe_url and recipe_url not in self.recipes_proposed:
            self.recipes_proposed.append(recipe_url)
        if ingredients:
            self.ingredients_mentioned.update({ing.lower() for ing in ingredients if ing})
        self.last_activity = datetime.now(UTC)

    def has_history(self) -> bool:
        """Return True if at least one conversation turn exists."""
        return bool(self.conversation_history)

    def get_last_turn(self) -> ConversationTurn | None:
        return self.conversation_history[-1] if self.conversation_history else None

    def should_exclude(self, url: str | None) -> bool:
        return bool(url and url in self.recipes_proposed)

    def get_context_for_continuation(self) -> dict[str, Any]:
        last = self.get_last_turn()
        if not last:
            return {}
        return {
            "last_intent": last.intent,
            "last_dish": last.primary_dish,
            "last_ingredients": last.ingredients,
            "last_recipe_url": last.recipe_url,
            "all_ingredients": list(self.ingredients_mentioned),
            "preferences": self.preferences,
        }


class SessionManager:
    """LRU + TTL in-memory session manager."""

    def __init__(self, max_sessions: int = 2000, ttl_minutes: int = 30) -> None:
        self._sessions: OrderedDict[str, SessionState] = OrderedDict()
        self._lock = threading.Lock()
        self.max_sessions = max_sessions
        self.ttl = timedelta(minutes=ttl_minutes)

    def get_or_create(self, session_id: str) -> SessionState:
        with self._lock:
            self._cleanup_expired()
            if session_id in self._sessions:
                session = self._sessions.pop(session_id)
                session.last_activity = datetime.now(UTC)
                self._sessions[session_id] = session
                return session
            session = SessionState(session_id=session_id)
            self._sessions[session_id] = session
            return session

    def save(self, session: SessionState) -> None:
        with self._lock:
            session.last_activity = datetime.now(UTC)
            if session.session_id in self._sessions:
                self._sessions.pop(session.session_id)
            self._sessions[session.session_id] = session

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active_sessions": len(self._sessions),
                "max_sessions": self.max_sessions,
                "ttl_minutes": self.ttl.total_seconds() / 60,
            }

    def get_context_hints(self, session_id: str | None) -> dict[str, Any]:
        if not session_id:
            return {}
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return {}
            last_turn = session.get_last_turn()
            return {
                "recipes_proposed": list(session.recipes_proposed),
                "ingredients_mentioned": list(session.ingredients_mentioned),
                "last_intent": last_turn.intent if last_turn else None,
                "turn_count": len(session.conversation_history),
            }

    def _cleanup_expired(self) -> None:
        now = datetime.now(UTC)
        expired = [
            sid for sid, session in self._sessions.items() if now - session.last_activity > self.ttl
        ]
        for sid in expired:
            self._sessions.pop(sid, None)
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)


CONTINUATION_PATTERNS = [
    r"^oui\b",
    r"^non\b",
    r"^ok\b",
    r"^d'accord\b",
    r"^une autre",
    r"^autre chose",
    r"^plutot",
    r"^version",
    r"^avec\b",
    r"^sans\b",
    r"^et (?:si|avec|pour)",
    r"^la meme",
    r"^celle",
    r"^pareil",
    r"^encore",
    r"^aussi",
    r"^sinon",
    r"^peut-?etre",
    r"^je (?:veux|voudrais|prefere)",
    r"^(?:vegetarien|vegetalien|vegan|sans viande|sans gluten)",
]


def is_continuation(message: str) -> bool:
    """Detect whether a short reply continues the previous discussion."""
    if not message:
        return False
    message_lower = message.strip().lower()
    normalized = (
        message_lower.replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("à", "a")
        .replace("â", "a")
        .replace("î", "i")
        .replace("ï", "i")
        .replace("ô", "o")
        .replace("ù", "u")
        .replace("û", "u")
    )
    return any(re.match(pattern, normalized) for pattern in CONTINUATION_PATTERNS)


__all__ = ["SessionManager", "SessionState", "ConversationTurn", "is_continuation"]
