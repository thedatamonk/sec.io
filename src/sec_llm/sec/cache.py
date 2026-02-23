"""Simple in-memory TTL cache."""

from __future__ import annotations

import time
from typing import Any


class TTLCache:
    """Thread-safe-ish TTL cache backed by a plain dict."""

    def __init__(self, ttl_seconds: int = 900):
        self._ttl = ttl_seconds
        self._store: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        ts, value = entry
        if time.monotonic() - ts > self._ttl:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (time.monotonic(), value)

    def clear(self) -> None:
        self._store.clear()

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None
