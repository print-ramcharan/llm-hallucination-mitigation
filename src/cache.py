"""Thread-safe LRU cache with per-entry time-to-live expiration."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


@dataclass
class _Entry(Generic[V]):
    value: V
    expires_at: float


class LruTtlCache(Generic[K, V]):
    """Bounded LRU cache whose entries expire after a configurable TTL."""

    def __init__(self, max_size: int = 1024, ttl_seconds: float = 300.0) -> None:
        if max_size < 1 or ttl_seconds <= 0:
            raise ValueError("max_size and ttl_seconds must be positive")
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._entries: OrderedDict[K, _Entry[V]] = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def get(self, key: K) -> V | None:
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            if entry.expires_at <= now:
                del self._entries[key]
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return entry.value

    def set(self, key: K, value: V) -> None:
        with self._lock:
            self._entries[key] = _Entry(value, time.monotonic() + self.ttl_seconds)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_size:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def stats(self) -> dict[str, int | float]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "ttl_seconds": self.ttl_seconds,
            }
