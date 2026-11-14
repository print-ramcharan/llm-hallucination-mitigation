"""Tests for the bounded LRU/TTL cache used by embedding and retrieval paths."""

import time

import numpy as np

from src.cache import LruTtlCache
from src.indexing.embeddings import EmbeddingEngine


def test_cache_returns_recent_values_and_tracks_hits() -> None:
    cache: LruTtlCache[str, int] = LruTtlCache(max_size=2, ttl_seconds=5)
    cache.set("a", 1)

    assert cache.get("a") == 1
    assert cache.stats()["hits"] == 1


def test_cache_evicts_least_recently_used_entry() -> None:
    cache: LruTtlCache[str, int] = LruTtlCache(max_size=2, ttl_seconds=5)
    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.get("a") == 1
    cache.set("c", 3)

    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3


def test_cache_expires_entries_after_ttl() -> None:
    cache: LruTtlCache[str, int] = LruTtlCache(max_size=2, ttl_seconds=0.01)
    cache.set("a", 1)
    time.sleep(0.02)

    assert cache.get("a") is None
    assert cache.stats()["entries"] == 0


def test_cache_clear_removes_entries() -> None:
    cache: LruTtlCache[str, int] = LruTtlCache(max_size=2, ttl_seconds=5)
    cache.set("a", 1)
    cache.clear()

    assert cache.get("a") is None


def test_query_embeddings_are_cached_and_return_isolated_arrays() -> None:
    class Model:
        calls = 0

        def encode(self, texts, **_kwargs):
            self.calls += 1
            return np.asarray([[3.0, 4.0]], dtype=np.float32)

    engine = EmbeddingEngine("test-model")
    fake_model = Model()
    engine._model = fake_model
    first = engine.embed_query(" repeated   query ")
    first[0] = 0
    second = engine.embed_query("REPEATED query")

    assert fake_model.calls == 1
    assert np.allclose(second, np.asarray([0.6, 0.8], dtype=np.float32))
    assert engine.query_cache.stats()["hits"] == 1
