"""Tests for the dedicated embedding pool + recall concurrency semaphore (Task #20).

Recall previously funnelled every embedding through the *default* asyncio
thread pool (also used by BM25 rebuilds) with no real ceiling.  These tests
pin the new behaviour:

  * ``core.search.async_encode`` / ``async_encode_batch`` run on a dedicated,
    bounded pool and are gated by a semaphore so concurrency is capped.
  * The circuit-open (EmbeddingUnavailable) error still propagates.
  * Batching collapses N texts into one executor call.

The real embedder is unavailable in CI (no model, no network), so we monkeypatch
``embedder.encode`` / ``encode_batch`` with a fake that records how many run
concurrently.  In the concurrency test we install a *large* pool and a *tiny*
semaphore so the SEMAPHORE — not the pool size — is the binding constraint,
isolating the ceiling guarantee.
"""

import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import core.search as search_mod
from core import embedder


@pytest.fixture
def patched(monkeypatch):
    lock = threading.Lock()
    state = {"cur": 0, "max": 0, "batch_calls": 0}

    def fake_encode(text):
        if embedder.embedding_unavailable():
            raise embedder.EmbeddingUnavailable("circuit open")
        with lock:
            state["cur"] += 1
            state["max"] = max(state["max"], state["cur"])
        time.sleep(0.02)
        with lock:
            state["cur"] -= 1
        return np.zeros(embedder.DIMS, dtype=np.float32)

    def fake_encode_batch(texts):
        state["batch_calls"] += 1
        return [fake_encode(t) for t in texts]

    monkeypatch.setattr(embedder, "encode", fake_encode)
    monkeypatch.setattr(embedder, "encode_batch", fake_encode_batch)
    monkeypatch.setattr(embedder, "embedding_unavailable", lambda: False)
    return state


def _install_large_pool_tiny_sem():
    """Swap in a large pool + 3-slot semaphore so the semaphore binds.

    Returns the originals for restoration.  Must be called inside a running loop.
    """
    orig_pool, orig_sem = search_mod._embed_pool, search_mod._embed_sem
    search_mod._embed_pool = ThreadPoolExecutor(
        max_workers=16, thread_name_prefix="test-embed"
    )
    search_mod._embed_sem = asyncio.Semaphore(3)
    return orig_pool, orig_sem


async def test_async_encode_returns_vector(patched):
    v = await search_mod.async_encode("hello world")
    assert isinstance(v, np.ndarray)
    assert v.shape == (embedder.DIMS,)


async def test_async_encode_batch_returns_list_and_batches(patched):
    vs = await search_mod.async_encode_batch(["a", "b", "c", "d"])
    assert len(vs) == 4
    # One executor call for the whole batch (not 4).
    assert patched["batch_calls"] == 1
    assert all(isinstance(v, np.ndarray) and v.shape == (embedder.DIMS,) for v in vs)


async def test_async_encode_batch_empty_is_noop(patched):
    assert await search_mod.async_encode_batch([]) == []
    assert patched["batch_calls"] == 0


async def test_semaphore_bounds_concurrency(patched):
    orig_pool, orig_sem = _install_large_pool_tiny_sem()
    try:
        tasks = [
            asyncio.create_task(search_mod.async_encode(f"q{i}"))
            for i in range(30)
        ]
        await asyncio.gather(*tasks)
        # The semaphore caps in-flight embeddings at 3 even though the pool has
        # 16 workers — the whole point of the recall concurrency ceiling.
        assert patched["max"] <= 3
        assert patched["max"] >= 1
    finally:
        search_mod._embed_pool.shutdown(wait=False)
        search_mod._embed_pool, search_mod._embed_sem = orig_pool, orig_sem


async def test_circuit_open_propagates(patched, monkeypatch):
    monkeypatch.setattr(embedder, "embedding_unavailable", lambda: True)
    with pytest.raises(embedder.EmbeddingUnavailable):
        await search_mod.async_encode("anything")
    with pytest.raises(embedder.EmbeddingUnavailable):
        await search_mod.async_encode_batch(["x", "y"])
