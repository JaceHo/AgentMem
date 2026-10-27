"""Regression tests for the deferred-write path of POST /store-procedure.

Why these exist (measured 2026-10-24): the handler persisted procedures inside a Starlette
background task whose first statement was ``encode(task)``. When the embedding circuit is open,
``encode`` raises ``EmbeddingUnavailable`` inside that task, the exception dies where no client can
see it, and the caller has already been answered ``200 {"status":"queued"}``. Net effect: three
procedures from ops/memory.sh and any nightly dream write simply vanished while the API claimed
success. These tests pin the two properties that make that impossible to reintroduce quietly:
  1. an outage is answered truthfully (status=deferred), not optimistically;
  2. the write survives the outage and replays when the embedder returns (no silent drop).
"""

import json
import time

import numpy as np
import pytest
import redis.asyncio as aioredis
from fastapi import BackgroundTasks

from api import state
from api.schemas.capability import StoreProcedureRequest
from core import embedder as embed_mod
from core import store as mem_store
from api.routes import capability as cap


@pytest.fixture
async def r():
    client = aioredis.Redis(host="localhost", port=6379, db=15, decode_responses=True)
    await client.flushdb()
    yield client
    await client.flushdb()
    await client.aclose()


def _req(tag: str) -> StoreProcedureRequest:
    return StoreProcedureRequest(task=f"defer test {tag}", procedure=f"steps for {tag}",
                                 tools_used=["web_search"], domain="testing")


async def test_outage_defers_instead_of_claiming_queued(r, monkeypatch):
    """With the breaker open the caller must be told "deferred", and the bytes must be in Redis."""
    monkeypatch.setattr(state, "redis", r)   # restored automatically: leaking a test-DB client into the global would poison later tests
    monkeypatch.setattr(embed_mod, "embedding_unavailable", lambda: True)
    before = cap.proc_guard_stats()["deferred"]

    resp = await cap.store_procedure(_req("A"), BackgroundTasks())

    assert resp["status"] == "deferred", f"outage must not be reported as queued: {resp}"
    assert resp["pending"] == 1
    assert cap.proc_guard_stats()["deferred"] == before + 1
    parked = json.loads(await r.lindex(cap.PROC_PENDING_KEY, 0))
    assert parked["task"].endswith("A") and parked["tools_used"] == ["web_search"], parked
    assert await r.llen(cap.PROC_PENDING_KEY) == 1


async def test_deferred_write_replays_when_embedder_recovers(r, monkeypatch):
    """The parked write must land once the backend answers — not be quietly discarded."""
    monkeypatch.setattr(state, "redis", r)   # restored automatically: leaking a test-DB client into the global would poison later tests
    saved = []

    async def fake_save(_r, **kw):
        saved.append(kw)
        return "elem-1"

    monkeypatch.setattr(mem_store, "save_procedure", fake_save)
    monkeypatch.setattr(mem_store, "find_procedure_by_task", lambda *_a, **_k: _none())
    monkeypatch.setattr(mem_store, "nearest_procedures", lambda *_a, **_k: _empty())
    monkeypatch.setattr(cap, "encode", lambda text: np.zeros(384, dtype=np.float32))
    monkeypatch.setattr(embed_mod, "embedding_unavailable", lambda: True)

    await cap.store_procedure(_req("B"), BackgroundTasks())
    assert await r.llen(cap.PROC_PENDING_KEY) == 1 and not saved

    monkeypatch.setattr(embed_mod, "embedding_unavailable", lambda: False)
    drained = await cap.drain_proc_pending()

    assert drained == 1, "the deferred procedure must replay exactly once"
    assert len(saved) == 1 and saved[0]["task"].endswith("B"), saved
    assert await r.llen(cap.PROC_PENDING_KEY) == 0
    assert cap.proc_guard_stats()["drained"] >= 1


async def test_breaker_reopening_mid_drain_keeps_the_order(r, monkeypatch):
    """If the backend fails again during replay the queue must stay intact, head preserved."""
    monkeypatch.setattr(state, "redis", r)   # restored automatically: leaking a test-DB client into the global would poison later tests
    await r.rpush(cap.PROC_PENDING_KEY, json.dumps({"task": "C", "procedure": "p",
                                                    "tools_used": [], "domain": "testing",
                                                    "session_id": ""}))
    monkeypatch.setattr(embed_mod, "embedding_unavailable", lambda: False)

    def boom(_text):
        raise embed_mod.EmbeddingUnavailable("embedding circuit open")

    monkeypatch.setattr(cap, "encode", boom)
    assert await cap.drain_proc_pending() == 0
    assert await r.llen(cap.PROC_PENDING_KEY) == 1, "a failed replay must not consume the record"


async def test_healthy_path_is_still_persisted_and_reported_as_queued(r, monkeypatch):
    """The fix must not change normal behaviour: queued + the background task stores it."""
    monkeypatch.setattr(state, "redis", r)   # restored automatically: leaking a test-DB client into the global would poison later tests
    saved = []

    async def fake_save(_r, **kw):
        saved.append(kw)
        return "elem-x"

    monkeypatch.setattr(embed_mod, "embedding_unavailable", lambda: False)
    monkeypatch.setattr(mem_store, "save_procedure", fake_save)
    monkeypatch.setattr(mem_store, "find_procedure_by_task", lambda *_a, **_k: _none())
    monkeypatch.setattr(mem_store, "nearest_procedures", lambda *_a, **_k: _empty())
    monkeypatch.setattr(cap, "encode", lambda text: np.zeros(384, dtype=np.float32))

    bg = BackgroundTasks()
    resp = await cap.store_procedure(_req(f"D{time.time_ns()}"), bg)
    assert resp["status"] == "queued" and resp["pending"] == 0
    assert not saved, "queued must return before the work happens — the store is the task"
    for task in bg.tasks:                       # FastAPI runs these after the response; do it here
        await task()
    assert len(saved) == 1 and saved[0]["task"].startswith("defer test D"), saved


async def _none():
    return None


async def _empty():
    return []


async def test_dedup_is_loud_not_silent(r, monkeypatch, caplog):
    """A write that adds no element must still leave a trace (measured 2026-10-24).

    The ops/memory.sh canary stored four near-identical sentences; the 4th was absorbed by the
    >0.90 dedup branch, which logged nothing and persisted nothing — from the outside that is
    byte-for-byte indistinguishable from the silent loss this file exists to prevent.
    """
    monkeypatch.setattr(state, "redis", r)
    monkeypatch.setattr(embed_mod, "embedding_unavailable", lambda: False)
    monkeypatch.setattr(cap, "encode", lambda text: np.zeros(384, dtype=np.float32))
    monkeypatch.setattr(mem_store, "find_procedure_by_task", lambda *_a, **_k: _none())

    async def known(*_a, **_k):
        return [{"score": 0.97, "_element": "elem-old", "attrs": {"task": "the known task"}}]

    monkeypatch.setattr(mem_store, "nearest_procedures", known)
    saved = []

    async def fake_save(_r, **kw):
        saved.append(kw)
        return "elem-new"

    monkeypatch.setattr(mem_store, "save_procedure", fake_save)
    before = cap.proc_guard_stats()["deduped"]

    with caplog.at_level("INFO", logger="mem"):
        bg = BackgroundTasks()
        resp = await cap.store_procedure(_req("dup"), bg)
        for task in bg.tasks:
            await task()

    assert resp["status"] == "queued" and not saved, "a dedup must not create a second element"
    assert cap.proc_guard_stats()["deduped"] == before + 1
    msg = [x.getMessage() for x in caplog.records if "deduped" in x.getMessage()]
    assert msg, "dedup was invisible: no log line, no counter, caller only saw 'queued'"
    assert "elem-old" in msg[0] and "0.970" in msg[0], msg[0]
