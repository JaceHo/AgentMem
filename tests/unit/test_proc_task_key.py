"""The procedure dedup key must not depend on how the task was typed.

Measured 2026-10-28 on the live service: the dedup hash was keyed on the raw task string, so a
re-worded task wrote a second row for the same workflow, and 12 of 312 rows had no key at all
(only reachable through the vector net). ``proc_task_key`` folds case / punctuation / whitespace —
those are not meaning — and every write path now maps the canonical key.
"""

import numpy as np
import pytest
import redis.asyncio as aioredis

from core import store as mem_store


@pytest.fixture
async def r():
    client = aioredis.Redis(host="localhost", port=6379, db=15, decode_responses=True)
    await client.flushdb()
    yield client
    await client.flushdb()
    await client.aclose()


def test_key_folds_only_what_is_not_meaning():
    k = mem_store.proc_task_key
    assert k("Task: Rotate  the_vault.") == k("task rotate the vault")
    assert k("  Publish   check: 何杰  ") == k("publish check 何杰")
    assert k("rotate the vault") != k("rotate the other vault")
    assert k("") == "" and k(None) == ""


def test_key_is_bounded_like_the_hash_field():
    assert len(mem_store.proc_task_key("x " * 500)) <= 300


async def test_save_then_lookup_survives_a_retyping(r):
    emb = np.zeros(mem_store._dims(), dtype=np.float32)
    await mem_store.save_procedure(r, task="Rotate the vault.", procedure="1. open 2. rotate",
                                   embedding=emb)
    assert await r.hget(mem_store.PROC_TASK_KEY, "rotate the vault")

    hit = await mem_store.find_procedure_by_task(r, emb, "rotate THE vault")
    assert hit and hit["_element"], "a re-typed task must dedupe through the canonical key"
    assert (hit["attrs"] or {}).get("task_key") == "rotate the vault"


async def test_delete_clears_the_canonical_and_the_legacy_key(r):
    emb = np.zeros(mem_store._dims(), dtype=np.float32)
    elem = await mem_store.save_procedure(r, task="Rotate the vault.", procedure="steps",
                                          embedding=emb)
    await r.hset(mem_store.PROC_TASK_KEY, "rotate the vault.", elem)   # pre-migration shape
    await mem_store._unlink_proc_side_indexes(r, elem, {"task": "rotate the vault."})
    assert not await r.hget(mem_store.PROC_TASK_KEY, "rotate the vault")
    assert not await r.hget(mem_store.PROC_TASK_KEY, "rotate the vault.")
