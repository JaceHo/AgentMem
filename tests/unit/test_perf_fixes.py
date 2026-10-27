"""Tests for the 2026-10-22 performance fixes.

Covers:
  * ``core.heat.atomic_bump_many`` — pipelined heat bump, with the NOSCRIPT
    fallback back to per-element calls.
  * ``main._query_specificity`` — the retrieval-free gate that decides whether
    HyDE/MIRIX are worth their latency.
  * ``main._recall_augmentations(skip_llm=True)`` — short-circuit path.
"""

import time

import numpy as np
import pytest
import redis.asyncio as aioredis

from core import heat as heat_mod
from core import store as mem_store


# ─── heat batching ───────────────────────────────────────────────────────────


@pytest.fixture
async def r():
    client = aioredis.Redis(host="localhost", port=6379, db=15, decode_responses=True)
    await client.flushdb()
    await mem_store.ensure_indexes(client)
    yield client
    await client.flushdb()
    await client.aclose()


class TestAtomicBumpMany:
    async def test_bumps_all_elements(self, r):
        z = np.zeros(384, dtype=np.float32)
        uids = [
            await mem_store.save_fact(r, content=f"heat fact {i}", category="rule",
                                      confidence=0.9, embedding=z)
            for i in range(5)
        ]
        bumped = await heat_mod.atomic_bump_many(r, mem_store.FACT_KEY, uids)
        assert bumped == 5
        for uid in uids:
            attrs = await mem_store.get_attrs(r, mem_store.FACT_KEY, uid)
            assert attrs["access_count"] == 1
            assert attrs["last_accessed"] > 0
            # Ebbinghaus anchor must move too, same as the single-element path
            assert attrs["last_confirmed_ts"] > 0

    async def test_empty_list_is_noop(self, r):
        assert await heat_mod.atomic_bump_many(r, mem_store.FACT_KEY, []) == 0

    async def test_repeated_bumps_accumulate(self, r):
        z = np.zeros(384, dtype=np.float32)
        uid = await mem_store.save_fact(r, content="accumulating fact", category="rule",
                                        confidence=0.9, embedding=z)
        for _ in range(3):
            await heat_mod.atomic_bump_many(r, mem_store.FACT_KEY, [uid])
        attrs = await mem_store.get_attrs(r, mem_store.FACT_KEY, uid)
        assert attrs["access_count"] == 3

    async def test_unknown_elements_do_not_raise(self, r, caplog):
        """A vanished element must be reported as NOT bumped, not counted as success.

        The old assertion (`bumped == 2`) encoded the exact bug that hid this for a month: the
        pipeline accepted the write and the Lua no-op'd on a non-existent element name, so callers
        believed heat was recorded while access_count stayed 0 forever. Honest counting is the
        regression guard — a silent no-op has to look like a failure in the return value and the log.
        """
        import logging
        with caplog.at_level(logging.WARNING, logger="mem"):
            bumped = await heat_mod.atomic_bump_many(
                r, mem_store.FACT_KEY, ["nope-1", "nope-2"]
            )
        assert bumped == 0                      # nothing was written -> nothing counted
        assert any("heat bump" in rec.message or "did nothing" in rec.message
                   for rec in caplog.records), "a no-op bump must be logged, not swallowed"

    async def test_sha_matches_redis_script_load(self, r):
        """_BUMP_HEAT_SHA must be the real script SHA or EVALSHA returns NOSCRIPT."""
        import hashlib

        expected = hashlib.sha1(heat_mod._BUMP_HEAT_LUA.encode("utf-8")).hexdigest()
        assert heat_mod._BUMP_HEAT_SHA == expected
        exists = await r.execute_command("SCRIPT", "EXISTS", heat_mod._BUMP_HEAT_SHA)
        # redis returns a list of 0/1 per SHA; [1] proves the constant is the
        # real SHA and EVALSHA will resolve rather than NOSCRIPT.
        assert list(exists) == [1]

    async def test_pipeline_is_one_round_trip(self, r, monkeypatch):
        """The whole point: N elements, 1 execute() call on the connection."""
        z = np.zeros(384, dtype=np.float32)
        uids = [
            await mem_store.save_fact(r, content=f"rt fact {i}", category="rule",
                                      confidence=0.9, embedding=z)
            for i in range(4)
        ]
        calls = {"n": 0}
        real_pipeline = r.pipeline

        def counting_pipeline(*a, **kw):
            calls["n"] += 1
            return real_pipeline(*a, **kw)

        monkeypatch.setattr(r, "pipeline", counting_pipeline)
        await heat_mod.atomic_bump_many(r, mem_store.FACT_KEY, uids)
        assert calls["n"] == 1


# ─── augmentation gate ───────────────────────────────────────────────────────


class TestQuerySpecificity:
    def test_specific_queries_score_high(self):
        import main
        assert main._query_specificity("redis vectorset VSIM connection pool") >= 3
        assert main._query_specificity("fix the cascade fallback in core/store.py") >= 3
        assert main._query_specificity("hyperqwen accelerate qwen inference") >= 3

    def test_vague_queries_score_low(self):
        import main
        assert main._query_specificity("what did we do") < 3
        assert main._query_specificity("hey") < 3
        assert main._query_specificity("can you help me with the thing") < 3

    def test_identifiers_and_paths_add_signal(self):
        import main
        # short but identifier-ish tokens still count
        assert main._query_specificity("aiserv_api deploy") >= 2
        assert main._query_specificity("deploy aiserv now") >= 1

    def test_empty_query(self):
        import main
        assert main._query_specificity("") == 0

    def test_gate_constant_is_sane(self):
        import main
        assert main._AUG_SPECIFICITY_FLOOR >= 1


class TestAugmentationSkip:
    async def test_skip_llm_makes_no_llm_calls(self, monkeypatch):
        """skip_llm=True must not touch retrieval_planner at all."""
        import main

        calls = {"hyde": 0, "topic": 0, "plan": 0}

        async def _hyde(q):
            calls["hyde"] += 1
            return "doc"

        async def _topic(q):
            calls["topic"] += 1
            return "topic"

        async def _plan(q):
            calls["plan"] += 1
            return ["q1"]

        monkeypatch.setattr(main.retrieval_planner, "generate_hyde_doc", _hyde)
        monkeypatch.setattr(main.retrieval_planner, "generate_retrieval_topic", _topic)
        monkeypatch.setattr(main.retrieval_planner, "plan_queries", _plan)

        class _Req:
            query = "redis vectorset VSIM connection pool"
            enable_hyde = True
            enable_planning = False

        out = await main._recall_augmentations(_Req(), skip_llm=True)
        assert out == {}
        assert calls == {"hyde": 0, "topic": 0, "plan": 0}

    async def test_no_skip_still_runs_augmentations(self, monkeypatch):
        import main

        async def _hyde(q):
            return "hypothetical document"

        async def _topic(q):
            return "retrieval topic"

        monkeypatch.setattr(main.retrieval_planner, "generate_hyde_doc", _hyde)
        monkeypatch.setattr(main.retrieval_planner, "generate_retrieval_topic", _topic)

        class _Req:
            query = "what did we decide about the deployment strategy last week"
            enable_hyde = True
            enable_planning = False

        out = await main._recall_augmentations(_Req(), skip_llm=False)
        assert out.get("hyde") == "hypothetical document"
        assert out.get("topic") == "retrieval topic"
