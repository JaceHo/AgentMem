"""Tests for the element-ID shadow index, multi-tier BM25, and tier routing.

These cover the 2026-10-22 correctness fixes:

  * ``core.store`` shadow index (``mem:idx:*``) — reliable enumeration where
    VSIM is not, because Redis vectorset's HNSW traversal does not guarantee
    COUNT results.
  * ``core.search`` multi-tier BM25 — episodes and procedures used to be
    keyword-invisible.
  * ``core.search.extract_text`` / ``tier_of`` — per-tier text assembly.

Redis-backed tests use db=15 and flush it.  Everything else is pure.
"""

import json
import time

import numpy as np
import pytest
import redis.asyncio as aioredis

from core import search as search_mod
from core import store as mem_store
from core.search import BM25Index, extract_text, tier_of


# ─── pure: tier helpers ──────────────────────────────────────────────────────


class TestTierHelpers:
    def test_tier_of(self):
        assert tier_of("mem:facts") == "facts"
        assert tier_of("mem:episodes") == "episodes"
        assert tier_of("mem:procedures") == "procedures"

    def test_idx_key(self):
        assert mem_store.idx_key("mem:facts") == "mem:idx:facts"
        assert mem_store.idx_key("mem:episodes") == "mem:idx:episodes"
        assert mem_store.idx_key("mem:procedures") == "mem:idx:procedures"

    def test_extract_text_facts(self):
        attrs = {"content": "redis vectorset is fast", "triple_str": "a | uses | b"}
        assert extract_text("mem:facts", attrs) == "redis vectorset is fast a | uses | b"

    def test_extract_text_episodes(self):
        assert extract_text("mem:episodes", {"content": "we discussed deploy"}) == "we discussed deploy"

    def test_extract_text_procedures_prefers_task_then_procedure(self):
        """Older procedure rows have no `content` — task + procedure is the body."""
        attrs = {"task": "deploy aiserv", "procedure": "run docker compose up"}
        assert extract_text("mem:procedures", attrs) == "deploy aiserv run docker compose up"

    def test_extract_text_procedures_falls_back_to_content(self):
        assert extract_text("mem:procedures", {"content": "legacy row"}) == "legacy row"

    def test_extract_text_empty_when_no_text(self):
        assert extract_text("mem:facts", {"category": "rule"}) == ""
        assert extract_text("mem:procedures", {"tools_used": ["bash"]}) == ""

    def test_extract_text_skips_blank_strings(self):
        assert extract_text("mem:facts", {"content": "   "}) == ""


# ─── pure: BM25Index tier plumbing ───────────────────────────────────────────


class TestBM25IndexTiers:
    @pytest.fixture
    def idx(self):
        return BM25Index()

    async def test_add_records_tier(self, idx):
        await idx.add("f1", "redis vectorset", {"category": "rule"}, "facts")
        assert idx._docs[0][3] == "facts"

    async def test_add_skips_empty_content(self, idx):
        await idx.add("f1", "   ", {}, "facts")
        assert idx._docs == []

    async def test_search_returns_tier(self, idx):
        """Each tier's hit carries its own tier tag.

        The corpus needs filler documents: with a 2-doc corpus where every
        query term appears in every document, BM25's IDF collapses to ~0 and
        every score is 0, which search() correctly treats as "no match".
        """
        filler = [
            ("x1", "quarterly revenue forecast spreadsheet", "facts"),
            ("x2", "kubernetes ingress controller rewrite", "facts"),
            ("x3", "afternoon coffee with the design team", "episodes"),
            ("x4", "refactor the billing invoice renderer", "procedures"),
        ]
        for uid, text, tier in filler:
            await idx.add(uid, text, {}, tier)
        await idx.add("e1", "we discussed the docker compose deploy", {}, "episodes")
        await idx.add("p1", "deploy aiserv with docker compose", {}, "procedures")
        await idx._rebuild_async()
        hits = await idx.search("docker compose deploy", k=5)
        assert hits
        assert {h["tier"] for h in hits} == {"episodes", "procedures"}

    async def test_remove_drops_by_uid(self, idx):
        await idx.add("f1", "alpha beta", {}, "facts")
        await idx.add("f2", "alpha gamma", {}, "facts")
        await idx.remove({"f1"})
        assert [d[0] for d in idx._docs] == ["f2"]

    async def test_stats_counts_by_tier(self, idx):
        await idx.add("f1", "a", {}, "facts")
        await idx.add("f2", "b", {}, "facts")
        await idx.add("e1", "c", {}, "episodes")
        st = await idx.stats()
        assert st == {"total": 3, "by_tier": {"facts": 2, "episodes": 1}}

    async def test_populate_from_items_derives_text_per_tier(self, idx):
        items = [
            {"_element": "f1", "attrs": {"content": "a fact about redis"}, "tier": "facts"},
            # procedure with no `content` — text must come from task+procedure
            {"_element": "p1", "attrs": {"task": "deploy", "procedure": "compose up"}, "tier": "procedures"},
            {"_element": "e1", "attrs": {"content": "an episode"}, "tier": "episodes"},
        ]
        await idx.populate_from_items(items)
        by_uid = {d[0]: d for d in idx._docs}
        assert by_uid["p1"][1] == "deploy compose up"
        assert by_uid["f1"][3] == "facts"
        assert by_uid["e1"][3] == "episodes"


# ─── Redis: shadow index ─────────────────────────────────────────────────────


@pytest.fixture
async def r():
    client = aioredis.Redis(host="localhost", port=6379, db=15, decode_responses=True)
    await client.flushdb()
    # ensure_indexes writes the __seed__ placeholder each vectorset needs for
    # VSIM on an empty set — without it a 1-element set has VCARD == 1.
    await mem_store.ensure_indexes(client)
    yield client
    await client.flushdb()
    await client.aclose()


def _zero_vec() -> np.ndarray:
    """Zero vector at the live embedder dimension.

    ``core.search._zero_vec`` resolves the provider dimension (384 locally,
    768 for nomic), so a hard-coded 8-dim vector makes every VSIM in the
    fallback path fail with a dimension mismatch.
    """
    from core import embedder
    return np.zeros(embedder.DIMS, dtype=np.float32)


class TestShadowIndex:
    async def test_register_and_enumerate_roundtrip(self, r):
        vset = "mem:facts"
        for eid, content in (("e1", "first fact"), ("e2", "second fact")):
            await r.execute_command(
                "VADD", vset, "FP32", _zero_vec().tobytes(),
                eid, "SETATTR", json.dumps({"content": content}),
            )
            await mem_store.register_element(r, vset, eid)
        assert int(await r.scard(mem_store.idx_key(vset))) == 2

        items = await mem_store.enumerate_elements(r, vset)
        assert {i["element"] for i in items} == {"e1", "e2"}
        assert {i["attrs"]["content"] for i in items} == {"first fact", "second fact"}

    async def test_enumerate_empty_when_no_shadow(self, r):
        await r.execute_command(
            "VADD", "mem:facts", "FP32", _zero_vec().tobytes(),
            "e1", "SETATTR", json.dumps({"content": "orphan"}),
        )
        # No register_element call -> shadow index is empty -> enumerate returns []
        assert await mem_store.enumerate_elements(r, "mem:facts") == []

    async def test_enumerate_prunes_vanished_elements(self, r):
        await r.execute_command(
            "VADD", "mem:facts", "FP32", _zero_vec().tobytes(),
            "e1", "SETATTR", json.dumps({"content": "x"}),
        )
        await mem_store.register_element(r, "mem:facts", "e1")
        # Element deleted behind the index's back
        await r.execute_command("VREM", "mem:facts", "e1")
        items = await mem_store.enumerate_elements(r, "mem:facts")
        assert items == []
        assert int(await r.scard(mem_store.idx_key("mem:facts"))) == 0

    async def test_remove_element_drops_both(self, r):
        await r.execute_command(
            "VADD", "mem:facts", "FP32", _zero_vec().tobytes(),
            "e1", "SETATTR", json.dumps({"content": "x"}),
        )
        await mem_store.register_element(r, "mem:facts", "e1")
        await mem_store.remove_element(r, "mem:facts", "e1")
        assert int(await r.scard(mem_store.idx_key("mem:facts"))) == 0
        raw = await r.execute_command("VGETATTR", "mem:facts", "e1")
        assert not raw

    async def test_save_fact_registers_to_shadow(self, r):
        emb = _zero_vec()
        uid = await mem_store.save_fact(
            r, content="shadow registered fact", category="rule",
            confidence=0.9, embedding=emb,
        )
        assert int(await r.sismember(mem_store.idx_key(mem_store.FACT_KEY), uid)) == 1

    async def test_save_episode_registers_to_shadow(self, r):
        emb = _zero_vec()
        uid = await mem_store.save_episode(
            r, session_id="s1", content="shadow registered episode", embedding=emb,
        )
        assert int(await r.sismember(mem_store.idx_key(mem_store.EPISODE_KEY), uid)) == 1

    async def test_save_procedure_registers_to_shadow(self, r):
        emb = _zero_vec()
        uid = await mem_store.save_procedure(
            r, task="deploy the thing", procedure="run compose up", embedding=emb,
        )
        assert int(await r.sismember(mem_store.idx_key(mem_store.PROC_KEY), uid)) == 1

    async def test_scan_all_procedures_prefers_shadow(self, r):
        emb = _zero_vec()
        await mem_store.save_procedure(r, task="alpha task", procedure="alpha steps", embedding=emb)
        await mem_store.save_procedure(r, task="beta task", procedure="beta steps", embedding=emb)
        procs = await mem_store.scan_all_procedures(r)
        tasks = sorted(p["task"] for p in procs)
        assert tasks == ["alpha task", "beta task"]

    async def test_backfill_reports_shortfall(self, r):
        """Backfill registers what VSIM can reach and logs the rest."""
        emb = _zero_vec()
        for i in range(3):
            await mem_store.save_procedure(r, task=f"task {i}", procedure=f"steps {i}", embedding=emb)
        # Simulate elements written before the shadow index existed.
        await r.delete(mem_store.idx_key(mem_store.PROC_KEY))
        registered = await mem_store.backfill_shadow_index(r, mem_store.PROC_KEY)
        assert registered >= 1
        assert int(await r.scard(mem_store.idx_key(mem_store.PROC_KEY))) == registered

    async def test_enumerate_respects_max_count(self, r):
        emb = _zero_vec()
        for i in range(5):
            await mem_store.save_procedure(r, task=f"t{i}", procedure=f"s{i}", embedding=emb)
        items = await mem_store.enumerate_elements(r, mem_store.PROC_KEY, max_count=2)
        assert len(items) == 2


# ─── Redis: multi-tier BM25 population ───────────────────────────────────────


class TestMultiTierPopulation:
    async def test_populate_covers_all_tiers(self, r):
        emb = _zero_vec()
        # facts
        await mem_store.save_fact(r, content="redis vectorset backend", category="rule",
                                  confidence=0.9, embedding=emb)
        # episode
        await mem_store.save_episode(r, session_id="s", content="we deployed aiserv today",
                                     embedding=emb)
        # procedure
        await mem_store.save_procedure(r, task="deploy aiserv", procedure="docker compose up",
                                       embedding=emb)

        idx = BM25Index()
        await search_mod.populate_bm25_from_redis(r, idx)
        st = await idx.stats()
        assert st["by_tier"].get("facts", 0) >= 1
        assert st["by_tier"].get("episodes", 0) >= 1
        assert st["by_tier"].get("procedures", 0) >= 1
        assert st["total"] == sum(st["by_tier"].values())

    async def test_procedure_keyword_now_reachable(self, r):
        """The whole point of the fix: a procedure is keyword-findable."""
        emb = _zero_vec()
        await mem_store.save_procedure(
            r, task="restart the nginx gateway",
            procedure="systemctl restart nginx && check health endpoint",
            embedding=emb,
        )
        for i in range(4):  # filler so BM25 IDF is non-degenerate
            await mem_store.save_fact(r, content=f"unrelated filler fact number {i}",
                                      category="general", confidence=0.5, embedding=emb)
        idx = BM25Index()
        await search_mod.populate_bm25_from_redis(r, idx)
        hits = await idx.search("nginx gateway restart", k=5)
        assert hits, "procedure text must be reachable by keyword search"
        assert hits[0]["tier"] == "procedures"

    async def test_episode_keyword_now_reachable(self, r):
        emb = _zero_vec()
        await mem_store.save_episode(
            r, session_id="s", content="migrated the postgres cluster to a new region",
            embedding=emb,
        )
        for i in range(4):
            await mem_store.save_fact(r, content=f"unrelated filler fact number {i}",
                                      category="general", confidence=0.5, embedding=emb)
        idx = BM25Index()
        await search_mod.populate_bm25_from_redis(r, idx)
        hits = await idx.search("postgres cluster migration", k=5)
        assert hits
        assert hits[0]["tier"] == "episodes"

    async def test_superseded_facts_excluded(self, r):
        emb = _zero_vec()
        for i in range(4):
            await mem_store.save_fact(r, content=f"unrelated filler fact number {i}",
                                      category="general", confidence=0.5, embedding=emb)
        uid = await mem_store.save_fact(r, content="superseded fact about zebras",
                                        category="rule", confidence=0.9, embedding=emb)
        await mem_store.soft_delete_fact(r, uid, superseded_by="winner", reason="merged")
        idx = BM25Index()
        await search_mod.populate_bm25_from_redis(r, idx)
        hits = await idx.search("zebras", k=5)
        assert hits == []

    async def test_populate_is_idempotent(self, r):
        emb = _zero_vec()
        await mem_store.save_fact(r, content="idempotent fact", category="rule",
                                  confidence=0.9, embedding=emb)
        idx = BM25Index()
        await search_mod.populate_bm25_from_redis(r, idx)
        first = await idx.stats()
        await search_mod.populate_bm25_from_redis(r, idx)
        second = await idx.stats()
        assert first == second


# ─── Redis: procedure recovery via the reverse index ─────────────────────────


class TestProcedureRecovery:
    """mem:procedures' HNSW graph is degraded, but the element IDs survive in
    mem:proc_by_tool:* and mem:proc_tasks — recovery must harvest them and
    validate each one before registering it."""

    async def test_recovers_ids_from_reverse_index(self, r):
        emb = _zero_vec()
        uids = [
            await mem_store.save_procedure(
                r, task=f"recover task {i}", procedure=f"steps {i}",
                embedding=emb, tools_used=["bash", "docker"],
            )
            for i in range(3)
        ]
        # Simulate elements written before the shadow index existed.
        await r.delete(mem_store.idx_key(mem_store.PROC_KEY))
        assert int(await r.scard(mem_store.idx_key(mem_store.PROC_KEY))) == 0

        stats = await mem_store.recover_procedures_from_reverse_index(r)
        assert stats["live"] >= 3
        assert stats["newly_registered"] >= 3
        for uid in uids:
            assert int(await r.sismember(mem_store.idx_key(mem_store.PROC_KEY), uid)) == 1
        # Recovery makes the procedures enumerable again.
        procs = await mem_store.scan_all_procedures(r)
        assert sorted(p["task"] for p in procs) == [
            "recover task 0", "recover task 1", "recover task 2",
        ]

    async def test_recovers_ids_from_task_hash(self, r):
        """A procedure with no tools_used is only referenced by mem:proc_tasks."""
        emb = _zero_vec()
        uid = await mem_store.save_procedure(
            r, task="hash only task", procedure="steps", embedding=emb,
        )
        await r.delete(mem_store.idx_key(mem_store.PROC_KEY))
        stats = await mem_store.recover_procedures_from_reverse_index(r)
        assert uid in {
            m for m in await r.smembers(mem_store.idx_key(mem_store.PROC_KEY))
        }
        assert stats["task_index_rows"] >= 1

    async def test_stale_refs_are_pruned_not_registered(self, r):
        emb = _zero_vec()
        uid = await mem_store.save_procedure(
            r, task="live task", procedure="steps", embedding=emb, tools_used=["bash"],
        )
        await r.delete(mem_store.idx_key(mem_store.PROC_KEY))
        # A reference to a procedure that no longer exists.
        await r.sadd(f"{mem_store.PROC_BY_TOOL_PRE}bash", "deleted-proc-id")

        stats = await mem_store.recover_procedures_from_reverse_index(r)
        assert stats["stale"] >= 1
        assert stats["stale_refs_pruned"] >= 1
        members = await r.smembers(f"{mem_store.PROC_BY_TOOL_PRE}bash")
        assert "deleted-proc-id" not in members
        assert uid in members
        # The phantom must never enter the shadow index.
        assert int(await r.sismember(mem_store.idx_key(mem_store.PROC_KEY), "deleted-proc-id")) == 0

    async def test_recovery_is_idempotent(self, r):
        emb = _zero_vec()
        await mem_store.save_procedure(
            r, task="idempotent task", procedure="steps", embedding=emb,
            tools_used=["bash"],
        )
        await r.delete(mem_store.idx_key(mem_store.PROC_KEY))
        first = await mem_store.recover_procedures_from_reverse_index(r)
        second = await mem_store.recover_procedures_from_reverse_index(r)
        assert first["newly_registered"] >= 1
        assert second["newly_registered"] == 0
        assert second["already_registered"] == first["live"]

    async def test_prune_can_be_disabled(self, r):
        emb = _zero_vec()
        await mem_store.save_procedure(
            r, task="keep refs task", procedure="steps", embedding=emb, tools_used=["bash"],
        )
        await r.delete(mem_store.idx_key(mem_store.PROC_KEY))
        await r.sadd(f"{mem_store.PROC_BY_TOOL_PRE}bash", "deleted-proc-id")
        stats = await mem_store.recover_procedures_from_reverse_index(r, prune_stale=False)
        assert stats["stale"] >= 1
        assert stats["stale_refs_pruned"] == 0
        assert "deleted-proc-id" in await r.smembers(f"{mem_store.PROC_BY_TOOL_PRE}bash")

    async def test_no_reverse_index_is_a_noop(self, r):
        """Empty side structures must not invent elements or crash."""
        stats = await mem_store.recover_procedures_from_reverse_index(r)
        assert stats["ids_collected"] == 0
        assert stats["live"] == 0
        assert stats["newly_registered"] == 0

    async def test_ensure_skips_when_shadow_covers_vcard(self, r):
        """The guard is exact: shadow >= VCARD means nothing left to recover.

        VCARD counts the __seed__ placeholder too, so a "complete" index has
        to include it — hence the explicit SADD below.
        """
        emb = _zero_vec()
        await mem_store.save_procedure(
            r, task="complete task", procedure="steps", embedding=emb,
        )
        await r.sadd(mem_store.idx_key(mem_store.PROC_KEY), "__seed__")
        assert await mem_store.ensure_procedures_recovered(r) is None

    async def test_ensure_runs_when_shadow_short(self, r):
        emb = _zero_vec()
        uid = await mem_store.save_procedure(
            r, task="short task", procedure="steps", embedding=emb, tools_used=["bash"],
        )
        await r.delete(mem_store.idx_key(mem_store.PROC_KEY))
        stats = await mem_store.ensure_procedures_recovered(r)
        assert stats is not None
        assert int(await r.sismember(mem_store.idx_key(mem_store.PROC_KEY), uid)) == 1

    async def test_ensure_skips_empty_vectorset(self, r):
        """A vectorset holding only __seed__ has VCARD == 1 — nothing to recover."""
        assert await mem_store.ensure_procedures_recovered(r) is None


class TestProcedureDeleteCleansSideIndexes:
    """A procedure delete must not leave dangling IDs in the reverse indexes.

    30,977 stale references had accumulated in mem:proc_by_tool:* on this
    instance because nothing cleaned them; every /tool-procedures lookup paid
    a VGETATTR per corpse.
    """

    async def test_remove_element_cleans_reverse_index_and_task_hash(self, r):
        emb = _zero_vec()
        uid = await mem_store.save_procedure(
            r, task="delete me", procedure="steps", embedding=emb,
            tools_used=["bash", "Docker"],
        )
        bash_set = f"{mem_store.PROC_BY_TOOL_PRE}bash"
        docker_set = f"{mem_store.PROC_BY_TOOL_PRE}docker"
        assert uid in await r.smembers(bash_set)
        assert uid in await r.smembers(docker_set)
        assert await r.hget(mem_store.PROC_TASK_KEY, "delete me") == uid

        await mem_store.remove_element(r, mem_store.PROC_KEY, uid)

        assert uid not in await r.smembers(bash_set)
        assert uid not in await r.smembers(docker_set)
        assert await r.hget(mem_store.PROC_TASK_KEY, "delete me") is None
        assert int(await r.sismember(mem_store.idx_key(mem_store.PROC_KEY), uid)) == 0

    async def test_remove_element_leaves_other_procedures_indexed(self, r):
        emb = _zero_vec()
        keep = await mem_store.save_procedure(
            r, task="keep me", procedure="steps", embedding=emb, tools_used=["bash"],
        )
        drop = await mem_store.save_procedure(
            r, task="drop me", procedure="steps", embedding=emb, tools_used=["bash"],
        )
        await mem_store.remove_element(r, mem_store.PROC_KEY, drop)
        members = await r.smembers(f"{mem_store.PROC_BY_TOOL_PRE}bash")
        assert keep in members
        assert drop not in members

    async def test_remove_element_on_facts_does_not_touch_proc_indexes(self, r):
        """The side-index cleanup must be scoped to procedures only."""
        emb = _zero_vec()
        uid = await mem_store.save_fact(
            r, content="a plain fact", category="rule", confidence=0.9, embedding=emb,
        )
        proc_uid = await mem_store.save_procedure(
            r, task="unrelated", procedure="steps", embedding=emb, tools_used=["bash"],
        )
        await mem_store.remove_element(r, mem_store.FACT_KEY, uid)
        assert proc_uid in await r.smembers(f"{mem_store.PROC_BY_TOOL_PRE}bash")
        assert await r.hget(mem_store.PROC_TASK_KEY, "unrelated") == proc_uid

    async def test_remove_element_tolerates_missing_attrs(self, r):
        """A procedure row with no tools_used/task must not raise."""
        emb = _zero_vec()
        await r.execute_command(
            "VADD", mem_store.PROC_KEY, "FP32", _zero_vec().tobytes(),
            "bare-proc", "SETATTR", json.dumps({"success_count": 1}),
        )
        await mem_store.remove_element(r, mem_store.PROC_KEY, "bare-proc")
        assert not await r.execute_command("VGETATTR", mem_store.PROC_KEY, "bare-proc")
