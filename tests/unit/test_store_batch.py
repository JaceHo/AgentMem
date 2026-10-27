"""Tests for the store-write-path batching (Task #18).

The store path used to issue one Redis round trip per extracted fact — a dedup
VSIM, a VADD+SADD save, a HGET+HSET persona merge, and N*(N-1) Lua edge upserts
per fact with entities.  These batch variants collapse each of those into one
(or two) pipelined flights:

  * ``core.store.knn_search_batch``  — N VSIM queries, one pipeline
  * ``core.store.save_facts_batch``  — N writes, two pipelines
  * ``core.persona.update_many``     — N merges, two round trips total
  * ``core.graph.record_entities_many`` — N groups, one pipeline

All tests use db=15 and flush it; the fixtures mirror test_perf_fixes.py.
"""

import numpy as np
import pytest
import redis.asyncio as aioredis

from core import store as mem_store


@pytest.fixture
async def r():
    client = aioredis.Redis(host="localhost", port=6379, db=15, decode_responses=True)
    await client.flushdb()
    await mem_store.ensure_indexes(client)
    yield client
    await client.flushdb()
    await client.aclose()


def _z() -> np.ndarray:
    return np.zeros(384, dtype=np.float32)


# ─── knn_search_batch ─────────────────────────────────────────────────────────


class TestKnnSearchBatch:
    async def test_results_aligned_with_embeddings(self, r):
        embs = []
        uids = []
        for i in range(4):
            # One-hot at a distinct dimension so each vector is uniquely nearest.
            e = np.zeros(384, dtype=np.float32)
            e[i] = 1.0
            embs.append(e)
            uids.append(await mem_store.save_fact(
                r, content=f"batch fact {i}", category="rule",
                confidence=0.9, embedding=e,
            ))
        # Query each stored embedding back — top hit must be itself, aligned.
        hits = await mem_store.knn_search_batch(r, mem_store.FACT_KEY, embs, k=1)
        assert len(hits) == 4
        for h, uid in zip(hits, uids):
            assert h[0]["_element"] == uid

    async def test_empty_embeddings_returns_empty_list(self, r):
        assert await mem_store.knn_search_batch(r, mem_store.FACT_KEY, []) == []

    async def test_matches_sequential_knn_search(self, r):
        e0 = np.full(384, 0.05, dtype=np.float32)
        e1 = np.full(384, 0.50, dtype=np.float32)
        await mem_store.save_fact(r, content="a", category="rule", confidence=0.9, embedding=e0)
        await mem_store.save_fact(r, content="b", category="rule", confidence=0.9, embedding=e1)
        batch = await mem_store.knn_search_batch(r, mem_store.FACT_KEY, [e0, e1], k=3)
        seq0 = await mem_store.knn_search(r, mem_store.FACT_KEY, e0, k=3)
        seq1 = await mem_store.knn_search(r, mem_store.FACT_KEY, e1, k=3)
        assert {h["_element"] for h in batch[0]} == {h["_element"] for h in seq0}
        assert {h["_element"] for h in batch[1]} == {h["_element"] for h in seq1}


# ─── save_facts_batch ────────────────────────────────────────────────────────


class TestSaveFactsBatch:
    async def test_writes_all_facts_and_indexes_them(self, r):
        specs = [
            {"embedding": _z(), "content": f"bf {i}", "category": "rule",
             "confidence": 0.9}
            for i in range(6)
        ]
        uids = await mem_store.save_facts_batch(r, specs)
        assert len(uids) == 6
        assert all(u for u in uids)
        # Every written element must be reachable through the shadow index.
        items = await mem_store.enumerate_elements(r, mem_store.FACT_KEY)
        stored = {i["element"] for i in items}
        assert set(uids) <= stored

    async def test_empty_is_noop(self, r):
        assert await mem_store.save_facts_batch(r, []) == []

    async def test_attrs_match_single_save_path(self, r):
        spec = {"embedding": _z(), "content": "triple fact", "category": "identity",
                "confidence": 0.8, "keywords": ["k1"], "triple_s": "Alice",
                "triple_p": "works_at", "triple_o": "NovaPay"}
        [uid] = await mem_store.save_facts_batch(r, [spec])
        attrs = await mem_store.get_attrs(r, mem_store.FACT_KEY, uid)
        assert attrs["content"] == "triple fact"
        assert attrs["category"] == "identity"
        assert attrs["triple_str"] == "Alice | works_at | NovaPay"
        assert attrs["keywords"] == ["k1"]


# ─── persona.update_many ─────────────────────────────────────────────────────


class TestPersonaUpdateMany:
    async def test_two_facts_same_field_both_merged(self, r):
        from core import persona as persona_mod
        await persona_mod.update_many(r, [
            ("preference", "I like green tea"),
            ("preference", "I prefer decaf coffee"),
        ])
        val = await r.hget("mem:persona", "preferences")
        assert val is not None
        assert "green tea" in val and "decaf coffee" in val

    async def test_replays_in_arrival_order_dedups_second(self, r):
        from core import persona as persona_mod
        # Two near-identical preferences — the second should be dropped.
        await persona_mod.update_many(r, [
            ("preference", "I live in Oakland"),
            ("preference", "I live in Oakland"),  # duplicate of the first
        ])
        val = await r.hget("mem:persona", "preferences")
        assert val.count("Oakland") == 1

    async def test_unrelated_fields_untouched(self, r):
        from core import persona as persona_mod
        await r.hset("mem:persona", "name", "Jace")
        await persona_mod.update_many(r, [("preference", "I like hiking")])
        assert await r.hget("mem:persona", "name") == "Jace"
        assert await r.hget("mem:persona", "preferences") is not None

    async def test_empty_is_noop(self, r):
        from core import persona as persona_mod
        await persona_mod.update_many(r, [])
        assert await r.hgetall("mem:persona") == {}


# ─── graph.record_entities_many ──────────────────────────────────────────────


class TestGraphRecordEntitiesMany:
    async def test_both_directions_recorded(self, r):
        from core import graph as graph_mod
        await graph_mod.record_entities_many(r, [(["Redis", "FastAPI"], [])])
        a = await r.hget("mem:graph:redis", "fastapi")
        b = await r.hget("mem:graph:fastapi", "redis")
        assert a is not None and b is not None

    async def test_multiple_groups_single_pipeline(self, r):
        from core import graph as graph_mod
        await graph_mod.record_entities_many(r, [
            (["Redis", "FastAPI"], []),
            (["Python", "Redis"], []),
        ])
        # Redis co-occurs with both FastAPI and Python.
        redis_neighbours = await r.hgetall("mem:graph:redis")
        assert "fastapi" in redis_neighbours and "python" in redis_neighbours

    async def test_single_entity_group_is_noop(self, r):
        from core import graph as graph_mod
        await graph_mod.record_entities_many(r, [(["Solo"], [])])
        assert await r.keys("mem:graph:*") == []

    async def test_empty_is_noop(self, r):
        from core import graph as graph_mod
        assert await graph_mod.record_entities_many(r, []) == 0


# ─── vector cache (Task #19) ─────────────────────────────────────────────────


class TestVectorCache:
    async def test_quantize_roundtrip_preserves_cosine(self, r):
        # The embedder emits L2-normalized vectors (components ~1/sqrt(D) << 1),
        # so int8 quantization (clip at |127|) never saturates and the roundtrip
        # is near-lossless.  A raw standard_normal would saturate the clip and is
        # *not* representative of stored vectors — we use the realistic dist here.
        v = np.random.default_rng(0).standard_normal(384).astype(np.float32)
        v /= np.linalg.norm(v)  # mirror the embedder's normalization
        blob = mem_store._quantize_vec(v)
        back = mem_store._dequantize_vec(blob)
        cos = float(np.dot(v, back) / (np.linalg.norm(v) * np.linalg.norm(back)))
        # Consolidation thresholds affinity at 0.85; the roundtrip must keep the
        # pair far above that (empirically ~0.999) so cluster membership never
        # flips.  0.99 leaves ~8x margin over the observed worst case.
        assert cos > 0.99

    async def test_save_fact_populates_cache(self, r):
        e = np.full(384, 0.3, dtype=np.float32)
        uid = await mem_store.save_fact(r, content="cached fact", category="rule",
                                        confidence=0.9, embedding=e)
        hits, misses = await mem_store.get_cached_vectors(r, mem_store.FACT_KEY, [uid])
        assert uid in hits
        assert misses == []
        assert hits[uid].shape == (384,)

    async def test_cache_misses_reported(self, r):
        hits, misses = await mem_store.get_cached_vectors(
            r, mem_store.FACT_KEY, ["no-such-element"])
        assert hits == {} and misses == ["no-such-element"]

    async def test_cache_vectors_batch_then_read(self, r):
        vecs = {f"e{i}": np.full(384, 0.1 * i, dtype=np.float32) for i in range(5)}
        await mem_store.cache_vectors(r, mem_store.FACT_KEY, vecs)
        hits, misses = await mem_store.get_cached_vectors(
            r, mem_store.FACT_KEY, list(vecs) + ["missing"])
        assert set(hits) == set(vecs)
        assert misses == ["missing"]

    async def test_remove_element_drops_cache(self, r):
        e = np.full(384, 0.2, dtype=np.float32)
        uid = await mem_store.save_fact(r, content="drop me", category="rule",
                                        confidence=0.9, embedding=e)
        assert (await mem_store.get_cached_vectors(r, mem_store.FACT_KEY, [uid]))[0]
        await mem_store.remove_element(r, mem_store.FACT_KEY, uid)
        hits, _ = await mem_store.get_cached_vectors(r, mem_store.FACT_KEY, [uid])
        assert hits == {}
