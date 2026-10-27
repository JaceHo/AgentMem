"""BM25 index completeness after warm-up.

Measured on the live service 2026-10-27: /kb/status reported docs=1775 while the
index had been built from 1771. add() only rebuilds once 10 docs accumulate and
search() uses the same threshold, so the tail of a warm-up stays invisible to the
keyword leg — and which chunks are missing follows Redis hgetall order, so every
restart silently searches a slightly different corpus.
"""
import asyncio
import json

import pytest

from core import kb as kb_mod
from core import search as search_mod


pytestmark = pytest.mark.skipif(not search_mod.BM25_AVAILABLE,
                                reason="rank_bm25 not installed")


def _run(coro):
    return asyncio.run(coro)


def test_tail_below_batch_threshold_is_not_indexed():
    async def scenario():
        idx = search_mod.BM25Index()
        await idx.add("c0", "alpha bravo zulu0 document", {"content": "d0"})
        await asyncio.sleep(0.1)      # the task created by the first add settles
        await idx.force_rebuild()     # built == 1
        # Two more docs: 2 < _rebuild_threshold, so no background rebuild runs.
        await idx.add("c1", "charlie delta zulu1 document", {"content": "d1"})
        await idx.add("c2", "echo foxtrot zulu2 document", {"content": "d2"})
        await asyncio.sleep(0.1)
        return idx

    idx = _run(scenario())
    assert idx._built_at_len == 1
    assert [h["_element"] for h in _run(idx.search("zulu2", 5))] == []


def test_force_rebuild_indexes_the_tail():
    async def scenario():
        idx = search_mod.BM25Index()
        await idx.add("c0", "alpha bravo zulu0 document", {"content": "d0"})
        await idx.add("c1", "charlie delta zulu1 document", {"content": "d1"})
        await idx.add("c2", "echo foxtrot zulu2 document", {"content": "d2"})
        await asyncio.sleep(0.05)
        await idx.force_rebuild()
        return idx

    idx = _run(scenario())
    assert idx._built_at_len == 3
    assert _run(idx.search("zulu2", 5))[0]["_element"] == "c2"


class _FakeRedis:
    """Only the two reads warm() performs."""

    def __init__(self, docs):
        self._docs = docs          # element -> content
        self._meta = {k.split(":")[0]: {"title": f"t-{k}", "path": f"/p/{k}"}
                      for k in docs}

    async def hgetall(self, key):
        assert key == kb_mod.KB_TEXT_KEY
        return dict(self._docs)

    async def hget(self, key, doc_id):
        assert key == kb_mod.KB_DOC_KEY
        meta = self._meta.get(doc_id)
        return json.dumps(meta) if meta else None


def test_warm_leaves_nothing_out_of_the_index():
    # 7 chunks: below the 10-doc threshold, so the old warm() ended with an
    # index built from 0 of them.
    docs = {f"kb_doc{i}:{i}": f"contents number{i} token{i}" for i in range(7)}
    kb = kb_mod.KBStore()
    n = _run(kb.warm(_FakeRedis(docs)))
    assert n == 7
    assert kb.bm25._built_at_len == 7
    assert _run(kb.bm25.search("token6", 3))[0]["_element"] == "kb_doc6:6"
