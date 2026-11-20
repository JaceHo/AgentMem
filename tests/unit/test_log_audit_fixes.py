"""Regression tests for the v1.1.0 log audit fixes."""

import asyncio

import httpx
import pytest

from api.routes.capability import _chain_signature
from api.schemas.memory import StoreRequest
from core import embedder, extractor
from core.extractor import _looks_like_empty_array
from core.http import async_post_json_detailed


def test_store_request_keeps_the_tail_instead_of_422():
    messages = [{"role": "user", "content": str(i)} for i in range(80)]
    req = StoreRequest(messages=messages, session_id="s")
    assert len(req.messages) == 50
    assert req.messages[0].content == "30"
    assert req.messages[-1].content == "79"


def test_store_request_short_transcript_unchanged():
    req = StoreRequest(messages=[{"role": "assistant", "content": "hi"}])
    assert len(req.messages) == 1
    assert req.messages[0].content == "hi"


def test_chain_signature_ignores_order_and_weights():
    left = [
        {"chain": ["read", "edit", "bash"], "strength": 9},
        {"chain": ["bash", "grep", "read"], "strength": 3},
    ]
    right = [
        {"chain": ["bash", "grep", "read"], "strength": 100},
        {"chain": ["read", "edit", "bash"], "strength": 1},
    ]
    assert _chain_signature(left) == _chain_signature(right)
    changed = [{"chain": ["bash", "grep", "read"]}]
    assert _chain_signature(changed) != _chain_signature(left)


def test_empty_array_is_not_a_parse_failure():
    assert _looks_like_empty_array("[]")
    assert _looks_like_empty_array("```json\n[]\n```")
    assert not _looks_like_empty_array("no facts here")
    assert not _looks_like_empty_array('[{"content": "x"}]')


def test_extractor_circuit_opens_only_after_repeated_failures():
    extractor._model_fail_times.clear()
    extractor._model_fail_counts.clear()
    try:
        extractor.mark_model_failed("audit-test-model")
        assert not extractor.is_model_circuit_broken("audit-test-model")
        extractor.mark_model_failed("audit-test-model")
        assert extractor.is_model_circuit_broken("audit-test-model")
        extractor.clear_model_circuit("audit-test-model")
        assert not extractor.is_model_circuit_broken("audit-test-model")
    finally:
        extractor.clear_model_circuit("audit-test-model")


def test_embedding_circuit_fails_fast_after_an_outage():
    class _Down:
        def encode(self, text):
            raise httpx.ConnectError("connection refused")

    embedder._embed_down_until = 0.0
    try:
        original = embedder._get_provider
        embedder._get_provider = lambda measure=True: _Down()
        # Soft-fail: first outage becomes EmbeddingUnavailable (not raw ConnectError)
        with pytest.raises(embedder.EmbeddingUnavailable):
            embedder.encode("audit")
        assert embedder.embedding_unavailable()
        with pytest.raises(embedder.EmbeddingUnavailable):
            embedder.encode("audit-again")
    finally:
        embedder._get_provider = original
        embedder._embed_down_until = 0.0


async def test_post_json_classifies_timeout():
    class _Slow:
        async def post(self, *args, **kwargs):
            await asyncio.sleep(2)

    original = None

    async def _client():
        return _Slow()

    import core.http as http_mod
    original = http_mod._get_client
    http_mod._get_client = _client
    try:
        result = await async_post_json_detailed("http://127.0.0.1:9/v1", timeout=0.05)
    finally:
        http_mod._get_client = original
    assert result.data is None
    assert result.error == "timeout"


# ── v1.1.1: 2026-11-19 ollama outage audit ───────────────────────────────────
# One upstream outage produced 3422 HTTP 500s, 2008 tracebacks and ~800
# duplicate warnings. These pin the degradation behaviour that fixes it.


def test_is_expected_outage_classifies_only_dependency_failures():
    from core import embedder as emb

    assert emb.is_expected_outage(emb.EmbeddingUnavailable("circuit open"))
    assert emb.is_expected_outage(httpx.ConnectError("connection refused"))
    assert emb.is_expected_outage(TimeoutError("timed out"))
    # A genuine code bug must NOT be downgraded to a one-line warning.
    assert not emb.is_expected_outage(ValueError("bad argument"))
    assert not emb.is_expected_outage(KeyError("missing key"))


async def test_kb_search_degrades_to_bm25_when_circuit_open(monkeypatch):
    """An embed outage must return keyword results, never a 500."""
    from core import kb as kb_mod

    kb = kb_mod.get_kb()

    async def _boom(_text):
        raise embedder.EmbeddingUnavailable("embedding circuit open")

    async def _bm25(_query, _k):
        return [{"uid": "u", "attrs": {"content": "match"}, "score": 1.0}]

    async def _no_expand(_query):
        return ""

    monkeypatch.setattr(kb_mod, "async_encode", _boom)
    monkeypatch.setattr(kb_mod, "expand_query", _no_expand)
    monkeypatch.setattr(kb, "bm25", type("B", (), {"search": staticmethod(_bm25)})())

    out = await kb.search(object(), "match", k=3)
    assert out and out[0]["text"] == "match", "BM25 leg must survive the outage"
    assert all(c["dense"] == 0.0 for c in out), "no dense leg should have run"


async def test_async_encode_batch_chunked_fails_fast_on_open_circuit(monkeypatch):
    """Three backoffs cannot recover a 30s circuit — stop on the first failure."""
    from core import search as search_mod

    calls = []

    def _encode_batch(_texts):
        calls.append(1)
        raise embedder.EmbeddingUnavailable("embedding circuit open")

    monkeypatch.setattr(search_mod, "encode_batch", _encode_batch)
    out = await search_mod.async_encode_batch_chunked(["a", "b"])
    assert out is None
    assert len(calls) == 1, f"expected one attempt, got {len(calls)} retries"
