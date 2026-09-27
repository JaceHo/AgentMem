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
        embedder._get_provider = lambda: _Down()
        with pytest.raises(httpx.ConnectError):
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
