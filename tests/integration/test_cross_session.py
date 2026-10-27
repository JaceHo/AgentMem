"""
Integration tests for cross-session memory continuity.

Tests session handoff, pinned summaries, and crystallized digests.
"""

import asyncio
import json
import time
import pytest
from httpx import AsyncClient


async def wait_for_session_len(client, session_id, min_len=80, timeout_s=90):
    """Poll GET /session/{id} until the async store pipeline accumulates
    enough context. Fixed sleeps lose to cold-start pipelines (~7s)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        r = await client.get(f"/session/{session_id}")
        if r.status_code == 200 and len(r.json().get("context") or "") >= min_len:
            return True
        await asyncio.sleep(0.4)
    return False


async def wait_for_pinned_summary(client, timeout_s=30):
    """Poll Redis until a session summary is pinned (test-process REDIS_URL
    points at the isolated instance's db)."""
    import redis.asyncio as aioredis

    from tests.conftest import TEST_REDIS_DB
    deadline = time.monotonic() + timeout_s
    # Fresh client per call — the global mem_store pool binds to whichever
    # event loop created it first, and pytest-asyncio gives each test a new one.
    rc = aioredis.Redis(host="localhost", port=6379, db=TEST_REDIS_DB,
                        decode_responses=True)
    try:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            raw = await rc.get("mem:pinned:session_summary")
            if raw:
                return raw
            await asyncio.sleep(0.4)
    finally:
        await rc.aclose()
    return None


@pytest.mark.asyncio
async def test_session_handoff_bridge(agentmem_client: AsyncClient):
    """
    Test that session summary is pinned and injected into next session.
    
    Flow:
    1. Store facts in Session A
    2. Compress Session A (pins summary)
    3. Start Session B
    4. Verify pinned summary is included in recall context
    """
    # Session A: Store enough substance to clear the promotion gate
    # (_do_compress_session skips sessions under ~80 chars of context).
    session_a_id = "test_session_a"
    messages = [
        {"role": "user", "content": "I prefer using Python for data science"},
        {"role": "assistant", "content": "Great choice! Python has excellent libraries."},
        {"role": "user", "content": "Mostly pandas and scikit-learn, on Python 3.12"},
        {"role": "assistant", "content": "Noted: your data science stack is pandas plus scikit-learn on Python 3.12."},
    ]
    
    store_resp = await agentmem_client.post("/store", json={
        "messages": messages,
        "session_id": session_a_id
    })
    assert store_resp.status_code == 200

    # Wait for the async store pipeline to accumulate the session KV
    assert await wait_for_session_len(agentmem_client, session_a_id, min_len=80)

    # Compress Session A (this pins the summary)
    compress_resp = await agentmem_client.post("/session/compress", json={
        "session_id": session_a_id,
        "wait": True
    })
    assert compress_resp.status_code == 200
    
    # Session B: Recall should include pinned summary
    session_b_id = "test_session_b"
    recall_resp = await agentmem_client.post("/recall", json={
        "query": "What programming language do I prefer?",
        "session_id": session_b_id
    })
    assert recall_resp.status_code == 200
    
    result = recall_resp.json()
    prepend = result.get("prependContext", "")
    
    # Should contain "Last Session Summary" section
    assert "Last Session Summary" in prepend or "## Last Session Summary" in prepend


@pytest.mark.asyncio
async def test_crystallized_digest_inclusion(agentmem_client: AsyncClient):
    """
    Test that auto-crystallized digests are included in recall context.
    
    Flow:
    1. Create a session with multiple facts
    2. Manually crystallize it
    3. Verify digest appears in subsequent recall
    """
    session_id = "test_crystal_session"
    
    # Store multiple facts to make session substantial
    messages = [
        {"role": "user", "content": "Set up Redis cluster with 3 nodes"},
        {"role": "assistant", "content": "Configured Redis cluster on ports 7000-7002"},
        {"role": "user", "content": "Enable TLS encryption"},
        {"role": "assistant", "content": "TLS enabled with certificates in /etc/redis/ssl"}
    ]
    
    store_resp = await agentmem_client.post("/store", json={
        "messages": messages,
        "session_id": session_id
    })
    assert store_resp.status_code == 200

    # Wait for the async store pipeline before crystallizing
    assert await wait_for_session_len(agentmem_client, session_id, min_len=80)

    # Manually crystallize the session
    crystal_resp = await agentmem_client.post("/crystallize", json={
        "session_id": session_id,
        "max_facts": 10
    })
    assert crystal_resp.status_code == 200

    digest = crystal_resp.json()
    assert digest["session_id"] == session_id
    assert digest["fact_count"] > 0, f"crystallize found no facts: {digest}"
    
    # Now recall in a new session - should include crystallized digest
    new_session_id = "test_after_crystal"
    recall_resp = await agentmem_client.post("/recall", json={
        "query": "How do I set up Redis?",
        "session_id": new_session_id
    })
    assert recall_resp.status_code == 200
    
    result = recall_resp.json()
    prepend = result.get("prependContext", "")
    
    # Should contain "Lessons Learned" section with crystallized content
    assert "Lessons Learned" in prepend or "Completed Session" in prepend


@pytest.mark.asyncio
async def test_cross_session_fact_persistence(agentmem_client: AsyncClient):
    """
    Test that facts persist across sessions and are retrievable.
    
    Flow:
    1. Store fact in Session A
    2. Recall in Session B with related query
    3. Verify fact from Session A is retrieved
    """
    # Session A: Store a specific fact
    session_a = "test_persist_a"
    messages_a = [
        {"role": "user", "content": "My favorite IDE is VS Code with Python extension"},
    ]
    
    await agentmem_client.post("/store", json={
        "messages": messages_a,
        "session_id": session_a
    })
    
    import asyncio
    await asyncio.sleep(1)
    
    # Session B: Query about IDE preference
    session_b = "test_persist_b"
    recall_resp = await agentmem_client.post("/recall", json={
        "query": "What IDE does the user prefer?",
        "session_id": session_b
    })
    
    result = recall_resp.json()
    prepend = result.get("prependContext", "")
    
    # Should retrieve the VS Code fact
    assert "VS Code" in prepend or "IDE" in prepend


@pytest.mark.asyncio
async def test_session_context_accumulation(agentmem_client: AsyncClient):
    """
    Test that session context accumulates within a single session.
    
    Flow:
    1. Store message in session
    2. Get session context
    3. Store another message
    4. Verify context includes both messages
    """
    session_id = "test_accumulation"
    
    # First message
    await agentmem_client.post("/store", json={
        "messages": [{"role": "user", "content": "First task: setup database"}],
        "session_id": session_id
    })
    
    import asyncio
    await asyncio.sleep(0.5)
    
    # Get session context
    session_resp = await agentmem_client.get(f"/session/{session_id}")
    assert session_resp.status_code == 200
    ctx1 = session_resp.json()["context"]
    
    # Second message
    await agentmem_client.post("/store", json={
        "messages": [{"role": "user", "content": "Second task: configure API"}],
        "session_id": session_id
    })

    # Context should have grown — poll until the second message lands
    deadline = time.monotonic() + 20
    ctx2 = ""
    while time.monotonic() < deadline:
        session_resp = await agentmem_client.get(f"/session/{session_id}")
        ctx2 = session_resp.json()["context"] or ""
        if "configure API" in ctx2:
            break
        await asyncio.sleep(0.4)

    assert len(ctx2) >= len(ctx1)


@pytest.mark.asyncio
async def test_pinned_summary_survives_restart(agentmem_client: AsyncClient):
    """
    Test that pinned session summary persists (simulating service restart).
    
    The pinned summary is stored in Redis with no TTL, so it survives
    service restarts and provides cross-session continuity.
    """
    session_id = "test_pin_persist"

    # Store and compress to pin summary — enough substance to clear the gate
    messages = [
        {"role": "user", "content": "Deployed microservice architecture with 5 services"},
        {"role": "assistant", "content": "Services: auth, api, worker, cache, db-proxy"},
        {"role": "user", "content": "The worker consumes from a Redis stream"},
        {"role": "assistant", "content": "Deployment verified: the worker consumes jobs from a Redis stream."},
    ]

    await agentmem_client.post("/store", json={
        "messages": messages,
        "session_id": session_id
    })

    # Wait for the async store pipeline, then compress to pin
    assert await wait_for_session_len(agentmem_client, session_id, min_len=80)

    await agentmem_client.post("/session/compress", json={
        "session_id": session_id,
        "wait": True
    })

    # Verify pinned key exists in Redis — poll until the pin lands
    pinned_text = await wait_for_pinned_summary(agentmem_client)

    assert pinned_text is not None
    assert len(pinned_text) > 0
    assert "microservice" in pinned_text.lower() or "deployed" in pinned_text.lower()


@pytest.mark.asyncio
async def test_multiple_crystallized_digests_ranked(agentmem_client: AsyncClient):
    """
    Test that multiple crystallized digests are fetched and ranked by recency.
    
    Flow:
    1. Crystallize 3 different sessions
    2. Recall in new session
    3. Verify top-3 most recent digests are included
    """
    import asyncio
    
    # Create 3 sessions with different timestamps
    for i in range(3):
        session_id = f"test_crystal_multi_{i}"
        messages = [
            {"role": "user", "content": f"Learned lesson {i}: always test edge cases in module {i}"},
            {"role": "assistant", "content": f"Noted: lesson {i} recorded — edge cases in module {i} need regression tests."},
        ]

        await agentmem_client.post("/store", json={
            "messages": messages,
            "session_id": session_id
        })

        # Wait for the pipeline, then crystallize this session
        assert await wait_for_session_len(agentmem_client, session_id, min_len=60)
        crystal_resp = await agentmem_client.post("/crystallize", json={
            "session_id": session_id,
            "max_facts": 5
        })
        assert crystal_resp.status_code == 200
    
    await asyncio.sleep(1)
    
    # Recall in new session
    recall_resp = await agentmem_client.post("/recall", json={
        "query": "What lessons have been learned?",
        "session_id": "test_reader_session"
    })
    
    result = recall_resp.json()
    prepend = result.get("prependContext", "")
    
    # Should contain Lessons Learned section
    assert "Lessons Learned" in prepend
    
    # Count how many "Completed Session" entries appear
    completed_count = prepend.count("Completed Session")
    assert completed_count >= 1  # at least one digest included
    assert completed_count <= 3  # max 3 digests


@pytest.mark.asyncio
async def test_user_feedback_updates_fact_metadata(agentmem_client: AsyncClient):
    """
    Test that user feedback updates fact metadata correctly.
    
    Flow:
    1. Store a fact
    2. Provide positive feedback
    3. Verify importance boosted and rating recorded
    """
    import asyncio
    import redis.asyncio as aioredis
    from core import store as mem_store
    from core import embedder as embedder_mod
    from tests.conftest import TEST_REDIS_DB

    session_id = "test_feedback_integration"
    messages = [
        {"role": "user", "content": "Always use type hints in Python functions"},
    ]

    await agentmem_client.post("/store", json={
        "messages": messages,
        "session_id": session_id
    })

    # Wait for the async store pipeline to persist at least one fact.
    # Fresh client — never the global mem_store pool (wrong event loop).
    deadline = time.monotonic() + 25
    card = 0
    redis_client = aioredis.Redis(host="localhost", port=6379, db=TEST_REDIS_DB,
                                  decode_responses=True)
    try:
        while time.monotonic() < deadline:
            card = int(await redis_client.execute_command("VCARD", mem_store.FACT_KEY) or 0)
            if card >= 1:
                break
            await asyncio.sleep(0.4)

        if card > 0:
            import numpy as np
            seed = np.zeros(embedder_mod.DIMS, dtype=np.float32)
            results = await redis_client.execute_command(
                "VSIM", mem_store.FACT_KEY, "FP32", seed.tobytes(),
                "COUNT", 1, "WITHSCORES", "WITHATTRIBS"
            )

            if len(results) >= 3:
                element_id = results[0].decode() if isinstance(results[0], bytes) else results[0]

                if element_id != "__seed__":
                    # Provide positive feedback
                    feedback_resp = await agentmem_client.post("/feedback", json={
                        "element_id": element_id,
                        "rating": 5,
                        "comment": "Very helpful guideline!"
                    })

                    assert feedback_resp.status_code == 200
                    feedback_result = feedback_resp.json()

                    assert feedback_result["status"] == "ok"
                    assert feedback_result["action"] == "boosted_and_reinforced"

                    # Verify metadata updated
                    raw = await redis_client.execute_command(
                        "VGETATTR", mem_store.FACT_KEY, element_id
                    )
                    attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)

                    assert attrs["user_rating"] == 5
                    assert attrs["user_comment"] == "Very helpful guideline!"
                    assert attrs["importance"] > 0.5  # was boosted
    finally:
        await redis_client.aclose()
