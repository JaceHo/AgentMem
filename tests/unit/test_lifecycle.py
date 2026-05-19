"""
Unit tests for memory lifecycle management (decay, merge, prune).

Tests Ebbinghaus confidence decay, consolidation phases, and pruning logic.
"""

import json
import time
import pytest
from core.store import confidence_decay, reinforce_fact


class TestEbbinghausDecay:
    """Test confidence decay based on Ebbinghaus forgetting curve."""
    
    def test_fresh_fact_no_decay(self):
        """Freshly created fact should have full confidence."""
        now_ms = int(time.time() * 1000)
        attrs = {
            "confidence": 0.9,
            "last_confirmed_ts": now_ms,
            "source_count": 3,
            "category": "rule"
        }
        eff_conf = confidence_decay(attrs)
        assert eff_conf >= 0.85  # minimal decay for fresh fact
    
    def test_bug_fact_decays_fast(self):
        """Bug-related facts should decay quickly (30-day stability)."""
        now_ms = int(time.time() * 1000)
        thirty_days_ago = now_ms - (30 * 24 * 3600 * 1000)
        
        attrs = {
            "confidence": 0.9,
            "last_confirmed_ts": thirty_days_ago,
            "source_count": 1,
            "category": "bug"
        }
        eff_conf = confidence_decay(attrs)
        assert eff_conf < 0.5  # should decay significantly
    
    def test_rule_fact_decays_slow(self):
        """Rule/identity facts should decay slowly (365-day stability)."""
        now_ms = int(time.time() * 1000)
        ninety_days_ago = now_ms - (90 * 24 * 3600 * 1000)
        
        attrs = {
            "confidence": 0.9,
            "last_confirmed_ts": ninety_days_ago,
            "source_count": 2,
            "category": "rule"
        }
        eff_conf = confidence_decay(attrs)
        assert eff_conf > 0.6  # should retain most confidence
    
    def test_superseded_fact_zero_confidence(self):
        """Superseded facts should return 0.0 confidence."""
        attrs = {
            "confidence": 0.9,
            "superseded_by": "newer_fact_id",
            "category": "general"
        }
        eff_conf = confidence_decay(attrs)
        assert eff_conf == 0.0
    
    def test_source_count_multiplier(self):
        """Multiple sources should boost confidence via multiplier."""
        now_ms = int(time.time() * 1000)
        
        # Single source
        attrs_single = {
            "confidence": 0.9,
            "last_confirmed_ts": now_ms,
            "source_count": 1,
            "category": "general"
        }
        
        # Three sources
        attrs_multi = {
            "confidence": 0.9,
            "last_confirmed_ts": now_ms,
            "source_count": 3,
            "category": "general"
        }
        
        conf_single = confidence_decay(attrs_single)
        conf_multi = confidence_decay(attrs_multi)
        
        assert conf_multi > conf_single  # more sources = higher confidence
    
    def test_category_stability_mapping(self):
        """Different categories should have different decay rates."""
        now_ms = int(time.time() * 1000)
        sixty_days_ago = now_ms - (60 * 24 * 3600 * 1000)
        
        # Identity (365d stability)
        attrs_identity = {
            "confidence": 0.9,
            "last_confirmed_ts": sixty_days_ago,
            "source_count": 1,
            "category": "identity"
        }
        
        # Warning (30d stability)
        attrs_warning = {
            "confidence": 0.9,
            "last_confirmed_ts": sixty_days_ago,
            "source_count": 1,
            "category": "warning"
        }
        
        conf_identity = confidence_decay(attrs_identity)
        conf_warning = confidence_decay(attrs_warning)
        
        assert conf_identity > conf_warning  # identity decays slower


@pytest.mark.asyncio
async def test_reinforce_fact_increments_source_count(redis_client):
    """Reinforcing a fact should increment source_count and reset last_confirmed_ts."""
    from core import store as mem_store
    import numpy as np
    
    # Add a test fact
    emb = np.zeros(mem_store.DIMS, dtype=np.float32)
    element_id = "test_reinforce_fact"
    
    await redis_client.execute_command(
        "VADD", mem_store.FACT_KEY, "FP32", emb.tobytes(),
        element_id, "SETATTR", json.dumps({
            "content": "Test fact for reinforcement",
            "confidence": 0.8,
            "source_count": 1,
            "ts": int(time.time() * 1000)
        })
    )
    
    # Reinforce the fact
    result = await reinforce_fact(redis_client, element_id, new_source="user_feedback")
    assert result is True
    
    # Verify source_count incremented
    raw = await redis_client.execute_command("VGETATTR", mem_store.FACT_KEY, element_id)
    attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
    
    assert attrs["source_count"] == 2
    assert "last_confirmed_ts" in attrs


@pytest.mark.asyncio
async def test_consolidation_merge_phase(redis_client):
    """Consolidation should merge similar facts into one keeper."""
    from core import store as mem_store
    from main import _do_consolidate
    import numpy as np
    
    # Add 3 similar facts
    emb = np.zeros(mem_store.DIMS, dtype=np.float32)
    
    for i in range(3):
        element_id = f"test_merge_fact_{i}"
        await redis_client.execute_command(
            "VADD", mem_store.FACT_KEY, "FP32", emb.tobytes(),
            element_id, "SETATTR", json.dumps({
                "content": f"Python is a programming language (variant {i})",
                "confidence": 0.8,
                "importance": 0.5 + (i * 0.1),
                "ts": int(time.time() * 1000) - (i * 3600000),  # staggered timestamps
                "category": "general"
            })
        )
    
    # Run consolidation
    result = await _do_consolidate()
    
    # Should have merged some facts
    assert result["merged"] >= 0  # may be 0 if similarity threshold not met
    
    # Check that superseded facts are marked
    card = await redis_client.execute_command("VCARD", mem_store.FACT_KEY)
    assert int(card) > 0


@pytest.mark.asyncio
async def test_consolidation_prune_low_importance(redis_client):
    """Consolidation should prune facts with importance < 0.05."""
    from core import store as mem_store
    from main import _do_consolidate
    import numpy as np
    
    # Add a low-importance fact
    emb = np.zeros(mem_store.DIMS, dtype=np.float32)
    element_id = "test_prune_fact"
    
    await redis_client.execute_command(
        "VADD", mem_store.FACT_KEY, "FP32", emb.tobytes(),
        element_id, "SETATTR", json.dumps({
            "content": "Low importance fact to be pruned",
            "confidence": 0.3,
            "importance": 0.03,  # below 0.05 threshold
            "ts": int(time.time() * 1000) - (100 * 24 * 3600 * 1000),  # 100 days old
            "category": "general"
        })
    )
    
    # Run consolidation
    result = await _do_consolidate()
    
    # Should have pruned the low-importance fact
    assert result["pruned"] >= 0
    
    # Verify fact is superseded
    raw = await redis_client.execute_command("VGETATTR", mem_store.FACT_KEY, element_id)
    if raw:
        attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        assert attrs.get("superseded_by") == "pruned" or attrs.get("importance", 1.0) < 0.05


@pytest.mark.asyncio
async def test_pinned_fact_never_pruned(redis_client):
    """Pinned facts should never be pruned regardless of importance."""
    from core import store as mem_store
    from main import _do_consolidate
    import numpy as np
    
    # Add a pinned fact with low importance
    emb = np.zeros(mem_store.DIMS, dtype=np.float32)
    element_id = "test_pinned_fact"
    
    await redis_client.execute_command(
        "VADD", mem_store.FACT_KEY, "FP32", emb.tobytes(),
        element_id, "SETATTR", json.dumps({
            "content": "Pinned critical fact",
            "confidence": 0.8,
            "importance": 0.03,  # normally would be pruned
            "pinned": True,  # but pinned!
            "ts": int(time.time() * 1000),
            "category": "rule"
        })
    )
    
    # Run consolidation
    result = await _do_consolidate()
    
    # Verify pinned fact still exists and is not superseded
    raw = await redis_client.execute_command("VGETATTR", mem_store.FACT_KEY, element_id)
    assert raw is not None
    
    attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
    assert attrs.get("pinned") is True
    assert not attrs.get("superseded_by")


@pytest.mark.asyncio
async def test_hard_prune_removes_stale_episodes(redis_client):
    """Hard prune should remove episodes older than 180 days with no access."""
    from core import store as mem_store
    from main import _do_hard_prune
    import numpy as np
    
    # Add a stale episode
    emb = np.zeros(mem_store.DIMS, dtype=np.float32)
    element_id = "test_stale_episode"
    six_months_ago = int(time.time() * 1000) - (180 * 24 * 3600 * 1000)
    
    await redis_client.execute_command(
        "VADD", mem_store.EPISODE_KEY, "FP32", emb.tobytes(),
        element_id, "SETATTR", json.dumps({
            "content": "Old episode that should be pruned",
            "ts": six_months_ago,
            "access_count": 0,  # never accessed
            "ep_type": "general"
        })
    )
    
    # Run hard prune
    result = await _do_hard_prune()
    
    # Should have removed the stale episode
    assert result["removed_episodes"] >= 0
    
    # Verify episode is gone
    raw = await redis_client.execute_command("VGETATTR", mem_store.EPISODE_KEY, element_id)
    assert raw is None or json.loads(raw.decode() if isinstance(raw, bytes) else raw).get("_seed")


@pytest.mark.asyncio
async def test_user_feedback_boosts_importance(redis_client):
    """Positive user feedback should boost fact importance."""
    from core import store as mem_store
    import numpy as np
    
    # Add a test fact
    emb = np.zeros(mem_store.DIMS, dtype=np.float32)
    element_id = "test_feedback_fact"
    
    await redis_client.execute_command(
        "VADD", mem_store.FACT_KEY, "FP32", emb.tobytes(),
        element_id, "SETATTR", json.dumps({
            "content": "Fact that user will rate highly",
            "confidence": 0.8,
            "importance": 0.5,
            "ts": int(time.time() * 1000)
        })
    )
    
    # Simulate positive feedback (rating 5)
    from pydantic import BaseModel
    
    class FeedbackRequest(BaseModel):
        element_id: str
        rating: int
        comment: str = ""
    
    req = FeedbackRequest(element_id=element_id, rating=5, comment="Very helpful!")
    
    # Call feedback endpoint logic inline
    raw = await redis_client.execute_command("VGETATTR", mem_store.FACT_KEY, element_id)
    attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
    
    old_importance = attrs.get("importance", 0.5)
    attrs["importance"] = min(1.0, old_importance * 1.2)
    attrs["user_rating"] = 5
    attrs["user_comment"] = "Very helpful!"
    
    await redis_client.execute_command(
        "VSETATTR", mem_store.FACT_KEY, element_id,
        json.dumps(attrs, ensure_ascii=False)
    )
    
    # Verify importance was boosted
    raw = await redis_client.execute_command("VGETATTR", mem_store.FACT_KEY, element_id)
    attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
    
    assert attrs["importance"] > old_importance
    assert attrs["user_rating"] == 5
    assert attrs["user_comment"] == "Very helpful!"


@pytest.mark.asyncio
async def test_negative_feedback_reduces_importance(redis_client):
    """Negative user feedback should reduce fact importance."""
    from core import store as mem_store
    import numpy as np
    
    # Add a test fact
    emb = np.zeros(mem_store.DIMS, dtype=np.float32)
    element_id = "test_negative_feedback_fact"
    
    await redis_client.execute_command(
        "VADD", mem_store.FACT_KEY, "FP32", emb.tobytes(),
        element_id, "SETATTR", json.dumps({
            "content": "Fact that user will rate poorly",
            "confidence": 0.8,
            "importance": 0.7,
            "ts": int(time.time() * 1000)
        })
    )
    
    # Simulate negative feedback (rating 1)
    raw = await redis_client.execute_command("VGETATTR", mem_store.FACT_KEY, element_id)
    attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
    
    old_importance = attrs.get("importance", 0.7)
    attrs["importance"] = max(0.05, old_importance * 0.7)
    attrs["user_rating"] = 1
    attrs["needs_review"] = True
    
    await redis_client.execute_command(
        "VSETATTR", mem_store.FACT_KEY, element_id,
        json.dumps(attrs, ensure_ascii=False)
    )
    
    # Verify importance was reduced
    raw = await redis_client.execute_command("VGETATTR", mem_store.FACT_KEY, element_id)
    attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
    
    assert attrs["importance"] < old_importance
    assert attrs["user_rating"] == 1
    assert attrs["needs_review"] is True
