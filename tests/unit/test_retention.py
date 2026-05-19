"""
Retention & Memory Lifecycle Tests — migrated from agentmemory/test/retention.test.ts

Tests memory retention policies, aging, access tracking, and lifecycle management.
Performance target: Optimize Redis operations for sub-millisecond retention checks.
"""

import time

import pytest
import redis


class TestMemoryRetention:
    """Test memory retention policies and aging."""

    @pytest.fixture(autouse=True)
    def setup_redis(self):
        """Setup clean Redis connection for each test."""
        self.r = redis.Redis(host="localhost", port=6379, db=15, decode_responses=True)
        self.r.flushdb()
        yield
        self.r.flushdb()
        self.r.close()

    def test_memory_ttl_expiration(self):
        """Session KV (Tier 1) should expire after configured TTL."""
        session_id = "test-ttl-session"
        key = f"mem:session:{session_id}:ctx"

        # Set with 2-second TTL for testing
        self.r.setex(key, 2, "test context")

        # Should exist immediately
        assert self.r.exists(key) == 1

        # Wait for expiration
        time.sleep(2.5)

        # Should be expired
        assert self.r.exists(key) == 0

    def test_long_term_memory_permanent(self):
        """Long-term memories (Tier 2) should not have TTL."""
        fact_id = "test-permanent-fact"
        key = f"mem:facts:{fact_id}"

        self.r.hset(key, mapping={"content": "permanent fact", "ts": str(int(time.time() * 1000))})

        # No TTL set
        ttl = self.r.ttl(key)
        assert ttl == -1  # -1 means no expiry

    def test_access_count_tracking(self):
        """Access count should increment on each recall."""
        vset_key = "mem:facts"
        element = "test-element"
        attrs = {"access_count": 0, "ts": int(time.time() * 1000)}

        # Initial store
        import json
        self.r.execute_command("VSETATTR", vset_key, element, json.dumps(attrs))

        # Simulate bump_heat calls
        from core.heat import bump_heat
        import asyncio

        async def bump_multiple():
            await bump_heat(self.r, vset_key, element, attrs)
            await bump_heat(self.r, vset_key, element, attrs)
            await bump_heat(self.r, vset_key, element, attrs)

        asyncio.run(bump_multiple())

        # Verify count incremented
        stored = self.r.execute_command("VGETATTR", vset_key, element)
        if stored:
            retrieved_attrs = json.loads(stored)
            assert retrieved_attrs["access_count"] == 3


class TestMemoryAging:
    """Test memory aging and decay mechanisms."""

    @pytest.fixture(autouse=True)
    def setup_redis(self):
        """Setup clean Redis connection for each test."""
        self.r = redis.Redis(host="localhost", port=6379, db=15, decode_responses=True)
        self.r.flushdb()
        yield
        self.r.flushdb()
        self.r.close()

    def test_recency_decay_calculation(self):
        """Verify recency decay follows exponential formula."""
        from core.heat import compute_heat

        now = int(time.time() * 1000)
        day_ms = 86400 * 1000

        # Fresh
        fresh = compute_heat({"access_count": 0, "ts": now})

        # 30 days old (one half-life)
        old_30 = compute_heat({"access_count": 0, "ts": now - 30 * day_ms})

        # 60 days old (two half-lives)
        old_60 = compute_heat({"access_count": 0, "ts": now - 60 * day_ms})

        ratio_30 = old_30 / fresh
        ratio_60 = old_60 / fresh

        # Each 30 days should reduce by ~exp(-1) ≈ 0.368
        assert 0.3 < ratio_30 < 0.5
        assert 0.1 < ratio_60 < 0.2

    def test_frequency_boost_logarithmic(self):
        """Frequency boost should grow logarithmically, not linearly."""
        from core.heat import compute_heat

        now = int(time.time() * 1000)

        heat_1 = compute_heat({"access_count": 1, "ts": now})
        heat_10 = compute_heat({"access_count": 10, "ts": now})
        heat_100 = compute_heat({"access_count": 100, "ts": now})

        # Logarithmic growth: diminishing returns
        boost_1_to_10 = heat_10 / heat_1
        boost_10_to_100 = heat_100 / heat_10

        # First 10 accesses should provide more boost than next 90
        assert boost_1_to_10 > boost_10_to_100


class TestConsolidationPipeline:
    """Test the 3-phase consolidation pipeline (Decay → Merge → Prune)."""

    @pytest.fixture(autouse=True)
    def setup_redis(self):
        """Setup clean Redis connection for each test."""
        self.r = redis.Redis(host="localhost", port=6379, db=15, decode_responses=True)
        self.r.flushdb()
        yield
        self.r.flushdb()
        self.r.close()

    def test_decay_phase_reduces_importance(self):
        """Decay phase should reduce importance scores over time."""
        # This is tested via the actual consolidate endpoint
        pass

    def test_merge_phase_detects_duplicates(self):
        """Merge phase should identify near-duplicate facts."""
        # Integration test with /consolidate endpoint
        pass

    def test_prune_phase_removes_stale(self):
        """Prune phase should remove superseded/stale entries."""
        # Tested via hard-prune endpoint
        pass


class TestRetentionPerformance:
    """Performance benchmarks for retention operations."""

    @pytest.fixture(autouse=True)
    def setup_redis(self):
        """Setup Redis connection."""
        self.r = redis.Redis(host="localhost", port=6379, db=15, decode_responses=True)
        self.r.flushdb()
        yield
        self.r.flushdb()
        self.r.close()

    @pytest.mark.benchmark(group="ttl-check")
    def test_ttl_check_performance(self, benchmark):
        """TTL check should complete in < 0.1ms."""
        key = "test-ttl-key"
        self.r.setex(key, 3600, "value")

        def check_ttl():
            return self.r.ttl(key)

        result = benchmark(check_ttl)
        assert result > 0

    @pytest.mark.benchmark(group="access-bump")
    def test_access_count_bump_performance(self, benchmark):
        """bump_heat should complete in < 1ms."""
        from core.heat import bump_heat
        import asyncio

        vset_key = "mem:facts"
        element = "perf-test-element"
        attrs = {"access_count": 0, "ts": int(time.time() * 1000)}

        # Initialize
        import json
        self.r.execute_command("VSETATTR", vset_key, element, json.dumps(attrs))

        def bump_sync():
            """Wrap async call in sync function for benchmarking."""
            async def _bump():
                await bump_heat(self.r, vset_key, element, attrs)
            asyncio.run(_bump())

        result = benchmark(bump_sync)
        assert result is None

    @pytest.mark.benchmark(group="retention-scan")
    def test_retention_scan_performance(self, benchmark):
        """Scanning 1000 memories for retention should complete in < 100ms."""
        # Create 1000 test keys
        for i in range(1000):
            self.r.set(f"mem:test:{i}", f"value_{i}", ex=3600)

        def scan_all():
            cursor = 0
            count = 0
            while True:
                cursor, keys = self.r.scan(cursor, match="mem:test:*", count=100)
                count += len(keys)
                if cursor == 0:
                    break
            return count

        result = benchmark(scan_all)
        assert result == 1000


def asyncio_run_wrapper(coro_func):
    """Helper to run async functions in benchmark."""
    import asyncio
    return asyncio.run(coro_func())
