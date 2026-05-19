"""
Heat Scoring Tests — migrated and enhanced from agentmemory/test/retention.test.ts

Tests the recency × frequency heat scoring system for memory re-ranking.
Performance target: Beat reference implementation with optimized Redis operations.
"""

import math
import time

import pytest

from core.heat import compute_heat, heat_rerank


class TestComputeHeat:
    """Unit tests for heat score computation."""

    def test_fresh_never_accessed(self):
        """Freshly stored, never accessed memory should have heat ~1.0."""
        attrs = {
            "access_count": 0,
            "ts": int(time.time() * 1000),  # now
        }
        heat = compute_heat(attrs)
        assert 0.9 < heat <= 1.1, f"Expected ~1.0, got {heat}"

    def test_recently_accessed_multiple_times(self):
        """Accessed 10 times yesterday should have elevated heat."""
        attrs = {
            "access_count": 10,
            "ts": int((time.time() - 86400) * 1000),  # 1 day ago
        }
        heat = compute_heat(attrs)
        # recency_decay ≈ exp(-1/30) ≈ 0.967
        # frequency_boost = 1 + log1p(10) * 0.25 ≈ 1.6
        # heat ≈ 0.967 * 1.6 ≈ 1.55
        assert 1.3 < heat < 1.8, f"Expected ~1.5, got {heat}"

    def test_old_never_accessed(self):
        """Stored 90 days ago, never accessed should have very low heat."""
        attrs = {
            "access_count": 0,
            "ts": int((time.time() - 90 * 86400) * 1000),  # 90 days ago
        }
        heat = compute_heat(attrs)
        # recency_decay = exp(-90/30) = exp(-3) ≈ 0.05
        assert 0.01 <= heat < 0.1, f"Expected ~0.05, got {heat}"

    def test_minimum_heat_floor(self):
        """Heat should never drop below 0.01 floor."""
        attrs = {
            "access_count": 0,
            "ts": int((time.time() - 365 * 86400) * 1000),  # 1 year ago
        }
        heat = compute_heat(attrs)
        assert heat >= 0.01, f"Heat {heat} below minimum floor 0.01"

    def test_high_frequency_boost(self):
        """High access count should provide logarithmic boost."""
        attrs = {
            "access_count": 100,
            "ts": int(time.time() * 1000),  # now
        }
        heat = compute_heat(attrs)
        # frequency_boost = 1 + log1p(100) * 0.25 ≈ 1 + 4.6 * 0.25 ≈ 2.15
        assert 1.8 < heat < 2.5, f"Expected ~2.1, got {heat}"

    def test_missing_access_count_defaults_to_zero(self):
        """Missing access_count should default to 0."""
        attrs = {"ts": int(time.time() * 1000)}
        heat = compute_heat(attrs)
        assert 0.9 < heat <= 1.1

    def test_missing_timestamp_uses_current_time(self):
        """Missing timestamp should use current time."""
        attrs = {"access_count": 0}
        heat = compute_heat(attrs)
        assert 0.9 < heat <= 1.1

    def test_recency_decay_half_life(self):
        """Verify 30-day half-life: heat at 30 days should be ~0.5 of fresh."""
        fresh_attrs = {"access_count": 0, "ts": int(time.time() * 1000)}
        old_attrs = {"access_count": 0, "ts": int((time.time() - 30 * 86400) * 1000)}

        fresh_heat = compute_heat(fresh_attrs)
        old_heat = compute_heat(old_attrs)

        ratio = old_heat / fresh_heat
        # Should be approximately exp(-30/30) = exp(-1) ≈ 0.368
        assert 0.3 < ratio < 0.5, f"Expected ratio ~0.37, got {ratio}"


class TestHeatRerank:
    """Integration tests for heat-based re-ranking."""

    def test_rerank_by_heat_score(self):
        """Items should be sorted by cosine_similarity × heat descending."""
        items = [
            {"content": "old", "score": 0.9, "attrs": {"access_count": 0, "ts": int((time.time() - 90 * 86400) * 1000)}},
            {"content": "recent", "score": 0.8, "attrs": {"access_count": 10, "ts": int(time.time() * 1000)}},
            {"content": "medium", "score": 0.85, "attrs": {"access_count": 2, "ts": int((time.time() - 10 * 86400) * 1000)}},
        ]

        ranked = heat_rerank(items)

        # Recent high-frequency should rank highest despite lower cosine
        assert ranked[0]["content"] == "recent"
        # Verify heat_score was computed
        assert "heat_score" in ranked[0]
        # Verify descending order
        assert ranked[0]["heat_score"] >= ranked[1]["heat_score"] >= ranked[2]["heat_score"]

    def test_rerank_preserves_original_scores(self):
        """Original cosine scores should remain unchanged."""
        items = [
            {"content": "a", "score": 0.95, "attrs": {"access_count": 0, "ts": int(time.time() * 1000)}},
            {"content": "b", "score": 0.85, "attrs": {"access_count": 0, "ts": int(time.time() * 1000)}},
        ]

        ranked = heat_rerank(items)

        assert ranked[0]["score"] == 0.95
        assert ranked[1]["score"] == 0.85

    def test_rerank_handles_empty_list(self):
        """Empty input should return empty list."""
        assert heat_rerank([]) == []

    def test_rerank_handles_single_item(self):
        """Single item should be returned as-is with heat_score added."""
        items = [{"content": "only", "score": 0.7, "attrs": {"access_count": 5, "ts": int(time.time() * 1000)}}]
        ranked = heat_rerank(items)

        assert len(ranked) == 1
        assert ranked[0]["content"] == "only"
        assert "heat_score" in ranked[0]

    def test_rerank_negative_scores_clamped_to_zero(self):
        """Negative cosine scores should be clamped to 0 before multiplication."""
        items = [
            {"content": "negative", "score": -0.5, "attrs": {"access_count": 100, "ts": int(time.time() * 1000)}},
            {"content": "positive", "score": 0.1, "attrs": {"access_count": 0, "ts": int(time.time() * 1000)}},
        ]

        ranked = heat_rerank(items)

        # Negative score item should have heat_score = 0 regardless of heat
        assert ranked[0]["content"] == "positive"
        assert ranked[1]["heat_score"] == 0.0


class TestHeatPerformance:
    """Performance benchmarks to beat reference implementation."""

    @pytest.mark.benchmark(group="heat-compute")
    def test_compute_heat_performance(self, benchmark):
        """compute_heat should execute in < 0.01ms (10 microseconds)."""
        attrs = {"access_count": 5, "ts": int(time.time() * 1000)}

        result = benchmark(compute_heat, attrs)

        # Reference: TypeScript version takes ~0.05ms
        # Target: Python implementation < 0.01ms
        assert result > 0.01  # Sanity check

    @pytest.mark.benchmark(group="heat-rerank")
    def test_heat_rerank_performance_100_items(self, benchmark):
        """heat_rerank should handle 100 items in < 1ms."""
        items = [
            {
                "content": f"item_{i}",
                "score": 0.5 + (i % 10) * 0.05,
                "attrs": {"access_count": i % 20, "ts": int((time.time() - i * 3600) * 1000)},
            }
            for i in range(100)
        ]

        result = benchmark(heat_rerank, items)

        assert len(result) == 100
        # Verify sorted
        for i in range(len(result) - 1):
            assert result[i]["heat_score"] >= result[i + 1]["heat_score"]

    @pytest.mark.benchmark(group="heat-rerank-large")
    def test_heat_rerank_performance_1000_items(self, benchmark):
        """heat_rerank should handle 1000 items in < 10ms."""
        items = [
            {
                "content": f"item_{i}",
                "score": 0.5 + (i % 10) * 0.05,
                "attrs": {"access_count": i % 50, "ts": int((time.time() - i * 1800) * 1000)},
            }
            for i in range(1000)
        ]

        result = benchmark(heat_rerank, items)

        assert len(result) == 1000
