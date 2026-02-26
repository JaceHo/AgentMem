"""
Heat scoring: recency × access frequency re-ranking.

Inspired by MemoryOS "heat-driven promotion":
  heat = frequency_boost × recency_decay
  final_score = cosine_similarity × heat

Stored per-element in Redis VSETATTR JSON:
  { "access_count": int, "last_accessed": unix_ms, ... }
"""

import json
import math
import time

import redis.asyncio as aioredis


def compute_heat(attrs: dict) -> float:
    """
    Returns a multiplier > 0.
    - Freshly stored, never accessed: ~1.0
    - Accessed 10 times, stored yesterday: ~1.5
    - Stored 90 days ago, never accessed: ~0.05
    """
    access_count: int = attrs.get("access_count", 0)
    ts_ms: int = attrs.get("ts", int(time.time() * 1000))

    days_old = (time.time() * 1000 - ts_ms) / (1000 * 86400)
    recency_decay = math.exp(-days_old / 30)          # 30-day half-life
    frequency_boost = 1.0 + math.log1p(access_count) * 0.25

    return max(0.01, recency_decay * frequency_boost)


def heat_rerank(items: list[dict]) -> list[dict]:
    """Re-rank a list of {content, score, attrs} dicts by cosine × heat."""
    for item in items:
        raw_score = float(item.get("score", 0))
        # Redis VSIM returns cosine DISTANCE (0=identical, 2=opposite).
        # Convert to similarity: sim = 1 - dist/2
        sim = max(0.0, 1.0 - raw_score / 2.0)
        h = compute_heat(item.get("attrs", {}))
        item["heat_score"] = sim * h
    return sorted(items, key=lambda x: x["heat_score"], reverse=True)


async def bump_heat(r: aioredis.Redis, vset_key: str, element: str, attrs: dict) -> None:
    """Increment access_count and update last_accessed (fire-and-forget ok)."""
    attrs = dict(attrs)  # copy
    attrs["access_count"] = attrs.get("access_count", 0) + 1
    attrs["last_accessed"] = int(time.time() * 1000)
    try:
        await r.execute_command("VSETATTR", vset_key, element, json.dumps(attrs))
    except Exception:
        pass
