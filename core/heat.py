"""
Heat scoring: recency × access frequency re-ranking.

Inspired by MemoryOS "heat-driven promotion":
  heat = frequency_boost × recency_decay
  final_score = cosine_similarity × heat

Stored per-element in Redis VSETATTR JSON:
  { "access_count": int, "last_accessed": unix_ms, ... }
"""

import asyncio
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
    access_count: int = attrs.get("access_count") or 0
    ts_ms: int = attrs.get("ts") or int(time.time() * 1000)

    days_old = (time.time() * 1000 - ts_ms) / (1000 * 86400)
    recency_decay = math.exp(-days_old / 30)          # 30-day half-life
    frequency_boost = 1.0 + math.log1p(access_count) * 0.25

    return max(0.01, recency_decay * frequency_boost)


def heat_rerank(items: list[dict]) -> list[dict]:
    """Re-rank a list of {content, score, attrs} dicts by cosine × heat."""
    for item in items:
        # Redis VSIM returns cosine SIMILARITY (1=identical, 0=opposite).
        sim = max(0.0, float(item.get("score", 0)))
        h = compute_heat(item.get("attrs", {}))
        item["heat_score"] = sim * h
    return sorted(items, key=lambda x: x["heat_score"], reverse=True)


# ── Atomic heat bump via Lua script ───────────────────────────────────────────
# Race condition fix: the old Python read-modify-write pattern lost increments
# under concurrent access (two coroutines both read access_count=N, both write
# N+1 instead of N+2). This Lua script does the read+increment+write atomically
# inside Redis, which is single-threaded for Lua execution.

_BUMP_HEAT_LUA = """
local key = KEYS[1]
local elem = ARGV[1]
local now_ms = ARGV[2]
local count_field = ARGV[3] or 'access_count'
local time_field = ARGV[4] or 'last_accessed'

local raw = redis.call('VGETATTR', key, elem)
if not raw then return 0 end

local ok, attrs = pcall(cjson.decode, raw)
if not ok or type(attrs) ~= 'table' then return 0 end

attrs[count_field] = (attrs[count_field] or 0) + 1
attrs[time_field] = tonumber(now_ms)
-- Reinforce Ebbinghaus confidence: recall counts as confirmation
attrs['last_confirmed_ts'] = tonumber(now_ms)

local new_json = cjson.encode(attrs)
redis.call('VSETATTR', key, elem, new_json)
return 1
"""

_bump_script = None  # cached Script object
_bump_script_lock = asyncio.Lock()  # protects _bump_script initialization


async def atomic_bump(
    r: aioredis.Redis,
    vset_key: str,
    element: str,
    count_field: str = "access_count",
    time_field: str = "last_accessed",
) -> None:
    """Atomically increment a counter field and timestamp in a vectorset element's attrs.

    Uses a Redis Lua script to avoid the read-modify-write race condition.
    """
    global _bump_script
    now_ms = int(time.time() * 1000)
    try:
        if _bump_script is None:
            async with _bump_script_lock:
                if _bump_script is None:
                    _bump_script = r.register_script(_BUMP_HEAT_LUA)
        await _bump_script(
            keys=[vset_key],
            args=[element, str(now_ms), count_field, time_field],
        )
    except Exception:
        pass


async def bump_heat(r: aioredis.Redis, vset_key: str, element: str, attrs: dict) -> None:
    """Atomically increment access_count and update last_accessed.

    Uses atomic_bump (Redis Lua script) to avoid the read-modify-write race.
    The `attrs` parameter is kept for API compatibility but is no longer used
    for the write — the Lua script reads fresh attrs from Redis.
    """
    await atomic_bump(r, vset_key, element, "access_count", "last_accessed")
