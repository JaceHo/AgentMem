"""
Heat scoring: recency × access frequency re-ranking.

Inspired by MemoryOS "heat-driven promotion":
  heat = frequency_boost × recency_decay
  final_score = cosine_similarity × heat

Stored per-element in Redis VSETATTR JSON:
  { "access_count": int, "last_accessed": unix_ms, ... }
"""

import asyncio
import hashlib
import logging
import math
import time

import redis.asyncio as aioredis

log = logging.getLogger("mem")


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
#
# ── Why the bump path is now LOUD ────────────────────────────────────────────
# For the whole of v1.0.x every heat bump was a silent no-op: VSIM element names
# came back as bytes, ``force_str`` rendered them as ``b'01M3…'``, and the Lua
# script's VGETATTR on that nonexistent element returned nil → 0. Nothing was
# logged, so 1,362 real /recall calls produced access_count=0 on 1,149/1,149
# facts and A-MEM evolution never ran. A no-op is now an *observed* outcome:
# every bump is counted (ok / missing / error) and exported through /stats as
# ``heat``, and any miss or error is logged at WARNING.

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

_bump_script_lock = asyncio.Lock()  # serializes per-client Script registration

# ── Bump observability counters (process lifetime, reset on restart) ──────────
# Exported by GET /stats as ``heat``. ``missing`` is the number of bumps that
# found no such element in the vectorset — exactly the failure mode that went
# unnoticed the first time. A non-zero ``missing`` with recent ``attempts`` means
# element names are being mangled again somewhere upstream.
_attempts = 0
_ok = 0
_missing = 0
_errors = 0
_counts_lock = asyncio.Lock()


async def _count(kind: str, n: int = 1) -> None:
    """Bump one observability counter. Never raises."""
    global _attempts, _ok, _missing, _errors
    async with _counts_lock:
        if kind == "attempt":
            _attempts += n
        elif kind == "ok":
            _ok += n
        elif kind == "missing":
            _missing += n
        elif kind == "error":
            _errors += n
        else:
            return


def stats_snapshot() -> dict:
    """Current bump counters, for /stats. Cheap and synchronous."""
    return {
        "attempts": _attempts,
        "ok": _ok,
        "missing": _missing,
        "errors": _errors,
    }


async def reset_stats() -> None:
    """Zero the bump counters (used by tests)."""
    global _attempts, _ok, _missing, _errors
    async with _counts_lock:
        _attempts = _ok = _missing = _errors = 0


def _classify(reply: object) -> str:
    """Map one Lua reply to 'ok' | 'missing'. Anything unexpected counts missing.

    redis-py hands EVAL replies back as int, bytes or str depending on the
    client's decode_responses; accept every shape rather than trusting one.
    """
    if isinstance(reply, bytes):
        reply = reply.decode("utf-8", "replace")
    try:
        return "ok" if int(reply) == 1 else "missing"
    except (TypeError, ValueError):
        return "missing"


async def atomic_bump(
    r: aioredis.Redis,
    vset_key: str,
    element: str,
    count_field: str = "access_count",
    time_field: str = "last_accessed",
) -> bool:
    """Atomically increment a counter field and timestamp in a vectorset element's attrs.

    Uses a Redis Lua script to avoid the read-modify-write race condition.
    Returns True only when the script actually found the element and wrote the
    attrs back. Never raises: a bump is fire-and-forget by design, but a failure
    is now logged at WARNING and counted, so a silent no-op cannot recur.
    """
    now_ms = int(time.time() * 1000)
    await _count("attempt")
    try:
        # Cached on the CLIENT, never in a module global: a Script object is bound to the pool it
        # was registered from, so a global hands the next caller someone else's connection.
        script = getattr(r, "_heat_bump_script", None)
        if script is None:
            async with _bump_script_lock:
                script = getattr(r, "_heat_bump_script", None)
                if script is None:
                    script = r.register_script(_BUMP_HEAT_LUA)
                    setattr(r, "_heat_bump_script", script)
        result = await script(
            keys=[vset_key],
            args=[element, str(now_ms), count_field, time_field],
        )
    except Exception as e:
        await _count("error")
        log.warning("heat bump ERROR on %s element=%r: %s", vset_key, element, e)
        return False
    kind = _classify(result)
    await _count(kind)
    if kind == "missing":
        # The element does not exist in this vectorset. Almost always means the
        # caller passed a mangled name (bytes repr, wrong vset) rather than real
        # data loss — but it must never be silent either way.
        log.warning(
            "heat bump NO-OP: element %r not found in %s (field=%s) — "
            "access_count will not move", element, vset_key, count_field,
        )
        return False
    return True


# SHA-1 of the script body is exactly what Redis computes for SCRIPT LOAD, so we
# can address it directly from a pipeline without going through the Script
# wrapper (which only exposes a single-call API).
_BUMP_HEAT_SHA = hashlib.sha1(_BUMP_HEAT_LUA.encode("utf-8")).hexdigest()

# Public aliases — the startup path pre-loads the script and logs its sha, so a
# flushed script cache (the other way bumps can silently stop) is visible.
BUMP_HEAT_LUA_SRC = _BUMP_HEAT_LUA


def bump_script_sha() -> str:
    """SHA-1 Redis assigns to the heat-bump script."""
    return _BUMP_HEAT_SHA


async def atomic_bump_many(
    r: aioredis.Redis,
    vset_key: str,
    elements: list[str],
    count_field: str = "access_count",
    time_field: str = "last_accessed",
) -> int:
    """Bump heat for many elements in one pipeline round trip.

    The per-element path awaits ``atomic_bump`` N times, i.e. N sequential Lua
    round trips for a recall that hit N items.  Pipelining sends them together;
    Redis still executes the Lua calls one at a time server-side, so the
    read-modify-write atomicity is unchanged — the N-1 network round trips just
    disappear.

    Returns the number of elements actually bumped.  Replies are inspected (they
    are ``1``/``0`` from the Lua script), so a batch that silently found nothing
    is reported at WARNING instead of looking like success.
    """
    if not elements:
        return 0
    now_ms = int(time.time() * 1000)
    await _count("attempt", len(elements))
    try:
        pipe = r.pipeline(transaction=False)
        for el in elements:
            pipe.execute_command(
                "EVALSHA", _BUMP_HEAT_SHA, 1, vset_key,
                el, str(now_ms), count_field, time_field,
            )
        replies = await pipe.execute(raise_on_error=False)
    except Exception as e:
        # NOSCRIPT means the cache was flushed — fall back to per-element calls
        # so a cold script cache degrades to the old behaviour instead of
        # silently dropping every heat bump.
        log.warning("atomic_bump_many pipeline failed (%s); falling back to %d "
                    "per-element bumps on %s", e, len(elements), vset_key)
        bumped = 0
        for el in elements:
            if await atomic_bump(r, vset_key, el, count_field, time_field):
                bumped += 1
        return bumped

    # Per-command pipeline failures come back as exception objects in the reply
    # list (raise_on_error=False); NOSCRIPT there must also trigger the fallback.
    n_exc = sum(1 for x in replies if isinstance(x, Exception))
    if n_exc:
        first_err = next((x for x in replies if isinstance(x, Exception)), None)
        log.warning("atomic_bump_many: %d/%d pipelined EVALSHA errors on %s (%s); "
                    "retrying per element", n_exc, len(elements), vset_key, first_err)
        await _count("error", n_exc)
        bumped = 0
        for el in elements:
            if await atomic_bump(r, vset_key, el, count_field, time_field):
                bumped += 1
        return bumped

    kinds = [_classify(rep) for rep in replies]
    ok = kinds.count("ok")
    missing = kinds.count("missing")
    await _count("ok", ok)
    if missing:
        await _count("missing", missing)
        log.warning(
            "heat bump NO-OP: %d/%d elements not found in %s (e.g. %r) — "
            "access_count will not move for them",
            missing, len(elements), vset_key, elements[kinds.index("missing")],
        )
    if ok == 0:
        log.warning("heat bump batch did nothing: 0/%d elements bumped on %s",
                    len(elements), vset_key)
    return ok


async def bump_heat(r: aioredis.Redis, vset_key: str, element: str, attrs: dict) -> bool:
    """Atomically increment access_count and update last_accessed.

    Uses atomic_bump (Redis Lua script) to avoid the read-modify-write race.
    The `attrs` parameter is kept for API compatibility but is no longer used
    for the write — the Lua script reads fresh attrs from Redis.
    """
    return await atomic_bump(r, vset_key, element, "access_count", "last_accessed")
