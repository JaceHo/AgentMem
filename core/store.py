"""
Redis 8 native vectorset store — v1.1 (LLM Wiki v2-aligned)

Two vector sets:
  mem:episodes  — episodic memory (conversation turns)
  mem:facts     — semantic memory (distilled lossless facts, SimpleMem Section 3.1)

Session working memory in plain Redis strings (TTL-based).
User persona in Redis Hash (mem:persona).

SimpleMem additions (v0.9.1):
  - importance field (float 0-1) on each fact: decays with age, used for pruning
  - superseded_by field: soft-delete — facts merged by consolidation are marked
    superseded_by = winning entry's element ID rather than hard-deleted
  - topic and location fields: SimpleMem symbolic layer (R_k metadata)
  - knn_search() filters out superseded entries by default

LLM Wiki v2 additions (v1.1):
  - source_count: how many independent sources support this fact (reinforcement)
  - last_confirmed_ts: timestamp of most recent reinforcement (Ebbinghaus decay)
  - superseded_at: timestamp when supersession occurred (version chain)
  - superseded_reason: "merged" | "contradicted" | "updated" | "pruned"
  - version: incrementing counter for each fact (tracks edits)
  - confidence_decay(): Ebbinghaus forgetting curve — confidence decays
    exponentially with time since last confirmation, resets on reinforcement

Redis 8 vectorset API summary:
  VADD  key FP32 <blob> <element> [SETATTR <json>]
  VSIM  key FP32 <blob> COUNT k WITHSCORES WITHATTRIBS [FILTER <expr>]
  VSETATTR key element <json>   — FILTER expr: .field == "value"
"""

import asyncio
import base64
import json
import logging
import math
import os
import re
import time
import unicodedata
from typing import Optional

import numpy as np
import redis.asyncio as aioredis
import ulid as _ulid_mod

from .utils import decode_bytes, decode_json
from .scope import Scope, scope_of

log = logging.getLogger(__name__)

REDIS_URL    = os.getenv("REDIS_URL", "redis://localhost:6379")
EPISODE_KEY  = "mem:episodes"
FACT_KEY     = "mem:facts"
TOOL_KEY        = "mem:tools"          # capability: tool/skill definitions
TOOL_GRAPH_KEY  = "mem:tool_graph"     # AutoTool TIG: tool transition counts hash (a:b → count)
PROC_KEY        = "mem:procedures"     # procedural: successful workflows/patterns (4th cognitive tier)
PROC_TASK_KEY   = "mem:proc_tasks"     # exact task → element id, so dedup does not depend on VSIM
PROC_BY_TOOL_PRE = "mem:proc_by_tool:" # reverse index: tool_name → Set of proc element IDs
ENV_KEY      = "mem:env"        # capability: environment state hash
AGENT_KEY    = "mem:agent"      # capability: agent self-model hash
SESSION_PRE  = "mem:session:"
PERSONA_KEY  = "mem:persona"

# ── Element-ID shadow index ──────────────────────────────────────────────────
#
# Redis vectorset has NO command to enumerate a set's elements: VSCAN does not
# exist, and VSIM is an approximate HNSW traversal that is not guaranteed to
# return COUNT results.  Measured on this instance (2026-10-22):
#
#   mem:facts       VCARD 1140   -> VSIM returns 1139   (99.9%)
#   mem:episodes    VCARD 1611   -> VSIM returns 1609   (99.8%)
#   mem:procedures  VCARD 13260  -> VSIM returns ~1730  (13.0%)
#                                       30-query union 6479  (48.9%)
#
# A freshly built 16k-element set still returns 98.3%, so this is not a size
# limit — mem:procedures' HNSW graph has degraded and over half its elements
# are unreachable by any similarity query.  vscan() therefore silently loses
# them, which quietly corrupts consolidation, procedure scans and BM25
# population.
#
# The fix is an explicit shadow SET of element IDs maintained at write time.
# SSCAN is exact and complete, and VGETATTR works by ID regardless of graph
# health, so enumeration through the shadow index stays correct even for a
# degraded vectorset.

IDX_PRE = "mem:idx:"           # mem:idx:<tier> -> Set of element IDs

# ── Cached raw embeddings ─────────────────────────────────────────────────────
#
# Redis vectorset has no command to read a stored vector back (no VGET), so the
# merge phase of consolidation used to re-embed the *entire* fact corpus every
# run — an hourly batch of N MiniLM/nomic API calls that times out on remote
# providers and burns tokens.  We keep the raw embedding alongside each element
# in a side hash, written once at save time.  The vector is int8-quantized and
# base64-encoded so it stays small (≈1 B/dim) and utf-8 safe; cosine is scale-
# invariant so the quantization does not change cluster membership at the 0.85
# affinity threshold.  Consolidation reads the whole tier in one HGETALL and only
# re-embeds the pre-cache legacy rows.

VEC_PRE = "mem:vec:"           # mem:vec:<tier> -> Hash element_id -> b64 int8 vector


def vec_key(vset: str) -> str:
    """Cached-vector hash key for a vectorset (mem:facts -> mem:vec:facts)."""
    return VEC_PRE + vset.split(":", 1)[-1]


def _quantize_vec(vec: np.ndarray) -> str:
    """fp32 embedding -> base64 int8 blob (≈1 byte/dim, utf-8 safe)."""
    q = np.clip(vec * 127.0, -127, 127).round().astype(np.int8)
    return base64.b64encode(q.tobytes()).decode("ascii")


def _dequantize_vec(blob: str) -> np.ndarray:
    """base64 int8 blob -> fp32 embedding (inverse of _quantize_vec)."""
    return np.frombuffer(base64.b64decode(blob), dtype=np.int8).astype(np.float32) / 127.0


def idx_key(vset: str) -> str:
    """Shadow-index key for a vectorset (mem:episodes -> mem:idx:episodes)."""
    return IDX_PRE + vset.split(":", 1)[-1]


async def register_element(r: aioredis.Redis, vset: str, element_id: str) -> None:
    """Record an element ID in the shadow index. Called on every VADD."""
    if not element_id:
        return
    try:
        await r.sadd(idx_key(vset), element_id)
    except Exception as e:  # pragma: no cover - shadow index is best-effort
        log.warning("register_element failed for %s/%s: %s", vset, element_id, e)


async def unregister_element(r: aioredis.Redis, vset: str, element_id: str) -> None:
    """Drop an element ID from the shadow index. Called on every VREM."""
    if not element_id:
        return
    try:
        await r.srem(idx_key(vset), element_id)
    except Exception as e:  # pragma: no cover
        log.warning("unregister_element failed for %s/%s: %s", vset, element_id, e)


async def remove_element(r: aioredis.Redis, vset: str, element_id: str) -> None:
    """VREM an element and drop it from the shadow index in one place.

    Every physical delete must go through here, otherwise the shadow index
    accumulates IDs for elements that no longer exist and enumeration returns
    phantom entries.

    For mem:procedures the element ID is *also* duplicated in the
    mem:proc_by_tool:<tool> reverse-index sets and the mem:proc_tasks hash,
    neither of which the vectorset knows about.  A bare VREM therefore leaves
    dangling references behind — 30,977 of them had piled up on this instance,
    every one costing a VGETATTR round trip in GET /tool-procedures/<tool>
    before being discarded.  So procedure deletes clean the side indexes too.

    The cached raw vector (mem:vec:<tier>) is dropped as well so the cache does
    not leak memory for deleted elements.
    """
    if not element_id:
        return
    if vset == PROC_KEY:
        # Read attrs BEFORE the VREM so we know which side keys reference it.
        try:
            raw = await r.execute_command("VGETATTR", vset, element_id)
            attrs = decode_json(raw) or {} if raw else {}
        except Exception:
            attrs = {}
        await r.execute_command("VREM", vset, element_id)
        await unregister_element(r, vset, element_id)
        await _unlink_proc_side_indexes(r, element_id, attrs)
        await _drop_vec_cache(r, vset, element_id)
        return
    await r.execute_command("VREM", vset, element_id)
    await unregister_element(r, vset, element_id)
    await _drop_vec_cache(r, vset, element_id)


async def _drop_vec_cache(r: aioredis.Redis, vset: str, element_id: str) -> None:
    """Drop a deleted element's cached vector (best-effort)."""
    try:
        await r.hdel(vec_key(vset), element_id)
    except Exception as e:  # pragma: no cover
        log.warning("drop vec cache failed for %s/%s: %s", vset, element_id, e)


async def get_cached_vectors(
    r: aioredis.Redis,
    vset: str,
    element_ids: list[str],
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Read cached embeddings for a tier.

    Returns ``(hits, misses)`` where ``hits`` maps element_id -> fp32 vector
    for every ID present in the cache, and ``misses`` is the subset of IDs the
    caller must still re-embed.  One HGETALL against the tier's vector hash.
    """
    if not element_ids:
        return {}, []
    try:
        raw = await r.hgetall(vec_key(vset))
    except Exception as e:
        log.warning("get_cached_vectors HGETALL failed: %s", e)
        return {}, list(element_ids)
    id_set = set(element_ids)
    hits: dict[str, np.ndarray] = {}
    # hgetall returns bytes/str depending on the client; normalise keys.
    for k, blob in (raw or {}).items():
        key = decode_bytes(k)
        if key in id_set:
            try:
                hits[key] = _dequantize_vec(decode_bytes(blob))
            except Exception:
                continue
    misses = [e for e in element_ids if e not in hits]
    return hits, misses


async def cache_vectors(
    r: aioredis.Redis,
    vset: str,
    vectors: dict[str, np.ndarray],
) -> None:
    """Write a batch of embeddings into the tier's vector cache (best-effort)."""
    if not vectors:
        return
    pipe = r.pipeline(transaction=False)
    for eid, vec in vectors.items():
        try:
            pipe.hset(vec_key(vset), eid, _quantize_vec(vec))
        except Exception:
            continue
    try:
        await pipe.execute()
    except Exception as e:  # pragma: no cover
        log.warning("cache_vectors pipeline failed: %s", e)


def proc_task_key(task: str) -> str:
    """Canonical dedup key for a procedure's task.

    The dedup hash used to be the raw string, so ``Task: rotate the vault.`` and ``task rotate the
    vault`` were two rows for one workflow — the write-side half of the tier's duplicate mass. Case,
    punctuation and whitespace are not meaning here, so they are folded out before the key is made.
    The vector path (``nearest_procedures`` > 0.90) stays as a second net: two people can describe
    the same workflow in genuinely different words.
    """
    t = unicodedata.normalize("NFKC", str(task or "")).casefold()
    t = re.sub(r"[\s_]+", " ", t)
    t = re.sub(r"[^\w\u4e00-\u9fff]+", " ", t)
    return re.sub(r" +", " ", t).strip()[:300]


async def _unlink_proc_side_indexes(
    r: aioredis.Redis,
    element_id: str,
    attrs: dict,
) -> None:
    """Drop a procedure's element ID from mem:proc_by_tool:* and mem:proc_tasks."""
    tools = attrs.get("tools_used") or []
    if isinstance(tools, str):
        tools = [tools]
    task = attrs.get("task") or ""
    if not tools and not task:
        return
    pipe = r.pipeline(transaction=False)
    for tool in tools:
        if tool:
            pipe.execute_command(
                "SREM", f"{PROC_BY_TOOL_PRE}{str(tool).lower()}", element_id,
            )
    if task:
        # Both shapes: rows written before the canonical key existed still hold the raw string.
        pipe.execute_command("HDEL", PROC_TASK_KEY, proc_task_key(task))
        pipe.execute_command("HDEL", PROC_TASK_KEY, task[:300])
    try:
        await pipe.execute()
    except Exception as e:  # pragma: no cover - side indexes are best-effort
        log.warning("unlink proc side indexes failed for %s: %s", element_id, e)


async def enumerate_elements(
    r: aioredis.Redis,
    vset: str,
    max_count: int = 100_000,
) -> list[dict]:
    """Enumerate every element of a vectorset via the shadow index.

    SSCAN is complete and exact, and VGETATTR resolves by ID regardless of HNSW
    graph health, so this is the reliable counterpart to ``core.search.vscan``.

    Returns the same ``{element, attrs}`` shape ``vscan`` produces, minus the
    (meaningless, for enumeration purposes) similarity score.

    If the shadow index has never been populated — i.e. elements written before
    this index existed — the result is empty.  Callers should treat that as
    "unknown" rather than "empty" and fall back to vscan.
    """
    key = idx_key(vset)
    ids: list[str] = []
    cursor = 0
    while True:
        cursor, batch = await r.sscan(key, cursor=cursor, count=500)
        ids.extend(decode_bytes(i) for i in batch)
        if int(cursor) == 0 or len(ids) >= max_count:
            break
    if not ids:
        return []

    out: list[dict] = []
    CHUNK = 500
    for i in range(0, len(ids), CHUNK):
        chunk = ids[i:i + CHUNK]
        pipe = r.pipeline(transaction=False)
        for eid in chunk:
            pipe.execute_command("VGETATTR", vset, eid)
        try:
            raws = await pipe.execute()
        except Exception as e:
            log.warning("enumerate_elements VGETATTR batch failed on %s: %s", vset, e)
            continue
        for eid, raw in zip(chunk, raws):
            if not raw:
                # Element vanished between SSCAN and VGETATTR — drop from shadow.
                await unregister_element(r, vset, eid)
                continue
            try:
                attrs = decode_json(raw) or {}
            except Exception:
                continue
            out.append({"element": eid, "attrs": attrs, "score": 0.0})
    return out[:max_count]


async def backfill_shadow_index(r: aioredis.Redis, vset: str) -> int:
    """Populate the shadow index from whatever vscan can reach.

    Best-effort repair for a degraded vectorset: only the reachable subset gets
    registered, but that is strictly better than nothing and the index becomes
    complete for every element written from now on.  Logs the shortfall loudly
    so the unreachable fraction is visible rather than silent.
    """
    from core.search import vscan

    try:
        card = int(await r.execute_command("VCARD", vset))
    except Exception:
        return 0
    scanned = await vscan(r, vset, max_count=max(card, 1000))
    if not scanned:
        return 0
    ids = [it["element"] for it in scanned]
    pipe = r.pipeline(transaction=False)
    for eid in ids:
        pipe.sadd(idx_key(vset), eid)
    await pipe.execute()
    missing = card - len(ids)
    if missing > 0:
        log.warning(
            "shadow-index backfill %s: registered %d/%d elements; %d (%.1f%%) are "
            "unreachable via VSIM and cannot be recovered without the element IDs",
            vset, len(ids), card, missing, missing / card * 100,
        )
    return len(ids)


# ── Procedure recovery via the tool reverse index ─────────────────────────────
#
# backfill_shadow_index() can only register what VSIM can still reach, and on
# mem:procedures that is ~13%.  The element IDs themselves were never lost,
# though — they are duplicated in two side structures that never depended on
# the HNSW graph:
#
#   mem:proc_by_tool:<tool>   SET    tool name -> proc element IDs (86 sets)
#   mem:proc_tasks            HASH   task -> proc element ID
#
# Measured on this instance (2026-10-22): the union holds 44,226 IDs, of which
# 13,249 resolve via VGETATTR against VCARD 13,260 — 99.9% of the set.  The
# remaining 30,977 are stale references to procedures that were deleted but
# never cleaned out of the reverse index, so every ID must be validated before
# it is registered, and the stale ones pruned while we are here (otherwise
# GET /tool-procedures/<tool> keeps paying a VGETATTR round trip per corpse).

async def recover_procedures_from_reverse_index(
    r: aioredis.Redis,
    *,
    prune_stale: bool = True,
    chunk: int = 500,
) -> dict:
    """Rebuild ``mem:idx:procedures`` from ``mem:proc_by_tool:*`` + ``mem:proc_tasks``.

    Returns a stats dict: sets scanned, IDs collected, live vs stale, how many
    were newly registered, how many stale references were pruned, and the
    resulting shadow coverage.  Safe to run repeatedly — SADD/SREM are
    idempotent and the work is a no-op once the index is complete.
    """
    # 1. Every reverse-index set and its members, in one round trip.
    keys: list[str] = []
    async for key in r.scan_iter(match=f"{PROC_BY_TOOL_PRE}*", count=200):
        keys.append(decode_bytes(key))
    members_by_key: dict[str, set[str]] = {}
    if keys:
        pipe = r.pipeline(transaction=False)
        for key in keys:
            pipe.smembers(key)
        try:
            batches = await pipe.execute()
        except Exception as e:
            log.warning("proc recovery: smembers batch failed: %s", e)
            batches = [[] for _ in keys]
        for key, batch in zip(keys, batches):
            ids = {decode_bytes(m) for m in (batch or []) if m}
            if ids:
                members_by_key[key] = ids

    # 2. mem:proc_tasks: task -> element id.
    task_ids: set[str] = set()
    try:
        raw_tasks = await r.hgetall(PROC_TASK_KEY)
        for v in (raw_tasks or {}).values():
            if v:
                task_ids.add(decode_bytes(v))
    except Exception as e:
        log.warning("proc recovery: hgetall %s failed: %s", PROC_TASK_KEY, e)

    all_ids: set[str] = set(task_ids)
    for ids in members_by_key.values():
        all_ids |= ids

    # 3. Validate every ID with batched VGETATTR (resolves by ID, graph-health
    #    independent) and split into live vs stale.
    live: list[str] = []
    stale: set[str] = set()
    ids_sorted = sorted(all_ids)
    for i in range(0, len(ids_sorted), chunk):
        part = ids_sorted[i:i + chunk]
        pipe = r.pipeline(transaction=False)
        for eid in part:
            pipe.execute_command("VGETATTR", PROC_KEY, eid)
        try:
            raws = await pipe.execute()
        except Exception as e:
            log.warning("proc recovery: VGETATTR batch failed: %s", e)
            continue  # unknown, not dead — leave these IDs alone
        for eid, raw in zip(part, raws):
            if raw:
                live.append(eid)
            else:
                stale.add(eid)

    # 4. Register the live IDs (SADD is idempotent; the return count is the
    #    number that were genuinely missing).
    newly = 0
    for i in range(0, len(live), chunk):
        pipe = r.pipeline(transaction=False)
        for eid in live[i:i + chunk]:
            pipe.sadd(idx_key(PROC_KEY), eid)
        try:
            res = await pipe.execute()
            newly += sum(int(x or 0) for x in res)
        except Exception as e:
            log.warning("proc recovery: shadow SADD batch failed: %s", e)

    # 5. Drop stale references from the reverse index so tool lookups stop
    #    returning corpses.
    pruned = 0
    if prune_stale and stale and members_by_key:
        pipe = r.pipeline(transaction=False)
        ops = 0
        for key, ids in members_by_key.items():
            dead = sorted(ids & stale)
            if dead:
                pipe.srem(key, *dead)
                ops += 1
        if ops:
            try:
                res = await pipe.execute()
                pruned = sum(int(x or 0) for x in res)
            except Exception as e:
                log.warning("proc recovery: stale-ref prune failed: %s", e)

    try:
        card = int(await r.execute_command("VCARD", PROC_KEY))
    except Exception:
        card = 0
    try:
        shadow = int(await r.scard(idx_key(PROC_KEY)))
    except Exception:
        shadow = 0

    stats = {
        "reverse_index_sets": len(members_by_key),
        "task_index_rows": len(task_ids),
        "ids_collected": len(all_ids),
        "live": len(live),
        "stale": len(stale),
        "newly_registered": newly,
        "already_registered": len(live) - newly,
        "stale_refs_pruned": pruned,
        "vcard": card,
        "shadow_after": shadow,
        "coverage": round(shadow / max(1, card), 4),
    }
    log.info(
        "proc recovery: %d ids from %d sets + %d task rows -> %d live "
        "(%d newly registered, %d already present), %d stale refs pruned; "
        "shadow now %d/%d (%.1f%%)",
        len(all_ids), len(members_by_key), len(task_ids), len(live), newly,
        len(live) - newly, pruned, shadow, card, shadow / max(1, card) * 100,
    )
    return stats


async def ensure_procedures_recovered(r: aioredis.Redis) -> dict | None:
    """Guarded wrapper: recover only when the shadow index is still short.

    Two round trips when already complete, so it is cheap enough to call on
    every boot; the full harvest only runs the first time.
    """
    try:
        card = int(await r.execute_command("VCARD", PROC_KEY))
    except Exception:
        return None
    if card <= 1:
        return None
    try:
        shadow = int(await r.scard(idx_key(PROC_KEY)))
    except Exception:
        shadow = 0
    if shadow >= card:
        return None
    return await recover_procedures_from_reverse_index(r)
from core import embedder as _embed_mod

def _dims() -> int:
    """Current embedding dimensions — resolved lazily after provider init."""
    return _embed_mod._get_provider().dims

# ── Shared connection pool (singleton) ────────────────────────────────────────
# Prevents connection leaks: get_client() returns the SAME Redis instance
# backed by a shared ConnectionPool instead of creating a new pool per call.
# Must call close_pool() at shutdown (called from lifespan).
_pool: aioredis.ConnectionPool | None = None
_client: aioredis.Redis | None = None

# ── Redis Lua scripts for atomic read-modify-write ────────────────────────────
# Race condition fix: concurrent reinforce_fact or soft_delete_fact calls could
# lose updates when two coroutines read the same attrs, modify independently,
# and overwrite each other. Lua scripts execute atomically in Redis, preventing
# lost increments and partial overwrites.

_SOFT_DELETE_LUA = """
local key = KEYS[1]
local elem = ARGV[1]
local superseded_by = ARGV[2]
local reason = ARGV[3]
local now_ms = ARGV[4]

local raw = redis.call('VGETATTR', key, elem)
if not raw then return 0 end

local ok, attrs = pcall(cjson.decode, raw)
if not ok or type(attrs) ~= 'table' then return 0 end

-- Don't re-supersede an already-superseded fact (idempotent)
if attrs["superseded_by"] and attrs["superseded_by"] ~= "" then
    return 1
end

attrs["superseded_by"] = superseded_by
attrs["superseded_at"] = tonumber(now_ms)
attrs["superseded_reason"] = reason

local new_json = cjson.encode(attrs)
redis.call('VSETATTR', key, elem, new_json)
return 1
"""

_FEEDBACK_LUA = """
local key = KEYS[1]
local elem = ARGV[1]
local action = ARGV[2]
local rating = tonumber(ARGV[3])
local now_ms = ARGV[4]
local comment = ARGV[5]

local raw = redis.call('VGETATTR', key, elem)
if not raw then return 0 end

local ok, attrs = pcall(cjson.decode, raw)
if not ok or type(attrs) ~= 'table' then return 0 end

attrs['user_rating'] = rating
attrs['user_rating_ts'] = tonumber(now_ms)

if comment and comment ~= '' then
    attrs['user_comment'] = comment
end

if action == 'boost' then
    local old_imp = attrs['importance'] or 0.5
    attrs['importance'] = math.min(1.0, old_imp * 1.2)
elseif action == 'reduce' then
    local old_imp = attrs['importance'] or 0.5
    attrs['importance'] = math.max(0.05, old_imp * 0.7)
    attrs['needs_review'] = true
end

local new_json = cjson.encode(attrs)
redis.call('VSETATTR', key, elem, new_json)
return 1
"""

_feedback_script_lock = asyncio.Lock()

_REINFORCE_LUA = """
local key = KEYS[1]
local elem = ARGV[1]
local now_ms = ARGV[2]
local new_source = ARGV[3]

local raw = redis.call('VGETATTR', key, elem)
if not raw then return 0 end

local ok, attrs = pcall(cjson.decode, raw)
if not ok or type(attrs) ~= 'table' then return 0 end

-- Don't reinforce superseded facts
if attrs["superseded_by"] and attrs["superseded_by"] ~= "" then
    return 0
end

attrs["source_count"] = (attrs["source_count"] or 1) + 1
attrs["last_confirmed_ts"] = tonumber(now_ms)
attrs["version"] = (attrs["version"] or 1) + 1

if new_source and new_source ~= "" then
    local sources = attrs["sources"] or {}
    local found = false
    for i, s in ipairs(sources) do
        if s == new_source then found = true; break end
    end
    if not found then
        table.insert(sources, new_source)
        if #sources > 10 then
            -- Keep only last 10 sources
            local trimmed = {}
            for i = #sources - 9, #sources do
                trimmed[#trimmed+1] = sources[i]
            end
            sources = trimmed
        end
        attrs["sources"] = sources
    end
end

local new_json = cjson.encode(attrs)
redis.call('VSETATTR', key, elem, new_json)
return 1
"""


# ── LLM Wiki v2: Confidence decay (Ebbinghaus forgetting curve) ───────────────
# Confidence decays exponentially with time since last confirmation.
# Each reinforcement (new source, re-observation) resets the curve.
# S (stability) varies by category: rules/identity decay slowly, bugs fast.

_CATEGORY_STABILITY_DAYS = {
    "identity": 365, "rule": 365, "preference": 180, "decision": 180,
    "work": 120, "personal": 120, "location": 120,
    "procedure": 90, "context": 60, "tool_use": 60,
    "entity": 90, "warning": 30, "reminder": 60,
    "general": 90, "capability_gained": 90, "env_change": 60,
    "env_context": 45, "deadline": 30,
}

_DEFAULT_STABILITY_DAYS = 90


def confidence_decay(attrs: dict) -> float:
    """
    Compute current effective confidence using Ebbinghaus forgetting curve.

    R(t) = e^(-t/S)  where t = days since last confirmation, S = stability.
    effective_confidence = base_confidence × R(t) × source_multiplier

    source_multiplier:
      - source_count ≥ 3 → 1.0 (well-supported fact)
      - source_count = 2 → 0.8 (corroborated)
      - source_count = 1 → 0.6 (single source, less reliable but not penalized to 0.33)
    - Reinforcement resets last_confirmed_ts → R(t) → 1.0 again
    - Superseded facts return 0.0 (no decay computation needed)
    """
    if attrs.get("superseded_by"):
        return 0.0
    base = attrs.get("confidence", 0.8)
    source_count = max(1, attrs.get("source_count", 1))
    last_confirmed = attrs.get("last_confirmed_ts", attrs.get("ts", 0))
    category = attrs.get("category", "general")

    now_ms = int(time.time() * 1000)
    days_since = max(0, (now_ms - last_confirmed) / 86_400_000)

    stability = _CATEGORY_STABILITY_DAYS.get(category, _DEFAULT_STABILITY_DAYS)
    retention = math.exp(-days_since / stability)
    if source_count >= 3:
        source_multiplier = 1.0
    elif source_count == 2:
        source_multiplier = 0.8
    else:
        source_multiplier = 0.6

    return round(base * retention * source_multiplier, 4)


async def reinforce_fact(
    r: aioredis.Redis,
    element_id: str,
    new_source: str | None = None,
) -> bool:
    """
    Reinforce a fact — increment source_count, reset last_confirmed_ts.

    LLM Wiki v2: "Confidence strengthens with reinforcement. Each reinforcement
    (access, confirmation from a new source) resets the [Ebbinghaus] curve."

    Uses atomic Lua script to prevent race conditions during concurrent
    reinforcements (e.g., two store operations reinforcing the same fact).

    Returns True if reinforcement succeeded.
    """
    now_ms = int(time.time() * 1000)
    try:
        result = await r.eval(
            _REINFORCE_LUA,
            1,                    # numkeys
            FACT_KEY,             # KEYS[1]
            element_id,           # ARGV[1]
            str(now_ms),          # ARGV[2]
            new_source or "",     # ARGV[3]
        )
        return int(result) == 1
    except Exception as e:
        log.warning("reinforce_fact Lua failed for %s: %s", element_id, e)
        return False


def _get_pool() -> aioredis.ConnectionPool:
    global _pool
    if _pool is None:
        _pool = aioredis.ConnectionPool.from_url(
            REDIS_URL,
            decode_responses=False,
            max_connections=50,
            socket_timeout=10,
            socket_connect_timeout=5,
            retry_on_timeout=True,
        )
    return _pool


async def get_client() -> aioredis.Redis:
    """Return the shared Redis client (backed by a connection pool).

    Safe to call from any async context. Always returns the same instance.
    """
    global _client
    if _client is None:
        _client = aioredis.Redis(connection_pool=_get_pool())
    return _client


async def close_pool() -> None:
    """Close the shared Redis client and connection pool. Call at shutdown."""
    global _client, _pool
    if _client is not None:
        await _client.aclose()
        _client = None
    if _pool is not None:
        await _pool.disconnect()
        _pool = None


def _blob(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


async def ensure_indexes(r: aioredis.Redis) -> None:
    """Seed vectorsets so VSIM works on empty sets."""
    for key in (EPISODE_KEY, FACT_KEY):
        card = await r.execute_command("VCARD", key)
        if not card:
            seed = np.zeros(_dims(), dtype=np.float32)
            attr = json.dumps({"_seed": True, "content": "",
                               "language": "en", "domain": "general",
                               "access_count": 0, "ts": 0,
                               "importance": 0.0, "superseded_by": ""})
            await r.execute_command("VADD", key, "FP32", _blob(seed), "__seed__",
                                    "SETATTR", attr)
    # Tool capability vectorset
    from .capability import ensure_tool_index
    await ensure_tool_index(r, _dims())
    # Procedural memory vectorset
    card = await r.execute_command("VCARD", PROC_KEY)
    if not card:
        seed = np.zeros(_dims(), dtype=np.float32)
        attr = json.dumps({"_seed": True, "task": "", "procedure": "",
                           "tools_used": [], "success_count": 0, "ts": 0})
        await r.execute_command("VADD", PROC_KEY, "FP32", _blob(seed), "__seed__",
                                "SETATTR", attr)


def _elem_name(elem: object) -> str:
    """Decode a VSIM / VGETATTR element name out of a raw Redis reply.

    Single choke point for element names, with a built-in regression tripwire:
    a bytes-repr name (``b'01M3…'``) is exactly what silently broke every heat
    bump and A-MEM evolution write once (commit e3e6b2b, decode_responses=False
    + ``str(bytes)``). Such a name matches no element, so writes keyed on it
    return 0 and nothing is logged. It can never be a valid ULID element id —
    so refuse it and shout, instead of quietly returning an unusable hit.
    """
    name = decode_bytes(elem)
    if (len(name) > 3 and name[:2].lower() in ("b'", 'b"')
            and name[-1] in ("'", '"')):
        log.warning(
            "element name arrived as a bytes repr (%r) — Redis replies are not "
            "being decoded; heat bumps / VSETATTR writes for it will no-op", name)
        return ""
    return name


def _parse_vsim_results(results, include_superseded: bool = False) -> list[dict]:
    """Parse a raw VSIM WITHSCORES WITHATTRIBS reply into hit dicts.

    Shared by ``knn_search`` and ``knn_search_batch`` so the two can never
    drift apart in what they consider a valid hit.
    """
    items: list[dict] = []
    i = 0
    while i + 2 < len(results):
        elem  = results[i]
        score = results[i + 1]
        raw   = results[i + 2]
        i += 3

        elem_str = _elem_name(elem)
        if elem_str == "__seed__":
            continue

        try:
            attrs = decode_json(raw) or {}
        except Exception as e:
            log.warning("knn_search: failed to parse attrs for %s: %s", elem_str, e)
            attrs = {}

        if attrs.get("_seed") or not attrs.get("content"):
            continue

        # SimpleMem soft-delete: skip entries marked as superseded by consolidation
        if not include_superseded and attrs.get("superseded_by"):
            continue

        item = {
            "content":   attrs.get("content", ""),
            "category":  attrs.get("category", attrs.get("domain", "")),
            "language":  attrs.get("language", ""),
            "domain":    attrs.get("domain", ""),
            "score":     float(score) if score else 0.0,
            "attrs":     attrs,
            "_element":  elem_str,
        }
        items.append(item)
    return items


async def knn_search_batch(
    r: aioredis.Redis,
    vset_key: str,
    embeddings: list[np.ndarray],
    k: int = 1,
    filter_expr: Optional[str] = None,
    include_superseded: bool = False,
) -> list[list[dict]]:
    """Run one VSIM per embedding, all in a single pipeline round trip.

    The store path used to issue one ``knn_search`` per extracted fact, so
    dedup cost N serial round trips on the critical path.  Every command in
    the pipeline is an independent read, so the results are identical to the
    sequential version — just in one network flight.

    Returns one hit list per input embedding, positionally aligned.
    """
    if not embeddings:
        return []
    pipe = r.pipeline(transaction=False)
    for emb in embeddings:
        cmd = ["VSIM", vset_key, "FP32", _blob(emb),
               "COUNT", k + 1, "WITHSCORES", "WITHATTRIBS"]
        if filter_expr:
            cmd += ["FILTER", filter_expr]
        pipe.execute_command(*cmd)
    try:
        replies = await pipe.execute()
    except Exception as e:
        log.error("knn_search_batch VSIM pipeline failed on %s: %s", vset_key, e)
        return [[] for _ in embeddings]
    return [_parse_vsim_results(reply, include_superseded) for reply in replies]


async def knn_search(
    r: aioredis.Redis,
    vset_key: str,
    embedding: np.ndarray,
    k: int,
    filter_expr: Optional[str] = None,
    bump_heat: bool = False,
    include_superseded: bool = False,
) -> list[dict]:
    """
    KNN search. Returns list of {content, category, score, attrs, _element}.

    filter_expr: e.g. '.language == "zh"' or '.domain == "trading"'
    bump_heat:   if True, async-increment access_count on each hit (non-blocking).
    include_superseded: if False (default), filter out soft-deleted entries.
                        SimpleMem consolidation marks losers as superseded_by=<winner_id>.
    """
    cmd = ["VSIM", vset_key, "FP32", _blob(embedding),
           "COUNT", k + 1, "WITHSCORES", "WITHATTRIBS"]
    if filter_expr:
        cmd += ["FILTER", filter_expr]

    try:
        results = await r.execute_command(*cmd)
    except Exception as e:
        log.error("knn_search VSIM failed on %s: %s", vset_key, e)
        return []

    items = _parse_vsim_results(results, include_superseded)

    # Batch heat bumps into a single async task to avoid connection pool exhaustion
    if bump_heat and items:
        import asyncio
        from .heat import atomic_bump_many

        elements = [item["_element"] for item in items]

        async def _batch_bump():
            # One pipeline round trip instead of N sequential Lua calls.
            await atomic_bump_many(r, vset_key, elements)

        asyncio.create_task(_batch_bump())

    return items


async def save_episode(
    r: aioredis.Redis,
    session_id: str,
    content: str,
    embedding: np.ndarray,
    language: str = "en",
    domain: str = "general",
    ep_type: str = "general",          # claude-mem taxonomy: bugfix|feature|discovery|decision|change|procedure|preference|context|general
    prev_episode_id: str = "",         # causal chain: predecessor episode UID
    user_id: str = "",
    agent_id: str = "",
) -> str:
    uid  = str(_ulid_mod.new())
    attr: dict = {
        "content":         content[:2000],
        "category":        "episode",
        "ep_type":         ep_type,
        "language":        language,
        "domain":          domain,
        "session_id":      session_id,
        "ts":              int(time.time() * 1000),
        "access_count":    0,
        "importance":      0.5,
        "superseded_by":   "",
        "prev_episode_id": prev_episode_id,
        "next_episode_id": "",         # filled in when a successor is stored
    }
    scoped = scope_of(user_id, agent_id)
    if not scoped.is_global:
        attr.update(scoped.attrs())
    await r.execute_command("VADD", EPISODE_KEY, "FP32", _blob(embedding), uid,
                            "SETATTR", json.dumps(attr))
    await register_element(r, EPISODE_KEY, uid)
    try:
        await r.hset(vec_key(EPISODE_KEY), uid, _quantize_vec(embedding))
    except Exception as e:  # pragma: no cover - cache is best-effort
        log.warning("episode vector cache failed for %s: %s", uid, e)
    return uid


async def get_last_episode_id(r: aioredis.Redis, session_id: str) -> str:
    """Return the most recently stored episode UID for this session (for causal chaining)."""
    if not session_id:
        return ""
    val = await r.get(f"{SESSION_PRE}{session_id}:last_ep")
    if val is None:
        return ""
    return decode_bytes(val)


async def set_last_episode_id(r: aioredis.Redis, session_id: str, uid: str) -> None:
    """Track the most recent episode UID per session (4h TTL matches session KV)."""
    if not session_id:
        return
    await r.set(f"{SESSION_PRE}{session_id}:last_ep", uid.encode(), ex=14400)


async def update_episode_next_id(r: aioredis.Redis, uid: str, next_uid: str) -> None:
    """Back-fill next_episode_id on the predecessor episode to complete the doubly-linked chain."""
    if not uid or not next_uid:
        return
    try:
        raw = await r.execute_command("VGETATTR", EPISODE_KEY, uid)
        if raw:
            attrs = decode_json(raw) or {}
            attrs["next_episode_id"] = next_uid
            await r.execute_command("VSETATTR", EPISODE_KEY, uid, json.dumps(attrs))
    except Exception as e:
        log.warning("update_episode_next_id failed for %s→%s: %s", uid, next_uid, e)


async def save_fact(
    r: aioredis.Redis,
    content: str,
    category: str,
    confidence: float,
    embedding: np.ndarray,
    language: str = "en",
    domain: str = "general",
    keywords: list[str] | None = None,
    persons: list[str] | None = None,
    entities: list[str] | None = None,
    # SimpleMem additions (v0.9.1)
    importance: float = 0.5,
    topic: str | None = None,
    location: str | None = None,
    # Dual-layer linking: source episode for narrative context retrieval (Memori §2.2)
    source_episode_id: str | None = None,
    # Memori semantic triple fields (arXiv:2603.19935)
    triple_s: str | None = None,
    triple_p: str | None = None,
    triple_o: str | None = None,
    # LLM Wiki v2 fields (v1.1)
    source_count: int = 1,
    sources: list[str] | None = None,
    superseded_reason: str = "",
    # Track scoping (EverOS parity): exactly one of these, or neither.
    user_id: str = "",
    agent_id: str = "",
) -> str:
    """
    Save a fact to the semantic memory vectorset.

    SimpleMem fields (v0.9.1):
      importance:    float 0-1, decays during consolidation (×0.9 per 90 days)
      topic:         SimpleMem topic phrase for the symbolic layer
      location:      SimpleMem location string for symbolic search
      superseded_by: empty string = active; set to winner's element ID on merge (soft-delete)

    Memori fields (v1.1):
      source_episode_id: UID of the episode this fact was extracted from (dual-layer linking)
      triple_s/p/o:      semantic triple subject/predicate/object for precision retrieval

    LLM Wiki v2 fields (v1.1):
      source_count:      how many independent sources support this fact (≥3 = well-supported)
      last_confirmed_ts: timestamp of most recent reinforcement (Ebbinghaus decay anchor)
      superseded_at:     timestamp when supersession occurred (0 = active)
      superseded_reason: "merged" | "contradicted" | "updated" | "pruned" | ""
      version:           incrementing counter for each fact (tracks edits)
      sources:           list of source identifiers that contributed to this fact
    """
    uid  = str(_ulid_mod.new())
    attr_dict = _build_fact_attrs(
        content=content, category=category, language=language, domain=domain,
        confidence=confidence, keywords=keywords, persons=persons, entities=entities,
        importance=importance, topic=topic, location=location,
        source_episode_id=source_episode_id,
        triple_s=triple_s, triple_p=triple_p, triple_o=triple_o,
        source_count=source_count, sources=sources,
        superseded_reason=superseded_reason,
        user_id=user_id, agent_id=agent_id,
    )

    await r.execute_command("VADD", FACT_KEY, "FP32", _blob(embedding), uid,
                            "SETATTR", json.dumps(attr_dict))
    await register_element(r, FACT_KEY, uid)
    # Cache the raw embedding so consolidation never re-embeds the corpus.
    try:
        await r.hset(vec_key(FACT_KEY), uid, _quantize_vec(embedding))
    except Exception as e:  # pragma: no cover - cache is best-effort
        log.warning("fact vector cache failed for %s: %s", uid, e)
    return uid


def _build_fact_attrs(
    *,
    content: str,
    category: str,
    language: str = "en",
    domain: str = "general",
    confidence: float = 0.8,
    keywords: list[str] | None = None,
    persons: list[str] | None = None,
    entities: list[str] | None = None,
    importance: float = 0.5,
    topic: str | None = None,
    location: str | None = None,
    source_episode_id: str | None = None,
    triple_s: str | None = None,
    triple_p: str | None = None,
    triple_o: str | None = None,
    source_count: int = 1,
    sources: list[str] | None = None,
    superseded_reason: str = "",
    user_id: str = "",
    agent_id: str = "",
) -> dict:
    """Build the vectorset attribute payload for one fact.

    Shared by ``save_fact`` and ``save_facts_batch`` so a batch write produces
    byte-identical rows to the single write path.
    """
    now_ms = int(time.time() * 1000)
    attr_dict: dict = {
        "content":           content[:500],
        "category":          category,
        "language":          language,
        "domain":            domain,
        "confidence":        confidence,
        "ts":                now_ms,
        "access_count":      0,
        "importance":        max(0.0, min(1.0, importance)),
        "superseded_by":     "",   # empty = active; set to winner_id on consolidation
        "source_count":      source_count,
        "last_confirmed_ts": now_ms,
        "superseded_at":     0,    # 0 = active; set to timestamp on supersession
        "superseded_reason": superseded_reason,
        "version":           1,
    }
    if keywords:
        attr_dict["keywords"] = keywords[:10]
    if persons:
        attr_dict["persons"] = persons[:5]
    if entities:
        attr_dict["entities"] = entities[:5]
    if topic:
        attr_dict["topic"] = topic[:100]
    if location:
        attr_dict["location"] = location[:100]
    if source_episode_id:
        attr_dict["source_episode_id"] = source_episode_id
    if triple_s and triple_p and triple_o:
        attr_dict["triple_s"] = triple_s[:100]
        attr_dict["triple_p"] = triple_p[:100]
        attr_dict["triple_o"] = triple_o[:200]
        # Also store compact triple string for BM25 keyword matching
        attr_dict["triple_str"] = f"{triple_s} | {triple_p} | {triple_o}"
    if sources:
        attr_dict["sources"] = sources[:10]
    # Written only when a track was named, so unscoped rows keep the byte shape
    # older builds wrote -- and stay readable as the global corpus.
    scoped = scope_of(user_id, agent_id)
    if not scoped.is_global:
        attr_dict.update(scoped.attrs())
    return attr_dict


async def save_facts_batch(
    r: aioredis.Redis,
    facts: list[dict],
) -> list[str]:
    """Write many facts in one pipeline instead of N × (VADD + SADD).

    ``facts`` items take the same keyword arguments as ``save_fact`` plus a
    required ``embedding``.  Returns the new element UIDs, positionally
    aligned with the input — a UID is only meaningful if the corresponding
    write actually landed, which is why the return list can contain "" for a
    failed item (the caller should treat those as not-saved).

    Two round trips total regardless of N: one for the VADDs, one for the
    shadow-index SADDs.
    """
    if not facts:
        return []
    uids: list[str] = []
    payloads: list[str] = []
    for spec in facts:
        uid = str(_ulid_mod.new())
        uids.append(uid)
        attr_dict = _build_fact_attrs(**{k: v for k, v in spec.items() if k != "embedding"})
        payloads.append(json.dumps(attr_dict))

    pipe = r.pipeline(transaction=False)
    for uid, payload, spec in zip(uids, payloads, facts):
        pipe.execute_command("VADD", FACT_KEY, "FP32", _blob(spec["embedding"]),
                             uid, "SETATTR", payload)
    try:
        await pipe.execute()
    except Exception as e:
        log.error("save_facts_batch VADD pipeline failed: %s", e)
        return ["" for _ in facts]

    pipe = r.pipeline(transaction=False)
    for uid in uids:
        pipe.sadd(idx_key(FACT_KEY), uid)
    try:
        await pipe.execute()
    except Exception as e:
        # The vectors themselves landed; the shadow index is best-effort and
        # can be rebuilt with POST /shadow-index/backfill.
        log.warning("save_facts_batch shadow SADD pipeline failed: %s", e)
    return uids


async def soft_delete_fact(
    r: aioredis.Redis,
    element_id: str,
    superseded_by: str,
    reason: str = "merged",
) -> None:
    """
    Mark a fact as superseded (soft-delete, SimpleMem consolidation pattern).

    Sets superseded_by = winner's element_id. knn_search() skips these by default.
    The entry is kept for audit/history but excluded from recall.

    LLM Wiki v2 additions (v1.1):
      superseded_at:     timestamp when supersession occurred (version chain)
      superseded_reason: "merged" | "contradicted" | "updated" | "pruned"

    Uses atomic Lua script to prevent race conditions during concurrent
    soft-deletes (e.g., consolidation + contradiction detection running in parallel).
    The script is idempotent: re-superseding an already-superseded fact is a no-op.

    Bug fix v0.9.2: removed the ×0.1 importance decay that was applied here.
    Phase 1 (Decay) already applies ×0.9 per 90 days; Phase 3 (Prune) removes
    entries that fall below 0.05. Applying a second ×0.1 during soft-delete
    caused double-decay: loser facts dropped from 0.5 → 0.045, landing just
    below the prune floor and being silently hard-removed the very next run.
    """
    now_ms = int(time.time() * 1000)
    try:
        result = await r.eval(
            _SOFT_DELETE_LUA,
            1,                    # numkeys
            FACT_KEY,             # KEYS[1]
            element_id,           # ARGV[1]
            superseded_by,        # ARGV[2]
            reason,               # ARGV[3]
            str(now_ms),          # ARGV[4]
        )
        if int(result) == 1:
            return
    except Exception as e:
        log.warning("soft_delete_fact Lua failed for %s: %s", element_id, e)

    # Fallback: VSIM ELE lookup (for Redis versions without EVAL support)
    try:
        scan = await r.execute_command(
            "VSIM", FACT_KEY, "ELE", element_id,
            "COUNT", 1, "WITHATTRIBS"
        )
        if scan and len(scan) >= 2:
            raw = scan[1]
            attrs = decode_json(raw) or {}
            if attrs.get("superseded_by"):
                return  # already superseded, idempotent
            attrs["superseded_by"] = superseded_by
            attrs["superseded_at"] = now_ms
            attrs["superseded_reason"] = reason
            await r.execute_command("VSETATTR", FACT_KEY, element_id, json.dumps(attrs))
            return
    except Exception as e:
        log.warning("soft_delete_fact VSIM ELE fallback failed for %s: %s", element_id, e)

    # Last resort: try VGETATTR + VSETATTR manually (preserves original attrs)
    try:
        raw = await r.execute_command("VGETATTR", FACT_KEY, element_id)
        if raw:
            attrs = decode_json(raw) or {}
            if attrs.get("superseded_by"):
                return  # already superseded, idempotent
            attrs["superseded_by"] = superseded_by
            attrs["superseded_at"] = now_ms
            attrs["superseded_reason"] = reason
            await r.execute_command("VSETATTR", FACT_KEY, element_id, json.dumps(attrs))
            return
    except Exception as e2:
        log.error("soft_delete_fact VGETATTR fallback also failed for %s: %s", element_id, e2)


async def link_proc_to_tools(
    r: aioredis.Redis,
    proc_uid: str,
    tools_used: list[str],
) -> None:
    """
    Add proc_uid to each tool's reverse-index set (mem:proc_by_tool:<tool>).
    Enables O(1) lookup: "which procedures use tool X?"
    """
    for tool in tools_used:
        if not tool:
            continue
        key = f"{PROC_BY_TOOL_PRE}{tool.lower()}"
        await r.sadd(key, proc_uid)


async def get_procs_for_tool(
    r: aioredis.Redis,
    tool_name: str,
) -> list[str]:
    """
    Return list of procedure UIDs that are indexed under tool_name.
    """
    key = f"{PROC_BY_TOOL_PRE}{tool_name.lower()}"
    members = await r.smembers(key)
    return [decode_bytes(m) for m in members]


async def scan_all_procedures(r: aioredis.Redis, max_count: int = 1000) -> list[dict]:
    """
    Scan all entries in mem:procedures and return list of
    {uid, task, tools_used, ...attrs}.  Used by the backfill endpoint.

    Prefers the element-ID shadow index over VSIM: mem:procedures' HNSW graph
    is degraded (only ~13% of elements reachable per VSIM call), so the old
    zero-vector VSIM scan silently lost most of the set.  Falls back to VSIM
    when the shadow index has not been populated yet, and logs the shortfall.
    """
    try:
        card = int(await r.execute_command("VCARD", PROC_KEY))
    except Exception:
        return []
    if card <= 1:
        return []

    shadow = await r.scard(idx_key(PROC_KEY))
    if shadow > 0:
        enumerated = await enumerate_elements(r, PROC_KEY, max_count=max_count)
        if enumerated:
            procs = []
            for it in enumerated:
                attrs = it.get("attrs") or {}
                if attrs.get("_seed") or not attrs.get("task"):
                    continue
                procs.append({"uid": it["element"], **attrs})
            if len(procs) < card:
                log.warning(
                    "scan_all_procedures: shadow index holds %d of %d elements",
                    len(procs), card,
                )
            return procs
        log.warning("scan_all_procedures: shadow index present but empty; falling back to VSIM")

    # Fallback: legacy zero-vector VSIM scan (incomplete on a degraded graph).
    seed = np.zeros(_dims(), dtype=np.float32)
    try:
        results = await r.execute_command(
            "VSIM", PROC_KEY, "FP32", _blob(seed),
            "COUNT", min(card, max_count), "WITHSCORES", "WITHATTRIBS"
        )
    except Exception as e:
        log.error("scan_all_procedures VSIM failed: %s", e)
        return []
    procs: list[dict] = []
    i = 0
    while i + 2 < len(results):
        elem = results[i]; raw = results[i + 2]
        i += 3
        elem_str = _elem_name(elem)
        if elem_str == "__seed__":
            continue
        try:
            attrs = decode_json(raw)
        except Exception:
            continue
        if attrs.get("_seed") or not attrs.get("task"):
            continue
        procs.append({"uid": elem_str, **attrs})
    if len(procs) < card:
        log.warning(
            "scan_all_procedures: VSIM reached only %d of %d elements — the HNSW "
            "graph is degraded; run the shadow-index backfill to recover the rest",
            len(procs), card,
        )
    return procs


async def nearest_procedures(
    r: aioredis.Redis,
    embedding: np.ndarray,
    k: int = 5,
    filter_expr: str | None = None,
) -> list[dict]:
    """Top-k procedures by vector score.

    Procedure rows store `task`, not `content`. knn_search drops them.
    """
    cmd = ["VSIM", PROC_KEY, "FP32", _blob(embedding),
           "COUNT", k + 1, "WITHSCORES", "WITHATTRIBS"]
    if filter_expr:
        cmd += ["FILTER", filter_expr]
    try:
        results = await r.execute_command(*cmd)
    except Exception as e:
        log.error("nearest_procedures VSIM failed: %s", e)
        return []
    items: list[dict] = []
    i = 0
    while i + 2 < len(results):
        elem, score, raw = results[i], results[i + 1], results[i + 2]
        i += 3
        elem_str = _elem_name(elem)
        if elem_str == "__seed__":
            continue
        try:
            attrs = decode_json(raw) or {}
        except Exception:
            continue
        if attrs.get("_seed") or not attrs.get("task"):
            continue
        items.append({
            "score": float(score) if score else 0.0,
            "attrs": attrs,
            "_element": elem_str,
        })
        if len(items) >= k:
            break
    return items


async def find_procedure_by_task(
    r: aioredis.Redis,
    embedding: np.ndarray,
    task: str,
    *,
    min_score: float = 0.80,
    k: int = 8,
    scope: Scope = Scope(),
) -> dict | None:
    """Return an existing procedure with this exact task, including legacy rows.

    The task hash is global while procedures can be per-track, so a mapped hit
    is only accepted when the row belongs to the caller's track; otherwise the
    lookup falls through to a scoped vector search, which never crosses tracks.
    """
    key = proc_task_key(task)
    for k_key in (key, task[:300]):          # canonical first, then the pre-migration shape
        mapped = await r.hget(PROC_TASK_KEY, k_key)
        if not mapped:
            continue
        elem = decode_bytes(mapped)
        try:
            raw = await r.execute_command("VGETATTR", PROC_KEY, elem)
            attrs = decode_json(raw) or {}
        except Exception:
            attrs = {}
        if attrs.get("task") and scope.matches(attrs):
            if k_key != key:
                try:
                    await r.hset(PROC_TASK_KEY, key, elem)
                except Exception:
                    pass
            return {"score": 1.0, "attrs": attrs, "_element": elem}
    for item in await nearest_procedures(r, embedding, k=k,
                                         filter_expr=scope.filter() or None):
        if (item["attrs"].get("task") == task or
                proc_task_key(item["attrs"].get("task")) == key) and item["score"] >= min_score:
            try:
                await r.hset(PROC_TASK_KEY, key, item["_element"])
            except Exception:
                pass
            return item
    return None


async def save_procedure(
    r: aioredis.Redis,
    task: str,
    procedure: str,
    embedding: np.ndarray,
    tools_used: list[str] | None = None,
    domain: str = "general",
    language: str = "en",
    user_id: str = "",
    agent_id: str = "",
) -> str:
    """
    Store a successful procedure/workflow in mem:procedures.
    Also updates the reverse index mem:proc_by_tool:<tool> for each tool.
    """
    uid = str(_ulid_mod.new())
    tools = (tools_used or [])[:10]
    task_key = task[:300]
    attrs = {
        "task":          task_key,
        "task_key":      proc_task_key(task),
        "content":       task_key,
        "procedure":     procedure[:1000],
        "tools_used":    tools,
        "domain":        domain,
        "language":      language,
        "success_count": 1,
        "ts":            int(time.time() * 1000),
    }
    scoped = scope_of(user_id, agent_id)
    if not scoped.is_global:
        attrs.update(scoped.attrs())
    attr = json.dumps(attrs)
    await r.execute_command("VADD", PROC_KEY, "FP32", _blob(embedding), uid,
                            "SETATTR", attr)
    await register_element(r, PROC_KEY, uid)
    try:
        await r.hset(vec_key(PROC_KEY), uid, _quantize_vec(embedding))
    except Exception as e:  # pragma: no cover - cache is best-effort
        log.warning("procedure vector cache failed for %s: %s", uid, e)
    try:
        await r.hset(PROC_TASK_KEY, attrs["task_key"], uid)
    except Exception:
        pass
    # Populate reverse index for each tool
    if tools:
        await link_proc_to_tools(r, uid, tools)
    return uid


async def get_session_context(r: aioredis.Redis, session_id: str) -> str | None:
    if not session_id:
        return None
    val = await r.get(f"{SESSION_PRE}{session_id}:ctx")
    if val is None:
        return None
    return val.decode("utf-8", errors="replace") if isinstance(val, bytes) else val


async def set_session_context(
    r: aioredis.Redis, session_id: str, content: str, ttl_s: int = 14400
) -> None:
    if not session_id:
        return
    await r.set(f"{SESSION_PRE}{session_id}:ctx", content.encode(), ex=ttl_s)


def session_owner_key(session_id: str) -> str:
    return f"{SESSION_PRE}{session_id}:owner"


async def set_session_owner(
    r: aioredis.Redis, session_id: str, scoped: Scope, ttl_s: int = 14400
) -> None:
    """Remember which track opened a session, so compress and recall can agree.

    A session key is a flat ``mem:session:<id>:*`` namespace, while the rows it
    later promotes are per-track. Recording the owner at store time is what
    lets a compress call (which carries only the session id) file the summary
    under the same track the conversation was stored in.
    """
    if not session_id or scoped.is_global:
        return
    await r.set(session_owner_key(session_id), json.dumps(scoped.attrs()), ex=ttl_s)


async def get_session_owner(r: aioredis.Redis, session_id: str) -> Scope:
    """The track that opened ``session_id``, or an empty scope when unrecorded."""
    if not session_id:
        return Scope()
    raw = await r.get(session_owner_key(session_id))
    if not raw:
        return Scope()
    try:
        attrs = decode_json(raw) or {}
    except Exception:
        return Scope()
    return scope_of(attrs.get("user_id"), attrs.get("agent_id"))


async def vcard(r: aioredis.Redis) -> dict:
    ep   = await r.execute_command("VCARD", EPISODE_KEY)
    fact = await r.execute_command("VCARD", FACT_KEY)
    return {"episodes": max(0, int(ep or 0) - 1),
            "facts":    max(0, int(fact or 0) - 1)}


async def get_attrs(r: aioredis.Redis, vset: str, element_id: str) -> dict | None:
    """Fetch and decode VGETATTR for an element. Returns dict or None if not found.

    Replaces the 13+ copy-paste VGETATTR+decode patterns across route modules.
    """
    from .utils import decode_attrs
    raw = await r.execute_command("VGETATTR", vset, element_id)
    if not raw:
        return None
    attrs = decode_attrs(raw)
    return attrs if attrs else None


async def set_attrs(r: aioredis.Redis, vset: str, element_id: str, attrs: dict) -> None:
    """Encode and write VSETATTR for an element."""
    await r.execute_command("VSETATTR", vset, element_id, json.dumps(attrs, ensure_ascii=False))


async def atomic_feedback(
    r: aioredis.Redis,
    element_id: str,
    action: str,
    rating: int,
    comment: str = "",
) -> dict | None:
    """Atomic feedback: adjust importance + record rating in a single Lua call.

    Prevents race conditions when two concurrent feedback requests hit the same fact.

    Args:
        action: "boost" (rating 4-5), "reduce" (rating 1-2), or "neutral" (rating 3)
        rating: 1-5 star rating
        comment: optional user comment

    Returns:
        Updated attrs dict, or None if element not found.
    """
    now_ms = int(time.time() * 1000)
    try:
        # Per-client: a Script object is bound to the pool it was registered from, so a module
        # global would hand the next caller someone else's connection.
        script = getattr(r, "_store_feedback_script", None)
        if script is None:
            async with _feedback_script_lock:
                script = getattr(r, "_store_feedback_script", None)
                if script is None:
                    script = r.register_script(_FEEDBACK_LUA)
                    setattr(r, "_store_feedback_script", script)
        result = await script(
            keys=[FACT_KEY],
            args=[element_id, action, str(rating), str(now_ms), comment[:500]],
        )
        if int(result) != 1:
            return None
        # Read back the updated attrs
        return await get_attrs(r, FACT_KEY, element_id)
    except Exception as e:
        log.warning("atomic_feedback Lua failed for %s: %s", element_id, e)
        return None


# ── Dimension delegation (PEP 562) ───────────────────────────────────────────
#
# ``embedder.DIMS`` is a mutable module global: it starts at the MiniLM default
# of 384 and is rewritten in place when the provider resolves (nomic-embed-text
# -> 768) or when the dimension guard force-switches providers.  A static
# ``from core.embedder import DIMS`` would therefore capture whatever value
# happened to be live at import time and go stale the moment the provider
# changed.
#
# Exposing it through a module-level ``__getattr__`` keeps ``store.DIMS`` (and
# ``mem_store.DIMS``) reading the live value on every access, which is what the
# lifecycle tests and the zero-vector scan path both rely on.


def __getattr__(name: str):
    if name == "DIMS":
        from core import embedder

        return embedder.DIMS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
