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
import json
import logging
import math
import os
import time
from typing import Optional

import numpy as np
import redis.asyncio as aioredis
import ulid as _ulid_mod

from .utils import decode_json, force_str

log = logging.getLogger(__name__)

REDIS_URL    = os.getenv("REDIS_URL", "redis://localhost:6379")
EPISODE_KEY  = "mem:episodes"
FACT_KEY     = "mem:facts"
TOOL_KEY        = "mem:tools"          # capability: tool/skill definitions
TOOL_GRAPH_KEY  = "mem:tool_graph"     # AutoTool TIG: tool transition counts hash (a:b → count)
PROC_KEY        = "mem:procedures"     # procedural: successful workflows/patterns (4th cognitive tier)
PROC_BY_TOOL_PRE = "mem:proc_by_tool:" # reverse index: tool_name → Set of proc element IDs
ENV_KEY      = "mem:env"        # capability: environment state hash
AGENT_KEY    = "mem:agent"      # capability: agent self-model hash
SESSION_PRE  = "mem:session:"
PERSONA_KEY  = "mem:persona"
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

_feedback_script = None
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

    items: list[dict] = []
    i = 0
    while i + 2 < len(results):
        elem  = results[i]
        score = results[i + 1]
        raw   = results[i + 2]
        i += 3

        elem_str = force_str(elem)
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

    # Batch heat bumps into a single async task to avoid connection pool exhaustion
    if bump_heat and items:
        import asyncio
        from .heat import atomic_bump

        async def _batch_bump():
            for item in items:
                try:
                    await atomic_bump(r, vset_key, item["_element"], "access_count", "last_accessed")
                except Exception:
                    pass  # non-critical

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
) -> str:
    uid  = str(_ulid_mod.new())
    attr = json.dumps({
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
    })
    await r.execute_command("VADD", EPISODE_KEY, "FP32", _blob(embedding), uid,
                            "SETATTR", attr)
    return uid


async def get_last_episode_id(r: aioredis.Redis, session_id: str) -> str:
    """Return the most recently stored episode UID for this session (for causal chaining)."""
    if not session_id:
        return ""
    val = await r.get(f"{SESSION_PRE}{session_id}:last_ep")
    if val is None:
        return ""
    return force_str(val)


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

    await r.execute_command("VADD", FACT_KEY, "FP32", _blob(embedding), uid,
                            "SETATTR", json.dumps(attr_dict))
    return uid


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
    return [force_str(m) for m in members]


async def scan_all_procedures(r: aioredis.Redis, max_count: int = 1000) -> list[dict]:
    """
    Scan all entries in mem:procedures and return list of
    {uid, task, tools_used, ...attrs}.  Used by the backfill endpoint.
    """
    card = await r.execute_command("VCARD", PROC_KEY)
    if not card or int(card) <= 1:
        return []
    seed = np.zeros(_dims(), dtype=np.float32)
    try:
        results = await r.execute_command(
            "VSIM", PROC_KEY, "FP32", _blob(seed),
            "COUNT", min(int(card), max_count), "WITHSCORES", "WITHATTRIBS"
        )
    except Exception as e:
        log.error("scan_all_procedures VSIM failed: %s", e)
        return []
    procs: list[dict] = []
    i = 0
    while i + 2 < len(results):
        elem = results[i]; raw = results[i + 2]
        i += 3
        elem_str = force_str(elem)
        if elem_str == "__seed__":
            continue
        try:
            attrs = decode_json(raw)
        except Exception:
            continue
        if attrs.get("_seed") or not attrs.get("task"):
            continue
        procs.append({"uid": elem_str, **attrs})
    return procs


async def save_procedure(
    r: aioredis.Redis,
    task: str,
    procedure: str,
    embedding: np.ndarray,
    tools_used: list[str] | None = None,
    domain: str = "general",
    language: str = "en",
) -> str:
    """
    Store a successful procedure/workflow in mem:procedures.
    Also updates the reverse index mem:proc_by_tool:<tool> for each tool.
    """
    uid = str(_ulid_mod.new())
    tools = (tools_used or [])[:10]
    attr = json.dumps({
        "task":          task[:300],
        "procedure":     procedure[:1000],
        "tools_used":    tools,
        "domain":        domain,
        "language":      language,
        "success_count": 1,
        "ts":            int(time.time() * 1000),
    })
    await r.execute_command("VADD", PROC_KEY, "FP32", _blob(embedding), uid,
                            "SETATTR", attr)
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
    global _feedback_script
    now_ms = int(time.time() * 1000)
    try:
        if _feedback_script is None:
            async with _feedback_script_lock:
                if _feedback_script is None:
                    _feedback_script = r.register_script(_FEEDBACK_LUA)
        result = await _feedback_script(
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
