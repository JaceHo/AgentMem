"""
Redis 8 native vectorset store — v0.9.1 (SimpleMem-aligned)

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

Redis 8 vectorset API summary:
  VADD  key FP32 <blob> <element> [SETATTR <json>]
  VSIM  key FP32 <blob> COUNT k WITHSCORES WITHATTRIBS [FILTER <expr>]
  VSETATTR key element <json>   — FILTER expr: .field == "value"
"""

import json
import time
from typing import Optional

import numpy as np
import redis.asyncio as aioredis
from ulid import ULID

REDIS_URL    = "redis://localhost:6379"
EPISODE_KEY  = "mem:episodes"
FACT_KEY     = "mem:facts"
TOOL_KEY     = "mem:tools"      # capability: tool/skill definitions
PROC_KEY     = "mem:procedures" # procedural: successful workflows/patterns (4th cognitive tier)
ENV_KEY      = "mem:env"        # capability: environment state hash
AGENT_KEY    = "mem:agent"      # capability: agent self-model hash
SESSION_PRE  = "mem:session:"
PERSONA_KEY  = "mem:persona"
DIMS         = 384


def _blob(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


async def get_client() -> aioredis.Redis:
    return aioredis.from_url(REDIS_URL, decode_responses=False)


async def ensure_indexes(r: aioredis.Redis) -> None:
    """Seed vectorsets so VSIM works on empty sets."""
    for key in (EPISODE_KEY, FACT_KEY):
        card = await r.execute_command("VCARD", key)
        if not card:
            seed = np.zeros(DIMS, dtype=np.float32)
            attr = json.dumps({"_seed": True, "content": "",
                               "language": "en", "domain": "general",
                               "access_count": 0, "ts": 0,
                               "importance": 0.0, "superseded_by": ""})
            await r.execute_command("VADD", key, "FP32", _blob(seed), "__seed__",
                                    "SETATTR", attr)
    # Tool capability vectorset
    from capability import ensure_tool_index
    await ensure_tool_index(r, DIMS)
    # Procedural memory vectorset
    card = await r.execute_command("VCARD", PROC_KEY)
    if not card:
        seed = np.zeros(DIMS, dtype=np.float32)
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
    except Exception:
        return []

    items: list[dict] = []
    i = 0
    while i + 2 < len(results):
        elem  = results[i]
        score = results[i + 1]
        raw   = results[i + 2]
        i += 3

        elem_str = elem.decode() if isinstance(elem, bytes) else elem
        if elem_str == "__seed__":
            continue

        try:
            attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        except Exception:
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

        if bump_heat:
            import asyncio
            from heat import bump_heat as _bump
            asyncio.create_task(_bump(r, vset_key, elem_str, attrs))

    return items


async def save_episode(
    r: aioredis.Redis,
    session_id: str,
    content: str,
    embedding: np.ndarray,
    language: str = "en",
    domain: str = "general",
) -> str:
    uid  = str(ULID())
    attr = json.dumps({
        "content":      content[:2000],
        "category":     "episode",
        "language":     language,
        "domain":       domain,
        "session_id":   session_id,
        "ts":           int(time.time() * 1000),
        "access_count": 0,
        "importance":   0.5,
        "superseded_by": "",
    })
    await r.execute_command("VADD", EPISODE_KEY, "FP32", _blob(embedding), uid,
                            "SETATTR", attr)
    return uid


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
) -> str:
    """
    Save a fact to the semantic memory vectorset.

    SimpleMem fields (v0.9.1):
      importance:    float 0-1, decays during consolidation (×0.9 per 90 days)
      topic:         SimpleMem topic phrase for the symbolic layer
      location:      SimpleMem location string for symbolic search
      superseded_by: empty string = active; set to winner's element ID on merge (soft-delete)
    """
    uid  = str(ULID())
    attr_dict: dict = {
        "content":       content[:500],
        "category":      category,
        "language":      language,
        "domain":        domain,
        "confidence":    confidence,
        "ts":            int(time.time() * 1000),
        "access_count":  0,
        "importance":    max(0.0, min(1.0, importance)),
        "superseded_by": "",   # empty = active; set to winner_id on consolidation
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

    await r.execute_command("VADD", FACT_KEY, "FP32", _blob(embedding), uid,
                            "SETATTR", json.dumps(attr_dict))
    return uid


async def soft_delete_fact(
    r: aioredis.Redis,
    element_id: str,
    superseded_by: str,
) -> None:
    """
    Mark a fact as superseded (soft-delete, SimpleMem consolidation pattern).

    Sets superseded_by = winner's element_id. knn_search() skips these by default.
    The entry is kept for audit/history but excluded from recall.

    Bug fix v0.9.2: removed the ×0.1 importance decay that was applied here.
    Phase 1 (Decay) already applies ×0.9 per 90 days; Phase 3 (Prune) removes
    entries that fall below 0.05. Applying a second ×0.1 during soft-delete
    caused double-decay: loser facts dropped from 0.5 → 0.045, landing just
    below the prune floor and being silently hard-removed the very next run.

    Redis 8 vectorset has no direct VGETATTR by element ID, so we use VGETATTR
    (available in Redis 8.0.1+) and fall back to the VSIM ELE scan.
    """
    try:
        # Redis 8.0.1+ supports VGETATTR element_id directly
        raw = await r.execute_command("VGETATTR", FACT_KEY, element_id)
        if raw:
            attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
            attrs["superseded_by"] = superseded_by
            await r.execute_command("VSETATTR", FACT_KEY, element_id, json.dumps(attrs))
            return
    except Exception:
        pass  # VGETATTR not available; fall through to VSIM scan

    # Fallback: VSIM ELE lookup (element-space traversal, Redis 8 native)
    try:
        scan = await r.execute_command(
            "VSIM", FACT_KEY, "ELE", element_id,
            "COUNT", 1, "WITHATTRIBS"
        )
        if scan and len(scan) >= 2:
            raw = scan[1]
            attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
            attrs["superseded_by"] = superseded_by
            await r.execute_command("VSETATTR", FACT_KEY, element_id, json.dumps(attrs))
            return
    except Exception:
        pass

    # Last resort: set superseded_by with minimal attrs to ensure
    # knn_search() will filter this entry out
    try:
        await r.execute_command(
            "VSETATTR", FACT_KEY, element_id,
            json.dumps({"superseded_by": superseded_by, "content": ""})
        )
    except Exception:
        pass


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
    """
    uid = str(ULID())
    attr = json.dumps({
        "task":          task[:300],
        "procedure":     procedure[:1000],
        "tools_used":    (tools_used or [])[:10],
        "domain":        domain,
        "language":      language,
        "success_count": 1,
        "ts":            int(time.time() * 1000),
    })
    await r.execute_command("VADD", PROC_KEY, "FP32", _blob(embedding), uid,
                            "SETATTR", attr)
    return uid


async def get_session_context(r: aioredis.Redis, session_id: str) -> str | None:
    if not session_id:
        return None
    val = await r.get(f"{SESSION_PRE}{session_id}:ctx")
    return val.decode() if val else None


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
