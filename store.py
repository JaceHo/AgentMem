"""
Redis 8 native vectorset store.

Two vector sets:
  mem:episodes  — episodic memory (conversation turns)
  mem:facts     — semantic memory (distilled facts)

Session working memory in plain Redis strings (TTL-based).
User persona in Redis Hash (mem:persona).

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
SESSION_PRE  = "mem:session:"
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
                               "access_count": 0, "ts": 0})
            await r.execute_command("VADD", key, "FP32", _blob(seed), "__seed__",
                                    "SETATTR", attr)


async def knn_search(
    r: aioredis.Redis,
    vset_key: str,
    embedding: np.ndarray,
    k: int,
    filter_expr: Optional[str] = None,
    bump_heat: bool = False,
) -> list[dict]:
    """
    KNN search. Returns list of {content, category, score, attrs, _element}.
    filter_expr: e.g. '.language == "zh"' or '.domain == "trading"'
    bump_heat:   if True, async-increment access_count on each hit (non-blocking).
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
            # Fire-and-forget: don't await, don't block recall latency
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
) -> str:
    uid  = str(ULID())
    attr_dict = {
        "content":      content[:500],
        "category":     category,
        "language":     language,
        "domain":       domain,
        "confidence":   confidence,
        "ts":           int(time.time() * 1000),
        "access_count": 0,
    }
    if keywords:
        attr_dict["keywords"] = keywords[:10]
    if persons:
        attr_dict["persons"] = persons[:5]
    if entities:
        attr_dict["entities"] = entities[:5]
    await r.execute_command("VADD", FACT_KEY, "FP32", _blob(embedding), uid,
                            "SETATTR", json.dumps(attr_dict))
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
