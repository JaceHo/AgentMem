"""Active working memory + prospective intention store (minimal).

Active focus is a small per-session hot buffer (goal / notes / updated_at).
Intentions are deferred cues (prospective memory) with pending/done/cancel.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

ACTIVE_PREFIX = "mem:active:"
INTENT_KEY = "mem:intentions"
INTENT_IDX = "mem:intentions:idx"


def _active_key(session_id: str) -> str:
    sid = (session_id or "default").strip()[:128] or "default"
    return f"{ACTIVE_PREFIX}{sid}"


async def get_active(redis, session_id: str) -> dict[str, Any]:
    raw = await redis.hgetall(_active_key(session_id))
    if not raw:
        return {"session_id": session_id or "default", "goal": "", "notes": "", "updated_at": 0}
    out = { (k.decode() if isinstance(k, bytes) else k):
            (v.decode() if isinstance(v, bytes) else v) for k, v in raw.items() }
    out["session_id"] = session_id or "default"
    try:
        out["updated_at"] = int(out.get("updated_at") or 0)
    except ValueError:
        out["updated_at"] = 0
    return out


async def set_active(redis, session_id: str, *, goal: str | None = None, notes: str | None = None) -> dict[str, Any]:
    key = _active_key(session_id)
    cur = await get_active(redis, session_id)
    if goal is not None:
        cur["goal"] = goal.strip()[:2000]
    if notes is not None:
        cur["notes"] = notes.strip()[:4000]
    cur["updated_at"] = int(time.time() * 1000)
    await redis.hset(key, mapping={
        "goal": cur.get("goal") or "",
        "notes": cur.get("notes") or "",
        "updated_at": str(cur["updated_at"]),
    })
    await redis.expire(key, 7 * 86400)
    return cur


async def list_intentions(redis, *, status: str | None = "pending", limit: int = 50) -> list[dict]:
    ids = await redis.zrevrange(INTENT_IDX, 0, max(limit * 3, 50) - 1)
    out: list[dict] = []
    for raw_id in ids:
        iid = raw_id.decode() if isinstance(raw_id, bytes) else raw_id
        blob = await redis.hget(INTENT_KEY, iid)
        if not blob:
            continue
        if isinstance(blob, bytes):
            blob = blob.decode()
        try:
            item = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if status and item.get("status") != status:
            continue
        out.append(item)
        if len(out) >= limit:
            break
    return out


async def add_intention(
    redis,
    *,
    text: str,
    cue: str = "",
    due_at: int | None = None,
    session_id: str = "",
) -> dict:
    iid = "int_" + uuid.uuid4().hex[:12]
    now = int(time.time() * 1000)
    item = {
        "id": iid,
        "text": text.strip()[:2000],
        "cue": (cue or "").strip()[:500],
        "due_at": int(due_at or 0),
        "session_id": (session_id or "")[:128],
        "status": "pending",
        "created_at": now,
        "updated_at": now,
    }
    await redis.hset(INTENT_KEY, iid, json.dumps(item, ensure_ascii=False))
    score = item["due_at"] or now
    await redis.zadd(INTENT_IDX, {iid: float(score)})
    return item


async def update_intention(redis, intention_id: str, *, status: str) -> dict | None:
    blob = await redis.hget(INTENT_KEY, intention_id)
    if not blob:
        return None
    if isinstance(blob, bytes):
        blob = blob.decode()
    try:
        item = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if status not in ("pending", "done", "cancel"):
        raise ValueError("status must be pending|done|cancel")
    item["status"] = status
    item["updated_at"] = int(time.time() * 1000)
    await redis.hset(INTENT_KEY, intention_id, json.dumps(item, ensure_ascii=False))
    if status != "pending":
        await redis.zrem(INTENT_IDX, intention_id)
    return item
