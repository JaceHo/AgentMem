"""Embedding-width integrity — one check for every vector set, fail closed.

Redis native vector sets fix their width on the first VADD and cannot be retyped.
A provider swap that changes width therefore does not degrade search: it rejects
every write, and the old rows stay readable at the wrong width. Before this module
the only check was a startup loop over three hard-coded keys (facts / episodes /
procedures) that, on mismatch, *force-switched the provider* instead of reporting
the problem — so `mem:tools`, `kb:chunks` and any set added later were never
checked, and a config error turned into a silent model downgrade.

Measured 2026-10-27, the concrete instance: aiserv's `/v1/embeddings` ignores the
requested model and always proxies Jetson `nomic-embed-text` (768), while the live
sets are 1024 (local qwen3). Anything that fell back to the gateway declared 1024
and wrote 768.

Rules:
  * the *index* is the source of truth for width,
  * the *model* is measured (one real vector), never declared,
  * disagreement refuses the write and fails the boot, with the repair command,
  * `AGENTMEM_DIM_POLICY=warn` exists only so a repair/read-only boot can start.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

from core import embedder

log = logging.getLogger("mem")

VECTORSET_TYPE = "vectorset"
REPAIR_CMD = "python scripts/reembed_vectorsets.py --vsets <key>"


def _s(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


async def scan_sets(redis) -> list[str]:
    """Every Redis key of type `vectorset`, whatever its name."""
    keys: list[str] = []
    cursor = 0
    while True:
        cursor, batch = await redis.scan(cursor, match="*", count=500)
        for raw in batch:
            key = _s(raw)
            if _s(await redis.type(key)) == VECTORSET_TYPE:
                keys.append(key)
        if cursor == 0:
            return sorted(keys)


async def snapshot(redis, dims: Optional[int] = None) -> Dict[str, Any]:
    """Width of the model vs width of every set on disk."""
    width = int(dims or embedder.measured_dims())
    sets: Dict[str, Dict[str, Any]] = {}
    for key in await scan_sets(redis):
        try:
            got = int(await redis.execute_command("VDIM", key) or 0)
            card = int(await redis.execute_command("VCARD", key) or 0)
        except Exception as exc:                                # noqa: BLE001
            sets[key] = {"dims": None, "card": None, "ok": False, "error": str(exc)[:120]}
            continue
        sets[key] = {"dims": got, "card": card, "ok": got == width}
    bad = {k: v for k, v in sets.items() if not v["ok"]}
    verified = embedder.dims_verified()
    return {
        "dims": width,
        "model": embedder.embed_signature(),
        "verified": verified,
        "sets": sets,
        "bad": bad,
        "ok": verified and not bad,
    }


def _explain(snap: Dict[str, Any]) -> str:
    width = snap["dims"]
    parts = []
    if not snap["verified"]:
        parts.append(f"the {snap['model']} width could not be measured (model unreachable?)")
    for key, info in snap["bad"].items():
        parts.append(f"{key} is {info['dims']}-d ({info['card']} rows) but the model is {width}-d")
    return "; ".join(parts) + f" — repair with `{REPAIR_CMD}`, or set AGENTMEM_DIM_POLICY=warn to boot read-only"


async def enforce(redis, policy: Optional[str] = None) -> Dict[str, Any]:
    """Boot-time gate. Raises on disagreement unless the policy says otherwise."""
    policy = (policy or os.environ.get("AGENTMEM_DIM_POLICY", "fail")).strip().lower()
    snap = await snapshot(redis)
    if snap["ok"]:
        log.info("[dim-guard] %d set(s) at %d dims — %s", len(snap["sets"]), snap["dims"],
                 snap["model"])
        return snap
    message = "[dim-guard] embedding width disagrees with the index: " + _explain(snap)
    if policy == "warn":
        log.error("%s (policy=warn: booting anyway, writes will fail)", message)
        return snap
    raise embedder.DimensionMismatch(message)


# Hot write paths ask "can this row land?" on every call; one round trip per set is
# wasted work at that rate, and the answer only changes when the provider does.
_CACHE_TTL_S = 30.0
_cache: Dict[str, Any] = {"at": 0.0, "dims": None, "bad": None}


async def assert_writeable(redis) -> Dict[str, Any]:
    """Cheap, cached gate for write endpoints (409 instead of a lying 200)."""
    now = time.monotonic()
    width = embedder.measured_dims()
    if _cache["bad"] is not None and _cache["dims"] == width and now - _cache["at"] < _CACHE_TTL_S:
        if _cache["bad"]:
            raise embedder.DimensionMismatch(_cache["bad"])
        return {"dims": width, "bad": {}}
    snap = await snapshot(redis, width)
    message = "" if snap["ok"] else "[dim-guard] " + _explain(snap)
    _cache.update({"at": now, "dims": width, "bad": message})
    if message:
        raise embedder.DimensionMismatch(message)
    return snap
