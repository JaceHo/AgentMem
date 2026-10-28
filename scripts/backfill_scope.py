"""Stamp existing unscoped rows with a track, and copy the global persona over.

Why: ``/recall`` with ``user_id`` or ``agent_id`` filters at the index, and a
scoped read never widens to the unscoped corpus. Rows written before scoping
existed carry no ids, so a caller that has always been user ``jace`` would see
an empty memory until the history is stamped. This script is that one-time
stamp; it never touches a row that already names a track.

What it does:
  - facts/episodes: for each element whose attrs carry neither ``user_id`` nor
    ``agent_id``, merge the requested ids and ``VSETATTR`` the result;
  - persona: copy ``mem:persona`` field-for-field into ``mem:persona:user:<id>``
    without re-running sanitisation, so the profile reads the same before and
    after, only under the new key.

Usage:
    python3 scripts/backfill_scope.py --user-id jace --dry-run
    python3 scripts/backfill_scope.py --user-id jace
    python3 scripts/backfill_scope.py --agent-id raven-code --tiers facts
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import redis.asyncio as aioredis  # noqa: E402

from core import store  # noqa: E402
from core.persona import PERSONA_KEY  # noqa: E402
from core.scope import strict_scope  # noqa: E402

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
_PINNED_KEY = "mem:pinned:session_summary"

TIERS = {
    "facts": store.FACT_KEY,
    "episodes": store.EPISODE_KEY,
    "procedures": store.PROC_KEY,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--user-id", default="", help="track to stamp rows with")
    p.add_argument("--agent-id", default="", help="agent track (XOR with --user-id)")
    p.add_argument("--tiers", default="facts,episodes",
                   help="comma-separated subset of: facts, episodes")
    p.add_argument("--dry-run", action="store_true", help="report, change nothing")
    p.add_argument("--no-persona", action="store_true",
                   help="skip copying mem:persona to the per-track key")
    p.add_argument("--redis-url", default=REDIS_URL)
    return p.parse_args()


async def stamp_tier(r: aioredis.Redis, vset: str, scope, dry_run: bool) -> dict:
    stats = {"seen": 0, "already_scoped": 0, "stamped": 0, "failed": 0}
    rows = await store.enumerate_elements(r, vset)
    if not rows:
        stats["failed"] = -1  # no shadow index: enumeration is unavailable
        return stats
    extra = scope.attrs()
    for row in rows:
        stats["seen"] += 1
        attrs = row.get("attrs") or {}
        if attrs.get("user_id") or attrs.get("agent_id"):
            stats["already_scoped"] += 1
            continue
        attrs.update(extra)
        if dry_run:
            stats["stamped"] += 1
            continue
        try:
            await r.execute_command(
                "VSETATTR", vset, row["element"], json.dumps(attrs)
            )
            stats["stamped"] += 1
        except Exception as exc:  # element vanished mid-run, or bad JSON
            stats["failed"] += 1
            if stats["failed"] <= 5:
                print(f"  ! {vset}/{row['element']}: {exc}", file=sys.stderr)
    return stats


async def copy_persona(r: aioredis.Redis, scope, dry_run: bool) -> tuple[int, str]:
    # The profile is a person's, not an agent's; an agent-track run leaves it alone.
    if scope.is_global or not scope.user_id:
        return 0, ""
    dst = scope.persona_key()
    raw = await r.hgetall(PERSONA_KEY) or {}
    if not raw:
        return 0, dst
    if dry_run:
        return len(raw), dst
    await r.hset(dst, mapping=raw)
    return len(raw), dst


async def copy_pinned_summary(r: aioredis.Redis, scope, dry_run: bool) -> str:
    """Move the handoff pin to this track's key, or report why there was none."""
    if scope.is_global or not scope.user_id:
        return ""
    dst = _PINNED_KEY + ":" + scope.track_suffix()
    raw = await r.get(_PINNED_KEY)
    if not raw:
        return ""
    if not dry_run:
        await r.set(dst, raw)
    return dst


async def main() -> int:
    args = parse_args()
    try:
        scope = strict_scope(args.user_id, args.agent_id)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if scope.is_global:
        print("error: pass --user-id or --agent-id; nothing to stamp", file=sys.stderr)
        return 2

    wanted = [t.strip() for t in args.tiers.split(",") if t.strip()]
    unknown = [t for t in wanted if t not in TIERS]
    if unknown:
        print(f"error: unknown tier(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    r = aioredis.from_url(args.redis_url, decode_responses=True)
    try:
        print(f"scope: {scope.attrs()}  dry_run={args.dry_run}")
        for tier in wanted:
            vset = TIERS[tier]
            stats = await stamp_tier(r, vset, scope, args.dry_run)
            if stats["failed"] == -1:
                print(f"{vset}: shadow index empty, cannot enumerate "
                      f"(run POST /admin/shadow-index/backfill first)")
                continue
            print(f"{vset}: seen={stats['seen']} "
                  f"already_scoped={stats['already_scoped']} "
                  f"stamped={stats['stamped']} failed={stats['failed']}")
        if not args.no_persona:
            n, dst = await copy_persona(r, scope, args.dry_run)
            if dst:
                print(f"{PERSONA_KEY} -> {dst}: {n} field(s)")
            pin_dst = await copy_pinned_summary(r, scope, args.dry_run)
            if pin_dst:
                print(f"{_PINNED_KEY} -> {pin_dst}")
    finally:
        await r.aclose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
