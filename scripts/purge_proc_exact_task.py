#!/usr/bin/env python3
"""OPEN #17: hard-dedup mem:procedures by exact `task` string.

Per cluster: keep the row with the newest `ts` (tie-break: highest
success_count, then element id); sum success_count onto the keeper; hard-delete
losers via DELETE /procedures/{id} so side indexes stay consistent; re-HSET
mem:proc_tasks → keeper (remove_element HDELs the shared task key).

Usage:
  python3 scripts/purge_proc_exact_task.py --dry-run
  python3 scripts/purge_proc_exact_task.py --execute
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

import redis

REDIS_URL = "redis://localhost:6379/0"
PROC_KEY = "mem:procedures"
IDX_KEY = "mem:idx:procedures"
PROC_TASK_KEY = "mem:proc_tasks"
BASE = "http://127.0.0.1:18800"
CHUNK = 400


def _decode(v):
    if isinstance(v, bytes):
        return v.decode()
    return v


def load_all(r: redis.Redis) -> list[dict]:
    card = int(r.execute_command("VCARD", PROC_KEY))
    ids = [_decode(x) for x in r.smembers(IDX_KEY)]
    if len(ids) < card * 0.9:
        print(
            f"WARN: shadow {len(ids)}/{card} (<90%); run POST /shadow-index/backfill first",
            file=sys.stderr,
        )
    out: list[dict] = []
    for i in range(0, len(ids), CHUNK):
        part = ids[i : i + CHUNK]
        pipe = r.pipeline(transaction=False)
        for eid in part:
            pipe.execute_command("VGETATTR", PROC_KEY, eid)
        raws = pipe.execute()
        for eid, raw in zip(part, raws):
            if not raw:
                continue
            try:
                attrs = json.loads(_decode(raw))
            except Exception:
                continue
            if attrs.get("_seed") or eid == "__seed__":
                continue
            task = (attrs.get("task") or "").strip()
            if not task:
                continue
            out.append(
                {
                    "id": eid,
                    "task": task,
                    "ts": float(attrs.get("ts") or 0),
                    "success_count": int(attrs.get("success_count") or 0),
                    "attrs": attrs,
                }
            )
    return out


def pick_keeper(rows: list[dict]) -> dict:
    return max(rows, key=lambda x: (x["ts"], x["success_count"], x["id"]))


def plan(entries: list[dict]) -> list[dict]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        by_task[e["task"]].append(e)
    clusters = []
    for task, rows in sorted(by_task.items(), key=lambda kv: -len(kv[1])):
        keeper = pick_keeper(rows)
        losers = [x for x in rows if x["id"] != keeper["id"]]
        total_sc = sum(x["success_count"] for x in rows)
        clusters.append(
            {
                "task": task,
                "n": len(rows),
                "keeper": keeper["id"],
                "keeper_ts": keeper["ts"],
                "keeper_sc": keeper["success_count"],
                "sum_sc": total_sc,
                "losers": [x["id"] for x in losers],
            }
        )
    return clusters


def bump_keeper(r: redis.Redis, cluster: dict, keeper_attrs: dict) -> None:
    """Re-point mem:proc_tasks and fold success_count onto the survivor."""
    task = cluster["task"][:300]
    eid = cluster["keeper"]
    attrs = dict(keeper_attrs)
    attrs["success_count"] = int(cluster["sum_sc"])
    attrs["dedup_cluster_n"] = int(cluster["n"])
    attrs["dedup_purged_at"] = time.time()
    r.execute_command("VSETATTR", PROC_KEY, eid, json.dumps(attrs, ensure_ascii=False))
    r.hset(PROC_TASK_KEY, task, eid)


async def direct_purge(losers: list[str], concurrency: int = 8) -> tuple[int, list]:
    """Call core.store.remove_element in-process — no HTTP stampede."""
    import asyncio
    import sys as _sys

    root = Path(__file__).resolve().parents[1]
    if str(root) not in _sys.path:
        _sys.path.insert(0, str(root))
    import redis.asyncio as aioredis
    from core import store as mem_store

    ar = aioredis.from_url(REDIS_URL, decode_responses=False)
    sem = asyncio.Semaphore(concurrency)
    deleted = 0
    errors: list = []
    lock = asyncio.Lock()

    async def one(eid: str) -> None:
        nonlocal deleted
        async with sem:
            try:
                attrs = await mem_store.get_attrs(ar, mem_store.PROC_KEY, eid)
                if not attrs:
                    async with lock:
                        deleted += 1
                    return
                await mem_store.remove_element(ar, mem_store.PROC_KEY, eid)
                async with lock:
                    deleted += 1
                    if deleted % 500 == 0:
                        print(f"  deleted {deleted}/{len(losers)}…", flush=True)
            except Exception as e:
                async with lock:
                    errors.append({"id": eid, "error": str(e)})

    await asyncio.gather(*(one(eid) for eid in losers))
    await ar.aclose()
    return deleted, errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", default=False)
    ap.add_argument("--execute", action="store_true", default=False)
    ap.add_argument("--workers", type=int, default=8, help="in-process remove_element concurrency")
    ap.add_argument(
        "--report",
        default=str(Path.home() / "tmp/artifacts/2026-10-24-proc-purge-dryrun.json"),
    )
    args = ap.parse_args()
    if not args.dry_run and not args.execute:
        args.dry_run = True

    r = redis.Redis.from_url(REDIS_URL, decode_responses=False)
    t0 = time.time()
    card_before = int(r.execute_command("VCARD", PROC_KEY))
    entries = load_all(r)
    clusters = plan(entries)
    multi = [c for c in clusters if c["n"] > 1]
    delete_n = sum(c["n"] - 1 for c in multi)
    keep_n = len(clusters)

    report = {
        "mode": "execute" if args.execute else "dry-run",
        "vcard_before": card_before,
        "enumerated": len(entries),
        "distinct_tasks": keep_n,
        "clusters_with_dupes": len(multi),
        "would_delete": delete_n,
        "would_keep": keep_n,
        "top_clusters": [
            {
                "n": c["n"],
                "task": c["task"][:120],
                "keeper": c["keeper"],
                "sum_sc": c["sum_sc"],
            }
            for c in multi[:25]
        ],
        "elapsed_s": round(time.time() - t0, 2),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.dry_run and not args.execute:
        print(f"\nDry-run only. Report → {args.report}")
        print(f"Re-run with --execute to delete {delete_n} losers, keep {keep_n}.")
        return 0

    import asyncio

    all_losers = [lid for c in multi for lid in c["losers"]]
    print(
        f"Deleting {len(all_losers)} via in-process remove_element "
        f"(concurrency={args.workers})…",
        flush=True,
    )
    deleted, errors = asyncio.run(direct_purge(all_losers, concurrency=args.workers))

    keeper_map = {e["id"]: e["attrs"] for e in entries}
    # Only bump keepers that still exist
    for c in clusters:
        raw = r.execute_command("VGETATTR", PROC_KEY, c["keeper"])
        if not raw:
            errors.append({"id": c["keeper"], "error": "keeper missing after purge"})
            continue
        bump_keeper(r, c, keeper_map[c["keeper"]])

    card_after = int(r.execute_command("VCARD", PROC_KEY))
    result = {
        "deleted": deleted,
        "errors": len(errors),
        "error_samples": errors[:10],
        "vcard_after": card_after,
        "proc_tasks_hlen": int(r.hlen(PROC_TASK_KEY)),
        "elapsed_s": round(time.time() - t0, 2),
    }
    out = Path(args.report).with_name("2026-10-24-proc-purge-result.json")
    out.write_text(json.dumps({**report, **result}, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Result → {out}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
