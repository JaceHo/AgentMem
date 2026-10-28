"""Give every procedure the canonical task key and rebuild mem:proc_tasks around it.

Why: the dedup hash (mem:proc_tasks) was keyed on the raw task string, so ``Task: X.`` and ``task x``
were different procedures, and a dedup hit did not always write a mapping at all — measured 2026-10-28,
12 live rows were reachable only through the vector net, and one normalised duplicate pair was live.
``store.proc_task_key`` is now the write-side rule; this brings the rows written before it in line.

Safe by construction: the old hash is dumped before anything is written, attrs are only ADDED
(task_key), and no element is deleted. Re-running is a no-op.

Usage:
    python3 scripts/rekey_procedures.py --dump-dir ~/tmp/attic/proc-rekey-$(date +%F)
    python3 scripts/rekey_procedures.py --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import redis.asyncio as aioredis  # noqa: E402

from core import store  # noqa: E402


async def run(a) -> int:
    r = aioredis.from_url(store.REDIS_URL, decode_responses=False)
    try:
        card = int(await r.execute_command("VCARD", store.PROC_KEY) or 0)
        procs = await store.scan_all_procedures(r)
        old_hash = {k if isinstance(k, str) else k.decode(): v if isinstance(v, str) else v.decode()
                    for k, v in (await r.hgetall(store.PROC_TASK_KEY)).items()}
        print(f"procedures={card} scanned={len(procs)} hash_fields={len(old_hash)}")
        if a.dump_dir:
            d = Path(a.dump_dir).expanduser()
            d.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
            (d / f"proc_tasks-{stamp}.json").write_text(
                json.dumps(old_hash, indent=1, ensure_ascii=False), encoding="utf-8")
            (d / f"procedures-{stamp}.json").write_text(
                json.dumps({"procedures": procs}, indent=1, ensure_ascii=False), encoding="utf-8")
            print(f"dump → {d}")

        mapping: dict[str, str] = {}
        added_attr = collisions = 0
        for p in procs:                       # scan_all_procedures flattens attrs next to `uid`
            elem = p.get("uid")
            attrs = {k: v for k, v in p.items() if k != "uid"}
            task = attrs.get("task") or ""
            if not elem or not task:
                continue
            key = store.proc_task_key(task)
            if key in mapping and mapping[key] != elem:
                collisions += 1           # two rows normalise to one task: keep the first, report it
                continue
            mapping[key] = elem
            if not attrs.get("task_key"):
                attrs["task_key"] = key
                if not a.dry_run:
                    await r.execute_command("VSETATTR", store.PROC_KEY, elem,
                                            json.dumps(attrs, ensure_ascii=False))
                added_attr += 1

        if a.dry_run:
            print(f"dry-run: would write {len(mapping)} keys ({collisions} collisions), "
                  f"{added_attr} attrs")
            return 0
        pipe = r.pipeline(transaction=True)
        pipe.delete(store.PROC_TASK_KEY)
        if mapping:
            pipe.hset(store.PROC_TASK_KEY, mapping=mapping)
        await pipe.execute()
        print(f"wrote {len(mapping)} canonical keys ({collisions} collisions), "
              f"{added_attr} attrs gained task_key")
        return 0
    finally:
        await r.aclose()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", help="write the pre-change hash + rows here first")
    ap.add_argument("--dry-run", action="store_true")
    return asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
