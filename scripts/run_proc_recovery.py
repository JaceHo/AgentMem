"""One-shot: run the procedure recovery against the live instance and time it."""
import asyncio
import sys
import time

sys.path.insert(0, "/Users/jace/code/agentmem")

import redis.asyncio as aioredis

from core import store as mem_store


async def main() -> int:
    r = aioredis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    before = {
        "vcard": int(await r.execute_command("VCARD", mem_store.PROC_KEY)),
        "shadow": int(await r.scard(mem_store.idx_key(mem_store.PROC_KEY))),
    }
    t0 = time.perf_counter()
    stats = await mem_store.recover_procedures_from_reverse_index(r)
    elapsed = time.perf_counter() - t0
    after = {
        "vcard": int(await r.execute_command("VCARD", mem_store.PROC_KEY)),
        "shadow": int(await r.scard(mem_store.idx_key(mem_store.PROC_KEY))),
    }
    print("before:", before)
    print("stats :", stats)
    print("after :", after)
    print(f"elapsed: {elapsed:.2f}s")

    # Second run must be a no-op — this is the steady-state boot cost.
    t1 = time.perf_counter()
    stats2 = await mem_store.recover_procedures_from_reverse_index(r)
    print("second run:", {k: stats2[k] for k in
                          ("ids_collected", "live", "newly_registered",
                           "stale", "stale_refs_pruned", "shadow_after", "coverage")})
    print(f"second elapsed: {time.perf_counter() - t1:.2f}s")

    # Enumeration now reaches everything.
    t2 = time.perf_counter()
    items = await mem_store.enumerate_elements(r, mem_store.PROC_KEY)
    with_task = [i for i in items if (i.get("attrs") or {}).get("task")]
    print(f"enumerate_elements: {len(items)} elements, {len(with_task)} with task "
          f"in {time.perf_counter() - t2:.2f}s")
    await r.aclose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
