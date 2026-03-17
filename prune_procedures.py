"""
Procedure deduplication + pruning script.

Clusters the 2422-entry procedure vectorset by cosine similarity,
keeps the best representative per cluster (highest success_count),
and soft-deletes duplicates by marking them superseded_by.

Target: reduce ~2422 → ~200 high-signal procedures.

Usage:
    cd ~/code/agentmem && source venv/bin/activate
    python prune_procedures.py [--dry-run] [--threshold 0.92]
"""
import argparse
import asyncio
import json
import sys
import time

import numpy as np
import redis.asyncio as aioredis

# Use local MiniLM embedder (same model as agentmem service)
sys.path.insert(0, "/Users/jace/code/agentmem")
import embedder as _embedder

REDIS_URL = "redis://localhost:6379"
PROC_KEY  = "mem:procedures"


def embed(text: str) -> np.ndarray:
    """Embed text using local MiniLM model (384-dim)."""
    return _embedder.encode(text)


async def load_all_procedures(r):
    """Fetch all non-seed, non-superseded procedures with their attributes."""
    # Use VSIM with a zero vector to enumerate all entries
    dims = 384
    zero = np.zeros(dims, dtype=np.float32).tobytes()
    raw = await r.execute_command(
        "VSIM", PROC_KEY, "FP32", zero,
        "COUNT", 5000, "WITHSCORES", "WITHATTRIBS"
    )
    entries = []
    i = 0
    while i + 2 < len(raw):
        elem  = raw[i].decode()   if isinstance(raw[i],   bytes) else raw[i]
        score = float(raw[i+1])
        attrs_raw = raw[i+2]
        try:
            attrs = json.loads(attrs_raw.decode() if isinstance(attrs_raw, bytes) else attrs_raw)
        except Exception:
            attrs = {}
        i += 3
        if elem == "__seed__" or attrs.get("_seed"):
            continue
        if attrs.get("superseded_by", ""):
            continue  # already soft-deleted
        entries.append({"id": elem, "score": score, "attrs": attrs})
    return entries


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",   action="store_true", help="Print what would be pruned, don't write")
    parser.add_argument("--threshold", type=float, default=0.92, help="Cosine similarity threshold for duplicates")
    args = parser.parse_args()

    r = aioredis.from_url(REDIS_URL, decode_responses=False)
    t0 = time.time()

    print(f"Loading procedures (threshold={args.threshold})...")
    entries = await load_all_procedures(r)
    print(f"  Found {len(entries)} active (non-superseded) procedures")

    # Re-embed task text using local MiniLM (VEMB not available in this Redis build)
    print("  Embedding procedure tasks (MiniLM)...")
    vecs = {}
    for idx, e in enumerate(entries):
        task_text = e["attrs"].get("task", "") or e["attrs"].get("procedure", "")
        if not task_text:
            continue
        vecs[e["id"]] = embed(task_text[:256])
        if (idx + 1) % 100 == 0:
            print(f"    {idx+1}/{len(entries)} embedded...")

    print(f"  Embedded {len(vecs)} procedures")

    # Greedy clustering: O(N^2) but N≈2400 so ~2.8M comparisons — manageable
    ids = list(vecs.keys())
    clusters: list[list[str]] = []   # each cluster = list of elem IDs
    assigned = set()

    for i, eid in enumerate(ids):
        if eid in assigned:
            continue
        cluster = [eid]
        assigned.add(eid)
        for j in range(i+1, len(ids)):
            other = ids[j]
            if other in assigned:
                continue
            sim = cosine_sim(vecs[eid], vecs[other])
            if sim >= args.threshold:
                cluster.append(other)
                assigned.add(other)
        clusters.append(cluster)

    print(f"  {len(clusters)} clusters from {len(ids)} procedures")

    # For each cluster, keep the entry with the highest success_count (or first if tie)
    # Soft-delete the rest by setting superseded_by = winner_id
    keep_count  = 0
    prune_count = 0

    for cluster in clusters:
        if len(cluster) == 1:
            keep_count += 1
            continue

        # Find the best entry: highest success_count, then highest importance/score
        def rank(eid):
            a = next((e["attrs"] for e in entries if e["id"] == eid), {})
            return (a.get("success_count", 0), a.get("importance", 0.0))

        winner = max(cluster, key=rank)
        losers = [x for x in cluster if x != winner]
        keep_count  += 1
        prune_count += len(losers)

        winner_task = next((e["attrs"].get("task","")[:60] for e in entries if e["id"] == winner), "")
        print(f"  cluster({len(cluster)}): keep [{winner_task}] prune {len(losers)}")

        if not args.dry_run:
            for loser in losers:
                try:
                    raw = await r.execute_command("VGETATTR", PROC_KEY, loser)
                    if raw:
                        attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
                        attrs["superseded_by"] = winner
                        await r.execute_command("VSETATTR", PROC_KEY, loser, json.dumps(attrs))
                except Exception as ex:
                    print(f"    WARN: could not prune {loser}: {ex}")

    elapsed = time.time() - t0
    action = "Would prune" if args.dry_run else "Pruned"
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Kept:    {keep_count}")
    print(f"  {action}: {prune_count}")
    print(f"  Reduction: {len(entries)} → {keep_count} ({100*(1-keep_count/max(len(entries),1)):.0f}% pruned)")

    await r.aclose()


if __name__ == "__main__":
    asyncio.run(main())
