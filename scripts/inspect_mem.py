"""
AgentMem inspector CLI — query your own memory from the terminal.

Usage:
    python inspect_mem.py <query>
    python inspect_mem.py "aiserv routing"
    python inspect_mem.py "jace preferences" --limit 10
    python inspect_mem.py --stats
"""
import argparse
import json
import sys

import httpx

BASE = "http://127.0.0.1:18800"


def stats():
    r = httpx.get(f"{BASE}/stats", timeout=5)
    d = r.json()
    print("── AgentMem Stats ──────────────────────────────")
    for k, v in d.items():
        print(f"  {k:30s}: {v}")
    print()
    # Store health
    rate = d.get("store_success_rate")
    if rate is not None:
        health = "OK" if rate >= 0.8 else ("WARN" if rate >= 0.5 else "CRIT")
        print(f"  Store health: {health} ({rate*100:.0f}% success, avg {d.get('store_avg_ms')}ms)")
    else:
        print("  Store health: no data yet (restarted recently?)")


def recall(query: str, limit: int, procedures: bool):
    r = httpx.post(f"{BASE}/recall", timeout=15, json={
        "query":               query,
        "memory_limit_number": limit,
        "include_tools":       True,
        "include_procedures":  procedures,
    })
    d = r.json()
    ctx = d.get("prependContext", "")
    lat = d.get("latency_ms", "?")
    print(f"── Query: {query!r}  ({lat}ms) ─────────────────")
    if ctx:
        print(ctx)
    else:
        print("  (no memories matched)")


def main():
    p = argparse.ArgumentParser(description="Query AgentMem from terminal")
    p.add_argument("query", nargs="*", help="Search query")
    p.add_argument("--limit", type=int, default=8, help="Max memories to return")
    p.add_argument("--no-procs", action="store_true", help="Skip procedures")
    p.add_argument("--stats", action="store_true", help="Show stats only")
    args = p.parse_args()

    if args.stats or not args.query:
        stats()
        if not args.query:
            return

    query = " ".join(args.query)
    recall(query, args.limit, not args.no_procs)


if __name__ == "__main__":
    main()
