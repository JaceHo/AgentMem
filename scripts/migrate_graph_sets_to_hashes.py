"""
One-shot migration: legacy Set-based graph edges (v0.9) → typed Hash edges (v1.1).

For each `mem:graph:{slug}` key that is still a Redis Set, replace it with a Hash
mapping {neighbour_slug: edge_json}, where edge_json carries the default
related_to edge. The per-key conversion runs as an atomic Lua script (TYPE check
+ SMEMBERS + DEL + HSET), so it is safe against concurrent graph writes and
idempotent — re-running skips keys that are already Hashes.

Usage:
    python scripts/migrate_graph_sets_to_hashes.py            # dry-run (report only)
    python scripts/migrate_graph_sets_to_hashes.py --apply    # perform migration
"""

from __future__ import annotations

import os
import sys
import json

import redis

GRAPH_PREFIX = "mem:graph:"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

DEFAULT_EDGE = json.dumps(
    {"type": "related_to", "confidence": 0.5, "source_count": 1},
    separators=(",", ":"),
)

# Atomic per-key convert: only acts on Sets, preserves all members as hash fields.
CONVERT_LUA = """
local k = KEYS[1]
if redis.call('TYPE', k).ok ~= 'set' then return -1 end
local members = redis.call('SMEMBERS', k)
if #members == 0 then redis.call('DEL', k); return 0 end
redis.call('DEL', k)
for i = 1, #members do
  redis.call('HSET', k, members[i], ARGV[1])
end
return #members
"""


def main() -> int:
    apply = "--apply" in sys.argv
    r = redis.Redis.from_url(REDIS_URL)
    convert = r.register_script(CONVERT_LUA)

    set_keys: list[str] = []
    hash_keys = 0
    for raw in r.scan_iter(match=f"{GRAPH_PREFIX}*", count=500):
        key = raw.decode() if isinstance(raw, bytes) else raw
        t = r.type(key).decode()
        if t == "set":
            set_keys.append(key)
        elif t == "hash":
            hash_keys += 1

    print(f"Found {len(set_keys)} set keys, {hash_keys} hash keys under {GRAPH_PREFIX}*")
    if not apply:
        print("DRY-RUN. Re-run with --apply to migrate.")
        for k in set_keys[:10]:
            print(f"  would convert: {k} ({r.scard(k)} members)")
        if len(set_keys) > 10:
            print(f"  ... and {len(set_keys) - 10} more")
        return 0

    migrated = 0
    edges = 0
    for key in set_keys:
        n = convert(keys=[key], args=[DEFAULT_EDGE])
        if n >= 0:
            migrated += 1
            edges += int(n)
    print(f"Migrated {migrated} keys ({edges} edges) set → hash.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
