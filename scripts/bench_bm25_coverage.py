#!/usr/bin/env python3
"""Real-data measurement: BM25 keyword coverage gap in agentmem.

``core/search.py::populate_bm25_from_redis`` loads ONLY ``mem:facts`` into the
in-memory BM25 corpus (cap 5000).  Every ``bm25_index.add`` call site passes
``fact.content``.  So ``mem:episodes`` and ``mem:procedures`` have zero keyword
coverage — they are reachable only through VSIM cosine.

This script measures the size of that gap against the live Redis instance:

  1. Count vectors per tier.
  2. Build a BM25 corpus from real procedure + episode text.
  3. Run representative queries through fact-only BM25 vs all-tiers BM25.
  4. Report how many keyword-relevant items each misses.

Read-only: it never writes to Redis.

Run: .venv/bin/python scripts/bench_bm25_coverage.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import redis.asyncio as aioredis  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402

TIERS = {
    "mem:facts": "facts",
    "mem:episodes": "episodes",
    "mem:procedures": "procedures",
}

QUERIES = [
    "deploy aiserv with docker compose",
    "redis vectorset VSIM connection pool",
    "huber loss lightgbm rank slot",
    "fix cascade fallback health matrix",
    "ssh jace jetson inference throughput",
    "hooks prompt submit recall timeout",
    "bun instead of npm package manager",
    "hyperqwen accelerate qwen inference",
    "session diversity prevent one session dominating",
    "ebbinghaus forgetting curve confidence decay",
]


def tokenize(text: str) -> list[str]:
    return [t for t in text.lower().replace("_", " ").replace("-", " ").replace(":", " ").split() if len(t) > 1]


async def scan_tier(r: aioredis.Redis, key: str, max_count: int = 20000, dim: int = 768) -> list[dict]:
    """Read a vectorset's elements + attributes the way agentmem's own vscan does.

    Redis vectorset has no native VSCAN; the project emulates a full scan with a
    zero-vector VSIM (``core/search.py::vscan``).  Read-only.

    ``dim`` must match the dimension the vectorset was written with — the live
    instance is 768 (nomic-embed-text), while ``embedder.DIMS`` defaults to 384
    (local MiniLM), so the project's own ``vscan`` would also fail here unless
    the provider resolves to the 768-dim model.
    """
    import numpy as np

    from core.search import decode_attrs, decode_bytes

    try:
        card = int(await r.execute_command("VCARD", key))
    except Exception as e:  # pragma: no cover
        print(f"  ! VCARD {key} failed: {e}")
        return []
    if card <= 1:  # only the __seed__ placeholder
        return []
    zero = np.zeros(dim, dtype=np.float32).tobytes()
    try:
        results = await r.execute_command(
            "VSIM", key, "FP32", zero,
            "COUNT", min(card, max_count), "WITHSCORES", "WITHATTRIBS",
        )
    except Exception as e:  # pragma: no cover
        print(f"  ! VSIM {key} (dim={dim}) failed: {e}")
        return []
    out: list[dict] = []
    i = 0
    while i + 2 < len(results):
        elem = decode_bytes(results[i])
        raw = results[i + 2]
        i += 3
        if elem == "__seed__":
            continue
        try:
            attrs = decode_attrs(raw)
        except Exception:
            continue
        if attrs.get("_seed"):
            continue
        # Different tiers store their text under different attribute names:
        #   facts/episodes -> content,  procedures -> task + procedure
        content = (
            attrs.get("content")
            or attrs.get("text")
            or attrs.get("summary")
            or attrs.get("body")
            or " ".join(
                str(attrs[k]) for k in ("task", "procedure") if attrs.get(k)
            )
            or ""
        )
        if content:
            out.append({"element": elem, "content": str(content), "attrs": attrs, "tier": TIERS[key]})
    return out


async def main() -> int:
    r = aioredis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    try:
        print("Reading live Redis (read-only)...\n")
        corpus: dict[str, list[dict]] = {}
        for key in TIERS:
            items = await scan_tier(r, key)
            corpus[TIERS[key]] = items
            print(f"  {key:<18} {len(items):>6} text-bearing items")

        facts = corpus["facts"]
        all_items = facts + corpus["episodes"] + corpus["procedures"]
        print(f"\n  total text-bearing: {len(all_items)}")
        print(f"  BM25 corpus today : {len(facts)} facts  ({len(facts) / max(1, len(all_items)) * 100:.1f}% of corpus)")
        print(f"  keyword-invisible : {len(corpus['episodes']) + len(corpus['procedures'])} "
              f"episodes+procedures")

        # Cap check — the silent truncation.
        cap = 5000
        print(f"\n  populate_bm25_from_redis cap = {cap}")
        if len(facts) > cap:
            print(f"  !! TRUNCATION ACTIVE: {len(facts) - cap} facts are silently absent from BM25")
        else:
            print(f"  .. facts ({len(facts)}) still under the cap; headroom {cap - len(facts)}")

        # Build both corpora.
        facts_tok = [tokenize(d["content"]) for d in facts]
        all_tok = [tokenize(d["content"]) for d in all_items]
        bm25_facts = BM25Okapi(facts_tok) if facts_tok else None
        bm25_all = BM25Okapi(all_tok) if all_tok else None

        print(f"\n{'query':<52}{'fact-only':>11}{'all-tiers':>11}{'gained':>8}")
        print("-" * 82)
        tot_f = tot_a = 0
        for q in QUERIES:
            toks = tokenize(q)
            hits_f = set()
            hits_a = set()
            if bm25_facts:
                for i in bm25_facts.get_scores(toks).argsort()[::-1][:10]:
                    if bm25_facts.get_scores(toks)[i] > 0:
                        hits_f.add(facts[i]["element"])
            if bm25_all:
                sc = bm25_all.get_scores(toks)
                for i in sc.argsort()[::-1][:10]:
                    if sc[i] > 0:
                        hits_a.add(all_items[i]["element"])
            tot_f += len(hits_f)
            tot_a += len(hits_a)
            gained = len(hits_a - hits_f)
            print(f"{q[:50]:<52}{len(hits_f):>11}{len(hits_a):>11}{gained:>+8}")

        print("-" * 82)
        print(f"{'TOTAL':<52}{tot_f:>11}{tot_a:>11}{tot_a - tot_f:>+8}")
        if tot_f:
            print(f"\n  all-tiers BM25 surfaces {(tot_a - tot_f) / tot_f * 100:+.1f}% more keyword hits")
        print(f"  (tiers reached only by all-tiers: "
              f"{sum(1 for d in all_items if d['tier'] != 'facts')} of {len(all_items)} docs)")
        return 0
    finally:
        await r.aclose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
