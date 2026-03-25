#!/usr/bin/env python3
"""One-off: Remove 'Always send a report' fact from mem:facts vectorset.
Uses zero-vector scan (no embedder) to avoid network/proxy issues."""
import asyncio
import json
import sys

import numpy as np

from store import FACT_KEY, DIMS, get_client


async def main():
    r = await get_client()
    seed = np.zeros(DIMS, dtype=np.float32)
    try:
        results = await r.execute_command(
            "VSIM", FACT_KEY, "FP32", seed.tobytes(),
            "COUNT", 500, "WITHSCORES", "WITHATTRIBS"
        )
    except Exception as e:
        print(f"VSIM failed: {e}")
        await r.aclose()
        return 1

    removed = 0
    i = 0
    while i + 2 < len(results):
        elem, score, raw = results[i], results[i + 1], results[i + 2]
        i += 3
        elem_str = elem.decode() if isinstance(elem, bytes) else elem
        if elem_str == "__seed__":
            continue
        try:
            attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        except Exception:
            continue
        content = attrs.get("content", "")
        if "Always send a report" in content or "always send a report" in content.lower():
            await r.execute_command("VREM", FACT_KEY, elem_str)
            removed += 1
            print(f"Removed: {content[:80]}...")
    await r.aclose()
    print(f"Removed {removed} fact(s)")
    return 0
