"""
Memory backfill: import OpenClaw workspace memory into memos-local Redis.

Sources:
  workspace/memory/*.md         — daily event summaries
  workspace/memory/warm-store.md — project paths + operating principles
  workspace/memory/cold-store.md — tool paths + config
  workspace/memory/self-programming-log.md — system health + decisions

Strategy: split each file by ## sections, post each as a synthetic episode.
The /store endpoint handles dedup, scene detection, summarization, fact extraction.
Session JSONL files skipped — all cron-automated or MemOS-injected, no real user content.
"""

import asyncio
import glob
import json
import re
import sys
from pathlib import Path

import httpx

BASE_URL   = "http://127.0.0.1:18790"
MEMORY_DIR = Path.home() / ".openclaw/workspace/memory"

SKIP_PREFIXES = (
    "## Long-Term Memory", "## Recent Relevant Episodes",
    "## Current Session Context", "## User Profile",
)

# Source files with their semantic role used as query context
SOURCES = [
    # (filename, query_hint, priority)
    ("warm-store.md",          "project paths, node management, operating principles", 1),
    ("cold-store.md",          "tool paths, search config, historical setup",          2),
    ("self-programming-log.md","system health, token costs, cron improvements",        2),
    ("2026-02-21.md",          "events and tasks on 2026-02-21",                       3),
    ("2026-02-22.md",          "events and tasks on 2026-02-22",                       3),
    ("2026-02-23.md",          "events and tasks on 2026-02-23",                       3),
    ("2026-02-25.md",          "events and tasks on 2026-02-25",                       3),
    ("2026-02-26.md",          "events and tasks on 2026-02-26",                       3),
    ("T11-self-improve-report.md", "self-improvement session T11 full report",         4),
    ("message-queue.md",       "pending message queue state",                          5),
]


def _split_sections(text: str) -> list[str]:
    """Split markdown by ## headings; keep sections >= 80 chars."""
    sections = re.split(r"\n(?=## )", text.strip())
    return [s.strip() for s in sections if len(s.strip()) >= 80]


def _is_injected(text: str) -> bool:
    return any(text.strip().startswith(p) for p in SKIP_PREFIXES)


def build_batches() -> list[dict]:
    batches = []
    for filename, query_hint, priority in SOURCES:
        path = MEMORY_DIR / filename
        if not path.exists():
            continue

        text = path.read_text(encoding="utf-8")
        sections = _split_sections(text)

        for section in sections:
            if _is_injected(section):
                continue
            # Use first line of section as the topic
            first_line = section.split("\n")[0].lstrip("#").strip()[:120]
            batches.append({
                "source":   filename,
                "priority": priority,
                "messages": [
                    {"role": "user",      "content": f"[{query_hint}] {first_line}"},
                    {"role": "assistant", "content": section[:2000]},
                ],
            })

    # Sort by priority so highest-value content is stored first (dedup hits later dupes)
    batches.sort(key=lambda x: x["priority"])
    return batches


async def run(dry_run: bool = False):
    print("=== memos-local backfill (workspace memory) ===\n")

    async with httpx.AsyncClient(timeout=15) as c:
        h = (await c.get(f"{BASE_URL}/health")).json()
        if h.get("status") != "ok":
            print(f"ERROR: service not healthy: {h}")
            return

        stats_before = (await c.get(f"{BASE_URL}/stats")).json()
        print(f"Before: episodes={stats_before['episodes']}  "
              f"facts={stats_before['facts']}  "
              f"persona={stats_before['persona_fields']}\n")

        batches = build_batches()
        print(f"Total sections to import: {len(batches)}")
        for src in dict.fromkeys(b["source"] for b in batches):
            n = sum(1 for b in batches if b["source"] == src)
            print(f"  {n:3d}  {src}")
        print()

        if dry_run:
            print("[DRY RUN — not writing to Redis]\n")
            for b in batches[:8]:
                print(f"  [{b['source']}] {b['messages'][0]['content'][:100]}")
            print(f"  ... and {len(batches)-8} more")
            return

        ok = 0
        for i, batch in enumerate(batches):
            try:
                r = await c.post(f"{BASE_URL}/store", json={
                    "messages":   batch["messages"],
                    "session_id": f"backfill:{batch['source']}",
                })
                if r.status_code == 200:
                    ok += 1
            except Exception as e:
                print(f"  WARN {batch['source']}: {e}")

            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(batches)}] queued…")
            await asyncio.sleep(0.08)   # let background worker breathe

        print(f"\nQueued {ok}/{len(batches)} sections.")
        print("Waiting for background workers to finish (~45s)…")
        await asyncio.sleep(45)

        stats_after = (await c.get(f"{BASE_URL}/stats")).json()
        ep_d   = stats_after["episodes"] - stats_before["episodes"]
        fact_d = stats_after["facts"]    - stats_before["facts"]
        per_d  = stats_after["persona_fields"] - stats_before["persona_fields"]

        print(f"\n{'='*50}")
        print(f"Before: ep={stats_before['episodes']}  facts={stats_before['facts']}  persona={stats_before['persona_fields']}")
        print(f"After:  ep={stats_after['episodes']}  facts={stats_after['facts']}  persona={stats_after['persona_fields']}")
        print(f"Added:  +{ep_d} episodes  +{fact_d} facts  +{per_d} persona fields")
        print("=== Done ===")


if __name__ == "__main__":
    dry = "--dry" in sys.argv
    asyncio.run(run(dry_run=dry))
