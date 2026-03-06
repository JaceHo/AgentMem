"""
Session Compactor  v1.0
=======================
Trims bloated OpenClaw session .jsonl files to prevent slow TTFT.

Strategy:
  1. Extract old messages to memory service (/store) so facts survive.
  2. Keep header events (session, model_change, etc.) intact.
  3. Keep last KEEP_LAST_N_TURNS user+assistant message pairs.
  4. Write back a trimmed .jsonl.

Usage:
  python3 compact.py [--dry-run] [--sessions-dir PATH] [--threshold-kb 200]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

MEMORY_BASE = "http://127.0.0.1:18800"
SESSIONS_DIR = Path.home() / ".openclaw" / "agents" / "main" / "sessions"
SESSIONS_JSON = SESSIONS_DIR / "sessions.json"
THRESHOLD_KB = 200          # compact if session > this size
KEEP_LAST_N_TURNS = 30      # keep this many user+assistant pairs
HARD_LIMIT_KB = 300         # force-compact even if pairs < KEEP_LAST_N_TURNS (tool-heavy sessions)
NON_MESSAGE_TYPES = {"session", "model_change", "thinking_level_change", "tool_change"}


def load_sessions_index():
    if not SESSIONS_JSON.exists():
        return {}
    with open(SESSIONS_JSON) as f:
        return json.load(f)


def compact_session(session_path: Path, dry_run: bool = False) -> dict:
    """Compact a single session file. Returns stats dict."""
    lines = session_path.read_text().splitlines()
    entries = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    if not entries:
        return {"skipped": True, "reason": "empty"}

    # Separate header events from messages
    headers = []
    messages = []
    for entry in entries:
        if entry.get("type") in NON_MESSAGE_TYPES:
            headers.append(entry)
        elif entry.get("type") == "message":
            messages.append(entry)
        else:
            # custom events, tool calls etc — keep with headers
            headers.append(entry)

    # Build user+assistant pairs
    pairs = []
    pending = None
    for msg in messages:
        role = msg.get("message", {}).get("role")
        if role == "user":
            if pending:
                pairs.append([pending])  # unpaired user
            pending = msg
        elif role == "assistant":
            if pending:
                pairs.append([pending, msg])
                pending = None
            else:
                pairs.append([msg])  # unpaired assistant
    if pending:
        pairs.append([pending])

    total_pairs = len(pairs)
    if total_pairs <= 1:
        return {"skipped": True, "reason": "only 1 pair, cannot trim"}

    file_size_kb = session_path.stat().st_size // 1024
    if total_pairs <= KEEP_LAST_N_TURNS:
        if file_size_kb < HARD_LIMIT_KB:
            return {"skipped": True, "reason": f"only {total_pairs} pairs, under limit"}
        # Large file with few pairs (tool-heavy session) — keep last half
        keep_n = max(1, total_pairs // 2)
    else:
        keep_n = KEEP_LAST_N_TURNS

    old_pairs = pairs[: total_pairs - keep_n]
    new_pairs = pairs[total_pairs - keep_n :]

    # Ship old messages to memory service so facts are preserved
    old_messages = []
    for pair in old_pairs:
        for entry in pair:
            msg = entry.get("message", {})
            old_messages.append({"role": msg.get("role", "user"), "content": str(msg.get("content", ""))})

    if old_messages and not dry_run:
        try:
            resp = httpx.post(
                f"{MEMORY_BASE}/store",
                json={"messages": old_messages, "session_id": f"compact:{session_path.stem}"},
                timeout=10,
            )
            stored = resp.status_code == 200
        except Exception as e:
            print(f"  [compact] store warning: {e}", file=sys.stderr)
            stored = False
    else:
        stored = False

    if dry_run:
        print(f"  [dry-run] would trim {len(old_pairs)} old pairs -> {keep_n} kept, store={len(old_messages)} msgs")
        return {"dry_run": True, "old_pairs": len(old_pairs), "kept": len(new_pairs)}

    # Rebuild trimmed file: headers + recent messages
    kept_entries = list(headers)
    for pair in new_pairs:
        kept_entries.extend(pair)

    # Write back (backup old first)
    backup_path = session_path.with_suffix(f".jsonl.bak.{int(time.time())}")
    session_path.rename(backup_path)
    with open(session_path, "w") as f:
        for entry in kept_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    old_size = backup_path.stat().st_size
    new_size = session_path.stat().st_size
    print(
        f"  [compact] trimmed: {old_size//1024}KB -> {new_size//1024}KB "
        f"({len(old_pairs)} pairs dropped, {len(new_pairs)} kept, stored={stored})"
    )
    return {"old_kb": old_size // 1024, "new_kb": new_size // 1024, "dropped": len(old_pairs)}


def run(dry_run: bool, sessions_dir: Path, threshold_kb: int):
    index = load_sessions_index()
    compacted = 0
    skipped = 0

    # Collect active session IDs from index
    active_ids = set()
    for k, v in index.items():
        sid = v["sessionId"] if isinstance(v, dict) else v
        active_ids.add(sid)

    # Scan active sessions above threshold
    for sid in active_ids:
        path = sessions_dir / f"{sid}.jsonl"
        if not path.exists():
            continue
        size_kb = path.stat().st_size // 1024
        if size_kb < threshold_kb:
            continue
        print(f"[compact] {path.name[:36]} {size_kb}KB -- compacting...")
        result = compact_session(path, dry_run=dry_run)
        if result.get("skipped"):
            print(f"  [compact] skipped: {result.get('reason')}")
            skipped += 1
        else:
            compacted += 1

    print(f"[compact] Done. compacted={compacted} skipped={skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compact OpenClaw session files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--sessions-dir", default=str(SESSIONS_DIR))
    parser.add_argument("--threshold-kb", type=int, default=THRESHOLD_KB)
    args = parser.parse_args()
    run(
        dry_run=args.dry_run,
        sessions_dir=Path(args.sessions_dir),
        threshold_kb=args.threshold_kb,
    )
