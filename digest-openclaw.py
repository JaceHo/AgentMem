#!/usr/bin/env python3
"""
digest-openclaw.py — Bootstrap AgentMem from OpenClaw chat history.

Scans ~/.openclaw/agents/main/sessions/*.jsonl, extracts user/assistant
turns, and bulk-ingests them into AgentMem via POST /store.

Processed sessions are tracked in .digest-state.json to enable
incremental re-runs (only new sessions are ingested).

Usage:
    python3 digest-openclaw.py [--sessions-dir PATH] [--api URL]
                               [--min-turns N] [--dry-run] [--reset]
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_SESSIONS_DIR = Path.home() / ".openclaw/agents/main/sessions"
DEFAULT_API           = "http://localhost:18800"
STATE_FILE            = Path(__file__).parent / ".digest-state.json"
MIN_TURNS_DEFAULT     = 2      # skip sessions shorter than this many turns
RATE_LIMIT_MS         = 200    # ms between /store calls (be polite to Redis)
WINDOW_SIZE           = 4      # messages per /store call — matches _do_store clean[-4:]

# Strip injected memory context before storing — avoids circular echoes
MEMORY_BLOCK_RE = re.compile(
    r"<cross_session_memory>.*?</cross_session_memory>",
    re.DOTALL,
)

# ── helpers ───────────────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"processed": {}}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def extract_messages(jsonl_path: Path) -> tuple[str, list[dict]]:
    """Parse an OpenClaw JSONL session file.

    Returns (session_id, messages) where messages is a list of
    {role, content} dicts suitable for /store.
    """
    session_id = jsonl_path.stem  # UUID filename without .jsonl
    messages: list[dict] = []

    try:
        with jsonl_path.open(encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                # Session header — grab the stored session id if present
                if obj.get("type") == "session" and "id" in obj:
                    session_id = obj["id"]
                    continue

                # Only care about message events
                if obj.get("type") != "message":
                    continue

                msg = obj.get("message", {})
                role = msg.get("role", "")
                if role not in ("user", "assistant"):
                    continue

                content = msg.get("content", "")
                # Flatten list content blocks → plain text
                if isinstance(content, list):
                    parts = []
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                parts.append(block.get("text", ""))
                            elif block.get("type") == "tool_result":
                                # skip tool results — low signal for memory
                                pass
                        elif isinstance(block, str):
                            parts.append(block)
                    content = "\n".join(p for p in parts if p).strip()
                else:
                    content = str(content).strip()

                # Strip injected memory blocks (avoid circular ingestion)
                content = MEMORY_BLOCK_RE.sub("", content).strip()

                if not content:
                    continue

                messages.append({"role": role, "content": content})

    except OSError as exc:
        print(f"  [warn] Cannot read {jsonl_path.name}: {exc}", file=sys.stderr)

    return session_id, messages


def store(api_base: str, session_id: str, messages: list[dict]) -> bool:
    """POST /store and return True on success."""
    payload = json.dumps(
        {"messages": messages, "session_id": f"openclaw:{session_id}"}
    ).encode()
    req = urllib.request.Request(
        f"{api_base}/store",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status < 300
    except Exception as exc:
        print(f"  [error] /store failed: {exc}", file=sys.stderr)
        return False


def store_windowed(api_base: str, session_id: str, messages: list[dict],
                   window: int = WINDOW_SIZE) -> int:
    """Chunk messages into windows of `window` and POST each chunk.

    Returns the number of successful /store calls.  Each chunk is sent as
    "openclaw:{session_id}:wN" so AgentMem can deduplicate across re-runs.

    Why: _do_store in main.py takes clean[-4:] (last 4 messages).  Sending a
    full session in one call discards everything except the final 4 messages.
    Chunking ensures every window is fully processed.
    """
    ok_count = 0
    for chunk_idx, start in enumerate(range(0, len(messages), window)):
        chunk = messages[start: start + window]
        if len(chunk) < 2:   # need at least 1 user+assistant pair
            continue
        chunk_id = f"openclaw:{session_id}:w{chunk_idx}"
        if store(api_base, chunk_id, chunk):
            ok_count += 1
        time.sleep(RATE_LIMIT_MS / 1000)
    return ok_count


def check_health(api_base: str) -> bool:
    try:
        with urllib.request.urlopen(f"{api_base}/health", timeout=5) as resp:
            data = json.loads(resp.read())
            return data.get("status") == "ok"
    except Exception:
        return False


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sessions-dir", type=Path, default=DEFAULT_SESSIONS_DIR,
                        help=f"Path to OpenClaw sessions directory (default: {DEFAULT_SESSIONS_DIR})")
    parser.add_argument("--api", default=DEFAULT_API,
                        help=f"AgentMem API base URL (default: {DEFAULT_API})")
    parser.add_argument("--min-turns", type=int, default=MIN_TURNS_DEFAULT,
                        help=f"Minimum user+assistant turns to ingest (default: {MIN_TURNS_DEFAULT})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse sessions and report counts without calling /store")
    parser.add_argument("--reset", action="store_true",
                        help="Clear processed-session state and re-ingest everything")
    args = parser.parse_args()

    # ── preflight ─────────────────────────────────────────────────────────────
    if not args.dry_run:
        print(f"Checking AgentMem health at {args.api} …", end=" ", flush=True)
        if not check_health(args.api):
            print("FAIL\nService not reachable. Start it with: bash start.sh", file=sys.stderr)
            sys.exit(1)
        print("OK")

    if not args.sessions_dir.is_dir():
        print(f"Sessions directory not found: {args.sessions_dir}", file=sys.stderr)
        sys.exit(1)

    state = load_state()
    if args.reset:
        state = {"processed": {}}
        print("State reset — all sessions will be re-ingested.")

    # ── discover sessions ─────────────────────────────────────────────────────
    jsonl_files = sorted(
        p for p in args.sessions_dir.iterdir()
        if p.suffix == ".jsonl"
        and ".deleted" not in p.name
        and ".bak"     not in p.name
        and ".lock"    not in p.name
        and ".reset"   not in p.name
        and ".tmp"     not in p.name
    )

    already_done = set(state["processed"].keys())
    pending = [p for p in jsonl_files if p.stem not in already_done]

    print(f"\nSessions: {len(jsonl_files)} total, "
          f"{len(already_done)} already ingested, "
          f"{len(pending)} pending\n")

    if not pending:
        print("Nothing to do — all sessions already ingested.")
        print("Use --reset to re-ingest everything.")
        return

    # ── ingest ────────────────────────────────────────────────────────────────
    ingested = skipped_short = skipped_empty = errors = 0
    total_turns = total_windows = 0

    for i, path in enumerate(pending, 1):
        session_id, messages = extract_messages(path)

        turns = len(messages)
        windows = max(0, (turns + WINDOW_SIZE - 1) // WINDOW_SIZE)
        label = (f"[{i:>4}/{len(pending)}] {path.name[:36]:<36} "
                 f"{turns:>4} turns  {windows:>3}w")

        if turns < args.min_turns:
            print(f"  skip  {label}  (< {args.min_turns} turns)")
            skipped_short += 1
            state["processed"][path.stem] = {
                "skipped": "too_short", "turns": turns,
                "at": datetime.now(timezone.utc).isoformat()
            }
            continue

        if not messages:
            print(f"  empty {label}")
            skipped_empty += 1
            state["processed"][path.stem] = {
                "skipped": "empty", "at": datetime.now(timezone.utc).isoformat()
            }
            continue

        if args.dry_run:
            print(f"  dry   {label}")
            ingested += 1
            total_turns += turns
            total_windows += windows
            continue

        ok_wins = store_windowed(args.api, session_id, messages)
        if ok_wins > 0:
            print(f"  ok    {label}  ({ok_wins}/{windows} windows stored)")
            ingested += 1
            total_turns += turns
            total_windows += ok_wins
            state["processed"][path.stem] = {
                "ok": True, "turns": turns, "windows": ok_wins,
                "at": datetime.now(timezone.utc).isoformat()
            }
        else:
            print(f"  FAIL  {label}")
            errors += 1

        # Save state periodically (every 10 sessions)
        if i % 10 == 0:
            save_state(state)

    # ── summary ───────────────────────────────────────────────────────────────
    save_state(state)
    verb = "Would ingest" if args.dry_run else "Ingested"
    print(f"\n{'─'*60}")
    print(f"{verb}:      {ingested} sessions / {total_turns} turns / {total_windows} windows")
    print(f"Skipped (short): {skipped_short}")
    print(f"Skipped (empty): {skipped_empty}")
    print(f"Errors:          {errors}")
    if not args.dry_run and ingested:
        print(f"\nState saved to {STATE_FILE}")
        print("Run `curl -X POST http://localhost:18800/consolidate/sync` to merge duplicates.")


if __name__ == "__main__":
    main()
