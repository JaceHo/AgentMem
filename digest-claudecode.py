#!/usr/bin/env python3
"""
digest-claudecode.py — Bootstrap AgentMem from Claude Code project session histories.

Scans ~/.claude/projects/*/  JSONL session files, extracts human prompts
and assistant text responses, and bulk-ingests them into AgentMem via POST /store.

Processed sessions are tracked in .digest-claudecode-state.json to enable
incremental re-runs (only new sessions are ingested).

Usage:
    python3 digest-claudecode.py [--projects-dir PATH] [--api URL]
                                 [--min-turns N] [--project SLUG]
                                 [--dry-run] [--reset] [--list-projects]
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
DEFAULT_PROJECTS_DIR = Path.home() / ".claude/projects"
DEFAULT_API          = "http://localhost:18800"
STATE_FILE           = Path(__file__).parent / ".digest-claudecode-state.json"
MIN_TURNS_DEFAULT    = 2      # skip sessions with fewer human+assistant turns
RATE_LIMIT_MS        = 200    # ms between /store calls

# Strip injected memory context from user messages
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


def extract_text_from_content(content) -> str:
    """Flatten Claude content (string or list of blocks) into plain text.

    Keeps only 'text' type blocks; skips tool_use, tool_result, thinking.
    """
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype == "text":
                parts.append(block.get("text", ""))
            # Skip: tool_use, tool_result, thinking, image
        return "\n".join(p for p in parts if p).strip()
    return ""


def extract_messages(jsonl_path: Path, project_slug: str) -> tuple[str, list[dict]]:
    """Parse a Claude Code session JSONL file.

    Returns (session_id, messages) where messages is a list of
    {role, content} dicts suitable for /store.

    Only real human turns are extracted — tool result callbacks
    (identified by toolUseResult key) are skipped since they are
    just shell/file output, not user intent.
    """
    session_uuid = jsonl_path.stem
    session_id = f"claudecode:{project_slug}:{session_uuid}"
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

                event_type = obj.get("type", "")

                # ── User turn (real human prompt) ──────────────────────────
                if event_type == "user":
                    # Skip tool result callbacks — these are command outputs
                    # fed back to the model, not actual user messages.
                    if obj.get("toolUseResult") or obj.get("sourceToolAssistantUUID"):
                        continue

                    msg = obj.get("message", {})
                    content = msg.get("content", "")

                    # Skip if content is a list of tool_result blocks only
                    if isinstance(content, list):
                        has_real_text = any(
                            b.get("type") == "text" for b in content
                            if isinstance(b, dict)
                        )
                        if not has_real_text:
                            continue

                    text = extract_text_from_content(content)
                    # Strip injected memory context (avoid circular echo)
                    text = MEMORY_BLOCK_RE.sub("", text).strip()

                    if not text:
                        continue

                    messages.append({"role": "user", "content": text})

                # ── Assistant turn (text responses only) ───────────────────
                elif event_type == "assistant":
                    msg = obj.get("message", {})
                    content = msg.get("content", [])
                    text = extract_text_from_content(content)
                    if not text:
                        continue
                    messages.append({"role": "assistant", "content": text})

                # Skip: file-history-snapshot, progress, summary, etc.

    except OSError as exc:
        print(f"  [warn] Cannot read {jsonl_path.name}: {exc}", file=sys.stderr)

    return session_id, messages


def store(api_base: str, session_id: str, messages: list[dict]) -> bool:
    """POST /store and return True on success."""
    payload = json.dumps(
        {"messages": messages, "session_id": session_id}
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


def check_health(api_base: str) -> bool:
    try:
        with urllib.request.urlopen(f"{api_base}/health", timeout=5) as resp:
            data = json.loads(resp.read())
            return data.get("status") == "ok"
    except Exception:
        return False


def iter_sessions(projects_dir: Path, project_filter: str | None):
    """Yield (project_slug, jsonl_path) for all main session files.

    Subagents (sessions inside a UUID subfolder) are excluded.
    """
    for project_dir in sorted(projects_dir.iterdir()):
        if not project_dir.is_dir():
            continue
        slug = project_dir.name
        if project_filter and project_filter not in slug:
            continue

        for p in sorted(project_dir.glob("*.jsonl")):
            # Skip if inside a subagent subfolder (parent would be a UUID dir)
            if p.parent.name != project_dir.name:
                continue
            yield slug, p


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--projects-dir", type=Path, default=DEFAULT_PROJECTS_DIR,
                        help=f"Claude Code projects directory (default: {DEFAULT_PROJECTS_DIR})")
    parser.add_argument("--api", default=DEFAULT_API,
                        help=f"AgentMem API base URL (default: {DEFAULT_API})")
    parser.add_argument("--min-turns", type=int, default=MIN_TURNS_DEFAULT,
                        help=f"Min human+assistant turns to ingest (default: {MIN_TURNS_DEFAULT})")
    parser.add_argument("--project", metavar="SLUG",
                        help="Only process projects whose slug contains this string "
                             "(e.g. 'agentmem', 'ios2and')")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse and report counts without calling /store")
    parser.add_argument("--reset", action="store_true",
                        help="Clear state and re-ingest all sessions")
    parser.add_argument("--list-projects", action="store_true",
                        help="List all project directories and exit")
    args = parser.parse_args()

    # ── list projects mode ────────────────────────────────────────────────────
    if args.list_projects:
        print(f"Projects in {args.projects_dir}:\n")
        for d in sorted(args.projects_dir.iterdir()):
            if d.is_dir():
                count = len(list(d.glob("*.jsonl")))
                print(f"  {d.name:<50}  {count:>4} sessions")
        return

    # ── preflight ─────────────────────────────────────────────────────────────
    if not args.dry_run:
        print(f"Checking AgentMem health at {args.api} …", end=" ", flush=True)
        if not check_health(args.api):
            print("FAIL\nService not reachable. Start it with: bash start.sh",
                  file=sys.stderr)
            sys.exit(1)
        print("OK")

    if not args.projects_dir.is_dir():
        print(f"Projects directory not found: {args.projects_dir}", file=sys.stderr)
        sys.exit(1)

    state = load_state()
    if args.reset:
        state = {"processed": {}}
        print("State reset — all sessions will be re-ingested.")

    # ── discover sessions ─────────────────────────────────────────────────────
    all_sessions   = list(iter_sessions(args.projects_dir, args.project))
    already_done   = set(state["processed"].keys())
    # State key = project_slug + session uuid (filename stem)
    pending = [
        (slug, p) for slug, p in all_sessions
        if f"{slug}/{p.stem}" not in already_done
    ]

    total_projects = len({slug for slug, _ in all_sessions})
    print(f"\nProjects: {total_projects}  |  "
          f"Sessions: {len(all_sessions)} total, "
          f"{len(already_done)} already ingested, "
          f"{len(pending)} pending\n")

    if not pending:
        print("Nothing to do — all sessions already ingested.")
        print("Use --reset to re-ingest everything.")
        return

    # ── ingest ────────────────────────────────────────────────────────────────
    ingested = skipped_short = skipped_empty = errors = 0
    total_turns = 0
    prev_slug = None

    for i, (slug, path) in enumerate(pending, 1):
        if slug != prev_slug:
            print(f"\n  [{slug}]")
            prev_slug = slug

        session_id, messages = extract_messages(path, slug)
        turns = len(messages)
        state_key = f"{slug}/{path.stem}"
        label = f"    [{i:>4}/{len(pending)}] {path.name[:36]:<36}  {turns:>4} turns"

        if not messages:
            print(f"  empty {label}")
            skipped_empty += 1
            state["processed"][state_key] = {
                "skipped": "empty",
                "at": datetime.now(timezone.utc).isoformat(),
            }
            continue

        if turns < args.min_turns:
            print(f"  skip  {label}  (< {args.min_turns})")
            skipped_short += 1
            state["processed"][state_key] = {
                "skipped": "too_short", "turns": turns,
                "at": datetime.now(timezone.utc).isoformat(),
            }
            continue

        if args.dry_run:
            print(f"  dry   {label}")
            ingested += 1
            total_turns += turns
            continue

        ok = store(args.api, session_id, messages)
        if ok:
            print(f"  ok    {label}")
            ingested += 1
            total_turns += turns
            state["processed"][state_key] = {
                "ok": True, "turns": turns,
                "at": datetime.now(timezone.utc).isoformat(),
            }
        else:
            print(f"  FAIL  {label}")
            errors += 1

        if i % 10 == 0:
            save_state(state)

        time.sleep(RATE_LIMIT_MS / 1000)

    # ── summary ───────────────────────────────────────────────────────────────
    save_state(state)
    verb = "Would ingest" if args.dry_run else "Ingested"
    print(f"\n{'─'*60}")
    print(f"{verb}:      {ingested} sessions / {total_turns} turns")
    print(f"Skipped (short): {skipped_short}")
    print(f"Skipped (empty): {skipped_empty}")
    print(f"Errors:          {errors}")
    if not args.dry_run and ingested:
        print(f"\nState saved to {STATE_FILE}")
        print("Run consolidation to merge duplicate facts:")
        print("  curl -X POST http://localhost:18800/consolidate/sync")


if __name__ == "__main__":
    main()
