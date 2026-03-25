#!/usr/bin/env python3
"""
digest-metaclaw.py — Ingest MetaClaw SKILL.md files into AgentMem procedural memory.

Reads all memory_data/skills/*/SKILL.md files from a MetaClaw skills directory
and stores each as a searchable procedure in AgentMem's mem:procedures tier.

Once ingested, the top-2 most relevant skills will automatically appear in
Claude Code's prependContext on every prompt (when include_procedures=True).

Usage:
    python3 digest-metaclaw.py                          # default skills dir
    python3 digest-metaclaw.py --skills-dir /path/to/metaclaw/memory_data/skills
    python3 digest-metaclaw.py --reset                  # re-ingest all skills

MetaClaw skill format (SKILL.md):
    ---
    name: debug-systematically
    description: Use when diagnosing a bug...
    category: coding        # optional, defaults to "general"
    ---
    # Debug Systematically
    ...markdown content...

Maps to AgentMem:
    task       = "[skill:{name}] {description}"
    procedure  = markdown body content
    domain     = category (coding→software-dev, research→research, etc.)
"""

import argparse
import glob
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
AGENTMEM_URL  = "http://localhost:18800"
DEFAULT_SKILLS_DIRS = [
    # MetaClaw repo default (relative to this file)
    Path(__file__).parent.parent / "MetaClaw" / "memory_data" / "skills",
    # /tmp clone (from 'git clone ... /tmp/MetaClaw')
    Path("/tmp/MetaClaw/memory_data/skills"),
    # Typical MetaClaw config dir
    Path.home() / ".metaclaw" / "skills",
    Path.home() / "Library" / "Application Support" / "metaclaw" / "skills",
]
STATE_FILE = Path(__file__).parent / ".digest_metaclaw_state.json"

# Domain mapping: MetaClaw category → AgentMem domain
DOMAIN_MAP = {
    "coding":        "software-dev",
    "research":      "research",
    "data_analysis": "analysis",
    "security":      "devops",
    "communication": "general",
    "automation":    "devops",
    "agentic":       "software-dev",
    "productivity":  "general",
    "common_mistakes": "general",
    "general":       "general",
}


# ── SKILL.md parser ───────────────────────────────────────────────────────────
def parse_skill_md(path: Path) -> dict | None:
    """Parse a SKILL.md file into {name, description, category, content}."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        print(f"  [warn] could not read {path}: {e}")
        return None

    if not raw.startswith("---"):
        return None

    end_idx = raw.find("\n---", 3)
    if end_idx == -1:
        return None

    fm_text = raw[3:end_idx].strip()
    body    = raw[end_idx + 4:].strip()

    fm: dict[str, str] = {}
    for line in fm_text.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            fm[key.strip()] = val.strip()

    name        = fm.get("name", "").strip()
    description = fm.get("description", "").strip()
    category    = fm.get("category", "general").strip()

    if not name or not description:
        print(f"  [warn] skipping {path} — missing name or description")
        return None

    return {"name": name, "description": description, "category": category, "content": body}


# ── State file helpers ─────────────────────────────────────────────────────────
def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"ingested": {}}   # ingested: {name: ts}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ── AgentMem API call ─────────────────────────────────────────────────────────
def store_procedure(skill: dict) -> bool:
    """POST one skill to AgentMem /store-procedure. Returns True on success."""
    name     = skill["name"]
    desc     = skill["description"]
    category = skill["category"]
    content  = skill["content"]

    task      = f"[skill:{name}] {desc}"
    procedure = content if content else desc
    domain    = DOMAIN_MAP.get(category, "general")

    payload = json.dumps({
        "task":       task,
        "procedure":  procedure,
        "domain":     domain,
        "tools_used": [],
        "session_id": "",
    }).encode()

    req = urllib.request.Request(
        f"{AGENTMEM_URL}/store-procedure",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"  [error] store-procedure failed for '{name}': {e}")
        return False


# ── Health check ──────────────────────────────────────────────────────────────
def check_health() -> bool:
    try:
        with urllib.request.urlopen(f"{AGENTMEM_URL}/health", timeout=3) as resp:
            data = json.loads(resp.read())
            return data.get("status") == "ok"
    except Exception:
        return False


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global AGENTMEM_URL
    parser = argparse.ArgumentParser(
        description="Ingest MetaClaw SKILL.md files into AgentMem procedural memory."
    )
    parser.add_argument(
        "--skills-dir", metavar="PATH",
        help="Path to MetaClaw skills directory (contains subdirs with SKILL.md files). "
             "Defaults to auto-detect.",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Clear state and re-ingest all skills (even already-ingested ones).",
    )
    parser.add_argument(
        "--url", default=AGENTMEM_URL,
        help=f"AgentMem service URL (default: {AGENTMEM_URL})",
    )
    args = parser.parse_args()

    AGENTMEM_URL = args.url.rstrip("/")

    # ── Find skills directory ──────────────────────────────────────────────────
    if args.skills_dir:
        skills_dir = Path(args.skills_dir)
        if not skills_dir.is_dir():
            print(f"[error] skills directory not found: {skills_dir}", file=sys.stderr)
            sys.exit(1)
    else:
        skills_dir = None
        for candidate in DEFAULT_SKILLS_DIRS:
            if candidate.is_dir():
                skills_dir = candidate
                break
        if skills_dir is None:
            print("[error] Could not auto-detect MetaClaw skills directory.")
            print("  Tried:")
            for c in DEFAULT_SKILLS_DIRS:
                print(f"    {c}")
            print("  Use --skills-dir PATH to specify it.")
            sys.exit(1)

    print(f"=== digest-metaclaw.py ===")
    print(f"Skills dir:  {skills_dir}")
    print(f"AgentMem:    {AGENTMEM_URL}")

    # ── Health check ──────────────────────────────────────────────────────────
    if not check_health():
        print(f"\n[error] AgentMem not reachable at {AGENTMEM_URL}/health")
        print("  Start it with: bash start.sh")
        sys.exit(1)
    print("Service:     healthy\n")

    # ── Load state ────────────────────────────────────────────────────────────
    state = load_state() if not args.reset else {"ingested": {}}
    already = state.get("ingested", {})

    # ── Discover SKILL.md files ───────────────────────────────────────────────
    patterns = sorted(glob.glob(str(skills_dir / "*" / "SKILL.md")))
    if not patterns:
        print(f"[warn] No SKILL.md files found in {skills_dir}")
        sys.exit(0)
    print(f"Found {len(patterns)} SKILL.md files\n")

    # ── Ingest each skill ─────────────────────────────────────────────────────
    ingested = 0
    skipped  = 0
    failed   = 0

    for path_str in patterns:
        skill = parse_skill_md(Path(path_str))
        if skill is None:
            failed += 1
            continue

        name = skill["name"]
        if name in already:
            skipped += 1
            continue

        ok = store_procedure(skill)
        if ok:
            already[name] = int(time.time() * 1000)
            ingested += 1
            print(f"  ✓ [{skill['category']:16s}] {name}")
        else:
            failed += 1
            print(f"  ✗ [{skill['category']:16s}] {name} (failed)")

        time.sleep(0.05)   # small delay to avoid overwhelming embedder

    # ── Save state ────────────────────────────────────────────────────────────
    state["ingested"] = already
    save_state(state)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nIngested: {ingested}  Skipped: {skipped}  Failed: {failed}")
    print(f"Total skills in AgentMem: {len(already)}")

    if ingested > 0:
        print(f"\nDone. Skills are now searchable via /recall-procedures")
        print(f"and will appear in prependContext when include_procedures=true.")


if __name__ == "__main__":
    main()
