#!/usr/bin/env python3
"""pre-tool-use hook — enriches tool context with relevant memories."""

import json
import os
import sys

try:
    import httpx
except ImportError:
    sys.exit(0)

REST_URL = os.getenv("AGENTMEMORY_URL", "http://localhost:18800")
SECRET = os.getenv("AGENTMEMORY_SECRET", "")

ENRICH_TOOLS = {"Edit", "Write", "Read", "Glob", "Grep", "MultiEdit"}


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        return

    if os.getenv("AGENTMEMORY_SDK_CHILD") == "1":
        return
    if isinstance(data, dict) and data.get("entrypoint") == "sdk-ts":
        return

    tool_name = data.get("tool_name", "")
    if tool_name not in ENRICH_TOOLS:
        return

    session_id = data.get("session_id") or "unknown"

    headers = {"Content-Type": "application/json"}
    if SECRET:
        headers["Authorization"] = f"Bearer {SECRET}"

    tool_input = data.get("tool_input", {})
    files = []
    if isinstance(tool_input, dict):
        for key in ("file_path", "path", "glob", "pattern"):
            val = tool_input.get(key, "")
            if val and isinstance(val, str):
                files.append(val)

    payload = {
        "sessionId": session_id,
        "project": data.get("cwd") or os.getcwd(),
        "cwd": data.get("cwd") or os.getcwd(),
        "files": files,
        "query": f"{tool_name} " + " ".join(files[:3]),
    }

    try:
        resp = httpx.post(
            f"{REST_URL}/agentmemory/enrich",
            json=payload,
            headers=headers,
            timeout=1.5,
        )
        if resp.status_code == 200:
            result = resp.json()
            context = result.get("context")
            if context:
                sys.stdout.write(context)
    except Exception:
        pass


if __name__ == "__main__":
    main()
