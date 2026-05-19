#!/usr/bin/env python3
"""post-tool-use hook — observes tool usage and stores to memory."""

import json
import os
import sys

try:
    import httpx
except ImportError:
    sys.exit(0)

REST_URL = os.getenv("AGENTMEMORY_URL", "http://localhost:18800")
SECRET = os.getenv("AGENTMEMORY_SECRET", "")


def _truncate(value, max_len=8000):
    if isinstance(value, str) and len(value) > max_len:
        return value[:max_len] + "\n[...truncated]"
    if isinstance(value, dict) or isinstance(value, list):
        s = json.dumps(value, ensure_ascii=False)
        if len(s) > max_len:
            return s[:max_len] + "...[truncated]"
    return value


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        return

    if os.getenv("AGENTMEMORY_SDK_CHILD") == "1":
        return
    if isinstance(data, dict) and data.get("entrypoint") == "sdk-ts":
        return

    session_id = data.get("session_id") or "unknown"

    headers = {"Content-Type": "application/json"}
    if SECRET:
        headers["Authorization"] = f"Bearer {SECRET}"

    payload = {
        "hookType": "post_tool_use",
        "sessionId": session_id,
        "project": data.get("cwd") or os.getcwd(),
        "cwd": data.get("cwd") or os.getcwd(),
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "data": {
            "tool_name": data.get("tool_name", ""),
            "tool_input": data.get("tool_input", ""),
            "tool_output": _truncate(data.get("tool_output", "")),
        },
    }

    try:
        httpx.post(
            f"{REST_URL}/agentmemory/observe",
            json=payload,
            headers=headers,
            timeout=3.0,
        )
    except Exception:
        pass


if __name__ == "__main__":
    main()
