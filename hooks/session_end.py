#!/usr/bin/env python3
"""session-end hook — compresses session, triggers consolidation."""

import json
import os
import sys

try:
    import httpx
except ImportError:
    sys.exit(0)

REST_URL = os.getenv("AGENTMEMORY_URL", "http://localhost:18800")
SECRET = os.getenv("AGENTMEMORY_SECRET", "")


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

    try:
        httpx.post(
            f"{REST_URL}/agentmemory/session/end",
            json={"sessionId": session_id, "cwd": data.get("cwd") or os.getcwd()},
            headers=headers,
            timeout=3.0,
        )
    except Exception:
        pass

    try:
        httpx.post(
            f"{REST_URL}/agentmemory/consolidate-pipeline",
            json={"sessionId": session_id},
            headers=headers,
            timeout=3.0,
        )
    except Exception:
        pass

    try:
        httpx.post(
            f"{REST_URL}/agentmemory/claude-bridge/sync",
            json={"sessionId": session_id, "cwd": data.get("cwd") or os.getcwd()},
            headers=headers,
            timeout=3.0,
        )
    except Exception:
        pass


if __name__ == "__main__":
    main()
