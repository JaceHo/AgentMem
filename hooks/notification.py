#!/usr/bin/env python3
"""notification hook — observes notification events."""

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
    message = data.get("message", data.get("notification", ""))

    if not message:
        return

    headers = {"Content-Type": "application/json"}
    if SECRET:
        headers["Authorization"] = f"Bearer {SECRET}"

    payload = {
        "hookType": "notification",
        "sessionId": session_id,
        "project": data.get("cwd") or os.getcwd(),
        "cwd": data.get("cwd") or os.getcwd(),
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "data": {
            "tool_name": "notification",
            "tool_input": "",
            "tool_output": str(message)[:4000],
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
