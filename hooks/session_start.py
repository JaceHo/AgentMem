#!/usr/bin/env python3
"""session-start hook — registers session, optionally injects context."""

import json
import os
import sys

try:
    import httpx
except ImportError:
    sys.exit(0)

REST_URL = os.getenv("AGENTMEMORY_URL", "http://localhost:18800")
SECRET = os.getenv("AGENTMEMORY_SECRET", "")
INJECT_CONTEXT = os.getenv("AGENTMEMORY_INJECT_CONTEXT", "true").lower() == "true"
REGISTER_TIMEOUT = 0.8
INJECT_TIMEOUT = 1.5


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        return

    if os.getenv("AGENTMEMORY_SDK_CHILD") == "1":
        return
    if isinstance(data, dict) and data.get("entrypoint") == "sdk-ts":
        return

    session_id = data.get("session_id") or f"ses_{os.getpid()}"
    project = data.get("cwd") or os.getcwd()

    headers = {"Content-Type": "application/json"}
    if SECRET:
        headers["Authorization"] = f"Bearer {SECRET}"

    payload = {"sessionId": session_id, "project": project, "cwd": project}

    if not INJECT_CONTEXT:
        try:
            httpx.post(
                f"{REST_URL}/agentmemory/session/start",
                json=payload,
                headers=headers,
                timeout=REGISTER_TIMEOUT,
            )
        except Exception:
            pass
        return

    try:
        resp = httpx.post(
            f"{REST_URL}/agentmemory/session/start",
            json=payload,
            headers=headers,
            timeout=INJECT_TIMEOUT,
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
