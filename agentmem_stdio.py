#!/usr/bin/env python3
"""
agentmem_stdio.py — MCP stdio bridge to the embedded HTTP MCP endpoint.

Use when Cursor (or other stdio clients) should talk to the running AgentMem
service on :18800 instead of spawning a second in-process MCP server.

~/.cursor/mcp.json:
    "agentmem": {
      "command": "/Users/jace/code/agentmem/venv/bin/python3",
      "args": ["/Users/jace/code/agentmem/agentmem_stdio.py"]
    }
"""

import json
import os
import sys
import urllib.error
import urllib.request

MCP_URL = os.getenv("AGENTMEMORY_MCP_URL", "http://127.0.0.1:18800/mcp").rstrip("/")
_session_id = ""


def _post(body: dict) -> dict:
    global _session_id
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if _session_id:
        headers["mcp-session-id"] = _session_id

    req = urllib.request.Request(
        MCP_URL,
        data=json.dumps(body).encode(),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            new_sid = resp.headers.get("mcp-session-id", "")
            if new_sid:
                _session_id = new_sid
            ct = resp.headers.get("content-type", "")
            if "application/json" in ct:
                return json.loads(resp.read().decode())
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if line.startswith("data:"):
                    try:
                        return json.loads(line[5:].lstrip())
                    except json.JSONDecodeError:
                        pass
    except urllib.error.URLError as e:
        return {
            "jsonrpc": "2.0",
            "id": body.get("id"),
            "error": {"code": -32603, "message": f"agentmem MCP unreachable: {e}"},
        }
    return {
        "jsonrpc": "2.0",
        "id": body.get("id"),
        "error": {"code": -32603, "message": "No response from agentmem MCP"},
    }


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            body = json.loads(line)
        except json.JSONDecodeError:
            continue
        if body.get("method", "").startswith("notifications/"):
            continue
        sys.stdout.write(json.dumps(_post(body)) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
