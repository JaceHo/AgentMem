#!/usr/bin/env python3
"""post-tool-use hook — observes tool usage and stores to memory."""

import os
from datetime import datetime

from hooks.common import current_project, current_session_id, post, read_input, skip_hook


def _truncate(value, max_len=8000):
    if isinstance(value, str) and len(value) > max_len:
        return value[:max_len] + "\n[...truncated]"
    if isinstance(value, dict) or isinstance(value, list):
        s = json.dumps(value, ensure_ascii=False)
        if len(s) > max_len:
            return s[:max_len] + "...[truncated]"
    return value


def main():
    data = read_input()
    if not data or skip_hook(data):
        return

    session_id = current_session_id(data)
    project = current_project(data)

    payload = {
        "hookType": "post_tool_use",
        "sessionId": session_id,
        "project": project,
        "cwd": project,
        "timestamp": datetime.now().isoformat(),
        "data": {
            "tool_name": data.get("tool_name", ""),
            "tool_input": data.get("tool_input", ""),
            "tool_output": _truncate(data.get("tool_output", "")),
        },
    }

    post("/agentmemory/observe", payload, timeout=3.0)


if __name__ == "__main__":
    main()
