#!/usr/bin/env python3
"""pre-tool-use hook — enriches tool context with relevant memories."""

from hooks.common import current_project, current_session_id, post_and_write_output, read_input, skip_hook

ENRICH_TOOLS = {"Edit", "Write", "Read", "Glob", "Grep", "MultiEdit"}


def main():
    data = read_input()
    if not data or skip_hook(data):
        return

    tool_name = data.get("tool_name", "")
    if tool_name not in ENRICH_TOOLS:
        return

    session_id = current_session_id(data)
    project = current_project(data)

    tool_input = data.get("tool_input", {})
    files = []
    if isinstance(tool_input, dict):
        for key in ("file_path", "path", "glob", "pattern"):
            val = tool_input.get(key, "")
            if val and isinstance(val, str):
                files.append(val)

    payload = {
        "sessionId": session_id,
        "project": project,
        "cwd": project,
        "files": files,
        "query": f"{tool_name} " + " ".join(files[:3]),
    }

    post_and_write_output("/agentmemory/enrich", payload, timeout=1.5)


if __name__ == "__main__":
    main()
