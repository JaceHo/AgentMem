#!/usr/bin/env python3
"""session-start hook — registers session, optionally injects context."""

import os

from hooks.common import current_project, current_session_id, post_and_write_output, read_input, skip_hook

INJECT_TIMEOUT = 1.5


def main():
    data = read_input()
    if not data or skip_hook(data):
        return

    session_id = current_session_id(data, default=f"ses_{os.getpid()}")
    project = current_project(data)

    payload = {"sessionId": session_id, "project": project, "cwd": project}

    if not os.getenv("AGENTMEMORY_INJECT_CONTEXT", "true").lower() == "true":
        post_and_write_output("/agentmemory/session/start", payload, timeout=INJECT_TIMEOUT)
        return

    post_and_write_output("/agentmemory/session/start", payload, timeout=INJECT_TIMEOUT)


if __name__ == "__main__":
    main()
