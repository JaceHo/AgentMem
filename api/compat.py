"""
Compat helpers for the agentmemory API surface.

Handles dual-format fields (camelCase + snake_case) and bearer token auth.
"""

import os
import time

from fastapi import Request


def compat_sid(req) -> str:
    """Extract session ID from compat request (supports both camelCase and snake_case)."""
    return req.sessionId or req.session_id or f"ses_{int(time.time())}"


def check_auth(request: Request) -> dict | None:
    """Check AGENTMEMORY_SECRET bearer token if configured.

    Returns None on success, or an error dict on failure.
    """
    secret = os.getenv("AGENTMEMORY_SECRET", "")
    if not secret:
        return None
    auth = request.headers.get("authorization", "")
    if auth != f"Bearer {secret}":
        return {"status_code": 401, "error": "unauthorized"}
    return None
