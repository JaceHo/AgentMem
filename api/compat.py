"""
Compat helpers for the agentmemory API surface.

Handles dual-format fields (camelCase + snake_case) and bearer token auth.
"""

import hmac
import os
import time
import uuid

from fastapi import Request
from fastapi.exceptions import HTTPException


def compat_sid(req) -> str:
    """Extract session ID from compat request (supports both camelCase and snake_case)."""
    return req.sessionId or req.session_id or f"ses_{int(time.time())}_{uuid.uuid4().hex[:8]}"


def check_auth(request: Request) -> None:
    """Check AGENTMEMORY_SECRET bearer token if configured.

    Raises HTTPException(401) on auth failure.  Returns None on success.
    Uses constant-time comparison to prevent timing attacks.
    """
    secret = os.getenv("AGENTMEMORY_SECRET", "")
    if not secret:
        return None
    auth = request.headers.get("authorization", "")
    expected = f"Bearer {secret}"
    if not hmac.compare_digest(auth, expected):
        raise HTTPException(status_code=401, detail="unauthorized")
    return None
