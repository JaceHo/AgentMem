#!/usr/bin/env python3
"""Shared hook helpers for AgentMem hook scripts."""

import json
import os
import sys
from pathlib import Path

# Ensure package imports resolve when scripts are invoked directly from the hooks
# directory or from the repository root.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import httpx
except ImportError:
    sys.exit(0)

REST_URL = os.getenv("AGENTMEMORY_URL", "http://localhost:18800")
SECRET = os.getenv("AGENTMEMORY_SECRET", "")


def _auth_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if SECRET:
        headers["Authorization"] = f"Bearer {SECRET}"
    return headers


def read_input() -> dict | None:
    try:
        return json.load(sys.stdin)
    except Exception:
        return None


def skip_hook(data: object) -> bool:
    return (
        os.getenv("AGENTMEMORY_SDK_CHILD") == "1"
        or (isinstance(data, dict) and data.get("entrypoint") == "sdk-ts")
    )


def current_project(data: dict | None) -> str:
    if not isinstance(data, dict):
        return os.getcwd()
    return data.get("cwd") or os.getcwd()


def current_session_id(data: dict | None, default: str = "unknown") -> str:
    if not isinstance(data, dict):
        return default
    return data.get("session_id") or default


def post(path: str, payload: dict, timeout: float = 3.0) -> httpx.Response | None:
    try:
        return httpx.post(f"{REST_URL}{path}", json=payload, headers=_auth_headers(), timeout=timeout)
    except Exception:
        return None


def post_and_write_output(path: str, payload: dict, timeout: float = 3.0) -> None:
    resp = post(path, payload, timeout=timeout)
    if not resp or resp.status_code != 200:
        return
    try:
        data = resp.json()
        context = data.get("context")
        if context:
            sys.stdout.write(context)
    except Exception:
        pass
