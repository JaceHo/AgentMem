"""Shared utility helpers for AgentMem core modules."""

import json
from typing import Any


def decode_json(raw: str | bytes | None) -> Any:
    """Decode JSON payload from bytes or string in a safe, repeated way."""
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    return json.loads(raw)


def force_str(value: object) -> str:
    """Convert a value to string safely."""
    if value is None:
        return ""
    return str(value)
