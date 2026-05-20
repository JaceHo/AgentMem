"""Shared utility helpers for AgentMem core modules."""

import json
from typing import Any


def decode_bytes(value: str | bytes | None) -> str:
    """Decode bytes to str, pass through str/None unchanged.

    Replaces the 80+ copy-paste ``x.decode() if isinstance(x, bytes) else x``
    patterns scattered across the codebase.
    """
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return value


def decode_json(raw: str | bytes | None) -> Any:
    """Decode JSON payload from bytes or string in a safe, repeated way."""
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    return json.loads(raw)


def decode_attrs(raw: str | bytes | None) -> dict:
    """Decode VGETATTR JSON payload into a dict, returning {} on any failure.

    Replaces the 35+ copy-paste ``json.loads(raw.decode() if isinstance(raw, bytes) else raw)``
    patterns used after VGETATTR calls.
    """
    if raw is None:
        return {}
    try:
        return decode_json(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}


def force_str(value: object) -> str:
    """Convert a value to string safely."""
    if value is None:
        return ""
    return str(value)
