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
    """Coerce a Redis reply (or any scalar) to ``str`` — bytes-aware.

    The shared pool is opened with ``decode_responses=False`` on purpose (binary
    FP32 blobs round-trip through it), so *every* value coming back from Redis —
    VSIM element names, hash fields, SET/SMEMBERS payloads — is ``bytes``.

    ``str(b"01M3AB")`` is ``"b'01M3AB'"``, not ``"01M3AB"``. That repr is a name
    that matches no element in any vectorset, so every write keyed on it (heat
    bumps, ``VSETATTR`` in A-MEM evolution, task→procedure lookups) silently
    no-oped for the whole of v1.0.x. Decode explicitly instead of ``str()``-ing.

    Non-text scalars (ints from ``HLEN``, floats, ...) keep the old ``str()``
    behaviour so this stays usable as the generic "make it a string" helper.
    """
    if value is None:
        return ""
    if isinstance(value, bytes):
        # utf-8 with replacement so a malformed element name degrades to a
        # visibly-weird key rather than raising inside a fire-and-forget task.
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    return str(value)
