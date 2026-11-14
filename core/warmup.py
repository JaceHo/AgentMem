"""Startup readiness tracking — the warmup gate.

Why this module exists
----------------------
agentmem's cold start is dominated by work that liveness does **not** need:
loading the local embedding model (qwen3-embedding-0.6b), connecting to Redis,
and rebuilding the in-RAM BM25 index over every knowledge-base chunk.

Measured on 2026-11-05 (macOS, 24 GiB host, memory-pressured):

    07:22:54  [mem] Connecting to Redis…
    07:27:58  [mem] [kb] BM25 warm: 2083 chunks      <- ~5 min of silence
    07:59:52  [mem] Load pretrained SentenceTransformer …   (next restart)

~10 min total cold start. All of it ran inside ``FastAPI(lifespan=...)``, which
executes **before uvicorn binds the listening socket**. So for the whole warmup
the port answered ``ECONNREFUSED`` -- indistinguishable from a crash. The
supervisor then counted two 30 s failures and escalated a P0 for a service that
was merely booting:

    07:57:06  agentmem: DOWN (000) fails=2
    07:59:32  agentmem: ESCALATED to P0 (OPEN #651)
    07:59:52  [mem] Load pretrained SentenceTransformer …   <- model starts AFTER the P0

The fix is the standard liveness/readiness split, plus a bounded wait:

    GET /livez   -> 200 the moment the socket answers   (is the process alive?)
    GET /health  -> 200 with warmup detail              (is it usable? say so honestly)
    GET /ready   -> 503 while warming, 200 when ready    (may I send real work?)

Real work that arrives during warmup is held for up to ``AGENTMEM_WARMUP_WAIT_S``
seconds (default 20) and then answered ``503`` + ``Retry-After`` instead of
dying or returning a misleading 500. That is the "wait a little, else fail fast
with a retry hint" contract -- no request is silently dropped, and no caller has
to guess whether the service is dead or merely busy being born.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from typing import Any, Optional

# How long a real request may block on warmup before we give it a Retry-After.
WAIT_S: float = float(os.environ.get("AGENTMEM_WARMUP_WAIT_S", "20"))

# Paths that must answer during warmup: liveness, readiness, docs, static assets.
_EXEMPT_EXACT = frozenset({
    "/livez", "/health", "/ready", "/openapi.json", "/redoc", "/docs",
    "/favicon.ico", "/favicon.svg", "/apple-touch-icon.png",
})
_EXEMPT_PREFIX = ("/static/", "/docs/", "/redoc/", "/css/", "/js/")

_lock = threading.Lock()
_started_monotonic = time.monotonic()
_started_wall = time.time()

_phase: str = "booting"
_detail: str = ""
_ready: bool = False                    # stage A: core (redis/indexes/kb) — gates /ready
_embedding_ready: bool = False          # stage B: model loaded + dim-guard verified
_failed: Optional[str] = None
_embedding_failed: Optional[str] = None
_finished_monotonic: Optional[float] = None


def set_phase(phase: str, detail: str = "") -> None:
    """Record warmup progress. Visible to /health and /ready as it advances.

    Once stage A has flipped ``ready``, later background work (dim-guard /
    embedding load) must not stomp the phase back to a pre-ready label — that
    made /health lie with ``ready:true, phase:dim_guard`` after boot.
    """
    global _phase, _detail
    with _lock:
        if _ready and phase not in ("ready", "failed", "embedding_ready"):
            if detail:
                _detail = detail
            return
        _phase = phase
        _detail = detail


def mark_ready() -> None:
    """Stage A done: the service can serve core work. Model warmup continues."""
    global _ready, _phase, _finished_monotonic, _detail
    with _lock:
        _ready = True
        _phase = "ready"
        _detail = "core ready; embedding warming"
        _finished_monotonic = time.monotonic()


def mark_embedding_ready() -> None:
    """Stage B done: model loaded, dim-guard verified. Non-gating — /ready flipped
    at stage A, and the lazy loader in embedder covers any request that arrives
    before this fires."""
    global _embedding_ready, _embedding_failed, _phase, _detail
    with _lock:
        _embedding_ready = True
        _embedding_failed = None
        _phase = "ready"
        _detail = "embedding ready"


def mark_embedding_failed(error: str) -> None:
    global _embedding_ready, _embedding_failed
    with _lock:
        _embedding_ready = False
        _embedding_failed = error[:500]


def mark_failed(error: str) -> None:
    """Core warmup blew up. Stay un-ready so /ready keeps reporting the failure loudly."""
    global _failed, _phase, _detail
    with _lock:
        _failed = error[:500]
        _phase = "failed"
        _detail = error[:500]


def is_ready() -> bool:
    return _ready


def is_embedding_ready() -> bool:
    return _embedding_ready


def snapshot() -> dict[str, Any]:
    """Serialisable view for /health and /ready."""
    with _lock:
        elapsed = (_finished_monotonic or time.monotonic()) - _started_monotonic
        snap: dict[str, Any] = {
            "ready": _ready,
            "phase": _phase,
            "detail": _detail,
            "embedding_ready": _embedding_ready,
            "elapsed_s": round(elapsed, 1),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(_started_wall)),
            "wait_budget_s": WAIT_S,
        }
        if _embedding_failed:
            snap["embedding_error"] = _embedding_failed
        if _failed:
            snap["error"] = _failed
        return snap


async def wait_ready(timeout: Optional[float] = None) -> bool:
    """Block until ready, the budget expires, or warmup fails.

    Returns True if the service became ready inside the budget. Callers that get
    False should answer 503 + Retry-After rather than pretending to work.
    """
    budget = WAIT_S if timeout is None else timeout
    deadline = time.monotonic() + max(0.0, budget)
    # Fast path avoids a sleep when warmup already finished.
    if _ready:
        return True
    while time.monotonic() < deadline:
        if _ready:
            return True
        if _failed:
            return False
        await asyncio.sleep(0.15)
    return _ready


def is_exempt(path: str) -> bool:
    """True for paths allowed to answer while warmup is still running."""
    if path in _EXEMPT_EXACT:
        return True
    return any(path.startswith(p) for p in _EXEMPT_PREFIX)
