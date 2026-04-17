"""
Real-time log streaming via Server-Sent Events + persistent ring buffer.

Taps into Python's logging system — zero changes to business logic needed.
The LogSSEHandler intercepts every log record, appends it to an in-memory
ring buffer (also mirrored to disk so logs survive restarts), and broadcasts
it to all connected SSE clients via asyncio queues.

Endpoints:
    GET /logs/recent?limit=500   → JSON array of recent log lines (for backfill)
    GET /logs/stream             → SSE stream of new log lines
    GET /logs/connections        → count of active SSE clients

Usage in main.py:
    from core import log_sse

    log_sse.init_persistence()                   # load on-disk tail into buffer
    h = log_sse.LogSSEHandler(level=logging.INFO)
    h.setFormatter(logging.Formatter("%(message)s"))   # raw msg; we add ts/level
    logging.getLogger().addHandler(h)
    app.include_router(log_sse.log_sse_router)
"""

import asyncio
import json
import logging
import os
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

# ── Configuration ─────────────────────────────────────────────────────────────

# Keep the last N records in memory for backfill on dashboard reload.
_BUFFER_SIZE = 2000

# Persist a rolling tail to disk so the dashboard can show history that
# predates the current process (i.e. survives restarts and crashes).
_LOG_DIR  = Path(os.path.expanduser("~/.agentmem/logs"))
_LOG_FILE = _LOG_DIR / "dashboard.jsonl"
# Cap on-disk file at ~5 MiB; rotated to dashboard.jsonl.1 on overflow.
_MAX_DISK_BYTES = 5 * 1024 * 1024


# ── State ─────────────────────────────────────────────────────────────────────

_connections: Dict[str, asyncio.Queue] = {}        # connection_id → queue
_buffer: Deque[dict] = deque(maxlen=_BUFFER_SIZE)  # in-memory ring of log dicts


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_disk(entry: dict) -> None:
    """Append one JSONL record. Rotates once the file exceeds _MAX_DISK_BYTES."""
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        if _LOG_FILE.exists() and _LOG_FILE.stat().st_size > _MAX_DISK_BYTES:
            rot = _LOG_FILE.with_suffix(".jsonl.1")
            try:
                if rot.exists():
                    rot.unlink()
                _LOG_FILE.rename(rot)
            except Exception:
                pass
        with _LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass  # never let log persistence kill the process


def init_persistence() -> None:
    """Load the on-disk tail into the in-memory buffer at startup."""
    try:
        if not _LOG_FILE.exists():
            return
        # Read last _BUFFER_SIZE lines efficiently
        with _LOG_FILE.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            # Pull at most ~2 MiB tail to keep startup fast
            chunk = min(size, 2 * 1024 * 1024)
            f.seek(size - chunk, os.SEEK_SET)
            data = f.read().decode("utf-8", errors="replace")
        lines = data.splitlines()[-_BUFFER_SIZE:]
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                _buffer.append(json.loads(line))
            except Exception:
                continue
        # Mark the boundary between previous-process logs and current run
        boundary = {
            "ts":    _now_iso(),
            "event": "log",
            "level": "INFO",
            "color": "info",
            "name":  "agentmem",
            "msg":   f"── service restarted at {_now_iso()} "
                     f"(loaded {len(lines)} prior log lines) ──",
        }
        _buffer.append(boundary)
        _append_disk(boundary)
    except Exception:
        pass


# ── Logging handler ───────────────────────────────────────────────────────────

_LEVEL_COLOR = {
    "DEBUG":    "dim",
    "INFO":     "info",
    "WARNING":  "warn",
    "ERROR":    "error",
    "CRITICAL": "error",
}


class LogSSEHandler(logging.Handler):
    """
    Python logging handler that:
      1. Appends each record to the in-memory ring buffer
      2. Mirrors it to ~/.agentmem/logs/dashboard.jsonl
      3. Broadcasts it to all live SSE clients
    """

    def __init__(self, level: int = logging.INFO):
        super().__init__(level)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = {
                "ts":    datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                "event": "log",
                "level": record.levelname,
                "color": _LEVEL_COLOR.get(record.levelname, "info"),
                "name":  record.name,
                "msg":   self.format(record),
            }
            _buffer.append(entry)
            _append_disk(entry)
            payload = json.dumps(entry, ensure_ascii=False)
            for q in list(_connections.values()):
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    pass
        except Exception:
            pass  # logging must never raise


# ── HTTP / SSE endpoints ──────────────────────────────────────────────────────

log_sse_router = APIRouter()


@log_sse_router.get("/logs/recent")
async def logs_recent(limit: int = 500, level: str = "all"):
    """
    Return the most recent N log entries (from in-memory buffer + disk tail).
    Used by the dashboard to backfill history immediately on page load,
    so the user sees real activity instead of an empty box waiting for SSE.
    """
    limit = max(1, min(limit, _BUFFER_SIZE))
    LNUM = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
    min_level = {"all": 0, "INFO": 20, "WARNING": 30, "ERROR": 40}.get(level, 0)
    items: List[dict] = []
    for e in list(_buffer):
        if LNUM.get(e.get("level", "INFO"), 20) >= min_level:
            items.append(e)
    return JSONResponse(items[-limit:])


@log_sse_router.get("/logs/stream")
async def log_stream(request: Request):
    """
    Server-Sent Events stream of NEW log entries (does not backfill —
    use /logs/recent for that).
    """
    cid = str(uuid.uuid4())
    q: asyncio.Queue = asyncio.Queue(maxsize=500)
    _connections[cid] = q

    welcome = json.dumps({
        "ts":    _now_iso(),
        "event": "connected",
        "level": "INFO",
        "color": "info",
        "name":  "sse",
        "msg":   f"[mem-dashboard] log stream connected ({cid[:8]})",
    })
    await q.put(welcome)

    async def _generate():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    data = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield f"data: {data}\n\n"
                except asyncio.TimeoutError:
                    # SSE keepalive — sent as a comment line so EventSource
                    # ignores it entirely and it never reaches `onmessage`.
                    yield ": keepalive\n\n"
        finally:
            _connections.pop(cid, None)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


@log_sse_router.get("/logs/connections")
async def log_connections():
    return {"active": len(_connections)}


# ── Back-compat broadcast helper (kept so existing callers don't break) ───────

async def broadcast(event_type: str, data: dict) -> None:
    if not _connections:
        return
    payload = json.dumps({"ts": _now_iso(), "event": event_type, **data})
    for q in list(_connections.values()):
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            pass
