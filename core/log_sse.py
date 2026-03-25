"""
Real-time log streaming via Server-Sent Events.

Taps into Python's logging system — zero changes to business logic needed.
The LogSSEHandler intercepts every log record and broadcasts it to all
connected SSE clients via asyncio queues.

Usage in main.py:
    from log_sse import LogSSEHandler, broadcast_log_event, log_sse_router

    logging.getLogger().addHandler(LogSSEHandler())
    app.include_router(log_sse_router)
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

# ── Broadcaster ────────────────────────────────────────────────────────────────

_connections: Dict[str, asyncio.Queue] = {}   # connection_id → queue


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def broadcast(event_type: str, data: dict) -> None:
    """Push an event to all connected SSE clients."""
    if not _connections:
        return
    payload = json.dumps({"ts": _now_iso(), "event": event_type, **data})
    dead = []
    for cid, q in list(_connections.items()):
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            dead.append(cid)
    for cid in dead:
        _connections.pop(cid, None)


# ── Custom logging handler ─────────────────────────────────────────────────────

_LEVEL_COLOR = {
    "DEBUG":    "dim",
    "INFO":     "info",
    "WARNING":  "warn",
    "ERROR":    "error",
    "CRITICAL": "error",
}

class LogSSEHandler(logging.Handler):
    """
    Python logging handler that broadcasts log records as SSE events.
    Install once at startup:
        logging.getLogger().addHandler(LogSSEHandler(level=logging.INFO))
    """

    def __init__(self, level: int = logging.INFO):
        super().__init__(level)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_loop(self) -> Optional[asyncio.AbstractEventLoop]:
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return None

    def emit(self, record: logging.LogRecord) -> None:
        loop = self._get_loop()
        if loop is None:
            return
        try:
            payload = json.dumps({
                "ts":    datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                "event": "log",
                "level": record.levelname,
                "color": _LEVEL_COLOR.get(record.levelname, "info"),
                "name":  record.name,
                "msg":   self.format(record),
            })
            # Fire-and-forget broadcast to all connections
            for q in list(_connections.values()):
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    pass
        except Exception:
            pass   # Never let logging errors propagate


# ── SSE endpoint ───────────────────────────────────────────────────────────────

log_sse_router = APIRouter()


@log_sse_router.get("/logs/stream")
async def log_stream(request: Request):
    """
    Server-Sent Events stream of service logs.
    Connect from the browser:
        const es = new EventSource('/logs/stream');
        es.onmessage = e => console.log(JSON.parse(e.data));
    """
    cid = str(uuid.uuid4())
    q: asyncio.Queue = asyncio.Queue(maxsize=500)
    _connections[cid] = q

    # Send welcome message
    welcome = json.dumps({
        "ts":    _now_iso(),
        "event": "connected",
        "level": "INFO",
        "color": "info",
        "name":  "sse",
        "msg":   f"[mem-dashboard] Log stream connected ({cid[:8]})",
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
                    # Heartbeat ping
                    ping = json.dumps({"ts": _now_iso(), "event": "ping"})
                    yield f"data: {ping}\n\n"
        finally:
            _connections.pop(cid, None)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@log_sse_router.get("/logs/connections")
async def log_connections():
    """How many SSE clients are currently watching the log stream."""
    return {"active": len(_connections)}
