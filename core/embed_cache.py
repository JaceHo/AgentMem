"""Persistent embedding cache — precompute once, reuse across restarts.

Why
---
Every restart re-encoded texts that were already encoded by the previous
process: recall queries that agents poll with verbatim, documents re-stored
after edits, and the dimension probe. On a memory-pressured host a single
encode of a large batch costs seconds, and the in-process ``lru_cache`` in
core/search.py dies with the process.

Design (deliberately boring)
----------------------------
* SQLite, one file, WAL mode. No Redis coupling: the encode path is
  synchronous and runs on worker threads, while the app's Redis client is
  asyncio — bridging the two would add a loop-hop and a failure mode to the
  hottest path in the service.
* Key = sha256(embed_signature + text). ``embed_signature()`` names the model
  AND its measured width, so swapping models (or a width change) can never
  serve stale vectors — cross-width protection lives in the key, exactly as it
  does for stored chunks.
* Row-level ``dim`` is a self-consistency check only: it catches corrupt rows
  (declared dim disagrees with the blob length). It cannot know the *current*
  model's width — by design, so a cache read never triggers a model load.
* Read/write paths are try/except-swallowed: a cache MUST never break an
  encode. The worst case is always "cache disabled, encode as before"; a
  wedged SQLite latches the cache off for the process instead of retry-storming.
* Bounded: over ``AGENTMEM_EMB_CACHE_CAP`` entries (default 200k) the oldest
  10% are dropped, so a runaway corpus cannot grow the file unbounded.

Disable with AGENTMEM_EMB_CACHE=0.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
import time
from typing import Dict, Iterable, Optional, Sequence

import numpy as np

_ENABLED = os.environ.get("AGENTMEM_EMB_CACHE", "1") not in ("0", "false", "no")
_CAP = int(os.environ.get("AGENTMEM_EMB_CACHE_CAP", "200000"))
_TRIM_EVERY = 512          # writes between trim checks
_TRIM_FRACTION = 0.10

_path = os.environ.get("AGENTMEM_EMB_CACHE_PATH") or os.path.expanduser(
    "~/.agentmem/cache/embeddings.sqlite3")

_lock = threading.RLock()
_conn: Optional[sqlite3.Connection] = None
_writes_since_trim = 0
_broken = False            # latched: stop touching the cache after a hard error
_warned = False


def _log_once(msg: str) -> None:
    global _warned
    if not _warned:
        _warned = True
        try:
            from core.log_sse import get_logger  # type: ignore
            get_logger("mem").warning(msg)
        except Exception:                                  # pragma: no cover
            import logging
            logging.getLogger("mem").warning(msg)


def _connect() -> Optional[sqlite3.Connection]:
    global _conn
    if _conn is not None:
        return _conn
    try:
        os.makedirs(os.path.dirname(_path), exist_ok=True)
        c = sqlite3.connect(_path, check_same_thread=False, timeout=5.0)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        c.execute(
            "CREATE TABLE IF NOT EXISTS emb ("
            " k TEXT PRIMARY KEY,"
            " dim INTEGER NOT NULL,"
            " vec BLOB NOT NULL,"
            " created REAL NOT NULL)"
        )
        _conn = c
        return c
    except Exception as exc:                               # noqa: BLE001
        _log_once(f"[emb-cache] disabled ({type(exc).__name__}: {exc})")
        return None


def cache_key(signature: str, text: str) -> str:
    """sha256(model signature + raw text). Signature change ⇒ key change."""
    return hashlib.sha256(f"{signature}\x00{text}".encode("utf-8", "replace")).hexdigest()


def _row_to_vec(dim: int, blob: bytes) -> Optional[np.ndarray]:
    vec = np.frombuffer(blob, dtype=np.float32)
    if vec.size != dim or dim == 0:
        return None                                        # stale/different width
    return vec


def get(key: str) -> Optional[np.ndarray]:
    if not _ENABLED or _broken:
        return None
    c = _connect()
    if c is None:
        return None
    try:
        with _lock:
            row = c.execute("SELECT dim, vec FROM emb WHERE k=?", (key,)).fetchone()
        if row is None:
            return None
        return _row_to_vec(int(row[0]), row[1])
    except Exception as exc:                               # noqa: BLE001
        _log_once(f"[emb-cache] read failed, ignoring ({type(exc).__name__}: {exc})")
        return None


def get_many(keys: Sequence[str]) -> Dict[str, np.ndarray]:
    """Batch lookup. Missing keys are simply absent from the result."""
    if not _ENABLED or _broken or not keys:
        return {}
    c = _connect()
    if c is None:
        return {}
    out: Dict[str, np.ndarray] = {}
    try:
        with _lock:
            cur = c.execute(
                f"SELECT k, dim, vec FROM emb WHERE k IN "
                f"({','.join('?' * len(keys))})", tuple(keys))
            for k, dim, blob in cur.fetchall():
                vec = _row_to_vec(int(dim), blob)
                if vec is not None:
                    out[k] = vec
    except Exception as exc:                               # noqa: BLE001
        _log_once(f"[emb-cache] batch read failed, ignoring ({type(exc).__name__}: {exc})")
    return out


def put(key: str, vec: np.ndarray) -> None:
    put_many([(key, vec)])


def put_many(items: Iterable[tuple[str, np.ndarray]]) -> None:
    """Write a batch in one transaction. Failures are swallowed by design."""
    global _writes_since_trim, _broken
    if not _ENABLED or _broken:
        return
    c = _connect()
    if c is None:
        return
    rows = []
    for key, vec in items:
        try:
            arr = np.asarray(vec, dtype=np.float32).reshape(-1)
            if arr.size == 0 or not np.isfinite(arr).all():
                continue
            rows.append((key, int(arr.size), arr.tobytes(), time.time()))
        except Exception:                                  # noqa: BLE001
            continue
    if not rows:
        return
    try:
        with _lock:
            c.executemany(
                "INSERT OR REPLACE INTO emb (k, dim, vec, created) VALUES (?,?,?,?)",
                rows)
            c.commit()
            _writes_since_trim += len(rows)
            if _writes_since_trim >= _TRIM_EVERY:
                _writes_since_trim = 0
                _trim(c)
    except Exception as exc:                               # noqa: BLE001
        # A wedged cache must not wedge encoding: latch off rather than retry-storm.
        _broken = True
        _log_once(f"[emb-cache] write failed — cache disabled this process "
                  f"({type(exc).__name__}: {exc})")


def _trim(c: sqlite3.Connection) -> None:
    """Keep the file bounded: drop the oldest 10% once over the cap."""
    try:
        n = c.execute("SELECT COUNT(*) FROM emb").fetchone()[0]
        if n <= _CAP:
            return
        drop = int(n - _CAP + _CAP * _TRIM_FRACTION)
        c.execute(
            "DELETE FROM emb WHERE k IN "
            "(SELECT k FROM emb ORDER BY created ASC LIMIT ?)", (drop,))
        c.commit()
    except Exception as exc:                               # noqa: BLE001
        _log_once(f"[emb-cache] trim failed, ignoring ({type(exc).__name__}: {exc})")


def stats() -> dict:
    """Introspection for /kb/status-style dashboards. Never raises."""
    if not _ENABLED or _broken:
        return {"enabled": False}
    c = _connect()
    if c is None:
        return {"enabled": False}
    try:
        with _lock:
            n = c.execute("SELECT COUNT(*) FROM emb").fetchone()[0]
        return {"enabled": True, "entries": n, "cap": _CAP, "path": _path}
    except Exception as exc:                               # noqa: BLE001
        return {"enabled": False, "error": str(exc)[:120]}
