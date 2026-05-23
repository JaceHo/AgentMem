"""Shared HTTP utility helpers for AgentMem core modules.

Uses a lazily-initialized shared httpx.AsyncClient to avoid creating and
destroying a connection pool on every request. The client is created on
first use and reused across all calls, reducing TCP/TLS handshake overhead.
"""

import logging
from typing import Any

import httpx

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10.0

# Shared client — lazily created on first use, reused across all calls.
# Avoids per-request connection pool creation (TCP handshake + TLS for HTTPS).
_shared_client: httpx.AsyncClient | None = None


def _get_client(timeout: float = DEFAULT_TIMEOUT) -> httpx.AsyncClient:
    """Return the shared AsyncClient, creating it on first call.

    The client uses the maximum timeout seen so far, ensuring no request
    is prematurely cut off. Connection pool limits prevent resource exhaustion.
    """
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        _shared_client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10,
                keepalive_expiry=30,
            ),
        )
    return _shared_client


async def close_client() -> None:
    """Close the shared client. Call at app shutdown."""
    global _shared_client
    if _shared_client is not None and not _shared_client.is_closed:
        await _shared_client.aclose()
    _shared_client = None


async def async_post_json(
    url: str,
    payload: dict | None = None,
    headers: dict | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict | None:
    """POST JSON and return parsed JSON result, or None on failure."""
    try:
        client = _get_client(timeout)
        resp = await client.post(url, json=payload or {}, headers=headers)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as exc:
        log.warning("[http] POST %s failed: %s", url, exc)
    except Exception as exc:
        log.debug("[http] POST %s error: %s", url, exc)
    return None


async def async_get_json(
    url: str,
    headers: dict | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict | None:
    """GET JSON and return parsed JSON result, or None on failure."""
    try:
        client = _get_client(timeout)
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as exc:
        log.warning("[http] GET %s failed: %s", url, exc)
    except Exception as exc:
        log.debug("[http] GET %s error: %s", url, exc)
    return None
