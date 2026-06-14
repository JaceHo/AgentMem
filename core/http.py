"""Shared HTTP utility helpers for AgentMem core modules.

Uses a lazily-initialized shared httpx.AsyncClient to avoid creating and
destroying a connection pool on every request. The client is created on
first use and reused across all calls, reducing TCP/TLS handshake overhead.

v1.1: Per-request timeout via asyncio.wait_for instead of shared client timeout.
The shared client uses a generous max timeout; individual requests enforce
their own deadline via asyncio.wait_for.
"""

import asyncio
import logging
from typing import Any

import httpx

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10.0
_MAX_CLIENT_TIMEOUT = 30.0

_shared_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    """Return the shared AsyncClient, creating it on first call.

    The client uses a generous timeout (_MAX_CLIENT_TIMEOUT) as the upper bound.
    Individual requests enforce their own deadline via asyncio.wait_for.
    Production-tuned connection pool: 100 max connections, 20 keepalive,
    60s keepalive expiry for long-running services.
    """
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        _shared_client = httpx.AsyncClient(
            timeout=httpx.Timeout(_MAX_CLIENT_TIMEOUT),
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=60,
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
    """POST JSON and return parsed JSON result, or None on failure.

    Uses asyncio.wait_for to enforce per-request timeout independently
    of the shared client's timeout setting.
    """
    try:
        client = _get_client()
        resp = await asyncio.wait_for(
            client.post(url, json=payload or {}, headers=headers),
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except asyncio.TimeoutError:
        log.debug("[http] POST %s timed out (%.1fs)", url, timeout)
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
    """GET JSON and return parsed JSON result, or None on failure.

    Uses asyncio.wait_for to enforce per-request timeout independently
    of the shared client's timeout setting.
    """
    try:
        client = _get_client()
        resp = await asyncio.wait_for(
            client.get(url, headers=headers),
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except asyncio.TimeoutError:
        log.debug("[http] GET %s timed out (%.1fs)", url, timeout)
    except httpx.HTTPStatusError as exc:
        log.warning("[http] GET %s failed: %s", url, exc)
    except Exception as exc:
        log.debug("[http] GET %s error: %s", url, exc)
    return None
