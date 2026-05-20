"""Shared HTTP utility helpers for AgentMem core modules."""

import logging
from typing import Any

import httpx

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10.0


async def async_post_json(
    url: str,
    payload: dict | None = None,
    headers: dict | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict | None:
    """POST JSON and return parsed JSON result, or None on failure."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
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
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        log.warning("[http] GET %s failed: %s", url, exc)
    except Exception as exc:
        log.debug("[http] GET %s error: %s", url, exc)
    return None
