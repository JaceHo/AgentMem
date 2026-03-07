"""
Shared async HTTP client for claw-memory REST API.

All framework adapters delegate to MemoryClient — they are thin idiomatic
wrappers around these three methods: recall, store, compress_session.
"""

from __future__ import annotations

import httpx


class MemoryClient:
    """
    Thin async client for the claw-memory REST API.

    Args:
        base_url:   URL of the running memory service (default: http://127.0.0.1:18800)
        session_id: Identifier that groups conversation turns into a session.
                    Pass the same value for every turn in one conversation.
        timeout:    HTTP timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:18800",
        session_id: str = "",
        timeout: float = 5.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id
        self.timeout = timeout

    async def recall(self, query: str, **kwargs) -> str | None:
        """
        Retrieve relevant memories for *query*.

        Keyword args are forwarded to POST /recall (e.g. memory_limit_number,
        time_from, time_to, include_tools, include_graph).

        Returns:
            prependContext string from the service, or None if empty.
        """
        payload = {"query": query, "session_id": self.session_id, **kwargs}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/recall", json=payload)
            resp.raise_for_status()
            return resp.json().get("prependContext")

    async def store(self, messages: list[dict], **kwargs) -> None:
        """
        Save a list of message dicts to long-term memory (fire-and-forget).

        Each dict must have ``role`` (str) and ``content`` (str or list) keys.
        """
        payload = {
            "messages": messages,
            "session_id": kwargs.pop("session_id", self.session_id),
            **kwargs,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/store", json=payload)
            resp.raise_for_status()

    async def store_text(self, text: str, role: str = "user") -> None:
        """Convenience wrapper: store a single text turn."""
        await self.store([{"role": role, "content": text}])

    async def compress_session(self) -> dict:
        """
        Promote accumulated Tier-1 session context to Tier-2 long-term memory.

        Call at the end of a conversation / agent session.
        Returns the result dict from POST /session/compress (with wait=True).
        """
        payload = {"session_id": self.session_id, "wait": True}
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{self.base_url}/session/compress", json=payload)
            resp.raise_for_status()
            return resp.json()
