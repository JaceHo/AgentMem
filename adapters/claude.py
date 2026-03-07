"""
Direct Claude API adapter for claw-memory.

Provides ClaudeMemorySession — the simplest possible integration:
recall before each turn → build messages → call Claude → store → return.

Usage::

    import anthropic
    from adapters.claude import ClaudeMemorySession

    client  = anthropic.Anthropic()
    session = ClaudeMemorySession(
        session_id="conv-abc",
        anthropic_client=client,
        model="claude-sonnet-4-6",
    )

    reply = await session.chat("What did we discuss about Redis yesterday?")
    print(reply)
    # At the end of the conversation:
    await session.end()   # promotes Tier-1 session to Tier-2 long-term memory
"""

from __future__ import annotations

from typing import Any

from .base import MemoryClient


class ClaudeMemorySession:
    """
    Minimal Claude API session with automatic memory recall and storage.

    Each call to chat():
      1. Recalls relevant memories from claw-memory
      2. Prepends them to the system prompt
      3. Calls anthropic.messages.create
      4. Stores the exchange (user + assistant turns) back to memory

    Args:
        base_url:           URL of the claw-memory service.
        session_id:         Session identifier (persists across process restarts).
        anthropic_client:   An instantiated anthropic.Anthropic() client.
        model:              Claude model ID (default: claude-sonnet-4-6).
        max_tokens:         Max output tokens per response.
        system:             Optional base system prompt (memory prepended on top).
    """

    def __init__(
        self,
        session_id: str = "",
        base_url: str = "http://127.0.0.1:18800",
        anthropic_client: Any = None,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 1024,
        system: str = "",
    ) -> None:
        if anthropic_client is None:
            try:
                import anthropic
                anthropic_client = anthropic.AsyncAnthropic()
            except ImportError:
                raise ImportError(
                    "anthropic SDK is not installed. Run: pip install anthropic"
                )
        self._client  = MemoryClient(base_url, session_id)
        self._claude  = anthropic_client
        self.model    = model
        self.max_tokens = max_tokens
        self._system  = system
        # In-session message history (only kept for multi-turn context within one process)
        self._history: list[dict] = []

    async def chat(self, user_msg: str) -> str:
        """
        Send a user message, recall memories, call Claude, store the exchange.

        Returns the assistant's reply text.
        """
        # 1. Recall memories relevant to this query
        memory_ctx = await self._client.recall(user_msg)

        # 2. Build system prompt (memory on top, user system prompt below)
        system_parts = []
        if memory_ctx:
            system_parts.append(memory_ctx)
        if self._system:
            system_parts.append(self._system)
        system_prompt = "\n\n".join(system_parts)

        # 3. Build messages list (include in-session history for multi-turn)
        self._history.append({"role": "user", "content": user_msg})
        messages = list(self._history)

        # 4. Call Claude
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = await self._claude.messages.create(**kwargs)
        reply = response.content[0].text if response.content else ""

        # 5. Append assistant turn to history
        self._history.append({"role": "assistant", "content": reply})

        # 6. Store the exchange to memory (fire-and-forget)
        await self._client.store([
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": reply},
        ])

        return reply

    async def end(self) -> dict:
        """
        End the session: promote Tier-1 session context to Tier-2 long-term memory.

        Call this when the conversation is finished (equivalent to agent_end hook).
        Returns the compress result dict.
        """
        return await self._client.compress_session()
