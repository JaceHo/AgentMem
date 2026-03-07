"""
AutoGen adapter for claw-memory.

Provides ClawMemoryHook — attach to any ConversableAgent to inject
memories before sending messages and save messages after receiving.

Usage::

    import autogen
    from adapters.autogen import ClawMemoryHook

    hook = ClawMemoryHook(base_url="http://127.0.0.1:18800", session_id="s1")

    agent = autogen.ConversableAgent("assistant", llm_config={...})
    agent.register_hook(
        "process_message_before_send",
        hook.process_message_before_send,
    )
    agent.register_hook(
        "process_last_received_message",
        hook.process_last_received_message,
    )
"""

from __future__ import annotations

import asyncio
from typing import Any

from .base import MemoryClient


def _run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


class ClawMemoryHook:
    """
    AutoGen agent hook that injects memories into outgoing messages
    and saves incoming messages to long-term memory.

    Register both methods via agent.register_hook().

    Args:
        base_url:   URL of the claw-memory service.
        session_id: Session identifier for grouping conversation turns.
        prepend_separator: String inserted between memory context and message.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:18800",
        session_id: str = "",
        prepend_separator: str = "\n\n---\n\n",
    ) -> None:
        self._client = MemoryClient(base_url, session_id)
        self._sep = prepend_separator

    def process_message_before_send(
        self,
        message: str | dict | list,
        sender: Any,
        recipient: Any,
        silent: bool,
    ) -> str | dict | list:
        """
        Hook: inject retrieved memories into the message before it is sent.

        AutoGen calls this with the raw message value. Handles str and dict
        message formats. Returns the message unchanged if no memories exist.
        """
        # Extract text content to use as recall query
        if isinstance(message, str):
            query = message
        elif isinstance(message, dict):
            query = message.get("content", "")
            if isinstance(query, list):
                query = " ".join(
                    b.get("text", "") for b in query if isinstance(b, dict)
                )
        else:
            return message  # list of messages — skip injection

        if not query:
            return message

        ctx = _run_async(self._client.recall(query))
        if not ctx:
            return message

        # Prepend memory context to the message
        prepended = f"{ctx}{self._sep}{query}"
        if isinstance(message, str):
            return prepended
        if isinstance(message, dict):
            new_msg = dict(message)
            new_msg["content"] = prepended
            return new_msg
        return message

    def process_last_received_message(self, message: str | dict) -> str | dict:
        """
        Hook: save the last received message to long-term memory.

        AutoGen calls this after each message is received. Stores the
        content as a 'user' turn (from the perspective of the other agent).
        """
        if isinstance(message, str):
            content = message
        elif isinstance(message, dict):
            content = message.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") for b in content if isinstance(b, dict)
                )
        else:
            return message

        if content and len(content) > 10:
            _run_async(self._client.store_text(content, role="user"))

        return message
