"""
LangChain adapter for claw-memory.

Drop-in BaseChatMemory subclass that persists conversation history in
the claw-memory service instead of in-process RAM.

Usage::

    from adapters.langchain import ClawMemory
    from langchain.chains import ConversationChain
    from langchain_anthropic import ChatAnthropic

    memory = ClawMemory(base_url="http://127.0.0.1:18800", session_id="my-session")
    chain  = ConversationChain(llm=ChatAnthropic(model="claude-sonnet-4-6"), memory=memory)
    reply  = chain.predict(input="Hello, who are you?")
"""

from __future__ import annotations

import asyncio
from typing import Any

from .base import MemoryClient

try:
    from langchain.memory import BaseChatMemory
    from langchain.schema import HumanMessage, AIMessage
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False
    BaseChatMemory = object   # type: ignore[assignment,misc]


def _run_async(coro):
    """Run an async coroutine from a synchronous context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


class ClawMemory(BaseChatMemory):
    """
    LangChain BaseChatMemory backed by claw-memory REST service.

    Compatible with LLMChain, ConversationChain, create_react_agent, etc.
    Memory is persisted across process restarts via Redis.

    Args:
        base_url:   URL of the claw-memory service.
        session_id: Session identifier (group turns into one conversation).
        memory_key: Key used to inject memories into the chain prompt.
    """

    # Pydantic v1 field declarations (LangChain uses pydantic v1 internally)
    base_url: str = "http://127.0.0.1:18800"
    session_id: str = ""
    memory_key: str = "history"

    class Config:
        arbitrary_types_allowed = True

    @property
    def _client(self) -> MemoryClient:
        return MemoryClient(self.base_url, self.session_id)

    @property
    def memory_variables(self) -> list[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Recall relevant memories and inject as 'history' into the chain prompt."""
        query = inputs.get("input") or inputs.get("human_input") or ""
        if not query:
            return {self.memory_key: ""}
        ctx = _run_async(self._client.recall(query))
        return {self.memory_key: ctx or ""}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        """Store the latest human/AI exchange into long-term memory."""
        human = inputs.get("input") or inputs.get("human_input") or ""
        ai    = outputs.get("response") or outputs.get("output") or outputs.get("text") or ""
        if not human:
            return
        messages = []
        if human:
            messages.append({"role": "user",      "content": human})
        if ai:
            messages.append({"role": "assistant", "content": ai})
        _run_async(self._client.store(messages))

    def clear(self) -> None:
        """Compress session context to long-term memory then reset."""
        _run_async(self._client.compress_session())
