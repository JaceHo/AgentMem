"""
CrewAI adapter for claw-memory.

Provides:
  RecallMemoryTool  — BaseTool that agents can call to retrieve memories
  StoreMemoryCallback — BaseCallbackHandler that auto-stores on task completion

Usage::

    from crewai import Agent, Task, Crew
    from adapters.crewai import RecallMemoryTool, StoreMemoryCallback

    recall_tool = RecallMemoryTool(base_url="http://127.0.0.1:18800", session_id="s1")
    callback    = StoreMemoryCallback(base_url="http://127.0.0.1:18800", session_id="s1")

    agent = Agent(
        role="Researcher",
        goal="Answer questions accurately",
        tools=[recall_tool],
        callbacks=[callback],
    )
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional, Type

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


try:
    from crewai_tools import BaseTool as CrewBaseTool
    from pydantic import BaseModel as PydanticBaseModel

    class _RecallInput(PydanticBaseModel):
        query: str

    class RecallMemoryTool(CrewBaseTool):
        """
        CrewAI tool that retrieves relevant long-term memories.

        Register this in an agent's tools list. The agent will call it
        automatically when it needs background knowledge.
        """
        name: str = "recall_memory"
        description: str = (
            "Recall relevant long-term memories and past conversation context. "
            "Use this whenever you need background knowledge about the user, "
            "previous decisions, or recurring topics."
        )
        args_schema: Type[PydanticBaseModel] = _RecallInput

        base_url: str = "http://127.0.0.1:18800"
        session_id: str = ""

        def _run(self, query: str) -> str:
            client = MemoryClient(self.base_url, self.session_id)
            ctx = _run_async(client.recall(query))
            return ctx or "(no relevant memories found)"

except ImportError:
    class RecallMemoryTool:  # type: ignore[no-redef]
        """Stub — install crewai-tools to use: pip install crewai-tools"""
        def __init__(self, **kwargs):
            raise ImportError("crewai-tools is not installed. Run: pip install crewai-tools")


try:
    from langchain_core.callbacks import BaseCallbackHandler

    class StoreMemoryCallback(BaseCallbackHandler):
        """
        CrewAI / LangChain callback that saves task output to long-term memory.

        Attach via agent = Agent(..., callbacks=[StoreMemoryCallback(...)]).
        """

        def __init__(
            self,
            base_url: str = "http://127.0.0.1:18800",
            session_id: str = "",
        ) -> None:
            super().__init__()
            self._client = MemoryClient(base_url, session_id)

        def on_tool_end(self, output: str, **kwargs: Any) -> None:
            """Store tool output snippets as context."""
            if output and len(output) > 30:
                _run_async(self._client.store_text(output, role="assistant"))

        def on_agent_finish(self, finish: Any, **kwargs: Any) -> None:
            """Store final agent output and compress session."""
            output = getattr(finish, "return_values", {}).get("output", "")
            if output:
                _run_async(self._client.store_text(output, role="assistant"))

except ImportError:
    class StoreMemoryCallback:  # type: ignore[no-redef]
        """Stub — install langchain-core to use: pip install langchain-core"""
        def __init__(self, **kwargs):
            raise ImportError("langchain-core is not installed. Run: pip install langchain-core")
