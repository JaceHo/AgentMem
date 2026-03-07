"""
LangGraph adapter for claw-memory.

Provides two ready-made StateGraph node functions:

  recall_node  — first node: injects memory context into graph state
  store_node   — last node:  saves new messages to long-term memory

Usage::

    from langgraph.graph import StateGraph, END
    from adapters.langgraph import recall_node, store_node, make_memory_nodes

    recall, store = make_memory_nodes(session_id="abc", base_url="http://127.0.0.1:18800")

    builder = StateGraph(dict)
    builder.add_node("recall",  recall)
    builder.add_node("llm",     my_llm_node)
    builder.add_node("store",   store)
    builder.set_entry_point("recall")
    builder.add_edge("recall", "llm")
    builder.add_edge("llm",    "store")
    builder.add_edge("store",  END)
    graph = builder.compile()

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Hello"}]})
    # result["memory_context"] contains the injected memory string
"""

from __future__ import annotations

from .base import MemoryClient


def make_memory_nodes(
    session_id: str = "",
    base_url: str = "http://127.0.0.1:18800",
    timeout: float = 5.0,
) -> tuple:
    """
    Factory that returns (recall_node, store_node) bound to the given session.

    Returns a tuple so you can unpack them with a single call:
        recall, store = make_memory_nodes(session_id="xyz")
    """
    client = MemoryClient(base_url=base_url, session_id=session_id, timeout=timeout)

    async def recall_node(state: dict) -> dict:
        """
        Read the last user message from state["messages"] and fetch memories.

        Injects result into state["memory_context"]. If no memories are found,
        memory_context is set to an empty string (never None).
        """
        messages = state.get("messages", [])
        query = ""
        for msg in reversed(messages):
            role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            if role == "user" and content:
                query = content if isinstance(content, str) else str(content)
                break

        if not query:
            return {**state, "memory_context": ""}

        ctx = await client.recall(query)
        return {**state, "memory_context": ctx or ""}

    async def store_node(state: dict) -> dict:
        """
        Save state["messages"] to long-term memory (side-effect only).

        Returns state unchanged. Place this as the last node before END
        or on the edge to END.
        """
        messages = state.get("messages", [])
        if not messages:
            return state

        # Normalise to plain dicts
        raw = []
        for m in messages:
            if isinstance(m, dict):
                raw.append({"role": m.get("role", "user"), "content": str(m.get("content", ""))})
            else:
                raw.append({"role": getattr(m, "role", "user"), "content": str(getattr(m, "content", ""))})

        await client.store(raw)
        return state

    return recall_node, store_node


# Module-level default nodes (session_id="" — override via make_memory_nodes)
async def recall_node(state: dict) -> dict:
    """Default recall node with empty session_id. Use make_memory_nodes() for sessions."""
    _recall, _ = make_memory_nodes()
    return await _recall(state)


async def store_node(state: dict) -> dict:
    """Default store node with empty session_id. Use make_memory_nodes() for sessions."""
    _, _store = make_memory_nodes()
    return await _store(state)
