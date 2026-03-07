"""
claw-memory framework adapters — v0.9.0

Thin HTTP wrappers that speak each framework's idiom.
The backend REST API is framework-agnostic; these are client-side only.

Quickstart
----------
    from adapters import MemoryClient           # raw async client
    from adapters import ClawMemory             # LangChain BaseChatMemory
    from adapters import make_memory_nodes      # LangGraph recall/store nodes
    from adapters import RecallMemoryTool       # CrewAI tool
    from adapters import ClawMemoryHook         # AutoGen hook
    from adapters import ClaudeMemorySession    # Direct Claude API
"""

from .base import MemoryClient
from .langchain import ClawMemory
from .langgraph import make_memory_nodes, recall_node, store_node
from .crewai import RecallMemoryTool, StoreMemoryCallback
from .autogen import ClawMemoryHook
from .claude import ClaudeMemorySession

__all__ = [
    "MemoryClient",
    "ClawMemory",
    "make_memory_nodes",
    "recall_node",
    "store_node",
    "RecallMemoryTool",
    "StoreMemoryCallback",
    "ClawMemoryHook",
    "ClaudeMemorySession",
]
