"""API schemas for core memory endpoints."""

from typing import Any

from pydantic import BaseModel


class RecallRequest(BaseModel):
    query: str
    session_id: str = ""
    memory_limit_number: int = 12
    include_tools: bool = True
    include_procedures: bool = False
    include_graph: bool = False
    auto_graph: bool = True
    enable_planning: bool = False
    enable_hyde: bool = True
    enable_reflection: bool = False
    token_budget: int | None = None
    time_from: int | None = None
    time_to: int | None = None


class Message(BaseModel):
    role: str
    content: Any


class StoreRequest(BaseModel):
    messages: list[Message]
    session_id: str = ""
    metadata: dict[str, Any] | None = None
