"""API schemas for core memory endpoints."""

from typing import Any

from pydantic import BaseModel, Field, field_validator

# The handler only keeps the last few turns. Rejecting longer transcripts
# dropped the whole Stop-hook store (422) once a session passed 50 messages.
_STORE_MESSAGE_CAP = 50


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
    messages: list[Message] = Field(max_length=_STORE_MESSAGE_CAP)
    session_id: str = ""
    metadata: dict[str, Any] | None = None

    @field_validator("messages", mode="before")
    @classmethod
    def _keep_recent_messages(cls, value: Any) -> Any:
        if isinstance(value, list) and len(value) > _STORE_MESSAGE_CAP:
            return value[-_STORE_MESSAGE_CAP:]
        return value
