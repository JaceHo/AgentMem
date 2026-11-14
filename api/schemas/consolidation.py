"""API schemas for consolidation and lifecycle endpoints."""

from typing import Any

from pydantic import AliasChoices, BaseModel, Field, field_validator


class CompactRequest(BaseModel):
    # Hooks/plugins send session_id | sessionId; null must not 400 the PostToolUse storm.
    session_id: str = Field(
        default="",
        validation_alias=AliasChoices("session_id", "sessionId", "session"),
    )
    force: bool = False
    threshold_chars: int = 3000

    @field_validator("session_id", mode="before")
    @classmethod
    def _coerce_session_id(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v).strip()


class AnswerRequest(BaseModel):
    query: str
    context: str = ""
    session_id: str = ""


class CompressSessionRequest(BaseModel):
    session_id: str
    wait: bool = False
    # Optional: when omitted, the track recorded when the session was stored
    # decides where the promoted rows land.
    user_id: str = ""
    agent_id: str = ""


class FeedbackRequest(BaseModel):
    element_id: str
    rating: int              # 1-5 stars (1-2: negative, 3: neutral, 4-5: positive)
    reason: str = ""
    comment: str = ""


class CrystallizeRequest(BaseModel):
    session_id: str
    max_facts: int = 20
