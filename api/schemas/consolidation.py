"""API schemas for consolidation and lifecycle endpoints."""

from pydantic import BaseModel


class CompactRequest(BaseModel):
    session_id: str = ""
    force: bool = False
    threshold_chars: int = 3000


class AnswerRequest(BaseModel):
    query: str
    context: str = ""
    session_id: str = ""


class CompressSessionRequest(BaseModel):
    session_id: str
    wait: bool = False


class FeedbackRequest(BaseModel):
    element_id: str
    rating: int              # -1 (negative) | 0 (neutral) | 1 (positive)
    reason: str = ""
    comment: str = ""


class CrystallizeRequest(BaseModel):
    session_id: str
    max_facts: int = 20
