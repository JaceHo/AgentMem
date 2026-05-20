"""API schemas for consolidation and lifecycle endpoints."""

from pydantic import BaseModel


class CompactRequest(BaseModel):
    session_id: str = ""
    force: bool = False


class AnswerRequest(BaseModel):
    query: str
    session_id: str = ""


class CompressSessionRequest(BaseModel):
    session_id: str


class FeedbackRequest(BaseModel):
    fact_id: str
    rating: int              # -1 (negative) | 0 (neutral) | 1 (positive)
    reason: str = ""


class CrystallizeRequest(BaseModel):
    session_id: str
    max_facts: int = 20
