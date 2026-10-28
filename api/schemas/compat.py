"""API schemas for compat/bridge endpoints (agentmemory API surface)."""

from pydantic import BaseModel


class CompatSessionStartRequest(BaseModel):
    sessionId: str = ""
    session_id: str = ""
    project: str = ""
    cwd: str = ""
    user_id: str = ""
    agent_id: str = ""


class CompatSessionEndRequest(BaseModel):
    sessionId: str = ""
    session_id: str = ""
    project: str = ""
    cwd: str = ""


class ObserveRequest(BaseModel):
    hookType: str = ""
    sessionId: str = ""
    session_id: str = ""
    project: str = ""
    cwd: str = ""
    timestamp: str = ""
    data: dict = {}


class SummarizeRequest(BaseModel):
    sessionId: str = ""
    session_id: str = ""
    project: str = ""
    cwd: str = ""


class EnrichRequest(BaseModel):
    sessionId: str = ""
    session_id: str = ""
    project: str = ""
    cwd: str = ""
    files: list[str] = []
    query: str = ""


class ContextRequest(BaseModel):
    sessionId: str = ""
    session_id: str = ""
    project: str = ""
    cwd: str = ""
    query: str = ""
    token_budget: int = 1500
    user_id: str = ""
    agent_id: str = ""


class SessionCommitRequest(BaseModel):
    sessionId: str = ""
    session_id: str = ""
    sha: str = ""
    message: str = ""
    branch: str = ""
    repo: str = ""
    cwd: str = ""


class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    format: str = "full"
    token_budget: int | None = None
    sessionId: str = ""
    session_id: str = ""
    user_id: str = ""
    agent_id: str = ""


class RememberRequest(BaseModel):
    content: str
    type: str = "fact"
    concepts: str = ""
    files: str = ""
    sessionId: str = ""
    session_id: str = ""
    # Writes carry the same track the reader will query, or the row lands in
    # the unscoped corpus and a scoped recall will never see it back.
    user_id: str = ""
    agent_id: str = ""


class ForgetRequest(BaseModel):
    query: str
    limit: int = 10
    dry_run: bool = True
    user_id: str = ""
    agent_id: str = ""


class FileContextRequest(BaseModel):
    files: str
    sessionId: str = ""
    session_id: str = ""


class PatternsRequest(BaseModel):
    query: str = ""
    limit: int = 10


class SmartSearchRequest(BaseModel):
    query: str
    limit: int = 10
    depth: str = "auto"


class TimelineRequest(BaseModel):
    sessionId: str = ""
    session_id: str = ""
    from_: str = ""
    to: str = ""
    limit: int = 20


class ClaudeBridgeSyncRequest(BaseModel):
    sessionId: str = ""
    session_id: str = ""
    project: str = ""
    cwd: str = ""


class ImportRequest(BaseModel):
    data: dict = {}
    version: str = ""
    # Default track for restored rows; an item carrying its own ids keeps them.
    user_id: str = ""
    agent_id: str = ""
