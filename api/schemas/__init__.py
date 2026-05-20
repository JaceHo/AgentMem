"""API schema package for AgentMem."""

from api.schemas.memory import RecallRequest, Message, StoreRequest
from api.schemas.capability import (
    ToolDefinition, RegisterToolsRequest, EnvState,
    RecallToolsRequest, StoreProcedureRequest,
    ToolFeedbackRequest, ToolSequenceRequest, ProcedureFeedbackRequest,
)
from api.schemas.compat import (
    CompatSessionStartRequest, CompatSessionEndRequest,
    ObserveRequest, SummarizeRequest, EnrichRequest,
    ContextRequest, SessionCommitRequest,
    SearchRequest, RememberRequest, ForgetRequest,
    FileContextRequest, PatternsRequest, SmartSearchRequest,
    TimelineRequest, ClaudeBridgeSyncRequest, ImportRequest,
)
from api.schemas.consolidation import (
    CompactRequest, AnswerRequest, CompressSessionRequest,
    FeedbackRequest, CrystallizeRequest,
)
from api.schemas.graph import (
    GraphRecallRequest, TypedEdgeRequest, TraverseRequest, ReinforceRequest,
)

__all__ = [
    # Memory
    "RecallRequest", "Message", "StoreRequest",
    # Capability
    "ToolDefinition", "RegisterToolsRequest", "EnvState",
    "RecallToolsRequest", "StoreProcedureRequest",
    "ToolFeedbackRequest", "ToolSequenceRequest", "ProcedureFeedbackRequest",
    # Compat
    "CompatSessionStartRequest", "CompatSessionEndRequest",
    "ObserveRequest", "SummarizeRequest", "EnrichRequest",
    "ContextRequest", "SessionCommitRequest",
    "SearchRequest", "RememberRequest", "ForgetRequest",
    "FileContextRequest", "PatternsRequest", "SmartSearchRequest",
    "TimelineRequest", "ClaudeBridgeSyncRequest", "ImportRequest",
    # Consolidation
    "CompactRequest", "AnswerRequest", "CompressSessionRequest",
    "FeedbackRequest", "CrystallizeRequest",
    # Graph
    "GraphRecallRequest", "TypedEdgeRequest", "TraverseRequest", "ReinforceRequest",
]
