"""API schemas for knowledge graph endpoints."""

from pydantic import BaseModel


class GraphRecallRequest(BaseModel):
    entity: str
    depth: int = 2
    max_facts: int = 10


class TypedEdgeRequest(BaseModel):
    source_entity: str
    target_entity: str
    relationship_type: str   # uses | depends_on | contradicts | caused | fixed | supersedes
    confidence: float = 0.8
    source: str = ""         # optional provenance
    bidirectional: bool = False


class TraverseRequest(BaseModel):
    start_entity: str
    relation: str = ""
    direction: str = "out"  # out | in | both
    max_depth: int = 2
    max_nodes: int = 20


class ReinforceRequest(BaseModel):
    reason: str = "user_confirmed"
