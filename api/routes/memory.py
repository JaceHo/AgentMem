"""
API Routes - Memory operations (store/recall).

Extracted from main.py to provide clean separation between routing and business logic.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Any

from api.schemas.memory import RecallRequest, StoreRequest
from services import MemoryService, RetrievalService


router = APIRouter(prefix="/memory", tags=["memory"])


def get_memory_service() -> MemoryService:
    """Dependency injection for MemoryService."""
    from main import _redis, embedder, extractor, mem_store, _bm25_index, graph_mod
    return MemoryService(_redis, embedder, extractor, mem_store, _bm25_index, graph_mod)


def get_retrieval_service() -> RetrievalService:
    """Dependency injection for RetrievalService."""
    from main import _redis, embedder, _bm25_index, mem_store, graph_mod
    return RetrievalService(_redis, embedder, _bm25_index, mem_store, graph_mod)


@router.post("/store")
async def store_memory(
    request: StoreRequest,
    service: MemoryService = Depends(get_memory_service),
):
    """Store conversation messages and extract facts.
    
    Accepts messages and session_id, stores episode in Tier 2 memory,
    extracts facts via LLM, and stores them in semantic memory.
    """
    try:
        if not request.session_id or not request.messages:
            raise HTTPException(status_code=400, detail="session_id and messages required")

        episode_id = await service.store_episode(
            session_id=request.session_id,
            messages=[msg.dict() for msg in request.messages],
            metadata=request.metadata,
        )

        return {
            "status": "success",
            "episode_id": episode_id,
            "session_id": request.session_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Storage failed: {e}")


@router.post("/recall")
async def recall_memory(
    request: RecallRequest,
    service: RetrievalService = Depends(get_retrieval_service),
):
    """Retrieve relevant memories for a query.
    
    Performs hybrid search across vector, BM25, and graph sources,
    then formats results within token budget.
    """
    try:
        if not request.query or not request.session_id:
            raise HTTPException(status_code=400, detail="query and session_id required")

        result = await service.retrieve(
            query=request.query,
            session_id=request.session_id,
            memory_limit_number=request.memory_limit_number,
            token_budget=request.token_budget,
            include_tools=request.include_tools,
            include_procedures=request.include_procedures,
            include_graph=request.include_graph,
            auto_graph=request.auto_graph,
            enable_planning=request.enable_planning,
            enable_reflection=request.enable_reflection,
            time_from=request.time_from,
            time_to=request.time_to,
        )

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")
