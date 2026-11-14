"""Active working memory + prospective intentions HTTP API."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api import state
from core import focus as focus_mod

router = APIRouter(tags=["focus"])


class ActiveBody(BaseModel):
    session_id: str = Field(default="default", max_length=128)
    goal: str | None = Field(default=None, max_length=2000)
    notes: str | None = Field(default=None, max_length=4000)


class IntentionBody(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    cue: str = Field(default="", max_length=500)
    due_at: int | None = None
    session_id: str = Field(default="", max_length=128)


class IntentionStatusBody(BaseModel):
    status: str = Field(description="pending|done|cancel")


@router.get("/active-memory")
async def get_active_memory(session_id: str = "default"):
    r = state.redis
    if r is None:
        raise HTTPException(503, "Redis not connected")
    return await focus_mod.get_active(r, session_id)


@router.post("/active-memory")
async def set_active_memory(body: ActiveBody):
    r = state.redis
    if r is None:
        raise HTTPException(503, "Redis not connected")
    return await focus_mod.set_active(r, body.session_id, goal=body.goal, notes=body.notes)


@router.get("/intentions")
async def get_intentions(status: str = "pending", limit: int = 50):
    r = state.redis
    if r is None:
        raise HTTPException(503, "Redis not connected")
    st = status if status in ("pending", "done", "cancel", "all") else "pending"
    return {
        "intentions": await focus_mod.list_intentions(
            r, status=None if st == "all" else st, limit=min(limit, 200)
        )
    }


@router.post("/intentions")
async def create_intention(body: IntentionBody):
    r = state.redis
    if r is None:
        raise HTTPException(503, "Redis not connected")
    item = await focus_mod.add_intention(
        r, text=body.text, cue=body.cue, due_at=body.due_at, session_id=body.session_id
    )
    return {"ok": True, "intention": item}


@router.post("/intentions/{intention_id}/status")
async def set_intention_status(intention_id: str, body: IntentionStatusBody):
    r = state.redis
    if r is None:
        raise HTTPException(503, "Redis not connected")
    try:
        item = await focus_mod.update_intention(r, intention_id, status=body.status)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    if not item:
        raise HTTPException(404, "intention not found")
    return {"ok": True, "intention": item}
