"""KB routes — progressive-disclosure knowledge base.

  POST /kb/ingest            parse + chunk + enrich + digest a file/dir under ~
  GET  /kb/index             Tier 0: one line per document
  GET  /kb/doc/{id}/digest   Tier 1: summary + section map
  GET  /kb/doc/{id}/raw      original path + metadata (content stays on disk)
  POST /kb/search            Tier 2: hybrid vector+BM25 chunk retrieval
  POST /kb/remove            remove a document (and its chunks/index entries)
  GET  /kb/status            counts + BM25 warm state
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api import state
from core import dim_guard
from core import embedder
from core import kb as kb_mod

log = logging.getLogger("mem")
router = APIRouter(prefix="/kb", tags=["kb"])


def _redis():
    if state.redis is not None:
        return state.redis
    from main import _get_redis  # late import avoids circularity; honours test patching
    return _get_redis()


class KbIngestRequest(BaseModel):
    path: str = Field(min_length=1)
    enrich: bool = True
    force: bool = False
    library: str = ""


class KbSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = 6
    library: str = ""


class KbRemoveRequest(BaseModel):
    doc: str = Field(min_length=1, description="doc id or absolute path")


def _kb():
    return kb_mod.get_kb()


@router.post("/ingest")
async def kb_ingest(req: KbIngestRequest):
    try:
        return await _kb().ingest_path(_redis(), req.path, enrich=req.enrich,
                                       force=req.force, library=req.library)
    except embedder.DimensionMismatch as exc:
        raise HTTPException(status_code=409,
                            detail={"error": "dimension_mismatch", "message": str(exc)})
    except kb_mod.KBError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/index")
async def kb_index():
    """Tier 0 — compact enough to inject into an agent system prompt."""
    try:
        return {"docs": await _kb().index(_redis())}
    except Exception as exc:  # noqa: BLE001
        log.error("[kb/index] %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/doc/{doc_id}/digest")
async def kb_digest(doc_id: str):
    """Tier 1 — description + markdown digest + section map."""
    meta = await _kb().digest(_redis(), doc_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"doc {doc_id} not found")
    return meta


@router.get("/doc/{doc_id}/raw")
async def kb_raw(doc_id: str):
    meta = await _kb().digest(_redis(), doc_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"doc {doc_id} not found")
    return {"id": meta["id"], "path": meta.get("path", ""), "title": meta.get("title", ""),
            "sections": meta.get("sections", []), "size": meta.get("size", 0),
            "note": "original content stays on disk — read it with a file tool"}


@router.post("/search")
async def kb_search(req: KbSearchRequest):
    """Tier 2 — hybrid chunk retrieval (vector + BM25, wRRF fused)."""
    try:
        return {"query": req.query,
                "chunks": await _kb().search(_redis(), req.query, k=req.k, library=req.library)}
    except Exception as exc:  # noqa: BLE001
        log.error("[kb/search] %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/remove")
async def kb_remove(req: KbRemoveRequest):
    kb = _kb()
    r = _redis()
    doc_id = req.doc
    if not doc_id.startswith("kb_"):
        from core.utils import decode_bytes
        doc_id = decode_bytes(await r.hget(kb_mod.KB_PATH_KEY, req.doc))
    if not doc_id:
        raise HTTPException(status_code=404, detail=f"no kb document for {req.doc}")
    removed = await kb.remove_doc(r, doc_id)
    return {"status": "removed" if removed else "absent", "doc": doc_id}


@router.get("/status")
async def kb_status():
    r = _redis()
    docs = await r.hlen(kb_mod.KB_DOC_KEY)
    chunks = await r.hlen(kb_mod.KB_TEXT_KEY)
    # Docs written by another embedding provider are not comparable with today's
    # queries; they still occupy the vectorset and, left unreported, make search
    # look merely bad instead of broken. Re-ingest with force to clear this.
    import json as _json
    current = kb_mod.embed_signature()
    stale = 0
    for raw in await r.hvals(kb_mod.KB_DOC_KEY):
        try:
            if _json.loads(raw).get("embed_sig", "") != current:
                stale += 1
        except Exception:  # noqa: BLE001 — a corrupt meta row is stale by definition
            stale += 1
    # The width the model actually produces, next to the width the sets were built
    # with — a mismatch here is the one failure mode that blocks every write, so it
    # belongs in the status payload instead of only in the boot log.
    try:
        guard = await dim_guard.snapshot(r)
        dims_block = {"dims": guard["dims"], "dims_verified": guard["verified"],
                      "dims_mismatched": sorted(guard["bad"])}
    except Exception as exc:                                  # noqa: BLE001
        dims_block = {"dims": None, "dims_verified": False, "dims_mismatched": [],
                      "dims_error": str(exc)[:120]}
    return {"docs": docs, "chunks": chunks,
            "embed_sig": current, "stale_embed": stale,
            "bm25_ready": len(kb_mod.get_kb().bm25._docs) if hasattr(kb_mod.get_kb().bm25, "_docs") else None,
            # What the BM25 index was actually BUILT from. warm() appends to the
            # doc list and rebuilds asynchronously in 10-doc batches, so the tail
            # of the warm-up can stay out of the index — and bm25_ready above
            # keeps reporting the full list length while it does.
            "bm25_indexed": getattr(kb_mod.get_kb().bm25, "_built_at_len", None),
            **dims_block}
