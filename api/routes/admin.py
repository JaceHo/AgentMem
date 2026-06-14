"""Admin routes — consolidation, crystallization, feedback, fact management, lifecycle."""

import logging
import time

import numpy as np
from fastapi import APIRouter, BackgroundTasks

from api import state
from api.schemas.consolidation import CrystallizeRequest, FeedbackRequest
from core import embedder
from core import store as mem_store
from core.utils import decode_bytes, decode_attrs

log = logging.getLogger("mem")
router = APIRouter(tags=["admin"])


@router.post("/consolidate")
async def consolidate(background_tasks: BackgroundTasks):
    """Trigger memory consolidation — merges similar facts to reduce redundancy."""
    from services.consolidation_service import do_consolidate
    background_tasks.add_task(do_consolidate, state.redis, state.bm25_index, state.spawn)
    return {"status": "consolidation_queued"}


@router.post("/consolidate/sync")
async def consolidate_sync():
    """Synchronous consolidation — returns results immediately."""
    from services.consolidation_service import do_consolidate
    result = await do_consolidate(state.redis, state.bm25_index, state.spawn)
    return result


@router.post("/consolidate/hard-prune")
async def hard_prune(background_tasks: BackgroundTasks):
    """Physical VREM of soft-deleted entries (>7 days) and stale episodes (>180 days)."""
    from services.consolidation_service import do_hard_prune
    background_tasks.add_task(do_hard_prune, state.redis, state.bm25_index, state.spawn)
    return {"status": "hard_prune_queued"}


@router.post("/admin/delete-facts-by-content")
async def admin_delete_facts_by_content(pattern: str = "Always send a report"):
    """Delete facts whose content contains the given pattern (admin only)."""
    r = state.redis
    card = await r.execute_command("VCARD", mem_store.FACT_KEY)
    if not card or int(card) <= 1:
        return {"deleted": 0, "message": "no facts to scan"}

    seed = np.zeros(embedder._get_provider().dims, dtype=np.float32)
    results = await r.execute_command(
        "VSIM", mem_store.FACT_KEY, "FP32", seed.tobytes(),
        "COUNT", min(int(card), 5000), "WITHSCORES", "WITHATTRIBS"
    )
    deleted = 0
    i = 0
    while i + 2 < len(results):
        elem = results[i]; raw = results[i + 2]; i += 3
        elem_str = decode_bytes(elem)
        if elem_str == "__seed__":
            continue
        try:
            attrs = decode_attrs(raw)
        except Exception:
            continue
        content = attrs.get("content", "")
        if pattern.lower() in content.lower():
            await r.execute_command("VREM", mem_store.FACT_KEY, elem_str)
            deleted += 1
    return {"deleted": deleted, "pattern": pattern}


@router.post("/feedback")
async def provide_feedback(req: FeedbackRequest):
    """User rates memory relevance (1-5 stars). Adjusts importance and confidence."""
    r = state.redis
    if req.rating < 1 or req.rating > 5:
        return {"error": "rating must be 1-5"}

    try:
        attrs = await mem_store.get_attrs(r, mem_store.FACT_KEY, req.element_id)
        if not attrs:
            return {"status": "not_found", "element_id": req.element_id}

        if req.rating >= 4:
            old_importance = attrs.get("importance", 0.5)
            attrs["importance"] = min(1.0, old_importance * 1.2)
            attrs["user_rating"] = req.rating
            attrs["user_rating_ts"] = int(time.time() * 1000)
            if req.comment:
                attrs["user_comment"] = req.comment[:500]
            await mem_store.reinforce_fact(r, req.element_id, source=f"user_feedback_{req.rating}")
            await mem_store.set_attrs(r, mem_store.FACT_KEY, req.element_id, attrs)
            log.info(f"[feedback] positive rating {req.rating}/5 for {req.element_id}, "
                    f"importance {old_importance:.2f} → {attrs['importance']:.2f}")
            return {
                "status": "ok", "element_id": req.element_id,
                "new_importance": attrs["importance"], "action": "boosted_and_reinforced"
            }

        elif req.rating <= 2:
            old_importance = attrs.get("importance", 0.5)
            attrs["importance"] = max(0.05, old_importance * 0.7)
            attrs["user_rating"] = req.rating
            attrs["user_rating_ts"] = int(time.time() * 1000)
            attrs["needs_review"] = True
            if req.comment:
                attrs["user_comment"] = req.comment[:500]
            await mem_store.set_attrs(r, mem_store.FACT_KEY, req.element_id, attrs)
            log.info(f"[feedback] negative rating {req.rating}/5 for {req.element_id}, "
                    f"importance {old_importance:.2f} → {attrs['importance']:.2f}")
            return {
                "status": "ok", "element_id": req.element_id,
                "new_importance": attrs["importance"], "action": "reduced_and_flagged_for_review"
            }

        else:
            attrs["user_rating"] = req.rating
            attrs["user_rating_ts"] = int(time.time() * 1000)
            if req.comment:
                attrs["user_comment"] = req.comment[:500]
            await mem_store.set_attrs(r, mem_store.FACT_KEY, req.element_id, attrs)
            return {
                "status": "ok", "element_id": req.element_id,
                "action": "recorded_neutral_rating"
            }

    except Exception as e:
        log.error(f"[feedback] error processing feedback: {e}")
        return {"error": str(e)}


@router.post("/facts/{element_id}/pin")
async def pin_fact(element_id: str):
    """Pin fact permanently (importance = 1.0, never pruned)."""
    r = state.redis
    try:
        attrs = await mem_store.get_attrs(r, mem_store.FACT_KEY, element_id)
        if not attrs:
            return {"status": "not_found", "element_id": element_id}

        attrs["pinned"] = True
        attrs["pinned_at"] = int(time.time() * 1000)
        attrs["importance"] = 1.0

        await mem_store.set_attrs(r, mem_store.FACT_KEY, element_id, attrs)
        log.info(f"[pin] pinned fact {element_id}: {attrs.get('content', '')[:80]}")
        return {"status": "ok", "element_id": element_id, "action": "pinned_permanently"}

    except Exception as e:
        log.error(f"[pin] error pinning fact: {e}")
        return {"error": str(e)}


@router.delete("/facts/{element_id}")
async def delete_fact(element_id: str):
    """User-initiated hard delete (immediate VREM)."""
    r = state.redis
    try:
        attrs = await mem_store.get_attrs(r, mem_store.FACT_KEY, element_id)
        if not attrs:
            return {"status": "not_found", "element_id": element_id}

        content_preview = attrs.get("content", "")[:80]

        await r.execute_command("VREM", mem_store.FACT_KEY, element_id)

        # Incremental BM25 update instead of full reset
        await state.bm25_index.remove(element_id)

        log.info(f"[delete] user-deleted fact {element_id}: {content_preview}")
        return {"status": "ok", "element_id": element_id, "action": "hard_deleted"}

    except Exception as e:
        log.error(f"[delete] error deleting fact: {e}")
        return {"error": str(e)}


@router.get("/facts/{element_id}/metadata")
async def get_fact_metadata(element_id: str):
    """Get full metadata for a fact including user ratings, pins, lifecycle info."""
    r = state.redis
    try:
        attrs = await mem_store.get_attrs(r, mem_store.FACT_KEY, element_id)
        if not attrs:
            return {"status": "not_found", "element_id": element_id}

        eff_conf = mem_store.confidence_decay(attrs)

        return {
            "element_id": element_id,
            "content": attrs.get("content", "")[:500],
            "category": attrs.get("category", ""),
            "importance": attrs.get("importance", 0.5),
            "confidence": attrs.get("confidence", 0.8),
            "effective_confidence": eff_conf,
            "source_count": attrs.get("source_count", 1),
            "access_count": attrs.get("access_count", 0),
            "created_at": attrs.get("ts", 0),
            "last_confirmed_ts": attrs.get("last_confirmed_ts", 0),
            "version": attrs.get("version", 1),
            "pinned": attrs.get("pinned", False),
            "user_rating": attrs.get("user_rating"),
            "user_comment": attrs.get("user_comment"),
            "needs_review": attrs.get("needs_review", False),
            "superseded_by": attrs.get("superseded_by", ""),
            "superseded_reason": attrs.get("superseded_reason", ""),
        }

    except Exception as e:
        return {"error": str(e), "element_id": element_id}


@router.get("/lifecycle/stats")
async def lifecycle_stats():
    """Return lifecycle statistics: confidence distribution, supersession counts, etc."""
    r = state.redis
    card = await r.execute_command("VCARD", mem_store.FACT_KEY)
    if not card or int(card) <= 1:
        return {"total": 0}

    seed = np.zeros(embedder._get_provider().dims, dtype=np.float32)
    results = await r.execute_command(
        "VSIM", mem_store.FACT_KEY, "FP32", seed.tobytes(),
        "COUNT", min(int(card), 5000), "WITHSCORES", "WITHATTRIBS"
    )

    active = 0
    superseded = 0
    conf_buckets = {"high": 0, "medium": 0, "low": 0, "expired": 0}
    reason_counts: dict[str, int] = {}
    category_health: dict[str, dict] = {}

    i = 0
    while i + 2 < len(results):
        raw = results[i + 2]; i += 3
        try:
            attrs = decode_attrs(raw)
        except Exception:
            continue
        if attrs.get("_seed") or not attrs.get("content"):
            continue

        if attrs.get("superseded_by"):
            superseded += 1
            reason = attrs.get("superseded_reason", "")
            # Infer reason from superseded_by value when reason field is missing (pre-v1.1 facts)
            if not reason:
                sb = attrs.get("superseded_by", "")
                if sb == "pruned":
                    reason = "pruned"
                elif sb:
                    reason = "merged"
                else:
                    reason = "unknown"
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            continue

        active += 1
        eff = mem_store.confidence_decay(attrs)
        if eff >= 0.7:
            conf_buckets["high"] += 1
        elif eff >= 0.4:
            conf_buckets["medium"] += 1
        elif eff >= 0.1:
            conf_buckets["low"] += 1
        else:
            conf_buckets["expired"] += 1

        cat = attrs.get("category", "general")
        if cat not in category_health:
            category_health[cat] = {"count": 0, "avg_confidence": 0.0, "total_conf": 0.0}
        category_health[cat]["count"] += 1
        category_health[cat]["total_conf"] += eff

    for cat in category_health:
        c = category_health[cat]
        c["avg_confidence"] = round(c["total_conf"] / max(1, c["count"]), 4)
        del c["total_conf"]

    return {
        "total": active + superseded,
        "active": active,
        "superseded": superseded,
        "confidence_distribution": conf_buckets,
        "supersession_reasons": reason_counts,
        "category_health": category_health,
    }


@router.post("/crystallize")
async def crystallize(req: CrystallizeRequest):
    """Crystallize a session — distill it into a structured digest."""
    from services.consolidation_service import crystallize_session
    digest = await crystallize_session(state.redis, req.session_id, max_facts=req.max_facts)
    return digest
