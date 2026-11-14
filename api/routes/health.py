"""Health, stats, and config endpoints."""

from fastapi import APIRouter, HTTPException

from api import state
from config.settings import settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    """Liveness + honest warmup state.

    Deliberately 200 even while warming: this process IS alive, it is just not
    finished loading the embedding model / BM25 index yet. Returning "degraded"
    (or refusing the connection, as it used to) made supervisors kill a service
    that was merely booting -- see core/warmup.py. Callers that need to know
    whether real work will succeed must use /ready.
    """
    from core import warmup as warmup_mod

    snap = warmup_mod.snapshot()
    r = state.redis
    if not snap["ready"]:
        return {"status": "starting", "redis": "pending" if r is None else "ok",
                "version": settings.app_version, "warmup": snap}
    try:
        await r.ping()
        return {"status": "ok", "redis": "ok", "version": settings.app_version,
                "embedding": state.embedder.get_provider_info(), "warmup": snap}
    except Exception as e:
        return {"status": "degraded", "redis": "error", "error": str(e),
                "version": settings.app_version, "warmup": snap}


@router.get("/ready")
async def ready():
    """Readiness: 200 once core work (redis/indexes/kb) can be served.

    ``embedding_ready`` is reported separately: the model loads in a background
    stage and the embedder's lazy loader covers requests that arrive before it
    finishes. Callers that require a warm model can poll this field instead of
    blocking readiness — a stats/graph/metadata request must not wait behind a
    20 s torch import.
    """
    from fastapi.responses import JSONResponse

    from core import warmup as warmup_mod

    snap = warmup_mod.snapshot()
    if warmup_mod.is_ready():
        return {"status": "ready", "embedding_ready": snap["embedding_ready"],
                "warmup": snap}
    resp = JSONResponse(
        status_code=503,
        content={"status": snap["phase"], "ready": False, "retry_after_s": 5,
                 "warmup": snap},
    )
    resp.headers["Retry-After"] = "5"
    return resp


@router.get("/livez")
async def livez():
    return {"status": "ok"}


@router.get("/stats")
async def stats():
    r = state.redis
    if r is None:
        raise HTTPException(status_code=503, detail="Redis not connected")
    try:
        counts = await state.mem_store.vcard(r)
        persona_raw = await r.hgetall("mem:persona")
        counts["persona_fields"] = len(persona_raw)

        tool_card = await r.execute_command("VCARD", state.cap_mod.TOOL_KEY)
        counts["tools"] = max(0, int(tool_card or 0) - 1)

        proc_card = await r.execute_command("VCARD", state.mem_store.PROC_KEY)
        counts["procedures"] = max(0, int(proc_card or 0) - 1)

        # Deferred-by-outage writes are only "not lost" if the backlog is observable: a queue
        # nobody reads is the same silence that hid the dropped stores for 2 days.
        try:
            from api.routes import capability as _cap
            counts["procedures_pending"] = int(await r.llen(_cap.PROC_PENDING_KEY) or 0)
            counts["proc_guard"] = _cap.proc_guard_stats()
        except Exception as exc:
            counts["procedures_pending"] = -1
            counts["proc_guard"] = {"error": str(exc)[:80]}

        env_raw = await r.hgetall(state.cap_mod.ENV_KEY)
        counts["env_fields"] = len(env_raw)

        w_attempts = await state.store_attempts.get()
        w_successes = await state.store_successes.get()
        w_skips = await state.store_skips.get()
        w_errors = await state.store_errors.get()
        w_latency = await state.store_latency_sum_ms.get()
        completed = w_successes + w_errors
        counts["writer"] = {
            "attempts": w_attempts,
            "successes": w_successes,
            "skips": w_skips,
            "errors": w_errors,
            "success_rate": round(w_successes / completed, 3) if completed > 0 else None,
            "avg_ms": round(w_latency / w_attempts) if w_attempts > 0 else None,
        }

        # Heat-bump observability. ``missing`` > 0 means bumps are being keyed on
        # element names that do not exist (the bytes-repr regression that left
        # access_count at 0 on every fact); this is the counter that makes a
        # silent no-op impossible to miss a second time.
        counts["heat"] = state.heat_mod.stats_snapshot()

        return counts
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config")
async def config():
    r = state.redis
    consolidations = await state.stores_since_consolidation.get()
    return {
        "version": settings.app_version,
        "embedding_model": state.embedder.get_provider_info(),
        "auto_consolidate_every": state.AUTO_CONSOLIDATE_EVERY,
        "stores_since_consolidation": consolidations,
        "redis_connected": r is not None,
        "settings": {
            "dedup_similarity_threshold": settings.dedup_similarity_threshold,
            "default_token_budget": settings.default_token_budget,
            "bm25_top_k": settings.bm25_top_k,
            "rrf_k_constant": settings.rrf_k_constant,
        },
    }


@router.get("/config/flags")
async def config_flags():
    return {
        "auto_consolidate": True,
        "auto_graph_expansion": True,
        "bm25_hybrid": True,
        "ebbinghaus_decay": True,
        "superseded_tracking": True,
    }
