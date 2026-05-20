"""Health, stats, and config endpoints."""

from fastapi import APIRouter

from api import state
from config.settings import settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    r = state.redis
    try:
        await r.ping()
        return {"status": "ok", "redis": "ok", "version": settings.app_version,
                "embedding": state.embedder.get_provider_info()}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@router.get("/livez")
async def livez():
    return {"status": "ok"}


@router.get("/stats")
async def stats():
    r = state.redis
    mem_store = state.mem_store
    try:
        counts = await mem_store.vcard(r)
        persona_raw = await r.hgetall("mem:persona")
        counts["persona_fields"] = len(persona_raw)

        tool_card = await r.execute_command("VCARD", state.cap_mod.TOOL_KEY)
        counts["tools"] = max(0, int(tool_card or 0) - 1)

        proc_card = await r.execute_command("VCARD", mem_store.PROC_KEY)
        counts["procedures"] = max(0, int(proc_card or 0) - 1)

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

        return counts
    except Exception as e:
        return {"error": str(e)}


@router.get("/config")
async def config():
    r = state.redis
    mem_store = state.mem_store
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
