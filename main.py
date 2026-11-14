"""
AgentMem — Local Agent Memory Service  v1.1.0
==============================================
Replaces memos-cloud-openclaw-plugin with fully local, persistent memory.

2026 Layered Controlled Architecture (Tier 0-2):
  Tier 0 — Working memory    : in-context window (LLM prompt)
  Tier 1 — Session KV        : rolling accumulated session summary (Redis String, 4h TTL)
  Tier 2 — Long-term vectors : episodic + semantic + procedural (Redis HNSW, permanent)
  +       — Capability layer : tool/env/agent self-model (Redis Hash + vectorset)

Features (v1.1.0 — Refactored with atomic counters + centralized config):
  1. Heat-tiered recall        — frequently/recently accessed memories rank higher
  2. Scene isolation           — language + domain tags filter recall by context
  3. Evolving persona          — structured user profile updated from extracted facts
  4. Summarize-then-embed      — long turns compressed before MiniLM embedding
  5. Hybrid extraction         — regex (instant) + LLM (GLM-4-flash, async) for facts
  6. Agent capability registry — tool/skill/env/agent self-model (v0.6.0)
  7. Procedural memory         — 4th cognitive tier: how-to workflows (v0.6.0)
  8. Session promotion         — Tier 1→2 compression at session end (v0.7.0)
  9. SimpleMem Sec 3.1         — lossless restatement: Φ_coref (no pronouns) +
                                 Φ_time (absolute ISO 8601), topic/location/importance (v0.9.1)
 10. SimpleMem Sec 3.2         — 3-phase consolidation: Decay (×0.9/90d) → Merge
                                 (soft-delete via superseded_by) → Prune (<0.05) (v0.9.1)
 11. SimpleMem Sec 3.3         — Intent-Aware Retrieval Planning: plan_queries() +
                                 check_sufficiency() reflection loop (v0.9.1)
 12. Token-budget context      — greedy word-count packing into <cross_session_memory>
                                 tags; bug-fixed header accounting (v0.9.2)
 13. Multi-framework adapters  — LangChain, LangGraph, CrewAI, AutoGen, Claude API (v0.9.0)
 14. MCP server                — mcp_server.py for Claude Desktop/Code/Cursor (v0.9.0)
 15. Knowledge graph           — entity relationship graph in Redis Sets (v0.9.0)
 16. Auto-consolidation        — counter-triggered (every 50 stores) + hourly timer (v0.9.0)
 17. A-MAC admission gate      — 5-factor gate: semantic_novelty + entity_novelty +
                                 factual_confidence + temporal_signal + content_type_prior (v0.9.2)
 18. Equal-weight RRF fusion   — hand-tuned per-query-type wRRF presets removed after a
                                 7-scenario sweep lost nDCG@10 in 6/7 scenarios (v1.1)
 19. Category importance floors — 15-tier floor map; identity/rule pinned ≥0.80 (v0.9.2)
 20. Episode type taxonomy    — 8-type classification per claude-mem: bugfix|feature|
                                 discovery|decision|change|procedure|preference|context (v0.9.3)
 21. Causal episode chaining  — doubly-linked prev/next episode IDs in Redis attrs,
                                 tracks temporal narrative flow across sessions (v0.9.3)
 22. Mid-session compact      — /session/compact: LLM-compress Tier 1 KV when >3000
                                 chars, PostToolUse hook keeps context O(N) (v0.9.3)
 23. Session handoff bridge  — /session/compress pins last summary; always injected
                                 at next session start for cross-session continuity (v1.0)
 24. Hard-delete pruning     — daily VREM of superseded facts (>7d) and stale
                                 episodes (>180d, unaccessed); keeps HNSW fresh (v1.0)
 25. Auto graph expansion    — graph neighbourhood auto-triggers on entity queries
                                 without explicit include_graph=True flag (v1.0)
 26. Semantic triple extraction — Memori-style (s,p,o) triples extracted alongside
                                 lossless facts; triple_str stored for BM25 coverage (v1.1)
 27. Real BM25 hybrid search  — rank_bm25 BM25Okapi over in-memory fact corpus;
                                 4th RRF input alongside vector/symbolic/global (v1.1)
 28. A-MEM memory evolution   — near-duplicate facts (cos 0.80-0.95) trigger async
                                 keyword/topic enrichment on existing fact (v1.1)
 29. Dual-layer fact linking  — source_episode_id links each fact to source episode
                                 for Memori-style narrative context retrieval (v1.1)
 30. Atomic counters          — thread-safe state management (v1.1 refactor)
 31. Centralized config       — environment-variable tunable parameters (v1.1 refactor)

Endpoints:
  POST /recall              — before_agent_start hook (includes tool context + graph)
  POST /store               — agent_end hook (async, non-blocking)
  POST /session/compress    — promote Tier 1 session → Tier 2 long-term
  GET  /session/{id}        — inspect current Tier 1 session state
  POST /register-tools      — register agent's tool/skill index
  POST /register-env        — register current environment state
  POST /recall-tools        — semantic search over tool index
  POST /recall-procedures   — semantic search over procedural memory
  POST /store-procedure     — manually store a workflow/how-to
  GET  /capabilities        — full capability manifest
  GET  /graph/{entity}      — knowledge graph: entity neighbours + fact counts
  POST /graph/recall        — knowledge graph: neighbourhood fact retrieval
  GET  /graph/stats         — knowledge graph: node/edge counts
  GET  /config              — service configuration + auto-consolidation state
  POST /session/compact     — mid-session compress Tier 1 KV if > threshold (v0.9.3)
  POST /tool-feedback       — ToolMem: record success/fail per tool invocation (v0.9.5)
  POST /record-tool-sequence — AutoTool TIG: record tool transition sequence (v0.9.5)
  GET  /tool-graph/{name}   — AutoTool TIG: outgoing transitions from tool (v0.9.5)
  POST /tool-graph/detect-meta-tools — AWO: synthesize composite procedures from TIG chains (v0.9.6)
  POST /procedure-feedback  — MACLA Beta: record procedure success/fail (v0.9.5)
  GET  /tool-procedures/{name} — reverse index: procedures that use a given tool (v0.9.6)
  POST /proc-backfill-index — backfill proc_by_tool reverse index for existing procedures (v0.9.6)
  POST /consolidate/hard-prune — manual trigger: VREM soft-deleted + stale entries (v1.0)
  GET  /health
  GET  /stats
"""

import asyncio
import json
import hmac
import logging
import re
import time
from contextlib import asynccontextmanager

import httpx
import numpy as np
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ── Schemas (extracted to api/schemas/) ──────────────────────────────────────
from api.schemas.memory import RecallRequest, Message, StoreRequest
from api.schemas.consolidation import (
    CompactRequest, AnswerRequest, CompressSessionRequest,
)

# ── Search utilities (extracted to core/search.py) ───────────────────────────
from core.search import (
    encode as _encode,
    vscan as _vscan,
    populate_bm25_from_redis,
    async_encode,
    async_encode_batch,
)

# ── Shared state (single source of truth) ────────────────────────────────────
from api import state

# ── Import refactored modules ────────────────────────────────────────────────
from config.settings import settings
from utils.text_processing import (
    flatten_message_content,
    estimate_tokens,
    truncate_to_tokens,
    redact_secrets,
    contains_secret,
    strip_platform_noise,
    is_only_platform_noise,
    is_brief_option_reply,
    is_trivial,
    is_injected_system_content,
)
from core.utils import decode_bytes as _decode_bytes, decode_attrs as _decode_attrs, decode_json as _decode_json

# Private aliases — used throughout main.py with underscore prefix
_is_injected = is_injected_system_content
_strip_platform_noise = strip_platform_noise
_is_only_platform_noise = is_only_platform_noise
_is_brief_option_reply = is_brief_option_reply
_contains_secret = contains_secret
_redact_secrets = redact_secrets
_is_trivial = is_trivial
_flatten_message_content = flatten_message_content
_estimate_tokens = estimate_tokens

from core import capability as cap_mod
from core import dim_guard
from core import embedder
from core import extractor
from core import graph as graph_mod
from core import heat as heat_mod
from core import log_sse
from core import persona as persona_mod
from core import retrieval_planner
from core import scene as scene_mod
from core import store as mem_store
from core.scope import BadScope, Scope, scope_of, strict_scope
from core import summarizer
from core.http import async_post_json
from services.consolidation_service import (
    do_consolidate as _do_consolidate,
    do_hard_prune as _do_hard_prune,
    crystallize_session_inline as _crystallize_session_inline,
)

logging.basicConfig(level=getattr(logging, settings.log_level), format=settings.log_format)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Startup forensics: how long did the module imports above take? Under memory
# pressure the torch/fastapi import chain was suspected of eating the first
# minutes of boot (probe-timeout mystery: TCP accepted but no HTTP response).
# _BOOT_T0 is captured as early as possible; lifespan logs the split.
_BOOT_T0 = time.monotonic()


class _ConciseNotificationFilter(logging.Filter):
    """The MCP SDK logs a multi-KB pydantic validation dump (WARNING) for every
    unknown client notification — Trae sends custom notifications/trae/* types
    on every session stop. Unknown notifications are ignored by design per the
    MCP spec, so collapse the dump to a one-line DEBUG record."""

    _marker = "Failed to validate notification"

    def filter(self, record: logging.LogRecord) -> bool:
        if self._marker not in record.getMessage():
            return True
        import re as _re

        m = _re.search(r"input_value='([^']+)'", record.getMessage())
        record.levelno = record.level = logging.DEBUG
        record.levelname = logging.getLevelName(logging.DEBUG)
        record.msg = "MCP: ignoring unknown client notification %s" % (
            m.group(1) if m else "(unparseable)",
        )
        record.args = None
        return True


_concise_filter = _ConciseNotificationFilter()
logging.getLogger().addFilter(_concise_filter)
for _h in logging.getLogger().handlers:
    _h.addFilter(_concise_filter)

log = logging.getLogger("mem")
APP_VERSION = settings.app_version

# Attach SSE log handler — broadcasts every record to dashboard clients,
# stores it in an in-memory ring buffer, and persists it to
# ~/.agentmem/logs/dashboard.jsonl so logs survive process restarts.
log_sse.init_persistence()
_sse_handler = log_sse.LogSSEHandler(level=logging.INFO)
# Bare %(message)s — the SSE payload already carries `ts`, `level`, and `name`,
# and the dashboard formats them itself. This avoids double-stamped output.
_sse_handler.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger().addHandler(_sse_handler)

_redis = None  # backward compat — delegates to state.redis via property-like access
_shutting_down = False  # set during lifespan shutdown to prevent new bg operations


def _get_redis():
    """Get the Redis client — always returns state.redis (single source of truth)."""
    r = state.redis
    if r is not None:
        return r
    # Fallback to legacy _redis global during startup before lifespan sets state.redis
    return _redis

# ── Task manager and counters — delegate to api.state (single source of truth) ─
_task_manager = state.task_manager
_AUTO_CONSOLIDATE_EVERY = state.AUTO_CONSOLIDATE_EVERY

def _spawn(coro, name: str = "bg") -> None:
    """Fire-and-forget a coroutine with TaskManager supervision."""
    state.spawn(coro, name=name)

# ── Thread-safe counters — delegate to api.state ─────────────────────────────
_stores_since_consolidation = state.stores_since_consolidation
_periodic_prune_counter = state.periodic_prune_counter
_store_attempts = state.store_attempts
_store_successes = state.store_successes
_store_skips = state.store_skips
_store_errors = state.store_errors
_store_latency_sum_ms = state.store_latency_sum_ms

# ── BM25 index — delegate to api.state ───────────────────────────────────────
_bm25_index = state.bm25_index


async def _encode_batch_async(texts: list[str]) -> list[np.ndarray]:
    """Batch-encode texts via the dedicated, bounded embedding pool.

    Delegates to core.search.async_encode_batch, which runs the CPU-bound
    inference on the embed thread pool (separate from the default pool used by
    BM25 rebuilds) under a semaphore — the recall concurrency ceiling.  Same
    contract as before: propagates EmbeddingUnavailable, degrades partial
    failures to zero vectors so the pipeline never crashes.
    """
    if not texts:
        return []
    if embedder.embedding_unavailable():
        raise embedder.EmbeddingUnavailable("embedding circuit open")
    try:
        return await async_encode_batch(texts)
    except embedder.EmbeddingUnavailable:
        raise
    except Exception as e:
        # async_encode_batch already fell back per-text; re-raise circuit trips
        # but swallow other unexpected errors as zeros to keep callers safe.
        log.warning("[encode_batch] pool encode failed (%s: %s); zero-filling", type(e).__name__, e)
        return [np.zeros(embedder.DIMS, dtype=np.float32) for _ in texts]

# ── Session handoff: pinned last-session summary ───────────────────────────────
# Key stores the most recent session summary so the NEXT session can always
# retrieve it regardless of query similarity (cross-session continuity bridge).
_PINNED_SESSION_KEY       = "mem:pinned:session_summary"
_CRYSTALLIZED_INDEX_KEY   = "mem:crystallized_index"   # sorted set: key → crystallized_at score


# ── Lifespan ──────────────────────────────────────────────────────────────────
async def _periodic_consolidate() -> None:
    """Background task: consolidate memory every hour; hard-prune every 24 hours."""
    global _periodic_prune_counter
    while True:
        await asyncio.sleep(3600)
        r = _get_redis()
        if r is None:
            continue
        try:
            card = await r.execute_command("VCARD", mem_store.FACT_KEY)
            if int(card or 0) > 5:
                await asyncio.wait_for(
                    _do_consolidate(r, _bm25_index, _spawn),
                    timeout=180.0,
                )
        except asyncio.TimeoutError:
            log.warning("[periodic_consolidate] timed out (180s limit)")
        except Exception as e:
            log.warning(f"[periodic_consolidate] error: {e}")
        prune_count = await _periodic_prune_counter.increment()
        if prune_count >= 24:
            await _periodic_prune_counter.reset()
            try:
                await _do_hard_prune(r, _bm25_index, _spawn)
            except Exception as e:
                log.warning(f"[periodic_consolidate] hard-prune error: {e}")


async def _backfill_crystallized_index(r) -> None:
    """One-time startup backfill: add any existing mem:crystallized:* keys to the sorted set index."""
    try:
        index_size = await r.zcard(_CRYSTALLIZED_INDEX_KEY)
        if index_size > 0:
            return  # index already populated
        cursor = 0
        pipe = r.pipeline(transaction=False)
        while True:
            cursor, keys = await r.scan(cursor, match="mem:crystallized:*", count=100)
            for key in keys:
                pipe.get(key)
            if cursor == 0:
                break
        raw_values = await pipe.execute()
        zadd_map: dict = {}
        for raw in raw_values:
            if not raw:
                continue
            try:
                d = _decode_json(raw)
                session_id = d.get("session_id", "")
                ts = d.get("crystallized_at", 0)
                if session_id and ts:
                    zadd_map[f"mem:crystallized:{session_id}"] = ts
            except Exception:
                continue
        if zadd_map:
            await r.zadd(_CRYSTALLIZED_INDEX_KEY, zadd_map)
            log.info(f"[crystallize] backfilled index with {len(zadd_map)} entries")
    except Exception as e:
        log.warning(f"[crystallize] index backfill failed: {e}")


async def _startup_bm25_populate(r, bm25_index) -> None:
    """Recover procedures, then build the BM25 corpus from a trustworthy enumeration.

    mem:procedures' HNSW graph is degraded, so VSIM-based enumeration sees only
    ~13% of it.  The element IDs survive in mem:proc_by_tool:* / mem:proc_tasks,
    so recover them first — otherwise the BM25 corpus would silently exclude
    ~51% of the procedural tier.  The recovery is guarded and idempotent: two
    round trips when the shadow index is already complete.
    """
    try:
        rec = await mem_store.ensure_procedures_recovered(r)
        if rec:
            log.info(
                "[proc-recovery] %d live / %d collected, %d newly registered, "
                "%d stale refs pruned — shadow %d/%d (%.1f%%)",
                rec["live"], rec["ids_collected"], rec["newly_registered"],
                rec["stale_refs_pruned"], rec["shadow_after"], rec["vcard"],
                rec["coverage"] * 100,
            )
    except Exception as e:
        log.warning(f"[proc-recovery] skipped: {e}")
    await populate_bm25_from_redis(r, bm25_index)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _redis
    from api import state as _state
    from core import warmup as warmup_mod

    # ── BIND FIRST, WARM AFTER ────────────────────────────────────────────────
    # The heavy init used to run right here -- embedding model, Redis, KB BM25 warm,
    # ~10 min of cold start -- and FastAPI runs lifespan BEFORE uvicorn binds the
    # socket. So the port refused connections for the entire warmup, which the
    # supervisor can only read as "crashing" (watchdog.log 07:57:06 DOWN → 07:59:32
    # P0, while the model had not even started loading). See core/warmup.py for the
    # full receipts. We now yield immediately and warm in the background; /livez,
    # /health and /ready carry the truth in the meantime.
    _mcp_session_mgr = None
    try:
        from mcp_server import mcp as _mcp_srv
        _mcp_session_mgr = getattr(_mcp_srv, "session_manager", None)
    except Exception as _e:
        log.info("MCP session manager unavailable: %s", _e)

    async def _run_warmup() -> None:
        """Two stages.

        Stage A (gates /ready): everything core work needs — Redis, indexes, KB BM25.
        Stage B (does NOT gate): the embedding model + dim-guard. Measured on a
        healthy run, stage B costs ~25 s (SentenceTransformer import 20 s + weights
        2 s + MPS first encode 1 s); under memory pressure it stretched to 180 s+.
        Blocking /ready on it delayed a service that could already serve recall
        metadata, stats and graph queries. Stage B now preloads in the background so
        the first real embed is warm; if it lands before that, the embedder's own
        thread-safe lazy loader covers the gap.
        """
        try:
            def _elapsed() -> float:
                return time.monotonic() - _BOOT_T0

            log.info("[warmup] entered lifespan warmup task (t+%.1fs since import-start)",
                     _elapsed())
            warmup_mod.set_phase("redis", "connecting")
            log.info("Connecting to Redis…")
            _t = time.monotonic()
            _redis = await mem_store.get_client()
            _state.redis = _redis  # share with compat/health/graph routes
            log.info("[warmup] redis connect %.1fs (t+%.1fs)", time.monotonic() - _t, _elapsed())

            # KB plane: rebuild the in-RAM BM25 index for knowledge-base chunks.
            # The old form issued one HGET per chunk (N+1) — 250 s+ on 2,083 chunks;
            # one HGETALL brought this phase to ~10 s.
            try:
                from core import kb as kb_mod
                warmup_mod.set_phase("kb_bm25", "rebuilding BM25 index")
                _t = time.monotonic()
                _warmed = await kb_mod.get_kb().warm(_redis)
                log.info("[warmup] kb BM25 warm %.1fs -> %d chunks (t+%.1fs)",
                         time.monotonic() - _t, _warmed, _elapsed())
            except Exception as exc:
                log.warning("[kb] warm skipped: %s", exc)

            warmup_mod.set_phase("indexes", "ensuring vectorset indexes")
            log.info("Ensuring vectorset indexes…")
            _t = time.monotonic()
            await mem_store.ensure_indexes(_get_redis())
            log.info("[warmup] ensure_indexes %.1fs (t+%.1fs)", time.monotonic() - _t, _elapsed())
            # Persona hygiene: reject rules only guard *future* writes. Anything already
            # in mem:persona (e.g. the national-ID-card path that was being injected into
            # every agent's context on every recall) is re-filtered here once per boot,
            # so a stale profile can never outlive the rule that should have blocked it.
            try:
                warmup_mod.set_phase("persona", "sanitizing stored persona")
                _t = time.monotonic()
                _dropped = await persona_mod.sanitize_stored(_get_redis())
                log.info("[warmup] persona sanitize %.1fs (t+%.1fs)", time.monotonic() - _t,
                         _elapsed())
                if _dropped:
                    log.info("[persona] startup hygiene removed stale entries from %d field(s): %s",
                             len(_dropped), _dropped)
            except Exception as exc:
                log.warning("[persona] startup hygiene skipped: %s", exc)
            # Pre-load the heat-bump Lua so the first recall does not pay a NOSCRIPT
            # round trip (and so a flushed script cache is visible at startup, loudly).
            try:
                _t = time.monotonic()
                await _get_redis().execute_command(
                    "SCRIPT LOAD", heat_mod.BUMP_HEAT_LUA_SRC)
                log.info("[warmup] heat SCRIPT LOAD %.3fs (t+%.1fs)", time.monotonic() - _t,
                         _elapsed())
                log.info("[heat] bump script cached: sha=%s", heat_mod.bump_script_sha())
            except Exception as exc:
                log.warning("[heat] SCRIPT LOAD failed (%s) — bumps will use per-element "
                            "EVAL fallback", exc)
            _spawn(_periodic_consolidate(), "consolidate")
            _spawn(_auto_crystallize(), "crystallize")  # Auto-crystallize sessions every 6h
            # Populate BM25 corpus from existing Redis facts (non-blocking, best-effort)
            _spawn(_startup_bm25_populate(_get_redis(), state.bm25_index), "bm25-populate")
            # Backfill crystallized sorted-set index from existing keys (idempotent)
            _spawn(_backfill_crystallized_index(_get_redis()), "crystallize-index-backfill")

            warmup_mod.mark_ready()
            log.info("AgentMem v%s core ready (redis+indexes+kb); embedding warmup continues "
                     "in background (t+%.1fs)", APP_VERSION, _elapsed())

            # ── Stage B: embedding model + width guard (non-gating) ──────────────
            try:
                warmup_mod.set_phase("embedding", "loading qwen3-embedding-0.6b")
                log.info("Loading embedding model…")
                # MUST go off the loop: `_get_provider()` is a synchronous model load
                # (torch + SentenceTransformer). Called directly it blocks the event
                # loop for the whole load, and uvicorn only creates its listening
                # socket after lifespan startup — so the block would delay binding and
                # we would be right back at ECONNREFUSED during warmup.
                await asyncio.to_thread(embedder._get_provider)

                # Width integrity: the index is the source of truth, the model is
                # measured. This used to check three hard-coded keys and, on mismatch,
                # force-switch the provider — a config error became a silent model
                # downgrade. It needs the loaded model to measure, so it lives in
                # stage B next to the load it validates.
                warmup_mod.set_phase("dim_guard", "verifying embedding width")
                await dim_guard.enforce(_get_redis())

                warmup_mod.mark_embedding_ready()
                log.info("[warmup] embedding model warm + dim-guard verified")
            except Exception as exc:
                # Core stays ready; only the embedding plane is flagged. /ready keeps
                # reporting 200 with embedding_ready=false + the error, instead of
                # pretending a broken model is healthy or hiding the failure.
                warmup_mod.mark_embedding_failed(f"{type(exc).__name__}: {exc}")
                log.error("[warmup] embedding stage FAILED (core unaffected): %s", exc,
                          exc_info=True)
        except Exception as exc:
            # Stay un-ready: /ready keeps returning 503 with this error rather than
            # letting a half-initialised service look healthy and corrupt reads.
            warmup_mod.mark_failed(f"{type(exc).__name__}: {exc}")
            log.error("[warmup] FAILED — service stays un-ready: %s", exc, exc_info=True)

    warm_task = asyncio.create_task(_run_warmup(), name="warmup")

    if _mcp_session_mgr is not None:
        async with _mcp_session_mgr.run():
            log.info("MCP Streamable-HTTP session manager started")
            yield
    else:
        yield

    if not warm_task.done():
        warm_task.cancel()
        try:
            await warm_task
        except (asyncio.CancelledError, Exception):
            pass
    # Cancel tracked background tasks before closing Redis
    global _shutting_down
    _shutting_down = True
    await state.task_manager.shutdown(timeout=settings.bg_task_shutdown_timeout)
    from core.http import close_client
    await close_client()
    try:
        from mcp_server import close_client as _mcp_close_client
        await _mcp_close_client()
    except ImportError:
        pass  # mcp package not installed — nothing to close
    from core.embedder import close as _embedder_close
    _embedder_close()
    await mem_store.close_pool()


app = FastAPI(title="AgentMem — Local Agent Memory Service", version=APP_VERSION, lifespan=lifespan, redirect_slashes=False)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Log full traceback for unhandled exceptions instead of silently returning 500."""
    import traceback
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    log.error("[unhandled] %s %s → %s\n%s", request.method, request.url.path, exc, "".join(tb))
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=500, content={"detail": str(exc), "type": type(exc).__name__})


@app.middleware("http")
async def mcp_path_rewrite(request: Request, call_next):
    """Rewrite /mcp → /mcp/ before routing (MCP clients POST without trailing slash)."""
    if request.scope.get("path") == "/mcp":
        request.scope["path"] = "/mcp/"
        request.scope["raw_path"] = b"/mcp/"
    return await call_next(request)


@app.middleware("http")
async def redis_guard(request: Request, call_next):
    """Warmup gate: hold real work through cold start, or fail fast WITH a retry hint.

    Was: `if redis is None: return 503 "Redis not connected"`, immediately. During the
    ~10 min cold start every real request therefore got a flat 503 that reads like a
    broken service, and a caller had no way to tell "still booting" from "misconfigured".
    Now we wait up to AGENTMEM_WARMUP_WAIT_S for readiness and, if the budget runs out,
    answer 503 + Retry-After with the live warmup phase in the body. Liveness/readiness
    endpoints and static assets are never gated.
    """
    from fastapi.responses import JSONResponse
    from core import warmup as warmup_mod

    path = request.url.path
    if warmup_mod.is_exempt(path):
        return await call_next(request)

    if not warmup_mod.is_ready():
        # One bounded wait shared by every in-flight request during cold start.
        if await warmup_mod.wait_ready():
            return await call_next(request)
        snap = warmup_mod.snapshot()
        return JSONResponse(
            status_code=503,
            headers={"Retry-After": "5"},
            content={
                "detail": "agentmem is still warming up",
                "ready": False,
                "retry_after_s": 5,
                "warmup": snap,
            },
        )

    if _get_redis() is None:
        return JSONResponse(status_code=503, content={"detail": "Redis not connected"})
    return await call_next(request)


# Write-endpoint auth: LAN default is OPEN (AGENTMEM_AUTH unset/false). Set
# AGENTMEM_AUTH=1 (and AGENTMEM_API_KEY) only when exposing beyond a trusted LAN.
# When auth is on, require X-API-Key on every mutating request. Reads stay open
# (GET/HEAD/OPTIONS by method, plus a small allowlist of POST-as-read query routes).
#
# Design note (2026-10-24 re-audit): the previous rule matched request PATHS against a
# prefix list, which silently under-covered as routes were added — DELETE /facts/{id},
# DELETE /procedures/{id}, POST /facts/{id}/pin, POST /kb/remove, POST /graph/edge,
# POST /facts/{id}/reinforce and every /session/* write matched nothing and stayed
# unauthenticated on a 0.0.0.0-bound service. Default-deny by METHOD (when auth is on)
# inverts the failure mode: a new route is protected unless it is explicitly a read.
# Policy lives in pure functions (tests/unit/test_write_guard.py).
_WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_READ_POST_PREFIXES = (
    "/recall",           # /recall, /recall-tools, /recall-procedures
    "/search",           # kb /search, compat /search
    "/smart-search", "/smart_search",
    "/timeline",
    "/graph/recall", "/graph/traverse",
    "/context", "/file-context",
    "/patterns", "/summarize",
    # MCP streamable-HTTP / SSE transport. Cursor POSTs initialize here with no
    # API key; a 401 is misread as OAuth → namespace stuck on mcp_auth only.
    # Tool→REST write gating (when AGENTMEM_AUTH=1) still happens inside mcp_server.
    "/mcp",
)


def needs_write_auth(method: str, path: str) -> bool:
    """True when a request mutates state and must present the API key.

    Pure policy function — no app, no request object — so the matrix is testable.
    """
    if method.upper() not in _WRITE_METHODS:
        return False
    return not any(path.startswith(p) for p in _READ_POST_PREFIXES)


def check_write_auth(
    method: str,
    path: str,
    provided: str,
    configured: str,
    *,
    auth_enabled: bool = False,
) -> bool:
    """True = allow.

    LAN default: auth_enabled=False → open (no key needed).
    When AGENTMEM_AUTH=1, writes require a matching AGENTMEM_API_KEY.
    AUTH on with an empty configured key fails closed on writes.
    """
    if not auth_enabled:
        return True
    if not needs_write_auth(method, path):
        return True
    if not configured:
        return False
    return hmac.compare_digest(provided, configured)


@app.middleware("http")
async def api_key_guard(request: Request, call_next):
    """Require API key on write endpoints when AGENTMEM_AUTH=1."""
    if not check_write_auth(
        request.method,
        request.url.path,
        request.headers.get("X-API-Key", ""),
        settings.api_key or "",
        auth_enabled=bool(settings.auth),
    ):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})
    return await call_next(request)

# Request tracing: assign a unique ID to each request for log correlation
import uuid as _uuid

# Hot paths that poll / fire every tool call. Logging each 200 at INFO
# dominated stderr (~17k /tool-feedback lines in ~2h) and drowned real faults.
_QUIET_ACCESS_EXACT = frozenset({
    "/health", "/livez", "/ready", "/stats", "/ops", "/favicon.ico", "/favicon.svg",
    "/apple-touch-icon.png", "/tool-feedback", "/session/compact",
    "/kb/status", "/graph/stats", "/sessions", "/observations",
    "/active-memory", "/intentions",
})
_QUIET_ACCESS_PREFIX = ("/css/", "/js/", "/static/")


def _quiet_access(path: str, status: int, elapsed_ms: int) -> bool:
    """True → log at DEBUG. Always INFO for 5xx and slow responses."""
    if status >= 500 or elapsed_ms >= 500:
        return False
    return path in _QUIET_ACCESS_EXACT or path.startswith(_QUIET_ACCESS_PREFIX)


@app.middleware("http")
async def request_tracing(request: Request, call_next):
    """Assign a unique request ID and log request/response timing."""
    request_id = request.headers.get("X-Request-ID") or str(_uuid.uuid4())[:8]
    request.state.request_id = request_id

    import time as _time
    start = _time.monotonic()
    response = await call_next(request)
    elapsed_ms = int((_time.monotonic() - start) * 1000)

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{elapsed_ms}ms"

    path = request.url.path
    if _quiet_access(path, response.status_code, elapsed_ms):
        log.debug(
            "[access] %s %s → %d (%dms) rid=%s",
            request.method, path, response.status_code, elapsed_ms, request_id,
        )
    else:
        log.info(
            "[access] %s %s → %d (%dms) rid=%s",
            request.method, path, response.status_code, elapsed_ms, request_id,
        )
    return response


@app.exception_handler(embedder.EmbeddingUnavailable)
async def embedding_unavailable_handler(request: Request, exc: embedder.EmbeddingUnavailable):
    """Soft-fail: never let an open embed circuit become an ASGI ExceptionGroup."""
    return JSONResponse(
        status_code=503,
        content={
            "detail": "embedding unavailable",
            "error": str(exc) or "embedding circuit open",
            "retry_after_s": 30,
        },
        headers={"Retry-After": "30"},
    )

# ── Static files + SSE log router ─────────────────────────────────────────────
import os as _os
_static_dir = _os.path.join(_os.path.dirname(__file__), "static")
if _os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")
    # Mount css/js subdirs at root so <link href="/css/..."> and <script src="/js/..."> work
    _css_dir = _os.path.join(_static_dir, "css")
    _js_dir = _os.path.join(_static_dir, "js")
    if _os.path.isdir(_css_dir):
        app.mount("/css", StaticFiles(directory=_css_dir), name="css")
    if _os.path.isdir(_js_dir):
        app.mount("/js", StaticFiles(directory=_js_dir), name="js")

app.include_router(log_sse.log_sse_router)

# ── Extracted route modules ──────────────────────────────────────────────────
from api.routes.health import router as health_router
from api.routes.capability import router as capability_router
from api.routes.graph import router as graph_router
from api.routes.compat import router as compat_router
from api.routes.admin import router as admin_router
app.include_router(health_router)
app.include_router(capability_router)
app.include_router(graph_router)
app.include_router(compat_router)
app.include_router(admin_router)
from api.routes.kb import router as kb_router
app.include_router(kb_router)
from api.routes.focus import router as focus_router
app.include_router(focus_router)

# ── MCP transports: Streamable HTTP (preferred) + SSE (legacy) ───────────────
try:
    from mcp_server import build_mcp_app, build_sse_app
    _mcp_asgi = build_mcp_app()
    _sse_asgi = build_sse_app()

    app.mount("/mcp/sse", _sse_asgi, name="mcp_sse")

    from starlette.routing import Mount as _Mount, Match as _Match

    class _NoRedirectMount(_Mount):
        """Serve POST /mcp (no trailing slash) without a 307 redirect."""

        def matches(self, scope):
            if scope.get("type") in ("http", "websocket"):
                path = scope.get("path", "")
                if path == "/mcp/sse" or path.startswith("/mcp/sse/"):
                    return _Match.NONE, scope
                if path == "/mcp":
                    scope = dict(scope)
                    scope["path"] = "/mcp/"
                    scope["raw_path"] = b"/mcp/"
            return super().matches(scope)

    app.routes.insert(0, _NoRedirectMount("/mcp/", app=_mcp_asgi, name="mcp_http"))
except ImportError as _mcp_err:
    import logging as _logging
    _logging.getLogger(__name__).info("MCP server not available (install 'mcp' package): %s", _mcp_err)
except Exception as _mcp_err:
    import logging as _logging
    _logging.getLogger(__name__).warning("MCP HTTP mount failed: %s", _mcp_err)

# ── WebSocket for real-time dashboard updates ─────────────────────────────────
_ws_clients: set[WebSocket] = set()


async def _ws_broadcast(event: str, data: dict) -> None:
    """Push an event to all connected WebSocket clients in parallel."""
    if not _ws_clients:
        return
    msg = json.dumps({"event": event, **data, "ts": time.time()})

    async def _send(ws: WebSocket) -> WebSocket | None:
        try:
            await ws.send_text(msg)
            return None
        except Exception:
            return ws

    results = await asyncio.gather(*[_send(ws) for ws in _ws_clients])
    dead = {ws for ws in results if ws is not None}
    _ws_clients.difference_update(dead)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    _ws_clients.add(ws)
    try:
        # Send initial state
        await ws.send_text(json.dumps({
            "event": "connected",
            "version": APP_VERSION,
            "ts": time.time(),
        }))
        while True:
            # Keep alive — client pings or we just wait for disconnect
            data = await ws.receive_text()
            # Client can request specific data
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await ws.send_text(json.dumps({"event": "pong", "ts": time.time()}))
                elif msg.get("type") == "subscribe":
                    pass  # all events broadcast by default
            except (json.JSONDecodeError, KeyError):
                pass
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.discard(ws)


@app.get("/", include_in_schema=False)
async def dashboard():
    """Serve the memory service dashboard."""
    idx = _os.path.join(_static_dir, "index.html")
    if _os.path.exists(idx):
        return FileResponse(idx)
    return {"message": f"AgentMem v{APP_VERSION}", "docs": "/docs"}


def _static_asset(name: str, media_type: str | None = None):
    path = _os.path.join(_static_dir, name)
    if not _os.path.exists(path):
        from fastapi import HTTPException
        raise HTTPException(404, f"missing {name}")
    kw = {"media_type": media_type} if media_type else {}
    return FileResponse(path, **kw)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon_ico():
    return _static_asset("favicon.ico", "image/x-icon")


@app.get("/favicon.svg", include_in_schema=False)
async def favicon_svg():
    return _static_asset("favicon.svg", "image/svg+xml")


@app.get("/apple-touch-icon.png", include_in_schema=False)
async def apple_touch_icon():
    return _static_asset("apple-touch-icon.png", "image/png")


# ── Request / Response models ──────────────────────────────────────────────────
# All Pydantic request models are now in api/schemas/ — imported above.

# ── Helpers ────────────────────────────────────────────────────────────────────

def _msg_text(m: Message) -> str:
    raw = _flatten_message_content(m.content)
    return _strip_platform_noise(raw)

def _messages_to_text(messages: list[Message]) -> str:
    return "\n".join(
        f"{m.role}: {_msg_text(m)}" for m in messages if _msg_text(m)
    )


def _format_prepend(
    facts: list[dict],
    episodes: list[dict],
    session_ctx: str | None,
    persona_ctx: str,
    env_ctx: str = "",
    tool_ctx: str = "",
    proc_ctx: list[dict] | None = None,
    token_budget: int | None = 1500,
    last_session_summary: str | None = None,
    crystal_digests: list[dict] | None = None,
    active_focus: dict | None = None,
    intentions: list[dict] | None = None,
) -> str:
    """
    Build the context string injected into the agent's system prompt.

    SimpleMem Section 3.3 token-budget packing:
    Priority order: active focus → intentions → persona → last_session_summary →
    crystallized digests → env → tools → skills → session_ctx → facts → episodes
    Greedy packing: stop adding items when token_budget is exhausted.
    Output wrapped in <cross_session_memory> XML tags (SimpleMem convention).
    """
    if token_budget is None:
        token_budget = settings.default_token_budget
    # Reserve wrapper tokens so packing can't "succeed" then overflow-clamp.
    _WRAPPER_TOKENS = 30
    pack_budget = max(32, int(token_budget) - _WRAPPER_TOKENS)
    # Always-on persona must stay short (Mem0/Letta practice: stable prefix ≪ retrieval).
    persona_token_cap = max(64, min(320, pack_budget // 4))
    sections: list[str] = []
    tokens_used = 0

    def _add(text: str) -> bool:
        nonlocal tokens_used
        text = _redact_secrets(text)
        cost = _estimate_tokens(text)
        if tokens_used + cost > pack_budget:
            return False
        sections.append(text)
        tokens_used += cost
        return True

    if active_focus and (active_focus.get("goal") or active_focus.get("notes")):
        focus_lines = ["## Active Focus"]
        if active_focus.get("goal"):
            focus_lines.append(f"- goal: {active_focus['goal']}")
        if active_focus.get("notes"):
            focus_lines.append(f"- notes: {active_focus['notes']}")
        _add("\n".join(focus_lines))

    if intentions:
        intent_lines = ["## Pending Intentions"]
        for it in intentions[:5]:
            cue = f" (cue: {it['cue']})" if it.get("cue") else ""
            intent_lines.append(f"- {it.get('text', '')}{cue}")
        if len(intent_lines) > 1:
            _add("\n".join(intent_lines))

    def _add_fitted(text: str) -> bool:
        """Add a block, truncated to whatever budget is left rather than dropped.

        Priority-1 blocks (the persona) must survive a small budget: `_add`
        refuses them wholesale, and history showed the opposite failure too —
        `estimate_tokens` scored the Chinese persona as ~1 token, so a 600-token
        budget shipped ~1 600 real tokens. Truncating to the remaining budget is
        the only behaviour that is both lossless-within-budget and honest about
        the budget.
        """
        nonlocal tokens_used
        text = _redact_secrets(text)
        remaining = pack_budget - tokens_used
        if remaining <= 0:
            return False
        if _estimate_tokens(text) > remaining:
            text = truncate_to_tokens(text, remaining)
            if not text:
                return False
        cost = _estimate_tokens(text)
        sections.append(text)
        tokens_used += cost
        return True

    # Priority 1: persona (user profile — most stable context), hard-capped
    if persona_ctx:
        persona_block = persona_ctx
        if _estimate_tokens(persona_block) > persona_token_cap:
            persona_block = truncate_to_tokens(persona_block, persona_token_cap)
        if persona_block and _add_fitted(persona_block):
            _add("")

    # Priority 1.5: last session summary (cross-session continuity bridge)
    # Always injected when present so the agent knows what was accomplished
    # in the previous session, regardless of current query similarity.
    if last_session_summary and _add("## Last Session Summary"):
        _add(last_session_summary[:600])
        _add("")

    # Priority 1.7: crystallized digests (lessons learned from past sessions)
    # Auto-generated summaries of completed work chains with key findings
    if crystal_digests and _add("## Lessons Learned"):
        for digest in crystal_digests[:3]:  # top 3 digests
            summary = digest.get("summary", "")
            fact_count = digest.get("fact_count", 0)
            entities = digest.get("entities", [])
            
            # Format: summary + key stats
            digest_text = f"- **Completed Session** ({fact_count} facts): {summary}"
            if not _add(digest_text):
                break
            
            # Add key entities if space allows
            if entities and len(entities) <= 5:
                entity_str = ", ".join(entities[:5])
                if not _add(f"  Key entities: {entity_str}"):
                    break
        
        _add("")

    # Priority 2: environment context
    if env_ctx and _add(env_ctx):
        _add("")

    # Priority 3: tool context
    if tool_ctx and _add(tool_ctx):
        _add("")

    # Priority 3.5: procedural skills (MetaClaw-style behavioral guidelines)
    if proc_ctx and _add("## Relevant Skills"):
        for p in proc_ctx:
            task      = p.get("task", "")
            procedure = p.get("procedure", "")
            # task may be "[skill:name] description" — strip the prefix tag for display
            display = task.split("] ", 1)[-1] if task.startswith("[skill") else task
            # Include first 200 chars of procedure body as a hint
            body_hint = procedure[:200].replace("\n", " ").strip()
            line = f"- **{display}**" + (f": {body_hint}" if body_hint else "")
            if not _add(line):
                break
        _add("")

    # Priority 4: session context (Tier 1 rolling summary)
    if session_ctx and _add("## Current Session Context"):
        _add(session_ctx)
        _add("")

    # Priority 5: facts (SimpleMem lossless_restatement + Memori triple entries)
    # ALL appends go through _add() so headers count toward budget too.
    if facts and _add("## Long-Term Memory (Facts)"):
        for i, f in enumerate(facts, 1):
            tag = f.get("category") or f.get("domain") or "?"
            attrs = f.get("attrs", {})
            meta_parts = []
            if p := attrs.get("persons"):
                meta_parts.append(f"who:{','.join(p)}")
            if e := attrs.get("entities"):
                meta_parts.append(f"re:{','.join(e)}")
            if t := attrs.get("topic"):
                meta_parts.append(f"topic:{t}")
            # Show compact triple form when available (more precise, fewer tokens)
            triple_str = attrs.get("triple_str", "")
            if triple_str:
                meta_parts.append(f"triple:{triple_str}")
            meta = f" ({'; '.join(meta_parts)})" if meta_parts else ""
            line = f"{i}. [{tag}]{meta} {f['content']}"
            if not _add(line):
                break   # budget exhausted
        _add("")

    # Priority 6: episodes (lower priority than facts)
    if episodes and _add("## Recent Relevant Episodes"):
        for i, e in enumerate(episodes, 1):
            attrs = e.get("attrs", {})
            ep_type = attrs.get("ep_type", "")
            type_tag = f"[{ep_type}] " if ep_type and ep_type != "general" else ""
            line = f"{i}. {type_tag}{e['content'][:300]}"
            if not _add(line):
                break
        _add("")

    body = "\n".join(sections).strip()
    if not body:
        return ""

    # SimpleMem XML wrapper — clearly delineates injected memory from user content
    prefix = ("<cross_session_memory>\n"
              "The following is relevant context from long-term memory.\n"
              "Use it to inform your responses but do not repeat it verbatim.\n\n")
    suffix = "\n</cross_session_memory>"
    out = prefix + body + suffix

    # Final budget assertion. A single oversized line can still slip past the
    # per-item check; clamp and log (debug when only a few tokens over — that
    # used to spam WARNING every recall when persona was 2k chars).
    cost = _estimate_tokens(out)
    if cost > token_budget:
        over = cost - token_budget
        msg = ("[recall] token_budget overflow: packed %d tokens > budget %d"
               " — clamping (%d facts, %d episodes, persona=%d chars)")
        args = (cost, token_budget, len(facts), len(episodes), len(persona_ctx or ""))
        if over <= 40:
            log.debug(msg, *args)
        else:
            log.warning(msg, *args)
        body = truncate_to_tokens(body, max(16, token_budget - _WRAPPER_TOKENS))
        out = prefix + body + suffix

    return out


# ── Query complexity estimator (SimpleMem-inspired, zero-cost) ────────────────
_COMPLEX_SIGNALS = re.compile(
    r'\b(what|when|where|who|how|why|which|compare|difference|between|all|every|list'
    r'|什么|怎么|哪|为什么|比较|区别|所有|每个|列出)\b', re.I
)

def _estimate_depth(query: str, base_limit: int) -> tuple[int, int]:
    """
    Adaptive retrieval depth: k_dyn = k_base × (1 + δ × C_q).
    Returns (n_facts, n_episodes). C_q estimated from query features.
    """
    words = query.split()
    n_words = len(words)
    n_signals = len(_COMPLEX_SIGNALS.findall(query))
    # Simple complexity score: 0.0 (trivial) to 1.0 (complex)
    c_q = min(1.0, (n_words / 30) + (n_signals * 0.2))
    delta = 0.8   # scaling factor

    k_base_facts = max(1, base_limit // 2)
    k_base_eps   = base_limit

    n_facts    = max(1, int(k_base_facts * (1.0 + delta * c_q)))
    n_episodes = max(2, int(k_base_eps   * (1.0 + delta * c_q * 0.5)))

    return n_facts, n_episodes


# ── A-MAC 5-Factor Admission Gate (arXiv:2603.04549) ─────────────────────────
#
# A-MAC (Adaptive Memory Admission Control, March 2026) achieves F1=0.583 on
# a LoCoMo *admission classification* task (binary: should this memory be stored?
# precision=0.417, recall=0.972). NOTE: this 0.583 is NOT the same metric as
# SimpleMem's 43.24 QA token-F1 — they measure different things on different
# task definitions and cannot be compared on the same leaderboard.
#
# Ablation reveals factor importance order:
#   type_prior > novelty > utility > confidence > recency
# We implement all 5 with zero extra LLM calls (no LLM utility scoring).
# Weight order follows A-MAC ablation (dominant factor gets highest weight):
#   F1 — semantic_novelty    (w=0.25): novelty vs. recently-stored episodes
#   F2 — entity_novelty      (w=0.15): new named entities / dates / file refs
#   F3 — factual_confidence  (w=0.20): declarative predicate density (confidence)
#   F4 — temporal_signal     (w=0.10): explicit dates/times (recency proxy)
#   F5 — content_type_prior  (w=0.30): high-value category patterns — DOMINANT
#                                       (A-MAC ablation: most influential factor)
#
# Threshold: 0.40 (between SimpleMem 0.35 and A-MAC's learned θ*=0.55;
# balances recall coverage vs. precision for a local non-LLM gate)

_ENTITY_RE = re.compile(
    r'\b[A-Z][a-z]{1,}\b'              # Capitalized words (names, places)
    r'|\b\d{4}[-/]\d{2}[-/]\d{2}\b'   # ISO dates
    r'|\b[A-Z]{2,}\b'                  # Acronyms (API, MCP, ...)
    r'|\b\w+\.(py|js|ts|sh|json|md)\b' # File references
    r'|[\u4e00-\u9fff\u3400-\u4dbf]{2,}' # CJK entity names (≥2 chars)
)
_DATE_RE  = re.compile(
    r'\b\d{4}[-/]\d{2}[-/]\d{2}\b'    # ISO date
    r'|\b(?:today|tomorrow|yesterday|monday|tuesday|wednesday|thursday|friday'
    r'|january|february|march|april|may|june|july|august|september|october'
    r'|november|december)\b'
    r'|\d{4}年\d{1,2}月(?:\d{1,2}日)?'  # Chinese date: 2026年3月7日
    r'|(?:昨天|今天|明天|上周|下周|上个月|下个月)', re.I
)
_FACT_RE  = re.compile(                # declarative fact patterns (EN + ZH)
    r'\b(?:is|are|was|were|has|have|had|will|prefer|prefers|preferred|use|uses'
    r'|used|using|like|likes|liked|work|works|worked|working|live|lives|lived'
    r'|need|needs|needed|always|never|must|should|remember|remembers|note|notes'
    r'|default|workspace|directory|repo|repository|branch|shell|tool|stack|my|our)\b'
    r'|(?:是|有|在|住|工作|喜欢|需要|记住|总是|从不|必须|应该|我的)', re.I
)
_HIGH_VALUE_RE = re.compile(           # content type prior: high-value (EN + ZH)
    r'\b(?:my name|i am|i\'m|i work|i live|i prefer|i like|i use|i always|i never'
    r'|always use|never use|uses|prefers|default shell|package manager|workspace'
    r'|directory|repo|repository|branch|project|tool|stack|deadline|remember'
    r'|important|rule:|note:|prefer|password|token|key|api|model)\b'
    r'|(?:我叫|我是|我在|我住|我喜欢|总是用|截止|记住|重要|密码|令牌)', re.I
)

async def _admission_gate(user_text: str) -> bool:
    """
    A-MAC 5-factor admission control (arXiv:2603.04549).

    Returns True  → worth storing (any of the 5 factors fires strongly).
    Returns False → low-value turn, discard to save embedding + write cost.
    """
    text = user_text[:400]

    # F1 — Semantic novelty (w=0.25)
    # Offload CPU-bound embedding to thread pool so it doesn't block the event loop
    emb = await asyncio.to_thread(_encode, text)
    recent = await mem_store.knn_search(_get_redis(), mem_store.EPISODE_KEY, emb, k=3)
    if recent:
        max_sim = max(r.get("score", 0.0) for r in recent)
        semantic_novelty = 1.0 - max_sim
    else:
        semantic_novelty = 1.0  # first store → always novel

    # F2 — Entity novelty (w=0.15)
    entities = set(_ENTITY_RE.findall(text))
    entity_novelty = min(1.0, len(entities) / 4)

    # F3 — Factual confidence: declarative predicate density (w=0.20)
    # For Chinese text .split() returns 1 token for the whole string; use
    # character count ÷ avg_word_len as a proxy for mixed-language word count.
    en_words  = len(text.split())
    zh_chars  = len(re.findall(r'[\u4e00-\u9fff]', text))
    words = max(en_words, 1) + zh_chars // 2  # ~2 chars per Chinese word
    fact_hits = len(_FACT_RE.findall(text))
    factual_confidence = min(1.0, fact_hits / max(words * 0.15, 1))

    # F4 — Temporal signal: explicit date/time references (w=0.10)
    temporal_signal = min(1.0, len(_DATE_RE.findall(text)) / 2)

    # F5 — Content type prior: high-value category patterns (w=0.30, DOMINANT)
    # A-MAC ablation: type_prior is the single most influential factor.
    # Scalar 1.0/0.0 is coarse; use 0.5 for partial matches (soft categories).
    high_value_hits = len(_HIGH_VALUE_RE.findall(text))
    content_type_prior = min(1.0, high_value_hits * 0.5) if high_value_hits else 0.0

    score = (
        0.25 * semantic_novelty
        + 0.15 * entity_novelty
        + 0.20 * factual_confidence
        + 0.10 * temporal_signal
        + 0.30 * content_type_prior
    )

    _gate_threshold = settings.amac_threshold
    if score < _gate_threshold:
        log.info(
            f"[store] admission gate filtered (score={score:.2f} "
            f"nov={semantic_novelty:.2f} ent={entity_novelty:.2f} "
            f"fact={factual_confidence:.2f} tmp={temporal_signal:.2f} "
            f"typ={content_type_prior:.2f}): {text[:50]!r}"
        )
        return False
    return True


# ── SimpleMem: Keyword Lexical Boost (Phase 3 hybrid scoring) ─────────────────

def _tokenize_query(query: str) -> set[str]:
    """Extract meaningful tokens from query for BM25-lite keyword matching.

    Handles both English (space-separated, ≥3 chars) and Chinese (CJK character
    sequences ≥2 chars, since Chinese has no word boundaries).
    """
    STOP = {"the", "and", "for", "are", "you", "can", "how", "what",
            "when", "where", "who", "why", "this", "that", "was", "did",
            "的", "了", "是", "在", "有", "我", "你", "他", "她", "它",
            "这", "那", "吗", "呢", "啊", "哦", "嗯"}
    # English tokens (≥3 chars)
    en_tokens = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()))
    # Chinese tokens: sequences of CJK chars (≥2 chars)
    zh_tokens = set(re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]{2,}', query))
    return (en_tokens | zh_tokens) - STOP


def _keyword_boost(items: list[dict], query: str, boost: float = 0.06) -> list[dict]:
    """
    SimpleMem Phase 3: lexical layer boost (BM25-lite).
    For each retrieved item, add `boost` × (matching_kw_count) to score.
    Enables exact-name matches (Bob, 2025-11-20, Redis) to surface higher.
    """
    if not items:
        return items
    query_tokens = _tokenize_query(query)
    if not query_tokens:
        return items

    boosted = []
    for item in items:
        attrs    = item.get("attrs", {})
        keywords = [kw.lower() for kw in (attrs.get("keywords") or [])]
        content  = item.get("content", "").lower()

        kw_overlap    = len(query_tokens & set(keywords))
        ct_overlap    = sum(1 for t in query_tokens if t in content)
        overlap_score = kw_overlap + ct_overlap * 0.3

        new_item = dict(item)
        new_item["score"] = item.get("score", 0.0) + boost * overlap_score
        boosted.append(new_item)

    boosted.sort(key=lambda x: x["score"], reverse=True)
    return boosted


def _importance_boost(items: list[dict], weight: float = 0.15) -> list[dict]:
    """
    AgentMem advantage over SimpleMem: importance-weighted recall reranking.

    SimpleMem uses `importance` only during consolidation winner selection.
    Here we use it during retrieval so high-signal facts (rules, identity,
    preferences — importance 0.7-1.0) outrank transient context (0.3-0.5).

    weight=0.15 adds up to 0.15 to the base cosine score (0.0-1.0 range),
    enough to surface critical memories without overwhelming cosine signal.
    """
    if not items:
        return items
    boosted = []
    for item in items:
        importance = item.get("attrs", {}).get("importance", 0.5)
        new_item = dict(item)
        new_item["score"] = item.get("score", 0.0) + weight * float(importance)
        boosted.append(new_item)
    boosted.sort(key=lambda x: x["score"], reverse=True)
    return boosted


def _rrf_merge(
    lists: list[list[dict]],
    weights: list[float] | None = None,
    k: int = 60,
    limit: int = 20,
) -> list[dict]:
    """
    Reciprocal Rank Fusion (Cormack et al., SIGIR 2009):
      RRF(d) = Σ_i  w_i / (k + rank_i(d))   k=60 smoothing constant

    Accepts per-list weights, but production callers pass equal weights
    ([1.0, 1.0, 1.0, 1.0]).  The hand-tuned per-query-type wRRF presets that
    used to be documented here lost nDCG@10 in 6 of 7 scenarios of a held-out
    sweep (worst -12.6%) and were removed — see docs/EVEROS_ASSESSMENT.md §4.1.
    """
    if weights is None:
        weights = [1.0] * len(lists)
    if len(weights) < len(lists):
        weights = weights + [1.0] * (len(lists) - len(weights))

    rrf_scores: dict[str, float] = {}
    items_by_content: dict[str, dict] = {}

    for w, lst in zip(weights, lists):
        for rank, item in enumerate(lst, 1):
            content = item.get("content", "")
            if not content:
                continue
            rrf_scores[content] = rrf_scores.get(content, 0.0) + w / (k + rank)
            if content not in items_by_content:
                items_by_content[content] = item

    sorted_pairs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    result = []
    for content, rrf_score in sorted_pairs[:limit]:
        item = dict(items_by_content[content])
        item["score"] = rrf_score
        result.append(item)
    return result


# ── Session diversity: prevent one session dominating recall results ──────────

def _session_diversify(items: list[dict], max_per_session: int = 3) -> list[dict]:
    """Limit results from the same session to avoid recall dominated by one session.

    Preserves rank order within each session group. Items without a session_id
    (facts with no origin session) are always kept.
    """
    seen: dict[str, int] = {}
    result = []
    for item in items:
        sid = item.get("attrs", {}).get("session_id", "")
        if not sid:
            result.append(item)
            continue
        count = seen.get(sid, 0)
        if count < max_per_session:
            result.append(item)
            seen[sid] = count + 1
    return result


# ── Episode type taxonomy (claude-mem inspired, v0.9.3) ───────────────────────

def _infer_ep_type(facts: list) -> str:
    """
    Map extracted fact categories → claude-mem episode type taxonomy.

    Priority order ensures the most specific/actionable type wins:
      decision > procedure > feature > change > discovery > preference > context > general

    This mirrors claude-mem's observation type system (bugfix|feature|refactor|
    discovery|decision|change) but uses AgentMem's existing category vocabulary.
    """
    if not facts:
        return "general"
    cat_set = {f.category for f in facts}
    if "decision" in cat_set:
        return "decision"
    if "procedure" in cat_set:
        return "procedure"
    if "capability_gained" in cat_set:
        return "feature"
    if "env_change" in cat_set:
        return "change"
    if cat_set & {"identity", "work", "personal", "location"}:
        return "discovery"
    if cat_set & {"preference", "rule", "reminder"}:
        return "preference"
    if "tool_use" in cat_set:
        return "context"
    return "general"


# ── A-MEM memory evolution: refine existing similar facts ─────────────────────
async def _evolve_similar_fact(element_id: str, new_keywords: list[str], new_topic: str | None) -> None:
    """
    A-MEM-inspired memory evolution (arXiv:2502.12110 §3).

    When a new fact is nearly-duplicate (cosine 0.80-0.95) of an existing fact,
    instead of discarding the new fact silently, we use its richer metadata
    (keywords, topic) to enrich the existing fact's attributes.

    This ensures that as the agent learns more specific context about a topic,
    previously stored facts gain better retrieval metadata — mimicking how
    human memory evolves and refines rather than just appending.
    """
    if not element_id or not new_keywords:
        return
    try:
        raw = await _get_redis().execute_command("VGETATTR", mem_store.FACT_KEY, element_id)
        if not raw:
            return
        attrs = _decode_attrs(raw)
        existing_kws = set(attrs.get("keywords") or [])
        # Merge new keywords (deduplicated, capped at 12)
        merged_kws = list(existing_kws | set(new_keywords))[:12]
        attrs["keywords"] = merged_kws
        # Update topic if new one is more specific (longer = more specific)
        if new_topic and len(new_topic) > len(attrs.get("topic") or ""):
            attrs["topic"] = new_topic[:100]
        await _get_redis().execute_command("VSETATTR", mem_store.FACT_KEY, element_id, json.dumps(attrs))
        log.debug(f"[evolution] enriched fact {element_id}: +{len(new_keywords)} kws")
    except Exception as e:
        log.debug(f"[evolution] skipped: {e}")


async def _check_contradiction(existing_fact: dict, new_fact) -> None:
    """
    LLM Wiki v2: Contradiction detection on store.

    When a new fact is semantically similar (0.80-0.95) to an existing one,
    check if they contradict. If so, add a 'contradicts' graph edge and
    mark the older fact as superseded with reason='contradicted'.

    Uses simple heuristic: if the same subject has opposing values
    (e.g., "X is A" vs "X is B" where A ≠ B), it's a contradiction.
    """
    element_id = existing_fact.get("_element", "")
    if not element_id:
        return

    old_content = existing_fact.get("content", "").lower()
    new_content = (new_fact.content if hasattr(new_fact, "content") else str(new_fact)).lower()

    # Extract entities from both facts for graph edge
    old_ents = existing_fact.get("attrs", {}).get("entities") or []
    new_ents = (new_fact.entities if hasattr(new_fact, "entities") else None) or []

    # Simple contradiction heuristic: same entities, different content
    # (a proper implementation would use LLM judge, but this covers common cases)
    shared_ents = set(e.lower() for e in old_ents) & set(e.lower() for e in new_ents)
    if not shared_ents:
        return

    # Check for opposing patterns: "is X" vs "is not X", "uses A" vs "uses B"
    _OPPOSITES = [
        ("is not", "is"), ("does not", "does"), ("cannot", "can"),
        ("will not", "will"), ("has no", "has"), ("no longer", ""),
    ]
    is_contradiction = False
    for neg, pos in _OPPOSITES:
        if neg in old_content and pos in new_content:
            is_contradiction = True
            break
        if neg in new_content and pos in old_content:
            is_contradiction = True
            break

    if not is_contradiction:
        return

    # Add 'contradicts' graph edges between old and new fact entities
    # (not self-edges — connect entities from the old fact to entities
    # from the new fact that are different but share the same subject)
    for old_ent in old_ents:
        for new_ent in new_ents:
            if old_ent.lower() != new_ent.lower():
                await graph_mod.add_typed_edge(
                    _get_redis(), old_ent, new_ent, "contradicts",
                    confidence=0.9, source="contradiction_detector", bidirectional=True,
                )

    # Mark old fact as superseded
    await mem_store.soft_delete_fact(_get_redis(), element_id, "contradicted", reason="contradicted")
    log.info(f"[contradiction] fact {element_id} superseded: contradicted by new fact")


# ── Background store ──────────────────────────────────────────────────────────
async def _do_store(messages: list[Message], session_id: str, scope: Scope = Scope()) -> None:
    global _store_attempts, _store_successes, _store_skips, _store_errors, _store_latency_sum_ms
    if _shutting_down:
        return
    await _store_attempts.increment()
    t0 = time.time()
    outcome = "error"
    try:
        if session_id and not scope.is_global:
            try:
                await mem_store.set_session_owner(_get_redis(), session_id, scope)
            except Exception as exc:
                log.warning("[store] session owner write failed for %s: %s", session_id, exc)
        # 1. Filter system-injected and empty messages.
        #    _msg_text() strips <cross_session_memory> prefixes and [[platform_tags]]
        #    before returning text, so purely-noise messages collapse to "" here.
        clean = [m for m in messages if _msg_text(m) and not _is_injected(_msg_text(m))]
        if not clean:
            outcome = "skip"
            return
        clean = clean[-4:]   # last 4 messages = last 2 turns (user+assistant pairs)

        # 1b. Check if the raw (pre-strip) user messages were ONLY cross_session_memory
        #     wrappers — e.g. heartbeat sessions that contain nothing real.
        raw_user_msgs = [
            m for m in messages
            if m.role == "user" and (
                isinstance(m.content, str) and m.content.strip()
            )
        ]
        if raw_user_msgs and all(
            _is_only_platform_noise(m.content if isinstance(m.content, str) else "")
            for m in raw_user_msgs
        ):
            log.info("[store] skipped: all user messages are cross_session_memory wrappers")
            outcome = "skip"
            return

        # 2. Detect scene from user messages
        user_text_raw = " ".join(_msg_text(m) for m in clean if m.role == "user")
        if not user_text_raw:
            outcome = "skip"
            return

        # 2b. Skip trivial exchanges (greetings, single words) — prevents feedback loops
        if _is_trivial(user_text_raw):
            log.info(f"[store] skipped trivial exchange: {user_text_raw[:40]!r}")
            outcome = "skip"
            return

        # 2c. A-MAC 5-factor admission gate (arXiv:2603.04549).
        #     Discards low-value inputs before heavy processing (summarize/embed/LLM).
        #     type_prior is dominant (w=0.30); threshold 0.40.
        if not await _admission_gate(user_text_raw):
            outcome = "skip"
            return  # low-entropy input, skip (not worth storing)

        sc = scene_mod.detect(user_text_raw)
        lang, domain = sc["language"], sc["domain"]

        # 3. Summarize long turn before embedding
        turn_text = _messages_to_text(clean)
        summary = await summarizer.summarize(turn_text)

        # 4. Embed + dedup-check + save episode (with type taxonomy + causal chain)
        ep_emb = await asyncio.to_thread(_encode, summary[:500])
        # Dedup is per-track: a turn identical to another track's episode is a
        # new episode here, not a duplicate to drop.
        similar = await mem_store.knn_search(
            _get_redis(), mem_store.EPISODE_KEY, ep_emb, k=1,
            filter_expr=scope.filter() or None,
        )
        is_dup = bool(similar and similar[0].get("score", 0.0) > 0.95)
        ep_saved = 0
        new_ep_uid = ""
        prev_ep_id = ""
        if not is_dup:
            # Causal chain: look up the previous episode for this session
            prev_ep_id = await mem_store.get_last_episode_id(_get_redis(), session_id)
            new_ep_uid = await mem_store.save_episode(
                _get_redis(), session_id, turn_text[:2000], ep_emb, lang, domain,
                ep_type="general",        # placeholder; updated after fact extraction below
                prev_episode_id=prev_ep_id,
                user_id=scope.user_id, agent_id=scope.agent_id,
            )
            # Episodes used to be keyword-invisible: BM25 only ever indexed facts.
            await _bm25_index.add(new_ep_uid, turn_text[:2000], {"category": "episode"}, "episodes")
            ep_saved = 1

        # 5. Hybrid fact extraction (regex + LLM) → dedup → save → update persona
        # Extract from all roles: assistant messages contain rich facts too
        # (e.g. "I work at NovaPay", "I live in Oakland" are in assistant turns)
        raw_msgs = [{"role": m.role, "content": _msg_text(m)} for m in clean if _msg_text(m)]
        facts = await extractor.extract_hybrid(raw_msgs, turn_text)
        fact_saved = 0

        # Separate secret-bearing facts (never persisted) from the rest so the
        # embeddings and dedup searches stay positionally aligned.
        normal_facts = [f for f in facts if not _contains_secret(f.content)]
        # Batch-encode all remaining fact contents off the event loop in one pass
        # instead of one MiniLM inference per fact (N sequential forward passes).
        normal_embs = await _encode_batch_async([f.content for f in normal_facts])
        # One VSIM pipeline for every fact's dedup check instead of N serial
        # round trips (mem:facts is small, but this is on the store hot path).
        dedup_hits = await mem_store.knn_search_batch(
            _get_redis(), mem_store.FACT_KEY, normal_embs, k=1,
            filter_expr=scope.filter() or None,
        )

        # Decide save/dedup/evolve/contradiction in memory — no Redis calls here.
        to_save: list[tuple] = []  # (fact, emb)
        for fact, emb, hits in zip(normal_facts, normal_embs, dedup_hits):
            top = hits[0] if hits else None
            score = top.get("score", 0.0) if top else 0.0
            if score > 0.95:
                # A-MEM memory evolution: similar fact exists → async-update its
                # keywords/topic (arXiv:2502.12110 §3: new memories trigger
                # context refinement on neighbours).
                if fact.keywords and score > 0.80:
                    _spawn(_evolve_similar_fact(
                        top.get("_element", ""), fact.keywords, fact.topic,
                    ), "evolve")
                continue
            # LLM Wiki v2: contradiction detection — a high-similarity fact
            # (0.80-0.95) with opposing meaning marks the old one superseded.
            if score > 0.80:
                _spawn(_check_contradiction(top, fact), "contradiction")
            to_save.append((fact, emb))

        # One VADD+SADD pipeline for every fact instead of N x 2 serial writes.
        save_specs = [{
            "embedding": emb,
            "content": fact.content, "category": fact.category,
            "confidence": fact.confidence, "language": lang, "domain": domain,
            "keywords": fact.keywords, "persons": fact.persons, "entities": fact.entities,
            "importance": fact.importance, "topic": fact.topic, "location": fact.location,
            "source_episode_id": new_ep_uid or None,
            "triple_s": fact.triple_s, "triple_p": fact.triple_p, "triple_o": fact.triple_o,
            "user_id": scope.user_id, "agent_id": scope.agent_id,
        } for fact, emb in to_save]
        uids = await mem_store.save_facts_batch(_get_redis(), save_specs)

        # BM25 insertion (in-memory) + persona merge, the latter batched into
        # two round trips (one HGETALL, one HSET pipeline) instead of 2N.
        persona_entries: list[tuple[str, str]] = []
        for (fact, _emb), uid in zip(to_save, uids):
            if not uid:
                continue
            attrs = {
                "content": fact.content, "category": fact.category,
                "language": lang, "domain": domain,
                "keywords": fact.keywords or [], "importance": fact.importance,
            }
            if fact.triple_s and fact.triple_p and fact.triple_o:
                attrs["triple_str"] = f"{fact.triple_s} | {fact.triple_p} | {fact.triple_o}"
            attrs.update(scope.attrs())
            await _bm25_index.add(uid, fact.content, attrs)
            persona_entries.append((fact.category, fact.content))
            fact_saved += 1
        if persona_entries:
            await persona_mod.update_many(
                _get_redis(), persona_entries, key=scope.persona_key(),
            )

        # 5b. Back-fill ep_type on newly saved episode + complete causal chain
        if new_ep_uid:
            ep_type = _infer_ep_type(facts)
            # Update ep_type attr on the episode we just saved
            try:
                raw = await _get_redis().execute_command("VGETATTR", mem_store.EPISODE_KEY, new_ep_uid)
                if raw:
                    ep_attrs = _decode_attrs(raw)
                    ep_attrs["ep_type"] = ep_type
                    await _get_redis().execute_command(
                        "VSETATTR", mem_store.EPISODE_KEY, new_ep_uid, json.dumps(ep_attrs)
                    )
            except Exception:
                pass  # VGETATTR not critical — ep_type stays "general"
            # Complete doubly-linked chain: tell prev episode its successor
            # (prev_ep_id was already fetched before save_episode above)
            if prev_ep_id and prev_ep_id != new_ep_uid:
                _spawn(
                    mem_store.update_episode_next_id(_get_redis(), prev_ep_id, new_ep_uid),
                    "ep-chain",
                )
            # Advance the session's last-episode pointer
            await mem_store.set_last_episode_id(_get_redis(), session_id, new_ep_uid)
            log.debug(f"[store] episode {new_ep_uid} type={ep_type} prev={prev_ep_id or 'none'}")

        # 6. Extract and store procedures (the 4th cognitive tier)
        proc_facts = [f for f in facts if f.category == "procedure"]
        # Known tool name whitelist — matched case-insensitively against procedure text.
        # Using a whitelist (not heuristic capitalization) avoids storing random words
        # like ['When', 'HEARTBEAT.md,'] as "tool names" (bug fixed v0.9.5).
        _KNOWN_TOOLS = {
            "bash", "glob", "grep", "read", "edit", "write", "websearch", "webfetch",
            "agent", "task", "notebook", "notebookedit", "mcp", "skill",
        }
        # Batch-encode procedure contents off the event loop
        if proc_facts:
            proc_contents = [pf.content for pf in proc_facts]
            proc_embs = await _encode_batch_async(proc_contents)
            # One VSIM pipeline for every procedure's dedup check.
            proc_hits = await mem_store.knn_search_batch(
                _get_redis(), mem_store.PROC_KEY, proc_embs, k=1,
            )
            for pf, p_emb, hits in zip(proc_facts, proc_embs, proc_hits):
                top = hits[0] if hits else None
                score = top.get("score", 0.0) if top else 0.0
                if score >= 0.90:
                    continue
                content_lower = pf.content.lower()
                tools_in_proc = [t for t in _KNOWN_TOOLS if t in content_lower]
                proc_uid = await mem_store.save_procedure(
                    _get_redis(), task=pf.content, procedure=pf.content,
                    embedding=p_emb, tools_used=tools_in_proc[:8],
                    domain=domain, language=lang,
                )
                # Procedures were keyword-invisible too — the largest tier by count.
                await _bm25_index.add(
                    proc_uid, pf.content, {"category": "procedure"}, "procedures"
                )

        # 7. Record knowledge graph edges from persons + entities co-occurrences.
        #    Pass every fact's entity group at once so the edge upserts run in a
        #    single pipeline instead of N*(N-1) serial Lua round trips.
        graph_groups: list[tuple[list[str], list[str]]] = []
        for fact in facts:
            if fact.persons or fact.entities:
                all_ents = (fact.persons or []) + (fact.entities or [])
                if len(all_ents) >= 2:
                    graph_groups.append((all_ents, []))
        if graph_groups:
            await graph_mod.record_entities_many(_get_redis(), graph_groups)


        # 8. Accumulate rolling session summary (Tier 1 — layered controlled architecture)
        #    Append this turn's summary instead of overwriting, so Tier 1 grows across
        #    the whole session rather than holding only the last turn.
        await _accumulate_session(session_id, summary)

        # 9. Auto-consolidation: trigger after every settings.auto_consolidate_every stored facts
        global _stores_since_consolidation
        fact_count = await _stores_since_consolidation.increment(fact_saved)
        if fact_count >= settings.auto_consolidate_every:
            await _stores_since_consolidation.reset()
            _spawn(_do_consolidate(_get_redis(), _bm25_index, _spawn), "consolidate")
            log.info("[store] auto-consolidation triggered")

        ms = int((time.time() - t0) * 1000)
        await _store_successes.increment()
        outcome = "success"
        log.info(f"[store] session={session_id} lang={lang} domain={domain} "
                 f"ep+{ep_saved} facts+{fact_saved} {ms}ms")

    except Exception as e:
        log.warning(f"[store] background error: {e}", exc_info=True)
    finally:
        ms = int((time.time() - t0) * 1000)
        await _store_latency_sum_ms.add(ms)
        if outcome == "skip":
            await _store_skips.increment()
        elif outcome == "error":
            await _store_errors.increment()

# ── Session helpers (Tier 1 — rolling accumulation + promote) ─────────────────

async def _accumulate_session(session_id: str, turn_summary: str) -> None:
    """
    Tier 1 rolling session summary.

    v1.1 (MemAgent §3.1): uses overwrite_update() instead of naive concatenation.
    When accumulated text exceeds 1200 chars the LLM sees existing memory +
    new turn and selectively retains critical facts — mirrors MemAgent's overwrite
    strategy (arXiv:2507.02259). Memory stays bounded at ~1200 chars regardless
    of session length; total compute is O(N) not O(N²).

    Security: secrets are redacted before storing in Tier 1 KV so they never
    appear in prependContext injected into future prompts.
    """
    if not session_id:
        return
    # Redact secrets before writing to Tier 1 KV (prevents leakage via prependContext)
    turn_summary = _redact_secrets(turn_summary)
    existing = await mem_store.get_session_context(_get_redis(), session_id) or ""
    if not existing:
        await mem_store.set_session_context(_get_redis(), session_id, turn_summary[:1500])
        return
    combined = f"{existing}\n---\n{turn_summary}"
    if len(combined) > 1200:
        # MemAgent-style incremental overwrite: LLM decides what to keep
        combined = await summarizer.overwrite_update(existing, turn_summary, target_chars=900)
    await mem_store.set_session_context(_get_redis(), session_id, combined[:1500])


async def _do_compress_session(session_id: str, scope: Scope = Scope()) -> dict:
    """
    Promote Tier 1 session context → Tier 2 long-term vector memory.
    Called by /session/compress (explicit) or the JS plugin on agent_end.

    Steps:
      1. Read accumulated session summary from Redis KV
      2. Summarize the whole session via LLM
      3. Save as a labelled episode in mem:episodes
      4. Run hybrid fact extraction → save facts + update persona
      5. Delete session KV (consumed, now lives in long-term memory)
    """
    if not session_id:
        return {"status": "skipped", "reason": "no session_id"}

    # A compress call usually carries only the session id; the track recorded
    # when the session was stored is what keeps the summary with its owner.
    if scope.is_global:
        scope = await mem_store.get_session_owner(_get_redis(), session_id)
    scope_filter = scope.filter() or None

    ctx = await mem_store.get_session_context(_get_redis(), session_id)
    if not ctx or len(ctx) < 80:
        return {"status": "skipped", "reason": "session too short to promote"}

    # Strip platform tags and cross_session_memory wrapping from the accumulated
    # session context before it is summarised and stored as a long-term episode.
    ctx = _strip_platform_noise(ctx)
    if not ctx or len(ctx) < 80:
        return {"status": "skipped", "reason": "session context empty after noise strip"}

    sc = scene_mod.detect(ctx[:500])
    lang, domain = sc["language"], sc["domain"]

    # Summarize the entire accumulated session context
    session_summary = await summarizer.summarize(ctx)

    # Save as a session-summary episode (labelled so recall can distinguish)
    ep_emb  = await asyncio.to_thread(_encode, session_summary[:500])
    similar = await mem_store.knn_search(_get_redis(), mem_store.EPISODE_KEY, ep_emb, k=1,
                                         filter_expr=scope_filter)
    ep_saved = 0
    if not similar or similar[0].get("score", 0.0) < 0.95:
        ep_uid = await mem_store.save_episode(
            _get_redis(), session_id,
            f"[Session Summary] {session_summary[:2000]}",
            ep_emb, lang, domain,
            user_id=scope.user_id, agent_id=scope.agent_id,
        )
        await _bm25_index.add(
            ep_uid, session_summary[:2000],
            {"category": "episode", **scope.attrs()}, "episodes"
        )
        ep_saved = 1

    # Hybrid fact extraction on the full session text
    facts = await extractor.extract_hybrid(
        [{"role": "user", "content": ctx}], ctx
    )
    fact_saved = 0
    # Batch-encode all fact contents off the event loop
    non_secret_facts = [f for f in facts if not _contains_secret(f.content)]
    if non_secret_facts:
        fact_contents = [f.content for f in non_secret_facts]
        fact_embs = await _encode_batch_async(fact_contents)
        emb_iter = iter(fact_embs)
    else:
        emb_iter = iter([])
    for fact in facts:
        if _contains_secret(fact.content):
            continue
        f_emb = next(emb_iter)
        existing = await mem_store.knn_search(_get_redis(), mem_store.FACT_KEY, f_emb, k=1,
                                              filter_expr=scope_filter)
        if existing and existing[0].get("score", 0.0) > 0.95:
            if fact.keywords and existing[0].get("score", 0.0) > 0.80:
                _spawn(_evolve_similar_fact(
                    existing[0].get("_element", ""),
                    fact.keywords, fact.topic,
                ))
            continue
        uid = await mem_store.save_fact(
            _get_redis(), fact.content, fact.category, fact.confidence,
            f_emb, lang, domain,
            keywords=fact.keywords, persons=fact.persons, entities=fact.entities,
            importance=fact.importance, topic=fact.topic, location=fact.location,
            triple_s=fact.triple_s, triple_p=fact.triple_p, triple_o=fact.triple_o,
            user_id=scope.user_id, agent_id=scope.agent_id,
        )
        # Add to BM25 corpus
        attrs = {
            "content": fact.content, "category": fact.category,
            "language": lang, "domain": domain,
            "keywords": fact.keywords or [], "importance": fact.importance,
            **scope.attrs(),
        }
        if fact.triple_s and fact.triple_p and fact.triple_o:
            attrs["triple_str"] = f"{fact.triple_s} | {fact.triple_p} | {fact.triple_o}"
        await _bm25_index.add(uid, fact.content, attrs)
        await persona_mod.update(_get_redis(), fact.category, fact.content,
                                 key=scope.persona_key())
        fact_saved += 1

    # Delete Tier 1 KV — session has been crystallised into long-term memory
    await _get_redis().delete(f"{mem_store.SESSION_PRE}{session_id}:ctx")

    # Pin the session summary for the NEXT session's recall (cross-session handoff).
    # Strip platform tags before persisting — summaries must contain real content only.
    # Stored without TTL so it persists until the next compress overwrites it.
    # One pin per track: a summary of someone else's session is not this track's
    # continuity bridge.
    clean_summary = _strip_platform_noise(session_summary)
    if clean_summary:
        key = _PINNED_SESSION_KEY + ("" if scope.is_global else ":" + scope.track_suffix())
        await _get_redis().set(key, clean_summary[:600].encode())

    log.info(f"[compress] session={session_id} promoted → ep+{ep_saved} facts+{fact_saved}")
    return {
        "status":     "ok",
        "session_id": session_id,
        "ep_saved":   ep_saved,
        "facts_saved":fact_saved,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.post("/session/compact")
async def session_compact(req: CompactRequest):
    """
    Mid-session Tier 1 compression (v0.9.3 — inspired by claude-mem Endless Mode).

    Problem: Tier 1 session KV grows linearly with every turn accumulation.
    At ~1200 chars it is already LLM-compressed by _accumulate_session(), but
    across many tool uses / recall cycles the KV can still balloon to 3-5K chars.

    This endpoint re-compresses the KV on demand from a PostToolUse hook,
    keeping context complexity O(N) rather than O(N²).

    Idempotent: no-op if session KV is already below threshold_chars.
    """
    if not req.session_id:
        return {"status": "skipped", "reason": "no session_id"}

    # PostToolUse fires this on every tool. One compact per few seconds is enough;
    # the body is already a no-op when the KV is under the threshold.
    gate_ok = await _get_redis().set(
        f"mem:compact:gate:{req.session_id}", "1", nx=True, ex=8,
    )
    if not gate_ok:
        return {"status": "skipped", "reason": "debounced"}

    ctx = await mem_store.get_session_context(_get_redis(), req.session_id)
    if not ctx:
        return {"status": "skipped", "reason": "no session context"}

    size_before = len(ctx)
    if size_before <= req.threshold_chars:
        return {"status": "skipped", "reason": "below threshold",
                "size": size_before, "threshold": req.threshold_chars}

    # MemAgent-style chunked overwrite (arXiv:2507.02259 §3.1):
    # Split context into head (existing memory) + tail (new info to incorporate).
    # overwrite_update uses max_tokens=400 giving a proper ~900-char result,
    # unlike summarize()'s max_tokens=80 which truncates mid-sentence.
    # For very long contexts iterate over multiple chunks.
    _CHUNK = 1200
    if size_before > _CHUNK * 2:
        chunks = [ctx[i:i + _CHUNK] for i in range(0, len(ctx), _CHUNK)]
        memory = await summarizer.summarize(chunks[0])
        for chunk in chunks[1:]:
            memory = await summarizer.overwrite_update(memory, chunk, target_chars=1000)
        compacted = memory
    else:
        # Single chunk: split head/tail so overwrite_update has both parts in context
        mid = len(ctx) // 2
        compacted = await summarizer.overwrite_update(ctx[:mid], ctx[mid:], target_chars=1000)

    await mem_store.set_session_context(_get_redis(), req.session_id, compacted[:1500])

    size_after = len(compacted)
    log.info(
        f"[compact] session={req.session_id} {size_before}→{size_after} chars "
        f"({round(100*(1-size_after/size_before))}% reduction)"
    )
    return {
        "status":        "ok",
        "session_id":    req.session_id,
        "size_before":   size_before,
        "size_after":    size_after,
        "reduction_pct": round(100 * (1 - size_after / size_before), 1),
    }


@app.post("/answer")
async def answer(req: AnswerRequest):
    """
    Extract a concise answer from recalled context using the LLM.

    Mirrors SimpleMem's AnswerGenerator (core/answer_generator.py):
    LLM reads the retrieved context and produces a short span answer,
    enabling token-F1 computation comparable to published LoCoMo baselines
    (which all use LLM-in-the-loop answer generation, not raw context F1).

    Returns {"answer": "<short span>"} or {"answer": ""} if LLM unavailable.
    """
    _AISERV_OAI = "http://127.0.0.1:4000/v1/chat/completions"
    _AISERV_KEY = "sk-aiserv-local-dev"

    if not req.context.strip():
        return {"answer": ""}

    def _parse_answer_raw(raw: str) -> str:
        """Parse JSON answer from LLM response, handling fences and both shapes."""
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        obj_start = raw.find("{")
        arr_start = raw.find("[")
        starts = [x for x in [obj_start, arr_start] if x != -1]
        if not starts:
            return ""
        start = min(starts)
        end_ch = "}" if raw[start] == "{" else "]"
        end = raw.rfind(end_ch)
        if end == -1:
            return ""
        parsed = json.loads(raw[start:end + 1])
        if isinstance(parsed, dict):
            return str(parsed.get("answer", ""))
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return str(parsed[0].get("answer", ""))
        return ""

    prompt = (
        f"Based only on the following memory context, answer the question concisely "
        f"using the exact phrase or name from the context (1-8 words preferred).\n\n"
        f"Context:\n{req.context[:2500]}\n\n"
        f"Question: {req.query}\n\n"
        f'Return JSON: {{"answer": "<exact phrase from context>"}}'
    )
    system_msg = ("You are a precise question-answering assistant. "
                  "Extract the exact answer phrase from the provided context. "
                  "Output valid JSON only.")

    # Two attempts: first with best QA model (qa role = stronger reasoning),
    # retry with fallback on timeout/failure.
    tried: list[str] = []
    exclude: str | None = None
    for attempt in range(2):
        model, _ = await extractor._resolve_qa_model(exclude=exclude)
        if model in tried:
            break
        tried.append(model)
        try:
            data = await async_post_json(
                _AISERV_OAI,
                payload={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user",   "content": prompt},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 80,
                },
                headers={"Authorization": f"Bearer {_AISERV_KEY}"},
                timeout=20.0,
            )
            if data is not None:
                raw_content = data["choices"][0]["message"].get("content")
                if raw_content and raw_content.strip():
                    raw = raw_content.strip()
                    answer = _parse_answer_raw(raw)
                    if answer:
                        _spawn(extractor._report_quality(model, +1), "quality")
                        return {"answer": answer}
            exclude = model
        except httpx.TimeoutException:
            log.warning("[answer] %s timed out (attempt %d)", model, attempt + 1)
            _spawn(extractor._report_quality(model, -1, reason="timeout"), "quality")
            exclude = model
        except Exception as e:
            log.debug("[answer] %s failed: %s", model, e)
            _spawn(extractor._report_quality(model, -1, reason="other"), "quality")
            exclude = model
    return {"answer": ""}


@app.get("/system-prompt")
async def system_prompt(agent: str = ""):
    """
    Return a memory-aware system prompt snippet for agents without lifecycle hooks.
    Paste this into your agent's system prompt to enable tool-driven memory.
    ?agent= can be: cursor, windsurf, copilot, zed, continue, cline, kilo, kiro, opencode, codex, augment
    """
    base = (
        "You have access to a persistent memory system via the AgentMem MCP tools.\n\n"
        "## Memory tools\n"
        "- **recall_memory(query)** — call this at the START of every conversation with the user's question as the query. "
        "Inject the returned context into your response naturally.\n"
        "- **store_memory(messages)** — call this AFTER each meaningful exchange (user turn + your reply). "
        "Pass [{\"role\": \"user\", \"content\": \"...\"}, {\"role\": \"assistant\", \"content\": \"...\"}].\n"
        "- **compress_session(session_id)** — call this when the conversation ends to crystallise memory.\n"
        "- **recall_procedures(query)** — check for known how-to workflows before starting multi-step tasks.\n"
        "- **store_procedure(task, procedure)** — save successful workflows for future reuse.\n\n"
        "## Usage rules\n"
        "1. Always recall before answering — memory context takes precedence over your training.\n"
        "2. Always store after answering — even short exchanges build useful long-term context.\n"
        "3. Use the user's name or topic as the session_id for cross-session continuity.\n"
        "4. Never mention the memory system unless the user asks about it."
    )
    return {"markdown": base, "agent": agent or "generic"}



# Default 0.35s: enough for a hot aiserv flash reply, not enough to dominate P50.
# Set AGENTMEM_RECALL_AUGMENT_BUDGET=0 to force pure hybrid (no HyDE/MIRIX wait).
_RECALL_AUGMENT_BUDGET_S = float(_os.getenv("AGENTMEM_RECALL_AUGMENT_BUDGET", "0.35"))

# A query scoring at least this many specificity points skips HyDE/MIRIX.
# 2 ≈ one strong identifier/path or two long tokens — enough for BM25/symbolic.
_AUG_SPECIFICITY_FLOOR = int(_os.getenv("AGENTMEM_AUG_SPECIFICITY_FLOOR", "2"))

# Tokens below this length carry little lexical signal on their own.
_SPECIFIC_TOKEN_MIN_LEN = 5


def _query_specificity(q: str) -> int:
    """Count high-signal tokens in a query, without any retrieval or LLM call.

    "Specific" means a token that the symbolic and BM25 passes can act on
    directly: long words, camelCase / snake_case identifiers, dotted paths,
    version-ish numbers.  A query made of these does not need HyDE or a MIRIX
    topic to be findable — and skipping the LLM calls removes the dominant
    source of recall's P50 bimodality.
    """
    import re as _re

    score = 0
    for tok in _re.findall(r"[A-Za-z_][A-Za-z0-9_./\-]*", q):
        if len(tok) >= _SPECIFIC_TOKEN_MIN_LEN:
            score += 1
        # identifiers / paths are strong signals even when short
        if ("_" in tok and len(tok) >= 6) or ("/" in tok) or ("." in tok and len(tok) >= 6):
            score += 1
        if _re.search(r"[a-z][A-Z]", tok):  # camelCase
            score += 1
    return score


async def _recall_augmentations(req: RecallRequest, *, skip_llm: bool = False) -> dict:
    """At most two LLM augmentations, under one wall-clock budget.

    HyDE, MIRIX topic, query-structure, and plan_queries used to run as
    separate stages with an 8s multi-model retry each. That held /recall
    for 13–24s after the Claude hook had already abandoned the request at 4s.

    ``skip_llm`` short-circuits the whole thing for queries that already carry
    enough lexical specificity for the symbolic + BM25 passes.  Every skipped
    call is one fewer HTTP round trip to aiserv on the recall hot path.
    """
    if skip_llm:
        return {}
    tasks: dict[str, asyncio.Task] = {}
    q = req.query.strip()
    if req.enable_hyde and len(q) > 10:
        tasks["hyde"] = asyncio.create_task(retrieval_planner.generate_hyde_doc(req.query))
    if req.enable_planning:
        tasks["plan"] = asyncio.create_task(retrieval_planner.plan_queries(req.query))
    elif len(q) > 8:
        tasks["topic"] = asyncio.create_task(retrieval_planner.generate_retrieval_topic(req.query))
    if not tasks:
        return {}
    _done, pending = await asyncio.wait(set(tasks.values()), timeout=_RECALL_AUGMENT_BUDGET_S)
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
        log.info(
            "[recall] augmentation budget %.1fs hit, dropped %s",
            _RECALL_AUGMENT_BUDGET_S,
            ",".join(name for name, task in tasks.items() if task in pending),
        )
    out: dict = {}
    for name, task in tasks.items():
        if task in pending or task.cancelled():
            continue
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            continue
        if exc is not None:
            log.debug("[recall] %s augmentation failed: %s", name, exc)
            continue
        out[name] = task.result()
    return out


@app.post("/recall")
async def recall(req: RecallRequest):
    t0 = time.time()
    scope = _request_scope(req.user_id, req.agent_id)
    scope_filter = scope.filter() or None

    if _is_brief_option_reply(req.query):
        hint = (
            "<cross_session_memory>\n"
            "The user's latest message is a brief numeric option selection.\n"
            "Resolve it against the numbered options in the immediately preceding assistant message in this same conversation.\n"
            "Only ask for clarification if no numbered options were offered.\n"
            "</cross_session_memory>"
        )
        ms = int((time.time() - t0) * 1000)
        log.info(f"[recall] session={req.session_id} option-reply shortcut {ms}ms")
        return {"prependContext": hint, "latency_ms": ms}

    # Detect scene of incoming query
    query_scene = scene_mod.detect(req.query)
    q_lang, q_domain = query_scene["language"], query_scene["domain"]

    if embedder.embedding_unavailable():
        ms = int((time.time() - t0) * 1000)
        log.debug("[recall] embedding circuit open — skipping retrieval")
        return {"prependContext": "", "facts": [], "episodes": [], "latency_ms": ms, "error": "embedding service unavailable"}

    # Overlap the query embed with the augmentation LLM calls.
    emb_task = asyncio.create_task(async_encode(req.query))
    # HyDE/MIRIX are the dominant recall-latency term: each is an HTTP round trip
    # to the local aiserv LLM proxy, and the 2.5s budget is what turns a 7ms
    # recall into a 2.1s one whenever aiserv is slow.  A query that already
    # carries lexical specificity does not need them — the symbolic and BM25
    # passes act on those tokens directly.
    _aug_skip = _query_specificity(req.query) >= _AUG_SPECIFICITY_FLOOR
    if _aug_skip:
        log.debug("[recall] skipping LLM augmentation (specificity=%d >= %d)",
                  _query_specificity(req.query), _AUG_SPECIFICITY_FLOOR)
    aug_task = asyncio.create_task(_recall_augmentations(req, skip_llm=_aug_skip))
    emb_res, aug_res = await asyncio.gather(emb_task, aug_task, return_exceptions=True)
    if isinstance(emb_res, Exception):
        log.warning("[recall] embedding failed: %s", emb_res)
        ms = int((time.time() - t0) * 1000)
        return {"prependContext": "", "facts": [], "episodes": [], "latency_ms": ms, "error": "embedding service unavailable"}
    emb = emb_res
    aug = aug_res if isinstance(aug_res, dict) else {}

    # HyDE (Gao et al. 2022): blend a hypothetical memory into the query vector.
    # MIRIX topic, when planning is off, replaces the BM25 string for vague prompts.
    _hyde_doc = aug.get("hyde") if isinstance(aug.get("hyde"), str) else None
    _mirix_topic = aug.get("topic") if isinstance(aug.get("topic"), str) else None
    _planned = aug.get("plan") if isinstance(aug.get("plan"), list) else None

    if _hyde_doc:
        try:
            hyde_emb = await async_encode(_hyde_doc)
            merged = (emb * 0.5 + hyde_emb * 0.5).astype(np.float32)
            norm = np.linalg.norm(merged)
            emb = merged / norm if norm > 0 else merged
        except Exception:
            log.debug("[recall] HyDE embedding failed, using query embedding only")

    _bm25_query = _mirix_topic if _mirix_topic else req.query

    # Adaptive retrieval depth (SimpleMem-inspired C_q estimator)
    n_facts, n_episodes = _estimate_depth(req.query, req.memory_limit_number)

    # Build filter expressions for scene isolation
    lang_filter   = f'.language == "{q_lang}"'
    domain_filter = f'.domain == "{q_domain}"' if q_domain != "general" else None

    # SimpleMem symbolic filter: time-range hard constraint
    # Combines with lang_filter + domain_filter as a compound VSIM FILTER expression
    def _build_filter(base_filter: str | None, domain_f: str | None = None) -> str:
        parts = []
        # The track is a hard constraint like language and time range: it goes
        # into the VSIM expression, so the index returns only this track's rows
        # rather than this track's rows surviving a later filter.
        if scope_filter:
            parts.append(scope_filter)
        if base_filter:
            parts.append(base_filter)
        if domain_f:
            parts.append(domain_f)
        if req.time_from is not None:
            parts.append(f".ts >= {req.time_from}")
        if req.time_to is not None:
            parts.append(f".ts <= {req.time_to}")
        return " && ".join(parts) if parts else ""

    lang_filter_sym = _build_filter(lang_filter, domain_filter)

    # Recall strategy: scene-filtered first, supplement with global if sparse
    async def _fetch(vset: str, k: int, fexpr=None):
        return await mem_store.knn_search(
            _get_redis(), vset, emb, k, filter_expr=fexpr, bump_heat=True
        )

    # Detect if this is a capability query (needs tool context)
    is_cap_query = query_scene.get("is_capability_query", False)

    # Parallel: persona + env + session ctx + scene-filtered facts + episodes
    # + tool recall if capability query or include_tools=True
    # + symbolic structured pass (async, no extra latency via gather)
    # + pinned last-session summary (cross-session handoff bridge, index 6)
    gather_tasks = [
        persona_mod.get_context(_get_redis(), key=scope.persona_key()),
        cap_mod.get_env_context(_get_redis()),
        mem_store.get_session_context(_get_redis(), req.session_id),
        _fetch(mem_store.FACT_KEY,    n_facts,    lang_filter_sym or None),
        _fetch(mem_store.EPISODE_KEY, n_episodes, lang_filter_sym or None),
        asyncio.sleep(0),  # index 5 kept stable; structure LLM is outside the recall budget
        # Handoff bridge (index 6): one pin per track, never another track's.
        _get_redis().get(
            _PINNED_SESSION_KEY + ("" if scope.is_global else ":" + scope.track_suffix())
        ),
    ]
    if req.include_tools and is_cap_query:
        gather_tasks.append(cap_mod.recall_tools(_get_redis(), emb, k=6))
    else:
        gather_tasks.append(asyncio.sleep(0))
    # Procedure recall (index 7): fire in parallel if include_procedures=True
    # Uses raw VSIM (not knn_search) because procedures store 'task' not 'content'.
    async def _fetch_procs(embedding, k: int = 2) -> list[dict]:
        """Fetch top-k procedures with MACLA Beta posterior scoring (v0.9.5).

        Beta(success+1, fail+1) × cosine_sim reranks procedures by both
        relevance AND historical success rate (arXiv:2512.18950).
        """
        blob = embedding.astype("float32").tobytes()
        # Fetch more than needed so Beta reranking can reorder
        cmd  = ["VSIM", mem_store.PROC_KEY, "FP32", blob,
                "COUNT", max(k * 3, 6), "WITHSCORES", "WITHATTRIBS"]
        # No track FILTER: `/store-procedure` writes the shared how-to tier without an owner
        # attribute, so filtering here hid every procedure from any caller that names a track
        # (measured 2026-10-28). `Scope.sees` post-filters instead: unowned = shared, owned = own track.
        try:
            results = await _get_redis().execute_command(*cmd)
        except Exception:
            return []
        procs = []
        idx = 0
        while idx + 2 < len(results):
            elem = results[idx]; score = results[idx+1]; raw = results[idx+2]
            idx += 3
            elem_str = _decode_bytes(elem)
            if elem_str == "__seed__":
                continue
            try:
                attrs = _decode_attrs(raw)
            except Exception:
                continue
            if attrs.get("_seed") or not attrs.get("task"):
                continue
            if not scope.sees(attrs.get("user_id") or attrs.get("agent_id")):
                continue
            cosine = float(score) if score else 0.0
            # MACLA Beta posterior: E[Beta(α,β)] = α/(α+β) with α=success+1, β=fail+1
            success = attrs.get("success_count", 0)
            fail    = attrs.get("fail_count", 0)
            total   = success + fail
            beta_mean = (success + 1) / (total + 2) if total >= 3 else 1.0
            procs.append({
                "task":          attrs.get("task", ""),
                "procedure":     attrs.get("procedure", ""),
                "_adjusted_score": cosine * beta_mean,
                "_cosine":       cosine,
            })
        # Re-sort by Beta-weighted score
        procs.sort(key=lambda x: x["_adjusted_score"], reverse=True)
        return procs[:k]

    if req.include_procedures:
        gather_tasks.append(_fetch_procs(emb))
    else:
        gather_tasks.append(asyncio.sleep(0))
    
    # Crystallized digests (index 8): fetch recent auto-crystallized session summaries.
    # Uses a sorted set index (mem:crystallized_index, score=crystallized_at) instead of
    # SCAN so lookup is O(log N) regardless of total key count.
    async def _fetch_crystallized_digests() -> list[dict]:
        """Fetch top-3 most recent crystallized session digests via sorted-set index."""
        try:
            top_keys = await _get_redis().zrevrange(_CRYSTALLIZED_INDEX_KEY, 0, 9)
            if not top_keys:
                return []

            pipe = _get_redis().pipeline(transaction=False)
            for key in top_keys:
                pipe.get(key)
            raw_values = await pipe.execute()

            digests = []
            for raw in raw_values:
                if not raw:
                    continue
                try:
                    digest = _decode_json(raw)
                    digests.append(digest)
                except Exception:
                    continue

            digests.sort(key=lambda d: d.get("crystallized_at", 0), reverse=True)
            return digests[:3]

        except Exception as e:
            log.warning(f"[recall] failed to fetch crystallized digests: {e}")
            return []
    
    gather_tasks.append(_fetch_crystallized_digests())

    results = await asyncio.gather(*gather_tasks)
    persona_ctx  = results[0]
    env_ctx      = results[1]
    session_ctx  = results[2]
    # A flat session key belongs to whoever opened it; another track's live
    # conversation must not ride along in this prompt. Unrecorded (legacy)
    # sessions stay visible, which is the same rule unscoped rows follow.
    if not scope.is_global and session_ctx and req.session_id:
        owner = await mem_store.get_session_owner(_get_redis(), req.session_id)
        if (owner.user_id and owner.user_id != scope.user_id) or (
            owner.agent_id and owner.agent_id != scope.agent_id
        ):
            session_ctx = None
    facts_scene  = results[3]
    eps_scene    = results[4]
    query_struct = results[5] if isinstance(results[5], dict) else {}
    _pinned_raw  = results[6]
    last_session_summary = (
        _decode_bytes(_pinned_raw)
    ) if _pinned_raw else None
    tools_raw    = results[7] if isinstance(results[7], list) else []
    procs_raw    = results[8] if isinstance(results[8], list) else []
    crystal_digests = results[9] if isinstance(results[9], list) else []
    # A digest summarizes a session; which track wrote it is recorded with it.
    # Digests from before tracks existed carry no id and are global-only.
    if not scope.is_global:
        crystal_digests = [d for d in crystal_digests if scope.matches(d)]

    # Symbolic structured pass (AgentMem §3.3 extension):
    # If the query mentions specific persons or entities, do a targeted scan
    # for facts that explicitly contain those names — catches cases where
    # semantic similarity is low but the name is a direct match.
    sym_facts: list[dict] = []
    sym_entities = (query_struct.get("persons") or []) + (query_struct.get("entities") or [])
    if sym_entities:
        ent_vecs = await _encode_batch_async(sym_entities[:3])
        sym_tasks = [
            mem_store.knn_search(_get_redis(), mem_store.FACT_KEY, vec, k=3,
                                 filter_expr=lang_filter_sym or None)
            for vec in ent_vecs
        ]
        sym_batches = await asyncio.gather(*sym_tasks)
        seen_sym = {f["content"] for f in facts_scene}
        for batch in sym_batches:
            for item in (batch if isinstance(batch, list) else []):
                if item["content"] not in seen_sym:
                    sym_facts.append(item)
                    seen_sym.add(item["content"])
        if sym_facts:
            log.debug(f"[recall] symbolic pass: +{len(sym_facts)} facts for entities {sym_entities[:3]}")

    # Planning ran inside the augmentation budget above. Use the queries only
    # when that call finished; otherwise the original query search stands.
    if req.enable_planning and _planned and len(_planned) > 1:
        targeted_queries = _planned
        if len(targeted_queries) > 1:
            # Run targeted queries in parallel (exclude original already fetched above)
            extra_queries = [q for q in targeted_queries if q != req.query]
            try:
                extra_embs = await _encode_batch_async(extra_queries)
            except embedder.EmbeddingUnavailable:
                extra_embs = []
            extra_tasks = [
                mem_store.knn_search(_get_redis(), mem_store.FACT_KEY, emb, n_facts,
                                     filter_expr=lang_filter_sym or None, bump_heat=True)
                for emb in extra_embs
            ]
            extra_results = await asyncio.gather(*extra_tasks)
            seen_extra = {f["content"] for f in facts_scene}
            for batch in extra_results:
                for item in (batch if isinstance(batch, list) else []):
                    if item["content"] not in seen_extra:
                        facts_scene.append(item)
                        seen_extra.add(item["content"])
            log.info(f"[recall] planning: {len(targeted_queries)} queries, "
                     f"total facts after planning: {len(facts_scene)}")

    # Supplement with global search if scene results are sparse.
    # BM25 is pure in-memory (no I/O) but async — fold it into this gather so it
    # runs concurrently with the Redis supplement fetches at zero extra latency cost.
    supplement_tasks = []
    if len(facts_scene) < n_facts:
        supplement_tasks.append(
            _fetch(mem_store.FACT_KEY, n_facts - len(facts_scene), scope_filter)
        )
    else:
        supplement_tasks.append(asyncio.sleep(0))

    if len(eps_scene) < max(2, n_episodes // 2):
        supplement_tasks.append(
            _fetch(mem_store.EPISODE_KEY, n_episodes - len(eps_scene), scope_filter)
        )
    else:
        supplement_tasks.append(asyncio.sleep(0))

    supplement_tasks.append(_bm25_index.search(_bm25_query, k=n_facts))

    supp_results = await asyncio.gather(*supplement_tasks)
    facts_global = supp_results[0] if isinstance(supp_results[0], list) else []
    eps_global   = supp_results[1] if isinstance(supp_results[1], list) else []
    bm25_hits    = supp_results[2] if isinstance(supp_results[2], list) else []
    # BM25 ranks the in-memory corpus without knowing about tracks, so its hits
    # carry their attributes and are filtered here.
    if not scope.is_global:
        bm25_hits = [h for h in bm25_hits if scope.matches(h.get("attrs"))]
    if bm25_hits:
        log.debug(f"[bm25] {len(bm25_hits)} hits for: {req.query[:60]!r}")

    # BM25 now indexes every tier, so route each hit back to the tier it came
    # from.  Before the multi-tier fix the corpus was facts-only, which left
    # ~79% of the text-bearing memory unreachable by keyword search.
    bm25_by_tier: dict[str, list[dict]] = {"facts": [], "episodes": [], "procedures": []}
    for hit in bm25_hits:
        tier = hit.get("tier") or "facts"
        bm25_by_tier.setdefault(tier, []).append(hit)
    bm25_facts     = bm25_by_tier.get("facts", [])
    bm25_episodes  = bm25_by_tier.get("episodes", [])
    bm25_procedures = bm25_by_tier.get("procedures", [])

    # Keyword hits on procedures feed the same MACLA-scored procedure list, so
    # a lexical match surfaces a how-to even when its embedding is mediocre.
    if bm25_procedures and isinstance(procs_raw, list):
        seen_proc = {p.get("task") for p in procs_raw}
        for hit in bm25_procedures:
            attrs = hit.get("attrs") or {}
            task = attrs.get("task") or ""
            if task and task not in seen_proc:
                seen_proc.add(task)
                procs_raw.append({
                    "task":          task,
                    "procedure":     attrs.get("procedure", ""),
                    "_adjusted_score": float(hit.get("score", 0.0)),
                    "_cosine":       0.0,
                    "_from_bm25":    True,
                })
        procs_raw.sort(key=lambda x: x.get("_adjusted_score", 0.0), reverse=True)

    # Equal-weight RRF.  The query-type hand-tuned weight presets that used to
    # live here ([0.8,0.8,1.4,1.2] / [0.9,0.9,1.2,1.1] / [0.9,0.9,1.2,0.8] /
    # [1.0,1.0,0.6,0.9]) were measured against equal weights on a 7-scenario
    # synthetic sweep: they lost nDCG@10 in 6 of 7 scenarios, by up to -12.6%.
    # See docs/EVEROS_ASSESSMENT.md §4.1.
    _rrf_weights = [1.0, 1.0, 1.0, 1.0]

    def _merge(primary, supplement, symbolic, bm25, limit):
        p_ranked   = heat_mod.heat_rerank(primary)
        s_ranked   = heat_mod.heat_rerank(supplement) if supplement else []
        sym_ranked = heat_mod.heat_rerank(symbolic)   if symbolic   else []
        # BM25 hits are already scored by TF-IDF; pass as-is (no heat reranking)
        fused = _rrf_merge(
            [p_ranked, s_ranked, sym_ranked, bm25],
            weights=_rrf_weights,
            limit=limit * 2,
        )
        return fused[:limit]

    facts    = _merge(facts_scene, facts_global, sym_facts,  bm25_facts,     n_facts)
    episodes = _merge(eps_scene,   eps_global,   [],         bm25_episodes,  n_episodes)

    # SimpleMem Phase 3: lexical keyword boost (BM25-lite additive layer)
    # Note: BM25 above handles full TF-IDF scoring; this adds a lightweight keyword
    # overlap boost on top for any residual exact-match signal not captured by BM25.
    facts    = _keyword_boost(facts,    req.query)
    episodes = _keyword_boost(episodes, req.query, boost=0.03)

    # AgentMem advantage: importance-weighted reranking.
    # High-importance facts (rules, identity, preferences) surface above transient context.
    facts    = _importance_boost(facts)
    episodes = _importance_boost(episodes, weight=0.05)  # lighter for episodes

    # Session diversity: cap at 3 results per source session to prevent
    # one verbose session drowning out facts from other sessions.
    facts    = _session_diversify(facts,    max_per_session=3)
    episodes = _session_diversify(episodes, max_per_session=4)

    # SimpleMem reflection loop (opt-in): check if results sufficient, if not re-fetch
    if req.enable_reflection and facts:
        is_sufficient = await retrieval_planner.check_sufficiency(req.query, facts)
        if not is_sufficient:
            # One additional targeted pass with a different angle
            extra_q = await retrieval_planner.plan_queries(
                f"Find additional information about: {req.query}"
            )
            if extra_q:
                reflection_emb = await async_encode(extra_q[0])
                reflection_facts = await mem_store.knn_search(
                    _get_redis(), mem_store.FACT_KEY, reflection_emb,
                    k=max(3, n_facts // 2), filter_expr=scope_filter,
                    bump_heat=True,
                )
                seen_reflect = {f["content"] for f in facts}
                for rf in reflection_facts:
                    if rf["content"] not in seen_reflect:
                        facts.append(rf)
                        seen_reflect.add(rf["content"])
                facts = facts[:n_facts + 3]
                log.info(f"[recall] reflection pass: added {len(reflection_facts)} candidates")

    # Knowledge graph expansion (v0.9.0 + auto-trigger v1.0):
    # Fire when explicitly requested OR when query has named entities (auto_graph=True).
    # Auto-trigger catches "what did Bob tell me about Redis?" without needing include_graph.
    _auto_graph_trigger = req.auto_graph and bool(sym_entities)
    if (req.include_graph or _auto_graph_trigger) and facts:
        graph_ents: list[str] = []
        for f in facts[:3]:
            attrs = f.get("attrs", {})
            graph_ents += attrs.get("persons", []) + attrs.get("entities", [])
        graph_ents = list(dict.fromkeys(graph_ents))[:5]

        if graph_ents:
            graph_tasks = [
                graph_mod.entity_recall(_get_redis(), ent, emb, k=2)
                for ent in graph_ents
            ]
            graph_results = await asyncio.gather(*graph_tasks, return_exceptions=True)
            seen_contents = {f["content"] for f in facts}
            for batch in graph_results:
                if isinstance(batch, Exception):
                    log.warning("[recall] graph recall failed: %s", batch)
                    continue
                for gf in batch:
                    if not scope.matches(gf.get("attrs")):
                        continue
                    if gf["content"] not in seen_contents:
                        facts.append(gf)
                        seen_contents.add(gf["content"])
            facts = facts[:n_facts + 3]

    # AriadneMem Steiner tree bridge discovery (arXiv:2603.03290 §3.3):
    # Find facts that structurally bridge pairs of retrieved terminal facts.
    # Fires when include_graph=True and we have ≥2 retrieved facts.
    # Deterministic — no extra LLM call, single graph+vector pass.
    if (req.include_graph or _auto_graph_trigger) and len(facts) >= 2:
        try:
            bridge_facts = await graph_mod.find_bridge_nodes(_get_redis(), facts, emb, k=3)
            if bridge_facts:
                seen_bridge = {f["content"] for f in facts}
                for bf in bridge_facts:
                    if not scope.matches(bf.get("attrs")):
                        continue
                    if bf["content"] not in seen_bridge:
                        facts.append(bf)
                        seen_bridge.add(bf["content"])
                facts = facts[:n_facts + 5]
                log.debug(f"[recall] bridge discovery: +{len(bridge_facts)} bridge nodes")
        except Exception as _be:
            log.debug(f"[recall] bridge discovery skipped: {_be}")

    # AutoTool TIG: get transition hints from most-used recalled tool (v0.9.5)
    tig_hints: list[str] = []
    if tools_raw:
        top_tool_elem = tools_raw[0].get("_element", "")
        if top_tool_elem:
            try:
                # HSCAN with MATCH lets Redis do the prefix filtering.  The old
                # HGETALL pulled the whole transition graph into Python on every
                # recall, so cost grew linearly with total historical tool calls.
                prefix  = f"{top_tool_elem}:"
                trans: dict[str, int] = {}
                cursor = 0
                for _ in range(64):  # bounded; the graph is small in practice
                    cursor, batch = await _get_redis().hscan(
                        mem_store.TOOL_GRAPH_KEY, cursor=cursor,
                        match=f"{prefix}*", count=200,
                    )
                    for k_b, v_b in batch.items():
                        target = _decode_bytes(k_b)[len(prefix):]
                        try:
                            trans[target] = int(v_b)
                        except (TypeError, ValueError):
                            continue
                    if int(cursor) == 0:
                        break
                if trans:
                    total_trans = sum(trans.values())
                    sorted_trans = sorted(trans.items(), key=lambda x: x[1], reverse=True)
                    tig_hints = [
                        f"{t} ({round(c/total_trans*100)}%)"
                        for t, c in sorted_trans[:3]
                        if c / total_trans >= 0.15  # only show ≥15% probability edges
                    ]
            except Exception:
                pass

    # Format tool context (only include env if it's non-trivial)
    tool_ctx = cap_mod.format_tool_context(tools_raw, tig_hints=tig_hints or None) if tools_raw else ""
    env_ctx_formatted = env_ctx if env_ctx and is_cap_query else ""

    # proc_ctx: already decoded by _fetch_procs (list of {task, procedure} dicts)
    proc_ctx: list[dict] = procs_raw if isinstance(procs_raw, list) else []

    # Active working-memory + pending intentions (prospective memory layer).
    _active: dict | None = None
    _intents: list[dict] = []
    try:
        from core import focus as focus_mod
        _active = await focus_mod.get_active(_get_redis(), req.session_id or "default")
        if not (_active.get("goal") or _active.get("notes")):
            _active = None
        _intents = await focus_mod.list_intentions(_get_redis(), status="pending", limit=5)
    except Exception as e:
        log.debug("[recall] focus/intentions unavailable: %s", e)

    # SimpleMem Section 3.3 token-budget injection with XML wrapper
    # Skip pinned session summary if this session has already accumulated context
    # (session_ctx means we're mid-session, not starting fresh).
    _handoff = last_session_summary if not session_ctx else None
    prepend = _format_prepend(
        facts, episodes, session_ctx, persona_ctx,
        env_ctx=env_ctx_formatted, tool_ctx=tool_ctx,
        proc_ctx=proc_ctx or None,
        token_budget=req.token_budget or settings.default_token_budget,
        last_session_summary=_handoff,
        crystal_digests=crystal_digests or None,  # Auto-crystallized lessons learned
        active_focus=_active,
        intentions=_intents or None,
    )
    ms = int((time.time() - t0) * 1000)

    planning_info = f" planning={req.enable_planning}" if req.enable_planning else ""
    hyde_info = f" hyde={bool(_hyde_doc)}" if req.enable_hyde else ""
    topic_info = f" topic={_mirix_topic[:30]!r}" if _mirix_topic else ""
    log.info(f"[recall] session={req.session_id} lang={q_lang} domain={q_domain} "
             f"facts={len(facts)} ep={len(episodes)} tools={len(tools_raw)} procs={len(proc_ctx)}"
             f"{planning_info}{hyde_info}{topic_info} {ms}ms")

    return {"prependContext": prepend if prepend else None, "latency_ms": ms}


def _request_scope(user_id: object, agent_id: object) -> Scope:
    """The track a request names, or 422 when the ids cannot mean one."""
    try:
        return strict_scope(user_id, agent_id)
    except BadScope as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _store_scope(req: StoreRequest) -> Scope:
    """The track a store names: the fields, else the per-call metadata override.

    Raven's contract carries ``user_id`` / ``agent_id`` inside a trace write's
    ``metadata`` when one process writes on behalf of another; the top-level
    fields are the same two ids for callers that have them directly.
    """
    meta = req.metadata or {}
    return _request_scope(
        req.user_id or meta.get("user_id"),
        req.agent_id or meta.get("agent_id"),
    )


@app.post("/store")
async def store_memory(req: StoreRequest):
    # A row this index cannot hold is not "queued" — say so while the caller can
    # still act on it. The background task cannot report a VADD it was never able
    # to attempt; measured 2026-10-27, a 768-d provider against 1024-d sets answered
    # 200 forever while every write was rejected.
    try:
        await dim_guard.assert_writeable(_get_redis())
    except embedder.DimensionMismatch as exc:
        raise HTTPException(status_code=409,
                            detail={"error": "dimension_mismatch", "message": str(exc)})
    _spawn(_do_store(req.messages, req.session_id, _store_scope(req)), "store")
    await _ws_broadcast("store", {"session_id": req.session_id})
    return {"status": "queued"}


# ── Session Tier-1 endpoints (v0.7.0) ─────────────────────────────────────────

@app.post("/session/compress")
async def compress_session(req: CompressSessionRequest, background_tasks: BackgroundTasks):
    """
    Promote Tier 1 session KV → Tier 2 long-term vector memory.

    Call this from the agent_end hook when a session is finishing.
    The accumulated session summary is compressed, saved as an episode,
    and fact-extracted — then the session KV is deleted.

    Set wait=true to block until promotion completes (returns full result).
    """
    scope = _request_scope(req.user_id, req.agent_id)
    if req.wait:
        result = await _do_compress_session(req.session_id, scope)
        return result
    background_tasks.add_task(_do_compress_session, req.session_id, scope)
    return {"status": "queued", "session_id": req.session_id}


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """
    Inspect current Tier 1 session context for a given session_id.
    Useful for debugging what the agent has accumulated in this session.
    """
    ctx = await mem_store.get_session_context(_get_redis(), session_id)
    owner = await mem_store.get_session_owner(_get_redis(), session_id)
    return {
        "session_id": session_id,
        "context":    ctx,
        "length":     len(ctx) if ctx else 0,
        "tier":       "Tier 1 / Session KV",
        # The track that wrote this session, so a caller can tell an owned
        # session from one whose id it merely happens to know.
        "owner":      None if owner.is_global else {
            "user_id":  owner.user_id,
            "agent_id": owner.agent_id,
        },
    }


# ── Consolidation logic now in services/consolidation_service.py ──────────────
# ═══════════════════════════════════════════════════════════════════════════════
# Absorbed agentmemory scaffold endpoints (no /agentmemory prefix)
# These provide full compatibility with rohitg00/agentmemory hooks & MCP tools.
# ═══════════════════════════════════════════════════════════════════════════════
# _compat_sid and _compat_check_auth are now in api/compat.py — imported above.
# _observe_internal is now in services/store_service.py — imported by route modules.


# ── Automatic Crystallization Background Task ────────────────────────────────
# _crystallize_session_inline now in services/consolidation_service.py

async def _auto_crystallize() -> None:
    """
    Background task: automatically crystallize sessions older than 24h.
    
    Crystallization distills completed work chains into structured digests.
    Runs every 6 hours.
    """
    while True:
        await asyncio.sleep(21600)  # 6 hours
        try:
            now_ms = int(time.time() * 1000)
            twenty_four_hours_ms = 24 * 3600 * 1000

            # Phase 1: Batch-scan all session keys and crystallized keys
            session_pattern = "mem:session:*"
            cursor = 0
            all_session_keys = []

            while True:
                cursor, keys = await _get_redis().scan(cursor, match=session_pattern, count=100)
                all_session_keys.extend(keys)
                if cursor == 0:
                    break

            if not all_session_keys:
                continue

            # Batch check which sessions are already crystallized + get session data
            # Pipeline: exists(crystal_key) + get(session_key) per session
            pipe = _get_redis().pipeline(transaction=False)
            key_map = []  # track (session_key, session_id, crystal_key) for each pipeline pair
            for key in all_session_keys:
                key_str = _decode_bytes(key)
                session_id = key_str.replace("mem:session:", "")
                crystal_key = f"mem:crystallized:{session_id}"
                pipe.exists(crystal_key)
                pipe.get(key)
                key_map.append((key_str, session_id, crystal_key))

            pipe_results = await pipe.execute()

            # Parse pipeline results (pairs: exists, get for each session)
            candidates = []
            for idx, (key_str, session_id, crystal_key) in enumerate(key_map):
                exists = pipe_results[idx * 2]
                session_data = pipe_results[idx * 2 + 1]
                if exists:
                    continue
                if not session_data:
                    continue
                try:
                    session_obj = _decode_json(session_data)
                except Exception:
                    continue

                ts = session_obj.get("ts", 0)
                if not ts:
                    started_at = session_obj.get("started_at", 0)
                    if started_at:
                        ts = int(started_at * 1000)
                if not ts:
                    continue
                age_ms = now_ms - ts
                if age_ms < twenty_four_hours_ms:
                    continue

                candidates.append((session_id, session_obj, ts, crystal_key))

            if not candidates:
                continue

            # Phase 2: Fetch all facts once (not per-session) and count per session
            scanned = await _vscan(_get_redis(), mem_store.FACT_KEY, max_count=200)
            if len(scanned) <= 5:
                continue

            # Build fact timestamp list once
            fact_timestamps = [
                item["attrs"].get("ts", 0)
                for item in scanned
                if not item["attrs"].get("superseded_by")
            ]

            # Phase 3: Crystallize qualifying sessions
            crystallized_count = 0
            for session_id, session_obj, ts, crystal_key in candidates:
                # Count facts within ±1 hour of session start
                fact_count = sum(1 for ft in fact_timestamps if abs(ft - ts) < 3600000)
                if fact_count < 5:
                    continue

                log.info(f"[crystallize] auto-crystallizing session {session_id} ({fact_count} facts, age={(now_ms - ts)/3600000:.1f}h)")

                digest = await _crystallize_session_inline(session_id, session_obj, fact_count)
                if digest:
                    await _get_redis().setex(
                        crystal_key,
                        90 * 86400,
                        json.dumps(digest, ensure_ascii=False)
                    )
                    # Maintain sorted-set index so /recall can use ZREVRANGE instead of SCAN
                    await _get_redis().zadd(
                        _CRYSTALLIZED_INDEX_KEY,
                        {crystal_key: digest["crystallized_at"]},
                    )
                    crystallized_count += 1
                    await _ws_broadcast("crystallize_auto", {
                        "session_id": session_id,
                        "fact_count": fact_count,
                        "digest_summary": digest.get("summary", "")[:200]
                    })

            if crystallized_count > 0:
                log.info(f"[crystallize] auto-crystallized {crystallized_count} sessions")

        except Exception as e:
            log.warning(f"[crystallize] auto-crystallization error: {e}", exc_info=True)


# ── Viewer ────────────────────────────────────────────────────────────────────

@app.get("/viewer")
async def compat_viewer():
    idx = _os.path.join(_static_dir, "index.html")
    if _os.path.exists(idx):
        return FileResponse(idx)
    return {"message": "AgentMem Viewer", "version": "1.2.0"}
