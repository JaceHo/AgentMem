"""Capability routes — tools, environment, procedures, TIG, AWO, MACLA."""

import asyncio
import hashlib
import json
import logging
import time

from fastapi import APIRouter, BackgroundTasks, HTTPException

from api import state
from api.schemas.capability import (
    RegisterToolsRequest, EnvState, RecallToolsRequest, RecallProceduresRequest,
    StoreProcedureRequest, ToolFeedbackRequest, ToolSequenceRequest,
    ProcedureFeedbackRequest,
)
from core import capability as cap_mod
from core import embedder as embed_mod
from core import persona as persona_mod
from core import scene as scene_mod
from core import store as mem_store
from core.scope import BadScope, strict_scope
from core.search import async_encode, encode
from core.utils import decode_attrs, decode_bytes

log = logging.getLogger("mem")
router = APIRouter(tags=["capability"])

AWO_SIG_KEY = "mem:awo:chain_sig"


def _chain_signature(chains: list[dict]) -> str:
    """Stable id of the chain set. Edge weights can change without a new chain."""
    idents = sorted("→".join(ch["chain"]) for ch in chains)
    return hashlib.sha1("\n".join(idents).encode()).hexdigest()


@router.post("/register-tools")
async def register_tools(req: RegisterToolsRequest, background_tasks: BackgroundTasks):
    """Register the agent's available tools/skills into mem:tools vectorset."""
    if not req.tools:
        return {"status": "ok", "registered": 0}

    async def _do_register():
        count = 0
        for tool in req.tools:
            try:
                emb = await async_encode(f"{tool.name}: {tool.description}")
                await cap_mod.register_tool(
                    state.redis,
                    name=tool.name,
                    description=tool.description,
                    embedding=emb,
                    category=tool.category,
                    source=tool.source,
                    parameters=tool.parameters or [],
                    agent_id=req.agent_id,
                )
                count += 1
            except Exception as e:
                log.warning("[tools] failed to register %s: %s", tool.name, e)
        log.info(f"[tools] registered {count} tools (agent={req.agent_id})")

    background_tasks.add_task(_do_register)
    return {"status": "queued", "tool_count": len(req.tools)}


@router.post("/register-env")
async def register_env(req: EnvState):
    """Store current environment state in mem:env hash."""
    r = state.redis
    env_data: dict = {}
    if req.os:             env_data["os"]             = req.os
    if req.os_version:     env_data["os_version"]     = req.os_version
    if req.shell:          env_data["shell"]           = req.shell
    if req.cwd:            env_data["cwd"]             = req.cwd
    if req.git_repo:       env_data["git_repo"]        = req.git_repo
    if req.git_branch:     env_data["git_branch"]      = req.git_branch
    if req.runtime:        env_data["runtime"]         = req.runtime
    if req.agent_model:    env_data["agent_model"]     = req.agent_model
    if req.agent_version:  env_data["agent_version"]   = req.agent_version
    if req.session_id:     env_data["session_id"]      = req.session_id
    if req.active_mcps:    env_data["active_mcps"]     = req.active_mcps
    if req.active_plugins: env_data["active_plugins"]  = req.active_plugins
    if req.active_skills:  env_data["active_skills"]   = req.active_skills
    if req.extra:
        for k, v in req.extra.items():
            env_data[k] = str(v)

    await cap_mod.set_env(r, env_data)

    agent_data: dict = {}
    if req.agent_model:   agent_data["model"]      = req.agent_model
    if req.agent_version: agent_data["version"]    = req.agent_version
    if req.session_id:    agent_data["session_id"] = req.session_id
    if req.runtime:       agent_data["runtime"]    = req.runtime
    if agent_data:
        await cap_mod.set_agent(r, agent_data)

    log.info(f"[env] registered env: {list(env_data.keys())}")
    return {"status": "ok", "fields": list(env_data.keys()), "agent_fields": list(agent_data.keys())}


@router.post("/recall-tools")
async def recall_tools_endpoint(req: RecallToolsRequest):
    """Semantic search over registered tools."""
    t0 = time.time()
    emb = encode(req.query)
    tools = await cap_mod.recall_tools(
        state.redis, emb,
        k=req.k,
        category_filter=req.category or None,
        source_filter=req.source or None,
    )
    ms = int((time.time() - t0) * 1000)
    return {"tools": tools, "count": len(tools), "latency_ms": ms}


@router.get("/capabilities")
async def get_capabilities():
    """Return the full agent capability manifest."""
    return await cap_mod.get_capability_summary(state.redis)


@router.post("/recall-procedures")
async def recall_procedures(req: RecallProceduresRequest):
    """Semantic search over procedural memory (4th cognitive tier)."""
    t0 = time.time()
    r = state.redis
    try:
        scope = strict_scope(req.user_id, req.agent_id)
    except BadScope as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    emb = encode(req.query)
    blob = emb.astype("float32").tobytes()
    cmd = ["VSIM", mem_store.PROC_KEY, "FP32", blob,
           "COUNT", req.k + 1, "WITHSCORES", "WITHATTRIBS"]
    if not scope.is_global:
        cmd += ["FILTER", scope.filter()]
    try:
        results = await r.execute_command(*cmd)
    except Exception:
        return {"procedures": [], "latency_ms": 0}

    procs = []
    i = 0
    while i + 2 < len(results):
        elem = results[i]; score = results[i+1]; raw = results[i+2]
        i += 3
        elem_str = decode_bytes(elem)
        if elem_str == "__seed__":
            continue
        try:
            attrs = decode_attrs(raw)
        except Exception:
            continue
        if attrs.get("_seed") or not attrs.get("task"):
            continue
        procs.append({
            "task":          attrs.get("task", ""),
            "procedure":     attrs.get("procedure", ""),
            "tools_used":    attrs.get("tools_used", []),
            "domain":        attrs.get("domain", ""),
            "success_count": attrs.get("success_count", 1),
            "score":         float(score) if score else 0.0,
        })

    ms = int((time.time() - t0) * 1000)
    return {"procedures": procs, "count": len(procs), "latency_ms": ms}


PROC_PENDING_KEY = "mem:proc_pending"   # durable parking lot for writes the embedder blocked
_DRAIN_BATCH = 25
# Every write outcome the endpoint can take is counted here, so "200 + nothing new stored" is
# always attributable to one of stored / deduped / deferred instead of being a mystery.
_proc_guard = {"deferred": 0, "errors": 0, "drained": 0, "stored": 0, "deduped": 0}
_queue_task = None                      # the replay supervisor's handle


def proc_guard_stats() -> dict:
    return dict(_proc_guard)


def _proc_payload(req: "StoreProcedureRequest") -> str:
    return json.dumps({"task": req.task, "procedure": req.procedure,
                       "tools_used": list(req.tools_used or []), "domain": req.domain,
                       "session_id": req.session_id,
                       "user_id": req.user_id, "agent_id": req.agent_id},
                      ensure_ascii=False)


async def _defer_proc(r, req, reason: str) -> int:
    """Park the write so an outage costs latency, never data."""
    n = int(await r.rpush(PROC_PENDING_KEY, _proc_payload(req)) or 0)
    _proc_guard["deferred"] += 1
    log.warning("[proc] deferred store (%s) task=%r pending=%d", reason, req.task[:60], n)
    return n


async def _store_proc_now(r, req, scope=None) -> str:
    """Embed + persist one procedure. Raises if the embedder is down — callers must handle it."""
    if scope is None:
        scope = strict_scope(req.user_id, req.agent_id)
    emb = encode(req.task)
    sc = scene_mod.detect(req.task)
    hit = await mem_store.find_procedure_by_task(r, emb, req.task, scope=scope)
    dedup_score = 1.0
    if not hit:
        near = await mem_store.nearest_procedures(
            r, emb, k=1, filter_expr=scope.filter() or None)
        if near and near[0]["score"] > 0.90:
            hit = near[0]
            dedup_score = near[0]["score"]
    if hit:
        elem = hit.get("_element")
        if elem:
            attrs = dict(hit.get("attrs") or {})
            attrs["success_count"] = attrs.get("success_count", 1) + 1
            try:
                await r.execute_command("VSETATTR", mem_store.PROC_KEY, elem, json.dumps(attrs))
            except Exception as exc:
                log.warning("[proc] dedup success_count bump failed for %s: %s", elem, exc)
        # Loud on purpose: an accepted write that adds no element must still leave a trace,
        # otherwise "already known" is indistinguishable from "silently lost".
        log.info("[proc] deduped: %r -> existing %s (score=%.3f, known_task=%r)",
                 req.task[:60], elem or "<no element>", dedup_score,
                 (hit.get("attrs") or {}).get("task", "")[:60])
        _proc_guard["deduped"] += 1
        return "deduped"
    await mem_store.save_procedure(
        r, task=req.task, procedure=req.procedure, embedding=emb,
        tools_used=req.tools_used, domain=req.domain or sc["domain"], language=sc["language"],
        user_id=scope.user_id, agent_id=scope.agent_id,
    )
    log.info("[proc] stored procedure: %s", req.task[:60])
    _proc_guard["stored"] += 1
    return "stored"


async def drain_proc_pending(limit: int = _DRAIN_BATCH) -> int:
    """Replay parked writes, oldest first, while the embedder is actually answering.

    Stops without popping on EmbeddingUnavailable so nothing is dropped and order is kept; a
    malformed record is discarded loudly rather than blocking the queue forever.
    """
    r = state.redis
    if r is None or embed_mod.embedding_unavailable():
        return 0
    drained = 0
    for _ in range(max(1, limit)):
        raw = await r.lindex(PROC_PENDING_KEY, 0)
        if raw is None:
            break
        try:
            data = json.loads(decode_bytes(raw))
            await _store_proc_now(r, StoreProcedureRequest(**data))
        except embed_mod.EmbeddingUnavailable:
            break
        except Exception as exc:
            await r.lpop(PROC_PENDING_KEY)
            _proc_guard["errors"] += 1
            log.error("[proc] dropped malformed pending record: %s", exc)
            continue
        await r.lpop(PROC_PENDING_KEY)
        drained += 1
    if drained:
        _proc_guard["drained"] += drained
        log.info("[proc] replayed %d deferred procedure(s) from %s", drained, PROC_PENDING_KEY)
    return drained


async def proc_queue_loop(interval_s: int = 60) -> None:
    """Safety net: a deferred write must not wait for the next store call to be replayed."""
    # Replay anything parked while the process was down, then keep watching.
    try:
        await asyncio.sleep(5)
        await drain_proc_pending(limit=200)
    except Exception as exc:
        log.warning("[proc] startup replay failed: %s", exc)
    while True:
        await asyncio.sleep(interval_s)
        try:
            if state.redis is not None and await state.redis.llen(PROC_PENDING_KEY):
                await drain_proc_pending()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("[proc] supervisor drain failed: %s", exc)


def start_proc_queue_supervisor() -> None:
    """Launch the replay loop as a private task.

    Deliberately not wired through main.py's startup block: this module is imported by the app
    factory, so a failure here must degrade to "no supervisor" instead of failing service boot.
    """
    global _queue_task
    if _queue_task is not None and not _queue_task.done():
        return
    try:
        loop = asyncio.get_running_loop()          # check BEFORE building the coroutine
    except RuntimeError as exc:                    # no loop: imported under tests / CLI use
        log.warning("[proc] supervisor not started: %s", exc)
        return
    _queue_task = loop.create_task(proc_queue_loop(), name="proc-queue-drain")
    log.info("[proc] deferred-write supervisor started (pending key=%s)", PROC_PENDING_KEY)


try:
    start_proc_queue_supervisor()
except Exception as _exc:                # never let this block break app import
    log.warning("[proc] supervisor bootstrap failed: %s", _exc)


@router.post("/store-procedure")
async def store_procedure(req: StoreProcedureRequest, background_tasks: BackgroundTasks):
    """Store a procedural memory (agent workflow / how-to pattern).

    ``status`` is honest about what will happen to the write:
      queued    — embedder is answering; the background task will persist it
      deferred  — embedder circuit is open; the write is parked in Redis and will be replayed
    Both are durable. The pre-2026-10-24 behaviour was neither: the background task's very first
    statement was ``encode(task)``, which raises ``EmbeddingUnavailable`` whenever the circuit is
    open, so the caller received 200 ``queued`` and the procedure silently never existed.
    """
    r = state.redis
    try:
        scope = strict_scope(req.user_id, req.agent_id)
    except BadScope as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if embed_mod.embedding_unavailable():
        n = await _defer_proc(r, req, "embedding circuit open")
        return {"status": "deferred", "pending": n, "reason": "embedding circuit open"}

    async def _do_store_proc():
        try:
            await _store_proc_now(r, req, scope)
        except Exception as exc:
            # Loud, and never at the cost of the write.
            log.error("[proc] store failed (%s) — parking it instead of dropping it",
                      exc, exc_info=True)
            try:
                await _defer_proc(r, req, f"{type(exc).__name__}: {str(exc)[:120]}")
            except Exception as exc2:
                _proc_guard["errors"] += 1
                log.error("[proc] UNRECOVERABLE: store failed AND defer failed: %s", exc2)
            return
        try:
            await drain_proc_pending()
        except Exception as exc:
            log.warning("[proc] inline drain failed: %s", exc)

    background_tasks.add_task(_do_store_proc)
    try:
        pending = int(await r.llen(PROC_PENDING_KEY) or 0)
    except Exception:
        pending = -1
    return {"status": "queued", "pending": pending}


@router.post("/tool-feedback")
async def tool_feedback(req: ToolFeedbackRequest):
    """Record success/failure for a tool invocation (ToolMem)."""
    r = state.redis
    elem_key = req.tool_name.lower().replace(" ", "_").replace("/", "_")[:64]

    attrs = await cap_mod.atomic_tool_feedback(r, elem_key, req.success)
    observed = False
    if not attrs:
        # Hooks report every tool Claude actually ran. Most 404s were tools
        # outside the static builtin list, so ToolMem never learned them.
        observed = True
        try:
            await cap_mod.register_observed_tool(
                r, name=req.tool_name, agent_id=req.session_id,
            )
            attrs = await cap_mod.atomic_tool_feedback(r, elem_key, req.success)
        except Exception as e:
            log.warning("[tool-feedback] could not observe %s: %s", elem_key, e)
            attrs = None
    if not attrs:
        raise HTTPException(status_code=404, detail="tool not found")

    log.debug(f"[tool-feedback] {elem_key} success={req.success} "
              f"s={attrs.get('success_count',0)} f={attrs.get('fail_count',0)}")
    return {
        "ok":                True,
        "tool":              elem_key,
        "observed":          observed,
        "success_count":     attrs.get("success_count", 0),
        "fail_count":        attrs.get("fail_count", 0),
        "capability_summary": attrs.get("capability_summary", ""),
    }


@router.post("/record-tool-sequence")
async def record_tool_sequence(req: ToolSequenceRequest):
    """Record an ordered tool-use sequence into the Tool Inertia Graph."""
    r = state.redis
    seq = [t.strip() for t in req.sequence if t.strip()]
    if len(seq) < 2:
        return {"ok": False, "transitions": 0}
    count = 0
    for i in range(len(seq) - 1):
        a = seq[i].lower().replace(" ", "_")[:64]
        b = seq[i + 1].lower().replace(" ", "_")[:64]
        if a == b:
            continue
        await r.hincrby(mem_store.TOOL_GRAPH_KEY, f"{a}:{b}", 1)
        count += 1
    log.debug(f"[tig] recorded {count} transitions for session={req.session_id}")
    return {"ok": True, "transitions": count}


@router.get("/tool-graph/{tool_name}")
async def tool_graph(tool_name: str, k: int = 5):
    """Return top-k outgoing transitions from tool_name in the TIG."""
    k = max(1, min(k, 50))
    r = state.redis
    elem_key = tool_name.lower().replace(" ", "_")[:64]
    try:
        all_entries = await r.hgetall(mem_store.TOOL_GRAPH_KEY)
    except Exception:
        return {"tool": tool_name, "transitions": []}
    prefix = f"{elem_key}:"
    trans: dict[str, int] = {}
    for k_b, v_b in all_entries.items():
        key_s = decode_bytes(k_b)
        if key_s.startswith(prefix):
            target = key_s[len(prefix):]
            trans[target] = int(v_b)
    total = sum(trans.values())
    sorted_t = sorted(trans.items(), key=lambda x: x[1], reverse=True)[:k]
    return {
        "tool":        tool_name,
        "transitions": [
            {"next": t, "count": c, "prob": round(c / total, 2) if total else 0}
            for t, c in sorted_t
        ],
        "total_transitions": total,
    }


@router.post("/tool-graph/detect-meta-tools")
async def detect_meta_tools(threshold: int = 5, background_tasks: BackgroundTasks = None):
    """AWO meta-tool detection — find high-frequency 2-hop chains in TIG."""
    r = state.redis
    try:
        all_entries = await r.hgetall(mem_store.TOOL_GRAPH_KEY)
    except Exception:
        return {"detected": 0, "chains": [], "new_procedures": 0}

    if not all_entries:
        return {"detected": 0, "chains": [], "new_procedures": 0}

    adj: dict[str, dict[str, int]] = {}
    for k_b, v_b in all_entries.items():
        key_s = decode_bytes(k_b)
        val = int(v_b) if v_b else 0
        if ":" not in key_s:
            continue
        a, b = key_s.split(":", 1)
        adj.setdefault(a, {})[b] = val

    chains: list[dict] = []
    for a, a_neighbors in adj.items():
        for b, ab_count in a_neighbors.items():
            if ab_count < threshold:
                continue
            b_neighbors = adj.get(b, {})
            for c, bc_count in b_neighbors.items():
                if bc_count < threshold or c == a:
                    continue
                chains.append({
                    "chain":    [a, b, c],
                    "ab_count": ab_count,
                    "bc_count": bc_count,
                    "strength": min(ab_count, bc_count),
                })

    chains.sort(key=lambda x: x["strength"], reverse=True)

    if not chains:
        return {"detected": 0, "chains": [], "new_procedures": 0}

    sig = _chain_signature(chains)
    prev = await r.get(AWO_SIG_KEY)
    if decode_bytes(prev) == sig:
        return {
            "detected": len(chains),
            "chains": chains[:10],
            "new_procedures": 0,
            "synthesis": "skipped",
            "reason": "unchanged",
            "threshold": threshold,
        }

    async def _synthesize_chains(chains_to_add: list[dict]):
        new_count = 0
        embed_failures = 0
        for ch in chains_to_add[:20]:
            a, b, c = ch["chain"]
            task = f"[meta-tool] {a} → {b} → {c} workflow"
            procedure = (
                f"High-frequency tool chain discovered by AWO analysis "
                f"(TIG count: {a}→{b}={ch['ab_count']}, {b}→{c}={ch['bc_count']}).\n"
                f"1. Use {a}\n2. Use {b}\n3. Use {c}"
            )
            mapped = await r.hget(mem_store.PROC_TASK_KEY, task[:300])
            if mapped:
                continue
            try:
                emb = encode(task)
            except Exception as e:
                embed_failures += 1
                log.warning("[awo] embed failed for %s: %s", a, e)
                continue
            if await mem_store.find_procedure_by_task(r, emb, task):
                continue
            await mem_store.save_procedure(
                r, task=task, procedure=procedure, embedding=emb,
                tools_used=[a, b, c], domain="meta", language="en",
            )
            new_count += 1
        if embed_failures == 0:
            try:
                await r.set(AWO_SIG_KEY, sig)
            except Exception:
                pass
        if new_count:
            log.info(
                "[awo] synthesized %d new meta-tool procedures from %d chains",
                new_count, len(chains_to_add),
            )
        else:
            log.info(
                "[awo] no new meta-tool procedures (%d chains, embed_failures=%d)",
                len(chains_to_add), embed_failures,
            )

    if background_tasks is not None:
        background_tasks.add_task(_synthesize_chains, chains)
        synthesis_status = "queued"
        new_procs = None
    else:
        await _synthesize_chains(chains)
        synthesis_status = "done"
        new_procs = min(len(chains), 20)

    return {
        "detected":       len(chains),
        "chains":         chains[:10],
        "new_procedures": new_procs,
        "synthesis":      synthesis_status,
        "threshold":      threshold,
    }


@router.post("/procedure-feedback")
async def procedure_feedback(req: ProcedureFeedbackRequest):
    """Record success/failure for a procedure retrieval (MACLA)."""
    r = state.redis
    emb = encode(req.task_prefix[:80])
    blob = emb.astype("float32").tobytes()
    try:
        results = await r.execute_command(
            "VSIM", mem_store.PROC_KEY, "FP32", blob,
            "COUNT", 3, "WITHSCORES", "WITHATTRIBS"
        )
    except Exception:
        raise HTTPException(status_code=503, detail="redis error")

    best_elem = None
    best_score = 0.0
    best_attrs: dict = {}
    idx = 0
    while idx + 2 < len(results):
        elem = results[idx]; score = results[idx+1]; raw = results[idx+2]
        idx += 3
        elem_str = decode_bytes(elem)
        if elem_str == "__seed__":
            continue
        try:
            attrs = decode_attrs(raw)
        except Exception:
            continue
        if attrs.get("_seed") or not attrs.get("task"):
            continue
        s = float(score) if score else 0.0
        if s > best_score:
            best_score = s; best_elem = elem_str; best_attrs = attrs

    if not best_elem or best_score < 0.50:
        raise HTTPException(status_code=404, detail="no matching procedure found")

    if req.success:
        best_attrs["success_count"] = best_attrs.get("success_count", 0) + 1
    else:
        best_attrs["fail_count"] = best_attrs.get("fail_count", 0) + 1

    try:
        await r.execute_command("VSETATTR", mem_store.PROC_KEY, best_elem, json.dumps(best_attrs))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    log.debug(f"[proc-feedback] {best_elem[:40]} success={req.success} score={best_score:.2f}")
    return {
        "ok":            True,
        "matched_task":  best_attrs.get("task", "")[:60],
        "score":         best_score,
        "success_count": best_attrs.get("success_count", 0),
        "fail_count":    best_attrs.get("fail_count", 0),
    }


@router.get("/tool-procedures/{tool_name}")
async def tool_procedures(tool_name: str):
    """Return all procedures that use a given tool (reverse index lookup)."""
    r = state.redis
    uids = await mem_store.get_procs_for_tool(r, tool_name)
    if not uids:
        return {"tool": tool_name, "procedures": [], "count": 0}

    procs = []
    for uid in uids:
        try:
            raw = await r.execute_command("VGETATTR", mem_store.PROC_KEY, uid)
        except Exception:
            continue
        if not raw:
            continue
        try:
            attrs = decode_attrs(raw)
        except Exception:
            continue
        if not attrs.get("task"):
            continue
        procs.append({
            "uid":           uid,
            "task":          attrs.get("task", ""),
            "procedure":     attrs.get("procedure", "")[:200],
            "tools_used":    attrs.get("tools_used", []),
            "domain":        attrs.get("domain", ""),
            "success_count": attrs.get("success_count", 1),
            "fail_count":    attrs.get("fail_count", 0),
        })

    return {"tool": tool_name, "procedures": procs, "count": len(procs)}


@router.post("/proc-backfill-index")
async def proc_backfill_index(background_tasks: BackgroundTasks):
    """Backfill mem:proc_by_tool reverse index for all existing procedures."""
    r = state.redis
    async def _do_backfill():
        procs = await mem_store.scan_all_procedures(r)
        count = 0
        for p in procs:
            uid = p.get("uid", "")
            tools = p.get("tools_used", [])
            if uid and tools:
                await mem_store.link_proc_to_tools(r, uid, tools)
                count += len(tools)
        log.info(f"[backfill] reverse-indexed {len(procs)} procedures, {count} tool→proc links")
        return len(procs), count

    background_tasks.add_task(_do_backfill)
    return {"status": "queued", "message": "backfilling proc_by_tool reverse index in background"}


@router.get("/capabilities/context")
async def get_capability_context():
    """Return environment context as formatted string."""
    r = state.redis
    env_ctx = await cap_mod.get_env_context(r)
    persona = await persona_mod.get_context(r)
    agent = await cap_mod.get_agent(r)
    all_tools = await cap_mod.list_all_tools(r)

    by_cat: dict[str, list] = {}
    for t in all_tools:
        cat = t.get("category", "general")
        by_cat.setdefault(cat, []).append(t["name"])

    tool_lines = []
    for cat, names in sorted(by_cat.items()):
        tool_lines.append(f"  [{cat}]: {', '.join(names)}")

    return {
        "persona_context":     persona,
        "env_context":         env_ctx,
        "agent_self_model":    agent,
        "tool_index_by_category": by_cat,
        "tool_count":          len(all_tools),
    }
