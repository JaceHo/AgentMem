"""Capability routes — tools, environment, procedures, TIG, AWO, MACLA."""

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException

from api import state
from api.schemas.capability import (
    RegisterToolsRequest, EnvState, RecallToolsRequest,
    StoreProcedureRequest, ToolFeedbackRequest, ToolSequenceRequest,
    ProcedureFeedbackRequest,
)
from core import capability as cap_mod
from core import persona as persona_mod
from core import scene as scene_mod
from core import store as mem_store
from core.search import encode
from core.utils import decode_bytes

log = logging.getLogger("mem")
router = APIRouter(tags=["capability"])


@router.post("/register-tools")
async def register_tools(req: RegisterToolsRequest, background_tasks: BackgroundTasks):
    """Register the agent's available tools/skills into mem:tools vectorset."""
    if not req.tools:
        return {"status": "ok", "registered": 0}

    async def _do_register():
        count = 0
        for tool in req.tools:
            emb = encode(f"{tool.name}: {tool.description}")
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
async def recall_procedures(req: RecallToolsRequest):
    """Semantic search over procedural memory (4th cognitive tier)."""
    t0 = time.time()
    r = state.redis
    emb = encode(req.query)
    blob = emb.astype("float32").tobytes()
    cmd = ["VSIM", mem_store.PROC_KEY, "FP32", blob,
           "COUNT", req.k + 1, "WITHSCORES", "WITHATTRIBS"]
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


@router.post("/store-procedure")
async def store_procedure(req: StoreProcedureRequest, background_tasks: BackgroundTasks):
    """Store a procedural memory (agent workflow / how-to pattern)."""
    r = state.redis
    async def _do_store_proc():
        emb = encode(req.task)
        sc = scene_mod.detect(req.task)
        existing = await mem_store.knn_search(r, mem_store.PROC_KEY, emb, k=1)
        if existing and existing[0].get("score", 0.0) > 0.90:
            elem = existing[0].get("_element")
            if elem:
                attrs = dict(existing[0].get("attrs", {}))
                attrs["success_count"] = attrs.get("success_count", 1) + 1
                try:
                    await r.execute_command("VSETATTR", mem_store.PROC_KEY, elem, json.dumps(attrs))
                except Exception:
                    pass
            return
        await mem_store.save_procedure(
            r,
            task=req.task,
            procedure=req.procedure,
            embedding=emb,
            tools_used=req.tools_used,
            domain=req.domain or sc["domain"],
            language=sc["language"],
        )
        log.info(f"[proc] stored procedure: {req.task[:60]}")

    background_tasks.add_task(_do_store_proc)
    return {"status": "queued"}


@router.post("/tool-feedback")
async def tool_feedback(req: ToolFeedbackRequest):
    """Record success/failure for a tool invocation (ToolMem)."""
    r = state.redis
    elem_key = req.tool_name.lower().replace(" ", "_").replace("/", "_")[:64]

    attrs = await cap_mod.atomic_tool_feedback(r, elem_key, req.success)
    if not attrs:
        raise HTTPException(status_code=404, detail="tool not found")

    log.debug(f"[tool-feedback] {elem_key} success={req.success} "
              f"s={attrs.get('success_count',0)} f={attrs.get('fail_count',0)}")
    return {
        "ok":                True,
        "tool":              elem_key,
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

    async def _synthesize_chains(chains_to_add: list[dict]):
        new_count = 0
        for ch in chains_to_add[:20]:
            a, b, c = ch["chain"]
            task = f"[meta-tool] {a} → {b} → {c} workflow"
            procedure = (
                f"High-frequency tool chain discovered by AWO analysis "
                f"(TIG count: {a}→{b}={ch['ab_count']}, {b}→{c}={ch['bc_count']}).\n"
                f"1. Use {a}\n2. Use {b}\n3. Use {c}"
            )
            emb = encode(task)
            existing = await mem_store.knn_search(r, mem_store.PROC_KEY, emb, k=1)
            if existing and existing[0].get("score", 0.0) > 0.90:
                continue
            await mem_store.save_procedure(
                r, task=task, procedure=procedure, embedding=emb,
                tools_used=[a, b, c], domain="meta", language="en",
            )
            new_count += 1
        log.info(f"[awo] synthesized {new_count} meta-tool procedures from {len(chains_to_add)} chains")

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
