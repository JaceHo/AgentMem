"""
OpenClaw Local Memory Service  v0.4.0
======================================
Replaces memos-cloud-openclaw-plugin with fully local, persistent memory.

Features:
  1. Heat-tiered recall  — frequently/recently accessed memories rank higher
  2. Scene isolation     — language + domain tags filter recall by context
  3. Evolving persona    — structured user profile updated from extracted facts
  4. Summarize-then-embed— long turns compressed before MiniLM embedding
  5. Hybrid extraction   — regex (instant) + LLM (GLM-4-flash, async) for facts

Endpoints:
  POST /recall   — before_agent_start hook
  POST /store    — agent_end hook (async, non-blocking)
  GET  /health
  GET  /stats
"""

import asyncio
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any

import httpx
import numpy as np
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

import embedder
import extractor
import heat as heat_mod
import persona as persona_mod
import scene as scene_mod
import store as mem_store
import summarizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [mem] %(message)s")
log = logging.getLogger("mem")

_redis = None

# ── System-injected content filter ────────────────────────────────────────────
_SKIP_PREFIXES = (
    "## Long-Term Memory",
    "## Recent Relevant Episodes",
    "## Current Session Context",
    "## User Profile",
    "[cron:",
)

def _is_injected(text: str) -> bool:
    t = text.strip()
    return any(t.startswith(p) for p in _SKIP_PREFIXES)


# ── Trivial message filter ─────────────────────────────────────────────────────
_TRIVIAL_MIN_CHARS = 15  # user text shorter than this → skip episode store
_TRIVIAL_PATTERNS = re.compile(
    r"^(hi+|hello+|hey+|yo+|ok+|okay|sure|thanks?|thx|bye+|good\s*(morning|night|day)|"
    r"好的|嗯|哦|呵|哈|谢谢|再见|早|晚安|你好|您好|ji+|test+|ping)[\s!?.。！？]*$",
    re.IGNORECASE,
)

def _is_trivial(user_text: str) -> bool:
    t = user_text.strip()
    if len(t) < _TRIVIAL_MIN_CHARS:
        return True
    return bool(_TRIVIAL_PATTERNS.match(t))


# ── Embedding cache ────────────────────────────────────────────────────────────
@lru_cache(maxsize=1024)
def _cached_encode(text: str) -> bytes:
    return embedder.encode(text).tobytes()

def _encode(text: str) -> np.ndarray:
    return np.frombuffer(_cached_encode(text), dtype=np.float32).copy()


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _redis
    log.info("Loading embedding model…")
    embedder.get_model()
    log.info("Connecting to Redis…")
    _redis = await mem_store.get_client()
    log.info("Ensuring vectorset indexes…")
    await mem_store.ensure_indexes(_redis)
    log.info("Memory service v0.4.0 ready")
    yield
    await _redis.aclose()


app = FastAPI(title="OpenClaw Memory Service", version="0.4.0", lifespan=lifespan)


# ── Request / Response models ──────────────────────────────────────────────────
class RecallRequest(BaseModel):
    query: str
    session_id: str = ""
    memory_limit_number: int = 6

class Message(BaseModel):
    role: str
    content: Any

class StoreRequest(BaseModel):
    messages: list[Message]
    session_id: str = ""


# ── Helpers ────────────────────────────────────────────────────────────────────
def _msg_text(m: Message) -> str:
    if isinstance(m.content, str):
        return m.content.strip()
    if isinstance(m.content, list):
        return " ".join(b.get("text","") for b in m.content if isinstance(b, dict)).strip()
    return ""

def _messages_to_text(messages: list[Message]) -> str:
    return "\n".join(
        f"{m.role}: {_msg_text(m)}" for m in messages if _msg_text(m)
    )

def _format_prepend(
    facts: list[dict],
    episodes: list[dict],
    session_ctx: str | None,
    persona_ctx: str,
) -> str:
    lines: list[str] = []

    if persona_ctx:
        lines += [persona_ctx, ""]

    if session_ctx:
        lines += ["## Current Session Context", session_ctx, ""]

    if facts:
        lines.append("## Long-Term Memory (Facts)")
        for i, f in enumerate(facts, 1):
            tag = f.get("category") or f.get("domain") or "?"
            attrs = f.get("attrs", {})
            meta_parts = []
            if p := attrs.get("persons"):
                meta_parts.append(f"who:{','.join(p)}")
            if e := attrs.get("entities"):
                meta_parts.append(f"re:{','.join(e)}")
            meta = f" ({'; '.join(meta_parts)})" if meta_parts else ""
            lines.append(f"{i}. [{tag}]{meta} {f['content']}")
        lines.append("")

    if episodes:
        lines.append("## Recent Relevant Episodes")
        for i, e in enumerate(episodes, 1):
            lines.append(f"{i}. {e['content'][:300]}")
        lines.append("")

    return "\n".join(lines).strip()


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


# ── Background store ──────────────────────────────────────────────────────────
async def _do_store(messages: list[Message], session_id: str) -> None:
    t0 = time.time()
    try:
        # 1. Filter system-injected and empty messages
        clean = [m for m in messages if _msg_text(m) and not _is_injected(_msg_text(m))]
        if not clean:
            return
        clean = clean[-4:]   # last 2 real turns

        # 2. Detect scene from user messages
        user_text_raw = " ".join(_msg_text(m) for m in clean if m.role == "user")
        if not user_text_raw:
            return

        # 2b. Skip trivial exchanges (greetings, single words) — prevents feedback loops
        if _is_trivial(user_text_raw):
            log.info(f"[store] skipped trivial exchange: {user_text_raw[:40]!r}")
            return

        sc = scene_mod.detect(user_text_raw)
        lang, domain = sc["language"], sc["domain"]

        # 3. Summarize long turn before embedding
        turn_text = _messages_to_text(clean)
        summary = await summarizer.summarize(turn_text)

        # 4. Embed + dedup-check + save episode
        ep_emb = _encode(summary[:500])
        similar = await mem_store.knn_search(_redis, mem_store.EPISODE_KEY, ep_emb, k=1)
        # VSIM returns similarity (1=identical). Dedup if score > 0.95.
        is_dup = bool(similar and similar[0].get("score", 0.0) > 0.95)
        ep_saved = 0
        if not is_dup:
            await mem_store.save_episode(
                _redis, session_id, turn_text[:2000], ep_emb, lang, domain
            )
            ep_saved = 1

        # 5. Hybrid fact extraction (regex + LLM) → dedup → save → update persona
        raw_msgs = [m.model_dump() for m in clean if m.role == "user"]
        facts = await extractor.extract_hybrid(raw_msgs, turn_text)
        fact_saved = 0
        for fact in facts:
            f_emb = _encode(fact.content)
            existing = await mem_store.knn_search(_redis, mem_store.FACT_KEY, f_emb, k=1)
            if existing and existing[0].get("score", 0.0) > 0.95:
                continue
            await mem_store.save_fact(
                _redis, fact.content, fact.category, fact.confidence, f_emb, lang, domain,
                keywords=fact.keywords, persons=fact.persons, entities=fact.entities,
            )
            await persona_mod.update(_redis, fact.category, fact.content)
            fact_saved += 1

        # 6. Update session working memory
        await mem_store.set_session_context(_redis, session_id, user_text_raw[:400])

        ms = int((time.time() - t0) * 1000)
        log.info(f"[store] session={session_id} lang={lang} domain={domain} "
                 f"ep+{ep_saved} facts+{fact_saved} {ms}ms")

    except Exception as e:
        log.warning(f"[store] background error: {e}", exc_info=True)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    try:
        await _redis.ping()
        return {"status": "ok", "redis": "ok", "version": "0.4.0"}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.get("/stats")
async def stats():
    try:
        counts = await mem_store.vcard(_redis)
        persona_raw = await _redis.hgetall(mem_store.PERSONA_KEY
                                           if hasattr(mem_store, "PERSONA_KEY")
                                           else "mem:persona")
        counts["persona_fields"] = len(persona_raw)
        return counts
    except Exception as e:
        return {"error": str(e)}


@app.post("/recall")
async def recall(req: RecallRequest):
    t0 = time.time()

    # Detect scene of incoming query
    query_scene = scene_mod.detect(req.query)
    q_lang, q_domain = query_scene["language"], query_scene["domain"]
    emb = _encode(req.query)

    # Adaptive retrieval depth (SimpleMem-inspired C_q estimator)
    n_facts, n_episodes = _estimate_depth(req.query, req.memory_limit_number)

    # Build filter expressions for scene isolation
    lang_filter   = f'.language == "{q_lang}"'
    domain_filter = f'.domain == "{q_domain}"' if q_domain != "general" else None

    # Recall strategy: scene-filtered first, supplement with global if sparse
    async def _fetch(vset: str, k: int, fexpr=None):
        return await mem_store.knn_search(
            _redis, vset, emb, k, filter_expr=fexpr, bump_heat=True
        )

    # Parallel: persona + session ctx + scene-filtered facts + scene-filtered episodes
    persona_ctx, session_ctx, facts_scene, eps_scene = await asyncio.gather(
        persona_mod.get_context(_redis),
        mem_store.get_session_context(_redis, req.session_id),
        _fetch(mem_store.FACT_KEY,    n_facts,    lang_filter),
        _fetch(mem_store.EPISODE_KEY, n_episodes, lang_filter),
    )

    # Supplement with global search if scene results are sparse
    supplement_tasks = []
    if len(facts_scene) < n_facts:
        supplement_tasks.append(_fetch(mem_store.FACT_KEY, n_facts - len(facts_scene)))
    else:
        supplement_tasks.append(asyncio.sleep(0))

    if len(eps_scene) < max(2, n_episodes // 2):
        supplement_tasks.append(
            _fetch(mem_store.EPISODE_KEY, n_episodes - len(eps_scene))
        )
    else:
        supplement_tasks.append(asyncio.sleep(0))

    supp_results = await asyncio.gather(*supplement_tasks)
    facts_global   = supp_results[0] if isinstance(supp_results[0], list) else []
    eps_global     = supp_results[1] if isinstance(supp_results[1], list) else []

    # Merge, dedup by content, heat-rerank
    def _merge(primary, supplement, limit):
        seen = {i["content"] for i in primary}
        merged = list(primary) + [i for i in supplement if i["content"] not in seen]
        ranked = heat_mod.heat_rerank(merged)
        return ranked[:limit]

    facts    = _merge(facts_scene,   facts_global,  n_facts)
    episodes = _merge(eps_scene,     eps_global,    n_episodes)

    prepend = _format_prepend(facts, episodes, session_ctx, persona_ctx)
    ms = int((time.time() - t0) * 1000)

    log.info(f"[recall] session={req.session_id} lang={q_lang} domain={q_domain} "
             f"facts={len(facts)} ep={len(episodes)} {ms}ms")

    return {"prependContext": prepend if prepend else None, "latency_ms": ms}


@app.post("/store")
async def store_memory(req: StoreRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(_do_store, req.messages, req.session_id)
    return {"status": "queued"}


# ── Consolidation (SimpleMem-inspired recursive merging) ─────────────────────

async def _do_consolidate(similarity_threshold: float = 0.85) -> dict:
    """
    Find clusters of similar facts and merge them into consolidated entries.
    Uses VSIM to find near-duplicates, then LLM to produce a merged fact.
    """
    t0 = time.time()
    all_facts = []

    # Fetch all fact elements with their attributes
    card = await _redis.execute_command("VCARD", mem_store.FACT_KEY)
    if not card or int(card) <= 1:
        return {"merged": 0, "removed": 0, "ms": 0}

    # Get all elements via VRANDMEMBER (Redis 8 vectorset supports this)
    try:
        # Use a large VSIM with a zero vector to get all entries
        # (Not ideal but Redis 8 vectorset has no SCAN equivalent)
        seed = np.zeros(mem_store.DIMS, dtype=np.float32)
        results = await _redis.execute_command(
            "VSIM", mem_store.FACT_KEY, "FP32", seed.tobytes(),
            "COUNT", min(int(card), 500), "WITHSCORES", "WITHATTRIBS"
        )
    except Exception as e:
        log.warning(f"[consolidate] failed to scan facts: {e}")
        return {"error": str(e)}

    # Parse all facts
    i = 0
    while i + 2 < len(results):
        elem = results[i]
        score = results[i + 1]
        raw = results[i + 2]
        i += 3
        elem_str = elem.decode() if isinstance(elem, bytes) else elem
        if elem_str == "__seed__":
            continue
        try:
            attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        except Exception:
            continue
        if attrs.get("_seed") or not attrs.get("content"):
            continue
        all_facts.append({"element": elem_str, "attrs": attrs})

    if len(all_facts) < 2:
        return {"merged": 0, "removed": 0, "total": len(all_facts), "ms": 0}

    # Find clusters: for each fact, find similar ones
    merged_count = 0
    removed_elements = set()

    for idx, fact in enumerate(all_facts):
        if fact["element"] in removed_elements:
            continue

        f_emb = _encode(fact["attrs"]["content"])
        similar = await mem_store.knn_search(
            _redis, mem_store.FACT_KEY, f_emb, k=5
        )

        # Find cluster members (similar facts above threshold)
        cluster = [fact]
        for s in similar:
            if s["_element"] == fact["element"]:
                continue
            if s["_element"] in removed_elements:
                continue
            if s.get("score", 0) >= similarity_threshold:
                # Find the full fact entry
                for f2 in all_facts:
                    if f2["element"] == s["_element"]:
                        cluster.append(f2)
                        break

        if len(cluster) < 2:
            continue

        # Merge cluster into a single consolidated fact via LLM
        contents = [c["attrs"]["content"] for c in cluster]
        merged_content = await _llm_merge_facts(contents)
        if not merged_content:
            continue

        # Keep the newest entry, update its content, remove the rest
        cluster.sort(key=lambda c: c["attrs"].get("ts", 0), reverse=True)
        keeper = cluster[0]

        # Update keeper with merged content
        new_attrs = dict(keeper["attrs"])
        new_attrs["content"] = merged_content[:500]
        new_attrs["access_count"] = max(
            c["attrs"].get("access_count", 0) for c in cluster
        )
        new_attrs["consolidated_from"] = len(cluster)

        # Collect keywords/persons/entities from all cluster members
        all_kw = set()
        all_persons = set()
        all_entities = set()
        for c in cluster:
            for kw in c["attrs"].get("keywords", []):
                all_kw.add(kw)
            for p in c["attrs"].get("persons", []):
                all_persons.add(p)
            for e in c["attrs"].get("entities", []):
                all_entities.add(e)
        if all_kw:
            new_attrs["keywords"] = list(all_kw)[:10]
        if all_persons:
            new_attrs["persons"] = list(all_persons)[:5]
        if all_entities:
            new_attrs["entities"] = list(all_entities)[:5]

        # Re-embed the merged content and replace keeper
        m_emb = _encode(merged_content)
        try:
            await _redis.execute_command(
                "VADD", mem_store.FACT_KEY, "FP32", m_emb.tobytes(),
                keeper["element"], "SETATTR", json.dumps(new_attrs)
            )
        except Exception as e:
            log.warning(f"[consolidate] failed to update keeper: {e}")
            continue

        # Remove other cluster members
        for c in cluster[1:]:
            try:
                await _redis.execute_command("VREM", mem_store.FACT_KEY, c["element"])
                removed_elements.add(c["element"])
            except Exception:
                pass

        merged_count += 1

    ms = int((time.time() - t0) * 1000)
    log.info(f"[consolidate] merged={merged_count} removed={len(removed_elements)} {ms}ms")
    return {
        "merged": merged_count,
        "removed": len(removed_elements),
        "total_before": len(all_facts),
        "total_after": len(all_facts) - len(removed_elements),
        "ms": ms,
    }


async def _llm_merge_facts(contents: list[str]) -> str | None:
    """Use GLM-4-flash to merge multiple similar facts into one consolidated fact."""
    key = extractor._load_key()
    if not key:
        # Fallback: just pick the longest fact
        return max(contents, key=len)

    facts_text = "\n".join(f"- {c}" for c in contents)
    prompt = (
        f"以下是关于同一主题的多条记忆，请合并为一条完整、准确的事实陈述。"
        f"保留所有不同的细节，去除重复。输出仅包含合并后的一句话，不要解释。\n\n{facts_text}"
    )

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                extractor.ZAI_URL,
                json={
                    "model": "glm-4-flash",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 150,
                    "temperature": 0.1,
                },
                headers={"Authorization": f"Bearer {key}"},
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log.warning(f"[consolidate] LLM merge failed: {e}")

    return max(contents, key=len)


@app.post("/consolidate")
async def consolidate(background_tasks: BackgroundTasks):
    """Trigger memory consolidation — merges similar facts to reduce redundancy."""
    background_tasks.add_task(_do_consolidate)
    return {"status": "consolidation_queued"}


@app.post("/consolidate/sync")
async def consolidate_sync():
    """Synchronous consolidation — returns results immediately."""
    result = await _do_consolidate()
    return result


@app.post("/admin/delete-facts-by-content")
async def admin_delete_facts_by_content(pattern: str = "Always send a report"):
    """Delete facts whose content contains the given pattern (admin only)."""
    card = await _redis.execute_command("VCARD", mem_store.FACT_KEY)
    if not card or int(card) <= 1:
        return {"deleted": 0, "message": "no facts to scan"}

    seed = np.zeros(mem_store.DIMS, dtype=np.float32)
    results = await _redis.execute_command(
        "VSIM", mem_store.FACT_KEY, "FP32", seed.tobytes(),
        "COUNT", min(int(card), 500), "WITHSCORES", "WITHATTRIBS"
    )
    deleted = 0
    i = 0
    while i + 2 < len(results):
        elem = results[i]
        raw = results[i + 2]
        i += 3
        elem_str = elem.decode() if isinstance(elem, bytes) else elem
        if elem_str == "__seed__":
            continue
        try:
            attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        except Exception:
            continue
        content = attrs.get("content", "")
        if pattern.lower() in content.lower():
            await _redis.execute_command("VREM", mem_store.FACT_KEY, elem_str)
            deleted += 1
    return {"deleted": deleted, "pattern": pattern}
