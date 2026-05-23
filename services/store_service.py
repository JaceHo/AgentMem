"""
Store Service — Core memory storage, recall, and session management.

Extracted from main.py to break the circular dependency where route modules
needed to `from main import _do_store, _format_prepend, ...`.

All functions here use api.state for Redis, embedder, BM25, and counters
instead of main.py globals.
"""

import asyncio
import json
import logging
import os
import re
import time

import httpx
import numpy as np

from api import state
from api.schemas.memory import Message
from config.settings import settings
from core import embedder
from core import extractor
from core import graph as graph_mod
from core import heat as heat_mod
from core import persona as persona_mod
from core import scene as scene_mod
from core import store as mem_store
from core import summarizer
from core.search import encode, vscan
from utils.text_processing import (
    flatten_message_content,
    estimate_tokens,
    redact_secrets,
    contains_secret,
    strip_platform_noise,
    is_only_platform_noise,
    is_brief_option_reply,
    is_trivial,
    is_injected_system_content,
)

log = logging.getLogger("mem")


# ── Helpers ────────────────────────────────────────────────────────────────────

def msg_text(m: Message) -> str:
    raw = flatten_message_content(m.content)
    return strip_platform_noise(raw)


def messages_to_text(messages: list[Message]) -> str:
    return "\n".join(
        f"{m.role}: {msg_text(m)}" for m in messages if msg_text(m)
    )


# ── Format prepend (token-budget context injection) ───────────────────────────

def format_prepend(
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
) -> str:
    """
    Build the context string injected into the agent's system prompt.

    SimpleMem Section 3.3 token-budget packing:
    Priority order: persona → last_session_summary → crystallized digests → env → tools → skills → session_ctx → facts → episodes
    Greedy packing: stop adding items when token_budget is exhausted.
    Output wrapped in <cross_session_memory> XML tags (SimpleMem convention).
    """
    if token_budget is None:
        token_budget = settings.default_token_budget
    sections: list[str] = []
    tokens_used = 0

    def _add(text: str) -> bool:
        nonlocal tokens_used
        text = redact_secrets(text)
        cost = estimate_tokens(text)
        if tokens_used + cost > token_budget:
            return False
        sections.append(text)
        tokens_used += cost
        return True

    # Priority 1: persona (user profile — most stable context)
    if persona_ctx and _add(persona_ctx):
        _add("")

    # Priority 1.5: last session summary (cross-session continuity bridge)
    if last_session_summary and _add("## Last Session Summary"):
        _add(last_session_summary[:600])
        _add("")

    # Priority 1.7: crystallized digests (lessons learned from past sessions)
    if crystal_digests and _add("## Lessons Learned"):
        for digest in crystal_digests[:3]:
            summary = digest.get("summary", "")
            fact_count = digest.get("fact_count", 0)
            entities = digest.get("entities", [])

            digest_text = f"- **Completed Session** ({fact_count} facts): {summary}"
            if not _add(digest_text):
                break

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

    # Priority 3.5: procedural skills
    if proc_ctx and _add("## Relevant Skills"):
        for p in proc_ctx:
            task = p.get("task", "")
            procedure = p.get("procedure", "")
            display = task.split("] ", 1)[-1] if task.startswith("[skill:") else task
            body_hint = procedure[:200].replace("\n", " ").strip()
            line = f"- **{display}**" + (f": {body_hint}" if body_hint else "")
            if not _add(line):
                break
        _add("")

    # Priority 4: session context (Tier 1 rolling summary)
    if session_ctx and _add("## Current Session Context"):
        _add(session_ctx)
        _add("")

    # Priority 5: facts
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
            triple_str = attrs.get("triple_str", "")
            if triple_str:
                meta_parts.append(f"triple:{triple_str}")
            meta = f" ({'; '.join(meta_parts)})" if meta_parts else ""
            line = f"{i}. [{tag}]{meta} {f['content']}"
            if not _add(line):
                break
        _add("")

    # Priority 6: episodes
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

    return (
        "<cross_session_memory>\n"
        "The following is relevant context from long-term memory.\n"
        "Use it to inform your responses but do not repeat it verbatim.\n\n"
        + body
        + "\n</cross_session_memory>"
    )


# ── Query complexity estimator ────────────────────────────────────────────────

_COMPLEX_SIGNALS = re.compile(
    r'\b(what|when|where|who|how|why|which|compare|difference|between|all|every|list'
    r'|什么|怎么|哪|为什么|比较|区别|所有|每个|列出)\b', re.I
)


def estimate_depth(query: str, base_limit: int) -> tuple[int, int]:
    """Adaptive retrieval depth: k_dyn = k_base × (1 + δ × C_q)."""
    words = query.split()
    n_words = len(words)
    n_signals = len(_COMPLEX_SIGNALS.findall(query))
    c_q = min(1.0, (n_words / 30) + (n_signals * 0.2))
    delta = 0.8

    k_base_facts = max(1, base_limit // 2)
    k_base_eps = base_limit

    n_facts = max(1, int(k_base_facts * (1.0 + delta * c_q)))
    n_episodes = max(2, int(k_base_eps * (1.0 + delta * c_q * 0.5)))

    return n_facts, n_episodes


# ── A-MAC 5-Factor Admission Gate ─────────────────────────────────────────────

_ENTITY_RE = re.compile(
    r'\b[A-Z][a-z]{1,}\b'
    r'|\b\d{4}[-/]\d{2}[-/]\d{2}\b'
    r'|\b[A-Z]{2,}\b'
    r'|\b\w+\.(py|js|ts|sh|json|md)\b'
    r'|[\u4e00-\u9fff\u3400-\u4dbf]{2,}'
)
_DATE_RE = re.compile(
    r'\b\d{4}[-/]\d{2}[-/]\d{2}\b'
    r'|\b(?:today|tomorrow|yesterday|monday|tuesday|wednesday|thursday|friday'
    r'|january|february|march|april|may|june|july|august|september|october'
    r'|november|december)\b'
    r'|\d{4}年\d{1,2}月(?:\d{1,2}日)?'
    r'|(?:昨天|今天|明天|上周|下周|上个月|下个月)', re.I
)
_FACT_RE = re.compile(
    r'\b(?:is|are|was|were|has|have|had|will|prefer|prefers|preferred|use|uses'
    r'|used|using|like|likes|liked|work|works|worked|working|live|lives|lived'
    r'|need|needs|needed|always|never|must|should|remember|remembers|note|notes'
    r'|default|workspace|directory|repo|repository|branch|shell|tool|stack|my|our)\b'
    r'|(?:是|有|在|住|工作|喜欢|需要|记住|总是|从不|必须|应该|我的)', re.I
)
_HIGH_VALUE_RE = re.compile(
    r'\b(?:my name|i am|i\'m|i work|i live|i prefer|i like|i use|i always|i never'
    r'|always use|never use|uses|prefers|default shell|package manager|workspace'
    r'|directory|repo|repository|branch|project|tool|stack|deadline|remember'
    r'|important|rule:|note:|prefer|password|token|key|api|model)\b'
    r'|(?:我叫|我是|我在|我住|我喜欢|总是用|截止|记住|重要|密码|令牌)', re.I
)


async def admission_gate(user_text: str) -> bool:
    """A-MAC 5-factor admission control (arXiv:2603.04549)."""
    r = state.redis
    text = user_text[:400]

    # F1 — Semantic novelty (w=0.25)
    emb = encode(text)
    recent = await mem_store.knn_search(r, mem_store.EPISODE_KEY, emb, k=3)
    if recent:
        max_sim = max(res.get("score", 0.0) for res in recent)
        semantic_novelty = 1.0 - max_sim
    else:
        semantic_novelty = 1.0

    # F2 — Entity novelty (w=0.15)
    entities = set(_ENTITY_RE.findall(text))
    entity_novelty = min(1.0, len(entities) / 4)

    # F3 — Factual confidence (w=0.20)
    en_words = len(text.split())
    zh_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    words = max(en_words, 1) + zh_chars // 2
    fact_hits = len(_FACT_RE.findall(text))
    factual_confidence = min(1.0, fact_hits / max(words * 0.15, 1))

    # F4 — Temporal signal (w=0.10)
    temporal_signal = min(1.0, len(_DATE_RE.findall(text)) / 2)

    # F5 — Content type prior (w=0.30, DOMINANT)
    high_value_hits = len(_HIGH_VALUE_RE.findall(text))
    content_type_prior = min(1.0, high_value_hits * 0.5) if high_value_hits else 0.0

    score = (
        0.25 * semantic_novelty
        + 0.15 * entity_novelty
        + 0.20 * factual_confidence
        + 0.10 * temporal_signal
        + 0.30 * content_type_prior
    )

    gate_threshold = float(os.getenv("AMAC_THRESHOLD", "0.40"))
    if score < gate_threshold:
        log.info(
            f"[store] admission gate filtered (score={score:.2f} "
            f"nov={semantic_novelty:.2f} ent={entity_novelty:.2f} "
            f"fact={factual_confidence:.2f} tmp={temporal_signal:.2f} "
            f"typ={content_type_prior:.2f}): {text[:50]!r}"
        )
        return False
    return True


# ── Keyword / importance boost helpers ────────────────────────────────────────

def _tokenize_query(query: str) -> set[str]:
    STOP = {"the", "and", "for", "are", "you", "can", "how", "what",
            "when", "where", "who", "why", "this", "that", "was", "did",
            "的", "了", "是", "在", "有", "我", "你", "他", "她", "它",
            "这", "那", "吗", "呢", "啊", "哦", "嗯"}
    en_tokens = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()))
    zh_tokens = set(re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]{2,}', query))
    return (en_tokens | zh_tokens) - STOP


def keyword_boost(items: list[dict], query: str, boost: float = 0.06) -> list[dict]:
    """SimpleMem Phase 3: lexical layer boost (BM25-lite)."""
    if not items:
        return items
    query_tokens = _tokenize_query(query)
    if not query_tokens:
        return items

    boosted = []
    for item in items:
        attrs = item.get("attrs", {})
        keywords = [kw.lower() for kw in (attrs.get("keywords") or [])]
        content = item.get("content", "").lower()

        kw_overlap = len(query_tokens & set(keywords))
        ct_overlap = sum(1 for t in query_tokens if t in content)
        overlap_score = kw_overlap + ct_overlap * 0.3

        new_item = dict(item)
        new_item["score"] = item.get("score", 0.0) + boost * overlap_score
        boosted.append(new_item)

    boosted.sort(key=lambda x: x["score"], reverse=True)
    return boosted


def importance_boost(items: list[dict], weight: float = 0.15) -> list[dict]:
    """Importance-weighted recall reranking."""
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


def rrf_merge(
    lists: list[list[dict]],
    weights: list[float] | None = None,
    k: int = 60,
    limit: int = 20,
) -> list[dict]:
    """Dynamic Weighted Reciprocal Rank Fusion."""
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


# ── Episode type taxonomy ─────────────────────────────────────────────────────

def infer_ep_type(facts: list) -> str:
    """Map extracted fact categories → claude-mem episode type taxonomy."""
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


# ── A-MEM memory evolution ────────────────────────────────────────────────────

async def evolve_similar_fact(element_id: str, new_keywords: list[str], new_topic: str | None) -> None:
    """A-MEM-inspired memory evolution (arXiv:2502.12110 §3)."""
    if not element_id or not new_keywords:
        return
    r = state.redis
    try:
        attrs = await mem_store.get_attrs(r, mem_store.FACT_KEY, element_id)
        if not attrs:
            return
        existing_kws = set(attrs.get("keywords") or [])
        merged_kws = list(existing_kws | set(new_keywords))[:12]
        attrs["keywords"] = merged_kws
        if new_topic and len(new_topic) > len(attrs.get("topic") or ""):
            attrs["topic"] = new_topic[:100]
        await mem_store.set_attrs(r, mem_store.FACT_KEY, element_id, attrs)
        log.debug(f"[evolution] enriched fact {element_id}: +{len(new_keywords)} kws")
    except Exception as e:
        log.debug(f"[evolution] skipped: {e}")


async def check_contradiction(existing_fact: dict, new_fact) -> None:
    """LLM Wiki v2: Contradiction detection on store."""
    r = state.redis
    element_id = existing_fact.get("_element", "")
    if not element_id:
        return

    old_content = existing_fact.get("content", "").lower()
    new_content = (new_fact.content if hasattr(new_fact, "content") else str(new_fact)).lower()

    old_ents = existing_fact.get("attrs", {}).get("entities") or []
    new_ents = (new_fact.entities if hasattr(new_fact, "entities") else None) or []

    shared_ents = set(e.lower() for e in old_ents) & set(e.lower() for e in new_ents)
    if not shared_ents:
        return

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

    for old_ent in old_ents:
        for new_ent in new_ents:
            if old_ent.lower() != new_ent.lower():
                await graph_mod.add_typed_edge(
                    r, old_ent, new_ent, "contradicts",
                    confidence=0.9, source="contradiction_detector", bidirectional=True,
                )

    await mem_store.soft_delete_fact(r, element_id, "contradicted", reason="contradicted")
    log.info(f"[contradiction] fact {element_id} superseded: contradicted by new fact")


# ── Background store ──────────────────────────────────────────────────────────

async def do_store(messages: list[Message], session_id: str) -> None:
    """Core store logic — extract facts, save episodes, update persona."""
    await state.store_attempts.increment()
    t0 = time.time()
    outcome = "error"
    try:
        clean = [m for m in messages if msg_text(m) and not is_injected_system_content(msg_text(m))]
        if not clean:
            outcome = "skip"
            return
        clean = clean[-4:]

        raw_user_msgs = [
            m for m in messages
            if m.role == "user" and (
                isinstance(m.content, str) and m.content.strip()
            )
        ]
        if raw_user_msgs and all(
            is_only_platform_noise(m.content if isinstance(m.content, str) else "")
            for m in raw_user_msgs
        ):
            log.info("[store] skipped: all user messages are cross_session_memory wrappers")
            outcome = "skip"
            return

        user_text_raw = " ".join(msg_text(m) for m in clean if m.role == "user")
        if not user_text_raw:
            outcome = "skip"
            return

        if is_trivial(user_text_raw):
            log.info(f"[store] skipped trivial exchange: {user_text_raw[:40]!r}")
            outcome = "skip"
            return

        if not await admission_gate(user_text_raw):
            outcome = "skip"
            return

        sc = scene_mod.detect(user_text_raw)
        lang, domain = sc["language"], sc["domain"]

        turn_text = messages_to_text(clean)
        summary = await summarizer.summarize(turn_text)

        ep_emb = encode(summary[:500])
        similar = await mem_store.knn_search(state.redis, mem_store.EPISODE_KEY, ep_emb, k=1)
        is_dup = bool(similar and similar[0].get("score", 0.0) > 0.95)
        ep_saved = 0
        new_ep_uid = ""
        prev_ep_id = ""
        if not is_dup:
            prev_ep_id = await mem_store.get_last_episode_id(state.redis, session_id)
            new_ep_uid = await mem_store.save_episode(
                state.redis, session_id, turn_text[:2000], ep_emb, lang, domain,
                ep_type="general",
                prev_episode_id=prev_ep_id,
            )
            ep_saved = 1

        # 5. Hybrid fact extraction
        raw_msgs = [{"role": m.role, "content": msg_text(m)} for m in clean if msg_text(m)]
        facts = await extractor.extract_hybrid(raw_msgs, turn_text)
        fact_saved = 0
        for fact in facts:
            if contains_secret(fact.content):
                continue
            f_emb = encode(fact.content)
            existing = await mem_store.knn_search(state.redis, mem_store.FACT_KEY, f_emb, k=1)
            if existing and existing[0].get("score", 0.0) > 0.95:
                if fact.keywords and existing[0].get("score", 0.0) > 0.80:
                    state.spawn(evolve_similar_fact(
                        existing[0].get("_element", ""),
                        fact.keywords,
                        fact.topic,
                    ), "evolve")
                continue
            if existing and existing[0].get("score", 0.0) > 0.80:
                state.spawn(check_contradiction(existing[0], fact), "contradiction")
            uid = await mem_store.save_fact(
                state.redis, fact.content, fact.category, fact.confidence, f_emb, lang, domain,
                keywords=fact.keywords, persons=fact.persons, entities=fact.entities,
                importance=fact.importance, topic=fact.topic, location=fact.location,
                source_episode_id=new_ep_uid or None,
                triple_s=fact.triple_s, triple_p=fact.triple_p, triple_o=fact.triple_o,
            )
            attrs = {
                "content": fact.content, "category": fact.category,
                "language": lang, "domain": domain,
                "keywords": fact.keywords or [], "importance": fact.importance,
            }
            if fact.triple_s and fact.triple_p and fact.triple_o:
                attrs["triple_str"] = f"{fact.triple_s} | {fact.triple_p} | {fact.triple_o}"
            await state.bm25_index.add(uid, fact.content, attrs)
            await persona_mod.update(state.redis, fact.category, fact.content)
            fact_saved += 1

        # 5b. Back-fill ep_type + causal chain
        if new_ep_uid:
            ep_type = infer_ep_type(facts)
            try:
                ep_attrs = await mem_store.get_attrs(state.redis, mem_store.EPISODE_KEY, new_ep_uid)
                if ep_attrs:
                    ep_attrs["ep_type"] = ep_type
                    await mem_store.set_attrs(state.redis, mem_store.EPISODE_KEY, new_ep_uid, ep_attrs)
            except Exception:
                pass
            if prev_ep_id and prev_ep_id != new_ep_uid:
                state.spawn(
                    mem_store.update_episode_next_id(state.redis, prev_ep_id, new_ep_uid),
                    "ep-chain",
                )
            await mem_store.set_last_episode_id(state.redis, session_id, new_ep_uid)

        # 6. Extract and store procedures
        proc_facts = [f for f in facts if f.category == "procedure"]
        _KNOWN_TOOLS = {
            "bash", "glob", "grep", "read", "edit", "write", "websearch", "webfetch",
            "agent", "task", "notebook", "notebookedit", "mcp", "skill",
        }
        for pf in proc_facts:
            p_emb = encode(pf.content)
            existing = await mem_store.knn_search(state.redis, mem_store.PROC_KEY, p_emb, k=1)
            if not existing or existing[0].get("score", 0.0) < 0.90:
                content_lower = pf.content.lower()
                tools_in_proc = [t for t in _KNOWN_TOOLS if t in content_lower]
                await mem_store.save_procedure(
                    state.redis, task=pf.content, procedure=pf.content,
                    embedding=p_emb, tools_used=tools_in_proc[:8],
                    domain=domain, language=lang,
                )

        # 7. Knowledge graph edges
        for fact in facts:
            if fact.persons or fact.entities:
                all_ents = (fact.persons or []) + (fact.entities or [])
                if len(all_ents) >= 2:
                    await graph_mod.record_entities(state.redis, all_ents, [])

        # 8. Accumulate rolling session summary
        await accumulate_session(session_id, summary)

        # 9. Auto-consolidation
        fact_count = await state.stores_since_consolidation.increment(fact_saved)
        if fact_count >= settings.auto_consolidate_every:
            await state.stores_since_consolidation.reset()
            from services.consolidation_service import do_consolidate
            state.spawn(do_consolidate(state.redis, state.bm25_index, state.spawn), "consolidate")
            log.info("[store] auto-consolidation triggered")

        ms = int((time.time() - t0) * 1000)
        await state.store_successes.increment()
        outcome = "success"
        log.info(f"[store] session={session_id} lang={lang} domain={domain} "
                 f"ep+{ep_saved} facts+{fact_saved} {ms}ms")

    except Exception as e:
        log.warning(f"[store] background error: {e}", exc_info=True)
    finally:
        ms = int((time.time() - t0) * 1000)
        await state.store_latency_sum_ms.add(ms)
        if outcome == "skip":
            await state.store_skips.increment()
        elif outcome == "error":
            await state.store_errors.increment()


# ── Session helpers ───────────────────────────────────────────────────────────

async def accumulate_session(session_id: str, turn_summary: str) -> None:
    """Tier 1 rolling session summary with MemAgent-style overwrite."""
    r = state.redis
    if not session_id:
        return
    turn_summary = redact_secrets(turn_summary)
    existing = await mem_store.get_session_context(r, session_id) or ""
    if not existing:
        await mem_store.set_session_context(r, session_id, turn_summary[:1500])
        return
    combined = f"{existing}\n---\n{turn_summary}"
    if len(combined) > 1200:
        combined = await summarizer.overwrite_update(existing, turn_summary, target_chars=900)
    await mem_store.set_session_context(r, session_id, combined[:1500])


async def do_compress_session(session_id: str) -> dict:
    """Promote Tier 1 session context → Tier 2 long-term vector memory."""
    r = state.redis
    if not session_id:
        return {"status": "skipped", "reason": "no session_id"}

    ctx = await mem_store.get_session_context(r, session_id)
    if not ctx or len(ctx) < 80:
        return {"status": "skipped", "reason": "session too short to promote"}

    ctx = strip_platform_noise(ctx)
    if not ctx or len(ctx) < 80:
        return {"status": "skipped", "reason": "session context empty after noise strip"}

    sc = scene_mod.detect(ctx[:500])
    lang, domain = sc["language"], sc["domain"]

    session_summary = await summarizer.summarize(ctx)

    ep_emb = encode(session_summary[:500])
    similar = await mem_store.knn_search(r, mem_store.EPISODE_KEY, ep_emb, k=1)
    ep_saved = 0
    if not similar or similar[0].get("score", 0.0) < 0.95:
        await mem_store.save_episode(
            r, session_id,
            f"[Session Summary] {session_summary[:2000]}",
            ep_emb, lang, domain,
        )
        ep_saved = 1

    facts = await extractor.extract_hybrid(
        [{"role": "user", "content": ctx}], ctx
    )
    fact_saved = 0
    for fact in facts:
        if contains_secret(fact.content):
            continue
        f_emb = encode(fact.content)
        existing = await mem_store.knn_search(r, mem_store.FACT_KEY, f_emb, k=1)
        if existing and existing[0].get("score", 0.0) > 0.95:
            if fact.keywords and existing[0].get("score", 0.0) > 0.80:
                state.spawn(evolve_similar_fact(
                    existing[0].get("_element", ""),
                    fact.keywords, fact.topic,
                ))
            continue
        uid = await mem_store.save_fact(
            r, fact.content, fact.category, fact.confidence,
            f_emb, lang, domain,
            keywords=fact.keywords, persons=fact.persons, entities=fact.entities,
            importance=fact.importance, topic=fact.topic, location=fact.location,
            triple_s=fact.triple_s, triple_p=fact.triple_p, triple_o=fact.triple_o,
        )
        attrs = {
            "content": fact.content, "category": fact.category,
            "language": lang, "domain": domain,
            "keywords": fact.keywords or [], "importance": fact.importance,
        }
        if fact.triple_s and fact.triple_p and fact.triple_o:
            attrs["triple_str"] = f"{fact.triple_s} | {fact.triple_p} | {fact.triple_o}"
        await state.bm25_index.add(uid, fact.content, attrs)
        await persona_mod.update(r, fact.category, fact.content)
        fact_saved += 1

    await r.delete(f"{mem_store.SESSION_PRE}{session_id}:ctx")

    clean_summary = strip_platform_noise(session_summary)
    if clean_summary:
        await r.set(state.PINNED_SESSION_KEY, clean_summary[:600].encode())

    log.info(f"[compress] session={session_id} promoted → ep+{ep_saved} facts+{fact_saved}")
    return {
        "status": "ok",
        "session_id": session_id,
        "ep_saved": ep_saved,
        "facts_saved": fact_saved,
    }


# ── Observe internal ──────────────────────────────────────────────────────────

async def observe_internal(session_id: str, _project: str, _cwd: str,
                           hook_type: str, data: dict) -> dict:
    """Core observe logic shared by /observe endpoint."""
    r = state.redis
    tool_name = data.get("tool_name", "")
    tool_input = data.get("tool_input", "")
    tool_output = data.get("tool_output", "")

    content_parts = []
    if tool_name:
        content_parts.append(f"Tool: {tool_name}")
    if tool_input:
        inp = tool_input if isinstance(tool_input, str) else json.dumps(tool_input, ensure_ascii=False)
        content_parts.append(f"Input: {inp[:2000]}")
    if tool_output:
        out = tool_output if isinstance(tool_output, str) else json.dumps(tool_output, ensure_ascii=False)
        content_parts.append(f"Output: {out[:4000]}")

    content = "\n".join(content_parts)
    if not content.strip():
        return {"status": "ok", "action": "skipped_empty"}

    if is_trivial(content) or is_injected_system_content(content):
        return {"status": "ok", "action": "skipped_filter"}
    if contains_secret(content):
        content = redact_secrets(content)

    if not await admission_gate(content):
        return {"status": "ok", "action": "skipped_gate"}

    emb = encode(content)
    uid = await mem_store.save_episode(
        r, session_id, content, emb,
        ep_type=hook_type,
    )

    try:
        facts = await extractor.extract_facts(content, session_id)
        for fact_text, fact_attrs in facts:
            fact_emb = encode(fact_text)
            await mem_store.save_fact(
                r,
                content=fact_text,
                category=fact_attrs.get("category", "fact"),
                confidence=fact_attrs.get("confidence", 0.7),
                embedding=fact_emb,
                language=fact_attrs.get("language", "en"),
                domain=fact_attrs.get("domain", "general"),
                keywords=fact_attrs.get("keywords"),
                importance=fact_attrs.get("importance", 0.5),
                source_episode_id=uid,
            )
    except Exception as e:
        log.warning("[observe] fact extraction failed: %s", e)

    return {"status": "ok", "action": "observed", "id": uid}
