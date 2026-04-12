"""
AgentMem — Local Agent Memory Service  v1.0.0
==============================================
Replaces memos-cloud-openclaw-plugin with fully local, persistent memory.

2026 Layered Controlled Architecture (Tier 0-2):
  Tier 0 — Working memory    : in-context window (LLM prompt)
  Tier 1 — Session KV        : rolling accumulated session summary (Redis String, 4h TTL)
  Tier 2 — Long-term vectors : episodic + semantic + procedural (Redis HNSW, permanent)
  +       — Capability layer : tool/env/agent self-model (Redis Hash + vectorset)

Features (v0.9.2 — A-MAC + wRRF + importance floors):
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
 18. Dynamic weighted RRF      — per-query-type wRRF weights: entity/temporal/semantic (v0.9.2)
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
                                 4th wRRF input alongside vector/symbolic/global (v1.1)
 28. A-MEM memory evolution   — near-duplicate facts (cos 0.80-0.95) trigger async
                                 keyword/topic enrichment on existing fact (v1.1)
 29. Dual-layer fact linking  — source_episode_id links each fact to source episode
                                 for Memori-style narrative context retrieval (v1.1)

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
import logging
import math
import re
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any

import httpx
import numpy as np
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from rank_bm25 import BM25Okapi as _BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False

from core import capability as cap_mod
from core import embedder
from core import extractor
from core import graph as graph_mod
from core import heat as heat_mod
from core import log_sse
from core import persona as persona_mod
from core import retrieval_planner
from core import scene as scene_mod
from core import store as mem_store
from core import summarizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [mem] %(message)s")
log = logging.getLogger("mem")

# Attach SSE log handler — broadcasts every log record to dashboard clients
_sse_handler = log_sse.LogSSEHandler(level=logging.INFO)
_sse_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s"))
logging.getLogger().addHandler(_sse_handler)

_redis = None

# ── Auto-consolidation counters ────────────────────────────────────────────────
_stores_since_consolidation: int = 0
_AUTO_CONSOLIDATE_EVERY: int = 50   # trigger after every N stored facts

# ── Hard-prune (physical VREM of superseded entries) runs every 24 hours ───────
_periodic_prune_counter: int = 0    # incremented each hourly _periodic_consolidate call

# ── Store observability counters (in-process, resets on restart) ───────────────
_store_attempts: int = 0
_store_successes: int = 0
_store_latency_sum_ms: float = 0.0

# ── System-injected content filter ────────────────────────────────────────────
_SKIP_PREFIXES = (
    "## Long-Term Memory",
    "## Recent Relevant Episodes",
    "## Current Session Context",
    "## User Profile",
    "[cron:",
)

# Platform tags that must never be stored as content
_PLATFORM_TAG_RE = re.compile(
    r"\[\[reply_to_current\]\]|\[\[reply_to:[^\]]*\]\]",
    re.IGNORECASE,
)

# Matches a message that is ONLY a <cross_session_memory> block (possibly followed
# by whitespace or a HEARTBEAT keyword — nothing real).
_ONLY_CROSS_SESSION_RE = re.compile(
    r"^\s*<cross_session_memory>.*?</cross_session_memory>\s*(HEARTBEAT[^\n]*)?\s*$",
    re.DOTALL | re.IGNORECASE,
)

# Matches the leading <cross_session_memory>…</cross_session_memory> block so we
# can strip it and keep whatever real message follows.
_CROSS_SESSION_PREFIX_RE = re.compile(
    r"^\s*<cross_session_memory>.*?</cross_session_memory>\s*",
    re.DOTALL | re.IGNORECASE,
)


_OPTION_REPLY_RE = re.compile(
    r"^\s*(?:option|choice|select|pick)?\s*[#(]?\s*(\d{1,2})\s*[)\].:,-]?\s*$",
    re.IGNORECASE,
)

_SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{12,}\b"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{10,}\b"), "[REDACTED_API_KEY]"),
    (re.compile(r"\btoken-active-\d+\b", re.IGNORECASE), "[REDACTED_TOKEN]"),
]

_SECRET_KEYWORD_VALUE_RE = re.compile(
    r"(?i)\b(password|token|api[_ -]?key|secret)\b"
    r"(\s*(?:is|=|:|for\s+[^:\n]{1,80}?\s+is)\s*)"
    r"([\'\"]?)([^\'\"\s;,\n]+)\3"
)

# ── BM25 in-memory index (Memori arXiv:2603.19935 hybrid retrieval) ────────────
# Maintains a live BM25Okapi corpus over stored facts for exact-term matching.
# Vector search is excellent for semantic similarity but misses exact names/dates.
# BM25 complements by scoring term frequency — together they form true hybrid search.

class _BM25Index:
    """Lazy-rebuild BM25 corpus over fact contents + triple strings.

    Thread-safe for asyncio: all mutations happen in the event loop.
    Rebuilt when corpus size changes by ≥10 entries or when forced.
    """
    def __init__(self):
        self._docs: list[tuple[str, str, dict]] = []   # (uid, content, attrs)
        self._bm25 = None
        self._built_at_len: int = 0
        self._rebuild_threshold: int = 10

    def add(self, uid: str, content: str, attrs: dict) -> None:
        """Add a fact to the corpus. Invalidates the BM25 index."""
        self._docs.append((uid, content, attrs))

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize for BM25: English words ≥3 chars + CJK bigrams."""
        text = text.lower()
        en = re.findall(r'\b[a-z]{3,}\b', text)
        zh = re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]{2,}', text)
        return en + zh

    def _ensure_built(self) -> bool:
        """Rebuild BM25 index if corpus has changed enough. Returns False if corpus empty."""
        n = len(self._docs)
        if n == 0:
            return False
        if self._bm25 is None or (n - self._built_at_len) >= self._rebuild_threshold:
            # Build corpus: content + triple_str for richer term coverage
            corpus = []
            for _, content, attrs in self._docs:
                text = content
                triple_str = attrs.get("triple_str", "")
                if triple_str:
                    text = f"{text} {triple_str}"
                corpus.append(self._tokenize(text))
            self._bm25 = _BM25Okapi(corpus)
            self._built_at_len = n
            log.debug(f"[bm25] rebuilt corpus n={n}")
        return True

    def search(self, query: str, k: int = 10) -> list[dict]:
        """Return top-k facts by BM25 score. Returns [] if corpus empty or BM25 unavailable."""
        if not _BM25_AVAILABLE or not self._ensure_built():
            return []
        tokens = self._tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        # Get top-k indices (non-zero scores only)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k * 2]
        results = []
        for idx in top_idx:
            if scores[idx] <= 0:
                break
            uid, content, attrs = self._docs[idx]
            results.append({
                "content": content,
                "category": attrs.get("category", ""),
                "language": attrs.get("language", "en"),
                "domain":   attrs.get("domain", "general"),
                "score":    float(scores[idx]),
                "attrs":    attrs,
                "_element": uid,
            })
            if len(results) >= k:
                break
        return results

    def populate_from_items(self, items: list[dict]) -> None:
        """Bulk-load from knn_search result items (called at startup)."""
        for item in items:
            uid = item.get("_element", "")
            content = item.get("content", "")
            attrs = item.get("attrs", {})
            if uid and content:
                self._docs.append((uid, content, attrs))

    def reset(self) -> None:
        self._docs.clear()
        self._bm25 = None
        self._built_at_len = 0


_bm25_index = _BM25Index()


def _is_injected(text: str) -> bool:
    t = text.strip()
    return any(t.startswith(p) for p in _SKIP_PREFIXES)


def _strip_platform_noise(text: str) -> str:
    """Remove platform tags and cross_session_memory prefixes from content.

    1. Strip leading <cross_session_memory>…</cross_session_memory> block.
    2. Remove any remaining [[reply_to_current]] / [[reply_to:<id>]] tags.
    """
    # Strip cross_session_memory prefix block (keep real content after it)
    text = _CROSS_SESSION_PREFIX_RE.sub("", text)
    # Strip platform reply tags anywhere in the text
    text = _PLATFORM_TAG_RE.sub("", text)
    return text.strip()


def _is_only_platform_noise(text: str) -> bool:
    """Return True when the entire message is a cross_session_memory wrapper
    with no meaningful user content after it (or just a HEARTBEAT line).
    """
    return bool(_ONLY_CROSS_SESSION_RE.match(text))


def _is_brief_option_reply(text: str) -> bool:
    return bool(_OPTION_REPLY_RE.match(text or ""))


def _contains_secret(text: str) -> bool:
    if not text:
        return False
    if any(pattern.search(text) for pattern, _ in _SECRET_PATTERNS):
        return True
    return bool(_SECRET_KEYWORD_VALUE_RE.search(text))


def _redact_secrets(text: str) -> str:
    if not text:
        return ""
    redacted = text
    for pattern, replacement in _SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return _SECRET_KEYWORD_VALUE_RE.sub(
        lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]",
        redacted,
    )


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

# ── Session handoff: pinned last-session summary ───────────────────────────────
# Key stores the most recent session summary so the NEXT session can always
# retrieve it regardless of query similarity (cross-session continuity bridge).
_PINNED_SESSION_KEY = "mem:pinned:session_summary"


# ── Lifespan ──────────────────────────────────────────────────────────────────
async def _periodic_consolidate() -> None:
    """Background task: consolidate memory every hour; hard-prune every 24 hours."""
    global _periodic_prune_counter
    while True:
        await asyncio.sleep(3600)
        try:
            card = await _redis.execute_command("VCARD", mem_store.FACT_KEY)
            if int(card or 0) > 5:
                await _do_consolidate()
        except Exception as e:
            log.warning(f"[periodic_consolidate] error: {e}")
        _periodic_prune_counter += 1
        if _periodic_prune_counter >= 24:
            _periodic_prune_counter = 0
            try:
                await _do_hard_prune()
            except Exception as e:
                log.warning(f"[periodic_consolidate] hard-prune error: {e}")


async def _populate_bm25_from_redis(r) -> None:
    """Load all existing facts from Redis into the in-memory BM25 corpus on startup."""
    if not _BM25_AVAILABLE:
        return
    try:
        card = await r.execute_command("VCARD", mem_store.FACT_KEY)
        if not card or int(card) <= 1:
            return
        seed = np.zeros(mem_store.DIMS, dtype=np.float32)
        results = await r.execute_command(
            "VSIM", mem_store.FACT_KEY, "FP32", seed.astype("float32").tobytes(),
            "COUNT", min(int(card), 5000), "WITHSCORES", "WITHATTRIBS"
        )
        items: list[dict] = []
        i = 0
        while i + 2 < len(results):
            elem = results[i]; raw = results[i + 2]
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
            if attrs.get("superseded_by"):
                continue
            items.append({"_element": elem_str, "content": attrs["content"], "attrs": attrs})
        _bm25_index.populate_from_items(items)
        log.info(f"[bm25] populated corpus from Redis: {len(items)} facts")
    except Exception as e:
        log.warning(f"[bm25] startup population failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _redis
    log.info("Loading embedding model…")
    embedder.get_model()
    log.info("Connecting to Redis…")
    _redis = await mem_store.get_client()
    log.info("Ensuring vectorset indexes…")
    await mem_store.ensure_indexes(_redis)
    asyncio.create_task(_periodic_consolidate())
    # Populate BM25 corpus from existing Redis facts (non-blocking, best-effort)
    asyncio.create_task(_populate_bm25_from_redis(_redis))
    log.info("AgentMem v1.0.0 ready (session-handoff+hard-prune+auto-graph+batch-MCP)")
    yield
    await _redis.aclose()


app = FastAPI(title="AgentMem — Local Agent Memory Service", version="1.0.0", lifespan=lifespan)

# ── Static files + SSE log router ─────────────────────────────────────────────
import os as _os
_static_dir = _os.path.join(_os.path.dirname(__file__), "static")
if _os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

app.include_router(log_sse.log_sse_router)


@app.get("/", include_in_schema=False)
async def dashboard():
    """Serve the memory service dashboard."""
    idx = _os.path.join(_static_dir, "index.html")
    if _os.path.exists(idx):
        return FileResponse(idx)
    return {"message": "AgentMem v0.9.4", "docs": "/docs"}


# ── Request / Response models ──────────────────────────────────────────────────
class RecallRequest(BaseModel):
    query: str
    session_id: str = ""
    memory_limit_number: int = 6
    include_tools: bool = True          # Whether to append relevant tool context
    include_procedures: bool = False    # Whether to append relevant skill/procedure context (v0.9.4)
    include_graph: bool = False         # Explicitly expand knowledge graph neighbours (v0.9.0)
    auto_graph: bool = True             # Auto-trigger graph expansion when query has named entities (v1.0)
    # SimpleMem Section 3.3: Intent-Aware Retrieval Planning (v0.9.1)
    enable_planning: bool = False       # LLM-based query planning (adds ~300-600ms)
    enable_reflection: bool = False     # LLM sufficiency check + second-pass retrieval
    token_budget: int = 1500            # Max tokens in context output (SimpleMem budget)
    # SimpleMem symbolic filter: restrict recall to a time window (Unix ms)
    time_from: int | None = None        # e.g. 1740000000000 (ms)
    time_to:   int | None = None        # e.g. 1742687999000 (ms)

class Message(BaseModel):
    role: str
    content: Any

class StoreRequest(BaseModel):
    messages: list[Message]
    session_id: str = ""

# ── Capability request models ──────────────────────────────────────────────────
class ToolDefinition(BaseModel):
    name: str
    description: str
    category: str = ""
    source: str = "builtin"             # builtin | mcp | plugin | skill
    parameters: list[str] = []

class RegisterToolsRequest(BaseModel):
    tools: list[ToolDefinition]
    agent_id: str = ""                  # optional agent identifier
    replace_all: bool = False           # if True, clear existing tools first

class EnvState(BaseModel):
    os: str = ""
    os_version: str = ""
    shell: str = ""
    cwd: str = ""
    git_repo: str = ""
    git_branch: str = ""
    runtime: str = ""                   # e.g. "python3.12", "node20"
    active_mcps: list[str] = []
    active_plugins: list[str] = []
    active_skills: list[str] = []
    agent_model: str = ""               # e.g. "claude-sonnet-4-6"
    agent_version: str = ""
    session_id: str = ""
    extra: dict = {}                    # any extra key/value pairs

class RecallToolsRequest(BaseModel):
    query: str
    k: int = 5
    category: str = ""                  # optional filter by tool category
    source: str = ""                    # optional filter by source

class StoreProcedureRequest(BaseModel):
    task: str                           # what kind of task (used as embedding key)
    procedure: str                      # the step-by-step procedure
    tools_used: list[str] = []          # tools/skills involved
    domain: str = ""
    session_id: str = ""

# ── ToolMem + TIG + MACLA request models (v0.9.5) ─────────────────────────────
class ToolFeedbackRequest(BaseModel):
    tool_name: str                      # slugified tool name (e.g. "web_search_prime")
    success: bool                       # did the tool invocation succeed?
    session_id: str = ""

class ToolSequenceRequest(BaseModel):
    sequence: list[str]                 # ordered list of tool names used in session
    session_id: str = ""

class ProcedureFeedbackRequest(BaseModel):
    task_prefix: str                    # first ~80 chars of task to identify procedure
    success: bool
    session_id: str = ""


# ── Helpers ────────────────────────────────────────────────────────────────────
def _msg_text(m: Message) -> str:
    if isinstance(m.content, str):
        raw = m.content.strip()
    elif isinstance(m.content, list):
        raw = " ".join(b.get("text","") for b in m.content if isinstance(b, dict)).strip()
    else:
        return ""
    # Strip cross_session_memory prefixes and platform tags before returning
    return _strip_platform_noise(raw)

def _messages_to_text(messages: list[Message]) -> str:
    return "\n".join(
        f"{m.role}: {_msg_text(m)}" for m in messages if _msg_text(m)
    )

def _estimate_tokens(text: str) -> int:
    """Fast word-count token estimate (SimpleMem Section 3.3 budget packing)."""
    return len(text.split())


def _format_prepend(
    facts: list[dict],
    episodes: list[dict],
    session_ctx: str | None,
    persona_ctx: str,
    env_ctx: str = "",
    tool_ctx: str = "",
    proc_ctx: list[dict] | None = None,
    token_budget: int = 1500,
    last_session_summary: str | None = None,
) -> str:
    """
    Build the context string injected into the agent's system prompt.

    SimpleMem Section 3.3 token-budget packing:
    Priority order: persona → last_session_summary → env → tools → skills → session_ctx → facts → episodes
    Greedy packing: stop adding items when token_budget is exhausted.
    Output wrapped in <cross_session_memory> XML tags (SimpleMem convention).
    """
    sections: list[str] = []
    tokens_used = 0

    def _add(text: str) -> bool:
        nonlocal tokens_used
        text = _redact_secrets(text)
        cost = _estimate_tokens(text)
        if tokens_used + cost > token_budget:
            return False
        sections.append(text)
        tokens_used += cost
        return True

    # Priority 1: persona (user profile — most stable context)
    if persona_ctx and _add(persona_ctx):
        _add("")

    # Priority 1.5: last session summary (cross-session continuity bridge)
    # Always injected when present so the agent knows what was accomplished
    # in the previous session, regardless of current query similarity.
    if last_session_summary and _add("## Last Session Summary"):
        _add(last_session_summary[:600])
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
            display = task.split("] ", 1)[-1] if task.startswith("[skill:") else task
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
    return (
        "<cross_session_memory>\n"
        "The following is relevant context from long-term memory.\n"
        "Use it to inform your responses but do not repeat it verbatim.\n\n"
        + body
        + "\n</cross_session_memory>"
    )


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
    r'\b(?:is|are|was|were|has|have|had|will|prefer|use|like|work|live|need'
    r'|always|never|must|should|remember|note|my|our)\b'
    r'|(?:是|有|在|住|工作|喜欢|需要|记住|总是|从不|必须|应该|我的)', re.I
)
_HIGH_VALUE_RE = re.compile(           # content type prior: high-value (EN + ZH)
    r'\b(?:my name|i am|i\'m|i work|i live|i prefer|i like|always use|never use'
    r'|deadline|remember|important|rule:|note:|prefer|password|token|key|api)\b'
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
    emb = _encode(text)
    recent = await mem_store.knn_search(_redis, mem_store.EPISODE_KEY, emb, k=3)
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

    _gate_threshold = float(_os.getenv("AMAC_THRESHOLD", "0.40"))
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
    Dynamic Weighted Reciprocal Rank Fusion.

    Standard RRF (Cormack et al., SIGIR 2009):
      RRF(d) = Σ_i  w_i / (k + rank_i(d))   k=60 smoothing constant

    Weights are query-type adaptive (arXiv:2511.18194 wRRF for agents):
      - Default equal weights [1.0, 1.0, 1.0] for [scene, global, symbolic]
      - Caller passes adjusted weights based on query type
      - Entity/person queries: [0.8, 0.8, 1.4] → upweight symbolic pass
      - Temporal queries: [0.9, 0.9, 1.2] → upweight symbolic (has timestamps)
      - Semantic/general: [1.0, 1.0, 0.6] → downweight symbolic if no entities
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
        raw = await _redis.execute_command("VGETATTR", mem_store.FACT_KEY, element_id)
        if not raw:
            return
        attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        existing_kws = set(attrs.get("keywords") or [])
        # Merge new keywords (deduplicated, capped at 12)
        merged_kws = list(existing_kws | set(new_keywords))[:12]
        attrs["keywords"] = merged_kws
        # Update topic if new one is more specific (longer = more specific)
        if new_topic and len(new_topic) > len(attrs.get("topic") or ""):
            attrs["topic"] = new_topic[:100]
        await _redis.execute_command("VSETATTR", mem_store.FACT_KEY, element_id, json.dumps(attrs))
        log.debug(f"[evolution] enriched fact {element_id}: +{len(new_keywords)} kws")
    except Exception as e:
        log.debug(f"[evolution] skipped: {e}")


# ── Background store ──────────────────────────────────────────────────────────
async def _do_store(messages: list[Message], session_id: str) -> None:
    global _store_attempts, _store_successes, _store_latency_sum_ms
    _store_attempts += 1
    t0 = time.time()
    try:
        # 1. Filter system-injected and empty messages.
        #    _msg_text() strips <cross_session_memory> prefixes and [[platform_tags]]
        #    before returning text, so purely-noise messages collapse to "" here.
        clean = [m for m in messages if _msg_text(m) and not _is_injected(_msg_text(m))]
        if not clean:
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
            return

        # 2. Detect scene from user messages
        user_text_raw = " ".join(_msg_text(m) for m in clean if m.role == "user")
        if not user_text_raw:
            return

        # 2b. Skip trivial exchanges (greetings, single words) — prevents feedback loops
        if _is_trivial(user_text_raw):
            log.info(f"[store] skipped trivial exchange: {user_text_raw[:40]!r}")
            return

        # 2c. A-MAC 5-factor admission gate (arXiv:2603.04549).
        #     Discards low-value inputs before heavy processing (summarize/embed/LLM).
        #     type_prior is dominant (w=0.30); threshold 0.40.
        if not await _admission_gate(user_text_raw):
            return  # low-entropy input, skip (not worth storing)

        sc = scene_mod.detect(user_text_raw)
        lang, domain = sc["language"], sc["domain"]

        # 3. Summarize long turn before embedding
        turn_text = _messages_to_text(clean)
        summary = await summarizer.summarize(turn_text)

        # 4. Embed + dedup-check + save episode (with type taxonomy + causal chain)
        ep_emb = _encode(summary[:500])
        similar = await mem_store.knn_search(_redis, mem_store.EPISODE_KEY, ep_emb, k=1)
        is_dup = bool(similar and similar[0].get("score", 0.0) > 0.95)
        ep_saved = 0
        new_ep_uid = ""
        if not is_dup:
            # Causal chain: look up the previous episode for this session
            prev_ep_id = await mem_store.get_last_episode_id(_redis, session_id)
            new_ep_uid = await mem_store.save_episode(
                _redis, session_id, turn_text[:2000], ep_emb, lang, domain,
                ep_type="general",        # placeholder; updated after fact extraction below
                prev_episode_id=prev_ep_id,
            )
            ep_saved = 1

        # 5. Hybrid fact extraction (regex + LLM) → dedup → save → update persona
        # Extract from all roles: assistant messages contain rich facts too
        # (e.g. "I work at NovaPay", "I live in Oakland" are in assistant turns)
        raw_msgs = [m.model_dump() for m in clean]
        facts = await extractor.extract_hybrid(raw_msgs, turn_text)
        fact_saved = 0
        for fact in facts:
            if _contains_secret(fact.content):
                continue
            f_emb = _encode(fact.content)
            existing = await mem_store.knn_search(_redis, mem_store.FACT_KEY, f_emb, k=1)
            if existing and existing[0].get("score", 0.0) > 0.95:
                # A-MEM memory evolution: similar fact exists → async-update its keywords/topic
                # (arXiv:2502.12110 §3: new memories trigger context refinement on neighbors)
                if fact.keywords and existing[0].get("score", 0.0) > 0.80:
                    asyncio.create_task(_evolve_similar_fact(
                        existing[0].get("_element", ""),
                        fact.keywords,
                        fact.topic,
                    ))
                continue
            uid = await mem_store.save_fact(
                _redis, fact.content, fact.category, fact.confidence, f_emb, lang, domain,
                keywords=fact.keywords, persons=fact.persons, entities=fact.entities,
                importance=fact.importance, topic=fact.topic, location=fact.location,
                source_episode_id=new_ep_uid or None,
                triple_s=fact.triple_s, triple_p=fact.triple_p, triple_o=fact.triple_o,
            )
            # Add to in-memory BM25 corpus for hybrid recall
            attrs = {
                "content": fact.content, "category": fact.category,
                "language": lang, "domain": domain,
                "keywords": fact.keywords or [], "importance": fact.importance,
            }
            if fact.triple_s and fact.triple_p and fact.triple_o:
                attrs["triple_str"] = f"{fact.triple_s} | {fact.triple_p} | {fact.triple_o}"
            _bm25_index.add(uid, fact.content, attrs)
            await persona_mod.update(_redis, fact.category, fact.content)
            fact_saved += 1

        # 5b. Back-fill ep_type on newly saved episode + complete causal chain
        if new_ep_uid:
            ep_type = _infer_ep_type(facts)
            # Update ep_type attr on the episode we just saved
            try:
                raw = await _redis.execute_command("VGETATTR", mem_store.EPISODE_KEY, new_ep_uid)
                if raw:
                    ep_attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
                    ep_attrs["ep_type"] = ep_type
                    await _redis.execute_command(
                        "VSETATTR", mem_store.EPISODE_KEY, new_ep_uid, json.dumps(ep_attrs)
                    )
            except Exception:
                pass  # VGETATTR not critical — ep_type stays "general"
            # Complete doubly-linked chain: tell prev episode its successor
            prev_ep_id = (await _redis.get(f"{mem_store.SESSION_PRE}{session_id}:last_ep") or b"").decode()
            if prev_ep_id and prev_ep_id != new_ep_uid:
                asyncio.create_task(
                    mem_store.update_episode_next_id(_redis, prev_ep_id, new_ep_uid)
                )
            # Advance the session's last-episode pointer
            await mem_store.set_last_episode_id(_redis, session_id, new_ep_uid)
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
        for pf in proc_facts:
            p_emb = _encode(pf.content)
            existing = await mem_store.knn_search(_redis, mem_store.PROC_KEY, p_emb, k=1)
            if not existing or existing[0].get("score", 0.0) < 0.90:
                # Match only known tool names mentioned in the procedure text
                content_lower = pf.content.lower()
                tools_in_proc = [t for t in _KNOWN_TOOLS if t in content_lower]
                await mem_store.save_procedure(
                    _redis, task=pf.content, procedure=pf.content,
                    embedding=p_emb, tools_used=tools_in_proc[:8],
                    domain=domain, language=lang,
                )

        # 7. Record knowledge graph edges from persons + entities co-occurrences
        for fact in facts:
            if fact.persons or fact.entities:
                all_ents = (fact.persons or []) + (fact.entities or [])
                if len(all_ents) >= 2:
                    await graph_mod.record_entities(_redis, all_ents, [])

        # 8. Accumulate rolling session summary (Tier 1 — layered controlled architecture)
        #    Append this turn's summary instead of overwriting, so Tier 1 grows across
        #    the whole session rather than holding only the last turn.
        await _accumulate_session(session_id, summary)

        # 9. Auto-consolidation: trigger after every _AUTO_CONSOLIDATE_EVERY stored facts
        global _stores_since_consolidation
        _stores_since_consolidation += fact_saved
        if _stores_since_consolidation >= _AUTO_CONSOLIDATE_EVERY:
            _stores_since_consolidation = 0
            asyncio.create_task(_do_consolidate())
            log.info("[store] auto-consolidation triggered")

        ms = int((time.time() - t0) * 1000)
        _store_successes += 1
        _store_latency_sum_ms += ms
        log.info(f"[store] session={session_id} lang={lang} domain={domain} "
                 f"ep+{ep_saved} facts+{fact_saved} {ms}ms")

    except Exception as e:
        ms = int((time.time() - t0) * 1000)
        _store_latency_sum_ms += ms
        log.warning(f"[store] background error: {e}", exc_info=True)


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
    existing = await mem_store.get_session_context(_redis, session_id) or ""
    if not existing:
        await mem_store.set_session_context(_redis, session_id, turn_summary[:1500])
        return
    combined = f"{existing}\n---\n{turn_summary}"
    if len(combined) > 1200:
        # MemAgent-style incremental overwrite: LLM decides what to keep
        combined = await summarizer.overwrite_update(existing, turn_summary, target_chars=900)
    await mem_store.set_session_context(_redis, session_id, combined[:1500])


async def _do_compress_session(session_id: str) -> dict:
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

    ctx = await mem_store.get_session_context(_redis, session_id)
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
    ep_emb  = _encode(session_summary[:500])
    similar = await mem_store.knn_search(_redis, mem_store.EPISODE_KEY, ep_emb, k=1)
    ep_saved = 0
    if not similar or similar[0].get("score", 0.0) < 0.95:
        await mem_store.save_episode(
            _redis, session_id,
            f"[Session Summary] {session_summary[:2000]}",
            ep_emb, lang, domain,
        )
        ep_saved = 1

    # Hybrid fact extraction on the full session text
    facts = await extractor.extract_hybrid(
        [{"role": "user", "content": ctx}], ctx
    )
    fact_saved = 0
    for fact in facts:
        if _contains_secret(fact.content):
            continue
        f_emb    = _encode(fact.content)
        existing = await mem_store.knn_search(_redis, mem_store.FACT_KEY, f_emb, k=1)
        if existing and existing[0].get("score", 0.0) > 0.95:
            if fact.keywords and existing[0].get("score", 0.0) > 0.80:
                asyncio.create_task(_evolve_similar_fact(
                    existing[0].get("_element", ""),
                    fact.keywords, fact.topic,
                ))
            continue
        uid = await mem_store.save_fact(
            _redis, fact.content, fact.category, fact.confidence,
            f_emb, lang, domain,
            keywords=fact.keywords, persons=fact.persons, entities=fact.entities,
            importance=fact.importance, topic=fact.topic, location=fact.location,
            triple_s=fact.triple_s, triple_p=fact.triple_p, triple_o=fact.triple_o,
        )
        # Add to BM25 corpus
        attrs = {
            "content": fact.content, "category": fact.category,
            "language": lang, "domain": domain,
            "keywords": fact.keywords or [], "importance": fact.importance,
        }
        if fact.triple_s and fact.triple_p and fact.triple_o:
            attrs["triple_str"] = f"{fact.triple_s} | {fact.triple_p} | {fact.triple_o}"
        _bm25_index.add(uid, fact.content, attrs)
        await persona_mod.update(_redis, fact.category, fact.content)
        fact_saved += 1

    # Delete Tier 1 KV — session has been crystallised into long-term memory
    await _redis.delete(f"{mem_store.SESSION_PRE}{session_id}:ctx")

    # Pin the session summary for the NEXT session's recall (cross-session handoff).
    # Strip platform tags before persisting — summaries must contain real content only.
    # Stored without TTL so it persists until the next compress overwrites it.
    clean_summary = _strip_platform_noise(session_summary)
    if clean_summary:
        await _redis.set(_PINNED_SESSION_KEY, clean_summary[:600].encode())

    log.info(f"[compress] session={session_id} promoted → ep+{ep_saved} facts+{fact_saved}")
    return {
        "status":     "ok",
        "session_id": session_id,
        "ep_saved":   ep_saved,
        "facts_saved":fact_saved,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

class CompactRequest(BaseModel):
    session_id: str
    threshold_chars: int = 3000   # only compact if session KV exceeds this


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

    ctx = await mem_store.get_session_context(_redis, req.session_id)
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

    await mem_store.set_session_context(_redis, req.session_id, compacted[:1500])

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


class AnswerRequest(BaseModel):
    query:   str
    context: str


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
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(
                    _AISERV_OAI,
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_msg},
                            {"role": "user",   "content": prompt},
                        ],
                        "temperature": 0.0,
                        "max_tokens": 80,
                    },
                    headers={"Authorization": f"Bearer {_AISERV_KEY}"},
                )
            if resp.status_code == 200:
                raw = resp.json()["choices"][0]["message"]["content"].strip()
                answer = _parse_answer_raw(raw)
                if answer:
                    asyncio.create_task(extractor._report_quality(model, +1))
                    return {"answer": answer}
            exclude = model
        except httpx.TimeoutException:
            log.warning("[answer] %s timed out (attempt %d)", model, attempt + 1)
            asyncio.create_task(extractor._report_quality(model, -1))
            exclude = model
        except Exception as e:
            log.debug("[answer] %s failed: %s", model, e)
            asyncio.create_task(extractor._report_quality(model, -1))
            exclude = model
    return {"answer": ""}


@app.get("/health")
async def health():
    try:
        await _redis.ping()
        return {"status": "ok", "redis": "ok", "version": "1.0.0"}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.get("/stats")
async def stats():
    try:
        counts = await mem_store.vcard(_redis)
        persona_raw = await _redis.hgetall("mem:persona")
        counts["persona_fields"] = len(persona_raw)

        # v0.6.0+: capability + procedural stats
        tool_card = await _redis.execute_command("VCARD", cap_mod.TOOL_KEY)
        counts["tools"] = max(0, int(tool_card or 0) - 1)

        proc_card = await _redis.execute_command("VCARD", mem_store.PROC_KEY)
        counts["procedures"] = max(0, int(proc_card or 0) - 1)

        env_raw = await _redis.hgetall(cap_mod.ENV_KEY)
        counts["env_fields"] = len(env_raw)

        # v0.9.7: store observability counters (since last restart)
        counts["store_attempts"] = _store_attempts
        counts["store_successes"] = _store_successes
        counts["store_failures"] = _store_attempts - _store_successes
        if _store_attempts > 0:
            counts["store_success_rate"] = round(_store_successes / _store_attempts, 3)
            counts["store_avg_ms"] = round(_store_latency_sum_ms / _store_attempts)
        else:
            counts["store_success_rate"] = None
            counts["store_avg_ms"] = None

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

    # Adaptive retrieval depth (SimpleMem-inspired C_q estimator)
    n_facts, n_episodes = _estimate_depth(req.query, req.memory_limit_number)

    # Build filter expressions for scene isolation
    lang_filter   = f'.language == "{q_lang}"'
    domain_filter = f'.domain == "{q_domain}"' if q_domain != "general" else None

    # SimpleMem symbolic filter: time-range hard constraint
    # Combines with lang_filter as a compound VSIM FILTER expression
    def _build_filter(base_filter: str) -> str:
        parts = [base_filter] if base_filter else []
        if req.time_from is not None:
            parts.append(f".ts >= {req.time_from}")
        if req.time_to is not None:
            parts.append(f".ts <= {req.time_to}")
        return " && ".join(parts) if parts else ""

    lang_filter_sym = _build_filter(lang_filter)

    # Recall strategy: scene-filtered first, supplement with global if sparse
    async def _fetch(vset: str, k: int, fexpr=None):
        return await mem_store.knn_search(
            _redis, vset, emb, k, filter_expr=fexpr, bump_heat=True
        )

    # Detect if this is a capability query (needs tool context)
    is_cap_query = query_scene.get("is_capability_query", False)

    # Parallel: persona + env + session ctx + scene-filtered facts + episodes
    # + tool recall if capability query or include_tools=True
    # + symbolic structured pass (async, no extra latency via gather)
    # + pinned last-session summary (cross-session handoff bridge, index 6)
    gather_tasks = [
        persona_mod.get_context(_redis),
        cap_mod.get_env_context(_redis),
        mem_store.get_session_context(_redis, req.session_id),
        _fetch(mem_store.FACT_KEY,    n_facts,    lang_filter_sym or None),
        _fetch(mem_store.EPISODE_KEY, n_episodes, lang_filter_sym or None),
        retrieval_planner.analyze_query_structure(req.query) if (req.query.strip() and req.enable_planning) else asyncio.sleep(0),  # symbolic layer (index 5) — LLM call only when enable_planning=True
        _redis.get(_PINNED_SESSION_KEY),                       # handoff bridge (index 6)
    ]
    if req.include_tools and is_cap_query:
        gather_tasks.append(cap_mod.recall_tools(_redis, emb, k=6))
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
        try:
            results = await _redis.execute_command(*cmd)
        except Exception:
            return []
        procs = []
        idx = 0
        while idx + 2 < len(results):
            elem = results[idx]; score = results[idx+1]; raw = results[idx+2]
            idx += 3
            elem_str = elem.decode() if isinstance(elem, bytes) else elem
            if elem_str == "__seed__":
                continue
            try:
                attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
            except Exception:
                continue
            if attrs.get("_seed") or not attrs.get("task"):
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

    results = await asyncio.gather(*gather_tasks)
    persona_ctx  = results[0]
    env_ctx      = results[1]
    session_ctx  = results[2]
    facts_scene  = results[3]
    eps_scene    = results[4]
    query_struct = results[5] if isinstance(results[5], dict) else {}
    _pinned_raw  = results[6]
    last_session_summary = (
        _pinned_raw.decode() if isinstance(_pinned_raw, bytes) else _pinned_raw
    ) if _pinned_raw else None
    tools_raw    = results[7] if isinstance(results[7], list) else []
    procs_raw    = results[8] if isinstance(results[8], list) else []

    # Symbolic structured pass (AgentMem §3.3 extension):
    # If the query mentions specific persons or entities, do a targeted scan
    # for facts that explicitly contain those names — catches cases where
    # semantic similarity is low but the name is a direct match.
    sym_facts: list[dict] = []
    sym_entities = (query_struct.get("persons") or []) + (query_struct.get("entities") or [])
    if sym_entities:
        sym_tasks = []
        for ent in sym_entities[:3]:   # cap at 3 to bound latency
            ent_slug = ent.lower().replace(" ", "_")
            sym_tasks.append(
                mem_store.knn_search(_redis, mem_store.FACT_KEY, _encode(ent), k=3,
                                     filter_expr=lang_filter_sym or None)
            )
        sym_batches = await asyncio.gather(*sym_tasks)
        seen_sym = {f["content"] for f in facts_scene}
        for batch in sym_batches:
            for item in (batch if isinstance(batch, list) else []):
                if item["content"] not in seen_sym:
                    sym_facts.append(item)
                    seen_sym.add(item["content"])
        if sym_facts:
            log.debug(f"[recall] symbolic pass: +{len(sym_facts)} facts for entities {sym_entities[:3]}")

    # SimpleMem Section 3.3 — Intent-Aware Planning (opt-in)
    # If enable_planning=True, run LLM query planning to generate targeted queries,
    # then re-fetch for each query and merge. Adds ~300-600ms but improves recall
    # for complex multi-part questions.
    if req.enable_planning:
        targeted_queries = await retrieval_planner.plan_queries(req.query)
        if len(targeted_queries) > 1:
            # Run targeted queries in parallel (exclude original already fetched above)
            extra_queries = [q for q in targeted_queries if q != req.query]
            extra_tasks = [
                mem_store.knn_search(_redis, mem_store.FACT_KEY, _encode(q), n_facts,
                                     filter_expr=lang_filter_sym or None, bump_heat=True)
                for q in extra_queries
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
    facts_global = supp_results[0] if isinstance(supp_results[0], list) else []
    eps_global   = supp_results[1] if isinstance(supp_results[1], list) else []

    # Dynamic Weighted RRF (arXiv:2511.18194 wRRF):
    # Adjust per-source weights based on what the query structure reveals.
    # Entity/person queries → boost symbolic + BM25 passes (exact name matches).
    # Temporal queries → boost symbolic (timestamps stored there).
    # Pure semantic queries → reduce symbolic/BM25 weight (no metadata to match).
    _has_entities = bool(sym_entities)
    _has_temporal  = bool(query_struct.get("time_expression"))
    if _has_entities and _has_temporal:
        _rrf_weights = [0.8, 0.8, 1.4, 1.2]   # strong symbolic + BM25 boost
    elif _has_entities:
        _rrf_weights = [0.9, 0.9, 1.2, 1.1]   # moderate symbolic + BM25 boost
    elif _has_temporal:
        _rrf_weights = [0.9, 0.9, 1.2, 0.8]   # symbolic boost, light BM25
    else:
        _rrf_weights = [1.0, 1.0, 0.6, 0.9]   # semantic: moderate BM25, low symbolic

    # Memori BM25 pass (arXiv:2603.19935): exact-term matching over fact corpus.
    # Runs in O(1) (no I/O) — BM25 corpus is maintained in-memory and searched locally.
    # Particularly effective for entity names, dates, tool names, specific terms.
    bm25_facts = _bm25_index.search(req.query, k=n_facts)
    if bm25_facts:
        log.debug(f"[bm25] {len(bm25_facts)} hits for: {req.query[:60]!r}")

    def _merge(primary, supplement, symbolic, limit):
        p_ranked   = heat_mod.heat_rerank(primary)
        s_ranked   = heat_mod.heat_rerank(supplement) if supplement else []
        sym_ranked = heat_mod.heat_rerank(symbolic)   if symbolic   else []
        # BM25 hits are already scored by TF-IDF; pass as-is (no heat reranking)
        fused = _rrf_merge(
            [p_ranked, s_ranked, sym_ranked, bm25_facts],
            weights=_rrf_weights,
            limit=limit * 2,
        )
        return fused[:limit]

    facts    = _merge(facts_scene, facts_global, sym_facts,  n_facts)
    episodes = _merge(eps_scene,   eps_global,   [],         n_episodes)

    # SimpleMem Phase 3: lexical keyword boost (BM25-lite additive layer)
    # Note: BM25 above handles full TF-IDF scoring; this adds a lightweight keyword
    # overlap boost on top for any residual exact-match signal not captured by BM25.
    facts    = _keyword_boost(facts,    req.query)
    episodes = _keyword_boost(episodes, req.query, boost=0.03)

    # AgentMem advantage: importance-weighted reranking.
    # High-importance facts (rules, identity, preferences) surface above transient context.
    facts    = _importance_boost(facts)
    episodes = _importance_boost(episodes, weight=0.05)  # lighter for episodes

    # SimpleMem reflection loop (opt-in): check if results sufficient, if not re-fetch
    if req.enable_reflection and facts:
        is_sufficient = await retrieval_planner.check_sufficiency(req.query, facts)
        if not is_sufficient:
            # One additional targeted pass with a different angle
            extra_q = await retrieval_planner.plan_queries(
                f"Find additional information about: {req.query}"
            )
            if extra_q:
                reflection_emb = _encode(extra_q[0])
                reflection_facts = await mem_store.knn_search(
                    _redis, mem_store.FACT_KEY, reflection_emb,
                    k=max(3, n_facts // 2), bump_heat=True
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
                graph_mod.entity_recall(_redis, ent, emb, k=2)
                for ent in graph_ents
            ]
            graph_results = await asyncio.gather(*graph_tasks)
            seen_contents = {f["content"] for f in facts}
            for batch in graph_results:
                for gf in batch:
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
            bridge_facts = await graph_mod.find_bridge_nodes(_redis, facts, emb, k=3)
            if bridge_facts:
                seen_bridge = {f["content"] for f in facts}
                for bf in bridge_facts:
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
                all_tig = await _redis.hgetall(mem_store.TOOL_GRAPH_KEY)
                prefix  = f"{top_tool_elem}:"
                trans: dict[str, int] = {}
                for k_b, v_b in all_tig.items():
                    key_s = k_b.decode() if isinstance(k_b, bytes) else k_b
                    if key_s.startswith(prefix):
                        target = key_s[len(prefix):]
                        trans[target] = int(v_b)
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

    # SimpleMem Section 3.3 token-budget injection with XML wrapper
    # Skip pinned session summary if this session has already accumulated context
    # (session_ctx means we're mid-session, not starting fresh).
    _handoff = last_session_summary if not session_ctx else None
    prepend = _format_prepend(
        facts, episodes, session_ctx, persona_ctx,
        env_ctx=env_ctx_formatted, tool_ctx=tool_ctx,
        proc_ctx=proc_ctx or None,
        token_budget=req.token_budget,
        last_session_summary=_handoff,
    )
    ms = int((time.time() - t0) * 1000)

    planning_info = f" planning={req.enable_planning}" if req.enable_planning else ""
    log.info(f"[recall] session={req.session_id} lang={q_lang} domain={q_domain} "
             f"facts={len(facts)} ep={len(episodes)} tools={len(tools_raw)} procs={len(proc_ctx)}"
             f"{planning_info} {ms}ms")

    return {"prependContext": prepend if prepend else None, "latency_ms": ms}


@app.post("/store")
async def store_memory(req: StoreRequest):
    asyncio.create_task(_do_store(req.messages, req.session_id))
    return {"status": "queued"}


# ── Session Tier-1 endpoints (v0.7.0) ─────────────────────────────────────────

class CompressSessionRequest(BaseModel):
    session_id: str
    wait: bool = False   # True → sync (returns results); False → background

@app.post("/session/compress")
async def compress_session(req: CompressSessionRequest, background_tasks: BackgroundTasks):
    """
    Promote Tier 1 session KV → Tier 2 long-term vector memory.

    Call this from the agent_end hook when a session is finishing.
    The accumulated session summary is compressed, saved as an episode,
    and fact-extracted — then the session KV is deleted.

    Set wait=true to block until promotion completes (returns full result).
    """
    if req.wait:
        result = await _do_compress_session(req.session_id)
        return result
    background_tasks.add_task(_do_compress_session, req.session_id)
    return {"status": "queued", "session_id": req.session_id}


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """
    Inspect current Tier 1 session context for a given session_id.
    Useful for debugging what the agent has accumulated in this session.
    """
    ctx = await mem_store.get_session_context(_redis, session_id)
    return {
        "session_id": session_id,
        "context":    ctx,
        "length":     len(ctx) if ctx else 0,
        "tier":       "Tier 1 / Session KV",
    }


# ── Capability endpoints (v0.6.0) ─────────────────────────────────────────────

@app.post("/register-tools")
async def register_tools(req: RegisterToolsRequest, background_tasks: BackgroundTasks):
    """
    Register the agent's available tools/skills into mem:tools vectorset.
    Tools are embedded for semantic recall ("find tools that can X").
    Call this from before_agent_start with the full tool list.
    """
    if not req.tools:
        return {"status": "ok", "registered": 0}

    async def _do_register():
        count = 0
        for tool in req.tools:
            emb = _encode(f"{tool.name}: {tool.description}")
            await cap_mod.register_tool(
                _redis,
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


@app.post("/register-env")
async def register_env(req: EnvState):
    """
    Store current environment state in mem:env hash.
    Call this from before_agent_start with OS/shell/cwd/git/MCP info.
    """
    env_data: dict = {}
    if req.os:           env_data["os"]            = req.os
    if req.os_version:   env_data["os_version"]    = req.os_version
    if req.shell:        env_data["shell"]          = req.shell
    if req.cwd:          env_data["cwd"]            = req.cwd
    if req.git_repo:     env_data["git_repo"]       = req.git_repo
    if req.git_branch:   env_data["git_branch"]     = req.git_branch
    if req.runtime:      env_data["runtime"]        = req.runtime
    if req.agent_model:  env_data["agent_model"]    = req.agent_model
    if req.agent_version:env_data["agent_version"]  = req.agent_version
    if req.session_id:   env_data["session_id"]     = req.session_id
    if req.active_mcps:  env_data["active_mcps"]    = req.active_mcps
    if req.active_plugins: env_data["active_plugins"] = req.active_plugins
    if req.active_skills:  env_data["active_skills"]  = req.active_skills
    if req.extra:
        for k, v in req.extra.items():
            env_data[k] = str(v)

    await cap_mod.set_env(_redis, env_data)
    log.info(f"[env] registered env: {list(env_data.keys())}")
    return {"status": "ok", "fields": list(env_data.keys())}


@app.post("/recall-tools")
async def recall_tools_endpoint(req: RecallToolsRequest):
    """
    Semantic search over registered tools.
    Query: natural language description of needed capability.
    Returns: ranked list of matching tools with scores.

    Example: {"query": "search the filesystem for files"} →
             [{"name": "Glob", "description": "...", "score": 0.92}]
    """
    t0 = time.time()
    emb = _encode(req.query)
    tools = await cap_mod.recall_tools(
        _redis, emb,
        k=req.k,
        category_filter=req.category or None,
        source_filter=req.source or None,
    )
    ms = int((time.time() - t0) * 1000)
    return {"tools": tools, "count": len(tools), "latency_ms": ms}


@app.get("/capabilities")
async def get_capabilities():
    """
    Return the full agent capability manifest:
    - tools: all registered tools with metadata
    - env: current environment state
    - agent: agent self-model
    - stats: summary counts by category
    """
    return await cap_mod.get_capability_summary(_redis)


@app.post("/recall-procedures")
async def recall_procedures(req: RecallToolsRequest):
    """
    Semantic search over procedural memory (4th cognitive tier).
    Query with a task description → get back how-to procedures.

    Example: {"query": "how to search files by pattern", "k": 3}
    → [{"task": "find Python files recursively", "procedure": "Use Glob with **/*.py pattern"}]
    """
    t0 = time.time()
    emb = _encode(req.query)
    blob = emb.astype("float32").tobytes()
    cmd = ["VSIM", mem_store.PROC_KEY, "FP32", blob,
           "COUNT", req.k + 1, "WITHSCORES", "WITHATTRIBS"]
    try:
        results = await _redis.execute_command(*cmd)
    except Exception:
        return {"procedures": [], "latency_ms": 0}

    procs = []
    i = 0
    while i + 2 < len(results):
        elem  = results[i]
        score = results[i+1]
        raw   = results[i+2]
        i += 3
        elem_str = elem.decode() if isinstance(elem, bytes) else elem
        if elem_str == "__seed__":
            continue
        try:
            attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
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


@app.post("/store-procedure")
async def store_procedure(req: StoreProcedureRequest, background_tasks: BackgroundTasks):
    """
    Explicitly store a procedural memory (agent workflow / how-to pattern).
    The task description is embedded for semantic retrieval.
    Agents or the JS plugin can call this after completing a successful task.
    """
    async def _do_store_proc():
        emb = _encode(req.task)
        sc = scene_mod.detect(req.task)
        existing = await mem_store.knn_search(_redis, mem_store.PROC_KEY, emb, k=1)
        if existing and existing[0].get("score", 0.0) > 0.90:
            # Bump success count on near-duplicate
            elem = existing[0].get("_element")
            if elem:
                attrs = dict(existing[0].get("attrs", {}))
                attrs["success_count"] = attrs.get("success_count", 1) + 1
                try:
                    await _redis.execute_command(
                        "VSETATTR", mem_store.PROC_KEY, elem, json.dumps(attrs)
                    )
                except Exception:
                    pass
            return
        await mem_store.save_procedure(
            _redis,
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


# ── ToolMem feedback endpoint (v0.9.5) ────────────────────────────────────────

@app.post("/tool-feedback")
async def tool_feedback(req: ToolFeedbackRequest):
    """
    Record success/failure for a tool invocation (ToolMem arXiv:2510.06664).

    Called from PostToolUse hook after each tool use. Updates success_count /
    fail_count on the tool's mem:tools entry and refreshes capability_summary
    once enough data is available (≥5 uses).

    Example: {"tool_name": "web_search_prime", "success": true}
    """
    elem_key = req.tool_name.lower().replace(" ", "_").replace("/", "_")[:64]
    try:
        raw = await _redis.execute_command("VGETATTR", cap_mod.TOOL_KEY, elem_key)
    except Exception:
        return {"ok": False, "reason": "redis error"}
    if not raw:
        return {"ok": False, "reason": "tool not found"}

    try:
        attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
    except Exception:
        return {"ok": False, "reason": "parse error"}

    if req.success:
        attrs["success_count"] = attrs.get("success_count", 0) + 1
    else:
        attrs["fail_count"]    = attrs.get("fail_count", 0) + 1
    attrs["use_count"] = attrs.get("use_count", 0) + 1

    # Refresh capability_summary once ≥5 feedback data points exist
    total = attrs.get("success_count", 0) + attrs.get("fail_count", 0)
    if total >= 5:
        rate = attrs.get("success_count", 0) / total
        if   rate >= 0.85: quality = "reliable"
        elif rate >= 0.60: quality = "mostly reliable"
        elif rate >= 0.40: quality = "mixed"
        else:              quality = "often fails"
        attrs["capability_summary"] = f"{quality} ({attrs.get('success_count',0)}/{total}✓)"

    try:
        await _redis.execute_command("VSETATTR", cap_mod.TOOL_KEY, elem_key, json.dumps(attrs))
    except Exception as e:
        return {"ok": False, "reason": str(e)}

    log.debug(f"[tool-feedback] {elem_key} success={req.success} "
              f"s={attrs.get('success_count',0)} f={attrs.get('fail_count',0)}")
    return {
        "ok":           True,
        "tool":         elem_key,
        "success_count": attrs.get("success_count", 0),
        "fail_count":    attrs.get("fail_count", 0),
        "capability_summary": attrs.get("capability_summary", ""),
    }


# ── AutoTool TIG endpoints (v0.9.5) ───────────────────────────────────────────

@app.post("/record-tool-sequence")
async def record_tool_sequence(req: ToolSequenceRequest):
    """
    Record an ordered tool-use sequence into the Tool Inertia Graph (AutoTool arXiv:2511.14650).

    Called from Stop hook after each session. For each consecutive pair (A→B),
    increments HINCRBY on mem:tool_graph key "A:B". These transition counts
    are used at recall time to suggest likely-next tools.

    Example: {"sequence": ["Glob", "Read", "Edit", "Bash"]}
    → increments Glob:Read, Read:Edit, Edit:Bash
    """
    seq = [t.strip() for t in req.sequence if t.strip()]
    if len(seq) < 2:
        return {"ok": False, "transitions": 0}
    count = 0
    for i in range(len(seq) - 1):
        a = seq[i].lower().replace(" ", "_")[:64]
        b = seq[i + 1].lower().replace(" ", "_")[:64]
        if a == b:
            continue
        await _redis.hincrby(mem_store.TOOL_GRAPH_KEY, f"{a}:{b}", 1)
        count += 1
    log.debug(f"[tig] recorded {count} transitions for session={req.session_id}")
    return {"ok": True, "transitions": count}


@app.get("/tool-graph/{tool_name}")
async def tool_graph(tool_name: str, k: int = 5):
    """
    Return the top-k outgoing transitions from tool_name in the Tool Inertia Graph.

    Example: GET /tool-graph/glob → [{"next": "read", "count": 42, "prob": 0.72}]
    """
    elem_key = tool_name.lower().replace(" ", "_")[:64]
    try:
        all_entries = await _redis.hgetall(mem_store.TOOL_GRAPH_KEY)
    except Exception:
        return {"tool": tool_name, "transitions": []}
    prefix = f"{elem_key}:"
    trans: dict[str, int] = {}
    for k_b, v_b in all_entries.items():
        key_s = k_b.decode() if isinstance(k_b, bytes) else k_b
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


# ── AWO meta-tool detection (v0.9.6) ──────────────────────────────────────────

@app.post("/tool-graph/detect-meta-tools")
async def detect_meta_tools(threshold: int = 5, background_tasks: BackgroundTasks = None):
    """
    AWO meta-tool detection (Autonomous Workflow Optimization arXiv:2601.22037).

    Scans the Tool Inertia Graph for high-frequency 2-hop chains A→B→C
    where both A:B and B:C have count ≥ threshold. Each discovered chain is
    auto-synthesized into a composite procedure in mem:procedures so future
    recall surfaces it as a recommended workflow pattern.

    Example: Glob→Read (28) + Read→Edit (23) → synthesize "Glob-Read-Edit pattern"
    Returns: {"detected": N, "chains": [...], "new_procedures": M}
    """
    try:
        all_entries = await _redis.hgetall(mem_store.TOOL_GRAPH_KEY)
    except Exception:
        return {"detected": 0, "chains": [], "new_procedures": 0}

    if not all_entries:
        return {"detected": 0, "chains": [], "new_procedures": 0}

    # Build adjacency: from_tool → {to_tool: count}
    adj: dict[str, dict[str, int]] = {}
    for k_b, v_b in all_entries.items():
        key_s = k_b.decode() if isinstance(k_b, bytes) else k_b
        val   = int(v_b) if v_b else 0
        if ":" not in key_s:
            continue
        a, b = key_s.split(":", 1)
        adj.setdefault(a, {})[b] = val

    # Find 2-hop chains: A→B (≥ threshold) + B→C (≥ threshold), A≠C
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
                    "strength": min(ab_count, bc_count),  # bottleneck strength
                })

    # Sort by bottleneck strength descending
    chains.sort(key=lambda x: x["strength"], reverse=True)

    if not chains:
        return {"detected": 0, "chains": [], "new_procedures": 0}

    async def _synthesize_chains(chains_to_add: list[dict]):
        new_count = 0
        for ch in chains_to_add[:20]:   # cap at 20 new meta-tools per run
            a, b, c = ch["chain"]
            task = f"[meta-tool] {a} → {b} → {c} workflow"
            procedure = (
                f"High-frequency tool chain discovered by AWO analysis "
                f"(TIG count: {a}→{b}={ch['ab_count']}, {b}→{c}={ch['bc_count']}).\n"
                f"1. Use {a}\n2. Use {b}\n3. Use {c}"
            )
            emb = _encode(task)
            # Check for near-duplicate before storing
            existing = await mem_store.knn_search(_redis, mem_store.PROC_KEY, emb, k=1)
            if existing and existing[0].get("score", 0.0) > 0.90:
                continue
            await mem_store.save_procedure(
                _redis,
                task=task,
                procedure=procedure,
                embedding=emb,
                tools_used=[a, b, c],
                domain="meta",
                language="en",
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
        "chains":         chains[:10],   # return top-10 for inspection
        "new_procedures": new_procs,
        "synthesis":      synthesis_status,
        "threshold":      threshold,
    }


# ── MACLA procedure feedback endpoint (v0.9.5) ────────────────────────────────

@app.post("/procedure-feedback")
async def procedure_feedback(req: ProcedureFeedbackRequest):
    """
    Record success/failure for a procedure retrieval (MACLA arXiv:2512.18950).

    Finds the best matching procedure by semantic similarity to task_prefix,
    then updates its fail_count. (success_count is already bumped by /store-procedure.)
    Used by agents after a recalled procedure was followed and succeeded/failed.

    Example: {"task_prefix": "how to search files by pattern", "success": false}
    """
    emb = _encode(req.task_prefix[:80])
    blob = emb.astype("float32").tobytes()
    try:
        results = await _redis.execute_command(
            "VSIM", mem_store.PROC_KEY, "FP32", blob,
            "COUNT", 3, "WITHSCORES", "WITHATTRIBS"
        )
    except Exception:
        return {"ok": False, "reason": "redis error"}

    best_elem = None
    best_score = 0.0
    best_attrs: dict = {}
    idx = 0
    while idx + 2 < len(results):
        elem  = results[idx]; score = results[idx+1]; raw = results[idx+2]
        idx += 3
        elem_str = elem.decode() if isinstance(elem, bytes) else elem
        if elem_str == "__seed__":
            continue
        try:
            attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        except Exception:
            continue
        if attrs.get("_seed") or not attrs.get("task"):
            continue
        s = float(score) if score else 0.0
        if s > best_score:
            best_score = s; best_elem = elem_str; best_attrs = attrs

    if not best_elem or best_score < 0.50:
        return {"ok": False, "reason": "no matching procedure found", "best_score": best_score}

    if req.success:
        best_attrs["success_count"] = best_attrs.get("success_count", 0) + 1
    else:
        best_attrs["fail_count"]    = best_attrs.get("fail_count", 0) + 1

    try:
        await _redis.execute_command("VSETATTR", mem_store.PROC_KEY, best_elem, json.dumps(best_attrs))
    except Exception as e:
        return {"ok": False, "reason": str(e)}

    log.debug(f"[proc-feedback] {best_elem[:40]} success={req.success} score={best_score:.2f}")
    return {
        "ok":           True,
        "matched_task": best_attrs.get("task", "")[:60],
        "score":        best_score,
        "success_count": best_attrs.get("success_count", 0),
        "fail_count":    best_attrs.get("fail_count", 0),
    }


@app.get("/tool-procedures/{tool_name}")
async def tool_procedures(tool_name: str):
    """
    Return all procedures that use a given tool (reverse index lookup).

    Uses mem:proc_by_tool:<tool> Redis Sets populated by save_procedure()
    and the /proc-backfill-index endpoint.

    Example: GET /tool-procedures/bash
    → [{"uid": "...", "task": "...", "tools_used": [...], "success_count": N}]
    """
    uids = await mem_store.get_procs_for_tool(_redis, tool_name)
    if not uids:
        return {"tool": tool_name, "procedures": [], "count": 0}

    procs = []
    for uid in uids:
        try:
            raw = await _redis.execute_command("VGETATTR", mem_store.PROC_KEY, uid)
        except Exception:
            continue
        if not raw:
            continue
        try:
            attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
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


@app.post("/proc-backfill-index")
async def proc_backfill_index(background_tasks: BackgroundTasks):
    """
    Backfill mem:proc_by_tool reverse index for all existing procedures.

    Run once after upgrading to v0.9.6 to populate the reverse index for
    procedures that were stored before link_proc_to_tools() was added.
    Idempotent — SADD is a no-op for already-present members.
    """
    async def _do_backfill():
        procs = await mem_store.scan_all_procedures(_redis)
        count = 0
        for p in procs:
            uid   = p.get("uid", "")
            tools = p.get("tools_used", [])
            if uid and tools:
                await mem_store.link_proc_to_tools(_redis, uid, tools)
                count += len(tools)
        log.info(f"[backfill] reverse-indexed {len(procs)} procedures, {count} tool→proc links")
        return len(procs), count

    background_tasks.add_task(_do_backfill)
    return {"status": "queued", "message": "backfilling proc_by_tool reverse index in background"}


@app.get("/capabilities/context")
async def get_capability_context():
    """
    Return environment context as formatted string (same as what /recall injects).
    Useful for debugging what an agent sees about its environment.
    """
    env_ctx   = await cap_mod.get_env_context(_redis)
    persona   = await persona_mod.get_context(_redis)
    agent     = await cap_mod.get_agent(_redis)
    all_tools = await cap_mod.list_all_tools(_redis)

    # Format tools by category
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


# ── Consolidation (SimpleMem-inspired recursive merging) ─────────────────────

async def _do_consolidate(
    similarity_threshold: float = 0.85,
    temporal_lambda: float = 0.03,      # temporal decay: λ=0.03 → 23-day half-life
) -> dict:
    """
    SimpleMem 3-phase consolidation (Section 3.2):
      Phase 1 — Decay:  importance × 0.9 for entries older than 90 days
      Phase 2 — Merge:  cluster near-duplicates (cosine×temporal ≥ threshold),
                        LLM-merge into keeper, soft-delete losers via superseded_by
      Phase 3 — Prune:  soft-delete any active entry with importance < 0.05
    """
    t0 = time.time()
    now_ms = int(time.time() * 1000)
    NINETY_DAYS_MS = 90 * 86_400_000

    card = await _redis.execute_command("VCARD", mem_store.FACT_KEY)
    if not card or int(card) <= 1:
        return {"merged": 0, "decayed": 0, "pruned": 0, "ms": 0}

    try:
        seed = np.zeros(mem_store.DIMS, dtype=np.float32)
        results = await _redis.execute_command(
            "VSIM", mem_store.FACT_KEY, "FP32", seed.tobytes(),
            "COUNT", min(int(card), 500), "WITHSCORES", "WITHATTRIBS"
        )
    except Exception as e:
        log.warning(f"[consolidate] failed to scan facts: {e}")
        return {"error": str(e)}

    # Parse only active (non-superseded) facts
    all_facts = []
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
        if attrs.get("superseded_by"):   # skip already-superseded entries
            continue
        all_facts.append({"element": elem_str, "attrs": attrs})

    if len(all_facts) < 2:
        return {"merged": 0, "decayed": 0, "pruned": 0, "total": len(all_facts), "ms": 0}

    # ── Phase 1: Decay ─────────────────────────────────────────────────────────
    # Entries older than 90 days lose 10% importance each consolidation run.
    decayed_count = 0
    for fact in all_facts:
        ts = fact["attrs"].get("ts", now_ms)
        if (now_ms - ts) > NINETY_DAYS_MS:
            old_imp = fact["attrs"].get("importance", 0.5)
            new_imp = round(old_imp * 0.9, 4)
            fact["attrs"]["importance"] = new_imp
            try:
                await _redis.execute_command(
                    "VSETATTR", mem_store.FACT_KEY, fact["element"],
                    json.dumps(fact["attrs"])
                )
                decayed_count += 1
            except Exception as e:
                log.debug(f"[consolidate] decay VSETATTR failed: {e}")

    # ── Phase 2: Merge ─────────────────────────────────────────────────────────
    # affinity = cosine_similarity × exp(−λ × |days_between|)
    # Cluster losers are soft-deleted (superseded_by = keeper_element), never VREM'd.
    merged_count = 0
    superseded_elements: set[str] = set()

    for fact in all_facts:
        if fact["element"] in superseded_elements:
            continue

        f_emb = _encode(fact["attrs"]["content"])
        similar = await mem_store.knn_search(
            _redis, mem_store.FACT_KEY, f_emb, k=5
        )

        cluster = [fact]
        fact_ts = fact["attrs"].get("ts", 0)

        for s in similar:
            if s["_element"] == fact["element"]:
                continue
            if s["_element"] in superseded_elements:
                continue

            cosine_sim = s.get("score", 0.0)
            s_ts = 0
            for f2 in all_facts:
                if f2["element"] == s["_element"]:
                    s_ts = f2["attrs"].get("ts", 0)
                    break

            days_between = abs(fact_ts - s_ts) / 86_400_000
            temporal_factor = math.exp(-temporal_lambda * days_between)
            affinity = cosine_sim * temporal_factor

            if affinity >= similarity_threshold:
                for f2 in all_facts:
                    if f2["element"] == s["_element"]:
                        cluster.append(f2)
                        break

        if len(cluster) < 2:
            continue

        contents = [c["attrs"]["content"] for c in cluster]
        merged_content = await _llm_merge_facts(contents)
        if not merged_content:
            continue

        # Keeper = highest importance in cluster (aligned with SimpleMem's winner
        # selection policy: importance beats recency). Ties broken by newer timestamp.
        cluster.sort(
            key=lambda c: (c["attrs"].get("importance", 0.5), c["attrs"].get("ts", 0)),
            reverse=True
        )
        keeper = cluster[0]
        keeper_element = keeper["element"]

        new_attrs = dict(keeper["attrs"])
        new_attrs["content"] = merged_content[:500]
        new_attrs["access_count"] = max(c["attrs"].get("access_count", 0) for c in cluster)
        new_attrs["importance"] = max(c["attrs"].get("importance", 0.5) for c in cluster)
        new_attrs["consolidated_from"] = len(cluster)

        all_kw: set[str] = set()
        all_persons: set[str] = set()
        all_entities: set[str] = set()
        for c in cluster:
            all_kw.update(c["attrs"].get("keywords", []))
            all_persons.update(c["attrs"].get("persons", []))
            all_entities.update(c["attrs"].get("entities", []))
        if all_kw:
            new_attrs["keywords"] = list(all_kw)[:10]
        if all_persons:
            new_attrs["persons"] = list(all_persons)[:5]
        if all_entities:
            new_attrs["entities"] = list(all_entities)[:5]

        m_emb = _encode(merged_content)
        try:
            await _redis.execute_command(
                "VADD", mem_store.FACT_KEY, "FP32", m_emb.tobytes(),
                keeper_element, "SETATTR", json.dumps(new_attrs)
            )
        except Exception as e:
            log.warning(f"[consolidate] failed to update keeper: {e}")
            continue

        # Soft-delete losers: mark superseded_by = keeper_element (never VREM)
        for c in cluster[1:]:
            if c["element"] not in superseded_elements:
                await mem_store.soft_delete_fact(_redis, c["element"], keeper_element)
                superseded_elements.add(c["element"])

        merged_count += 1

    # ── Phase 3: Prune ─────────────────────────────────────────────────────────
    # Soft-delete any remaining active entry whose importance has decayed below 0.05.
    pruned_count = 0
    for fact in all_facts:
        if fact["element"] in superseded_elements:
            continue
        if fact["attrs"].get("importance", 1.0) < 0.05:
            await mem_store.soft_delete_fact(_redis, fact["element"], "pruned")
            superseded_elements.add(fact["element"])
            pruned_count += 1

    ms = int((time.time() - t0) * 1000)
    log.info(
        f"[consolidate] decayed={decayed_count} merged={merged_count} "
        f"pruned={pruned_count} {ms}ms"
    )
    return {
        "decayed": decayed_count,
        "merged": merged_count,
        "pruned": pruned_count,
        "total_before": len(all_facts),
        "total_after": len(all_facts) - len(superseded_elements),
        "ms": ms,
    }


async def _do_hard_prune() -> dict:
    """
    Physically VREM entries that have been soft-deleted for > 7 days,
    and stale episodes (>180 days old, never accessed).

    Soft-delete (superseded_by != "") is the SimpleMem audit trail.
    Hard-delete (VREM) is the capacity management pass that actually
    reduces vectorset size and keeps HNSW index fresh.

    Returns counts of entries removed from each vectorset.
    """
    t0 = time.time()
    now_ms = int(time.time() * 1000)
    SEVEN_DAYS_MS   = 7  * 86_400_000
    SIX_MONTHS_MS   = 180 * 86_400_000

    removed_facts = 0
    removed_eps   = 0

    async def _hard_prune_vset(vset_key: str, max_scan: int = 2000) -> int:
        card = await _redis.execute_command("VCARD", vset_key)
        if not card or int(card) <= 1:
            return 0
        seed = np.zeros(mem_store.DIMS, dtype=np.float32)
        try:
            results = await _redis.execute_command(
                "VSIM", vset_key, "FP32", seed.tobytes(),
                "COUNT", min(int(card), max_scan), "WITHSCORES", "WITHATTRIBS"
            )
        except Exception as e:
            log.warning(f"[hard_prune] scan failed for {vset_key}: {e}")
            return 0

        to_remove: list[str] = []
        i = 0
        while i + 2 < len(results):
            elem = results[i]; raw = results[i + 2]; i += 3
            elem_str = elem.decode() if isinstance(elem, bytes) else elem
            if elem_str == "__seed__":
                continue
            try:
                attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
            except Exception:
                continue
            if attrs.get("_seed"):
                continue

            ts = attrs.get("ts", now_ms)
            age_ms = now_ms - ts

            # Hard-delete soft-deleted entries older than 7 days
            superseded = attrs.get("superseded_by", "")
            if superseded and age_ms > SEVEN_DAYS_MS:
                to_remove.append(elem_str)
                continue

            # Hard-delete stale episodes: 180+ days old, never recalled
            if vset_key == mem_store.EPISODE_KEY:
                if age_ms > SIX_MONTHS_MS and attrs.get("access_count", 0) == 0:
                    to_remove.append(elem_str)

        removed = 0
        for elem_str in to_remove:
            try:
                await _redis.execute_command("VREM", vset_key, elem_str)
                removed += 1
            except Exception:
                pass
        return removed

    removed_facts = await _hard_prune_vset(mem_store.FACT_KEY)
    removed_eps   = await _hard_prune_vset(mem_store.EPISODE_KEY)

    ms = int((time.time() - t0) * 1000)
    log.info(f"[hard_prune] removed facts={removed_facts} episodes={removed_eps} {ms}ms")
    return {"removed_facts": removed_facts, "removed_episodes": removed_eps, "ms": ms}


async def _llm_merge_facts(contents: list[str]) -> str | None:
    """Use role-based routing to merge multiple similar facts into one consolidated fact."""
    _AISERV_OAI = "http://127.0.0.1:4000/v1/chat/completions"
    _AISERV_KEY = "sk-aiserv-local-dev"

    # Use live health matrix — no hardcoded dead models.
    model, _ = await extractor._resolve_nlp_model()

    facts_text = "\n".join(f"- {c}" for c in contents)
    prompt = (
        f"以下是关于同一主题的多条记忆，请合并为一条完整、准确的事实陈述。"
        f"保留所有不同的细节，去除重复。输出仅包含合并后的一句话，不要解释。\n\n{facts_text}"
    )

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                _AISERV_OAI,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 150,
                    "temperature": 0.1,
                },
                headers={"Authorization": f"Bearer {_AISERV_KEY}"},
            )
            if resp.status_code == 200:
                asyncio.create_task(extractor._report_quality(model, +1))
                return resp.json()["choices"][0]["message"]["content"].strip()
    except httpx.TimeoutException:
        log.warning("[consolidate] %s timed out", model)
        asyncio.create_task(extractor._report_quality(model, -1))
    except Exception as e:
        log.warning(f"[consolidate] LLM merge failed: {e}")
        asyncio.create_task(extractor._report_quality(model, -1))

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


@app.post("/consolidate/hard-prune")
async def hard_prune(background_tasks: BackgroundTasks):
    """
    Physical VREM of soft-deleted entries (>7 days) and stale episodes (>180 days).
    Runs automatically every 24h; use this to trigger on demand.
    """
    background_tasks.add_task(_do_hard_prune)
    return {"status": "hard_prune_queued"}


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


# ── Knowledge Graph endpoints (v0.9.0) ────────────────────────────────────────

class GraphRecallRequest(BaseModel):
    entity: str
    k: int = 5


@app.get("/graph/{entity}")
async def graph_neighbors(entity: str):
    """
    Return entities related to *entity* in the knowledge graph, with connection counts.

    Example: GET /graph/bob  →  related entities Bob co-occurs with in stored facts.
    """
    neighbors = await graph_mod.get_entity_neighbors_with_counts(_redis, entity)
    return {
        "entity":    entity,
        "neighbors": neighbors,
        "count":     len(neighbors),
    }


@app.post("/graph/recall")
async def graph_recall(req: GraphRecallRequest):
    """
    Retrieve facts from the knowledge graph neighbourhood of an entity.

    Example: {"entity": "Bob", "k": 5}
    → facts that mention Bob's related entities.
    """
    emb = _encode(req.entity)
    facts = await graph_mod.entity_recall(_redis, req.entity, emb, k=req.k)
    return {
        "entity": req.entity,
        "facts":  [{"content": f["content"], "score": f.get("score", 0.0)} for f in facts],
        "count":  len(facts),
    }


@app.get("/graph/stats")
async def graph_stats_endpoint():
    """Return knowledge graph statistics: node count and edge count."""
    return await graph_mod.graph_stats(_redis)


# ── Config endpoint (v0.9.0) ──────────────────────────────────────────────────

@app.get("/config")
async def get_config():
    """Return current service configuration including auto-consolidation settings."""
    return {
        "version":                 "0.9.2",
        "auto_consolidate_every":  _AUTO_CONSOLIDATE_EVERY,
        "stores_since_last":       _stores_since_consolidation,
        "periodic_interval_s":     3600,
        "entropy_gate_threshold":  0.35,
        "dedup_threshold":         0.95,
        "consolidate_threshold":   0.85,
        "session_ttl_s":           14400,
    }
