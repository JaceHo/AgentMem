"""
MCP Server for AgentMem — v1.3.0

Exposes memory service tools via the Model Context Protocol (MCP).
Supports stdio, SSE, and Streamable HTTP transports.

Transport modes
---------------
Stdio (default — run standalone):
    python mcp_server.py
    python mcp_server.py --stdio

HTTP Streamable (MCP 2025-03-26 spec, preferred for HTTP agents):
    python mcp_server.py --http [--port 18801]

SSE (legacy HTTP, for older clients):
    python mcp_server.py --sse [--port 18801]

Embedding in FastAPI (used by main.py):
    from mcp_server import build_mcp_app, build_sse_app
    app.mount("/mcp",     build_mcp_app())
    app.mount("/mcp/sse", build_sse_app())

Per-agent config snippets
--------------------------
Claude Code (.mcp.json):
    {"mcpServers": {"agentmem": {"type": "http", "url": "http://localhost:18800/mcp"}}}

Cursor (.cursor/mcp.json) — remote HTTP, no "type" field (see cursor.com/docs/context/mcp):
    {"mcpServers": {"agentmem": {"url": "http://127.0.0.1:18800/mcp"}}}

Windsurf (~/.codeium/windsurf/mcp_config.json):
    {"mcpServers": {"agentmem": {"serverUrl": "http://localhost:18800/mcp/sse"}}}

GitHub Copilot (.vscode/mcp.json):
    {"servers": {"agentmem": {"type": "http", "url": "http://localhost:18800/mcp"}}}

Zed (~/.config/zed/settings.json):
    {"context_servers": {"agentmem": {"url": "http://localhost:18800/mcp/sse"}}}

Opencode (opencode.json):
    {"mcp": {"agentmem": {"type": "remote", "url": "http://localhost:18800/mcp/sse"}}}

Stdio (all others — Cline, Continue, Kilo Code, Kiro, etc.):
    {"mcpServers": {"agentmem": {"command": "python", "args": ["/path/to/agentmem/mcp_server.py"]}}}

Tools exposed (54 total)
------------------------
Core recall/store (9):
  recall_memory        — retrieve relevant memories for a query
  store_memory         — save a conversation turn to long-term memory
  recall_tools         — find agent tools by capability description
  recall_procedures    — find how-to workflows by task description
  store_procedure      — save a successful workflow
  get_stats            — memory counts across all tiers
  compress_session     — promote session to long-term memory
  batch_recall_memory  — recall for multiple queries in parallel
  batch_store_memory   — store multiple turns in parallel

Knowledge CRUD (5):
  remember_fact        — directly store a fact with category + concepts
  forget_memory        — fuzzy-delete memories by semantic query
  delete_memory        — hard-delete a specific fact by element_id
  pin_memory           — pin a fact permanently (importance=1.0, never pruned)
  feedback_memory      — rate a memory 1-5 stars (adjusts importance)

Search & Retrieval (4):
  search_memory        — compact vector search across facts + episodes
  smart_search         — wRRF fused multi-tier search with scores
  timeline             — chronological episode retrieval
  answer_from_memory   — LLM-extract a concise answer from recalled context

Graph (5):
  recall_graph         — retrieve facts via knowledge graph entity neighbourhood
  graph_neighbors      — direct neighbours of an entity in the graph
  graph_stats          — graph node/edge/cluster counts
  add_graph_edge       — manually add a typed relationship edge
  traverse_graph       — multi-hop path traversal from an entity

Memory introspection (4):
  get_memory_metadata  — full metadata for a fact (confidence, pins, ratings)
  get_memory_confidence — effective confidence with Ebbinghaus decay applied
  reinforce_memory     — boost a fact's importance + reset decay clock
  lifecycle_stats      — confidence distribution, supersession counts

Session (4):
  get_session          — session summary + accumulated context
  list_sessions        — recent sessions with metadata
  compact_session      — mid-session compress of Tier 1 KV
  get_context          — formatted cross-session context for a session

Persona & capabilities (3):
  get_profile          — user persona, rules, and preferences
  get_capabilities     — registered tools and skills with descriptions
  get_config           — current service configuration and feature flags

Procedures & Tools (4):
  procedure_feedback   — record procedure success/fail for MACLA Beta scoring
  get_tool_procedures  — O(1) reverse lookup: procedures associated with a tool
  get_tool_graph       — AutoTool TIG next-tool hints for a given tool
  detect_meta_tools    — AWO: scan TIG for 2-hop chains → auto-create meta-tools

Lifecycle & admin (4):
  consolidate          — trigger 3-phase consolidation pipeline (decay/merge/prune)
  export_memories      — export all facts + episodes as JSON
  find_patterns        — extract recurring patterns from episodic memory
  enrich_memory        — enrich a memory with additional context/keywords

Raw tier access (5):
  get_observations     — list raw observations with optional session filter
  list_raw_memories    — list all stored memories (facts + episodes)
  get_semantic_tier    — raw semantic memory entries
  get_procedural_tier  — raw procedural memory entries
  get_relations        — entity relation triples from the knowledge graph

Batch operations (3):
  batch_search         — parallel smart-search for multiple queries
  batch_recall_procedures — parallel procedure recall for multiple task descriptions
  batch_store_procedure — store multiple procedures in parallel

Misc (4):
  crystallize          — distill session summaries into high-importance facts
  get_health           — service health check with latency
  record_tool_sequence — record a tool usage sequence for TIG training
  get_system_prompt    — get current memory system prompt for context injection
"""

from __future__ import annotations

import asyncio
import json as _json
import os

import httpx

from mcp.server.fastmcp import FastMCP

BASE_URL = os.getenv("AGENTMEMORY_URL", "http://127.0.0.1:18800").rstrip("/")
TIMEOUT  = 12.0

mcp = FastMCP(
    "agentmem",
    json_response=True,
    instructions=(
        "AgentMem gives you persistent, cross-session memory. "
        "Call recall_memory at the start of each conversation with the user's question as the query. "
        "Call store_memory after each meaningful exchange. "
        "Call compress_session when the conversation ends. "
        "Use smart_search for targeted lookups, recall_graph for entity context, "
        "and get_profile to retrieve user preferences."
    ),
)


# ── HTTP helpers ──────────────────────────────────────────────────────────────

# Shared client — avoids creating a new connection pool per request.
# httpx.AsyncClient with connection pooling is significantly faster for
# repeated calls (MCP tools call the backend frequently).
_shared_client: httpx.AsyncClient | None = None
_client_lock = asyncio.Lock()


async def _get_client() -> httpx.AsyncClient:
    global _shared_client
    if _shared_client is not None and not _shared_client.is_closed:
        return _shared_client
    async with _client_lock:
        if _shared_client is not None and not _shared_client.is_closed:
            return _shared_client
        _shared_client = httpx.AsyncClient(
            timeout=httpx.Timeout(TIMEOUT),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=10, keepalive_expiry=30),
        )
    return _shared_client


async def close_client() -> None:
    """Close the shared httpx client (called during app shutdown)."""
    global _shared_client
    async with _client_lock:
        if _shared_client is not None and not _shared_client.is_closed:
            await _shared_client.aclose()
        _shared_client = None


async def _post(path: str, payload: dict) -> dict:
    client = await _get_client()
    resp = await client.post(f"{BASE_URL}{path}", json=payload)
    resp.raise_for_status()
    return resp.json()


async def _get(path: str, params: dict | None = None) -> dict:
    client = await _get_client()
    resp = await client.get(f"{BASE_URL}{path}", params=params or {})
    resp.raise_for_status()
    return resp.json()


async def _delete(path: str) -> dict:
    client = await _get_client()
    resp = await client.delete(f"{BASE_URL}{path}")
    resp.raise_for_status()
    return resp.json()


def _dead() -> str:
    return (
        f"Error: AgentMem service is not running at {BASE_URL}.\n"
        "Start it with: agentmem.sh start"
    )


def _fmt(d: dict, indent: int = 2) -> str:
    return _json.dumps(d, indent=indent, ensure_ascii=False, default=str)


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 1 — Core recall / store (9 tools)
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def recall_memory(
    query: str,
    session_id: str = "",
    memory_limit_number: int = 6,
    include_tools: bool = True,
    include_graph: bool = False,
    enable_hyde: bool = True,
) -> str:
    """
    Retrieve relevant long-term memories, past conversation context,
    user preferences, and learned facts for a given query.
    Call at the start of a conversation or when you need background knowledge.
    HyDE + MIRIX active retrieval are enabled by default for higher recall quality.
    """
    try:
        result = await _post("/recall", {
            "query": query,
            "session_id": session_id,
            "memory_limit_number": memory_limit_number,
            "include_tools": include_tools,
            "include_graph": include_graph,
            "enable_hyde": enable_hyde,
        })
        ctx = result.get("prependContext") or "(no relevant memories found)"
        return f"Memory context ({result.get('latency_ms', '?')}ms):\n\n{ctx}"
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def store_memory(
    messages: list[dict],
    session_id: str = "",
) -> str:
    """
    Save a conversation turn or important information to long-term memory.
    Provide messages as a list with role ('user'/'assistant'/'system') and content.
    Call after each meaningful exchange to build persistent memory.
    """
    try:
        result = await _post("/store", {"messages": messages, "session_id": session_id})
        return f"Memory stored: {result.get('status', 'ok')}"
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def recall_tools(
    query: str,
    k: int = 5,
    category: str = "",
    source: str = "",
) -> str:
    """
    Find registered agent tools and skills by semantic description.
    Use this to discover what capabilities are available for a task.
    """
    try:
        result = await _post("/recall-tools", {
            "query": query, "k": k, "category": category, "source": source,
        })
        tools = result.get("tools", [])
        if not tools:
            return "No matching tools found."
        lines = [f"Found {len(tools)} tool(s):"]
        for t in tools:
            lines.append(f"  • {t['name']} ({t.get('score', 0):.2f}): {t.get('description', '')[:100]}")
        return "\n".join(lines)
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def recall_procedures(
    query: str,
    k: int = 3,
) -> str:
    """
    Find how-to workflows and successful task patterns by description.
    MACLA Beta scoring re-ranks by success rate × similarity.
    Use this when approaching a task to check if there's a known procedure.
    """
    try:
        result = await _post("/recall-procedures", {"query": query, "k": k})
        procs = result.get("procedures", [])
        if not procs:
            return "No matching procedures found."
        lines = [f"Found {len(procs)} procedure(s):"]
        for p in procs:
            lines.append(f"\n**{p['task']}** (score={p.get('score', 0):.2f})\n{p.get('procedure', '')[:300]}")
        return "\n".join(lines)
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def store_procedure(
    task: str,
    procedure: str,
    tools_used: list[str] = [],
    domain: str = "",
    session_id: str = "",
) -> str:
    """
    Save a successful workflow or how-to pattern to procedural memory.
    Call after completing a multi-step task to remember the approach.
    """
    try:
        result = await _post("/store-procedure", {
            "task": task, "procedure": procedure,
            "tools_used": tools_used, "domain": domain, "session_id": session_id,
        })
        return f"Procedure stored: {result.get('status', 'ok')}"
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def get_stats() -> str:
    """Get memory counts across all tiers (episodes, facts, procedures, tools, persona)."""
    try:
        result = await _get("/stats")
        lines = ["Memory statistics:"]
        for k, v in result.items():
            if k != "writer":
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def compress_session(session_id: str) -> str:
    """
    Promote the current session's accumulated context to long-term memory.
    Extracts facts and episodes from the session KV and stores them permanently.
    Call at the end of a conversation to crystallise session knowledge.
    """
    try:
        result = await _post("/session/compress", {"session_id": session_id, "wait": True})
        return (
            f"Session compressed: ep_saved={result.get('ep_saved', 0)}, "
            f"facts_saved={result.get('facts_saved', 0)}, "
            f"status={result.get('status', '?')}"
        )
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def batch_recall_memory(
    queries: list[str],
    session_id: str = "",
    memory_limit_number: int = 4,
) -> str:
    """
    Retrieve relevant memories for multiple queries in parallel.
    More efficient than calling recall_memory N times. Max 5 queries per call.
    """
    try:
        qs = queries[:5]
        tasks = [
            _post("/recall", {
                "query": q, "session_id": session_id,
                "memory_limit_number": memory_limit_number,
                "enable_hyde": True,
            })
            for q in qs
        ]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        contexts: list[str] = []
        for q, res in zip(qs, results_list):
            if isinstance(res, Exception):
                continue
            ctx = res.get("prependContext")
            if ctx:
                contexts.append(f"[Query: {q}]\n{ctx}")
        if contexts:
            return "Batch memory context:\n\n" + "\n\n---\n\n".join(contexts)
        return "(no relevant memories found for any query)"
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def batch_store_memory(batches: list[dict]) -> str:
    """
    Save multiple conversation turns to long-term memory in parallel.
    Each batch is a dict with 'messages' (list) and optional 'session_id'.
    More efficient than calling store_memory N times.
    """
    try:
        tasks = [
            _post("/store", {
                "messages": b.get("messages", []),
                "session_id": b.get("session_id", ""),
            })
            for b in batches
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
        return f"Batch stored: {len(batches)} conversation turn(s) queued"
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 2 — Knowledge CRUD (5 tools)
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def remember_fact(
    content: str,
    category: str = "fact",
    concepts: str = "",
) -> str:
    """
    Directly store a specific fact with a category and optional concept keywords.
    Use for explicit facts the user states ('remember that I always use bun').
    Category options: fact, rule, preference, identity, decision, discovery.
    concepts: comma-separated keywords for BM25 retrieval (e.g. 'bun,npm,javascript').
    """
    try:
        result = await _post("/remember", {
            "content": content, "type": category, "concepts": concepts,
        })
        return f"Fact remembered: id={result.get('id', '?')}, status={result.get('status', 'ok')}"
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def forget_memory(
    query: str,
    limit: int = 3,
    dry_run: bool = True,
) -> str:
    """
    Fuzzy-delete memories that are semantically similar to the query.
    Use dry_run=True first to preview what would be deleted.
    Use dry_run=False to actually remove the memories.
    """
    try:
        result = await _post("/forget", {
            "query": query, "limit": limit, "dry_run": dry_run,
        })
        if dry_run:
            would_delete = result.get("would_delete", 0)
            previews = result.get("memories", [])
            lines = [f"Dry run: would delete {would_delete} memory/ies matching '{query}':"]
            for p in previews:
                lines.append(f"  - {p[:100]}")
            return "\n".join(lines)
        return f"Forgot: deleted={result.get('deleted', 0)}, status={result.get('status', 'ok')}"
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def delete_memory(element_id: str) -> str:
    """
    Hard-delete a specific fact from memory by its element_id.
    The element_id can be obtained from recall_memory, search_memory, or get_memory_metadata.
    Immediately removes the fact and rebuilds the BM25 index.
    """
    try:
        result = await _delete(f"/facts/{element_id}")
        return _fmt(result)
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def pin_memory(element_id: str) -> str:
    """
    Pin a fact permanently — sets importance=1.0 and marks it never-prunable.
    Pinned facts survive consolidation decay and are always injected in context.
    The element_id can be obtained from recall_memory or search_memory.
    """
    try:
        result = await _post(f"/facts/{element_id}/pin", {})
        return _fmt(result)
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def feedback_memory(
    element_id: str,
    rating: int,
    comment: str = "",
) -> str:
    """
    Rate a memory 1-5 stars to adjust its importance and confidence.
    4-5 stars: boosts importance × 1.2 and reinforces the fact.
    1-2 stars: reduces importance × 0.7 and flags for review.
    3 stars: records rating with no importance change.
    """
    try:
        result = await _post("/feedback", {
            "element_id": element_id, "rating": rating, "comment": comment,
        })
        return _fmt(result)
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 3 — Search & Retrieval (4 tools)
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def search_memory(
    query: str,
    limit: int = 10,
    session_id: str = "",
    format: str = "compact",
) -> str:
    """
    Compact vector search across facts and episodes tiers.
    Returns scored results without full context formatting.
    Faster than recall_memory when you only need raw search results.
    format: 'compact' (content+score) or 'full' (with persona/session context).
    """
    try:
        result = await _post("/search", {
            "query": query, "limit": limit, "session_id": session_id, "format": format,
        })
        results = result.get("results", [])
        if not results:
            return "No results found."
        lines = [f"Search results ({len(results)}):"]
        for r in results:
            lines.append(f"  [{r.get('score', 0):.3f}] ({r.get('category', '?')}) {r.get('content', '')[:120]}")
        return "\n".join(lines)
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def smart_search(
    query: str,
    limit: int = 10,
) -> str:
    """
    wRRF-fused search across facts + episodes with per-result scores.
    Combines BM25 keyword and vector similarity for higher precision.
    Use when you want ranked results with explicit relevance scores.
    """
    try:
        result = await _post("/smart-search", {"query": query, "limit": limit})
        results = result.get("results", [])
        if not results:
            return "No results found."
        lines = [f"Smart search results ({len(results)}):"]
        for r in results:
            lines.append(f"  [{r.get('score', 0):.4f}] ({r.get('category', '?')}) {r.get('content', '')[:150]}")
        return "\n".join(lines)
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def timeline(
    limit: int = 20,
    session_id: str = "",
) -> str:
    """
    Retrieve episodes in reverse-chronological order.
    Optionally filter to a specific session.
    Useful for understanding what happened in recent sessions.
    """
    try:
        result = await _post("/timeline", {"limit": limit, "session_id": session_id})
        items = result.get("timeline", [])
        if not items:
            return "No episodes found."
        lines = [f"Timeline ({len(items)} episodes):"]
        for item in items:
            ts = item.get("timestamp", 0)
            lines.append(f"  [{item.get('category', '?')}] {item.get('content', '')[:120]}")
        return "\n".join(lines)
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def answer_from_memory(
    question: str,
    session_id: str = "",
    memory_limit_number: int = 8,
) -> str:
    """
    Retrieve memory context and then ask an LLM to extract a concise factual answer.
    Returns both the extracted answer and the supporting context.
    Use when you need a direct answer drawn from memory, not just raw context.
    """
    try:
        result = await _post("/answer", {
            "query": question,
            "session_id": session_id,
            "memory_limit_number": memory_limit_number,
        })
        answer = result.get("answer", "")
        context = result.get("context", "")[:400]
        lines = []
        if answer:
            lines.append(f"Answer: {answer}")
        if context:
            lines.append(f"\nSupporting context:\n{context}")
        return "\n".join(lines) if lines else "No answer could be extracted from memory."
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 4 — Graph (5 tools)
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def recall_graph(
    entity: str,
    k: int = 5,
) -> str:
    """
    Retrieve facts from the knowledge graph neighbourhood of an entity.
    Finds memories semantically associated with the entity via graph edges.
    Use when you want contextually related facts for a person, project, or concept.
    """
    try:
        result = await _post("/graph/recall", {"entity": entity, "k": k})
        facts = result.get("facts", [])
        if not facts:
            return f"No graph-linked facts found for entity: {entity}"
        lines = [f"Graph facts for '{entity}' ({len(facts)}):"]
        for f in facts:
            lines.append(f"  [{f.get('score', 0):.3f}] {f.get('content', '')[:150]}")
        return "\n".join(lines)
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def graph_neighbors(entity: str) -> str:
    """
    Get direct neighbours of an entity in the knowledge graph.
    Returns related entities with relationship types and co-occurrence counts.
    """
    try:
        result = await _get(f"/graph/{entity}")
        neighbors = result.get("neighbors", [])
        if not neighbors:
            return f"No neighbours found for entity: {entity}"
        lines = [f"Neighbours of '{entity}' ({len(neighbors)}):"]
        for n in neighbors[:20]:
            if isinstance(n, dict):
                lines.append(f"  {n.get('entity', n)} — {n.get('relationship', '')} (count={n.get('count', 1)})")
            else:
                lines.append(f"  {n}")
        return "\n".join(lines)
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def graph_stats() -> str:
    """Get knowledge graph statistics: node count, edge count, top entities."""
    try:
        result = await _get("/graph/stats")
        return _fmt(result)
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def add_graph_edge(
    source_entity: str,
    target_entity: str,
    relationship_type: str,
    confidence: float = 0.8,
    bidirectional: bool = False,
) -> str:
    """
    Manually add a typed relationship edge between two entities in the knowledge graph.
    Example: source='jace', target='python', relationship_type='prefers', confidence=0.9.
    Bidirectional=True also adds the reverse edge.
    """
    try:
        result = await _post("/graph/edge", {
            "source_entity": source_entity,
            "target_entity": target_entity,
            "relationship_type": relationship_type,
            "confidence": confidence,
            "source": "manual",
            "bidirectional": bidirectional,
        })
        return _fmt(result)
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def traverse_graph(
    entity: str,
    relationship_types: list[str] = [],
    max_depth: int = 2,
    max_nodes: int = 20,
) -> str:
    """
    Walk outward through typed edges from an entity for impact/context analysis.
    Useful for 'what else is related to X' queries across multiple hops.
    relationship_types: filter to specific edge types (empty = all types).
    """
    try:
        result = await _post("/graph/traverse", {
            "entity": entity,
            "relationship_types": relationship_types,
            "max_depth": max_depth,
            "max_nodes": max_nodes,
        })
        nodes = result.get("nodes", [])
        if not nodes:
            return f"No connected nodes found from entity: {entity}"
        lines = [f"Graph traversal from '{entity}' ({len(nodes)} nodes, depth={max_depth}):"]
        for n in nodes[:25]:
            if isinstance(n, dict):
                lines.append(f"  [{n.get('depth', '?')}] {n.get('entity', n)} via {n.get('relationship', '?')}")
            else:
                lines.append(f"  {n}")
        return "\n".join(lines)
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 5 — Memory introspection (4 tools)
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def get_memory_metadata(element_id: str) -> str:
    """
    Get full metadata for a fact: importance, confidence, access count,
    user ratings, pin status, lifecycle info, and supersession chain.
    element_id is returned by recall_memory, search_memory, and smart_search.
    """
    try:
        result = await _get(f"/facts/{element_id}/metadata")
        return _fmt(result)
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def get_memory_confidence(element_id: str) -> str:
    """
    Get the effective confidence for a fact with Ebbinghaus temporal decay applied.
    Returns raw confidence, decay factor, and effective confidence after time adjustment.
    """
    try:
        result = await _get(f"/facts/{element_id}/confidence")
        return _fmt(result)
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def reinforce_memory(
    element_id: str,
    source: str = "manual",
) -> str:
    """
    Reinforce a fact — increments source_count, resets Ebbinghaus decay clock,
    and boosts effective confidence. Call when a memory is confirmed correct.
    """
    try:
        result = await _post(f"/facts/{element_id}/reinforce", {"source": source})
        return _fmt(result)
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def lifecycle_stats() -> str:
    """
    Get lifecycle statistics: confidence distribution (high/medium/low/expired),
    supersession counts, and facts pending review.
    Useful for understanding overall memory health.
    """
    try:
        result = await _get("/lifecycle/stats")
        return _fmt(result)
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 6 — Session (4 tools)
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def get_session(session_id: str) -> str:
    """
    Get the session summary, accumulated context, and episode list for a session.
    """
    try:
        result = await _get(f"/session/{session_id}")
        return _fmt(result)
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def list_sessions(limit: int = 10) -> str:
    """
    List recent sessions with metadata: session_id, start time, episode count, summary.
    """
    try:
        result = await _get("/sessions", {"limit": limit})
        sessions = result.get("sessions", [])
        if not sessions:
            return "No sessions found."
        lines = [f"Recent sessions ({len(sessions)}):"]
        for s in sessions:
            lines.append(f"  {s.get('session_id', '?')[:16]}… eps={s.get('ep_count', 0)} — {s.get('summary', '')[:80]}")
        return "\n".join(lines)
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def compact_session(
    session_id: str,
    threshold_chars: int = 3000,
) -> str:
    """
    Mid-session compress of Tier 1 session KV when context grows too large.
    Fires automatically via PostToolUse hook, but can be called manually.
    threshold_chars: only compress if session context exceeds this length.
    """
    try:
        result = await _post("/session/compact", {
            "session_id": session_id, "threshold_chars": threshold_chars,
        })
        return f"Session compact: {result.get('status', 'ok')} — chars_before={result.get('chars_before', '?')}"
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def get_context(
    session_id: str,
    query: str = "",
) -> str:
    """
    Get formatted cross-session context for a session, ready for prompt injection.
    Returns the same <cross_session_memory> block that the recall hook injects.
    """
    try:
        result = await _post("/context", {"session_id": session_id, "query": query})
        return result.get("context", result.get("prependContext", _fmt(result)))
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 7 — Persona & capabilities (3 tools)
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def get_profile() -> str:
    """
    Get the user persona and preferences stored in memory:
    rules, identity, preferred tools, working style, and registered environment.
    """
    try:
        result = await _get("/profile")
        profile = result.get("profile", {})
        if not profile:
            return "No user profile found. Use remember_fact with category='rule' or 'identity' to build one."
        lines = ["User profile:"]
        for k, v in profile.items():
            lines.append(f"  {k}: {str(v)[:120]}")
        return "\n".join(lines)
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def get_capabilities() -> str:
    """
    List all tools and skills registered in AgentMem's capability index.
    Shows tool names, descriptions, success rates, and last-used times.
    """
    try:
        result = await _get("/capabilities")
        return _fmt(result)
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def get_config() -> str:
    """
    Get current AgentMem service configuration: feature flags, thresholds,
    model settings, and active features (HyDE, MIRIX, A-MAC threshold, etc.).
    """
    try:
        result = await _get("/config/flags")
        return _fmt(result)
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 8 — Procedures & Tools (4 tools)
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def procedure_feedback(
    procedure_id: str,
    success: bool,
    session_id: str = "",
) -> str:
    """
    Record whether a procedure succeeded or failed.
    Updates the MACLA Beta posterior score (Beta(success+1, fail+1) × cosine),
    which improves future procedure ranking for similar tasks.
    """
    try:
        result = await _post("/procedure-feedback", {
            "procedure_id": procedure_id, "success": success, "session_id": session_id,
        })
        return f"Procedure feedback recorded: {result.get('status', 'ok')}"
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def get_tool_procedures(tool_name: str) -> str:
    """
    O(1) reverse lookup: get all procedures that use a given tool.
    Uses the mem:proc_by_tool reverse index built by AWO.
    Example: get_tool_procedures('Bash') returns all procedures that use Bash.
    """
    try:
        result = await _get(f"/tool-procedures/{tool_name}")
        procs = result.get("procedures", [])
        if not procs:
            return f"No procedures found for tool: {tool_name}"
        lines = [f"Procedures using {tool_name} ({len(procs)}):"]
        for p in procs:
            lines.append(f"  • {p.get('task', '?')[:80]} (score={p.get('score', 0):.2f})")
        return "\n".join(lines)
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def get_tool_graph(tool_name: str) -> str:
    """
    Get AutoTool TIG (Tool Interaction Graph) hints for a given tool.
    Returns the most likely next tools after this tool, ranked by transition frequency.
    Example: after 'Bash', Claude most often uses 'Read' or 'Edit'.
    """
    try:
        result = await _get(f"/tool-graph/{tool_name}")
        return _fmt(result)
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def detect_meta_tools(
    min_chain_strength: int = 5,
) -> str:
    """
    AWO: scan the Tool Interaction Graph for 2-hop chains A→B→C and
    auto-create meta-tool procedure entries for frequent patterns.
    Call periodically to synthesise new compound workflows from observed tool usage.
    min_chain_strength: minimum co-occurrence count to create a meta-tool.
    """
    try:
        result = await _post("/tool-graph/detect-meta-tools", {
            "min_chain_strength": min_chain_strength,
        })
        created = result.get("created", 0)
        chains = result.get("chains", [])
        lines = [f"AWO meta-tool synthesis: {created} new meta-tools created from {len(chains)} chains"]
        for c in chains[:10]:
            lines.append(f"  {c}")
        return "\n".join(lines)
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 9 — Lifecycle & admin (4 tools)
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def consolidate() -> str:
    """
    Trigger the 3-phase consolidation pipeline synchronously:
    Phase 1 — Decay: age >90d → importance × 0.9
    Phase 2 — Merge: similar facts (affinity ≥ 0.85) → LLM merge
    Phase 3 — Prune: importance <0.05 → soft-delete
    Returns merged, pruned, decayed, and before/after counts.
    """
    try:
        result = await _post("/consolidate/sync", {})
        return (
            f"Consolidation complete: "
            f"merged={result.get('merged', 0)}, "
            f"pruned={result.get('pruned', 0)}, "
            f"decayed={result.get('decayed', 0)}, "
            f"{result.get('total_before', '?')} → {result.get('total_after', '?')} facts, "
            f"{result.get('ms', '?')}ms"
        )
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def export_memories() -> str:
    """
    Export all facts and episodes as JSON (version 1.2.0 format).
    Includes full metadata for each entry. Use for backup or migration.
    Returns the export data up to 8000 chars; full export is available via GET /export.
    """
    try:
        result = await _get("/export")
        facts_count = len(result.get("facts", []))
        eps_count = len(result.get("episodes", []))
        summary = f"Export: {facts_count} facts, {eps_count} episodes (v{result.get('version', '?')})\n"
        return summary + _fmt(result)[:7500]
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def find_patterns(
    query: str = "",
    limit: int = 10,
) -> str:
    """
    Extract recurring patterns from episodic memory.
    Returns pattern clusters with frequency counts.
    Useful for understanding repeated behaviours or decisions across sessions.
    """
    try:
        result = await _post("/patterns", {"query": query, "limit": limit})
        patterns = result.get("patterns", [])
        if not patterns:
            return "No patterns found."
        lines = [f"Patterns found ({len(patterns)}):"]
        for p in patterns:
            lines.append(f"  [freq={p.get('frequency', 1)}] {p.get('content', '')[:120]}")
        return "\n".join(lines)
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def enrich_memory(
    content: str,
    existing_id: str = "",
) -> str:
    """
    Enrich a memory or new content with extracted keywords, entities,
    and semantic triple (subject, predicate, object) for better retrieval.
    If existing_id is provided, enriches that stored fact in-place.
    """
    try:
        result = await _post("/enrich", {"content": content, "existing_id": existing_id})
        return _fmt(result)
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 10 — Raw tier access (5 tools)
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def get_observations(
    session_id: str = "",
    limit: int = 20,
) -> str:
    """
    List raw observations (compressed conversation turns) with optional session filter.
    Returns the raw input to the memory pipeline before fact/episode extraction.
    """
    try:
        params: dict = {"limit": limit}
        if session_id:
            params["session_id"] = session_id
        result = await _get("/observations", params)
        obs = result.get("observations", [])
        if not obs:
            return "No observations found."
        lines = [f"Observations ({len(obs)}):"]
        for o in obs[:20]:
            lines.append(f"  [{o.get('session_id', '?')[:8]}] {str(o.get('content', ''))[:100]}")
        return "\n".join(lines)
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def list_raw_memories(limit: int = 20) -> str:
    """
    List all stored memories across types (facts + patterns + episodes).
    Returns raw memory entries sorted by recency.
    """
    try:
        result = await _get("/memories", {"limit": limit})
        memories = result if isinstance(result, list) else result.get("memories", [])
        if not memories:
            return "No memories found."
        lines = [f"Raw memories ({len(memories)}):"]
        for m in memories[:20]:
            lines.append(f"  [{m.get('type', '?')}] {str(m.get('content', ''))[:100]}")
        return "\n".join(lines)
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def get_semantic_tier(limit: int = 20) -> str:
    """
    Access raw semantic memory entries (cross-session distilled facts).
    These are facts that survived consolidation and represent persistent knowledge.
    """
    try:
        result = await _get("/semantic", {"limit": limit})
        items = result if isinstance(result, list) else result.get("semantic", [])
        if not items:
            return "No semantic memories found."
        lines = [f"Semantic tier ({len(items)} entries):"]
        for m in items[:20]:
            lines.append(f"  [conf={m.get('confidence', 0):.2f}] {str(m.get('fact', m.get('content', '')))[:120]}")
        return "\n".join(lines)
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def get_procedural_tier(limit: int = 20) -> str:
    """
    Access raw procedural memory entries (extracted how-to workflows).
    These are procedures that were distilled from pattern analysis.
    """
    try:
        result = await _get("/procedural", {"limit": limit})
        items = result if isinstance(result, list) else result.get("procedural", [])
        if not items:
            return "No procedural memories found."
        lines = [f"Procedural tier ({len(items)} entries):"]
        for m in items[:20]:
            lines.append(f"  [freq={m.get('frequency', 1)}] {m.get('name', '?')}: {str(m.get('steps', ''))[:100]}")
        return "\n".join(lines)
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def get_relations(limit: int = 30) -> str:
    """
    Get entity relation triples from the knowledge graph (subject, predicate, object).
    Returns the semantic triple index used for knowledge graph traversal.
    """
    try:
        result = await _get("/relations", {"limit": limit})
        relations = result if isinstance(result, list) else result.get("relations", [])
        if not relations:
            return "No relations found."
        lines = [f"Relations ({len(relations)}):"]
        for r in relations[:30]:
            if isinstance(r, dict):
                lines.append(f"  ({r.get('subject', '?')}) —[{r.get('predicate', '?')}]→ ({r.get('object', '?')})")
            else:
                lines.append(f"  {r}")
        return "\n".join(lines)
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 11 — Batch operations (3 tools)
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def batch_search(
    queries: list[str],
    limit: int = 5,
) -> str:
    """
    Run smart_search for multiple queries in parallel.
    Returns merged, deduplicated results ranked by score. Max 5 queries.
    """
    try:
        qs = queries[:5]
        tasks = [_post("/smart-search", {"query": q, "limit": limit}) for q in qs]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        seen: set[str] = set()
        all_results: list[str] = []
        for q, res in zip(qs, results_list):
            if isinstance(res, Exception):
                continue
            for r in res.get("results", []):
                content = r.get("content", "")[:120]
                key = content[:60]
                if key not in seen:
                    seen.add(key)
                    all_results.append(f"  [{r.get('score', 0):.3f}] {content}")
        if not all_results:
            return "No results found."
        return f"Batch search ({len(all_results)} results across {len(qs)} queries):\n" + "\n".join(all_results[:30])
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def batch_recall_procedures(
    task_descriptions: list[str],
    k: int = 2,
) -> str:
    """
    Recall procedures for multiple task descriptions in parallel.
    Returns the best-matching procedure for each task. Max 5 descriptions.
    """
    try:
        descs = task_descriptions[:5]
        tasks = [_post("/recall-procedures", {"query": d, "k": k}) for d in descs]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        lines = [f"Batch procedure recall ({len(descs)} tasks):"]
        for d, res in zip(descs, results_list):
            if isinstance(res, Exception):
                lines.append(f"\n[{d[:40]}]: error")
                continue
            procs = res.get("procedures", [])
            if procs:
                p = procs[0]
                lines.append(f"\n[{d[:40]}]:\n  {p.get('task', '?')} — {p.get('procedure', '')[:200]}")
            else:
                lines.append(f"\n[{d[:40]}]: no procedure found")
        return "\n".join(lines)
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def batch_store_procedure(procedures: list[dict]) -> str:
    """
    Store multiple procedures in parallel.
    Each item is a dict with: task (str), procedure (str), tools_used (list[str]),
    domain (str), session_id (str). More efficient than calling store_procedure N times.
    """
    try:
        tasks = [
            _post("/store-procedure", {
                "task": p.get("task", ""),
                "procedure": p.get("procedure", ""),
                "tools_used": p.get("tools_used", []),
                "domain": p.get("domain", ""),
                "session_id": p.get("session_id", ""),
            })
            for p in procedures
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
        return f"Batch stored: {len(procedures)} procedure(s)"
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 12 — Misc (4 tools)
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def crystallize(session_id: str = "") -> str:
    """
    Distill important session summaries into high-importance long-term facts.
    Crystallisation extracts durable facts from transient session context.
    Call after a productive session to permanently preserve key insights.
    """
    try:
        result = await _post("/crystallize", {"session_id": session_id})
        return _fmt(result)
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def get_health() -> str:
    """
    Check AgentMem service health: Redis connection, memory tier sizes,
    embedding model status, and background pipeline health.
    """
    try:
        result = await _get("/health")
        return _fmt(result)
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def record_tool_sequence(
    tool_names: list[str],
    session_id: str = "",
) -> str:
    """
    Record a sequence of tool usages to train the AutoTool TIG (Transition Graph).
    Subsequent get_tool_graph calls will incorporate these transitions.
    Call after completing a multi-tool workflow with the ordered list of tools used.
    """
    try:
        result = await _post("/record-tool-sequence", {
            "tool_names": tool_names, "session_id": session_id,
        })
        return f"Tool sequence recorded: {len(tool_names)} tools → {result.get('transitions', 0)} TIG transitions added"
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def get_system_prompt() -> str:
    """
    Get the current memory system prompt — the <cross_session_memory> block
    that AgentMem injects into every prompt via the recall hook.
    Use this to inspect what context is being injected without triggering a recall.
    """
    try:
        result = await _get("/system-prompt")
        return result.get("system_prompt", result.get("content", _fmt(result)))
    except httpx.ConnectError:
        return _dead()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


# ── ASGI app factories (used by main.py for HTTP/SSE mounting) ────────────────

def build_mcp_app():
    """Return a Starlette ASGI app for Streamable HTTP (MCP 2025-03-26).

    The app is mounted at ``/mcp`` in main.py, so the internal route must be
    ``/`` — otherwise the default ``/mcp`` route doubles to ``/mcp/mcp``.
    """
    mcp.settings.streamable_http_path = "/"
    return mcp.streamable_http_app()


def build_sse_app():
    """Return a Starlette ASGI app for SSE transport (legacy HTTP clients)."""
    return mcp.sse_app(mount_path="/mcp/sse")


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="AgentMem MCP Server")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--stdio", action="store_true", default=False, help="stdio transport (default)")
    group.add_argument("--http",  action="store_true", default=False, help="Streamable HTTP transport")
    group.add_argument("--sse",   action="store_true", default=False, help="SSE transport (legacy)")
    parser.add_argument("--port", type=int, default=18801, help="Port for HTTP/SSE mode (default 18801)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for HTTP/SSE mode")
    args = parser.parse_args()

    if args.http:
        print(f"AgentMem MCP server (Streamable HTTP) on http://{args.host}:{args.port}/mcp")
        uvicorn.run(build_mcp_app(), host=args.host, port=args.port)
    elif args.sse:
        print(f"AgentMem MCP server (SSE) on http://{args.host}:{args.port}/mcp/sse")
        uvicorn.run(build_sse_app(), host=args.host, port=args.port)
    else:
        asyncio.run(mcp.run_stdio_async())
