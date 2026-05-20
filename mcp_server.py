"""
MCP Server for AgentMem — v1.2.0

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

Cursor (.cursor/mcp.json):
    {"mcpServers": {"agentmem": {"url": "http://localhost:18800/mcp"}}}

Windsurf (~/.codeium/windsurf/mcp_config.json):
    {"mcpServers": {"agentmem": {"serverUrl": "http://localhost:18800/mcp/sse"}}}

GitHub Copilot (.vscode/mcp.json):
    {"servers": {"agentmem": {"type": "http", "url": "http://localhost:18800/mcp"}}}

Zed (~/.config/zed/settings.json):
    {"context_servers": {"agentmem": {"url": "http://localhost:18800/mcp/sse"}}}

Opencode (opencode.json):
    {"mcp": {"agentmem": {"type": "sse", "url": "http://localhost:18800/mcp/sse"}}}

Stdio (all others — Cline, Continue, Kilo Code, Kiro, etc.):
    {"mcpServers": {"agentmem": {"command": "python", "args": ["/path/to/agentmem/mcp_server.py"]}}}

Tools exposed
-------------
  recall_memory        — retrieve relevant memories for a query
  store_memory         — save a conversation turn to long-term memory
  recall_tools         — find agent tools by capability description
  recall_procedures    — find how-to workflows by task description
  store_procedure      — save a successful workflow
  get_stats            — memory counts across all tiers
  compress_session     — promote session to long-term memory
  batch_recall_memory  — recall for multiple queries in parallel
  batch_store_memory   — store multiple turns in parallel
"""

from __future__ import annotations

import asyncio
import httpx

from mcp.server.fastmcp import FastMCP

BASE_URL = "http://127.0.0.1:18800"
TIMEOUT = 10.0

mcp = FastMCP(
    "agentmem",
    instructions=(
        "AgentMem gives you persistent, cross-session memory. "
        "Call recall_memory at the start of each conversation with the user's question as the query. "
        "Call store_memory after each meaningful exchange. "
        "Call compress_session when the conversation ends."
    ),
)


# ── HTTP helpers ──────────────────────────────────────────────────────────────

async def _post(path: str, payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(f"{BASE_URL}{path}", json=payload)
        resp.raise_for_status()
        return resp.json()


async def _get(path: str) -> dict:
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.get(f"{BASE_URL}{path}")
        resp.raise_for_status()
        return resp.json()


def _dead_service_msg() -> str:
    return (
        f"Error: AgentMem service is not running at {BASE_URL}.\n"
        "Start it with: agentmem.sh start"
    )


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
async def recall_memory(
    query: str,
    session_id: str = "",
    memory_limit_number: int = 6,
    include_tools: bool = True,
    include_graph: bool = False,
) -> str:
    """
    Retrieve relevant long-term memories, past conversation context,
    user preferences, and learned facts for a given query.
    Call this at the start of a conversation or when you need background knowledge.
    """
    try:
        result = await _post("/recall", {
            "query": query,
            "session_id": session_id,
            "memory_limit_number": memory_limit_number,
            "include_tools": include_tools,
            "include_graph": include_graph,
        })
        ctx = result.get("prependContext") or "(no relevant memories found)"
        return f"Memory context ({result.get('latency_ms', '?')}ms):\n\n{ctx}"
    except httpx.ConnectError:
        return _dead_service_msg()
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
    Call this after each meaningful exchange to build persistent memory.
    """
    try:
        result = await _post("/store", {"messages": messages, "session_id": session_id})
        return f"Memory stored: {result.get('status', 'ok')}"
    except httpx.ConnectError:
        return _dead_service_msg()
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
        return _dead_service_msg()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def recall_procedures(
    query: str,
    k: int = 3,
) -> str:
    """
    Find how-to workflows and successful task patterns by description.
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
        return _dead_service_msg()
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
    Call this after completing a multi-step task to remember the approach.
    """
    try:
        result = await _post("/store-procedure", {
            "task": task, "procedure": procedure,
            "tools_used": tools_used, "domain": domain, "session_id": session_id,
        })
        return f"Procedure stored: {result.get('status', 'ok')}"
    except httpx.ConnectError:
        return _dead_service_msg()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def get_stats() -> str:
    """Get memory counts across all tiers (episodes, facts, procedures, tools)."""
    try:
        result = await _get("/stats")
        lines = ["Memory statistics:"]
        for k, v in result.items():
            if k != "writer":
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)
    except httpx.ConnectError:
        return _dead_service_msg()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def compress_session(session_id: str) -> str:
    """
    Promote the current session's accumulated context to long-term memory.
    Call this at the end of a conversation to crystallise session knowledge.
    """
    try:
        result = await _post("/session/compress", {"session_id": session_id, "wait": True})
        return (
            f"Session compressed: ep_saved={result.get('ep_saved', 0)}, "
            f"facts_saved={result.get('facts_saved', 0)}, "
            f"status={result.get('status', '?')}"
        )
    except httpx.ConnectError:
        return _dead_service_msg()
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
    More efficient than calling recall_memory N times. Returns combined context.
    """
    try:
        qs = queries[:5]
        tasks = [
            _post("/recall", {
                "query": q, "session_id": session_id,
                "memory_limit_number": memory_limit_number,
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
        return _dead_service_msg()
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
        return _dead_service_msg()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


# ── ASGI app factories (used by main.py for HTTP/SSE mounting) ────────────────

def build_mcp_app():
    """Return a Starlette ASGI app for Streamable HTTP (MCP 2025-03-26)."""
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
