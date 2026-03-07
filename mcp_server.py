"""
MCP Server for claw-memory — v0.9.0

Exposes memory service tools via the Model Context Protocol (MCP),
enabling any MCP-compatible client (Claude Desktop, Claude Code, Cursor,
Windsurf, LM Studio) to use long-term memory with zero extra code.

Transport: stdio (default) — suitable for Claude Desktop / local clients.

Usage
-----
Run directly:
    python mcp_server.py

Claude Desktop config (~/.config/claude/claude_desktop_config.json):
    {
      "mcpServers": {
        "claw-memory": {
          "command": "python",
          "args": ["/path/to/claw-memory/mcp_server.py"],
          "env": {}
        }
      }
    }

Tools exposed
-------------
  recall_memory       — retrieve relevant memories for a query
  store_memory        — save a conversation turn to long-term memory
  recall_tools        — find agent tools by capability description
  recall_procedures   — find how-to workflows by task description
  store_procedure     — save a successful workflow
  get_stats           — memory counts across all tiers
  compress_session    — promote session to long-term memory
"""

from __future__ import annotations

import httpx

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp import types
except ImportError:
    raise SystemExit(
        "mcp package not found. Install it with:\n"
        "    pip install mcp\n"
        "or:\n"
        "    pip install -r requirements.txt"
    )

BASE_URL = "http://127.0.0.1:18800"
TIMEOUT  = 10.0

server = Server("claw-memory")


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


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="recall_memory",
            description=(
                "Retrieve relevant long-term memories, past conversation context, "
                "user preferences, and learned facts for a given query. "
                "Call this at the start of a conversation or when you need background knowledge."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query":               {"type": "string",  "description": "What to search for in memory"},
                    "session_id":          {"type": "string",  "description": "Session identifier (optional)", "default": ""},
                    "memory_limit_number": {"type": "integer", "description": "Max memories to retrieve",      "default": 6},
                    "include_tools":       {"type": "boolean", "description": "Include tool capability context", "default": True},
                    "include_graph":       {"type": "boolean", "description": "Expand graph neighbours",        "default": False},
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="store_memory",
            description=(
                "Save a conversation turn or important information to long-term memory. "
                "Provide messages as a list with role and content. "
                "Call this after each meaningful exchange to build persistent memory."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "description": "List of message objects with role and content",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role":    {"type": "string", "enum": ["user", "assistant", "system"]},
                                "content": {"type": "string"},
                            },
                            "required": ["role", "content"],
                        },
                    },
                    "session_id": {"type": "string", "description": "Session identifier", "default": ""},
                },
                "required": ["messages"],
            },
        ),
        types.Tool(
            name="recall_tools",
            description=(
                "Find registered agent tools and skills by semantic description. "
                "Use this to discover what capabilities are available for a task."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query":    {"type": "string",  "description": "Capability description to search for"},
                    "k":        {"type": "integer", "description": "Number of results",          "default": 5},
                    "category": {"type": "string",  "description": "Filter by tool category",    "default": ""},
                    "source":   {"type": "string",  "description": "Filter by tool source",      "default": ""},
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="recall_procedures",
            description=(
                "Find how-to workflows and successful task patterns by description. "
                "Use this when approaching a task to check if there's a known procedure."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string",  "description": "Task description to find procedures for"},
                    "k":     {"type": "integer", "description": "Number of results", "default": 3},
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="store_procedure",
            description=(
                "Save a successful workflow or how-to pattern to procedural memory. "
                "Call this after completing a multi-step task to remember the approach."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "task":       {"type": "string", "description": "Task description (used as retrieval key)"},
                    "procedure":  {"type": "string", "description": "Step-by-step procedure text"},
                    "tools_used": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Tools or skills used", "default": [],
                    },
                    "domain":     {"type": "string", "description": "Domain/context", "default": ""},
                    "session_id": {"type": "string", "description": "Session identifier", "default": ""},
                },
                "required": ["task", "procedure"],
            },
        ),
        types.Tool(
            name="get_stats",
            description="Get memory counts across all tiers (episodes, facts, procedures, tools).",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="compress_session",
            description=(
                "Promote the current session's accumulated context to long-term memory. "
                "Call this at the end of a conversation to crystallise session knowledge."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Session identifier to compress"},
                },
                "required": ["session_id"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    try:
        if name == "recall_memory":
            result = await _post("/recall", arguments)
            ctx = result.get("prependContext") or "(no relevant memories found)"
            text = f"Memory context ({result.get('latency_ms', '?')}ms):\n\n{ctx}"

        elif name == "store_memory":
            result = await _post("/store", arguments)
            text = f"Memory stored: {result.get('status', 'ok')}"

        elif name == "recall_tools":
            result = await _post("/recall-tools", arguments)
            tools = result.get("tools", [])
            if not tools:
                text = "No matching tools found."
            else:
                lines = [f"Found {len(tools)} tool(s):"]
                for t in tools:
                    lines.append(f"  • {t['name']} ({t.get('score', 0):.2f}): {t.get('description', '')[:100]}")
                text = "\n".join(lines)

        elif name == "recall_procedures":
            result = await _post("/recall-procedures", arguments)
            procs = result.get("procedures", [])
            if not procs:
                text = "No matching procedures found."
            else:
                lines = [f"Found {len(procs)} procedure(s):"]
                for p in procs:
                    lines.append(f"\n**{p['task']}** (score={p.get('score', 0):.2f})\n{p.get('procedure', '')[:300]}")
                text = "\n".join(lines)

        elif name == "store_procedure":
            result = await _post("/store-procedure", arguments)
            text = f"Procedure stored: {result.get('status', 'ok')}"

        elif name == "get_stats":
            result = await _get("/stats")
            lines = ["Memory statistics:"]
            for k, v in result.items():
                lines.append(f"  {k}: {v}")
            text = "\n".join(lines)

        elif name == "compress_session":
            result = await _post("/session/compress", {**arguments, "wait": True})
            text = (
                f"Session compressed: ep_saved={result.get('ep_saved', 0)}, "
                f"facts_saved={result.get('facts_saved', 0)}, "
                f"status={result.get('status', '?')}"
            )

        else:
            text = f"Unknown tool: {name}"

    except httpx.ConnectError:
        text = (
            f"Error: claw-memory service is not running at {BASE_URL}.\n"
            "Start it with: ./start.sh (or python -m uvicorn main:app --port 18800)"
        )
    except Exception as e:
        text = f"Error calling {name}: {type(e).__name__}: {e}"

    return [types.TextContent(type="text", text=text)]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
