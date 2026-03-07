"""
Agent Capability Registry  v0.1.0
===================================
Stores and retrieves:
  1. Tool/Skill Index   (mem:tools  — HNSW vectorset, semantic search)
  2. Environment State  (mem:env    — Redis Hash, structured)
  3. Agent Self-Model   (mem:agent  — Redis Hash, identity + stats)

This gives the agent persistent self-awareness across sessions:
  - "What can I do?"  → semantic search over mem:tools
  - "What env am I in?" → structured mem:env
  - "What am I?" → mem:agent

Design principles:
  - Tool descriptions are embedded for semantic retrieval (not just keyword lookup)
  - Environment state is structured hash (exact lookup, no embedding needed)
  - All writes are idempotent — re-registering a tool updates it in-place
  - Usage tracking: tool use_count bumped on each recall hit

Redis keys:
  mem:tools    — vectorset (tool name → description embedding + attrs)
  mem:env      — hash     (field → value)
  mem:agent    — hash     (field → value)
"""

import json
import time
from typing import Optional

import redis.asyncio as aioredis

# ── Constants ──────────────────────────────────────────────────────────────────
TOOL_KEY   = "mem:tools"
ENV_KEY    = "mem:env"
AGENT_KEY  = "mem:agent"


# ── Tool categories ────────────────────────────────────────────────────────────
TOOL_CATEGORIES = {
    "file":       ["read", "write", "glob", "grep", "edit", "find", "ls", "cat"],
    "system":     ["bash", "shell", "exec", "terminal", "process", "command"],
    "search":     ["web", "search", "fetch", "browse", "scrape", "lookup"],
    "code":       ["run", "test", "lint", "format", "debug", "compile", "build"],
    "ai":         ["llm", "embed", "prompt", "model", "inference", "agent", "claude"],
    "data":       ["sql", "query", "database", "redis", "csv", "json", "parse"],
    "git":        ["commit", "push", "branch", "diff", "merge", "repo"],
    "web":        ["http", "api", "rest", "graphql", "webhook", "request"],
    "memory":     ["recall", "store", "remember", "forget", "summarize"],
    "skill":      ["skill", "slash", "command", "workflow", "task"],
    "mcp":        ["mcp", "server", "resource", "tool", "protocol"],
}


def _infer_category(name: str, description: str) -> str:
    text = (name + " " + description).lower()
    best, best_score = "general", 0
    for cat, kws in TOOL_CATEGORIES.items():
        score = sum(1 for kw in kws if kw in text)
        if score > best_score:
            best, best_score = cat, score
    return best


# ── Tool Registry ──────────────────────────────────────────────────────────────

async def ensure_tool_index(r: aioredis.Redis, dims: int = 384) -> None:
    """Seed mem:tools vectorset so VSIM works on empty sets."""
    import numpy as np
    card = await r.execute_command("VCARD", TOOL_KEY)
    if not card:
        seed = np.zeros(dims, dtype=np.float32)
        attr = json.dumps({"_seed": True, "name": "", "description": "",
                           "category": "general", "use_count": 0, "ts": 0})
        await r.execute_command("VADD", TOOL_KEY, "FP32", seed.tobytes(),
                                "__seed__", "SETATTR", attr)


async def register_tool(
    r: aioredis.Redis,
    name: str,
    description: str,
    embedding,                          # np.ndarray (384-dim)
    category: str = "",
    source: str = "builtin",            # builtin | mcp | plugin | skill
    parameters: list[str] | None = None,
    agent_id: str = "",
) -> str:
    """
    Store or update a tool in mem:tools.
    Element key is name (slugified) — re-registering updates attrs + vector.
    Returns the element key used.
    """
    import numpy as np
    elem_key = name.lower().replace(" ", "_").replace("/", "_")[:64]
    cat = category or _infer_category(name, description)

    attr = {
        "name":        name,
        "description": description[:400],
        "category":    cat,
        "source":      source,
        "agent_id":    agent_id,
        "use_count":   0,
        "ts":          int(time.time() * 1000),
    }
    if parameters:
        attr["parameters"] = parameters[:10]

    # VADD is upsert in Redis 8 vectorsets — overwrites existing element
    blob = embedding.astype(np.float32).tobytes()
    await r.execute_command("VADD", TOOL_KEY, "FP32", blob,
                            elem_key, "SETATTR", json.dumps(attr))
    return elem_key


async def recall_tools(
    r: aioredis.Redis,
    embedding,              # query embedding np.ndarray
    k: int = 5,
    category_filter: str | None = None,
    source_filter: str | None = None,
) -> list[dict]:
    """
    Semantic search over mem:tools.
    Returns list of {name, description, category, source, score, use_count}.
    """
    import numpy as np
    blob = embedding.astype(np.float32).tobytes()

    filter_expr = None
    if category_filter:
        filter_expr = f'.category == "{category_filter}"'
    elif source_filter:
        filter_expr = f'.source == "{source_filter}"'

    cmd = ["VSIM", TOOL_KEY, "FP32", blob, "COUNT", k + 1,
           "WITHSCORES", "WITHATTRIBS"]
    if filter_expr:
        cmd += ["FILTER", filter_expr]

    try:
        results = await r.execute_command(*cmd)
    except Exception:
        return []

    tools: list[dict] = []
    i = 0
    while i + 2 < len(results):
        elem  = results[i]
        score = results[i + 1]
        raw   = results[i + 2]
        i += 3
        elem_str = elem.decode() if isinstance(elem, bytes) else elem
        if elem_str == "__seed__":
            continue
        try:
            attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        except Exception:
            attrs = {}
        if attrs.get("_seed") or not attrs.get("name"):
            continue

        tools.append({
            "name":        attrs.get("name", elem_str),
            "description": attrs.get("description", ""),
            "category":    attrs.get("category", "general"),
            "source":      attrs.get("source", "builtin"),
            "parameters":  attrs.get("parameters", []),
            "use_count":   attrs.get("use_count", 0),
            "score":       float(score) if score else 0.0,
            "_element":    elem_str,
        })

        # Bump use_count (fire-and-forget)
        import asyncio
        asyncio.create_task(_bump_use_count(r, elem_str, attrs))

    return tools


async def _bump_use_count(r: aioredis.Redis, element: str, attrs: dict) -> None:
    attrs = dict(attrs)
    attrs["use_count"] = attrs.get("use_count", 0) + 1
    attrs["last_used"] = int(time.time() * 1000)
    try:
        await r.execute_command("VSETATTR", TOOL_KEY, element, json.dumps(attrs))
    except Exception:
        pass


async def list_all_tools(r: aioredis.Redis) -> list[dict]:
    """Return all registered tools (no embedding needed — scan via VSIM with zero vec)."""
    import numpy as np
    card = await r.execute_command("VCARD", TOOL_KEY)
    if not card or int(card) <= 1:
        return []

    seed = np.zeros(384, dtype=np.float32)
    try:
        results = await r.execute_command(
            "VSIM", TOOL_KEY, "FP32", seed.tobytes(),
            "COUNT", min(int(card), 300), "WITHSCORES", "WITHATTRIBS"
        )
    except Exception:
        return []

    tools = []
    i = 0
    while i + 2 < len(results):
        elem = results[i]
        raw  = results[i + 2]
        i += 3
        elem_str = elem.decode() if isinstance(elem, bytes) else elem
        if elem_str == "__seed__":
            continue
        try:
            attrs = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        except Exception:
            continue
        if attrs.get("_seed") or not attrs.get("name"):
            continue
        tools.append({
            "name":        attrs.get("name", ""),
            "description": attrs.get("description", ""),
            "category":    attrs.get("category", "general"),
            "source":      attrs.get("source", "builtin"),
            "use_count":   attrs.get("use_count", 0),
        })

    return tools


# ── Environment Registry ───────────────────────────────────────────────────────

async def set_env(r: aioredis.Redis, env_data: dict) -> None:
    """
    Store environment state in mem:env hash.
    All values must be strings (Redis hash requirement).
    Idempotent — overwrites existing fields.
    """
    env_data["last_updated"] = str(int(time.time() * 1000))
    # Serialize complex values
    for k, v in list(env_data.items()):
        if isinstance(v, (list, dict)):
            env_data[k] = json.dumps(v)
        elif not isinstance(v, str):
            env_data[k] = str(v)

    if env_data:
        await r.hset(ENV_KEY, mapping=env_data)


async def get_env(r: aioredis.Redis) -> dict:
    """Return current environment state as dict with parsed values."""
    raw = await r.hgetall(ENV_KEY)
    if not raw:
        return {}
    result = {}
    for k, v in raw.items():
        k_str = k.decode() if isinstance(k, bytes) else k
        v_str = v.decode() if isinstance(v, bytes) else v
        # Try to parse JSON lists/dicts
        if v_str.startswith(("[", "{")):
            try:
                v_str = json.loads(v_str)
            except Exception:
                pass
        result[k_str] = v_str
    return result


async def get_env_context(r: aioredis.Redis) -> str:
    """Return environment as formatted context block."""
    env = await get_env(r)
    if not env:
        return ""

    priority_fields = [
        "os", "os_version", "shell", "cwd", "git_repo", "git_branch",
        "active_mcps", "active_plugins", "active_skills",
        "agent_model", "agent_version", "session_id",
    ]

    lines = ["## Agent Environment"]
    seen = set()
    for field in priority_fields:
        val = env.get(field)
        if val:
            if isinstance(val, list):
                val = ", ".join(val)
            lines.append(f"- {field}: {val}")
            seen.add(field)

    # Any extra fields
    for k, v in env.items():
        if k not in seen and k != "last_updated":
            if isinstance(v, list):
                v = ", ".join(str(x) for x in v)
            lines.append(f"- {k}: {v}")

    return "\n".join(lines)


# ── Agent Self-Model ──────────────────────────────────────────────────────────

async def set_agent(r: aioredis.Redis, agent_data: dict) -> None:
    """Store agent identity and capability summary in mem:agent hash."""
    agent_data["last_seen"] = str(int(time.time() * 1000))
    for k, v in list(agent_data.items()):
        if isinstance(v, (list, dict)):
            agent_data[k] = json.dumps(v)
        elif not isinstance(v, str):
            agent_data[k] = str(v)
    if agent_data:
        await r.hset(AGENT_KEY, mapping=agent_data)


async def get_agent(r: aioredis.Redis) -> dict:
    """Return agent self-model as dict."""
    raw = await r.hgetall(AGENT_KEY)
    if not raw:
        return {}
    result = {}
    for k, v in raw.items():
        k_str = k.decode() if isinstance(k, bytes) else k
        v_str = v.decode() if isinstance(v, bytes) else v
        if v_str.startswith(("[", "{")):
            try:
                v_str = json.loads(v_str)
            except Exception:
                pass
        result[k_str] = v_str
    return result


# ── Capability Summary ────────────────────────────────────────────────────────

async def get_capability_summary(r: aioredis.Redis) -> dict:
    """
    Return a structured capability manifest:
    {
      tools: [{name, description, category, source, use_count}],
      env: {os, shell, cwd, ...},
      agent: {model, version, ...},
      stats: {tool_count, categories: [...]}
    }
    """
    tools = await list_all_tools(r)
    env   = await get_env(r)
    agent = await get_agent(r)

    categories = sorted({t["category"] for t in tools})

    return {
        "tools":  tools,
        "env":    env,
        "agent":  agent,
        "stats": {
            "tool_count":     len(tools),
            "categories":     categories,
            "mcp_count":      len(env.get("active_mcps", []) or []),
            "plugin_count":   len(env.get("active_plugins", []) or []),
            "skill_count":    len(env.get("active_skills", []) or []),
        },
    }


def format_tool_context(tools: list[dict], max_tools: int = 8) -> str:
    """Format top-k tools as a context block for prepending to agent."""
    if not tools:
        return ""
    lines = ["## Available Tools (Relevant)"]
    for t in tools[:max_tools]:
        cat = t.get("category", "")
        src = t.get("source", "")
        tag = f"[{cat}/{src}]" if src and src != "builtin" else f"[{cat}]"
        lines.append(f"- **{t['name']}** {tag}: {t['description'][:120]}")
    return "\n".join(lines)
