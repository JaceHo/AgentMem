#!/usr/bin/env bash
# Hook: SessionStart — register Claude Code tools into mem:tools (v0.9.7)
# Discovers tools from two sources:
#   1. Known built-in Claude Code tools (static, authoritative)
#   2. MCP tools parsed from permissions.allow in settings files
# Fire-and-forget: does NOT block session start.
unset PYTHONWARNINGS

python3 - << 'PYEOF'
import json, os, urllib.request

# ── 1. Known Claude Code built-in tools ──────────────────────────────────────
BUILTIN_TOOLS = [
    {"name": "Bash",            "description": "Execute shell commands in the user's environment",                          "category": "system",        "source": "builtin", "parameters": ["command"]},
    {"name": "Read",            "description": "Read files from the local filesystem",                                       "category": "filesystem",    "source": "builtin", "parameters": ["file_path"]},
    {"name": "Write",           "description": "Write or create files on the local filesystem",                              "category": "filesystem",    "source": "builtin", "parameters": ["file_path", "content"]},
    {"name": "Edit",            "description": "Edit files using exact string replacement",                                  "category": "filesystem",    "source": "builtin", "parameters": ["file_path", "old_string", "new_string"]},
    {"name": "Glob",            "description": "Find files matching glob patterns (e.g. **/*.py)",                          "category": "filesystem",    "source": "builtin", "parameters": ["pattern", "path"]},
    {"name": "Grep",            "description": "Search file contents using regex patterns (ripgrep)",                        "category": "search",        "source": "builtin", "parameters": ["pattern", "path"]},
    {"name": "WebFetch",        "description": "Fetch a URL and extract readable text content from the page",               "category": "web",           "source": "builtin", "parameters": ["url", "prompt"]},
    {"name": "WebSearch",       "description": "Search the web and return structured results with URLs",                     "category": "web",           "source": "builtin", "parameters": ["query"]},
    {"name": "Agent",           "description": "Launch a specialized subagent for complex or parallel tasks",               "category": "orchestration", "source": "builtin", "parameters": ["subagent_type", "prompt"]},
    {"name": "Task",            "description": "Run a shell command in the background, returns task_id",                    "category": "orchestration", "source": "builtin", "parameters": ["description", "prompt"]},
    {"name": "TaskCreate",      "description": "Create a tracked task entry in the session task list",                      "category": "orchestration", "source": "builtin", "parameters": ["subject", "description"]},
    {"name": "TaskUpdate",      "description": "Update task status (pending/in_progress/completed)",                        "category": "orchestration", "source": "builtin", "parameters": ["taskId", "status"]},
    {"name": "TaskList",        "description": "List all tasks in the current session",                                     "category": "orchestration", "source": "builtin", "parameters": []},
    {"name": "NotebookEdit",    "description": "Edit Jupyter notebook cells (code or markdown)",                            "category": "filesystem",    "source": "builtin", "parameters": ["notebook_path", "new_source"]},
    {"name": "AskUserQuestion", "description": "Present the user with multiple-choice questions to clarify requirements",  "category": "interaction",   "source": "builtin", "parameters": ["questions"]},
    {"name": "Skill",           "description": "Invoke a user-defined slash command skill by name",                         "category": "interaction",   "source": "skill",   "parameters": ["skill", "args"]},
    {"name": "EnterPlanMode",   "description": "Enter plan mode to design implementation approach before coding",           "category": "orchestration", "source": "builtin", "parameters": []},
    {"name": "ExitPlanMode",    "description": "Exit plan mode and present plan to user for approval",                     "category": "orchestration", "source": "builtin", "parameters": []},
]

# ── 2. Discover MCP tools from settings files ─────────────────────────────────
# Walk up from cwd looking for .claude/settings.local.json and check global
settings_candidates = [
    os.path.expanduser("~/.claude/settings.json"),
]
cwd = os.getcwd()
for _ in range(5):
    candidate = os.path.join(cwd, ".claude", "settings.local.json")
    if os.path.exists(candidate):
        settings_candidates.append(candidate)
    parent = os.path.dirname(cwd)
    if parent == cwd:
        break
    cwd = parent

# MCP tool name → human-readable description (known servers)
_MCP_DESCRIPTIONS = {
    "web_search_prime":       "Search the web via prime search API (structured results)",
    "webReader":              "Fetch and convert a URL to clean readable text",
    "search_doc":             "Search documentation, issues, and commits of a GitHub repo",
    "read_file":              "Read the full content of a file in a GitHub repository",
    "get_repo_structure":     "Get the directory structure and file list of a GitHub repo",
    "bash":                   "Execute a bash command in the toolserv sandbox",
    "get_time":               "Get the current date and time in any IANA timezone",
    "ip_info":                "Look up geographic and network info for an IP address",
    "exchange_rate":          "Get live currency exchange rates",
    "uuid_generate":          "Generate a cryptographically random UUID v4",
    "memory_recall_procedures": "Search AgentMem procedural memory for prior workflows",
    "memory_store_procedure": "Store a completed workflow as a procedure in AgentMem",
    "memory_graph_recall":    "Query AgentMem knowledge graph for entity relationships",
    "proxy_status":           "Check proxy pool status and outbound connectivity",
    "echo":                   "Echo back a message (connectivity test)",
}

mcp_tools = []
seen_names = {t["name"] for t in BUILTIN_TOOLS}

for path in settings_candidates:
    try:
        d = json.load(open(path))
        allow_list = d.get("permissions", {}).get("allow", [])
        for perm in allow_list:
            # mcp__server__tool_name format
            if not perm.startswith("mcp__"):
                continue
            if perm == "mcp__*":
                continue   # wildcard — no specific tool to register
            parts = perm.split("__", 2)
            if len(parts) < 3:
                continue
            server = parts[1]
            tool   = parts[2]
            full   = perm   # e.g. mcp__zread__read_file
            if full in seen_names:
                continue
            seen_names.add(full)
            desc = _MCP_DESCRIPTIONS.get(tool, f"MCP tool '{tool}' from {server} server")
            mcp_tools.append({
                "name":        full,
                "description": desc,
                "category":    "mcp",
                "source":      "mcp",
                "parameters":  [],
            })
    except Exception:
        pass

all_tools = BUILTIN_TOOLS + mcp_tools

payload = json.dumps({"tools": all_tools, "agent_id": ""}).encode()
req = urllib.request.Request(
    "http://localhost:18800/register-tools",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    urllib.request.urlopen(req, timeout=3)
except Exception:
    pass   # service not running or timeout — non-fatal
PYEOF
exit 0
