"""
agentmemory-compatible hooks for AgentMem — Python implementation

These hooks match the exact API contract of rohitg00/agentmemory's Node.js hooks.
They read JSON from stdin and make HTTP calls to the AgentMem backend.

Usage in Claude Code hooks.json:
  {
    "hooks": {
      "session-start": [{ "command": "python3 /path/to/hooks/session_start.py" }],
      "post-tool-use": [{ "command": "python3 /path/to/hooks/post_tool_use.py" }],
      ...
    }
  }

Environment variables:
  AGENTMEMORY_URL  — backend URL (default: http://localhost:18800)
  AGENTMEMORY_SECRET — auth secret (default: empty)
  AGENTMEMORY_INJECT_CONTEXT — inject context on session-start (default: true)
"""
