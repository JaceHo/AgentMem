"""AgentMem memory plugin for Hermes Agent.

Connects Hermes to a local AgentMem service (Redis-backed, HNSW vector search)
for persistent cross-session memory with heat-tiered recall, episodic/semantic/
procedural tiers, and auto-consolidation.

Config via environment variables:
  AGENTMEM_BASE_URL   — AgentMem REST API URL (default: http://127.0.0.1:18800)
  AGENTMEM_API_KEY    — Optional auth key (default: empty, local mode)

Or via $HERMES_HOME/agentmem.json.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List
from urllib.request import Request, urlopen
from urllib.error import URLError

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://127.0.0.1:18800"
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    from hermes_constants import get_hermes_home

    config = {
        "base_url": os.environ.get("AGENTMEM_BASE_URL", _DEFAULT_BASE_URL),
        "api_key": os.environ.get("AGENTMEM_API_KEY", ""),
    }

    config_path = get_hermes_home() / "agentmem.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({k: v for k, v in file_cfg.items()
                           if v is not None and v != ""})
        except Exception:
            pass

    return config


# ---------------------------------------------------------------------------
# Lightweight HTTP helpers (no external deps)
# ---------------------------------------------------------------------------

def _http_get(url: str, api_key: str = "", timeout: float = 5.0) -> dict | None:
    req = Request(url)
    req.add_header("Accept", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def _http_post(url: str, payload: dict, api_key: str = "", timeout: float = 5.0) -> dict | None:
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

RECALL_SCHEMA = {
    "name": "agentmem_recall",
    "description": (
        "Search persistent memory for relevant facts, episodes, and procedures. "
        "Returns ranked results from episodic, semantic, and procedural tiers. "
        "Use when you need to recall past context, decisions, or workflows."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for in memory."},
            "limit": {"type": "integer", "description": "Max results (default: 10)."},
        },
        "required": ["query"],
    },
}

REMEMBER_SCHEMA = {
    "name": "agentmem_remember",
    "description": (
        "Store a durable fact or procedure in persistent memory. "
        "Use for explicit preferences, corrections, decisions, or how-to workflows."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The fact or procedure to store."},
            "category": {
                "type": "string",
                "description": "Memory category: fact, preference, procedure, decision, rule, identity",
                "default": "fact",
            },
        },
        "required": ["content"],
    },
}

FORGET_SCHEMA = {
    "name": "agentmem_forget",
    "description": (
        "Remove memories matching a query. Use to delete outdated or incorrect info. "
        "Use dry_run=true to preview what would be deleted."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to forget."},
            "dry_run": {"type": "boolean", "description": "Preview only, don't delete (default: true)."},
            "limit": {"type": "integer", "description": "Max memories to delete (default: 5)."},
        },
        "required": ["query"],
    },
}

SEARCH_SCHEMA = {
    "name": "agentmem_search",
    "description": (
        "Deep semantic search across all memory tiers with hybrid RRF ranking. "
        "More thorough than recall — includes graph expansion and cross-session context."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "limit": {"type": "integer", "description": "Max results (default: 10)."},
        },
        "required": ["query"],
    },
}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class AgentMemProvider(MemoryProvider):
    """AgentMem local persistent memory with heat-tiered recall."""

    def __init__(self):
        self._base_url = _DEFAULT_BASE_URL
        self._api_key = ""
        self._session_id = ""
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread = None
        self._sync_thread = None
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

    @property
    def name(self) -> str:
        return "agentmem"

    def is_available(self) -> bool:
        cfg = _load_config()
        base_url = cfg.get("base_url", _DEFAULT_BASE_URL).rstrip("/")
        result = _http_get(f"{base_url}/health", api_key=cfg.get("api_key", ""), timeout=3.0)
        return result is not None and result.get("status") == "ok"

    def get_config_schema(self):
        return [
            {
                "key": "base_url",
                "description": "AgentMem REST API URL",
                "default": _DEFAULT_BASE_URL,
                "required": False,
            },
            {
                "key": "api_key",
                "description": "AgentMem API key (leave empty for local mode)",
                "secret": True,
                "required": False,
                "env_var": "AGENTMEM_API_KEY",
            },
        ]

    def save_config(self, values: dict, hermes_home: str) -> None:
        from pathlib import Path
        config_path = Path(hermes_home) / "agentmem.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        config_path.write_text(json.dumps(existing, indent=2))

    def initialize(self, session_id: str, **kwargs) -> None:
        cfg = _load_config()
        self._base_url = cfg.get("base_url", _DEFAULT_BASE_URL).rstrip("/")
        self._api_key = cfg.get("api_key", "")
        self._session_id = session_id

        # Start session in agentmem
        result = _http_post(
            f"{self._base_url}/session/start",
            {"session_id": session_id},
            api_key=self._api_key,
            timeout=5.0,
        )
        if result and result.get("context"):
            with self._prefetch_lock:
                self._prefetch_result = result["context"]

    def system_prompt_block(self) -> str:
        return (
            "# AgentMem Memory\n"
            "Active. Local persistent memory with heat-tiered recall across sessions.\n"
            "Use agentmem_recall to find memories, agentmem_remember to store facts, "
            "agentmem_search for deep semantic search, agentmem_forget to remove outdated info.\n"
            "Memory is automatically synced each turn — no manual save needed."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## AgentMem Memory\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._is_breaker_open():
            return

        sid = session_id or self._session_id

        def _run():
            try:
                result = _http_post(
                    f"{self._base_url}/recall",
                    {
                        "query": query,
                        "session_id": sid,
                        "memory_limit_number": 8,
                        "include_tools": True,
                        "enable_hyde": True,
                    },
                    api_key=self._api_key,
                    timeout=8.0,
                )
                if result and result.get("prependContext"):
                    with self._prefetch_lock:
                        self._prefetch_result = result["prependContext"]
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("AgentMem prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="agentmem-prefetch")
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if self._is_breaker_open():
            return

        sid = session_id or self._session_id

        def _sync():
            try:
                _http_post(
                    f"{self._base_url}/store",
                    {
                        "messages": [
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": assistant_content},
                        ],
                        "session_id": sid,
                    },
                    api_key=self._api_key,
                    timeout=5.0,
                )
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.warning("AgentMem sync failed: %s", e)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)

        self._sync_thread = threading.Thread(target=_sync, daemon=True, name="agentmem-sync")
        self._sync_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [RECALL_SCHEMA, REMEMBER_SCHEMA, FORGET_SCHEMA, SEARCH_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if self._is_breaker_open():
            return json.dumps({
                "error": "AgentMem API temporarily unavailable (multiple consecutive failures). Will retry automatically."
            })

        if tool_name == "agentmem_recall":
            query = args.get("query", "")
            if not query:
                return tool_error("Missing required parameter: query")
            limit = args.get("limit", 10)
            result = _http_post(
                f"{self._base_url}/recall",
                {
                    "query": query,
                    "session_id": self._session_id,
                    "memory_limit_number": limit,
                    "include_tools": True,
                    "enable_hyde": True,
                },
                api_key=self._api_key,
                timeout=10.0,
            )
            if result is None:
                self._record_failure()
                return tool_error("AgentMem recall request failed")
            self._record_success()
            context = result.get("prependContext", "")
            if not context:
                return json.dumps({"result": "No relevant memories found."})
            return json.dumps({"result": context})

        elif tool_name == "agentmem_remember":
            content = args.get("content", "")
            if not content:
                return tool_error("Missing required parameter: content")
            category = args.get("category", "fact")
            result = _http_post(
                f"{self._base_url}/remember",
                {
                    "content": content,
                    "type": category,
                    "session_id": self._session_id,
                },
                api_key=self._api_key,
                timeout=5.0,
            )
            if result is None:
                self._record_failure()
                return tool_error("AgentMem remember request failed")
            self._record_success()
            return json.dumps({"result": "Fact stored.", "id": result.get("id", "")})

        elif tool_name == "agentmem_forget":
            query = args.get("query", "")
            if not query:
                return tool_error("Missing required parameter: query")
            dry_run = args.get("dry_run", True)
            limit = args.get("limit", 5)
            result = _http_post(
                f"{self._base_url}/forget",
                {
                    "query": query,
                    "dry_run": dry_run,
                    "limit": limit,
                    "session_id": self._session_id,
                },
                api_key=self._api_key,
                timeout=5.0,
            )
            if result is None:
                self._record_failure()
                return tool_error("AgentMem forget request failed")
            self._record_success()
            return json.dumps(result)

        elif tool_name == "agentmem_search":
            query = args.get("query", "")
            if not query:
                return tool_error("Missing required parameter: query")
            limit = args.get("limit", 10)
            result = _http_post(
                f"{self._base_url}/search",
                {
                    "query": query,
                    "limit": limit,
                    "session_id": self._session_id,
                    "format": "compact",
                },
                api_key=self._api_key,
                timeout=10.0,
            )
            if result is None:
                self._record_failure()
                return tool_error("AgentMem search request failed")
            self._record_success()
            results = result.get("results", [])
            if not results:
                return json.dumps({"result": "No relevant memories found."})
            return json.dumps({"results": results, "count": len(results)})

        return tool_error(f"Unknown tool: {tool_name}")

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Compress session to long-term memory at session end."""
        def _do_compress():
            try:
                _http_post(
                    f"{self._base_url}/session/end",
                    {"session_id": self._session_id},
                    api_key=self._api_key,
                    timeout=15.0,
                )
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.warning("AgentMem session compress failed: %s", e)

        t = threading.Thread(target=_do_compress, daemon=True, name="agentmem-compress")
        t.start()

    def on_session_switch(self, new_session_id: str, *, parent_session_id: str = "",
                          reset: bool = False, **kwargs) -> None:
        if reset:
            self.on_session_end([])
        self._session_id = new_session_id
        _http_post(
            f"{self._base_url}/session/start",
            {"session_id": new_session_id},
            api_key=self._api_key,
            timeout=5.0,
        )

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        return ""

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)

    # -- Circuit breaker ----------------------------------------------------

    def _is_breaker_open(self) -> bool:
        if self._consecutive_failures < _BREAKER_THRESHOLD:
            return False
        if time.monotonic() >= self._breaker_open_until:
            self._consecutive_failures = 0
            return False
        return True

    def _record_success(self):
        self._consecutive_failures = 0

    def _record_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= _BREAKER_THRESHOLD:
            self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
            logger.warning(
                "AgentMem circuit breaker tripped after %d consecutive failures. "
                "Pausing API calls for %ds.",
                self._consecutive_failures, _BREAKER_COOLDOWN_SECS,
            )


def register(ctx) -> None:
    """Register AgentMem as a memory provider plugin."""
    ctx.register_memory_provider(AgentMemProvider())
