"""
agentmem connect — wire AgentMem into AI coding agents.

Ported from rohitg00/agentmemory's connect command.
Writes MCP server configs and hook entries to agent config files
so they use our Python backend as the memory service.

Supported agents:
  - claude-code   (MCP + hooks)
  - codex         (MCP via config.toml)
  - cursor        (MCP via .cursor/mcp.json)
  - gemini-cli    (MCP via config)
  - cline         (MCP via VS Code settings)
  - windsurf      (MCP via config)
  - roo-code      (MCP via VS Code settings)
  - hermes        (Memory provider plugin)
  - openclaw      (Lifecycle plugin)
  - trae          (MCP via ~/.trae/mcp.json)
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional


HOOKS_DIR = os.path.dirname(os.path.abspath(__file__))
AGENTMEM_URL = os.getenv("AGENTMEMORY_URL", "http://localhost:18800")
AGENTMEM_SECRET = os.getenv("AGENTMEMORY_SECRET", "")


def _backup(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    backup = path + ".bak"
    i = 0
    while os.path.exists(backup):
        backup = f"{path}.bak.{i}"
        i += 1
    shutil.copy2(path, backup)
    return backup


def _hooks_json_entry() -> dict:
    hooks_base = HOOKS_DIR
    return {
        "session-start": [{"command": f"python3 {hooks_base}/session_start.py"}],
        "post-tool-use": [{"command": f"python3 {hooks_base}/post_tool_use.py"}],
        "pre-tool-use": [{"command": f"python3 {hooks_base}/pre_tool_use.py"}],
        "session-end": [{"command": f"python3 {hooks_base}/session_end.py"}],
        "stop": [{"command": f"python3 {hooks_base}/stop.py"}],
        "pre-compact": [{"command": f"python3 {hooks_base}/pre_compact.py"}],
        "post-commit": [{"command": f"python3 {hooks_base}/post_commit.py"}],
        "post-tool-failure": [{"command": f"python3 {hooks_base}/post_tool_failure.py"}],
        "prompt-submit": [{"command": f"python3 {hooks_base}/prompt_submit.py"}],
        "notification": [{"command": f"python3 {hooks_base}/notification.py"}],
        "subagent-start": [{"command": f"python3 {hooks_base}/subagent_start.py"}],
        "subagent-stop": [{"command": f"python3 {hooks_base}/subagent_stop.py"}],
        "task-completed": [{"command": f"python3 {hooks_base}/task_completed.py"}],
    }


def _mcp_env_block() -> dict:
    env = {"AGENTMEMORY_URL": AGENTMEM_URL}
    if AGENTMEM_SECRET:
        env["AGENTMEMORY_SECRET"] = AGENTMEM_SECRET
    return env


# ── Claude Code ───────────────────────────────────────────────────────────────

def detect_claude_code() -> bool:
    return os.path.isdir(os.path.expanduser("~/.claude"))


def connect_claude_code(dry_run: bool = False, force: bool = False) -> str:
    claude_json = os.path.expanduser("~/.claude.json")
    hooks_json_path = os.path.expanduser("~/.claude/hooks.json")

    config = {}
    if os.path.exists(claude_json):
        with open(claude_json) as f:
            config = json.load(f)

    mcp_servers = config.get("mcpServers", {})
    already = "agentmem" in mcp_servers

    mcp_entry = {
        "command": "python3",
        "args": [os.path.abspath(os.path.join(HOOKS_DIR, "..", "mcp_server.py"))],
        "env": _mcp_env_block(),
    }

    if already and not force:
        return f"✓ claude-code (already wired in {claude_json})"

    if dry_run:
        return f"[dry-run] Would add mcpServers.agentmem to {claude_json}"

    backup = _backup(claude_json)
    mcp_servers["agentmem"] = mcp_entry
    config["mcpServers"] = mcp_servers
    with open(claude_json, "w") as f:
        json.dump(config, f, indent=2)

    hooks_entry = _hooks_json_entry()
    hooks_backup = _backup(hooks_json_path)
    with open(hooks_json_path, "w") as f:
        json.dump({"hooks": hooks_entry}, f, indent=2)

    result = f"✓ claude-code → {claude_json}"
    if backup:
        result += f" (backup: {backup})"
    result += f"\n  hooks → {hooks_json_path}"
    if hooks_backup:
        result += f" (backup: {hooks_backup})"
    result += "\n  Restart Claude Code (or /mcp) to pick up the new server."
    return result


# ── Codex CLI ─────────────────────────────────────────────────────────────────

def detect_codex() -> bool:
    return os.path.isdir(os.path.expanduser("~/.codex"))


def connect_codex(dry_run: bool = False, force: bool = False) -> str:
    codex_dir = os.path.expanduser("~/.codex")
    config_path = os.path.join(codex_dir, "config.toml")

    os.makedirs(codex_dir, exist_ok=True)

    env_lines = ""
    for k, v in _mcp_env_block().items():
        env_lines += f'\n{k} = "{v}"'

    toml_block = f'''
[mcp_servers.agentmem]
command = "python3"
args = ["{os.path.abspath(os.path.join(HOOKS_DIR, '..', 'mcp_server.py'))}"]
[mcp_servers.agentmem.env]{env_lines}
'''

    current = ""
    if os.path.exists(config_path):
        with open(config_path) as f:
            current = f.read()

    if "[mcp_servers.agentmem]" in current and not force:
        return f"✓ codex (already wired in {config_path})"

    if dry_run:
        return f"[dry-run] Would add [mcp_servers.agentmem] to {config_path}"

    backup = _backup(config_path)
    with open(config_path, "a") as f:
        f.write(toml_block)

    result = f"✓ codex → {config_path}"
    if backup:
        result += f" (backup: {backup})"
    return result


# ── Cursor ────────────────────────────────────────────────────────────────────

def detect_cursor() -> bool:
    return os.path.isdir(os.path.expanduser("~/.cursor"))


def connect_cursor(dry_run: bool = False, force: bool = False) -> str:
    cursor_dir = os.path.expanduser("~/.cursor")
    mcp_path = os.path.join(cursor_dir, "mcp.json")

    os.makedirs(cursor_dir, exist_ok=True)

    config = {}
    if os.path.exists(mcp_path):
        with open(mcp_path) as f:
            config = json.load(f)

    mcp_servers = config.get("mcpServers", {})
    already = "agentmem" in mcp_servers

    mcp_entry = {
        "type": "http",
        "url": f"{AGENTMEM_URL.rstrip('/')}/mcp",
    }
    if AGENTMEM_SECRET:
        mcp_entry["headers"] = {"X-API-Key": AGENTMEM_SECRET}

    if already and not force:
        return f"✓ cursor (already wired in {mcp_path})"

    if dry_run:
        return f"[dry-run] Would add mcpServers.agentmem to {mcp_path}"

    backup = _backup(mcp_path)
    mcp_servers["agentmem"] = mcp_entry
    config["mcpServers"] = mcp_servers
    with open(mcp_path, "w") as f:
        json.dump(config, f, indent=2)

    result = f"✓ cursor → {mcp_path}"
    if backup:
        result += f" (backup: {backup})"
    return result


# ── Gemini CLI ────────────────────────────────────────────────────────────────

def detect_gemini_cli() -> bool:
    return os.path.isfile(os.path.expanduser("~/.gemini/settings.json"))


def connect_gemini_cli(dry_run: bool = False, force: bool = False) -> str:
    settings_path = os.path.expanduser("~/.gemini/settings.json")

    config = {}
    if os.path.exists(settings_path):
        with open(settings_path) as f:
            config = json.load(f)

    mcp_servers = config.get("mcpServers", {})
    already = "agentmem" in mcp_servers

    mcp_entry = {
        "command": "python3",
        "args": [os.path.abspath(os.path.join(HOOKS_DIR, "..", "mcp_server.py"))],
        "env": _mcp_env_block(),
    }

    if already and not force:
        return f"✓ gemini-cli (already wired in {settings_path})"

    if dry_run:
        return f"[dry-run] Would add mcpServers.agentmem to {settings_path}"

    os.makedirs(os.path.dirname(settings_path), exist_ok=True)
    backup = _backup(settings_path)
    mcp_servers["agentmem"] = mcp_entry
    config["mcpServers"] = mcp_servers
    with open(settings_path, "w") as f:
        json.dump(config, f, indent=2)

    result = f"✓ gemini-cli → {settings_path}"
    if backup:
        result += f" (backup: {backup})"
    return result


# ── Cline / Roo Code (VS Code settings) ───────────────────────────────────────

def detect_cline() -> bool:
    vscode_dir = os.path.expanduser("~/Library/Application Support/Code/User")
    return os.path.isdir(vscode_dir)


def connect_cline(dry_run: bool = False, force: bool = False) -> str:
    vscode_dir = os.path.expanduser("~/Library/Application Support/Code/User")
    settings_path = os.path.join(vscode_dir, "settings.json")

    if not os.path.isdir(vscode_dir):
        return "⚠ cline (VS Code not detected)"

    config = {}
    if os.path.exists(settings_path):
        with open(settings_path) as f:
            config = json.load(f)

    key = "cline.mcpServers"
    mcp_servers = config.get(key, {})
    already = "agentmem" in mcp_servers

    mcp_entry = {
        "command": "python3",
        "args": [os.path.abspath(os.path.join(HOOKS_DIR, "..", "mcp_server.py"))],
        "env": _mcp_env_block(),
    }

    if already and not force:
        return f"✓ cline (already wired in VS Code settings)"

    if dry_run:
        return f"[dry-run] Would add cline.mcpServers.agentmem to VS Code settings"

    backup = _backup(settings_path)
    mcp_servers["agentmem"] = mcp_entry
    config[key] = mcp_servers
    with open(settings_path, "w") as f:
        json.dump(config, f, indent=2)

    result = f"✓ cline → VS Code settings"
    if backup:
        result += f" (backup: {backup})"
    return result


# ── Hermes Agent ──────────────────────────────────────────────────────────────

def detect_hermes() -> bool:
    return os.path.isdir(os.path.expanduser("~/.hermes"))


def connect_hermes(dry_run: bool = False, force: bool = False) -> str:
    """Install agentmem as a Hermes memory provider plugin + configure."""
    hermes_home = os.path.expanduser("~/.hermes")
    plugins_dir = os.path.join(hermes_home, "plugins", "agentmem")
    config_yaml = os.path.join(hermes_home, "config.yaml")
    project_plugins = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plugins", "hermes")

    # 1. Copy plugin files
    src_init = os.path.join(project_plugins, "__init__.py")
    src_yaml = os.path.join(project_plugins, "plugin.yaml")

    if not os.path.isfile(src_init):
        return f"❌ hermes: plugin source not found at {project_plugins}"

    already = os.path.isfile(os.path.join(plugins_dir, "__init__.py"))
    if already and not force:
        plugin_ok = "✓"
    else:
        if dry_run:
            return f"[dry-run] Would copy agentmem plugin to {plugins_dir}"
        os.makedirs(plugins_dir, exist_ok=True)
        shutil.copy2(src_init, os.path.join(plugins_dir, "__init__.py"))
        shutil.copy2(src_yaml, os.path.join(plugins_dir, "plugin.yaml"))
        plugin_ok = "✓"

    # 2. Write agentmem.json config
    agentmem_json = os.path.join(hermes_home, "agentmem.json")
    am_cfg = {"base_url": AGENTMEM_URL}
    if not os.path.exists(agentmem_json) or force:
        if not dry_run:
            with open(agentmem_json, "w") as f:
                json.dump(am_cfg, f, indent=2)

    # 3. Set memory.provider in config.yaml
    provider_set = False
    if os.path.isfile(config_yaml):
        with open(config_yaml) as f:
            content = f.read()
        # Check if already set
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("provider:") and "agentmem" in stripped:
                provider_set = True
                break

        if not provider_set and not dry_run:
            # Replace empty provider line
            import re
            new_content = re.sub(
                r"(\s+provider:\s*)(['\"]?)(\2)",
                r"\1agentmem",
                content,
            )
            # If the regex didn't match (provider: '' or provider: ""), try broader
            if new_content == content:
                new_content = re.sub(
                    r"(\s+provider:\s*).*",
                    r"\1agentmem",
                    content,
                )
            with open(config_yaml, "w") as f:
                f.write(new_content)
            provider_set = True

    result = f"{plugin_ok} hermes → {plugins_dir}"
    if provider_set:
        result += "\n  config.yaml: memory.provider = agentmem"
    result += f"\n  config: {agentmem_json}"
    result += "\n  Restart Hermes (or run `hermes memory setup`) to activate."
    return result


# ── OpenClaw ──────────────────────────────────────────────────────────────────

def detect_openclaw() -> bool:
    return os.path.isdir(os.path.expanduser("~/.openclaw"))


def connect_openclaw(dry_run: bool = False, force: bool = False) -> str:
    """Install agentmem as an OpenClaw lifecycle plugin + configure."""
    openclaw_home = os.path.expanduser("~/.openclaw")
    config_path = os.path.join(openclaw_home, "openclaw.json")
    project_plugin = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plugin")

    src_index = os.path.join(project_plugin, "index.js")
    src_manifest = os.path.join(project_plugin, "openclaw.plugin.json")
    src_package = os.path.join(project_plugin, "package.json")

    if not os.path.isfile(src_index):
        return f"❌ openclaw: plugin source not found at {project_plugin}"

    # 1. Copy plugin to ~/.openclaw/plugins/agentmem/
    plugin_dir = os.path.join(openclaw_home, "plugins", "agentmem")
    already = os.path.isfile(os.path.join(plugin_dir, "index.js"))
    if already and not force:
        plugin_ok = "✓"
    else:
        if dry_run:
            return f"[dry-run] Would copy agentmem plugin to {plugin_dir}"
        os.makedirs(plugin_dir, exist_ok=True)
        shutil.copy2(src_index, os.path.join(plugin_dir, "index.js"))
        shutil.copy2(src_manifest, os.path.join(plugin_dir, "openclaw.plugin.json"))
        shutil.copy2(src_package, os.path.join(plugin_dir, "package.json"))
        plugin_ok = "✓"

    # 2. Configure openclaw.json: enable plugin + set config
    if os.path.isfile(config_path):
        with open(config_path) as f:
            cfg = json.load(f)

        plugins = cfg.setdefault("plugins", {})
        entries = plugins.setdefault("entries", {})
        allow = plugins.setdefault("allow", [])

        plugin_id = "memos-local-openclaw-plugin"

        # Add to allow list
        if plugin_id not in allow:
            allow.append(plugin_id)

        # Set plugin entry
        entries[plugin_id] = {
            "enabled": True,
            "config": {
                "baseUrl": AGENTMEM_URL,
                "memoryLimitNumber": 8,
            },
        }

        if not dry_run:
            backup = _backup(config_path)
            with open(config_path, "w") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)

    result = f"{plugin_ok} openclaw → {plugin_dir}"
    result += f"\n  config: {config_path} (plugin enabled)"
    result += "\n  Restart OpenClaw to activate."
    return result


# ── Trae IDE ──────────────────────────────────────────────────────────────────

def detect_trae() -> bool:
    return os.path.isdir(os.path.expanduser("~/.trae"))


def _connect_trae_variant(trae_dir: str, label: str, dry_run: bool = False, force: bool = False) -> str:
    """Shared logic for Trae IDE and Trae CN — both use ~/.trae/mcp.json or ~/.trae-cn/mcp.json."""
    mcp_path = os.path.join(trae_dir, "mcp.json")
    project_root = os.path.dirname(os.path.abspath(__file__))
    mcp_script = os.path.join(project_root, "mcp_server.py")

    os.makedirs(trae_dir, exist_ok=True)

    config = {}
    if os.path.exists(mcp_path):
        with open(mcp_path) as f:
            config = json.load(f)

    mcp_servers = config.get("mcpServers", {})
    already = "agentmem" in mcp_servers

    # Use stdio transport — most reliable, no HTTP mount dependency
    mcp_entry = {
        "command": "uv",
        "args": ["run", "--project", project_root, "python", mcp_script],
    }
    if AGENTMEM_SECRET:
        mcp_entry["env"] = {"AGENTMEM_SECRET": AGENTMEM_SECRET}

    if already and not force:
        return f"✓ {label} (already wired in {mcp_path})"

    if dry_run:
        return f"[dry-run] Would add mcpServers.agentmem to {mcp_path}"

    backup = _backup(mcp_path)
    mcp_servers["agentmem"] = mcp_entry
    config["mcpServers"] = mcp_servers
    with open(mcp_path, "w") as f:
        json.dump(config, f, indent=2)

    result = f"✓ {label} → {mcp_path}"
    if backup:
        result += f" (backup: {backup})"
    result += "\n  Restart Trae IDE (or reload MCP in Settings) to activate."
    return result


def connect_trae(dry_run: bool = False, force: bool = False) -> str:
    """Configure AgentMem MCP server for Trae IDE via ~/.trae/mcp.json."""
    return _connect_trae_variant(os.path.expanduser("~/.trae"), "trae", dry_run, force)


# ── Trae CN IDE ───────────────────────────────────────────────────────────────

def detect_trae_cn() -> bool:
    return os.path.isdir(os.path.expanduser("~/.trae-cn"))


def connect_trae_cn(dry_run: bool = False, force: bool = False) -> str:
    """Configure AgentMem MCP server for Trae CN IDE via ~/.trae-cn/mcp.json."""
    return _connect_trae_variant(os.path.expanduser("~/.trae-cn"), "trae-cn", dry_run, force)


# ── Adapter registry ──────────────────────────────────────────────────────────

ADAPTERS = [
    {"name": "claude-code", "display": "Claude Code", "detect": detect_claude_code, "connect": connect_claude_code},
    {"name": "codex",       "display": "Codex CLI",   "detect": detect_codex,        "connect": connect_codex},
    {"name": "cursor",      "display": "Cursor",      "detect": detect_cursor,       "connect": connect_cursor},
    {"name": "gemini-cli",  "display": "Gemini CLI",  "detect": detect_gemini_cli,   "connect": connect_gemini_cli},
    {"name": "cline",       "display": "Cline",       "detect": detect_cline,        "connect": connect_cline},
    {"name": "hermes",      "display": "Hermes Agent", "detect": detect_hermes,      "connect": connect_hermes},
    {"name": "openclaw",    "display": "OpenClaw",     "detect": detect_openclaw,    "connect": connect_openclaw},
    {"name": "trae",        "display": "Trae IDE",     "detect": detect_trae,        "connect": connect_trae},
    {"name": "trae-cn",     "display": "Trae CN IDE",  "detect": detect_trae_cn,     "connect": connect_trae_cn},
]


def run_connect(agent_name: str = "", dry_run: bool = False, force: bool = False, all_agents: bool = False) -> None:
    print("\n🔧 agentmem connect\n")

    if all_agents:
        targets = [a for a in ADAPTERS if a["detect"]()]
    elif agent_name:
        targets = [a for a in ADAPTERS if a["name"] == agent_name.lower()]
        if not targets:
            print(f"❌ Unknown agent: {agent_name}")
            print(f"   Supported: {', '.join(a['name'] for a in ADAPTERS)}")
            sys.exit(1)
    else:
        detected = [a for a in ADAPTERS if a["detect"]()]
        if not detected:
            print("❌ No supported agents detected on this machine.")
            print(f"   Supported: {', '.join(a['name'] for a in ADAPTERS)}")
            sys.exit(1)
        targets = detected

    for adapter in targets:
        if not adapter["detect"]():
            print(f"⚠ {adapter['display']}: not detected (skipping)")
            continue
        try:
            result = adapter["connect"](dry_run=dry_run, force=force)
            print(result)
        except Exception as e:
            print(f"❌ {adapter['display']}: {e}")

    print("\nRestart any wired agent to pick up AgentMem.")
