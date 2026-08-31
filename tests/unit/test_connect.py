"""Regression tests for agent configuration wiring."""

import json
import tomllib
from pathlib import Path

import connect


def test_stdio_entry_uses_uv_and_project_paths():
    project_root = Path(connect.__file__).resolve().parent

    entry = connect._mcp_stdio_entry()

    assert Path(entry["command"]).name == "uv"
    assert entry["args"] == [
        "run",
        "--project",
        str(project_root),
        "python",
        str(project_root / "mcp_server.py"),
    ]
    assert Path(entry["args"][-1]).is_file()
    assert Path(connect.HOOKS_DIR) == project_root / "hooks"


def test_claude_hook_paths_point_to_existing_scripts():
    for registrations in connect._hooks_json_entry().values():
        for registration in registrations:
            script = registration["command"].removeprefix("python3 ")
            assert Path(script).is_file()


def test_connect_codex_repairs_stale_entry_and_preserves_other_servers(tmp_path, monkeypatch):
    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir()
    config_path = codex_dir / "config.toml"
    config_path.write_text(
        '''model = "gpt-test"

[mcp_servers.agentmem]
command = "python3"
args = ["/old/location/mcp_server.py"]
[mcp_servers.agentmem.env]
AGENTMEMORY_URL = "http://localhost:18800"

[mcp_servers.other]
command = "other-server"
'''
    )
    monkeypatch.setenv("HOME", str(tmp_path))

    result = connect.connect_codex()

    parsed = tomllib.loads(config_path.read_text())
    agentmem = parsed["mcp_servers"]["agentmem"]
    assert agentmem == connect._mcp_stdio_entry()
    assert parsed["mcp_servers"]["other"] == {"command": "other-server"}
    assert parsed["model"] == "gpt-test"
    assert "backup:" in result


def test_connect_codex_leaves_current_entry_unchanged(tmp_path, monkeypatch):
    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir()
    config_path = codex_dir / "config.toml"
    entry = connect._mcp_stdio_entry()
    config_path.write_text(
        "[mcp_servers.agentmem]\n"
        f"command = {json.dumps(entry['command'])}\n"
        f"args = [{', '.join(json.dumps(arg) for arg in entry['args'])}]\n"
        "[mcp_servers.agentmem.env]\n"
        f"AGENTMEMORY_URL = {json.dumps(entry['env']['AGENTMEMORY_URL'])}\n"
    )
    monkeypatch.setenv("HOME", str(tmp_path))

    before = config_path.read_text()
    result = connect.connect_codex()

    assert config_path.read_text() == before
    assert "already wired" in result
    assert not (codex_dir / "config.toml.bak").exists()
