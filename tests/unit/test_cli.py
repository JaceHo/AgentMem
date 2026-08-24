"""Regression tests for the standalone AgentMem CLI."""

from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path


def _load_cli():
    path = Path(__file__).parents[2] / "cli" / "agentmem"
    loader = SourceFileLoader("agentmem_cli_test", str(path))
    spec = spec_from_loader(loader.name, loader)
    assert spec is not None
    module = module_from_spec(spec)
    loader.exec_module(module)
    return module


def test_default_timeout_allows_normal_recall_latency(monkeypatch):
    monkeypatch.delenv("AGENTMEM_TIMEOUT", raising=False)

    cli = _load_cli()

    assert cli.TIMEOUT == 30


def test_store_uses_current_messages_contract(monkeypatch, capsys):
    cli = _load_cli()
    captured = {}

    def fake_req(method, path, body=None):
        captured.update(method=method, path=path, body=body)
        return 200, {"status": "queued"}

    monkeypatch.setattr(cli, "_req", fake_req)

    cli.cmd_store(["remember", "this"])

    assert captured == {
        "method": "POST",
        "path": "/store",
        "body": {
            "messages": [{"role": "user", "content": "remember this"}],
            "session_id": "agentmem-cli",
        },
    }
    assert '"status": "queued"' in capsys.readouterr().out
