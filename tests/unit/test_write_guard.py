"""Tests for the write-endpoint auth policy (main.api_key_guard).

Why these exist: the 2026-10-24 re-audit found the previous path-prefix rule left
DELETE /facts/{id}, DELETE /procedures/{id}, POST /facts/{id}/pin, POST /kb/remove,
POST /graph/edge, POST /facts/{id}/reinforce and every /session/* write unauthenticated
on a 0.0.0.0-bound service. The policy is now default-deny by method *when auth is
on*; LAN default keeps AGENTMEM_AUTH off so a key in .env alone does not gate writes.

Three layers, cheapest first:
  1. needs_write_auth  — pure policy matrix (no app, no redis)
  2. check_write_auth  — pure key comparison (auth_enabled=False = gate off)
  3. middleware        — the real main.api_key_guard on a minimal app, with
                         settings.api_key / settings.auth monkeypatched
"""

from __future__ import annotations

import importlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import main as agentmem_main


# ─── 1. policy matrix ────────────────────────────────────────────────────────

GATED = [
    # the routes the old prefix list missed — the reason this file exists
    ("DELETE", "/facts/01ABC"),
    ("DELETE", "/procedures/01ABC"),
    ("POST", "/facts/01ABC/pin"),
    ("POST", "/facts/01ABC/reinforce"),
    ("POST", "/shadow-index/backfill"),
    ("POST", "/proc-backfill-index"),
    ("POST", "/kb/remove"),
    ("POST", "/kb/ingest"),
    ("POST", "/graph/edge"),
    ("POST", "/session/start"),
    ("POST", "/session/end"),
    ("POST", "/session/commit"),
    ("POST", "/claude-bridge/sync"),
    ("POST", "/consolidate-pipeline"),
    ("POST", "/crystals/auto"),
    ("POST", "/enrich"),
    # routes the old prefix list did cover — must stay covered
    ("POST", "/store"),
    ("POST", "/store-procedure"),
    ("POST", "/register-env"),
    ("POST", "/register-tools"),
    ("POST", "/consolidate/hard-prune"),
    ("POST", "/crystallize"),
    ("POST", "/feedback"),
    ("POST", "/tool-feedback"),
    ("POST", "/procedure-feedback"),
    ("POST", "/observe"),
    ("POST", "/remember"),
    ("POST", "/forget"),
    ("POST", "/import"),
    ("POST", "/compress-file"),
    ("POST", "/record-tool-sequence"),
    ("POST", "/tool-graph/detect-meta-tools"),
    ("PUT", "/anything"),
    ("PATCH", "/anything"),
]

OPEN = [
    # reads by method
    ("GET", "/facts/01ABC/metadata"),
    ("GET", "/lifecycle/stats"),
    ("GET", "/shadow-index/stats"),
    ("GET", "/health"),
    ("GET", "/stats"),
    ("GET", "/"),
    ("HEAD", "/health"),
    ("OPTIONS", "/store"),
    # POST-as-read query routes
    ("POST", "/recall"),
    ("POST", "/recall-tools"),
    ("POST", "/recall-procedures"),
    ("POST", "/search"),
    ("POST", "/smart-search"),
    ("POST", "/smart_search"),
    ("POST", "/timeline"),
    ("POST", "/graph/recall"),
    ("POST", "/graph/traverse"),
    ("POST", "/context"),
    ("POST", "/file-context"),
    ("POST", "/patterns"),
    ("POST", "/summarize"),
    # MCP transport — must stay ungated or Cursor treats 401 as OAuth
    ("POST", "/mcp"),
    ("POST", "/mcp/"),
    ("POST", "/mcp/sse"),
    ("DELETE", "/mcp"),
]


class TestNeedsWriteAuth:
    @pytest.mark.parametrize("method,path", GATED)
    def test_mutations_are_gated(self, method, path):
        assert agentmem_main.needs_write_auth(method, path) is True

    @pytest.mark.parametrize("method,path", OPEN)
    def test_reads_stay_open(self, method, path):
        assert agentmem_main.needs_write_auth(method, path) is False

    def test_method_is_case_insensitive(self):
        assert agentmem_main.needs_write_auth("delete", "/procedures/x") is True
        assert agentmem_main.needs_write_auth("get", "/store") is False


class TestCheckWriteAuth:
    def test_lan_default_auth_off_allows_everything(self):
        # AGENTMEM_AUTH unset/false → open LAN, key irrelevant
        assert agentmem_main.check_write_auth(
            "DELETE", "/procedures/x", "", "k", auth_enabled=False
        ) is True

    def test_auth_on_without_configured_key_denies_writes(self):
        assert agentmem_main.check_write_auth(
            "POST", "/store", "", "", auth_enabled=True
        ) is False

    def test_correct_key_allows(self):
        assert agentmem_main.check_write_auth(
            "POST", "/store", "k", "k", auth_enabled=True
        ) is True

    def test_wrong_key_denied(self):
        assert agentmem_main.check_write_auth(
            "POST", "/store", "wrong", "k", auth_enabled=True
        ) is False

    def test_empty_key_denied_when_auth_on(self):
        assert agentmem_main.check_write_auth(
            "POST", "/store", "", "k", auth_enabled=True
        ) is False

    def test_read_ignores_key(self):
        assert agentmem_main.check_write_auth(
            "GET", "/facts/x", "", "k", auth_enabled=True
        ) is True
        assert agentmem_main.check_write_auth(
            "POST", "/recall", "", "k", auth_enabled=True
        ) is True


class TestMiddlewareWiring:
    """The real guard, on a minimal app — no lifespan, no prod redis."""

    def _client(self, monkeypatch, configured: str, *, auth: bool = True):
        app = FastAPI()
        app.middleware("http")(agentmem_main.api_key_guard)

        @app.get("/health")
        async def health():
            return {"ok": True}

        @app.post("/store")
        async def store():
            return {"stored": True}

        @app.delete("/procedures/{element_id}")
        async def drop(element_id: str):
            return {"deleted": element_id}

        @app.post("/recall")
        async def recall():
            return {"results": []}

        monkeypatch.setattr(agentmem_main.settings, "api_key", configured)
        monkeypatch.setattr(agentmem_main.settings, "auth", auth)
        return TestClient(app)

    def test_write_without_key_is_401(self, monkeypatch):
        c = self._client(monkeypatch, "secret-key", auth=True)
        assert c.post("/store").status_code == 401
        assert c.delete("/procedures/abc").status_code == 401

    def test_write_with_key_passes(self, monkeypatch):
        c = self._client(monkeypatch, "secret-key", auth=True)
        r = c.post("/store", headers={"X-API-Key": "secret-key"})
        assert r.status_code == 200
        r = c.delete("/procedures/abc", headers={"X-API-Key": "secret-key"})
        assert r.status_code == 200

    def test_reads_open_even_with_key_configured(self, monkeypatch):
        c = self._client(monkeypatch, "secret-key", auth=True)
        assert c.get("/health").status_code == 200
        assert c.post("/recall").status_code == 200

    def test_gate_off_when_auth_disabled(self, monkeypatch):
        # LAN default: key may exist in .env but AGENTMEM_AUTH unset → open
        c = self._client(monkeypatch, "secret-key", auth=False)
        assert c.post("/store").status_code == 200
        assert c.delete("/procedures/abc").status_code == 200

    def test_gate_off_when_unconfigured(self, monkeypatch):
        c = self._client(monkeypatch, "", auth=False)
        assert c.post("/store").status_code == 200
        assert c.delete("/procedures/abc").status_code == 200

    # NOTE: no importlib.reload test here on purpose. Reloading main re-executes
    # module state (new app object, fresh settings, lost conftest patches on
    # main._redis) — that is cross-test pollution, the exact failure mode this
    # suite was burned by before. Env-read semantics belong to pydantic; the
    # restart-is-required behaviour is an ops fact, not a unit-test subject.
