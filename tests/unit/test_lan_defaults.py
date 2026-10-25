"""LAN defaults: no cloud key, no auth, aiserv-local embedding path."""

from __future__ import annotations

import os

import pytest

import main as agentmem_main
from config.settings import Settings
from core import embedder as emb


class TestAuthLanDefault:
    def test_settings_auth_defaults_off(self, monkeypatch):
        monkeypatch.delenv("AGENTMEM_AUTH", raising=False)
        s = Settings(_env_file=None)
        assert s.auth is False

    def test_middleware_open_when_auth_flag_off(self, monkeypatch):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        app.middleware("http")(agentmem_main.api_key_guard)

        @app.post("/store")
        async def store():
            return {"ok": True}

        monkeypatch.setattr(agentmem_main.settings, "auth", False)
        monkeypatch.setattr(agentmem_main.settings, "api_key", "should-be-ignored")
        assert TestClient(app).post("/store").status_code == 200


class TestEmbedderLanDefaults:
    def test_lan_base_gets_local_dev_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:4000/v1")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        p = emb.OpenAIProvider()
        assert p.api_key == "sk-aiserv-local-dev"
        assert p.base_url.endswith("/v1")

    def test_cloud_base_still_requires_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            emb.OpenAIProvider()

    def test_detect_prefers_aiserv_when_reachable(self, monkeypatch):
        monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(emb, "_aiserv_reachable", lambda timeout=0.4: True)
        monkeypatch.setenv("AISERV_URL", "http://127.0.0.1:4000")
        p = emb._detect_provider()
        assert p.name == "openai"
        assert os.environ.get("OPENAI_API_KEY") == "sk-aiserv-local-dev"
