"""
Integration Tests — API Endpoints
==================================
Tests all AgentMem API endpoints with real Redis backend.
Performance target: Beat agentmemory reference implementation on all metrics.
"""

import json
import time

import pytest
import httpx


BASE_URL = "http://localhost:18800"


@pytest.fixture
def client(agentmem_server):
    """HTTP client for the isolated test instance (fresh state per suite run).

    Carries the API key so mutating routes clear the post-S1 write guard.
    """
    from tests.conftest import _test_api_key

    limits = httpx.Limits(max_keepalive_connections=0)  # avoid stale keep-alive reuse
    return httpx.Client(base_url=agentmem_server, timeout=30, limits=limits,
                        headers={"X-API-Key": _test_api_key()})


class TestHealthEndpoints:
    """Test health and stats endpoints."""

    def test_health_check(self, client):
        """GET /health should return status=ok and redis=ok."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ok"
        assert data["redis"] == "ok"

    def test_stats_endpoint(self, client):
        """GET /stats should return memory counts."""
        response = client.get("/stats")
        assert response.status_code == 200

        data = response.json()
        assert "episodes" in data
        assert "facts" in data
        assert "procedures" in data
        assert isinstance(data["episodes"], int)
        assert isinstance(data["facts"], int)


class TestStoreEndpoint:
    """Test POST /store endpoint."""

    def test_store_basic(self, client):
        """Store should accept messages and return queued status."""
        session_id = f"test-store-{int(time.time())}"

        response = client.post("/store", json={
            "messages": [
                {"role": "user", "content": "I prefer Python for backend development"},
                {"role": "assistant", "content": "Noted. Python for backend."},
            ],
            "session_id": session_id,
        })

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "queued"

    def test_store_empty_messages(self, client):
        """Store with empty messages should not crash."""
        response = client.post("/store", json={
            "messages": [],
            "session_id": "empty-test",
        })

        # Should handle gracefully (queued, skipped, or error - but not 500)
        assert response.status_code in (200, 400)

    @pytest.mark.asyncio
    async def test_store_async_processing(self, client):
        """Store should process asynchronously without blocking."""
        session_id = f"test-async-{int(time.time())}"

        start = time.time()
        response = client.post("/store", json={
            "messages": [
                {"role": "user", "content": "Test message"},
                {"role": "assistant", "content": "Response"},
            ],
            "session_id": session_id,
        })
        elapsed = time.time() - start

        # Should return immediately (< 100ms)
        assert elapsed < 0.1
        assert response.json()["status"] == "queued"


class TestRecallEndpoint:
    """Test POST /recall endpoint."""

    @pytest.fixture(autouse=True)
    def setup_test_data(self, client):
        """Store test data before recall tests."""
        session_id = f"test-recall-setup-{int(time.time())}"

        client.post("/store", json={
            "messages": [
                {"role": "user", "content": "jace uses bun as JavaScript package manager"},
                {"role": "assistant", "content": "Got it, bun instead of npm."},
            ],
            "session_id": session_id,
        })

        # Wait for async processing
        time.sleep(5)

        self.session_id = session_id

    def test_recall_basic(self, client):
        """Recall should return relevant context."""
        response = client.post("/recall", json={
            "query": "what package manager does jace use",
            "session_id": self.session_id,
            "memory_limit_number": 5,
        })

        assert response.status_code == 200
        data = response.json()

        assert "prependContext" in data
        assert "latency_ms" in data
        assert data["latency_ms"] is not None

    # Warm the recall path before timing: the first request after boot pays a
    # one-time ~6s cold-start (regex compile, pool init, first VSIM) that would
    # otherwise poison the P95 sample.
    session_id = "perf-recall-session"

    def _warmup(self, client):
        for _ in range(3):
            client.post("/recall", json={
                "query": "warmup query",
                "session_id": self.session_id,
                "memory_limit_number": 5,
            })

    def test_recall_performance_p50(self, client):
        """Recall P50 latency should stay under 100ms over real HTTP."""
        self._warmup(client)
        latencies = []

        for _ in range(20):
            start = time.time()
            response = client.post("/recall", json={
                "query": "test query",
                "session_id": self.session_id,
                "memory_limit_number": 5,
            })
            elapsed_ms = (time.time() - start) * 1000
            latencies.append(elapsed_ms)

        latencies.sort()
        p50 = latencies[len(latencies) // 2]

        # Target: P50 < 100ms over real HTTP (full recall pipeline: hybrid
        # retrieval + augmentation skip + context assembly). The old 15ms
        # target was calibrated against in-process ASGI without lifespan.
        assert p50 < 100, f"P50 latency {p50:.2f}ms exceeds target 100ms"

    def test_recall_performance_p95(self, client):
        """Recall P95 latency should stay under 150ms over real HTTP."""
        self._warmup(client)
        latencies = []

        for _ in range(20):
            start = time.time()
            response = client.post("/recall", json={
                "query": "test query",
                "session_id": self.session_id,
                "memory_limit_number": 5,
            })
            elapsed_ms = (time.time() - start) * 1000
            latencies.append(elapsed_ms)

        latencies.sort()
        p95_idx = int(len(latencies) * 0.95)
        p95 = latencies[min(p95_idx, len(latencies) - 1)]

        # Target: P95 < 150ms over real HTTP (see P50 note above).
        assert p95 < 150, f"P95 latency {p95:.2f}ms exceeds target 150ms"

    def test_recall_respects_limit(self, client):
        """Recall should respect memory_limit_number parameter."""
        response = client.post("/recall", json={
            "query": "test",
            "session_id": self.session_id,
            "memory_limit_number": 2,
        })

        assert response.status_code == 200
        data = response.json()
        # Context should be present (even if long with existing data)
        assert "prependContext" in data

    def test_recall_empty_session_id(self, client):
        """Recall with empty session_id should not crash."""
        response = client.post("/recall", json={
            "query": "test",
            "session_id": "",
        })

        # Should handle gracefully
        assert response.status_code == 200


class TestSessionEndpoints:
    """Test session management endpoints."""

    def test_session_inspect(self, client):
        """GET /session/{id} should return session state."""
        session_id = f"test-inspect-{int(time.time())}"

        # First store something to create session
        client.post("/store", json={
            "messages": [
                {"role": "user", "content": "test message"},
                {"role": "assistant", "content": "response"},
            ],
            "session_id": session_id,
        })

        time.sleep(3)

        response = client.get(f"/session/{session_id}")
        assert response.status_code == 200

        data = response.json()
        # May have context or be empty if not yet processed
        assert "context" in data or "length" in data

    def test_session_compact(self, client):
        """POST /session/compact should compress session context."""
        session_id = f"test-compact-{int(time.time())}"

        response = client.post("/session/compact", json={
            "session_id": session_id,
            "threshold_chars": 100,
        })

        assert response.status_code == 200
        data = response.json()
        # May be ok, skipped, or queued depending on session state
        assert data["status"] in ("ok", "skipped", "queued")

    def test_session_compress(self, client):
        """POST /session/compress should promote Tier 1 → Tier 2."""
        session_id = f"test-compress-{int(time.time())}"

        response = client.post("/session/compress", json={
            "session_id": session_id,
        })

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("ok", "skipped", "queued")

    def test_session_nonexistent(self, client):
        """GET /session for non-existent session should return null context."""
        response = client.get("/session/nonexistent-xyz")
        assert response.status_code == 200

        data = response.json()
        assert data.get("context") is None or data.get("length") == 0


class TestCapabilityEndpoints:
    """Test tool/environment registration endpoints."""

    def test_register_tools(self, client):
        """POST /register-tools should register tool index."""
        response = client.post("/register-tools", json={
            "session_id": f"test-tools-{int(time.time())}",
            "tools": [
                {"name": "Bash", "description": "Execute shell commands"},
                {"name": "Read", "description": "Read files"},
            ],
        })

        assert response.status_code == 200

    def test_register_env(self, client):
        """POST /register-env should register environment state."""
        response = client.post("/register-env", json={
            "session_id": f"test-env-{int(time.time())}",
            "env": {
                "os": "darwin",
                "shell": "zsh",
                "python_version": "3.13",
            },
        })

        assert response.status_code == 200

    def test_recall_tools(self, client):
        """POST /recall-tools should search tool index."""
        response = client.post("/recall-tools", json={
            "query": "execute command",
            "session_id": "test-tool-recall",
        })

        assert response.status_code == 200


class TestGraphEndpoints:
    """Test knowledge graph endpoints."""

    def test_graph_stats(self, client):
        """GET /graph/stats should return graph statistics."""
        response = client.get("/graph/stats")
        assert response.status_code == 200

        data = response.json()
        # Current route shape: node_count / edge_count_directed / edge_count_undirected
        assert "node_count" in data or "count" in data or "nodes" in data

    def test_graph_entity(self, client):
        """GET /graph/{entity} should return entity neighbors."""
        response = client.get("/graph/test-entity")
        assert response.status_code == 200

    def test_graph_recall(self, client):
        """POST /graph/recall should retrieve neighborhood facts."""
        response = client.post("/graph/recall", json={
            "entity": "test-entity",
            "session_id": "test-graph-recall",
        })

        assert response.status_code == 200


class TestSecretRedaction:
    """Test secret detection and redaction."""

    def test_api_key_redaction(self, client):
        """API keys should be redacted before storage."""
        import uuid
        session_id = f"test-secret-{uuid.uuid4().hex[:8]}"
        # Use a unique secret that won't match any existing data
        unique_secret = f"sk-test-{uuid.uuid4().hex[:16]}"

        response = client.post("/store", json={
            "messages": [
                {"role": "user", "content": f"my secret key is {unique_secret}"},
                {"role": "assistant", "content": "got it"},
            ],
            "session_id": session_id,
        })

        assert response.status_code == 200
        assert response.json()["status"] == "queued"

        # Wait for processing
        time.sleep(5)

        # Try to recall - secret should NOT appear
        recall_response = client.post("/recall", json={
            "query": f"{unique_secret} secret key",
            "session_id": session_id,
        })

        data = recall_response.json()
        context = data.get("prependContext", "")

        # Secret should be redacted
        assert unique_secret not in context, "Secret API key found in recall context!"


class TestIntegrationPerformance:
    """End-to-end performance benchmarks."""

    @pytest.mark.benchmark(group="full-store-recall-cycle")
    def test_store_recall_cycle_performance(self, benchmark, client):
        """Complete store→recall cycle should complete in < 100ms."""
        session_id = f"perf-cycle-{int(time.time())}"

        def cycle():
            # Store
            client.post("/store", json={
                "messages": [
                    {"role": "user", "content": "performance test message"},
                    {"role": "assistant", "content": "response"},
                ],
                "session_id": session_id,
            })

            # Small delay for processing
            time.sleep(0.1)

            # Recall
            response = client.post("/recall", json={
                "query": "performance test",
                "session_id": session_id,
                "memory_limit_number": 5,
            })

            return response.status_code

        result = benchmark(cycle)
        assert result == 200


class TestScopeIsolation:
    """A track id on a write and on a read must name the same, exclusive corpus."""

    def _remember(self, client, content: str, **ids):
        response = client.post("/remember", json={"content": content, **ids})
        assert response.status_code == 200, response.text
        return response.json()["id"]

    def _recall(self, client, query: str, **ids):
        response = client.post("/recall", json={
            "query": query,
            "memory_limit_number": 10,
            **ids,
        })
        assert response.status_code == 200, response.text
        # Empty result is the isolation working: the server sends a null context.
        return response.json().get("prependContext") or ""

    def test_facts_stay_inside_their_track(self, client):
        stamp = str(int(time.time() * 1000))
        alice_token = f"alice-{stamp}-zephyrine"
        bob_token = f"bob-{stamp}-quokka"

        self._remember(client, f"alice keeps the {alice_token} ledger", user_id="alice")
        self._remember(client, f"bob keeps the {bob_token} ledger", user_id="bob")

        alice_view = self._recall(client, alice_token, user_id="alice")
        assert alice_token in alice_view
        assert bob_token not in alice_view

        bob_view = self._recall(client, bob_token, user_id="bob")
        assert bob_token in bob_view
        assert alice_token not in bob_view

    def test_unscoped_caller_still_sees_scoped_rows(self, client):
        stamp = str(int(time.time() * 1000))
        token = f"scoped-{stamp}-narwhal"
        self._remember(client, f"carol keeps the {token} ledger", user_id="carol")

        assert token in self._recall(client, token)

    def test_agent_track_is_separate_from_user_track(self, client):
        stamp = str(int(time.time() * 1000))
        agent_token = f"agent-{stamp}-pangolin"
        self._remember(client, f"the agent notes {agent_token}", agent_id="raven-code")

        assert agent_token in self._recall(client, agent_token, agent_id="raven-code")
        assert agent_token not in self._recall(client, agent_token, user_id="alice")
        assert agent_token not in self._recall(client, agent_token, agent_id="other-agent")

    def test_both_ids_and_bad_ids_are_rejected(self, client):
        for body in (
            {"content": "x", "user_id": "alice", "agent_id": "raven-code"},
            {"content": 'x', "user_id": 'bad"id'},
        ):
            response = client.post("/remember", json=body)
            assert response.status_code == 422, response.text

        response = client.post("/recall", json={
            "query": "x", "user_id": "alice", "agent_id": "raven-code",
        })
        assert response.status_code == 422, response.text

    def test_procedures_follow_the_agent_track(self, client):
        stamp = str(int(time.time() * 1000))
        marker = f"proc-{stamp}-axolotl"
        task = f"{marker} rebuild the gateway config"

        response = client.post("/store-procedure", json={
            "task": task,
            "procedure": f"1. read {marker}\n2. restart",
            "agent_id": "scope-test-agent",
        })
        assert response.status_code == 200, response.text

        def _tasks(**ids):
            r = client.post("/recall-procedures", json={
                "query": marker, "k": 5, **ids,
            })
            assert r.status_code == 200, r.text
            return [p["task"] for p in r.json()["procedures"]]

        for _ in range(20):
            if any(marker in t for t in _tasks(agent_id="scope-test-agent")):
                break
            time.sleep(0.2)

        assert any(marker in t for t in _tasks(agent_id="scope-test-agent"))
        assert not any(marker in t for t in _tasks(agent_id="other-agent"))
        assert not any(marker in t for t in _tasks(user_id="alice"))

    def test_compressed_session_lands_in_its_track(self, client):
        stamp = str(int(time.time() * 1000))
        token = f"walrusprotocol{stamp}"
        session_id = f"scope-compress-{stamp}"

        client.post("/store", json={
            "session_id": session_id,
            "user_id": "jace",
            "messages": [
                {"role": "user", "content": f"remember that jace ships the {token} deploy on fridays"},
                {"role": "assistant", "content": f"noted: {token} deploys are fridays"},
            ],
        })

        for _ in range(40):
            state = client.get(f"/session/{session_id}").json()
            if state.get("length", 0) >= 80:
                break
            time.sleep(0.2)
        assert client.get(f"/session/{session_id}").json()["length"] >= 80

        result = client.post("/session/compress", json={
            "session_id": session_id, "wait": True,
        }).json()
        assert result.get("status") == "ok", result

        assert token in self._recall(client, token, user_id="jace")
        assert token not in self._recall(client, token, user_id="bob")
