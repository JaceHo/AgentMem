"""
Shared test fixtures and configuration.
========================================
Provides common fixtures for all test modules.
"""

import asyncio
import os
import time
from pathlib import Path

import pytest
import redis
import httpx


# Force the offline hashing embedder for tests (no HF model download / network).
# It is a deterministic lexical embedder: similar texts -> high cosine,
# dissimilar -> low cosine.  Keeps the whole suite green offline.
os.environ["EMBEDDING_PROVIDER"] = "hashing"

# Tests and the isolated integration server must share one private Redis DB —
# tests that read Redis directly via mem_store.get_client() land here too,
# instead of silently reading the live service's db 0.
os.environ["REDIS_URL"] = "redis://localhost:6379/14"

# Reset the embedder singleton so tests always get the local provider,
# even if the module was previously imported with a different provider.
from core import embedder as _embedder
_embedder._provider = None
_embedder.DIMS = 384


def _test_api_key() -> str:
    """Resolve an API key header for test HTTP clients (optional on LAN).

    LAN default keeps ``AGENTMEM_AUTH`` off, so mutating routes accept requests
    with or without a key. When a developer opts into ``AGENTMEM_AUTH=1`` (or
    an older .env still expects the header), clients send the same token the
    server loaded from ``~/.agentmem_api_key`` / ``agentmem/.env``.

    No secret is committed: this reads from disk at test time only.
    """
    p = Path.home() / ".agentmem_api_key"
    if p.is_file():
        tok = p.read_text(encoding="utf-8", errors="replace").strip()
        if tok:
            return tok
    # Fallback: parse agentmem/.env (repo_root/.env).
    env_f = Path(__file__).resolve().parent.parent / ".env"
    if env_f.is_file():
        for line in env_f.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.startswith("AGENTMEM_API_KEY") and "=" in line:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


@pytest.fixture
def redis_client():
    """Provide a clean Redis client for each test (db=15 for testing)."""
    r = redis.Redis(host="localhost", port=6379, db=15, decode_responses=True)
    r.flushdb()
    yield r
    r.flushdb()
    r.close()


@pytest.fixture
async def async_redis_client():
    """Provide an async Redis client for async tests."""
    import redis.asyncio as aioredis
    
    r = aioredis.Redis(host="localhost", port=6379, db=15, decode_responses=True)
    await r.flushdb()
    yield r
    
    await r.flushdb()
    await r.aclose()


@pytest.fixture(autouse=True)
async def patch_main_redis(async_redis_client):
    """Automatically patch main._redis global variable for all async tests."""
    import main
    original_redis = main._redis
    main._redis = async_redis_client
    yield
    main._redis = original_redis


@pytest.fixture
def http_client():
    """Provide HTTP client for API testing."""
    client = httpx.Client(base_url="http://localhost:18800", timeout=15)
    yield client
    client.close()


@pytest.fixture
def async_http_client():
    """Provide async HTTP client for async API testing."""
    client = httpx.AsyncClient(base_url="http://localhost:18800", timeout=15)
    yield client
    
    async def cleanup():
        await client.aclose()
    
    asyncio.get_event_loop().run_until_complete(cleanup())


# ── Isolated integration server ──────────────────────────────────────────────
# Integration tests used to run against the live :18800 instance or an
# in-process ASGI app whose lifespan never ran (state.redis stayed None →
# 'NoneType' has no attribute 'execute_command'), and fixed session ids
# collided with real data. Spawning an ephemeral instance on its own port
# with a private Redis DB makes them deterministic and isolated.

TEST_PORT = int(os.getenv("AGENTMEM_TEST_PORT", "18990"))
TEST_REDIS_DB = 14


@pytest.fixture(scope="session")
def agentmem_server():
    """Ephemeral AgentMem instance: own port, private Redis DB, hashing
    embedder, quiet background loops. Fresh state for every suite run."""
    import subprocess
    import sys
    import time

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {
        **os.environ,
        "AGENTMEM_PORT": str(TEST_PORT),
        # core/store.py reads REDIS_URL (not AGENTMEM_REDIS_URL) — private db 14
        "REDIS_URL": f"redis://localhost:6379/{TEST_REDIS_DB}",
        "EMBEDDING_PROVIDER": "hashing",
        "AGENTMEM_CONSOLIDATE_INTERVAL": "100000",
        "AGENTMEM_PRUNE_INTERVAL": "100000",
        "AGENTMEM_AUTO_CONSOLIDATE_EVERY": "100000",
        # Synthetic test facts score 0.16-0.35 on A-MAC; default 0.40 filters
        # them all, leaving sessions empty and handoff tests meaningless.
        "AGENTMEM_AMAC_THRESHOLD": "0",
        # Recall augmentation (LLM planning) burns its whole 2.5s budget per
        # call in tests — skip it; augmentation itself is covered by unit tests.
        "AGENTMEM_RECALL_AUGMENT_BUDGET": "0.05",
        # Dead port: the store pipeline's LLM extraction (and every other
        # aiserv enhancement) fails in milliseconds and falls back to the
        # regex path — pipelines drop from 10-52s to well under a second.
        "AISERV_URL": "http://127.0.0.1:9",
    }
    # Strip external LLM/embedding credentials so every path uses the offline
    # hashing embedder + regex extraction: deterministic, fast, one vector dim.
    for k in ("ZAI_API_KEY", "ZAI_PROXY", "OPENAI_API_KEY", "OPENAI_BASE_URL",
              "GEMINI_API_KEY", "OLLAMA_BASE_URL"):
        env.pop(k, None)
    log_path = "/tmp/agentmem-test-server.log"
    logf = open(log_path, "w")
    # Fresh state BEFORE boot: the startup dimension guard inspects mem:facts —
    # leftover vectors with different dims force-switch the provider off hashing.
    redis.Redis(host="localhost", port=6379, db=TEST_REDIS_DB).flushdb()
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1",
         "--port", str(TEST_PORT), "--log-level", "warning"],
        cwd=repo_root, env=env, stdout=logf, stderr=logf,
    )
    base = f"http://127.0.0.1:{TEST_PORT}"
    for _ in range(120):
        try:
            if httpx.get(f"{base}/health", timeout=2).status_code == 200:
                break
        except Exception:
            if proc.poll() is not None:
                break  # crashed — fail fast via the final raise
            time.sleep(0.5)
    else:
        proc.kill()
        raise RuntimeError(f"isolated test server failed to start — see {log_path}")
    if proc.poll() is not None:
        raise RuntimeError(f"isolated test server died at startup — see {log_path}")
    yield base
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()
    logf.close()
    redis.Redis(host="localhost", port=6379, db=TEST_REDIS_DB).flushdb()


@pytest.fixture
async def agentmem_client(agentmem_server):
    """
    Async HTTP client for the isolated test instance.

    Real HTTP against a real server process (lifespan ran, state.redis set) —
    the old in-process ASGITransport skipped lifespan startup, so every route
    reading api.state.redis saw None.
    """
    from httpx import AsyncClient

    # No keep-alive pooling: uvicorn's 5s idle timeout closes pooled sockets
    # between tests, and POSTing into a half-closed socket fails with
    # "Server disconnected without sending a response".
    limits = httpx.Limits(max_keepalive_connections=0)
    async with AsyncClient(base_url=agentmem_server, timeout=30, limits=limits,
                           headers={"X-API-Key": _test_api_key()}) as client:
        yield client


@pytest.fixture(scope="session")
async def kb_client():
    """
    In-process app WITH lifespan, for tests that monkeypatch core modules
    (KB tests patch kb._llm / encode — must run in the test process).

    Unlike the old bare ASGITransport client, the lifespan runs — so
    api.state.redis is bound (to the isolated REDIS_URL db) and routes work.
    The lifespan enters and exits inside its own dedicated task: anyio
    forbids exiting a cancel scope from a different task than the entry.
    """
    import asyncio

    from main import app
    from httpx import ASGITransport, AsyncClient

    import redis.asyncio as aioredis
    _kb_db = int(os.environ.get("REDIS_URL", "redis://localhost:6379").rsplit("/", 1)[-1])
    r = aioredis.Redis(host="localhost", port=6379, db=_kb_db)
    await r.flushdb()
    await r.aclose()

    started = asyncio.Event()
    stop = asyncio.Event()

    async def run_lifespan():
        async with app.router.lifespan_context(app):
            started.set()
            await stop.wait()

    task = asyncio.create_task(run_lifespan())
    try:
        await asyncio.wait_for(started.wait(), timeout=30)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test", timeout=30,
                              headers={"X-API-Key": _test_api_key()}) as client:
            yield client
    finally:
        stop.set()
        try:
            await asyncio.wait_for(task, timeout=15)
        except asyncio.TimeoutError:
            task.cancel()


@pytest.fixture
def sample_messages():
    """Sample conversation messages for testing."""
    return [
        {"role": "user", "content": "I prefer Python for backend development"},
        {"role": "assistant", "content": "Noted. Python is great for backend services."},
        {"role": "user", "content": "Also, I use Redis for caching and session storage"},
        {"role": "assistant", "content": "Good choice. Redis is fast and reliable."},
    ]


@pytest.fixture
def sample_session_id():
    """Generate a unique session ID for testing."""
    return f"test-session-{int(time.time())}"


@pytest.fixture
def clean_database(redis_client):
    """Ensure database is clean before and after test."""
    # Already cleaned by redis_client fixture
    yield


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "benchmark: mark test as performance benchmark"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically add markers based on test path."""
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        if "benchmark" in str(item.fspath):
            item.add_marker(pytest.mark.benchmark)
