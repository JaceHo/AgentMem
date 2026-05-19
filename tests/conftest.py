"""
Shared test fixtures and configuration.
========================================
Provides common fixtures for all test modules.
"""

import asyncio
import os
import time

import pytest
import redis
import httpx


# Force local embedding provider for tests (no external API calls)
os.environ["EMBEDDING_PROVIDER"] = "local"

# Reset the embedder singleton so tests always get the local provider,
# even if the module was previously imported with a different provider.
from core import embedder as _embedder
_embedder._provider = None
_embedder.DIMS = 384


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def redis_client():
    """Provide a clean Redis client for each test (db=15 for testing)."""
    r = redis.Redis(host="localhost", port=6379, db=15, decode_responses=True)
    r.flushdb()
    yield r
    r.flushdb()
    r.close()


@pytest.fixture
def async_redis_client():
    """Provide an async Redis client for async tests."""
    import redis.asyncio as aioredis
    
    async def _create():
        r = aioredis.Redis(host="localhost", port=6379, db=15, decode_responses=True)
        await r.flushdb()
        return r
    
    r = asyncio.get_event_loop().run_until_complete(_create())
    yield r
    
    async def _cleanup():
        await r.flushdb()
        await r.close()
    
    asyncio.get_event_loop().run_until_complete(_cleanup())


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


@pytest.fixture
async def agentmem_client():
    """
    Async HTTP client specifically for AgentMem API testing.
    
    Uses the same base URL as async_http_client but with better error handling
    and automatic cleanup for integration tests.
    """
    from main import app
    from httpx import AsyncClient, ASGITransport
    
    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://test")
    yield client
    await client.aclose()


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
