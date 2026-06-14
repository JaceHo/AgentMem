"""
Application state singletons — single source of truth for all runtime state.

This module owns the runtime singletons (Redis client, embedder, store, etc.)
that are initialized during lifespan startup and accessed by route handlers
and service modules.

Route modules and services import from here instead of from main.py, breaking
the circular-import problem that occurs when routes import from the app module
that also registers them.
"""

import logging

from concurrency import AtomicCounter, AtomicFloat, TaskManager
from config.settings import settings
from core import embedder
from core import store as mem_store
from core import graph as graph_mod
from core import capability as cap_mod
from core import persona as persona_mod
from core import heat as heat_mod
from core import scene as scene_mod
from core import extractor
from core import summarizer
from core import retrieval_planner
from core import log_sse
from core.search import BM25Index, encode, vscan

log = logging.getLogger("mem")

# ── Runtime singletons (initialized during lifespan) ──────────────────────────

redis = None                          # aioredis.Redis — set during lifespan

# Module references — used by route modules to access core functionality
# without importing from main.py (avoids circular imports)
embedder = embedder
mem_store = mem_store
cap_mod = cap_mod
graph_mod = graph_mod
persona_mod = persona_mod
heat_mod = heat_mod
scene_mod = scene_mod
extractor = extractor
summarizer = summarizer
retrieval_planner = retrieval_planner
log_sse = log_sse

# ── Task manager for fire-and-forget background work ──────────────────────────
task_manager = TaskManager(max_concurrent=settings.bg_task_limit)

AUTO_CONSOLIDATE_EVERY = settings.auto_consolidate_every


def spawn(coro, name: str = "bg") -> None:
    """Fire-and-forget a coroutine with TaskManager supervision."""
    task_manager.spawn(coro, name=name)


# ── Thread-safe counters ──────────────────────────────────────────────────────
stores_since_consolidation = AtomicCounter()
periodic_prune_counter = AtomicCounter()
store_attempts = AtomicCounter()
store_successes = AtomicCounter()
store_skips = AtomicCounter()
store_errors = AtomicCounter()
store_latency_sum_ms = AtomicFloat()

# ── BM25 in-memory index ──────────────────────────────────────────────────────
bm25_index = BM25Index()

# ── Redis key constants (shared across services and routes) ───────────────────
PINNED_SESSION_KEY = "mem:pinned:session_summary"
CRYSTALLIZED_INDEX_KEY = "mem:crystallized_index"
