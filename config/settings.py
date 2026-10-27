"""
Centralized configuration for AgentMem with environment variable support.

All magic numbers and tunable parameters are defined here with sensible defaults.
Override via environment variables or .env file.

Usage:
    from config.settings import settings
    
    if stores >= settings.auto_consolidate_every:
        await consolidate()
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """AgentMem configuration with environment variable overrides."""
    
    model_config = SettingsConfigDict(
        env_prefix="AGENTMEM_",
        env_file=".env",
        extra="ignore"
    )
    
    # ── Service Configuration ──────────────────────────────────────────────
    host: str = Field(default="0.0.0.0", description="Host to bind")
    port: int = Field(default=18800, description="Port to listen on")
    app_version: str = Field(default="1.1.0", description="Application version")
    # LAN default: auth OFF. Set AGENTMEM_AUTH=1 (and AGENTMEM_API_KEY) only when
    # exposing beyond a trusted LAN. A key alone does not enable the write gate.
    auth: bool = Field(
        default=False,
        description="Require X-API-Key on write endpoints (AGENTMEM_AUTH=1)",
    )
    api_key: str | None = Field(
        default=None,
        description="API key for write endpoints when auth=True (AGENTMEM_API_KEY)",
        json_schema_extra={"env": "AGENTMEM_API_KEY"},
    )
    
    # ── Redis Configuration ────────────────────────────────────────────────
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL"
    )
    
    # ── Embedding Configuration ───────────────────────────────────────────
    # LAN default path: leave unset → embedder auto-detects local aiserv
    # (http://127.0.0.1:4000/v1, key sk-aiserv-local-dev). Force with
    # EMBEDDING_PROVIDER=openai|local|ollama|hashing|gemini.
    embedding_provider: str = Field(
        default="",
        description="Embedding provider override: openai, local, ollama, hashing, gemini (empty = auto)"
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Model name for embeddings"
    )
    
    # ── Consolidation Configuration ───────────────────────────────────────
    auto_consolidate_every: int = Field(
        default=50,
        description="Trigger consolidation after N stored facts",
        json_schema_extra={"env": "AGENTMEM_AUTO_CONSOLIDATE_EVERY"}
    )
    periodic_consolidate_interval_hours: int = Field(
        default=1,
        description="Run consolidation check every N hours",
        json_schema_extra={"env": "AGENTMEM_CONSOLIDATE_INTERVAL"}
    )
    hard_prune_interval_hours: int = Field(
        default=24,
        description="Run hard prune every N hours",
        json_schema_extra={"env": "AGENTMEM_PRUNE_INTERVAL"}
    )
    
    # ── Session Management ─────────────────────────────────────────────────
    session_ttl_seconds: int = Field(
        default=14400,
        description="Session KV TTL in seconds (4 hours)",
        json_schema_extra={"env": "AGENTMEM_SESSION_TTL"}
    )
    session_compact_threshold_chars: int = Field(
        default=3000,
        description="Compact session when > N chars",
        json_schema_extra={"env": "AGENTMEM_COMPACT_THRESHOLD"}
    )
    session_overwrite_target_chars: int = Field(
        default=900,
        description="Target length after overwrite compaction",
        json_schema_extra={"env": "AGENTMEM_OVERWRITE_TARGET"}
    )
    
    # ── Retrieval Configuration ────────────────────────────────────────────
    default_token_budget: int = Field(
        default=1500,
        description="Default token budget for context injection",
        json_schema_extra={"env": "AGENTMEM_TOKEN_BUDGET"}
    )
    default_memory_limit: int = Field(
        default=6,
        description="Default number of memories to retrieve",
        json_schema_extra={"env": "AGENTMEM_MEMORY_LIMIT"}
    )
    dedup_similarity_threshold: float = Field(
        default=0.95,
        description="Similarity threshold for deduplication",
        json_schema_extra={"env": "AGENTMEM_DEDUP_THRESHOLD"}
    )
    bm25_top_k: int = Field(
        default=10,
        description="Top-k results to return from BM25 search",
        json_schema_extra={"env": "AGENTMEM_BM25_TOP_K"}
    )
    retrieval_top_k: int = Field(
        default=12,
        description="Top-k results to return from vector retrieval",
        json_schema_extra={"env": "AGENTMEM_RETRIEVAL_TOP_K"}
    )
    retrieval_max_facts: int = Field(
        default=50,
        description="Maximum number of fused facts to keep after ranking",
        json_schema_extra={"env": "AGENTMEM_RETRIEVAL_MAX_FACTS"}
    )
    graph_expansion_depth: int = Field(
        default=2,
        description="Max graph traversal depth for retrieval expansion",
        json_schema_extra={"env": "AGENTMEM_GRAPH_EXPANSION_DEPTH"}
    )
    graph_max_neighbors: int = Field(
        default=10,
        description="Max number of graph neighbors to return",
        json_schema_extra={"env": "AGENTMEM_GRAPH_MAX_NEIGHBORS"}
    )
    rrf_k_constant: int = Field(
        default=60,
        description="RRF smoothing constant",
        json_schema_extra={"env": "AGENTMEM_RRF_K"}
    )
    
    # ── Pruning Configuration ─────────────────────────────────────────────
    prune_importance_threshold: float = Field(
        default=0.05,
        description="Prune facts with importance below this threshold",
        json_schema_extra={"env": "AGENTMEM_PRUNE_THRESHOLD"}
    )
    
    # ── A-MAC Admission Gate ───────────────────────────────────────────────
    amac_threshold: float = Field(
        default=0.40,
        description="A-MAC admission gate threshold",
        json_schema_extra={"env": "AGENTMEM_AMAC_THRESHOLD"}
    )
    amac_weights_semantic_novelty: float = Field(
        default=0.25,
        description="Weight for semantic novelty factor"
    )
    amac_weights_entity_novelty: float = Field(
        default=0.15,
        description="Weight for entity novelty factor"
    )
    amac_weights_factual_confidence: float = Field(
        default=0.20,
        description="Weight for factual confidence factor"
    )
    amac_weights_temporal_signal: float = Field(
        default=0.10,
        description="Weight for temporal signal factor"
    )
    amac_weights_content_type_prior: float = Field(
        default=0.30,
        description="Weight for content type prior factor"
    )
    
    # ── Background Task Configuration ──────────────────────────────────────
    bg_task_limit: int = Field(
        default=50,
        description="Max concurrent background tasks",
        json_schema_extra={"env": "AGENTMEM_BG_TASK_LIMIT"}
    )
    bg_task_shutdown_timeout: float = Field(
        default=5.0,
        description="Timeout for task shutdown in seconds"
    )
    recall_concurrency: int = Field(
        default=8,
        description=(
            "Max concurrent embedding operations in the recall path. "
            "Embeddings run on a dedicated, bounded thread pool (separate "
            "from the default pool used by BM25 rebuilds) so the single-"
            "process server — which holds in-process BM25 index and "
            "TaskManager state that multi-worker would shard — can serve "
            "many concurrent recalls without saturating. Raise this to "
            "increase the recall concurrency ceiling (set via "
            "AGENTMEM_RECALL_CONCURRENCY)."
        ),
        json_schema_extra={"env": "AGENTMEM_RECALL_CONCURRENCY"}
    )
    
    # ── Trivial Message Filter ─────────────────────────────────────────────
    trivial_min_chars: int = Field(
        default=15,
        description="Skip messages shorter than N chars"
    )
    
    # ── BM25 Configuration ─────────────────────────────────────────────────
    bm25_rebuild_threshold: int = Field(
        default=10,
        description="Rebuild BM25 index when corpus changes by N entries"
    )
    bm25_search_multiplier: int = Field(
        default=2,
        description="Search 2x results then filter for top-k"
    )
    
    # ── LLM Configuration ──────────────────────────────────────────────────
    # LAN default: local aiserv gateway — no cloud key required.
    llm_provider: str = Field(
        default="aiserv/fast",
        description="LLM model id for extraction/summarization (aiserv role or OpenAI-compat id)"
    )
    llm_api_key: str | None = Field(
        default="sk-aiserv-local-dev",
        description="API key for LLM provider (LAN aiserv local-dev key by default)",
        json_schema_extra={"env": "AGENTMEM_LLM_API_KEY"}
    )
    llm_base_url: str | None = Field(
        default="http://127.0.0.1:4000/v1",
        description="Base URL for LLM API (LAN aiserv by default)",
        json_schema_extra={"env": "AGENTMEM_LLM_BASE_URL"}
    )
    llm_timeout_seconds: float = Field(
        default=30.0,
        description="Timeout for LLM API calls"
    )
    
    # ── Logging Configuration ──────────────────────────────────────────────
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR"
    )
    log_format: str = Field(
        default="%(asctime)s [mem] %(message)s",
        description="Log format string"
    )
    
    # ── Security Configuration ─────────────────────────────────────────────
    secret_redaction_enabled: bool = Field(
        default=True,
        description="Enable automatic secret redaction"
    )

# Singleton instance
settings = Settings()


def reload_settings() -> Settings:
    """Reload settings from environment (useful for testing)."""
    global settings
    settings = Settings()
    return settings
