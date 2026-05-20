"""
Custom exception hierarchy for AgentMem.

Provides typed exceptions for better error handling and debugging.
All exceptions inherit from AgentMemError for easy catching.

Usage:
    try:
        await memory_service.store(messages)
    except StorageError as e:
        log.error(f"Failed to store: {e}", extra={"session_id": e.session_id})
        raise HTTPException(status_code=500, detail="Storage failed")
"""


class AgentMemError(Exception):
    """Base exception for all AgentMem errors."""
    
    def __init__(self, message: str = "", cause: Exception | None = None):
        super().__init__(message)
        self.message = message
        self.cause = cause


# ── Storage Errors ─────────────────────────────────────────────────────────

class StorageError(AgentMemError):
    """Failed to store memory (episode, fact, procedure)."""
    
    def __init__(self, message: str = "", session_id: str = "", cause: Exception | None = None):
        super().__init__(message, cause)
        self.session_id = session_id


class EpisodeStorageError(StorageError):
    """Failed to store episode."""
    pass


class FactStorageError(StorageError):
    """Failed to store fact."""
    pass


class ProcedureStorageError(StorageError):
    """Failed to store procedure."""
    pass


# ── Retrieval Errors ───────────────────────────────────────────────────────

class RetrievalError(AgentMemError):
    """Failed to retrieve memories."""
    
    def __init__(self, query: str = "", message: str = "", cause: Exception | None = None):
        super().__init__(message or f"Retrieval failed for query: {query}", cause)
        self.query = query


class VectorSearchError(RetrievalError):
    """Vector similarity search failed."""
    pass


class BM25SearchError(RetrievalError):
    """BM25 keyword search failed."""
    pass


class GraphRetrievalError(RetrievalError):
    """Knowledge graph retrieval failed."""
    pass


# ── Consolidation Errors ───────────────────────────────────────────────────

class ConsolidationError(AgentMemError):
    """Memory consolidation failed."""
    
    def __init__(self, phase: str = "", message: str = "", cause: Exception | None = None):
        super().__init__(message or f"Consolidation failed in phase: {phase}", cause)
        self.phase = phase


class DecayError(ConsolidationError):
    """Ebbinghaus decay calculation failed."""
    pass


class MergeError(ConsolidationError):
    """Near-duplicate merge failed."""
    pass


class PruneError(ConsolidationError):
    """Hard prune operation failed."""
    pass


# ── Extraction Errors ──────────────────────────────────────────────────────

class ExtractionError(AgentMemError):
    """Fact/procedure extraction failed."""
    
    def __init__(self, message: str = "", source: str = "", cause: Exception | None = None):
        super().__init__(message or f"Extraction failed from source: {source}", cause)
        self.source = source


class RegexExtractionError(ExtractionError):
    """Regex-based extraction failed."""
    pass


class LLMExtractionError(ExtractionError):
    """LLM-based extraction failed."""
    
    def __init__(self, message: str = "", provider: str = "", cause: Exception | None = None):
        super().__init__(message, source=f"llm:{provider}", cause=cause)
        self.provider = provider


# ── Embedding Errors ───────────────────────────────────────────────────────

class EmbeddingError(AgentMemError):
    """Text embedding failed."""
    
    def __init__(self, text_preview: str = "", message: str = "", cause: Exception | None = None):
        preview = text_preview[:50] + "..." if len(text_preview) > 50 else text_preview
        super().__init__(message or f"Embedding failed for: {preview!r}", cause)
        self.text_preview = text_preview


class ModelLoadError(EmbeddingError):
    """Failed to load embedding model."""
    pass


# ── Configuration Errors ───────────────────────────────────────────────────

class ConfigurationError(AgentMemError):
    """Configuration validation or loading failed."""
    
    def __init__(self, key: str = "", message: str = "", cause: Exception | None = None):
        super().__init__(message or f"Configuration error for key: {key}", cause)
        self.key = key


# ── Lifecycle Errors ───────────────────────────────────────────────────────

class AdmissionError(AgentMemError):
    """A-MAC admission gate decision failed."""
    
    def __init__(self, score: float = 0.0, threshold: float = 0.0, message: str = ""):
        msg = message or f"Admission rejected: score={score:.2f} < threshold={threshold:.2f}"
        super().__init__(msg)
        self.score = score
        self.threshold = threshold


# ── API Errors ─────────────────────────────────────────────────────────────

class APIError(AgentMemError):
    """API request/response error."""
    
    def __init__(self, endpoint: str = "", status_code: int = 0, message: str = ""):
        super().__init__(message or f"API error on {endpoint}: {status_code}")
        self.endpoint = endpoint
        self.status_code = status_code


class ValidationError(APIError):
    """Request validation failed."""
    
    def __init__(self, field: str = "", message: str = ""):
        super().__init__(message=message or f"Validation failed for field: {field}")
        self.field = field
