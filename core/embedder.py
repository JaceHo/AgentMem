"""
Multi-provider embedder with auto-detection.

Supports:
  - local   : sentence-transformers (default, no API key needed)
  - ollama  : local Ollama /api/embeddings endpoint
  - openai  : OpenAI-compatible /v1/embeddings (also Azure, vLLM, LM Studio)
  - gemini  : Google Gemini embedding API

Selection priority:
  1. EMBEDDING_PROVIDER env var (explicit override)
  2. Auto-detect from available API keys
  3. Fallback to local sentence-transformers

Env vars:
  EMBEDDING_PROVIDER   — force provider: local | ollama | openai | gemini
  EMBEDDING_DIMS       — override vector dimensions (auto-detected if omitted)
  OLLAMA_BASE_URL      — Ollama API base (default http://localhost:11434)
  OLLAMA_EMBED_MODEL   — Ollama model name (default nomic-embed-text)
  OPENAI_API_KEY       — OpenAI / compatible API key
  OPENAI_BASE_URL      — OpenAI-compatible base URL
  OPENAI_EMBED_MODEL   — model name (default text-embedding-3-small)
  GEMINI_API_KEY       — Google Gemini API key
  GEMINI_EMBED_MODEL   — Gemini model (default models/gemini-embedding-001)
"""

from __future__ import annotations

import os
import threading
from abc import ABC, abstractmethod
from typing import Optional

# Prevent sentence-transformers from phoning home to HuggingFace Hub
# when using Ollama or other non-local providers. Avoids 30s+ timeouts.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import httpx

# ── Public API (unchanged) ──────────────────────────────────────────

DIMS: int = 384  # default for local MiniLM; overridden by provider

_provider: Optional[BaseProvider] = None
_provider_lock = threading.Lock()  # protects _provider and DIMS mutations


def _get_provider() -> BaseProvider:
    global _provider, DIMS
    with _provider_lock:
        if _provider is None:
            _provider = _detect_provider()
            DIMS = _provider.dims
        return _provider


def _reset_provider(name: str = "local") -> None:
    """Force-switch to a specific provider (used by dimension guard)."""
    global _provider, DIMS
    with _provider_lock:
        _provider = _create_provider(name)
        DIMS = _provider.dims


def encode(text: str) -> np.ndarray:
    """Encode a single text string to a normalized float32 vector."""
    return _get_provider().encode(text)


def encode_batch(texts: list[str]) -> list[np.ndarray]:
    """Encode multiple texts. Returns list of normalized float32 vectors."""
    if not texts:
        return []
    return _get_provider().encode_batch(texts)


# Alias for compatibility with tests that use embed()
def embed(text: str) -> np.ndarray:
    return encode(text)


# ── Provider base class ─────────────────────────────────────────────

class BaseProvider(ABC):
    name: str
    dims: int

    @abstractmethod
    def encode(self, text: str) -> np.ndarray: ...

    @abstractmethod
    def encode_batch(self, texts: list[str]) -> list[np.ndarray]: ...


# ── Local: sentence-transformers ────────────────────────────────────

class LocalProvider(BaseProvider):
    name = "local"
    dims = 384

    # Default model: benchmark-proven (LongMemEval-S R@5=95.2%),
    # 384-dim, fast (~3ms/text), no API key needed.
    # Can be overridden via EMBEDDING_MODEL env var or settings.embedding_model.
    _MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    _model = None
    _model_lock = threading.Lock()

    def __init__(self):
        dims_override = os.environ.get("EMBEDDING_DIMS")
        if dims_override:
            self.dims = int(dims_override)
        # Respect EMBEDDING_MODEL env var or settings if set
        model_override = os.environ.get("EMBEDDING_MODEL")
        if model_override:
            LocalProvider._MODEL_NAME = model_override

    def _get_model(self):
        with self._model_lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer
                LocalProvider._model = SentenceTransformer(self._MODEL_NAME)
            return self._model

    def encode(self, text: str) -> np.ndarray:
        return self._get_model().encode(text, normalize_embeddings=True)

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        return self._get_model().encode(texts, normalize_embeddings=True, batch_size=32)


# ── Ollama: local LLM embeddings ───────────────────────────────────

class OllamaProvider(BaseProvider):
    name = "ollama"
    dims = 768  # nomic-embed-text default; varies by model

    def __init__(self):
        self.base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self.model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        dims_override = os.environ.get("EMBEDDING_DIMS")
        if dims_override:
            self.dims = int(dims_override)
        # Shared sync client — reused across encode calls to avoid per-call
        # TCP+TLS handshake overhead. Created lazily on first use.
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(timeout=60)
        return self._client

    def encode(self, text: str) -> np.ndarray:
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        results = []
        r = self._get_client().post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts},
        )
        r.raise_for_status()
        data = r.json()
        embeddings = data.get("embeddings", [])
        for emb in embeddings:
            vec = np.array(emb, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            results.append(vec)
        return results


# ── OpenAI-compatible ───────────────────────────────────────────────

class OpenAIProvider(BaseProvider):
    name = "openai"
    dims = 1536  # text-embedding-3-small

    _MODEL_DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for openai embedding provider")
        self.base_url = os.environ.get(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        ).rstrip("/")
        self.model = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        dims_override = os.environ.get("EMBEDDING_DIMS")
        if dims_override:
            self.dims = int(dims_override)
        else:
            self.dims = self._MODEL_DIMS.get(self.model, 1536)
        # Shared sync client — reused across encode calls
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(timeout=60)
        return self._client

    def encode(self, text: str) -> np.ndarray:
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        url = f"{self.base_url}/embeddings"
        r = self._get_client().post(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.model, "input": texts},
        )
        r.raise_for_status()
        data = r.json()
        results = []
        for item in data["data"]:
            vec = np.array(item["embedding"], dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            results.append(vec)
        return results


# ── Gemini ──────────────────────────────────────────────────────────

class GeminiProvider(BaseProvider):
    name = "gemini"
    dims = 768

    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required for gemini embedding provider")
        self.model = os.environ.get("GEMINI_EMBED_MODEL", "models/gemini-embedding-001")
        dims_override = os.environ.get("EMBEDDING_DIMS")
        if dims_override:
            self.dims = int(dims_override)
        # Shared sync client — reused across encode calls
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(timeout=60)
        return self._client

    def encode(self, text: str) -> np.ndarray:
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/"
            f"{self.model}:batchEmbedContents?key={self.api_key}"
        )
        requests = [
            {
                "model": self.model,
                "content": {"parts": [{"text": t}]},
            }
            for t in texts
        ]
        r = self._get_client().post(
            url,
            headers={"Content-Type": "application/json"},
            json={"requests": requests},
        )
        r.raise_for_status()
        data = r.json()
        results = []
        for emb in data.get("embeddings", []):
            vec = np.array(emb["values"], dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            results.append(vec)
        return results


# ── Auto-detection ──────────────────────────────────────────────────

def _detect_provider() -> BaseProvider:
    """Detect embedding provider from environment, with local fallback."""
    forced = os.environ.get("EMBEDDING_PROVIDER", "").strip().lower()
    if forced:
        return _create_provider(forced)

    # Auto-detect from API keys
    if os.environ.get("GEMINI_API_KEY"):
        return GeminiProvider()
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIProvider()

    # Check if Ollama is running locally AND has an embedding model
    try:
        import httpx
        base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        r = httpx.get(f"{base}/api/tags", timeout=2)
        if r.status_code == 200:
            models = r.json().get("models", [])
            # Check if any model is an embedding model (nomic-embed-text, mxbai-embed, etc.)
            embed_keywords = ("embed", "e5", "bge-", "gte-", "jina-embed")
            has_embed = any(
                any(kw in m.get("name", "").lower() for kw in embed_keywords)
                for m in models
            )
            if has_embed:
                return OllamaProvider()
    except Exception:
        pass

    # Default: local sentence-transformers
    return LocalProvider()


def _create_provider(name: str) -> BaseProvider:
    providers = {
        "local": LocalProvider,
        "ollama": OllamaProvider,
        "openai": OpenAIProvider,
        "gemini": GeminiProvider,
    }
    cls = providers.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER '{name}'. "
            f"Choose from: {', '.join(providers.keys())}"
        )
    return cls()


# ── CLI helper ──────────────────────────────────────────────────────

def get_provider_info() -> dict:
    """Return current provider info for diagnostics."""
    p = _get_provider()
    return {
        "provider": p.name,
        "dims": p.dims,
        "model": getattr(p, "model", getattr(p, "_MODEL_NAME", "unknown")),
    }
