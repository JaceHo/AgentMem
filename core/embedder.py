"""
Multi-provider embedder with auto-detection.

Supports:
  - openai  : OpenAI-compatible /v1/embeddings (aiserv, Azure, vLLM, LM Studio)
  - local   : sentence-transformers (offline MiniLM)
  - ollama  : local Ollama /api/embeddings endpoint
  - gemini  : Google Gemini embedding API
  - hashing : deterministic offline embedder (tests / fully air-gapped)

Selection priority (LAN-first):
  1. EMBEDDING_PROVIDER env var (explicit override)
  2. Auto-detect from available cloud API keys (gemini / openai)
  3. Local aiserv at 127.0.0.1:4000 (no external key — sk-aiserv-local-dev)
  4. Local Ollama with an embedding model
  5. Fallback to local sentence-transformers

Env vars:
  EMBEDDING_PROVIDER   — force provider: local | ollama | openai | gemini | hashing
  EMBEDDING_DIMS       — override vector dimensions (auto-detected if omitted)
  OLLAMA_BASE_URL      — Ollama API base (default http://localhost:11434)
  OLLAMA_EMBED_MODEL   — Ollama model name (default nomic-embed-text)
  OPENAI_API_KEY       — OpenAI / compatible API key (aiserv local-dev when LAN)
  OPENAI_BASE_URL      — OpenAI-compatible base URL (default aiserv on LAN)
  OPENAI_EMBED_MODEL   — model name (default text-embedding-3-small)
  AISERV_URL           — aiserv root (default http://127.0.0.1:4000)
  GEMINI_API_KEY       — Google Gemini API key
  GEMINI_EMBED_MODEL   — Gemini model (default models/gemini-embedding-001)
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
import time
from abc import ABC, abstractmethod
from typing import Callable, Optional

# Prevent sentence-transformers from phoning home to HuggingFace Hub
# when using Ollama or other non-local providers. Avoids 30s+ timeouts.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import httpx

log = logging.getLogger("mem")

# Max texts per HTTP request for remote embedding providers.
_HTTP_EMBED_CHUNK = 32
# Read timeout for embedding API calls (remote proxies can be slow).
_HTTP_EMBED_TIMEOUT = httpx.Timeout(30.0, connect=5.0)


def _normalize_vectors(raw_embeddings: list) -> list[np.ndarray]:
    results = []
    for emb in raw_embeddings:
        vec = np.array(emb, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        results.append(vec)
    return results


def _chunked_http_encode(
    texts: list[str],
    encode_chunk: Callable[[list[str]], list[np.ndarray]],
) -> list[np.ndarray]:
    """Split large batches into HTTP-sized chunks for remote providers."""
    if not texts:
        return []
    if len(texts) <= _HTTP_EMBED_CHUNK:
        return encode_chunk(texts)
    results: list[np.ndarray] = []
    for i in range(0, len(texts), _HTTP_EMBED_CHUNK):
        results.extend(encode_chunk(texts[i:i + _HTTP_EMBED_CHUNK]))
    return results

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


def embed_signature() -> str:
    """Fingerprint of the provider actually in use: name + dims + model.

    Vectors from two signatures are not comparable. Nothing used to record which
    provider wrote a chunk, so swapping the provider (or its model) left the old
    rows in place and turned `/kb/search` into a flat ordering with no error —
    measured 2026-10-27, every query scored exactly 1/(60+0). Chunks now carry
    this string so the mismatch is countable instead of invisible.
    """
    p = _get_provider()
    model = (getattr(p, "model", None) or getattr(p, "model_name", None)
             or getattr(p, "_MODEL_NAME", None) or "")
    return f"{getattr(p, 'name', type(p).__name__)}:{getattr(p, 'dims', '?')}:{model}"


def _reset_provider(name: str = "local") -> None:
    """Force-switch to a specific provider (used by dimension guard)."""
    global _provider, DIMS
    with _provider_lock:
        # Close old provider's HTTP client if any
        if _provider is not None and hasattr(_provider, '_client') and _provider._client is not None:
            try:
                _provider._client.close()
            except Exception:
                pass
        _provider = _create_provider(name)
        DIMS = _provider.dims


def close() -> None:
    """Close the embedder's HTTP client. Call at app shutdown."""
    global _provider
    with _provider_lock:
        if _provider is not None and hasattr(_provider, '_client') and _provider._client is not None:
            try:
                _provider._client.close()
            except Exception:
                pass
            _provider._client = None


class EmbeddingUnavailable(RuntimeError):
    """Embedding backend is in cooldown after a transport or HTTP failure."""


_embed_down_until = 0.0
_EMBED_COOLDOWN_S = 30.0


def embedding_unavailable() -> bool:
    """True while the embedding circuit is open."""
    return time.monotonic() < _embed_down_until


def _is_embed_outage(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.HTTPError, TimeoutError, OSError)):
        return True
    msg = str(exc).lower()
    return "502" in msg or "bad gateway" in msg or "timed out" in msg


def _trip_embed_circuit(exc: BaseException) -> None:
    global _embed_down_until
    if time.monotonic() >= _embed_down_until:
        log.warning(
            "[embed] backend failed (%s: %s); skipping embeds for %.0fs",
            type(exc).__name__, exc, _EMBED_COOLDOWN_S,
        )
    _embed_down_until = time.monotonic() + _EMBED_COOLDOWN_S


def _clear_embed_circuit() -> None:
    global _embed_down_until
    _embed_down_until = 0.0


def encode(text: str) -> np.ndarray:
    """Encode a single text string to a normalized float32 vector."""
    if embedding_unavailable():
        raise EmbeddingUnavailable("embedding circuit open")
    try:
        vec = _get_provider().encode(text)
    except EmbeddingUnavailable:
        raise
    except Exception as exc:
        if _is_embed_outage(exc):
            _trip_embed_circuit(exc)
        raise
    _clear_embed_circuit()
    return vec


def encode_batch(texts: list[str]) -> list[np.ndarray]:
    """Encode multiple texts. Returns list of normalized float32 vectors."""
    if not texts:
        return []
    if embedding_unavailable():
        raise EmbeddingUnavailable("embedding circuit open")
    try:
        vecs = _get_provider().encode_batch(texts)
    except EmbeddingUnavailable:
        raise
    except Exception as exc:
        if _is_embed_outage(exc):
            _trip_embed_circuit(exc)
        raise
    _clear_embed_circuit()
    return vecs


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
            LocalProvider._MODEL_NAME = os.path.expanduser(model_override)
        # Short name for the embed signature — a full cache path in
        # /kb/status would be unreadable and machine-specific.
        self.model = os.path.basename(self._MODEL_NAME.rstrip("/"))
        # MiniLM is tiny and CPU wins; a 0.6B model is ~10x faster on Apple
        # silicon's GPU (measured 2026-10-27: 1.6 vs 16.6 texts/s on 64 chunks).
        self.device = os.environ.get("EMBEDDING_DEVICE") or None
        if not dims_override:
            # Class default is MiniLM's 384; a different model has a different
            # width, and a stale dims value corrupts the embed signature that
            # /kb/status uses to detect incomparable vectors.
            try:
                from transformers import AutoConfig
                cfg = AutoConfig.from_pretrained(self._MODEL_NAME)
                self.dims = int(getattr(cfg, "hidden_size", self.dims))
            except Exception:
                pass

    def _get_model(self):
        with self._model_lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer
                LocalProvider._model = SentenceTransformer(self._MODEL_NAME, device=self.device)
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
            self._client = httpx.Client(timeout=_HTTP_EMBED_TIMEOUT, trust_env=False)
        return self._client

    def encode(self, text: str) -> np.ndarray:
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        return _chunked_http_encode(texts, self._encode_batch_http)

    def _encode_batch_http(self, texts: list[str]) -> list[np.ndarray]:
        r = self._get_client().post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts},
        )
        r.raise_for_status()
        data = r.json()
        return _normalize_vectors(data.get("embeddings", []))


# ── OpenAI-compatible ───────────────────────────────────────────────

_AISERV_LOCAL_KEY = "sk-aiserv-local-dev"
_AISERV_DEFAULT_ROOT = "http://127.0.0.1:4000"


def _is_lan_openai_base(base_url: str) -> bool:
    """True for loopback / private aiserv-style OpenAI-compat bases."""
    u = (base_url or "").lower()
    return any(
        h in u
        for h in ("127.0.0.1", "localhost", "0.0.0.0", "[::1]", ".local/")
    )


def _aiserv_root() -> str:
    raw = (
        os.environ.get("AISERV_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or _AISERV_DEFAULT_ROOT
    ).rstrip("/")
    if raw.endswith("/v1"):
        raw = raw[:-3]
    return raw


def _aiserv_reachable(timeout: float = 0.4) -> bool:
    """Fast probe — LAN aiserv up → prefer it over downloading MiniLM."""
    try:
        root = _aiserv_root()
        r = httpx.get(
            f"{root}/v1/models",
            timeout=timeout,
            trust_env=False,
            headers={"Authorization": f"Bearer {_AISERV_LOCAL_KEY}"},
        )
        return r.status_code == 200
    except Exception:
        return False


class OpenAIProvider(BaseProvider):
    name = "openai"
    dims = 1536  # text-embedding-3-small

    _MODEL_DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
        "nomic-embed-text": 768,
    }

    def __init__(self):
        # LAN default: aiserv OpenAI-compat endpoint, local-dev key (no cloud key).
        default_base = f"{_AISERV_DEFAULT_ROOT}/v1"
        self.base_url = os.environ.get("OPENAI_BASE_URL", default_base).rstrip("/")
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            if _is_lan_openai_base(self.base_url):
                self.api_key = _AISERV_LOCAL_KEY
            else:
                raise ValueError(
                    "OPENAI_API_KEY is required for non-LAN openai embedding provider"
                )
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
            self._client = httpx.Client(timeout=_HTTP_EMBED_TIMEOUT, trust_env=False)
        return self._client

    def encode(self, text: str) -> np.ndarray:
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        return _chunked_http_encode(texts, self._encode_batch_http)

    def _encode_batch_http(self, texts: list[str]) -> list[np.ndarray]:
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
        return _normalize_vectors(item["embedding"] for item in data["data"])


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
            self._client = httpx.Client(timeout=_HTTP_EMBED_TIMEOUT, trust_env=False)
        return self._client

    def encode(self, text: str) -> np.ndarray:
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        return _chunked_http_encode(texts, self._encode_batch_http)

    def _encode_batch_http(self, texts: list[str]) -> list[np.ndarray]:
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
        return _normalize_vectors(emb["values"] for emb in data.get("embeddings", []))


# ── Offline hashing embedder (no model, no network) ────────────────────────
#
# Deterministic, dependency-free *lexical* embedder for offline / dev / test
# use.  It feature-hashes de-duplicated, stopword-filtered tokens into a
# fixed-dim vector and L2-normalizes.  Lexically similar texts get similar
# vectors (high cosine); dissimilar texts get near-orthogonal vectors (low
# cosine).  This replaces the MiniLM local provider in the test suite so the
# whole suite runs fully offline — no HuggingFace download, no network.  It is
# selected via EMBEDDING_PROVIDER=hashing (see tests/conftest.py); production
# still defaults to local/remote providers with the circuit breaker intact.

_HASHING_STOPWORDS = frozenset(
    """
    a an the and or but if then else when at by for with without about above below
    to of in on into is are was were be been being this that these those it its as
    from we you they he she i my your our their his her not no yes do does did have
    has had will would can could should may might must i'm you're he's she's it's
    we're they're there here what which who whom whose why how all any both each
    few more most other some such only own same so than too very just don now also
    """
    .split()
)


class HashingProvider(BaseProvider):
    name = "hashing"
    dims = 384

    def __init__(self):
        dims_override = os.environ.get("EMBEDDING_DIMS")
        if dims_override:
            self.dims = int(dims_override)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        text = text.lower()
        words = re.findall(r"[a-z0-9]+", text)
        cjk = re.findall(r"[\u4e00-\u9fff\u3400-\u4dbf]", text)
        return words + cjk

    def _vectorize(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dims, dtype=np.float32)
        seen: set[str] = set()
        for tok in self._tokenize(text):
            if tok in _HASHING_STOPWORDS or tok in seen:
                continue
            seen.add(tok)
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dims] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def encode(self, text: str) -> np.ndarray:
        return self._vectorize(text)

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self._vectorize(t) for t in texts]


# ── Auto-detection ──────────────────────────────────────────────────

def _detect_provider() -> BaseProvider:
    """Detect embedding provider from environment, LAN-aiserv first."""
    forced = os.environ.get("EMBEDDING_PROVIDER", "").strip().lower()
    if forced:
        return _create_provider(forced)

    # Explicit cloud keys win (user opted into a remote provider).
    if os.environ.get("GEMINI_API_KEY"):
        return GeminiProvider()
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if openai_key and openai_key != _AISERV_LOCAL_KEY:
        return OpenAIProvider()

    # LAN default: local aiserv gateway (no external key / no HF download).
    if _aiserv_reachable():
        os.environ.setdefault("OPENAI_BASE_URL", f"{_aiserv_root()}/v1")
        os.environ.setdefault("OPENAI_API_KEY", _AISERV_LOCAL_KEY)
        return OpenAIProvider()

    # Explicit local-dev key with no reachable aiserv — still try openai path
    # (caller may have set OPENAI_BASE_URL to something else on the LAN).
    if openai_key == _AISERV_LOCAL_KEY:
        return OpenAIProvider()

    # Check if Ollama is running locally AND has an embedding model
    try:
        base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        r = httpx.get(f"{base}/api/tags", timeout=2, trust_env=False)
        if r.status_code == 200:
            models = r.json().get("models", [])
            embed_keywords = ("embed", "e5", "bge-", "gte-", "jina-embed")
            has_embed = any(
                any(kw in m.get("name", "").lower() for kw in embed_keywords)
                for m in models
            )
            if has_embed:
                return OllamaProvider()
    except Exception:
        pass

    # Last resort: local sentence-transformers (may download on first use)
    return LocalProvider()


def _create_provider(name: str) -> BaseProvider:
    providers = {
        "local": LocalProvider,
        "ollama": OllamaProvider,
        "openai": OpenAIProvider,
        "gemini": GeminiProvider,
        "hashing": HashingProvider,
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
