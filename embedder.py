"""Singleton sentence-transformer embedder — loads model once, reuses."""
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DIMS = 384

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def encode(text: str) -> np.ndarray:
    return get_model().encode(text, normalize_embeddings=True)


def encode_batch(texts: list[str]) -> list[np.ndarray]:
    if not texts:
        return []
    return get_model().encode(texts, normalize_embeddings=True, batch_size=32)
