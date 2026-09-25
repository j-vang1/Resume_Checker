"""Semantic similarity backends for resume ↔ requirement matching.

Primary: local sentence-transformers embeddings (all-MiniLM-L6-v2).
Optional: OpenAI embeddings if OPENAI_API_KEY is set.
Fallback: improved TF-IDF if embeddings unavailable.
"""

from __future__ import annotations

import os
import threading
from functools import lru_cache
from typing import Literal

import numpy as np

BackendName = Literal["openai", "minilm", "tfidf"]

_LOCK = threading.RLock()
_MODEL = None
_BACKEND: BackendName | None = None
_LOAD_ERROR: str | None = None


def scoring_backend_info() -> dict[str, str]:
    """Return which backend is active (lazy-loads on first call)."""
    name = get_backend()
    labels = {
        "openai": "OpenAI embeddings",
        "minilm": "Local semantic embeddings (MiniLM)",
        "tfidf": "TF-IDF fallback (no embedding model loaded)",
    }
    return {
        "backend": name,
        "label": labels.get(name, name),
        "error": _LOAD_ERROR or "",
    }


def get_backend() -> BackendName:
    global _BACKEND, _LOAD_ERROR
    if _BACKEND is not None:
        return _BACKEND
    with _LOCK:
        if _BACKEND is not None:
            return _BACKEND
        if os.getenv("OPENAI_API_KEY"):
            try:
                _openai_embed(["warmup"])
                _BACKEND = "openai"
                return _BACKEND
            except Exception as exc:  # noqa: BLE001
                _LOAD_ERROR = f"OpenAI unavailable ({exc}); using MiniLM."
        try:
            _get_minilm()
            _BACKEND = "minilm"
            return _BACKEND
        except Exception as exc:  # noqa: BLE001
            _LOAD_ERROR = f"MiniLM unavailable ({exc}); using TF-IDF fallback."
            _BACKEND = "tfidf"
            return _BACKEND


def _get_minilm():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _LOCK:
        if _MODEL is not None:
            return _MODEL
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return _MODEL


def _openai_embed(texts: list[str]) -> np.ndarray:
    from openai import OpenAI

    client = OpenAI()
    # Batch for efficiency
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    vectors = [np.array(item.embedding, dtype=np.float32) for item in resp.data]
    mat = np.vstack(vectors)
    # L2 normalize
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


def _minilm_embed(texts: list[str]) -> np.ndarray:
    model = _get_minilm()
    return np.asarray(
        model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float32,
    )


def _tfidf_embed(texts: list[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    try:
        matrix = vec.fit_transform(texts).astype(np.float32).toarray()
    except ValueError:
        return np.zeros((len(texts), 8), dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    return matrix / norms


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts with the active backend. Returns L2-normalized vectors."""
    cleaned = [t if (t or "").strip() else " " for t in texts]
    backend = get_backend()
    if backend == "openai":
        return _openai_embed(cleaned)
    if backend == "minilm":
        return _minilm_embed(cleaned)
    return _tfidf_embed(cleaned)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim > 1:
        a = a.reshape(-1)
    if b.ndim > 1:
        b = b.reshape(-1)
    return float(np.dot(a, b))


def similarity_matrix(queries: list[str], documents: list[str]) -> np.ndarray:
    """
    Return shape (len(queries), len(documents)) cosine similarities.
    Embeds all texts in one batch for speed (important at ~100 resumes).
    """
    if not queries or not documents:
        return np.zeros((len(queries), len(documents)), dtype=np.float32)
    all_texts = list(queries) + list(documents)
    vectors = embed_texts(all_texts)
    q = vectors[: len(queries)]
    d = vectors[len(queries) :]
    return q @ d.T


@lru_cache(maxsize=1)
def warmup() -> str:
    """Force model load; return backend label."""
    info = scoring_backend_info()
    # Touch encode once so first real request is warm
    if info["backend"] != "tfidf":
        embed_texts(["hardware validation and root cause analysis"])
    return info["label"]
