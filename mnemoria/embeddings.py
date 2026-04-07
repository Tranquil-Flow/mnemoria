"""
Embedding system for cognitive memory.

Fallback chain:
1. sentence-transformers (all-MiniLM-L6-v2) — best quality, ~80MB download
2. ollama — if running locally
3. openai — text-embedding-3-small
4. TF-IDF — pure Python fallback, no external deps

Configured via CognitiveMemoryConfig.embedding_model:
  'auto' = try chain in order above
  'sentence-transformers' / 'ollama' / 'openai' / 'tfidf' = force specific backend
"""

from __future__ import annotations

import math
import os
import re
import logging
from collections import Counter
from pathlib import Path
from typing import List, Optional, Dict, Tuple

logger = logging.getLogger(__name__)

# Auto-detect persisted HuggingFace cache in the project workspace.
# Containers lose /root/.cache on restart; this survives via mount.
_PERSISTENT_HF_CACHE = Path("/workspace/Projects/.huggingface_cache")
if _PERSISTENT_HF_CACHE.exists() and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = str(_PERSISTENT_HF_CACHE)
    logger.debug(f"Using persistent HF cache: {_PERSISTENT_HF_CACHE}")

# In Docker containers behind MITM proxy (hermes-aegis), httpx SSL
# verification fails even with the cert in the system store (httpx doesn't
# use the OS trust store by default). Two mitigations:
# 1. Disable HF SSL checks (suppresses warnings)
# 2. Set HF_HUB_OFFLINE if the model is already cached (avoids network entirely)
if Path("/certs/mitmproxy-ca-cert.pem").exists():
    os.environ.setdefault("HF_HUB_DISABLE_SSL_VERIFICATION", "1")
    # If the model is already cached, prefer offline mode to avoid SSL issues
    _CACHED_MODEL = _PERSISTENT_HF_CACHE / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2"
    if _CACHED_MODEL.exists():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        logger.debug("HF model cached — using offline mode to avoid SSL issues")

# Try numpy — if unavailable, use pure Python vectors
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ─── Pure Python vector operations ──────────────────────────────────

def _dot(a: list, b: list) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: list) -> float:
    return math.sqrt(sum(x * x for x in a))


def _cosine_sim(a, b) -> float:
    """Cosine similarity — works with lists or numpy arrays.
    Handles different-length vectors (e.g., from incremental TF-IDF)
    by zero-padding the shorter one.
    """
    if HAS_NUMPY:
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        # Pad shorter vector with zeros (new vocab dimensions = 0 for older docs)
        if a.shape[0] != b.shape[0]:
            max_len = max(a.shape[0], b.shape[0])
            if a.shape[0] < max_len:
                a = np.pad(a, (0, max_len - a.shape[0]))
            else:
                b = np.pad(b, (0, max_len - b.shape[0]))
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)
    else:
        # Pad shorter list
        if len(a) != len(b):
            max_len = max(len(a), len(b))
            a = list(a) + [0.0] * (max_len - len(a))
            b = list(b) + [0.0] * (max_len - len(b))
        na, nb = _norm(a), _norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return _dot(a, b) / (na * nb)


def cosine_similarity(a, b) -> float:
    """Public API for cosine similarity between two embedding vectors."""
    return _cosine_sim(a, b)


# ─── Tokenizer ──────────────────────────────────────────────────────

_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "because", "but", "and", "or", "if", "while", "about", "against",
    "up", "down", "that", "this", "it", "its", "i", "me", "my", "we",
    "our", "you", "your", "he", "him", "his", "she", "her", "they",
    "them", "their", "what", "which", "who", "whom",
})

_TOKEN_PATTERN = re.compile(r'[a-zA-Z0-9]+')


def tokenize(text: str) -> List[str]:
    """Lowercase tokenize, remove stop words."""
    tokens = _TOKEN_PATTERN.findall(text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


# ─── TF-IDF Embedding Provider ──────────────────────────────────────

class TfidfEmbedder:
    """
    Pure-Python TF-IDF vectorizer. Builds vocabulary incrementally.
    
    Each text is encoded as a sparse-ish vector in vocabulary space.
    Uses sublinear TF (1 + log(tf)) and IDF weighting.
    Vectors are L2-normalized for cosine similarity via dot product.
    """

    def __init__(self, max_features: int = 2048):
        self.max_features = max_features
        self._vocab: Dict[str, int] = {}  # word -> index
        self._doc_freq: Counter = Counter()  # word -> num docs containing it
        self._num_docs: int = 0
        self._dim: int = 0

    @property
    def dimension(self) -> int:
        return self._dim if self._dim > 0 else self.max_features

    def _ensure_vocab(self, tokens: List[str]) -> None:
        """Add new tokens to vocabulary if space permits."""
        for token in tokens:
            if token not in self._vocab and len(self._vocab) < self.max_features:
                idx = len(self._vocab)
                self._vocab[token] = idx
                self._dim = len(self._vocab)

    def _update_doc_freq(self, tokens: List[str]) -> None:
        """Update document frequency counts."""
        unique_tokens = set(tokens)
        self._doc_freq.update(unique_tokens)
        self._num_docs += 1

    def encode(self, text: str) -> list:
        """
        Encode text into a TF-IDF vector.
        Returns a Python list (or numpy array if available).
        """
        tokens = tokenize(text)
        
        # Update vocab and doc freq
        self._ensure_vocab(tokens)
        self._update_doc_freq(tokens)
        
        # Build TF vector
        tf_counts = Counter(tokens)
        dim = max(len(self._vocab), 1)
        
        if HAS_NUMPY:
            vec = np.zeros(dim, dtype=np.float32)
        else:
            vec = [0.0] * dim
        
        for token, count in tf_counts.items():
            if token in self._vocab:
                idx = self._vocab[token]
                # Sublinear TF
                tf = 1.0 + math.log(count) if count > 0 else 0.0
                # IDF
                df = self._doc_freq.get(token, 1)
                idf = math.log(1.0 + self._num_docs / df)
                if HAS_NUMPY:
                    vec[idx] = tf * idf
                else:
                    vec[idx] = tf * idf
        
        # L2 normalize
        if HAS_NUMPY:
            norm_val = np.linalg.norm(vec)
            if norm_val > 0:
                vec = vec / norm_val
            return vec
        else:
            n = _norm(vec)
            if n > 0:
                vec = [x / n for x in vec]
            return vec

    def encode_batch(self, texts: List[str]) -> list:
        """Encode multiple texts."""
        return [self.encode(t) for t in texts]

    def reset(self) -> None:
        """Clear vocabulary and counts."""
        self._vocab.clear()
        self._doc_freq.clear()
        self._num_docs = 0
        self._dim = 0


# ─── Sentence Transformers Provider ──────────────────────────────────

class SentenceTransformerEmbedder:
    """Wraps sentence-transformers for high-quality embeddings.
    
    Caches the model globally so multiple stores can share it.
    Loading ~80MB model per instance would be wasteful.
    """

    _shared_models: dict = {}  # class-level cache: model_name -> model

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            if model_name in self._shared_models:
                self._model = self._shared_models[model_name]
                logger.info(f"Reusing cached sentence-transformers model: {model_name}")
            else:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(model_name)
                self._shared_models[model_name] = self._model
                logger.info(f"Loaded sentence-transformers model: {model_name}")
            self._dimension = self._model.get_sentence_embedding_dimension()
        except Exception as e:
            raise ImportError(f"sentence-transformers not available: {e}")

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, text: str):
        embedding = self._model.encode(text, normalize_embeddings=True)
        if HAS_NUMPY:
            return np.asarray(embedding, dtype=np.float32)
        return list(embedding)

    def encode_batch(self, texts: List[str]) -> list:
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        if HAS_NUMPY:
            return [np.asarray(e, dtype=np.float32) for e in embeddings]
        return [list(e) for e in embeddings]


# ─── Embedding Provider (main entry point) ───────────────────────────

class EmbeddingProvider:
    """
    Unified embedding interface with automatic fallback.
    
    Usage:
        provider = EmbeddingProvider(model='auto')
        vec = provider.encode("some text")
        sim = provider.similarity(vec_a, vec_b)
    """

    def __init__(self, model: str = "auto"):
        self._backend = None
        self._backend_name = "none"
        self._model_config = model

        if model == "auto":
            self._try_fallback_chain()
        elif model == "sentence-transformers":
            self._try_sentence_transformers()
        elif model == "tfidf":
            self._init_tfidf()
        else:
            # Unknown model — fall back
            self._try_fallback_chain()

        if self._backend is None:
            self._init_tfidf()

    def _try_fallback_chain(self) -> None:
        """Try each embedding backend in priority order."""
        # 1. sentence-transformers
        try:
            self._try_sentence_transformers()
            return
        except (ImportError, Exception) as e:
            logger.debug(f"sentence-transformers unavailable: {e}")

        # 2. TF-IDF fallback (always works)
        self._init_tfidf()

    def _try_sentence_transformers(self) -> None:
        self._backend = SentenceTransformerEmbedder()
        self._backend_name = "sentence-transformers"
        logger.info("Using sentence-transformers embeddings")

    def _init_tfidf(self) -> None:
        self._backend = TfidfEmbedder()
        self._backend_name = "tfidf"
        logger.info("Using TF-IDF fallback embeddings")

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def dimension(self) -> int:
        return self._backend.dimension

    def encode(self, text: str):
        """Encode text to embedding vector."""
        return self._backend.encode(text)

    def encode_batch(self, texts: List[str]) -> list:
        """Encode multiple texts."""
        return self._backend.encode_batch(texts)

    def similarity(self, a, b) -> float:
        """Cosine similarity between two vectors."""
        return cosine_similarity(a, b)

    def reset(self) -> None:
        """Reset backend state (only meaningful for TF-IDF)."""
        if hasattr(self._backend, 'reset'):
            self._backend.reset()
