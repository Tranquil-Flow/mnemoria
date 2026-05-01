"""Cross-encoder reranker for the recall pipeline.

Activation/embedding-based scoring is good at recall (getting the right facts
into the candidate pool) but weak at precision (ordering them correctly). A
cross-encoder reads the query and each candidate fact together and scores
their joint relevance directly. Used as a final stage: broaden the candidate
pool (top_k=8 -> top_k=50), score with the cross-encoder, return top_k.

Model defaults to ``cross-encoder/ms-marco-MiniLM-L-6-v2`` (~22MB,
~500 pairs/sec on CPU). For higher quality at a latency cost, swap to
``BAAI/bge-reranker-v2-m3`` (~568MB, ~150 pairs/sec on CPU) via the
``rerank_model`` config knob.
"""

from __future__ import annotations

import logging
from typing import List, Sequence

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Thin wrapper around sentence_transformers.CrossEncoder.

    Caches the loaded model class-level so multiple stores share it.
    Falls back to a no-op if sentence-transformers is unavailable.
    """

    _shared_models: dict = {}  # model_name -> CrossEncoder

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None
        try:
            if model_name in self._shared_models:
                self._model = self._shared_models[model_name]
                logger.debug(f"Reusing cached cross-encoder: {model_name}")
            else:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(model_name)
                self._shared_models[model_name] = self._model
                logger.info(f"Loaded cross-encoder: {model_name}")
        except Exception as e:
            logger.warning(f"cross-encoder unavailable ({e}); reranker is a no-op")
            self._model = None

    @property
    def is_available(self) -> bool:
        return self._model is not None

    def score(self, query: str, docs: Sequence[str]) -> List[float]:
        """Score (query, doc) pairs. Returns aligned list of relevance floats."""
        if not self._model or not docs:
            return [0.0] * len(docs)
        pairs = [(query, doc) for doc in docs]
        try:
            scores = self._model.predict(pairs, show_progress_bar=False)
        except Exception as e:
            logger.warning(f"cross-encoder predict failed: {e}")
            return [0.0] * len(docs)
        return [float(s) for s in scores]
