"""
LinUCB Contextual Bandit for Self-Optimizing Retrieval Pipeline.

Each retrieval stage (embedding, BM25, dampening, Q-reranking, etc.) is wrapped
in an independent LinUCB bandit that learns whether to run/skip that stage for
a given query type. An 8-dimensional feature vector captures query properties.

Ported from Ori-Mnemos tracking.ts + ACQO two-phase curriculum.

Architecture:
  - Each stage has its own LinUCB model (A matrix, b vector)
  - 3 actions per stage: RUN (1.0), SKIP (0.0), ABSTAIN (-1.0 = stop pipeline)
  - Essential stages (embedding similarity, RRF fusion) are never skipped
  - ACQO: run all stages for first N queries (exploration), then optimize
"""

from __future__ import annotations

import math
import re
import sqlite3
import logging
import json
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)

# Feature dimension
D = 8

# Actions
RUN = 1.0
SKIP = 0.0
ABSTAIN = -1.0

# Stages in the pipeline
STAGES = [
    "bm25_fusion",       # FTS5/BM25 keyword scoring + RRF
    "dampening",         # Gravity + hub + resolution boost
    "qvalue_reranking",  # Q-value blend + UCB exploration
    "intent_boost",      # Intent-based type boosting
]

# Essential stages that are never skipped
ESSENTIAL_STAGES = {"embedding"}  # Embedding similarity is always required


class LinUCBArm:
    """Single LinUCB arm for one (stage, action) pair."""

    def __init__(self, d: int = D):
        self.A = np.eye(d, dtype=np.float64)
        self.b = np.zeros(d, dtype=np.float64)
        self.n = 0  # selection count

    def predict(self, x: np.ndarray, alpha: float = 1.0) -> float:
        """Predict reward with UCB exploration bonus."""
        A_inv = np.linalg.inv(self.A)
        theta = A_inv @ self.b
        p = float(theta @ x + alpha * math.sqrt(float(x @ A_inv @ x)))
        return p

    def update(self, x: np.ndarray, reward: float) -> None:
        """Update with observed (feature, reward) pair."""
        self.A += np.outer(x, x)
        self.b += reward * x
        self.n += 1


class StageDecider:
    """LinUCB decider for a single pipeline stage."""

    def __init__(self, stage_name: str, d: int = D):
        self.stage_name = stage_name
        self.run_arm = LinUCBArm(d)
        self.skip_arm = LinUCBArm(d)
        self.is_essential = stage_name in ESSENTIAL_STAGES

    def decide(self, features: np.ndarray, alpha: float = 1.0) -> float:
        """Decide whether to RUN or SKIP this stage.

        Returns RUN (1.0) or SKIP (0.0).
        Essential stages always return RUN.
        """
        if self.is_essential:
            return RUN

        run_score = self.run_arm.predict(features, alpha)
        skip_score = self.skip_arm.predict(features, alpha)

        return RUN if run_score >= skip_score else SKIP

    def update(self, features: np.ndarray, action: float, reward: float) -> None:
        """Update the selected arm with the observed reward."""
        if action == RUN:
            self.run_arm.update(features, reward)
        else:
            self.skip_arm.update(features, reward)


class PipelineOptimizer:
    """Self-optimizing retrieval pipeline using LinUCB bandits.

    ACQO (All-Compute, then Query-Optimize) two-phase curriculum:
    Phase 1 (first N queries): run ALL stages, collect training data
    Phase 2 (after N queries): use LinUCB to decide which stages to run
    """

    def __init__(
        self,
        exploration_budget: int = 50,
        alpha: float = 1.0,
        d: int = D,
    ):
        self.exploration_budget = exploration_budget
        self.alpha = alpha
        self.d = d
        self.query_count = 0
        self.deciders: Dict[str, StageDecider] = {
            stage: StageDecider(stage, d) for stage in STAGES
        }
        self._last_features: Optional[np.ndarray] = None
        self._last_decisions: Dict[str, float] = {}

    @property
    def in_exploration_phase(self) -> bool:
        """True during ACQO Phase 1 (run everything)."""
        return self.query_count < self.exploration_budget

    def decide_stages(self, features: np.ndarray) -> Dict[str, bool]:
        """Decide which stages to run for this query.

        During exploration phase, runs all stages.
        After exploration, uses LinUCB to decide.

        Returns dict of {stage_name: should_run}.
        """
        self.query_count += 1
        self._last_features = features

        if self.in_exploration_phase:
            self._last_decisions = {stage: RUN for stage in STAGES}
        else:
            self._last_decisions = {
                stage: decider.decide(features, self.alpha)
                for stage, decider in self.deciders.items()
            }

        return {stage: (action == RUN) for stage, action in self._last_decisions.items()}

    def update_reward(self, reward: float) -> None:
        """Update all deciders with the observed reward for the last query."""
        if self._last_features is None:
            return

        for stage, action in self._last_decisions.items():
            self.deciders[stage].update(self._last_features, action, reward)

    def get_stats(self) -> Dict[str, Any]:
        """Return optimizer statistics."""
        stats = {
            "query_count": self.query_count,
            "phase": "exploration" if self.in_exploration_phase else "optimization",
            "stages": {},
        }
        for stage, decider in self.deciders.items():
            stats["stages"][stage] = {
                "run_count": decider.run_arm.n,
                "skip_count": decider.skip_arm.n,
            }
        return stats


# ─── Feature Extraction ──────────────────────────────────────


def extract_query_features(
    query: str,
    store_size: int = 0,
    embedding: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Extract 8-dimensional feature vector from a query.

    Features:
    1. query_length (normalized by 100)
    2. unique_terms (normalized by 20)
    3. has_question_mark (0/1)
    4. temporal_markers (0/1) — "when", "yesterday", "last week"
    5. named_entities (count of capitalized words, normalized by 5)
    6. embedding_entropy (Shannon entropy of embedding, normalized)
    7. store_size (log-normalized)
    8. query_depth (from intent — 0=surface, 1=deep)
    """
    words = query.strip().split()

    # 1. Query length
    f_length = min(len(query) / 100.0, 1.0)

    # 2. Unique terms
    unique = len(set(w.lower() for w in words))
    f_unique = min(unique / 20.0, 1.0)

    # 3. Has question mark
    f_question = 1.0 if "?" in query else 0.0

    # 4. Temporal markers
    temporal_patterns = r"\b(?:when|yesterday|today|last\s+(?:week|month|year)|recently|previously|ago)\b"
    f_temporal = 1.0 if re.search(temporal_patterns, query.lower()) else 0.0

    # 5. Named entities (capitalized words not at sentence start)
    caps = [w for w in words[1:] if w[0].isupper()] if len(words) > 1 else []
    f_entities = min(len(caps) / 5.0, 1.0)

    # 6. Embedding entropy
    f_entropy = 0.5  # default
    if embedding is not None:
        emb = np.abs(embedding)
        if emb.sum() > 0:
            p = emb / emb.sum()
            p = p[p > 0]
            entropy = -np.sum(p * np.log2(p))
            f_entropy = min(entropy / 10.0, 1.0)

    # 7. Store size (log-normalized)
    f_store = math.log1p(store_size) / 10.0 if store_size > 0 else 0.0

    # 8. Query depth
    from mnemoria.intent import classify_intent
    intent = classify_intent(query)
    depth_map = {
        "procedural": 0.8, "decision": 0.7, "episodic": 0.6,
        "constraint": 0.5, "value": 0.3, "semantic": 0.4, "general": 0.5,
    }
    f_depth = depth_map.get(intent.intent, 0.5)

    return np.array([
        f_length, f_unique, f_question, f_temporal,
        f_entities, f_entropy, f_store, f_depth,
    ], dtype=np.float64)


# ─── Session Reward Tracking ─────────────────────────────────


class SessionRewardTracker:
    """Heuristic session reward tracking.

    Auto-infers rewards from store-after-recall patterns:
    - If store() follows recall() within N seconds, credit recalled memories
    - If same content recalled again, credit re-recall signal
    """

    def __init__(self, credit_window_seconds: float = 300.0):
        self.credit_window = credit_window_seconds
        self._last_recall_ids: List[str] = []
        self._last_recall_time: float = 0.0
        self._recall_history: Dict[str, int] = {}  # id -> recall count

    def on_recall(self, result_ids: List[str], now: float) -> Dict[str, float]:
        """Called after recall(). Returns reward signals for re-recalled memories."""
        rewards: Dict[str, float] = {}

        # Check for re-recall signals
        for rid in result_ids:
            if rid in self._recall_history:
                rewards[rid] = 0.4  # re-recall signal
            self._recall_history[rid] = self._recall_history.get(rid, 0) + 1

        self._last_recall_ids = result_ids
        self._last_recall_time = now
        return rewards

    def on_store(self, now: float) -> Dict[str, float]:
        """Called after store(). Returns reward signals for previously recalled memories."""
        rewards: Dict[str, float] = {}

        if self._last_recall_ids and (now - self._last_recall_time) <= self.credit_window:
            for rid in self._last_recall_ids:
                rewards[rid] = 0.5  # store-after-recall credit

        return rewards
