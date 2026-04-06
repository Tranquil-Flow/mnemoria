"""
Unified Memory Configuration — all tunable parameters with profile presets.

Extends the parameter set of CognitiveMemoryConfig with unified-memory
specific knobs. The unified system is standalone: no imports from
cognitive_memory are required at runtime.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MnemoriaConfig:
    """All tunable parameters for the unified memory system.

    Fields inherited from CognitiveMemoryConfig are reproduced verbatim so
    that this module has no runtime dependency on the cognitive_memory package.
    """

    # ------------------------------------------------------------------
    # ACT-R Parameters  (from CognitiveMemoryConfig)
    # ------------------------------------------------------------------

    d: float = 0.3
    """ACT-R decay parameter in tᵢ^(-d). Lower = slower decay."""

    w_semantic: float = 0.5
    """Weight of cosine similarity in activation score."""

    w_importance: float = 0.4
    """Weight of importance score in activation."""

    hebbian_learning_rate: float = 0.12
    """How fast co-activation strengthens Hebbian links. η in ΔW = η × aᵢ × aⱼ."""

    # ------------------------------------------------------------------
    # Thresholds  (from CognitiveMemoryConfig)
    # ------------------------------------------------------------------

    contradiction_threshold: float = 0.12
    """Combined entity-overlap + embedding score above which to flag contradictions."""

    semantic_link_threshold: float = 0.70
    """Minimum cosine similarity to create a semantic link between memories."""

    high_link_threshold: float = 0.90
    """Cosine above which to link even across different categories."""

    links_per_memory: int = 5
    """Maximum outgoing links per memory."""

    enable_keyword_links: bool = True
    """If True, create weak keyword-overlap links at store() time."""

    keyword_link_threshold: float = 0.15
    """Minimum Jaccard similarity to create a keyword link."""

    keyword_link_min_shared: int = 3
    """Minimum number of shared tokens required to create a keyword link."""

    keyword_link_max_recent: int = 50
    """Maximum number of recent memories to check for keyword overlap."""

    prune_threshold: float = 0.01
    """Activation below which to prune archived memories."""

    scope_multiplier: float = 1.5
    """Boost multiplier for memories matching the current query scope."""

    top_k: int = 8
    """Number of memories returned per recall query."""

    # ------------------------------------------------------------------
    # Consolidation  (from CognitiveMemoryConfig)
    # ------------------------------------------------------------------

    core_promotion_count: int = 3
    """Minimum access count to promote a memory from working → core."""

    core_promotion_importance: float = 0.5
    """Minimum importance to promote from working → core."""

    archive_threshold: float = 0.1
    """Activation below which to demote from core → archive."""

    link_decay_rate: float = 0.95
    """Hebbian link weight multiplier per consolidation cycle."""

    link_prune_threshold: float = 0.01
    """Link weight below which to prune the link."""

    # ------------------------------------------------------------------
    # System  (from CognitiveMemoryConfig)
    # ------------------------------------------------------------------

    embedding_model: str = "auto"
    """Embedding provider: 'auto', 'sentence-transformers', 'ollama', 'openai', 'tfidf'."""

    db_path: str = ""
    """Path to SQLite database. Empty = default (~/.hermes/mnemoria.db)."""

    max_access_times: int = 100
    """Cap on stored access timestamps per memory."""

    contradiction_llm_model: Optional[str] = None
    """Model for LLM-assisted contradiction detection. None = embedding-only."""

    # ------------------------------------------------------------------
    # RRF Fusion  (from CognitiveMemoryConfig)
    # ------------------------------------------------------------------

    enable_rrf_fusion: bool = False
    """Enable BM25 keyword scoring + score-weighted RRF fusion in recall().
    Automatically skipped for episodic/temporal queries to preserve recency ordering.
    Disabled by default: helps compression/importance but hurts temporal/scale."""

    rrf_k: int = 60
    """RRF rank constant."""

    rrf_activation_weight: float = 0.92
    """Weight for activation-based score in RRF fusion."""

    rrf_keyword_weight: float = 0.08
    """Weight for BM25 keyword score in RRF fusion. Lower than cognitive default
    (0.15) because FTS5 OR-queries are broader and can disrupt temporal ordering."""

    # ------------------------------------------------------------------
    # Hebbian Upgrade  (from CognitiveMemoryConfig)
    # ------------------------------------------------------------------

    hebbian_glove_xmax: int = 100
    """GloVe-style saturation count for co-occurrence frequency weighting."""

    hebbian_strength_rate: float = 0.2
    """Ebbinghaus strength accumulation rate."""

    hebbian_homeostasis_target: float = 0.7
    """Turrigiano homeostasis target mean weight per node."""

    enable_hebbian_homeostasis: bool = True
    """Enable Turrigiano homeostatic scaling after each Hebbian update round."""

    # ------------------------------------------------------------------
    # Dampening Pipeline  (from CognitiveMemoryConfig)
    # ------------------------------------------------------------------

    enable_dampening: bool = True
    """Enable post-scoring dampening pipeline (gravity, hub, resolution boost)."""

    gravity_dampening_factor: float = 0.6
    """Score multiplier for 'cosine ghosts'."""

    hub_dampening_max_penalty: float = 0.3
    """Maximum fractional penalty for hub memories."""

    resolution_boost_factor: float = 1.25
    """Score multiplier for actionable-knowledge categories."""

    # ------------------------------------------------------------------
    # Q-Value Reranking  (from CognitiveMemoryConfig)
    # ------------------------------------------------------------------

    enable_qvalue_reranking: bool = True
    """Enable Phase B Q-value reranking in recall()."""

    qvalue_lambda_min: float = 0.05
    """Minimum blend weight for Q-values (at cold start)."""

    qvalue_lambda_max: float = 0.35
    """Maximum blend weight for Q-values (reached at ~200 total updates)."""

    qvalue_exploration_c: float = 0.2
    """UCB exploration constant."""

    # ------------------------------------------------------------------
    # Explore / PPR  (from CognitiveMemoryConfig)
    # ------------------------------------------------------------------

    ppr_alpha: float = 0.45
    """Teleport probability for Personalized PageRank."""

    ppr_boost: float = 0.2
    """How much PPR score boosts an existing memory's activation in explore()."""

    explore_max_rounds: int = 3
    """Maximum recursion depth for explore_recursive()."""

    explore_convergence_threshold: float = 0.1
    """Stop recursion when new_notes / total_notes < this fraction."""

    explore_max_notes: int = 50
    """Hard cap on total memories visited across all rounds in explore_recursive()."""

    # ------------------------------------------------------------------
    # Unified Memory Extensions
    # ------------------------------------------------------------------

    # Phase 1 — always on
    enable_typed_decay: bool = True
    """Use per-FactType metabolic rates when computing activation decay."""

    enable_supersession: bool = True
    """Auto-supersede older facts sharing the same type + target on store()."""

    enable_pressure: bool = True
    """Enable gauge pressure management to keep memory within gauge_max_chars."""

    gauge_max_chars: int = 10000
    """Maximum total character budget for in-context (working) memory."""

    activation_transfer_ratio: float = 0.7
    """Fraction of a superseded fact's activation transferred to its successor."""

    # Phase 2 — intent classification
    enable_intent_classification: bool = True
    """(Phase 2) Classify query intent to adjust retrieval strategy."""

    # Phase 3 — graph analytics
    enable_npmi: bool = True
    """(Phase 3) Normalize co-occurrence edges with NPMI during consolidation."""

    enable_tarjan_protection: bool = True
    """(Phase 3) Protect bridge nodes (articulation points) from pruning."""

    # Phase 4 — reinforcement learning
    enable_linucb: bool = True
    """(Phase 4) LinUCB contextual bandit for per-stage run/skip decisions.
    Enabled by default — PipelineOptimizer runs all stages for the first 50
    queries (ACQO exploration phase) before adapting."""

    enable_session_rewards: bool = True
    """(Phase 4) Auto-infer rewards from store-after-recall patterns."""

    enable_ips: bool = True
    """(Phase 4) IPS (Inverse Propensity Scoring) debiasing in recall().
    Boosts under-retrieved facts and penalises over-retrieved ones to
    counteract popularity bias.  Only activates when 5+ facts have been
    accessed at least once."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        if not self.db_path:
            hermes_home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
            self.db_path = os.path.join(hermes_home, "mnemoria.db")

    # ------------------------------------------------------------------
    # Profile constructors
    # ------------------------------------------------------------------

    @classmethod
    def balanced(cls) -> "MnemoriaConfig":
        """Default balanced profile — all Phase-1 features on, later phases off."""
        return cls()  # All defaults

    @classmethod
    def from_profile(cls, profile: str) -> "MnemoriaConfig":
        """Create a config from a named profile.

        Available profiles
        ------------------
        balanced : sensible defaults, Phase-1 unified features enabled.
        """
        profiles: dict[str, "MnemoriaConfig"] = {
            "balanced": cls.balanced(),
        }
        if profile not in profiles:
            raise ValueError(
                f"Unknown profile '{profile}'. Available: {list(profiles.keys())}"
            )
        return profiles[profile]
