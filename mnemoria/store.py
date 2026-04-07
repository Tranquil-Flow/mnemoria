"""
MnemoriaStore — the main engine for the Mnemoria cognitive memory system.

Combines:
- Cognitive memory: ACT-R activation, Hebbian links, Q-value RL, embeddings
- Structured memory: typed facts (C/D/V/?/✓/~), scopes, supersession, FTS5, gauge

Usage:
    from mnemoria.store import MnemoriaStore
    from mnemoria.config import MnemoriaConfig

    store = MnemoriaStore(MnemoriaConfig.balanced())
    fact_id = store.store("The API uses JWT tokens", fact_type="D", target="auth")
    results = store.recall("What authentication does the API use?")
"""

from __future__ import annotations

import hashlib
import math
import time
import uuid
import logging
import sqlite3
from collections import defaultdict
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

from mnemoria.types import (
    MemoryFact, ScoredFact, FactType,
    METABOLIC_RATES, FACT_TYPE_FROM_NOTATION,
    parse_notation,
)
from mnemoria.config import MnemoriaConfig
from mnemoria.schema import get_connection
from mnemoria import links as link_ops
from mnemoria.links import cosine_similarity
from mnemoria.retrieval import (
    score_candidates, fts5_search, apply_rrf_fusion,
    apply_qvalue_reranking, apply_dampening, apply_ips_debiasing,
    check_contradictions, _get_access_times,
)

logger = logging.getLogger(__name__)


class MnemoriaStore:
    """
    Unified memory engine combining cognitive + structured memory.

    Single store with:
    - ACT-R activation scoring with metabolic decay by fact type
    - Embedding-based semantic retrieval
    - FTS5 keyword search
    - Hebbian link formation (GloVe + Ebbinghaus + homeostasis)
    - Q-value reranking with RL
    - Typed facts with supersession
    - Scope lifecycle management
    - Gauge pressure management
    """

    def __init__(
        self,
        config: Optional[MnemoriaConfig] = None,
        db_path: Optional[str] = None,
    ):
        self._config = config or MnemoriaConfig.balanced()
        self._conn = get_connection(db_path or self._config.db_path)

        # Lazy-load embedding provider
        self._embedder = None
        self._embedder_initialized = False

        # Virtual clock for benchmarking
        self._simulated_time_offset: float = 0.0
        self._use_virtual_clock: bool = False
        self._virtual_clock: float = time.time()

        # Pipeline optimizer (LinUCB)
        self._pipeline_optimizer = None
        if self._config.enable_linucb:
            from mnemoria.bandit import PipelineOptimizer
            self._pipeline_optimizer = PipelineOptimizer(
                exploration_budget=50, alpha=1.0
            )

        # Session reward tracker
        self._reward_tracker = None
        if self._config.enable_session_rewards:
            from mnemoria.bandit import SessionRewardTracker
            self._reward_tracker = SessionRewardTracker()

        # Q-value store
        self._qvalue_store = None
        if self._config.enable_qvalue_reranking:
            try:
                from mnemoria.qvalue_store import QValueStore
                actual_db = db_path or self._config.db_path
                if ":memory:" in actual_db:
                    qvalue_db = ":memory:"
                else:
                    # Store Q-values alongside the main DB
                    qvalue_db = actual_db.replace(".db", "_qvalues.db")
                self._qvalue_store = QValueStore(qvalue_db)
            except ImportError:
                logger.warning("QValueStore not available — Q-value reranking disabled")

        logger.info(f"MnemoriaStore initialized: db={db_path or self._config.db_path}")

    @property
    def config(self) -> MnemoriaConfig:
        return self._config

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    def _get_embedder(self):
        """Lazy-initialize the embedding provider."""
        if not self._embedder_initialized:
            try:
                from mnemoria.embeddings import EmbeddingProvider
                self._embedder = EmbeddingProvider(model=self._config.embedding_model)
            except ImportError:
                logger.warning("EmbeddingProvider not available — semantic search disabled")
                self._embedder = None
            self._embedder_initialized = True
        return self._embedder

    def _now(self) -> float:
        """Current time, accounting for simulated time offset."""
        if self._use_virtual_clock:
            return self._virtual_clock
        return time.time() + self._simulated_time_offset

    def advance_time(self, seconds: float) -> None:
        """Advance simulated clock (for benchmarking)."""
        if self._use_virtual_clock:
            self._virtual_clock += seconds
        else:
            self._simulated_time_offset += seconds

    def enable_virtual_clock(self) -> None:
        """Switch to virtual clock mode — time only advances via advance_time()."""
        self._use_virtual_clock = True
        self._virtual_clock = time.time()

    # ─── Store ───────────────────────────────────────────────────

    def store(
        self,
        content: str,
        category: Optional[str] = None,
        scope: str = "global",
        importance: Optional[float] = None,
        source: str = "agent",
        pinned: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        fact_type: Optional[str] = None,
        target: Optional[str] = None,
    ) -> str:
        """Store a new memory/fact.

        Accepts plain text OR MEMORY_SPEC notation (e.g. "C[db]: UUIDs mandatory").
        Auto-classifies category and importance if not provided.
        Generates embedding. Checks for contradictions and supersession.
        Creates semantic and keyword links.

        Returns the fact ID.
        """
        now = self._now()

        # Parse MEMORY_SPEC notation if present
        parsed = parse_notation(content)
        if parsed:
            ft, tgt, raw_content = parsed
            fact_type = fact_type or ft.value
            target = target or tgt
            content = raw_content

        # Resolve fact type
        ft_enum = FACT_TYPE_FROM_NOTATION.get(fact_type, FactType.VALUE) if fact_type else FactType.VALUE
        target = target or "general"
        metabolic_rate = METABOLIC_RATES.get(ft_enum, 1.0)

        # Auto-classify if not provided
        if category is None or importance is None:
            try:
                from mnemoria.encoding import encode as encode_content
                auto_cat, auto_imp = encode_content(content)
                if category is None:
                    category = auto_cat
                if importance is None:
                    importance = auto_imp
            except ImportError:
                category = category or "factual"
                importance = importance if importance is not None else 0.5

        # Generate embedding — include target in the text for better keyword matching
        # "api.url" → "api url" so queries like "API URL" match the embedding
        embedder = self._get_embedder()
        embedding = None
        if embedder:
            embed_text = content
            if target and target != "general":
                # Convert dotted target to space-separated words for embedding
                target_words = target.replace(".", " ").replace("_", " ")
                embed_text = f"{target_words} {content}"
            embedding = embedder.encode(embed_text)

        # Create fact ID and source hash
        fact_id = str(uuid.uuid4())
        source_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        # Dedup check — exact content match via source hash
        existing = self._conn.execute(
            "SELECT id FROM um_facts WHERE source_hash = ? AND status = 'active'",
            (source_hash,)
        ).fetchone()
        if existing:
            logger.debug(f"Dedup: fact already exists as {existing['id'][:8]}")
            return existing["id"]

        # Semantic dedup — catch near-duplicates with different wording
        if embedding is not None:
            try:
                from mnemoria.ingestion import find_near_duplicates
                dupes = find_near_duplicates(self._conn, content, embedding, threshold=0.95)
                if dupes:
                    logger.debug(f"Semantic dedup: near-duplicate of {dupes[0][0][:8]} (sim={dupes[0][1]:.3f})")
                    return dupes[0][0]
            except Exception:
                pass

        # Supersession check — only for explicitly typed facts with specific targets
        # (not "general" which is the default for plain text)
        superseded_activation = 0.0
        if (self._config.enable_supersession
                and ft_enum not in (FactType.DONE, FactType.OBSOLETE)
                and target != "general"):
            superseded_id = self._check_supersession(ft_enum.value, target, scope, now)
            if superseded_id:
                # Get old activation for transfer
                old_row = self._conn.execute(
                    "SELECT activation FROM um_facts WHERE id = ?", (superseded_id,)
                ).fetchone()
                if old_row:
                    superseded_activation = old_row["activation"] * self._config.activation_transfer_ratio

        # Resolve scope
        scope_id = None
        if scope and scope.lower() not in ("global", "none", ""):
            scope_id = self._get_or_create_scope(scope, now)

        # Contradiction check
        check_contradictions(
            self._conn, content, embedding, self._config.contradiction_threshold,
            category=category, scope_id=scope_id,
        )

        # Store embedding as BLOB
        embedding_blob = embedding.tobytes() if embedding is not None else None

        # INSERT the fact
        self._conn.execute(
            """INSERT INTO um_facts
            (id, content, embedding, type, target, scope_id, status,
             activation, q_value, access_count, metabolic_rate,
             importance, category, layer, pinned,
             created_at, updated_at, last_accessed, source_hash)
            VALUES (?, ?, ?, ?, ?, ?, 'active',
                    ?, 0.5, 0, ?,
                    ?, ?, 'working', ?,
                    ?, ?, ?, ?)""",
            (fact_id, content, embedding_blob, ft_enum.value, target, scope_id,
             superseded_activation, metabolic_rate,
             importance, category, int(pinned),
             now, now, now, source_hash)
        )

        # Record initial access time
        self._conn.execute(
            "INSERT INTO um_access_times (fact_id, access_time) VALUES (?, ?)",
            (fact_id, now)
        )
        self._conn.commit()

        # Create links
        if embedding is not None:
            link_ops.create_semantic_links(
                self._conn, fact_id, embedding, now,
                threshold=self._config.semantic_link_threshold,
                high_threshold=self._config.high_link_threshold,
                max_links=self._config.links_per_memory,
            )
            if self._config.enable_keyword_links:
                link_ops.create_keyword_links(
                    self._conn, fact_id, content, now,
                    threshold=self._config.keyword_link_threshold,
                    min_shared=self._config.keyword_link_min_shared,
                    max_recent=self._config.keyword_link_max_recent,
                )
            self._conn.commit()

        # Gauge pressure check
        if self._config.enable_pressure:
            self._gauge_check()

        # Session reward tracking: credit recalled memories if store follows recall
        if self._reward_tracker:
            rewards = self._reward_tracker.on_store(now)
            for rid, signal in rewards.items():
                self.reward_memory(rid, signal)

        logger.debug(
            f"Stored fact {fact_id[:8]}: type={ft_enum.value}, target={target}, "
            f"cat={category}, imp={importance:.2f}, scope={scope}"
        )
        return fact_id

    # ─── Recall ──────────────────────────────────────────────────

    def recall(
        self,
        query: str,
        scope: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[ScoredFact]:
        """Recall facts using 4-signal fusion.

        Pipeline:
        1. Get candidates (active facts, optionally filtered by scope)
        2. Score via ACT-R activation + embedding similarity
        3. FTS5/BM25 keyword fusion (if enabled)
        4. Q-value reranking (if enabled)
        5. Dampening (gravity, hub, resolution boost)
        6. Update access stats and strengthen Hebbian links

        Returns top-K ScoredFact objects sorted by score.
        """
        top_k = top_k or self._config.top_k
        now = self._now()

        # Encode query
        embedder = self._get_embedder()
        query_embedding = embedder.encode(query) if embedder else None

        # Get candidates
        scope_id = None
        if scope and scope.lower() not in ("global", "none", ""):
            scope_id = self._resolve_scope(scope)

        candidates = self._get_active_facts(scope_id)
        if not candidates:
            return []

        # Score candidates (ACT-R activation + embedding)
        scored = score_candidates(
            self._conn, candidates, query_embedding, query,
            now, self._config, scope,
        )

        # FTS5/BM25 fusion — use FTS5 as a rescue signal when activation-only
        # ranking has low confidence or there's a strong keyword match.
        use_rrf = self._config.enable_rrf_fusion
        fts5_scores: Dict[str, float] = {}

        if scored:
            fts5_scores = fts5_search(self._conn, query, scope_id)

            # Auto-enable RRF in these cases:
            # 1. Strong FTS5 signal exists (clear keyword match) — let it rescue
            # 2. Typed facts exist and activation top-2 gap is small
            if not use_rrf and fts5_scores:
                # Case 1: FTS5 has a strong match (any score > 0.3 is meaningful)
                max_fts5 = max(fts5_scores.values())
                if max_fts5 > 0.3:
                    use_rrf = True
                # Case 2: Typed facts + tight activation scores
                else:
                    has_typed = self._conn.execute(
                        "SELECT 1 FROM um_facts WHERE type != 'V' AND status='active' LIMIT 1"
                    ).fetchone()
                    if has_typed and len(scored) >= 2:
                        gap = abs(scored[0].score - scored[1].score) if scored[0].score != 0 else 0
                        if gap < 0.5:
                            use_rrf = True

        if use_rrf and scored and fts5_scores:
            apply_rrf_fusion(scored, fts5_scores, self._config)

            # Strong-match override: if FTS5 has exactly one dominant match
            # (score >> others), ensure it's ranked first. This handles cases
            # like typed facts where target encodes semantic identity.
            sorted_fts5 = sorted(fts5_scores.items(), key=lambda x: x[1], reverse=True)
            if sorted_fts5 and sorted_fts5[0][1] > 0.3:
                top_fts5_id = sorted_fts5[0][0]
                second_fts5 = sorted_fts5[1][1] if len(sorted_fts5) > 1 else 0.0
                # Dominant if at least 5x stronger than next match (or only match)
                if second_fts5 == 0 or sorted_fts5[0][1] / max(second_fts5, 0.001) >= 5:
                    # Find this fact in scored and boost it above current top
                    for item in scored:
                        if item.fact.id == top_fts5_id:
                            current_max = max(s.score for s in scored)
                            item.score = current_max + abs(current_max) * 0.5 + 0.1
                            item.components['fts5_strong_match'] = sorted_fts5[0][1]
                            break

        # Q-value reranking
        if self._config.enable_qvalue_reranking and self._qvalue_store is not None:
            apply_qvalue_reranking(scored, self._qvalue_store, self._config)

        # Sort and apply dampening
        scored.sort(key=lambda s: s.score, reverse=True)
        scored = apply_dampening(self._conn, scored, query, self._config)

        # IPS debiasing — counteract popularity bias after dampening
        apply_ips_debiasing(scored, self._conn, self._config)
        scored.sort(key=lambda s: s.score, reverse=True)

        results = scored[:top_k]

        # Update access stats
        self._update_access_stats(results, now)

        # Session reward tracking: track recalled memories for re-recall signals
        if self._reward_tracker:
            result_ids = [r.fact.id for r in results]
            rewards = self._reward_tracker.on_recall(result_ids, now)
            for rid, signal in rewards.items():
                self.reward_memory(rid, signal)

        # Strengthen Hebbian links for co-recalled facts
        co_recalled = [(r.fact.id, r.score) for r in results]
        link_ops.strengthen_hebbian_links(
            self._conn, co_recalled, now,
            learning_rate=self._config.hebbian_learning_rate,
            glove_xmax=self._config.hebbian_glove_xmax,
            strength_rate=self._config.hebbian_strength_rate,
            max_links=self._config.links_per_memory,
            enable_homeostasis=self._config.enable_hebbian_homeostasis,
            homeostasis_target=self._config.hebbian_homeostasis_target,
        )

        return results

    def recall_with_ids(
        self,
        query: str,
        top_k: int = 10,
        scope: Optional[str] = None,
    ) -> List[Tuple[str, str, float]]:
        """Recall returning (id, content, score) tuples. Used by Q-learning benchmarks."""
        results = self.recall(query, scope=scope, top_k=top_k)
        return [(r.fact.id, r.fact.content, r.score) for r in results]

    # ─── Lifecycle ───────────────────────────────────────────────

    def simulate_time(self, days: float) -> None:
        """Advance simulated clock by N days."""
        self.advance_time(days * 86400)

    def simulate_access(self, content_substring: str) -> None:
        """Simulate accessing a memory by content substring."""
        now = self._now()
        row = self._conn.execute(
            "SELECT id FROM um_facts WHERE status = 'active' AND content LIKE ?",
            (f"%{content_substring}%",)
        ).fetchone()
        if row:
            self._conn.execute(
                "UPDATE um_facts SET last_accessed=?, access_count=access_count+1 WHERE id=?",
                (now, row["id"])
            )
            self._conn.execute(
                "INSERT INTO um_access_times (fact_id, access_time) VALUES (?, ?)",
                (row["id"], now)
            )
            self._conn.commit()
        elif self._get_embedder():
            # Fallback to semantic match
            emb = self._get_embedder().encode(content_substring)
            rows = self._conn.execute(
                "SELECT id, embedding FROM um_facts WHERE status='active' AND embedding IS NOT NULL"
            ).fetchall()
            best_id, best_sim = None, -1.0
            for r in rows:
                sim = cosine_similarity(emb, np.frombuffer(r["embedding"], dtype=np.float32))
                if sim > best_sim:
                    best_sim = sim
                    best_id = r["id"]
            if best_id and best_sim > 0.5:
                self._conn.execute(
                    "UPDATE um_facts SET last_accessed=?, access_count=access_count+1 WHERE id=?",
                    (now, best_id)
                )
                self._conn.execute(
                    "INSERT INTO um_access_times (fact_id, access_time) VALUES (?, ?)",
                    (best_id, now)
                )
                self._conn.commit()

    def consolidate(self) -> Dict[str, int]:
        """Run consolidation cycle.

        - Protect bridge nodes (Tarjan) from pruning
        - Promote working -> core (access_count >= 3)
        - Demote core -> archive (low activation)
        - Prune archived below threshold
        - Decay Hebbian links
        - Update NPMI on all links
        - Run gauge pressure check
        """
        now = self._now()
        cfg = self._config
        report = {"promoted": 0, "demoted": 0, "pruned": 0, "links_pruned": 0}

        # Tarjan bridge protection: pin structurally critical nodes
        if cfg.enable_tarjan_protection:
            try:
                from mnemoria.lifecycle import find_articulation_points, protect_bridge_nodes
                bridges = find_articulation_points(self._conn)
                if bridges:
                    protect_bridge_nodes(self._conn, bridges)
                    report["bridges_protected"] = len(bridges)
            except Exception:
                pass

        # Promote working -> core
        result = self._conn.execute(
            "UPDATE um_facts SET layer='core' WHERE layer='working' AND access_count >= 3"
        )
        report["promoted"] = result.rowcount

        # Demote core -> archive (low activation - check access times)
        core_facts = self._conn.execute(
            "SELECT id FROM um_facts WHERE layer='core' AND status='active'"
        ).fetchall()
        for row in core_facts:
            access_times = _get_access_times(self._conn, row["id"])
            if access_times:
                base = math.log(max(sum((now - t) ** (-cfg.d) for t in access_times if now - t > 0), 1e-10))
                if base < -5.0:  # Very low activation
                    self._conn.execute(
                        "UPDATE um_facts SET layer='archive' WHERE id=?", (row["id"],)
                    )
                    report["demoted"] += 1

        # Prune archived below threshold
        result = self._conn.execute(
            "DELETE FROM um_facts WHERE layer='archive' AND status='active' "
            "AND access_count < 2 AND (? - last_accessed) > 86400 * 30",
            (now,)
        )
        report["pruned"] = result.rowcount

        # Decay all links
        report["links_pruned"] = link_ops.decay_all_links(self._conn, 1.0 - cfg.link_decay_rate)

        # Update NPMI normalization on all links
        if cfg.enable_npmi:
            try:
                updated = link_ops.update_all_npmi(self._conn)
                report["npmi_updated"] = updated
            except Exception:
                pass

        # Gauge pressure check
        if cfg.enable_pressure:
            self._gauge_check()

        self._conn.commit()
        return report

    def reward_memory(self, memory_id: str, signal: float) -> None:
        """Apply a reward signal to a memory's Q-value."""
        if self._qvalue_store:
            self._qvalue_store.reward(memory_id, signal)

    def explore(
        self,
        query: str,
        top_k: int = 20,
        scope: Optional[str] = None,
    ) -> List[ScoredFact]:
        """Multi-hop exploration via Personalized PageRank.

        Seeds PPR from recall() results, walks the Hebbian link graph
        to discover associatively connected facts that pure similarity
        would miss.

        Falls back to recall() if the link graph is too sparse.
        """
        # Phase 1: seed from recall
        seeds = self.recall(query, scope=scope, top_k=min(top_k, 10))
        if not seeds or len(seeds) < 2:
            return self.recall(query, scope=scope, top_k=top_k)

        seed_ids = {s.fact.id: s.score for s in seeds}

        # Build adjacency from links
        all_links = link_ops.get_all_links(self._conn)
        adj: Dict[str, Dict[str, float]] = {}
        for link in all_links:
            adj.setdefault(link.source_id, {})[link.target_id] = link.strength
            adj.setdefault(link.target_id, {})[link.source_id] = link.strength

        if len(adj) < 3:
            return self.recall(query, scope=scope, top_k=top_k)

        # Phase 2: Personalized PageRank
        alpha = self._config.ppr_alpha  # teleport probability
        all_nodes = set(adj.keys()) | set(seed_ids.keys())
        scores: Dict[str, float] = {n: 0.0 for n in all_nodes}

        # Initialize with seed scores (normalized)
        total_seed = sum(max(s, 0.01) for s in seed_ids.values())
        personalization = {n: max(seed_ids.get(n, 0.0), 0.01) / total_seed for n in all_nodes}

        # PPR iteration
        for _ in range(20):
            new_scores: Dict[str, float] = {}
            for node in all_nodes:
                teleport = alpha * personalization.get(node, 0.0)
                neighbor_contrib = 0.0
                neighbors = adj.get(node, {})
                if neighbors:
                    total_weight = sum(neighbors.values())
                    for neighbor, weight in neighbors.items():
                        neighbor_contrib += (1 - alpha) * scores.get(neighbor, 0.0) * (weight / total_weight)
                new_scores[node] = teleport + neighbor_contrib
            scores = new_scores

        # Combine PPR scores with original recall scores
        combined = {}
        for sf in seeds:
            ppr_boost = scores.get(sf.fact.id, 0.0) * self._config.ppr_boost
            combined[sf.fact.id] = sf.score + ppr_boost
            sf.components['ppr_discovery'] = ppr_boost

        # Discover facts found by PPR but not in original recall
        recalled_ids = {sf.fact.id for sf in seeds}
        discovered_ids = [
            (nid, sc) for nid, sc in scores.items()
            if nid not in recalled_ids and sc > 0.01
        ]
        discovered_ids.sort(key=lambda x: x[1], reverse=True)

        # Fetch discovered facts from DB
        for nid, ppr_score in discovered_ids[:top_k - len(seeds)]:
            row = self._conn.execute(
                "SELECT * FROM um_facts WHERE id = ? AND status IN ('active', 'cold')",
                (nid,)
            ).fetchone()
            if row:
                from mnemoria.retrieval import _row_to_fact, _get_access_times
                access_times = _get_access_times(self._conn, nid)
                fact = _row_to_fact(dict(row), None, access_times)
                seeds.append(ScoredFact(
                    fact=fact,
                    score=ppr_score * self._config.ppr_boost,
                    components={"ppr_discovery": ppr_score},
                ))

        # Re-sort by combined score
        seeds.sort(key=lambda s: s.score, reverse=True)
        return seeds[:top_k]

    def get_stats(self) -> Dict[str, Any]:
        """Return store statistics."""
        fact_row = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM um_facts WHERE status IN ('active', 'cold')"
        ).fetchone()
        link_row = self._conn.execute("SELECT COUNT(*) as cnt FROM um_links").fetchone()
        scope_row = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM um_scopes WHERE status='active'"
        ).fetchone()
        gauge_row = self._conn.execute("SELECT * FROM um_gauge").fetchone()

        return {
            "fact_count": fact_row["cnt"] if fact_row else 0,
            "link_count": link_row["cnt"] if link_row else 0,
            "scope_count": scope_row["cnt"] if scope_row else 0,
            "gauge_pct": round(
                (gauge_row["used_chars"] / gauge_row["max_chars"]) * 100, 1
            ) if gauge_row and gauge_row["max_chars"] > 0 else 0.0,
            "used_chars": gauge_row["used_chars"] if gauge_row else 0,
            "max_chars": gauge_row["max_chars"] if gauge_row else 0,
        }

    def reset(self) -> None:
        """Clear all stored data. Used between benchmark runs."""
        self._conn.executescript("""
            DELETE FROM um_access_times;
            DELETE FROM um_links;
            DELETE FROM um_qvalues;
            DELETE FROM um_facts;
            DELETE FROM um_scopes;
            INSERT INTO um_facts_fts(um_facts_fts) VALUES('rebuild');
        """)
        self._conn.commit()
        if self._qvalue_store:
            try:
                self._qvalue_store._conn.execute("DELETE FROM memory_qvalues")
                self._qvalue_store._conn.commit()
            except Exception:
                pass

        # Reset TF-IDF embedding provider to prevent vocabulary leakage
        # across benchmark scenarios. TF-IDF builds vocab incrementally,
        # so previous scenarios' vocabulary pollutes subsequent ones.
        # Sentence-transformers don't need resetting (fixed vocab).
        if self._embedder_initialized and self._embedder is not None:
            backend = getattr(self._embedder, '_backend', None)
            if backend is not None and hasattr(backend, '_vocab'):
                # Re-initialize the TfidfEmbedder from scratch
                backend._vocab = {}
                backend._doc_freq.clear()
                backend._num_docs = 0
                backend._dim = 0

    # ─── Internal Helpers ────────────────────────────────────────

    def _get_active_facts(self, scope_id: Optional[str] = None) -> List[dict]:
        """Get all active/cold facts, optionally filtered by scope.

        When a scope is specified, returns facts IN that scope PLUS global
        facts (scope_id IS NULL). Global facts are always accessible —
        scope filtering narrows context but doesn't exclude globals.
        This matches cognitive memory's behavior.
        """
        if scope_id:
            rows = self._conn.execute(
                "SELECT * FROM um_facts WHERE status IN ('active', 'cold') "
                "AND (scope_id = ? OR scope_id IS NULL)",
                (scope_id,)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM um_facts WHERE status IN ('active', 'cold')"
            ).fetchall()
        return [dict(r) for r in rows]

    def _check_supersession(
        self, fact_type: str, target: str, scope: str, now: float,
    ) -> Optional[str]:
        """Check if a new fact supersedes an existing one (same type + target + scope)."""
        scope_id = self._resolve_scope(scope) if scope and scope.lower() not in ("global", "none", "") else None

        if scope_id:
            row = self._conn.execute(
                "SELECT id FROM um_facts WHERE type=? AND target=? AND scope_id=? AND status='active'",
                (fact_type, target, scope_id)
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT id FROM um_facts WHERE type=? AND target=? AND scope_id IS NULL AND status='active'",
                (fact_type, target)
            ).fetchone()

        if row:
            self._conn.execute(
                "UPDATE um_facts SET status='superseded', superseded_by=NULL, updated_at=? WHERE id=?",
                (now, row["id"])
            )
            self._conn.commit()
            logger.debug(f"Superseded fact {row['id'][:8]} (type={fact_type}, target={target})")
            return row["id"]
        return None

    def _get_or_create_scope(self, label: str, now: float) -> str:
        """Get or create a scope by label."""
        row = self._conn.execute(
            "SELECT id FROM um_scopes WHERE label=? AND status='active'", (label,)
        ).fetchone()
        if row:
            self._conn.execute(
                "UPDATE um_scopes SET last_referenced=? WHERE id=?", (now, row["id"])
            )
            return row["id"]

        scope_id = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO um_scopes (id, label, status, last_referenced, created_at) "
            "VALUES (?, ?, 'active', ?, ?)",
            (scope_id, label, now, now)
        )
        self._conn.commit()
        return scope_id

    def _resolve_scope(self, label: str) -> Optional[str]:
        """Resolve scope label to ID without creating."""
        row = self._conn.execute(
            "SELECT id FROM um_scopes WHERE label=? AND status='active'", (label,)
        ).fetchone()
        return row["id"] if row else None

    def _update_access_stats(self, results: List[ScoredFact], now: float) -> None:
        """Update access stats for recalled facts."""
        max_times = self._config.max_access_times
        for scored_fact in results:
            fid = scored_fact.fact.id
            self._conn.execute(
                "UPDATE um_facts SET last_accessed=?, access_count=access_count+1 WHERE id=?",
                (now, fid)
            )
            # Add access time (cap at max_times)
            self._conn.execute(
                "INSERT INTO um_access_times (fact_id, access_time) VALUES (?, ?)",
                (fid, now)
            )
            # Trim old access times if needed
            count_row = self._conn.execute(
                "SELECT COUNT(*) as cnt FROM um_access_times WHERE fact_id=?", (fid,)
            ).fetchone()
            if count_row and count_row["cnt"] > max_times:
                self._conn.execute(
                    "DELETE FROM um_access_times WHERE fact_id=? AND access_time IN "
                    "(SELECT access_time FROM um_access_times WHERE fact_id=? "
                    "ORDER BY access_time LIMIT ?)",
                    (fid, fid, count_row["cnt"] - max_times)
                )
        self._conn.commit()

    def _gauge_check(self) -> List[str]:
        """Run gauge pressure cascade. Returns list of actions taken."""
        actions = []
        gauge = self._conn.execute("SELECT * FROM um_gauge").fetchone()
        if not gauge:
            return actions

        used = gauge["used_chars"]
        max_chars = gauge["max_chars"]
        pct = (used / max_chars * 100) if max_chars > 0 else 0

        # 70%: merge duplicates (same type+target+scope, keep newest)
        if pct >= 70:
            result = self._conn.execute("""
                UPDATE um_facts SET status='superseded'
                WHERE id IN (
                    SELECT f1.id FROM um_facts f1
                    JOIN um_facts f2 ON f1.type=f2.type AND f1.target=f2.target
                        AND coalesce(f1.scope_id,'')=coalesce(f2.scope_id,'')
                    WHERE f1.status='active' AND f2.status='active'
                        AND f1.created_at < f2.created_at
                        AND f1.id != f2.id
                )
            """)
            if result.rowcount > 0:
                actions.append(f"merged {result.rowcount} duplicates")
                self._conn.commit()

        # 85%: archive cold-scope facts
        if pct >= 85:
            result = self._conn.execute("""
                UPDATE um_facts SET status='archived'
                WHERE status='active'
                AND scope_id IN (SELECT id FROM um_scopes WHERE status IN ('cold', 'closed'))
                AND (? - last_accessed) > 86400
            """, (self._now(),))
            if result.rowcount > 0:
                actions.append(f"archived {result.rowcount} cold facts")
                self._conn.commit()

        # 95%: push oldest to cold
        if pct >= 95:
            result = self._conn.execute("""
                UPDATE um_facts SET status='cold'
                WHERE id IN (
                    SELECT id FROM um_facts
                    WHERE status='active' AND pinned=0
                    ORDER BY last_accessed
                    LIMIT 10
                )
            """)
            if result.rowcount > 0:
                actions.append(f"cooled {result.rowcount} oldest facts")
                self._conn.commit()

        return actions
