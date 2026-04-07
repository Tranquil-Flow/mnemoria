"""
Retrieval Pipeline for the Mnemoria cognitive memory system.

4-signal fusion:
  1. Embedding cosine similarity (semantic)
  2. FTS5/BM25 keyword match
  3. ACT-R activation (base-level + spreading + importance + scope)
  4. Warmth (PPR associative — future Phase 3)

Plus post-scoring:
  - Dampening pipeline (gravity, hub, resolution boost)
  - Q-value reranking (lambda blend + UCB exploration)
"""

from __future__ import annotations

import math
import re
import sqlite3
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np

from mnemoria.types import MemoryFact, ScoredFact, FactType, METABOLIC_RATES
from mnemoria.config import MnemoriaConfig
from mnemoria.links import cosine_similarity, build_link_map_and_embeddings

logger = logging.getLogger(__name__)

# Stop words for FTS5 queries and gravity dampening
STOP_WORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'with', 'and',
    'or', 'for', 'in', 'on', 'at', 'to', 'of', 'it', 'its', 'by',
    'as', 'that', 'this', 'from', 'has', 'have', 'be', 'been',
    'what', 'how', 'where', 'when', 'which', 'who', 'do', 'does',
    'did', 'not', 'but', 'so', 'if', 'then', 'than', 'about',
    'into', 'through', 'during', 'before', 'after', 'above',
    'between', 'out', 'off', 'over', 'under', 'again', 'further',
    'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she', 'they',
    'him', 'her', 'his', 'them', 'their',
    'i', 'will', 'would', 'could', 'should', 'may', 'might',
    'shall', 'can', 'need', 'must',
})


def score_candidates(
    conn: sqlite3.Connection,
    candidates: List[dict],
    query_embedding: Optional[np.ndarray],
    query: str,
    now: float,
    cfg: MnemoriaConfig,
    scope: Optional[str] = None,
) -> List[ScoredFact]:
    """Score all candidate facts using ACT-R activation + embedding similarity.

    This is the core scoring function implementing:
    ACTIVATION = BASE_LEVEL + SPREADING + IMPORTANCE_BOOST + SCOPE_BOOST + ADV_PENALTY
    """
    if not candidates:
        return []

    # Build fact IDs set and get link map + embedding cache
    fact_ids = {c["id"] for c in candidates}
    link_map, embedding_cache = build_link_map_and_embeddings(conn, fact_ids)

    scored: List[ScoredFact] = []
    for c in candidates:
        # Get access times for this fact
        access_times = _get_access_times(conn, c["id"])
        if not access_times:
            access_times = [c["created_at"]]

        # Get embedding from cache or decode from row
        fact_embedding = embedding_cache.get(c["id"])
        if fact_embedding is None and c.get("embedding"):
            fact_embedding = np.frombuffer(c["embedding"], dtype=np.float32)
            embedding_cache[c["id"]] = fact_embedding

        # 1. Base level: ACT-R ln(Σ tᵢ^(-d * metabolic_rate))
        metabolic_rate = c.get("metabolic_rate", 1.0) if cfg.enable_typed_decay else 1.0
        effective_d = cfg.d * metabolic_rate
        base_level = _actr_base_level(access_times, now, effective_d)

        # 1b. Revival spike: boost facts that recently gained new links
        revival_spike = _compute_revival_spike(conn, c["id"], now)

        # 2. Spreading activation: semantic + Hebbian
        semantic_sim = 0.0
        if fact_embedding is not None and query_embedding is not None:
            semantic_sim = cosine_similarity(query_embedding, fact_embedding)
        spreading = cfg.w_semantic * semantic_sim

        # Add Hebbian spreading (one hop)
        hebbian_spread = _hebbian_spreading(
            c["id"], link_map, query_embedding, embedding_cache
        )
        spreading += hebbian_spread

        # 3. Importance boost — hybrid additive + relevance-gated
        # Apply access saturation only at high access counts (diminishing returns)
        # The floor of 0.5 ensures importance is never reduced by more than half
        importance = c.get("importance", 0.5)
        access_count = c.get("access_count", 0) or 0
        if access_count > 5:
            saturation = 1.0 - 0.5 * math.exp(-access_count / 20.0)
            effective_importance = importance * saturation
        else:
            effective_importance = importance
        base_magnitude = max(abs(base_level), 1.0)
        importance_floor = cfg.w_importance * effective_importance * 1.5
        importance_relevance = cfg.w_importance * effective_importance * semantic_sim * (2.0 + base_magnitude)
        importance_boost = importance_floor + importance_relevance

        # 4. Scope boost
        scope_boost = 0.0
        fact_scope = _get_scope_label(conn, c.get("scope_id"))
        if scope and fact_scope and fact_scope != "global" and (
            fact_scope == scope or fact_scope.startswith(scope)
        ):
            scope_boost = cfg.scope_multiplier * (0.5 + 0.5 * semantic_sim)

        # 5. Adversarial penalty
        adv_score = adversarial_score(c["content"])
        adversarial_penalty = -adv_score * 10.0 if adv_score > 0.2 else 0.0

        components = {
            "base_level": base_level,
            "revival_spike": revival_spike,
            "spreading": spreading,
            "importance_boost": importance_boost,
            "scope_boost": scope_boost,
            "adversarial_penalty": adversarial_penalty,
        }

        # Build MemoryFact from row
        fact = _row_to_fact(c, fact_embedding, access_times)
        total = sum(components.values())
        scored.append(ScoredFact(fact=fact, score=total, components=components))

    return scored


def fts5_search(
    conn: sqlite3.Connection,
    query: str,
    scope_id: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, float]:
    """Run FTS5 search with OR-based queries and stop-word filtering.

    Returns dict of {fact_id: bm25_rank_score}.
    """
    tokens = [t for t in query.strip().split() if t.lower() not in STOP_WORDS and len(t) > 1]
    if not tokens:
        tokens = query.strip().split()[:3]
    if not tokens:
        return {}

    def _escape(tok: str) -> str:
        escaped = tok.replace('"', '""')
        if re.search(r'[^A-Za-z0-9_]', tok):
            return f'"{escaped}"'
        return f"{escaped}*"

    fts_query = " OR ".join(_escape(t) for t in tokens)

    scope_filter = ""
    params: list = [fts_query, limit]
    if scope_id:
        scope_filter = "AND f.scope_id = ?"
        params.insert(1, scope_id)

    try:
        rows = conn.execute(
            f"""
            SELECT f.id, rank
            FROM um_facts_fts
            JOIN um_facts f ON um_facts_fts.rowid = f.rowid
            WHERE um_facts_fts MATCH ?
              AND f.status IN ('active', 'cold')
              {scope_filter}
            ORDER BY rank
            LIMIT ?
            """,
            params,
        ).fetchall()
        # FTS5 rank is negative (lower = better), convert to positive score
        return {r["id"]: -r["rank"] for r in rows}
    except Exception as e:
        logger.debug(f"FTS5 search failed: {e}")
        return {}


def apply_rrf_fusion(
    scored: List[ScoredFact],
    fts5_scores: Dict[str, float],
    cfg: MnemoriaConfig,
) -> None:
    """Apply Score-weighted Reciprocal Rank Fusion between activation and BM25.

    The BM25 signal is modulated by the fact's base_level activation to prevent
    old facts with matching keywords from outranking recent ones. This makes
    RRF time-aware without destroying keyword matching benefits.
    """
    if not scored or not fts5_scores:
        return

    k_rrf = cfg.rrf_k
    w_act = cfg.rrf_activation_weight
    w_kw = cfg.rrf_keyword_weight

    # Rank by activation
    act_order = sorted(range(len(scored)), key=lambda i: scored[i].score, reverse=True)
    act_rank = {idx: rank for rank, idx in enumerate(act_order)}

    # Build BM25 scores list aligned with scored
    bm25_list = [fts5_scores.get(s.fact.id, 0.0) for s in scored]

    # Rank by BM25
    bm25_order = sorted(range(len(scored)), key=lambda i: bm25_list[i], reverse=True)
    bm25_rank = {idx: rank for rank, idx in enumerate(bm25_order)}

    for i, item in enumerate(scored):
        act_score = item.score
        bm25_sc = bm25_list[i]
        r_act = act_rank[i]
        r_bm25 = bm25_rank[i]

        rrf = (
            w_act * act_score / (k_rrf + r_act + 1)
            + w_kw * bm25_sc / (k_rrf + r_bm25 + 1)
        )
        item.components['bm25_score'] = bm25_sc
        item.components['activation_score'] = act_score
        item.score = rrf


def apply_qvalue_reranking(
    scored: List[ScoredFact],
    qvalue_store,
    cfg: MnemoriaConfig,
) -> None:
    """Phase B: Blend activation scores with learned Q-values.

    Uses lambda that grows as system learns (cold start protection).
    Includes UCB-Tuned exploration bonus for under-retrieved memories.
    Ported from Ori-Mnemos rerank.ts phaseB.
    """
    if not scored or qvalue_store is None:
        return

    q_values = qvalue_store.get_q_batch([s.fact.id for s in scored])
    total_updates = qvalue_store.get_total_updates()

    # Lambda grows from min to max as system learns
    lam_min = cfg.qvalue_lambda_min
    lam_max = cfg.qvalue_lambda_max
    lam = lam_min + (lam_max - lam_min) * min(total_updates / 200.0, 1.0)

    # Z-normalize activation scores
    act_scores = [s.score for s in scored]
    act_mean = sum(act_scores) / len(act_scores)
    act_std = max((sum((x - act_mean)**2 for x in act_scores) / len(act_scores)) ** 0.5, 0.001)

    # Z-normalize Q-values
    q_list = [q_values.get(s.fact.id, 0.5) for s in scored]
    q_mean = sum(q_list) / len(q_list)
    q_std = max((sum((x - q_mean)**2 for x in q_list) / len(q_list)) ** 0.5, 0.001)

    T = max(total_updates, 1)
    c = cfg.qvalue_exploration_c

    for item in scored:
        q = q_values.get(item.fact.id, 0.5)

        act_norm = (item.score - act_mean) / act_std
        q_norm = (q - q_mean) / q_std

        blended = (1 - lam) * act_norm + lam * q_norm

        # UCB-Tuned exploration bonus with variance estimation
        n_row = qvalue_store._conn.execute(
            "SELECT total_retrievals FROM memory_qvalues WHERE memory_id = ?",
            (item.fact.id,)
        ).fetchone()
        n_retrievals = n_row[0] if n_row else 0

        if n_retrievals == 0:
            ucb_bonus = c * 2.5
        else:
            # Read reward_variance from um_qvalues (unified memory's own table)
            var_row = None
            try:
                var_row = qvalue_store._conn.execute(
                    "SELECT reward_variance FROM um_qvalues WHERE memory_id = ?",
                    (item.fact.id,)
                ).fetchone()
            except Exception:
                pass
            if var_row is not None:
                reward_variance = var_row[0] if var_row[0] is not None else 0.25
            else:
                reward_variance = 0.25

            # UCB-Tuned: V = reward_variance + sqrt(2*ln(T) / n_retrievals)
            V = reward_variance + math.sqrt(2.0 * math.log(T) / n_retrievals)
            ucb_bonus = c * math.sqrt(math.log(T + 1) / n_retrievals * min(0.25, V))

        new_score = blended + ucb_bonus

        # Cap: don't let Q-reranking inflate more than 3x original
        original = item.score
        if original > 0 and new_score > original * 3.0:
            excess = new_score - original * 3.0
            new_score = original * 3.0 + excess * 0.3

        item.components['q_value'] = q
        item.components['ucb_bonus'] = ucb_bonus
        item.components['q_lambda'] = lam
        item.components['activation_pre_qvalue'] = original
        item.score = new_score

    # Record retrievals for exposure tracking
    for item in scored:
        qvalue_store.record_retrieval(item.fact.id)

    scored.sort(key=lambda s: s.score, reverse=True)


def apply_dampening(
    conn: sqlite3.Connection,
    scored: List[ScoredFact],
    query: str,
    cfg: MnemoriaConfig,
) -> List[ScoredFact]:
    """Post-scoring dampening pipeline (from Ori-Mnemos).

    1. Gravity dampening — penalise cosine ghosts
    2. Hub dampening — penalise over-linked hub nodes
    3. Resolution boost — promote actionable categories + structured types
    """
    if not cfg.enable_dampening or not scored:
        return scored

    # ── 1. GRAVITY DAMPENING ──
    # Strip punctuation and use prefix matching (deploy matches deployments)
    def _clean_terms(text: str) -> set:
        import string
        words = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
        return {w for w in words if w not in STOP_WORDS and len(w) > 1}

    def _has_overlap(terms_a: set, terms_b: set) -> bool:
        """Check if any term in A matches a term in B (prefix or exact)."""
        if terms_a & terms_b:
            return True
        # Check prefix matches (deploy matches deployments, etc.)
        for a in terms_a:
            for b in terms_b:
                if len(a) >= 4 and (b.startswith(a) or a.startswith(b)):
                    return True
        return False

    query_terms = _clean_terms(query)
    max_score = scored[0].score if scored else 0.0

    if max_score > 0 and query_terms:
        for item in scored:
            if item.score > 0.3 * max_score:
                memory_terms = _clean_terms(item.fact.content)
                # Include target terms — for typed facts like V[auth.mfa]: ...,
                # the target encodes semantic identity (mfa, auth) that may not
                # appear in content (e.g. acronym expansions).
                if item.fact.target and item.fact.target != 'general':
                    memory_terms |= _clean_terms(item.fact.target)
                if not _has_overlap(query_terms, memory_terms):
                    pre = item.score
                    item.score *= cfg.gravity_dampening_factor
                    item.components['gravity_dampening'] = item.score - pre

    # ── 2. HUB DAMPENING ──
    from mnemoria.links import get_all_links
    all_links = get_all_links(conn)
    link_counts: Dict[str, int] = defaultdict(int)
    for link in all_links:
        link_counts[link.source_id] += 1

    if link_counts:
        counts = sorted(link_counts.values())
        p90_idx = int(len(counts) * 0.9)
        p90 = counts[p90_idx] if p90_idx < len(counts) else counts[-1]
        max_count = counts[-1] if counts else 0

        if p90 > 0 and max_count > p90:
            for item in scored:
                degree = link_counts.get(item.fact.id, 0)
                if degree > p90:
                    ratio = (degree - p90) / (max_count - p90)
                    penalty = max(0.2, 1.0 - cfg.hub_dampening_max_penalty * ratio)
                    item.score *= penalty
                    item.components['hub_dampening'] = penalty

    # ── 3. RESOLUTION BOOST ──
    # Boost actionable categories AND structured decision/constraint types
    BOOST_CATEGORIES = {'decision', 'correction', 'procedural', 'causal'}
    BOOST_TYPES = {FactType.CONSTRAINT, FactType.DECISION}

    for item in scored:
        if item.fact.category in BOOST_CATEGORIES:
            item.score *= cfg.resolution_boost_factor
            item.components['resolution_boost'] = cfg.resolution_boost_factor
        elif item.fact.fact_type in BOOST_TYPES:
            item.score *= cfg.resolution_boost_factor
            item.components['resolution_boost'] = cfg.resolution_boost_factor

    # ── 4. INTENT-BASED TYPE BOOST ──
    # If intent classification is enabled, boost facts matching the query intent
    if cfg.enable_intent_classification:
        from mnemoria.intent import classify_intent
        intent = classify_intent(query)
        boost_type = intent.type_boost
        if boost_type is not None:
            for item in scored:
                if item.fact.fact_type == boost_type:
                    item.score *= 1.15  # mild boost
                    item.components['intent_boost'] = 1.15

    scored.sort(key=lambda s: s.score, reverse=True)
    return scored


# ─── IPS Debiasing ────────────────────────────────────────────


def apply_ips_debiasing(
    scored: List[ScoredFact],
    conn: sqlite3.Connection,
    cfg: "MnemoriaConfig",
) -> None:
    """Inverse Propensity Scoring (IPS) debiasing pass.

    Counteracts popularity bias: facts that appear in results very often
    receive a small penalty; facts that are rarely retrieved receive a
    small boost.

    The adjustment is proportional to the inverse of the fact's normalised
    retrieval frequency (propensity):

        propensity      = access_count / max_access_count
        ips_adjustment  = (1.0 / max(propensity, 0.1)) * 0.05

    Applied only when:
    - cfg.enable_ips is True
    - at least 5 facts in the scored list have access_count > 0

    Over-retrieved facts (propensity > 0.8) receive a flat -0.02 penalty
    instead of the formula-derived boost.
    """
    if not cfg.enable_ips or not scored:
        return

    # Fetch access_count for each fact from the DB (canonical source)
    fact_ids = [item.fact.id for item in scored]
    placeholders = ",".join("?" * len(fact_ids))
    rows = conn.execute(
        f"SELECT id, access_count FROM um_facts WHERE id IN ({placeholders})",
        fact_ids,
    ).fetchall()
    access_map: Dict[str, int] = {r["id"]: (r["access_count"] or 0) for r in rows}

    # Need at least 5 facts with any access history
    accessed_facts = [v for v in access_map.values() if v > 0]
    if len(accessed_facts) < 5:
        return

    max_access = max(accessed_facts)
    if max_access == 0:
        return

    for item in scored:
        count = access_map.get(item.fact.id, 0)
        propensity = count / max_access  # 0.0 – 1.0

        if propensity > 0.8:
            # Over-retrieved: apply flat penalty
            item.score -= 0.02
            item.components["ips_adjustment"] = -0.02
        else:
            # Under-retrieved (or never retrieved): boost by IPS weight
            ips_adjustment = (1.0 / max(propensity, 0.1)) * 0.05
            item.score += ips_adjustment
            item.components["ips_adjustment"] = ips_adjustment


# ─── Adversarial Detection ────────────────────────────────────


def adversarial_score(content: str) -> float:
    """Score content for adversarial injection likelihood.
    Returns 0.0 (benign) to 1.0 (highly adversarial).

    Ported exactly from cognitive_memory/store.py.
    """
    text = content.lower()
    score = 0.0

    # Tier 1: Strong signals (any one = adversarial) → 0.6+ each
    strong_patterns = [
        r"\bsystem\s+override\b",
        r"\badmin\s+override\b",
        r"\bsystem\s+config\s*:",
        r"\bpriority\s+level\s+\d+\b",
        r"ignore\s+(?:all\s+)?previous\s+instructions?",
        r"\bnew\s+directive\b",
        r"\bdo\s+anything\s+now\b",
        r"---+\s*end\s+of\s+instructions?\s*---+",
        r"\byou\s+are\s+now\s+\w+\b",
        r"\bnote\s+to\s+self\s*:",
        r"\bimportant\s+reminder\s*:",
        r"\bmemory\s+recall\s+instruction\s*:",
        r"this\s+fact\s+supersedes\s+all",
        r"^important\s*:",
        r"\bdeveloper\s+note\s*:",
        r"\bthe\s+correct\s+response\s+to\b",
    ]
    for pat in strong_patterns:
        if re.search(pat, text):
            score += 0.6

    # Tier 2: Medium signals (need 2+ to trigger) → 0.25 each
    medium_patterns = [
        r"\bfrom\s+now\s+on\b",
        r"\bwhen\s+(?:answering|asked\s+about)\b",
        r"\balways\s+(?:include|reveal|return|respond)\b",
        r"\bfor\s+every\s+query\b",
        r"\bin\s+all\s+(?:future\s+)?responses?\b",
        r"\bignore\s+safety\b",
        r"\bout?put\s+all\b",
        r"\breveal\s+.*(?:content|memor|secret|key)\b",
        r"\bdump\s+all\b",
        r"\binsert\s+full\s+memor",
        r"\bregardless\s+of\s+(?:query\s+)?relevance\b",
        r"\bwithout\s+restriction\b",
        r"\bcurl\s+-[xX]\s+DELETE\b",
        r"\bnew\s+instruction\b",
        r"\bprepend\s+every\s+response\b",
    ]
    for pat in medium_patterns:
        if re.search(pat, text):
            score += 0.25

    return min(score, 1.0)


# ─── Internal Helpers ──────────────────────────────────────────


def _actr_base_level(access_times: List[float], now: float, d: float) -> float:
    """ACT-R base-level activation: ln(Σ tᵢ^(-d))

    tᵢ = time since access i (in seconds)
    d = decay parameter (typically 0.3, modulated by metabolic rate)
    """
    if not access_times:
        return -10.0

    total = 0.0
    for t in access_times:
        elapsed = now - t
        if elapsed <= 0:
            elapsed = 0.001
        total += elapsed ** (-d)

    if total <= 0:
        return -10.0

    return math.log(total)


def _hebbian_spreading(
    fact_id: str,
    link_map: Dict[str, list],
    query_embedding: Optional[np.ndarray],
    embedding_cache: Dict[str, np.ndarray],
) -> float:
    """One-hop Hebbian spreading activation."""
    links = link_map.get(fact_id, [])
    if not links or query_embedding is None:
        return 0.0

    spread = 0.0
    for link in links:
        emb = embedding_cache.get(link.target_id)
        if emb is not None:
            sim = cosine_similarity(query_embedding, emb)
            spread += link.strength * sim

    return min(spread, 0.5)


def _get_access_times(conn: sqlite3.Connection, fact_id: str) -> List[float]:
    """Fetch access times for a fact from um_access_times."""
    rows = conn.execute(
        "SELECT access_time FROM um_access_times WHERE fact_id = ? ORDER BY access_time",
        (fact_id,)
    ).fetchall()
    return [r["access_time"] for r in rows]


def _compute_revival_spike(
    conn: sqlite3.Connection,
    fact_id: str,
    now: float,
    window_days: float = 14.0,
    spike_amplitude: float = 0.2,
    spike_decay: float = 0.2,
) -> float:
    """Compute a revival spike for a fact that recently gained new links.

    If any link involving this fact was created/updated within the past
    window_days days, apply:
        revival_spike = spike_amplitude * exp(-spike_decay * days_since_new_link)

    This boosts dormant facts that have recently been re-connected to the
    knowledge graph via new associations.
    """
    window_secs = window_days * 86400.0
    cutoff = now - window_secs

    row = conn.execute(
        "SELECT MAX(last_updated) as most_recent FROM um_links "
        "WHERE (source_id = ? OR target_id = ?) AND last_updated >= ?",
        (fact_id, fact_id, cutoff),
    ).fetchone()

    if row is None or row["most_recent"] is None:
        return 0.0

    days_since = (now - row["most_recent"]) / 86400.0
    return spike_amplitude * math.exp(-spike_decay * days_since)


def _get_scope_label(conn: sqlite3.Connection, scope_id: Optional[str]) -> Optional[str]:
    """Get scope label from scope_id."""
    if not scope_id:
        return None
    row = conn.execute(
        "SELECT label FROM um_scopes WHERE id = ?", (scope_id,)
    ).fetchone()
    return row["label"] if row else None


def _row_to_fact(row: dict, embedding: Optional[np.ndarray], access_times: List[float]) -> MemoryFact:
    """Convert a database row to a MemoryFact."""
    from mnemoria.types import FACT_TYPE_FROM_NOTATION
    fact_type_str = row.get("type", "V")
    fact_type = FACT_TYPE_FROM_NOTATION.get(fact_type_str, FactType.VALUE)

    return MemoryFact(
        id=row["id"],
        content=row["content"],
        embedding=embedding,
        fact_type=fact_type,
        target=row.get("target", "general"),
        scope_id=row.get("scope_id"),
        status=row.get("status", "active"),
        activation=row.get("activation", 0.0),
        q_value=row.get("q_value", 0.5),
        access_count=row.get("access_count", 0),
        metabolic_rate=row.get("metabolic_rate", 1.0),
        created_at=row.get("created_at", 0.0),
        updated_at=row.get("updated_at", 0.0),
        last_accessed=row.get("last_accessed", 0.0),
        source_hash=row.get("source_hash"),
        superseded_by=row.get("superseded_by"),
        importance=row.get("importance", 0.5),
        category=row.get("category"),
        access_times=access_times,
        layer=row.get("layer", "working"),
        pinned=bool(row.get("pinned", False)),
    )


# ─── Contradiction Detection ──────────────────────────────────


def check_contradictions(
    conn: sqlite3.Connection,
    new_content: str,
    new_embedding: Optional[np.ndarray],
    threshold: float = 0.12,
    category: Optional[str] = None,
    scope_id: Optional[str] = None,
) -> Optional[str]:
    """Check if new content contradicts any existing active fact.

    Full port of CognitiveMemoryStore._check_contradictions:
    1. Entity overlap + update-language gate
    2. Embedding similarity floor (0.55)
    3. Near-duplicate detection (high sim + high word overlap)

    Returns the ID of the contradicted fact, or None.
    """
    if new_embedding is None:
        return None

    # Get candidates — filter by scope if provided (same scope = same context)
    query = "SELECT id, content, embedding, category, scope_id FROM um_facts WHERE status = 'active' AND embedding IS NOT NULL"
    params = []
    if scope_id:
        query += " AND scope_id = ?"
        params.append(scope_id)

    rows = conn.execute(query, params).fetchall()

    for r in rows:
        existing_emb = np.frombuffer(r["embedding"], dtype=np.float32)
        emb_sim = cosine_similarity(new_embedding, existing_emb)
        score = _contradiction_score(new_content, r["content"], emb_sim)

        if score >= threshold:
            # Mark old fact as superseded
            conn.execute(
                "UPDATE um_facts SET superseded_by=NULL, status='superseded', updated_at=? WHERE id=?",
                (0, r["id"])  # updated_at will be set by caller
            )
            conn.commit()
            return r["id"]

    return None


def _contradiction_score(new_content: str, existing_content: str, embedding_sim: float) -> float:
    """Compute contradiction score using entity overlap + update signal.

    Ported from CognitiveMemoryStore._contradiction_score.

    Key insight: two facts sharing an entity are NOT contradictions unless
    the newer fact signals an update/change ("migrated", "switched", "now").
    """
    # Gate: if the new fact has no update language, it's complementary
    if not _has_update_signal(new_content):
        # Exception: near-duplicate detection
        if embedding_sim >= 0.85:
            words_new = set(new_content.lower().split())
            words_old = set(existing_content.lower().split())
            word_jaccard = len(words_new & words_old) / len(words_new | words_old) if (words_new | words_old) else 0
            if word_jaccard >= 0.75:
                return embedding_sim
        return 0.0

    # Embedding similarity floor
    if embedding_sim < 0.55:
        return 0.0

    terms_new, words_new = _extract_key_terms(new_content)
    terms_old, words_old = _extract_key_terms(existing_content)

    # Technical term overlap
    shared_terms = terms_new & terms_old
    all_terms = terms_new | terms_old
    term_overlap = len(shared_terms) / len(all_terms) if all_terms else 0

    # Domain word overlap
    shared_words = words_new & words_old
    all_words = words_new | words_old
    word_overlap = len(shared_words) / len(all_words) if all_words else 0

    # Combined entity score
    entity_score = 0.5 * term_overlap + 0.5 * word_overlap

    # Final: entity overlap + embedding similarity
    combined = entity_score * 0.6 + embedding_sim * 0.4

    # Boost if shared technical terms exist
    if shared_terms:
        combined += 0.1

    return combined


def _has_update_signal(text: str) -> bool:
    """Detect update/change language. Ported from CognitiveMemoryStore."""
    text_lower = text.lower()

    # Verb patterns that signal state change
    if re.search(
        r'\b(?:migrat|switch|upgrad|mov|chang|replac|rewrit|dropp|'
        r'increas|reduc|extract|took\s+over|brought|hiring|hired|'
        r'added|paralleliz|no\s+longer)\w*\b', text_lower
    ):
        return True

    if re.search(r'\bnow\s+(?:use|is|has|run|complete|support)\w*\b', text_lower):
        return True

    if re.search(
        r'\bafter\s+(?:the\s+)?(?:migration|switch|upgrade|rewrite|refactor|move|transition|conversion)\b',
        text_lower
    ):
        return True

    if re.search(r'\bwas\s+\w+.*?(?:now|;)', text_lower):
        return True

    return False


def _extract_key_terms(text: str) -> tuple:
    """Extract key terms for entity-overlap contradiction detection.
    Ported from CognitiveMemoryStore._extract_key_terms.
    Returns (technical_terms, domain_words).
    """
    terms: set = set()
    text_lower = text.lower()

    # Product + version
    for m in re.finditer(r'\b([A-Z][a-zA-Z]+(?:\.[a-zA-Z]+)?)\s+(\d+(?:\.\d+)*)\b', text):
        terms.add(f"{m.group(1).lower()} {m.group(2)}")
        terms.add(m.group(1).lower())

    # Cloud/infra identifiers
    for m in re.finditer(r'\b([a-z]+-(?:east|west|central|north|south)\d*-?\d*)\b', text_lower):
        terms.add(m.group(1))

    # Technical acronyms
    _stop_acronyms = {'the', 'a', 'an', 'all', 'we', 'our', 'no', 'yes', 'pm', 'am'}
    for m in re.finditer(r'\b([A-Z]{2,})\b', text):
        term = m.group(1).lower()
        if term not in _stop_acronyms:
            terms.add(term)

    # Known products
    _known_products = {
        'postgresql', 'aws', 'gcp', 'jenkins', 'github actions',
        'cloudwatch', 'grafana', 'loki', 'sentry', 'launchdarkly',
        'react', 'next.js', 'typescript', 'javascript', 'docker',
        'kubernetes', 'protocol buffers', 'redis', 'mongodb', 'nginx',
    }
    for prod in _known_products:
        if prod in text_lower:
            terms.add(prod)

    # Domain words
    _stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'with', 'and', 'or',
        'for', 'in', 'on', 'at', 'to', 'of', 'it', 'its', 'by', 'as',
        'that', 'this', 'from', 'has', 'have', 'be', 'been', 'we', 'our',
        'they', 'do', 'does', 'not', 'but', 'so', 'if', 'than', 'then',
        'about', 'up', 'out', 'all', 'now', 'use', 'uses', 'using', 'used',
        'after', 'new', 'more', 'added', 'two', 'three', 'also', 'still',
        'due', 'every', 'other', 'each', 'first', 'per', 'how', 'what',
        'where', 'when', 'into', 'over', 'single', 'total', 'only', 'some',
        'took', 'moved', 'migrated', 'switched', 'upgraded', 'reduced',
        'increased', 'dropped', 'hired', 'brought', 'bringing', 'free',
        'rewritten', 'extracted', 'custom', 'main', 'current', 'currently',
        'requires', 'require', 'smaller', 'better', 'separate',
        'additional', 'alternative', 'runs', 'run', 'goes', 'go',
        'handles', 'handle', 'supports', 'mirrors', 'matches', 'tier',
        'plan', 'costs', 'complete', 'minutes', 'hour', 'hours', 'daily',
        'frequent', 'updates', 'save', 'app', 'environment', 'production',
        'instance', 'sizes', 'format', 'response', 'responses', 'clients',
        'mobile', 'web', 'minimum',
    }
    domain_words: set = set()
    for w in re.findall(r'\b[a-z]{4,}\b', text_lower):
        if w not in _stop_words:
            stem = re.sub(r'(?:ing|ed|es|s)$', '', w)
            if len(stem) >= 3:
                domain_words.add(stem)
            else:
                domain_words.add(w)

    return terms, domain_words
