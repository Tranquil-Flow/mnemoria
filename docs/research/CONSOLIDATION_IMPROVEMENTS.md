# Consolidation Improvements Research

Research date: 2026-04-23
Mnemoria version: v0.2.1 (0.927 benchmark, consolidation: 1.000)

## Current Consolidation Architecture

Mnemoria's `consolidate()` method (store.py L623-693) performs:

1. **Tarjan bridge protection**: Find articulation points in the link graph, pin them to prevent pruning
2. **Promote**: working -> core (access_count >= 3)
3. **Demote**: core -> archive (base-level activation < -5.0)
4. **Prune**: Delete archived facts with access_count < 2 and not accessed in 30 days
5. **Decay links**: Multiply all link strengths by (1 - link_decay_rate), prune below 0.01
6. **Update NPMI**: Recompute Normalized Pointwise Mutual Information on all links
7. **Gauge check**: Pressure cascade if memory usage exceeds thresholds

This is a single-pass, synchronous operation. It runs on demand -- there is no scheduled/periodic consolidation. The benchmark scores consolidation at 1.000 (perfect), meaning the current mechanics are correct. But there are significant opportunities to make consolidation smarter and more biologically inspired.

## Research: Sleep-Inspired Memory Consolidation

### Neuroscience Background

In biological brains, memory consolidation occurs primarily during sleep (Diekelmann & Born, 2010):

1. **Hippocampal replay**: During slow-wave sleep (SWS), the hippocampus replays recently encoded memories at accelerated speed (10-20x real-time). This transfers memories from hippocampal (working) to neocortical (long-term) storage.

2. **Synaptic homeostasis hypothesis** (Tononi & Cirelli, 2006): During waking, synapses strengthen through learning (Hebbian potentiation). During sleep, global synaptic downscaling normalizes total synaptic weight, preserving relative differences while reducing metabolic cost. Mnemoria already implements this via `_apply_homeostasis()` in links.py.

3. **Memory triage**: Not all memories are consolidated equally. Emotionally significant, survival-relevant, or goal-related memories are preferentially consolidated. Others are allowed to decay.

4. **Schema integration**: During consolidation, new memories are integrated with existing schemas (prior knowledge structures). This can distort individual memories but improves generalization.

### Mapping to Mnemoria

| Sleep Process | Mnemoria Equivalent | Status |
|---|---|---|
| Hippocampal replay | Re-scoring facts against recent queries | Not implemented |
| Synaptic downscaling | Hebbian homeostasis + link decay | Implemented |
| Memory triage | Promotion (working -> core -> archive) | Implemented, basic |
| Schema integration | Hierarchical abstraction (raw -> summary -> principle) | Not implemented |
| Sleep spindles (memory tagging) | Importance scoring at store time | Implemented |
| REM dream consolidation | Cross-domain link discovery | Partially (PPR exploration) |

### Proposed: Offline Replay Consolidation

During consolidation, simulate a "replay" pass where recent queries are re-run against the current fact store. Facts that consistently appear in replay results are strengthened; facts that never appear are candidates for demotion.

```python
def consolidate_with_replay(self, recent_queries: list[str] = None) -> dict:
    """Enhanced consolidation with offline replay.
    
    If recent_queries is None, reconstructs them from recent recall patterns.
    """
    report = self.consolidate()  # Standard consolidation
    
    # Phase 2: Replay
    if recent_queries is None:
        recent_queries = self._reconstruct_recent_queries()
    
    replay_hits: dict[str, int] = {}
    for query in recent_queries[:20]:  # Cap at 20 replays
        results = self.recall(query, top_k=5)
        for r in results:
            replay_hits[r.fact.id] = replay_hits.get(r.fact.id, 0) + 1
    
    # Strengthen frequently replayed facts
    for fact_id, hit_count in replay_hits.items():
        if hit_count >= 3:
            self._conn.execute(
                "UPDATE um_facts SET importance = MIN(importance + 0.05, 1.0) "
                "WHERE id = ? AND status = 'active'",
                (fact_id,)
            )
    
    # Flag never-replayed working-layer facts as demotion candidates
    all_working = self._conn.execute(
        "SELECT id FROM um_facts WHERE layer = 'working' AND status = 'active'"
    ).fetchall()
    never_replayed = [r["id"] for r in all_working if r["id"] not in replay_hits]
    report["never_replayed"] = len(never_replayed)
    
    self._conn.commit()
    return report

def _reconstruct_recent_queries(self, limit: int = 20) -> list[str]:
    """Reconstruct recent query patterns from access times.
    
    Groups facts that were accessed at similar times (within 5 seconds)
    and reconstructs likely queries from their shared content.
    """
    rows = self._conn.execute(
        "SELECT f.content, at.access_time "
        "FROM um_access_times at "
        "JOIN um_facts f ON at.fact_id = f.id "
        "ORDER BY at.access_time DESC "
        "LIMIT 100"
    ).fetchall()
    
    # Cluster by time proximity
    queries = []
    if rows:
        # Use the content of the most recently accessed fact as a proxy query
        # (the actual query text is not stored -- this is an approximation)
        seen_times = set()
        for row in rows:
            t = int(row["access_time"])
            if t not in seen_times:
                queries.append(row["content"][:100])
                seen_times.add(t)
    
    return queries[:limit]
```

### Implementation Consideration

Replay consolidation requires re-running `recall()` during consolidation, which involves embedding computation. This is expensive if the embedding model is a large sentence transformer. Options:

1. **Pre-compute**: Store query embeddings alongside access times. Replay uses cached embeddings.
2. **Scheduled**: Run replay consolidation only during low-activity periods (equivalent to "sleep").
3. **Sampling**: Replay a random sample (20%) of recent queries, not all.

## Research: Hierarchical Memory Abstraction

### The Abstraction Ladder

Raw facts (specific, concrete) -> Summaries (aggregate, contextual) -> Principles (general, abstract)

Example:
- **Raw facts**: "Deployed v2.1 to staging on Monday", "v2.1 staging deploy had 3 test failures", "Fixed test failures, redeployed v2.1 Tuesday"
- **Summary**: "v2.1 deployment to staging took 2 days due to test failures (Monday-Tuesday)"
- **Principle**: "New deployments should have all tests passing before staging push"

### Why Abstraction Matters

1. **Memory efficiency**: 3 raw facts -> 1 summary -> 1 principle. Reduces storage and retrieval noise.
2. **Generalization**: Principles apply to future situations. Raw facts are specific to past events.
3. **Cognitive fidelity**: Human memory naturally abstracts over time. You remember principles, not details of every deployment.
4. **Context window management**: Summaries and principles are compact. Agent's in-context memory budget is better spent on abstractions than raw events.

### How MemGPT (Letta) Implements Hierarchical Memory

MemGPT (Packer et al., 2023) uses a two-tier memory architecture:

1. **Main context** (working memory): Fixed-size, holds the most relevant information for the current conversation
2. **Archival memory** (long-term): Unlimited, stores all past information
3. **Recursive summarization**: When main context overflows, the oldest section is summarized and pushed to archival memory

MemGPT's summarization is LLM-driven:
```
Summarize the following conversation segment into key facts:
{old_context_section}
```

### Mnemoria's Hierarchical Abstraction Design

#### Layer Structure

```
um_facts (layer='working')    -- Raw facts, recent, concrete
    |
    v  [consolidation: summarize]
um_facts (layer='core')       -- Summaries, aggregated, important
    |
    v  [consolidation: abstract]
um_facts (layer='principle')  -- Principles, general rules, stable
```

Add a new layer value: `principle` (currently only working/core/archive).

#### Summary Generation During Consolidation

```python
def consolidate_with_abstraction(self) -> dict:
    """Consolidation pass that generates summaries from fact clusters."""
    report = self.consolidate()
    
    # Find clusters of related working-layer facts
    working_facts = self._get_facts_by_layer('working')
    if len(working_facts) < 5:
        return report
    
    # Cluster by topic (using existing topic segment infrastructure)
    clusters = self._cluster_by_similarity(working_facts, threshold=0.6)
    
    for cluster in clusters:
        if len(cluster) >= 3:
            # Generate summary using LLM (or rule-based for offline)
            summary = self._summarize_cluster(cluster)
            if summary:
                # Store summary as a core-layer fact
                summary_id = self.store(
                    summary,
                    fact_type='V',
                    importance=max(f.importance for f in cluster),
                    metadata={"abstraction_source": [f.id for f in cluster]},
                )
                # Mark source facts as incorporated
                for f in cluster:
                    self._conn.execute(
                        "UPDATE um_facts SET layer = 'archive' WHERE id = ?",
                        (f.id,)
                    )
                report.setdefault("summaries_created", 0)
                report["summaries_created"] += 1
    
    return report

def _summarize_cluster(self, facts: list) -> str:
    """Generate a summary from a cluster of related facts.
    
    Uses LLM if available, otherwise rule-based concatenation.
    """
    if self._config.consolidation_llm_model:
        # LLM-based summarization
        fact_texts = "\n".join(f"- {f.content}" for f in facts)
        prompt = (
            f"Summarize these {len(facts)} related facts into a single "
            f"concise statement that captures the essential information:\n"
            f"{fact_texts}\n\n"
            f"Summary (one sentence):"
        )
        return self._call_llm(prompt)
    else:
        # Rule-based: keep the highest-importance fact as representative
        best = max(facts, key=lambda f: f.importance)
        return best.content
```

#### Principle Extraction

Over longer time scales, summaries that persist and remain accessed can be further abstracted into principles:

```python
def extract_principles(self) -> list[str]:
    """Extract general principles from long-lived core-layer facts."""
    # Find core facts that have been accessed many times
    stable_facts = self._conn.execute(
        "SELECT content, access_count, category FROM um_facts "
        "WHERE layer = 'core' AND status = 'active' AND access_count >= 10 "
        "ORDER BY access_count DESC LIMIT 20"
    ).fetchall()
    
    if not stable_facts or not self._config.consolidation_llm_model:
        return []
    
    fact_texts = "\n".join(f"- {r['content']}" for r in stable_facts)
    prompt = (
        f"These are frequently used facts from an AI agent's memory. "
        f"Extract 1-3 general principles or rules that these facts imply:\n"
        f"{fact_texts}\n\n"
        f"Principles (one per line, start with 'PRINCIPLE:'):"
    )
    
    response = self._call_llm(prompt)
    principles = [
        line.replace("PRINCIPLE:", "").strip()
        for line in response.split("\n")
        if line.strip().startswith("PRINCIPLE:")
    ]
    return principles
```

## Research: Importance-Based Rehearsal

### Cognitive Background

Human memory rehearsal strengthens important memories through re-activation. Two types:

1. **Maintenance rehearsal**: Repeating information to keep it in short-term memory (rote repetition). Low-depth processing, temporary effect.
2. **Elaborative rehearsal**: Connecting new information to existing knowledge, creating meaningful associations. Deep processing, durable effect.

### Active Rehearsal for Mnemoria

Instead of passively waiting for queries to strengthen facts, actively rehearse important facts during consolidation:

```python
def rehearse_important(self, n: int = 10) -> dict:
    """Actively rehearse high-importance facts to prevent decay.
    
    Simulates a recall event for important facts that haven't been
    accessed recently, preventing them from decaying below retrieval
    threshold despite being important.
    """
    now = self._now()
    
    # Find important facts that are at risk of decay
    at_risk = self._conn.execute("""
        SELECT id, content, importance, last_accessed
        FROM um_facts
        WHERE status = 'active'
          AND importance >= 0.7
          AND (? - last_accessed) > 86400 * 7  -- Not accessed in 7 days
        ORDER BY importance DESC
        LIMIT ?
    """, (now, n)).fetchall()
    
    rehearsed = 0
    for row in at_risk:
        # Simulate access (adds access time, increments count)
        self._conn.execute(
            "UPDATE um_facts SET last_accessed = ?, access_count = access_count + 1 "
            "WHERE id = ?",
            (now, row["id"])
        )
        self._conn.execute(
            "INSERT INTO um_access_times (fact_id, access_time) VALUES (?, ?)",
            (row["id"], now)
        )
        rehearsed += 1
    
    self._conn.commit()
    return {"rehearsed": rehearsed}
```

### Elaborative Rehearsal: Link Discovery

During consolidation, actively search for new connections between existing facts:

```python
def elaborative_rehearsal(self) -> dict:
    """Find and create new links between existing facts.
    
    Models elaborative rehearsal: deepening connections between
    related memories that were stored at different times.
    """
    report = {"new_links": 0}
    
    # Get all active facts with embeddings
    facts = self._conn.execute(
        "SELECT id, embedding, content FROM um_facts "
        "WHERE status = 'active' AND embedding IS NOT NULL "
        "AND layer IN ('working', 'core')"
    ).fetchall()
    
    if len(facts) < 5:
        return report
    
    # For each fact, check if it should be linked to facts
    # it wasn't linked to before (cross-temporal connections)
    for i, fact in enumerate(facts):
        existing_links = {
            r["target_id"] for r in self._conn.execute(
                "SELECT target_id FROM um_links WHERE source_id = ?",
                (fact["id"],)
            ).fetchall()
        }
        
        fact_emb = np.frombuffer(fact["embedding"], dtype=np.float32)
        
        for other in facts[i+1:]:
            if other["id"] in existing_links:
                continue
            
            other_emb = np.frombuffer(other["embedding"], dtype=np.float32)
            sim = cosine_similarity(fact_emb, other_emb)
            
            if sim >= self._config.semantic_link_threshold:
                link_ops._upsert_link(
                    self._conn, fact["id"], other["id"],
                    sim * 0.3, self._now(), "elaborative"
                )
                report["new_links"] += 1
    
    if report["new_links"] > 0:
        self._conn.commit()
    
    return report
```

## Research: Spaced Repetition Algorithms

### SM-2 (SuperMemo Algorithm 2)

SM-2 (Wozniak, 1990) is the classic spaced repetition algorithm used in Anki and other flashcard systems:

```
After each review:
  if quality >= 3 (correct):
    if repetitions == 0:
      interval = 1 day
    elif repetitions == 1:
      interval = 6 days
    else:
      interval = interval * ease_factor
    repetitions += 1
  else (incorrect):
    repetitions = 0
    interval = 1 day
  
  ease_factor = max(1.3, ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)))
```

For Mnemoria, SM-2 could schedule which facts to rehearse and when:

```python
@dataclass
class SpacedRepetitionState:
    fact_id: str
    ease_factor: float = 2.5
    interval_days: float = 1.0
    repetitions: int = 0
    next_review: float = 0.0  # timestamp

def sm2_update(state: SpacedRepetitionState, quality: int) -> SpacedRepetitionState:
    """Update SM-2 state after a review event.
    
    quality: 0-5 (0=complete failure, 5=perfect recall)
    """
    if quality >= 3:
        if state.repetitions == 0:
            state.interval_days = 1.0
        elif state.repetitions == 1:
            state.interval_days = 6.0
        else:
            state.interval_days *= state.ease_factor
        state.repetitions += 1
    else:
        state.repetitions = 0
        state.interval_days = 1.0
    
    state.ease_factor = max(
        1.3,
        state.ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    )
    state.next_review = time.time() + state.interval_days * 86400
    return state
```

### FSRS (Free Spaced Repetition Scheduler)

FSRS (Ye, 2023) is a modern replacement for SM-2 used in Anki 23.10+. It uses a neural network to predict memory stability and retrievability:

Key concepts:
- **Stability (S)**: How long (in days) it takes for retrievability to drop from 100% to 90%. Higher = stronger memory.
- **Difficulty (D)**: Item-specific difficulty parameter (0-10). Harder items have lower initial stability.
- **Retrievability (R)**: Probability of successfully recalling the item. R = (1 + t/S)^(-1) where t = days since last review.

FSRS has 19 tunable parameters learned from review history. It outperforms SM-2 by 10-20% on recall prediction accuracy.

For Mnemoria, FSRS's retrievability formula could replace or augment ACT-R base-level activation:

```python
def fsrs_retrievability(stability: float, days_since_access: float) -> float:
    """FSRS retrievability: probability of successful recall.
    
    R = (1 + t/S)^(-1)
    """
    if stability <= 0:
        return 0.0
    return (1.0 + days_since_access / stability) ** (-1)
```

### Comparison: ACT-R vs SM-2 vs FSRS for Mnemoria

| Aspect | ACT-R (current) | SM-2 | FSRS |
|---|---|---|---|
| Decay function | t^(-d) power law | Scheduled intervals | (1 + t/S)^(-1) |
| Personalization | Metabolic rate per type | Ease factor per item | 19 learned parameters |
| Multi-access support | Yes (sum of all access times) | Single interval chain | Yes (stability grows with review) |
| Review scheduling | No (passive) | Yes (next_review date) | Yes (next_review date) |
| Cold start | Works with 1 access | Needs initial parameters | Needs training data |
| Complexity | Low | Low | Medium |
| Scientific basis | Cognitive architecture | Empirical flashcard data | Neural network on 70M reviews |

### Recommendation for Mnemoria

Keep ACT-R as the primary activation model (it works well for retrieval scoring). Add SM-2-style scheduling **only for active rehearsal** during consolidation:

```sql
-- Add spaced repetition fields to um_facts
ALTER TABLE um_facts ADD COLUMN sr_ease REAL DEFAULT 2.5;
ALTER TABLE um_facts ADD COLUMN sr_interval REAL DEFAULT 1.0;  -- days
ALTER TABLE um_facts ADD COLUMN sr_repetitions INTEGER DEFAULT 0;
ALTER TABLE um_facts ADD COLUMN sr_next_review REAL;  -- timestamp
```

During consolidation, facts whose `sr_next_review < now` are candidates for rehearsal. Rehearsal success is measured by whether the fact would rank in top-k for a synthetic query (using the fact's own content as the query).

## Recommended Implementation Plan

### Phase 1: Scheduled Consolidation (v0.3.0)

Currently, consolidation only runs when explicitly called. Add automatic consolidation:

1. **Write-triggered**: After every N stores (configurable, default 50), run a light consolidation
2. **Time-triggered**: If more than T seconds since last consolidation (configurable, default 3600s), run on next recall
3. **Track last consolidation**: Add `last_consolidated_at` to `um_meta`

```python
def _maybe_consolidate(self):
    """Check if consolidation is due and run if needed."""
    if not hasattr(self, '_consolidation_counter'):
        self._consolidation_counter = 0
    
    self._consolidation_counter += 1
    if self._consolidation_counter >= 50:
        self._consolidation_counter = 0
        self.consolidate()
```

### Phase 2: Active Rehearsal (v0.3.x)

1. Add `rehearse_important()` to consolidation pipeline
2. Add spaced repetition fields to um_facts
3. SM-2 scheduling for rehearsal timing
4. Rehearsal runs during consolidation, not during recall (no user-facing latency)

### Phase 3: Hierarchical Abstraction (v0.4.0)

1. Add `layer='principle'` to the layer hierarchy
2. During consolidation, cluster working-layer facts and generate summaries
3. LLM-assisted summarization (optional, with rule-based fallback)
4. Source facts archived after summarization (not deleted)

### Phase 4: Elaborative Rehearsal + Link Discovery (v0.4.x)

1. Cross-temporal link discovery during consolidation
2. Bibliographic coupling for new-to-old fact connections
3. NPMI-based link quality assessment

### Phase 5: Replay Consolidation (v0.5.0)

1. Store recent queries (or reconstruct from access patterns)
2. During consolidation, replay queries against current store
3. Strengthen consistently-retrieved facts, flag never-retrieved ones
4. Optional: store query embeddings for efficient replay

## Integration with Existing Architecture

| Current Feature | Consolidation Enhancement |
|---|---|
| Tarjan bridge protection | Already runs during consolidation. No change needed. |
| Promotion (working -> core) | Replace access_count threshold with SM-2 ease + replay consistency |
| Link decay | Add elaborative rehearsal after decay (creates new links to offset decay losses) |
| NPMI update | Run after link discovery to normalize new links |
| Gauge pressure | Summary generation reduces fact count, relieving gauge pressure |
| Q-value learning | Rehearsal events generate Q-value updates (rehearsed = "retrieved successfully") |
| LinUCB bandit | Can learn whether rehearsal/abstraction stages improve query outcomes |

## Expected Impact

| Metric | Current | Expected | Enhancement Source |
|---|---|---|---|
| consolidation | 1.000 | 1.000 | Already perfect |
| compression | 1.000 | 1.000 | Already perfect |
| compression_survival | 1.000 | 1.000 | Already perfect |
| importance_filtering | 0.925 | 0.95 | Active rehearsal preserves important facts |
| temporal_decay | 0.933 | 0.96 | SM-2 scheduling + rehearsal |
| capacity_stress | 1.000 | 1.000 | Hierarchical abstraction reduces fact count |
| **Overall** | **0.927** | **0.935-0.94** | Conservative estimate |

The primary value of consolidation improvements is not benchmark score (already strong) but **long-term agent effectiveness**: agents that consolidate intelligently retain useful knowledge indefinitely while shedding noise, rather than uniformly decaying everything.
