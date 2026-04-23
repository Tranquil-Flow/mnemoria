# Memory Safety & User Consent Research

Research date: 2026-04-23
Mnemoria version: v0.2.1 (0.927 benchmark)
Context: SaTML 2025 gap -- Mnemoria has adversarial detection (0.867 benchmark) but no formal safety framework or user consent model

## The Problem

AI agent memory systems that persist information across sessions create novel safety risks:

1. **Privacy leakage**: Memories from one conversation context bleed into another
2. **Stale beliefs**: Outdated facts drive incorrect agent behavior
3. **Manipulation**: Adversarial inputs plant false memories that persist
4. **User loss of control**: Users cannot see, edit, or delete what the agent "remembers"
5. **Regulatory non-compliance**: GDPR, CCPA, and other regulations grant users rights over their personal data -- including data stored in AI memory systems

Mnemoria currently addresses (3) via adversarial scoring (retrieval.py L655-709) and (2) via metabolic decay. But (1), (4), and (5) are largely unaddressed.

## DeChant's 4 Principles for Safe Episodic Memory

### Source: SaTML 2025

Michael DeChant's paper "Safety Considerations for Episodic Memory in AI Agents" (presented at IEEE SaTML 2025) proposes four principles that any episodic memory system should satisfy. This is the most comprehensive safety framework for agent memory to date.

### Principle 1: Transparency

**"The agent should clearly communicate what it remembers and how memories influence its behavior."**

Requirements:
- Memory contents are inspectable by the user at any time
- The agent can explain why it retrieved specific memories for a given response
- Memory influence on output is traceable (which memories contributed to which response)

**Mnemoria status**:
- Partial: `ScoredFact.components` dict tracks score breakdown (base_level, spreading, importance_boost, etc.)
- Partial: `mcp_umemory_pending` tool exposes pending facts
- Missing: No user-facing "what do you remember about me?" query
- Missing: No explanation of why specific memories were used in a response

**Implementation gap**: Mnemoria needs a `explain_recall()` method that returns human-readable explanations:

```python
def explain_recall(self, query: str, top_k: int = 5) -> list[dict]:
    """Recall with human-readable explanations for each result."""
    results = self.recall(query, top_k=top_k)
    explanations = []
    for sf in results:
        explanations.append({
            "content": sf.fact.content,
            "score": sf.score,
            "why": self._explain_components(sf.components),
            "stored_when": sf.fact.created_at,
            "accessed_times": sf.fact.access_count,
            "type": sf.fact.fact_type.value,
        })
    return explanations

def _explain_components(self, components: dict) -> str:
    """Convert score components to human-readable explanation."""
    parts = []
    if components.get("spreading", 0) > 0.1:
        parts.append("semantically similar to your query")
    if components.get("base_level", 0) > 0:
        parts.append("frequently accessed")
    if components.get("scope_boost", 0) > 0:
        parts.append("from the current conversation context")
    if components.get("target_match_boost", 0) > 0:
        parts.append("matches the topic you asked about")
    if components.get("resolution_boost", 0) > 1:
        parts.append("contains actionable information")
    return "; ".join(parts) if parts else "general relevance"
```

### Principle 2: User Control

**"Users should have meaningful control over what the agent remembers, including the ability to add, modify, and delete memories."**

Requirements:
- Users can view all stored memories
- Users can delete specific memories
- Users can correct inaccurate memories
- Users can set memory policies (e.g., "don't remember financial details")
- Deletions are permanent and verifiable

**Mnemoria status**:
- Partial: `mcp_umemory_retract` tool allows retracting pending facts or superseding confirmed facts
- Missing: No "list all my memories" command
- Missing: No "delete this memory permanently" command
- Missing: No memory policies or content filters
- Missing: No "correct this memory" workflow

**Required new MCP tools**:

```python
# List all memories (paginated)
def mcp_umemory_list(scope: str = None, type: str = None, limit: int = 50, offset: int = 0):
    """List stored memories with optional filters."""
    
# Delete a specific memory permanently
def mcp_umemory_delete(fact_id: str, reason: str = None):
    """Permanently delete a memory. Cannot be undone."""
    
# Edit a memory's content
def mcp_umemory_edit(fact_id: str, new_content: str):
    """Replace a memory's content. Old content is logged for audit."""
    
# Set memory policies
def mcp_umemory_policy(action: str, pattern: str):
    """Set memory policies. Actions: block, allow, forget.
    Examples:
      block "financial.*" - never store financial information
      forget "password.*" - delete all password-related memories
    """
```

### Principle 3: Bounded Influence

**"Memories should have bounded influence on agent behavior -- no single memory should dominate decision-making."**

Requirements:
- Single memory influence is capped
- Conflicting memories are surfaced, not silently resolved
- Memory staleness is tracked and communicated
- No runaway reinforcement (popularity bias)

**Mnemoria status**:
- Good: IPS debiasing (retrieval.py L592-649) counteracts popularity bias
- Good: Hub dampening limits over-linked nodes' influence
- Good: Homeostatic scaling prevents link weight accumulation
- Good: Q-value cap (3x original score, retrieval.py L423-424)
- Partial: Contradictions are detected but the old fact is silently superseded (not surfaced to user)
- Missing: No explicit influence cap per memory
- Missing: No staleness communication to user

**Enhancement**: Surface contradiction events to the user:

```python
def store_with_contradiction_notification(self, content, ...):
    """Store with contradiction detection that notifies the user."""
    contradicted_id = check_contradictions(...)
    if contradicted_id:
        old_fact = self._get_fact(contradicted_id)
        return {
            "fact_id": new_id,
            "contradiction_detected": True,
            "superseded_fact": old_fact.content,
            "message": f"Updated: '{old_fact.content}' has been replaced by '{content}'"
        }
```

### Principle 4: Graceful Degradation

**"When memory systems fail or produce uncertain results, agents should fail safely -- defaulting to conservative behavior rather than acting on uncertain memories."**

Requirements:
- Low-confidence retrievals are flagged, not silently used
- Memory system errors do not crash the agent
- Stale or uncertain memories are deprioritized
- The agent can operate without memory (graceful fallback)

**Mnemoria status**:
- Good: The entire memory pipeline is wrapped in try/except (agent continues without memory if anything fails)
- Good: Adversarial scoring penalizes suspicious content
- Partial: Activation scores encode confidence but this isn't surfaced
- Missing: No explicit confidence threshold below which memories are suppressed
- Missing: No "uncertainty flag" on retrieved memories

**Enhancement**: Add confidence thresholds to recall:

```python
def recall(self, query, min_confidence: float = 0.0, ...):
    """Recall with optional confidence floor."""
    results = self._recall_internal(query, ...)
    
    if min_confidence > 0:
        # Suppress results below confidence threshold
        results = [r for r in results if self._confidence(r) >= min_confidence]
    
    # Flag uncertain results
    for r in results:
        r.components["confidence"] = self._confidence(r)
        r.components["uncertain"] = self._confidence(r) < 0.3
    
    return results

def _confidence(self, scored_fact: ScoredFact) -> float:
    """Compute retrieval confidence from score components."""
    # Confidence is high when multiple signals agree
    semantic = scored_fact.components.get("spreading", 0)
    keyword = scored_fact.components.get("bm25_score", 0)
    base = scored_fact.components.get("base_level", 0)
    
    signals_positive = sum(1 for s in [semantic, keyword, base] if s > 0.1)
    return min(1.0, signals_positive / 3.0 + semantic * 0.3)
```

## GDPR Right to Be Forgotten

### Regulatory Context

GDPR Article 17 (Right to Erasure) grants data subjects the right to have their personal data deleted when:
1. The data is no longer necessary for the purpose it was collected
2. The subject withdraws consent
3. The subject objects to processing
4. The data was unlawfully processed
5. Legal obligation requires erasure

For AI agent memory, this means:
- User can request deletion of any personal data stored in Mnemoria
- Deletion must be **complete** -- not just marked as inactive, but removed from all storage
- Deletion must include derived data (embeddings, links, access history)
- Deletion should be **verifiable** -- the user can confirm the data is gone

### Current Mnemoria Gap

Mnemoria has several deletion-adjacent features but none that satisfy GDPR:

| Feature | What It Does | GDPR Compliant? |
|---|---|---|
| Supersession | Marks old fact as `status='superseded'` | No -- data persists |
| Archive | Moves to `layer='archive'` | No -- data persists |
| Pruning | Deletes archived facts below threshold | Partially -- not user-triggered |
| Gauge pressure | Pushes old facts to cold/archive | No -- not erasure |
| Reset | Deletes all data | Too broad -- erases everything |

### Required: Targeted Erasure API

```python
def forget(self, fact_id: str) -> dict:
    """Permanently erase a specific fact and all its traces.
    
    GDPR Article 17 compliant: removes the fact, its embedding,
    all links, all access history, and any pending/promoted references.
    
    Returns a deletion receipt for audit.
    """
    now = self._now()
    receipt = {"fact_id": fact_id, "deleted_at": now, "traces_removed": []}
    
    # 1. Delete the fact itself
    self._conn.execute("DELETE FROM um_facts WHERE id = ?", (fact_id,))
    receipt["traces_removed"].append("um_facts")
    
    # 2. Delete all links involving this fact
    self._conn.execute(
        "DELETE FROM um_links WHERE source_id = ? OR target_id = ?",
        (fact_id, fact_id)
    )
    receipt["traces_removed"].append("um_links")
    
    # 3. Delete all access history
    self._conn.execute(
        "DELETE FROM um_access_times WHERE fact_id = ?", (fact_id,)
    )
    receipt["traces_removed"].append("um_access_times")
    
    # 4. Delete Q-value data
    if self._qvalue_store:
        self._qvalue_store._conn.execute(
            "DELETE FROM memory_qvalues WHERE memory_id = ?", (fact_id,)
        )
        receipt["traces_removed"].append("qvalues")
    
    # 5. Delete pending references
    self._conn.execute(
        "DELETE FROM um_pending WHERE promoted_to = ?", (fact_id,)
    )
    receipt["traces_removed"].append("um_pending")
    
    # 6. Update FTS index
    self._conn.execute("INSERT INTO um_facts_fts(um_facts_fts) VALUES('rebuild')")
    
    self._conn.commit()
    return receipt

def forget_by_content(self, content_pattern: str) -> list[dict]:
    """Erase all facts matching a content pattern.
    
    Example: forget_by_content("%password%") removes all password-related memories.
    """
    rows = self._conn.execute(
        "SELECT id FROM um_facts WHERE content LIKE ?", (content_pattern,)
    ).fetchall()
    
    receipts = []
    for row in rows:
        receipts.append(self.forget(row["id"]))
    return receipts
```

### Deletion Audit Trail

For compliance, maintain a lightweight audit log of deletions:

```sql
CREATE TABLE um_deletion_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id TEXT NOT NULL,           -- The deleted fact's ID
    content_hash TEXT,               -- SHA-256 of deleted content (not content itself)
    deletion_time REAL NOT NULL,
    reason TEXT,                     -- "user_request", "gdpr_erasure", "policy_match"
    traces_removed TEXT              -- JSON list of tables cleaned
);
```

The log stores the hash (not content) of deleted facts, enabling proof of deletion without retaining personal data.

## User-Facing Controls: What Mnemoria Should Expose

### Control Surface Matrix

| Control | Priority | MCP Tool | Description |
|---|---|---|---|
| **View memories** | P0 | `mcp_umemory_list` | Paginated list with filters (type, scope, date, search) |
| **Search memories** | P0 | `mcp_umemory_search` | Semantic + keyword search over stored facts |
| **Delete memory** | P0 | `mcp_umemory_forget` | Permanent erasure with audit receipt |
| **Delete by pattern** | P0 | `mcp_umemory_forget_pattern` | Erase all matching facts ("forget everything about X") |
| **Edit memory** | P1 | `mcp_umemory_edit` | Correct a stored fact (creates new version, logs old) |
| **Pin/unpin** | P1 | `mcp_umemory_pin` | Mark facts as permanent (survives consolidation/pruning) |
| **Block topics** | P2 | `mcp_umemory_policy` | Content filters ("never remember financial data") |
| **Export memories** | P2 | `mcp_umemory_export` | GDPR data portability (Article 20) |
| **Explain recall** | P2 | `mcp_umemory_explain` | Why was this memory used? |
| **Memory stats** | P3 | `mcp_umemory_stats` | Already exists -- extend with safety metrics |

### Content Blocking Policies

```sql
CREATE TABLE um_policies (
    id TEXT PRIMARY KEY,
    action TEXT NOT NULL,         -- "block", "forget", "flag"
    pattern TEXT NOT NULL,        -- regex or LIKE pattern
    scope TEXT DEFAULT 'global',  -- scope restriction
    created_at REAL,
    active BOOLEAN DEFAULT 1
);
```

At store time, check new content against active policies:

```python
def _check_policies(self, content: str) -> str:
    """Check content against user-defined memory policies.
    Returns: 'allow', 'block', or 'flag'.
    """
    policies = self._conn.execute(
        "SELECT action, pattern FROM um_policies WHERE active = 1"
    ).fetchall()
    
    for policy in policies:
        if re.search(policy["pattern"], content, re.I):
            return policy["action"]
    
    return "allow"
```

### Natural Language Controls

Users should be able to control memory via natural language:
- "Forget everything about my medical appointments"
- "Don't remember my credit card details"
- "Show me what you know about my coding preferences"
- "Delete the memory about the PostgreSQL password"

These should map to MCP tool calls via intent recognition in the agent layer (not in Mnemoria itself -- Mnemoria provides the API, the agent provides the NL interface).

## Selective Forgetting Algorithms

### The Challenge

Forgetting is harder than remembering. When a fact is deleted, its influence may persist through:
1. **Hebbian links**: Other facts linked to the deleted fact retain strengthened connections
2. **Embeddings**: Other facts may have been re-ranked due to the deleted fact's presence
3. **Q-values**: Reward signals from the deleted fact may have influenced other facts' Q-values
4. **Consolidated summaries**: If the fact was incorporated into a summary fact, the summary retains the information

### Approach 1: Cascade Deletion

Delete the fact and propagate erasure through connected structures:

```python
def deep_forget(self, fact_id: str) -> dict:
    """Deep erasure: remove fact and cascade through connections."""
    # 1. Standard forget (delete fact, links, access history)
    receipt = self.forget(fact_id)
    
    # 2. Decay links to neighbors (they lose the deleted node's signal)
    neighbors = self._get_neighbor_ids(fact_id)
    for neighbor_id in neighbors:
        # Reduce neighbor's link strengths proportionally
        self._conn.execute(
            "UPDATE um_links SET strength = strength * 0.8 "
            "WHERE source_id = ? OR target_id = ?",
            (neighbor_id, neighbor_id)
        )
    
    # 3. Reset Q-values for closely connected facts
    if self._qvalue_store:
        for neighbor_id in neighbors:
            self._qvalue_store.reset_to_default(neighbor_id)
    
    receipt["cascade_applied"] = True
    receipt["neighbors_affected"] = len(neighbors)
    return receipt
```

### Approach 2: Machine Unlearning (Academic)

Machine unlearning (Bourtoule et al., 2021) researches how to remove the influence of specific training data from ML models. For Mnemoria:

- **SISA (Sharded, Isolated, Sliced, Aggregated) training**: Partition the embedding space so individual facts can be "un-trained" from the embedding model. Not applicable to Mnemoria (uses frozen pre-trained embeddings).
- **Influence function approximation**: Estimate how much a deleted fact influenced other facts' scores and reverse the influence. Computationally expensive.
- **Exact unlearning**: Re-train from scratch without the deleted data. For Mnemoria: rebuild all Hebbian links and Q-values from access history after removing the deleted fact's contributions.

For Mnemoria's scale (100-1000 facts), exact unlearning is feasible:

```python
def unlearn(self, fact_id: str):
    """Remove a fact and rebuild derived structures without its influence."""
    # 1. Delete the fact
    self.forget(fact_id)
    
    # 2. Rebuild Hebbian links from access history
    # (remove co-occurrences involving the deleted fact)
    self._conn.execute(
        "UPDATE um_links SET co_occurrence_count = "
        "co_occurrence_count - 1 "
        "WHERE source_id IN (SELECT target_id FROM um_links WHERE source_id = ?) "
        "   OR target_id IN (SELECT source_id FROM um_links WHERE target_id = ?)",
        (fact_id, fact_id)
    )
    
    # 3. Recompute NPMI for affected links
    link_ops.update_all_npmi(self._conn)
```

### Approach 3: Gradual Decay (Neuropsychological)

Inspired by how human forgetting works: rather than hard deletion, accelerate the fact's decay until it naturally falls below pruning threshold:

```python
def gradual_forget(self, fact_id: str, decay_multiplier: float = 10.0):
    """Accelerate a fact's decay to simulate natural forgetting."""
    self._conn.execute(
        "UPDATE um_facts SET metabolic_rate = metabolic_rate * ? WHERE id = ?",
        (decay_multiplier, fact_id)
    )
    # The fact will naturally decay below pruning threshold during next consolidation
```

This preserves link structure but makes the fact increasingly unlikely to be retrieved. For GDPR compliance, this is insufficient (data still exists), but for user-initiated "I'd rather you forgot this" requests, it provides a softer experience.

### Recommended Approach for Mnemoria

**Tiered forgetting**:
1. **User says "forget X"**: Gradual decay (set metabolic_rate to 10x). Fact fades naturally.
2. **User says "delete X"**: Hard deletion via `forget()`. Immediate, permanent.
3. **GDPR erasure request**: Deep forget with cascade deletion. Audit receipt generated.
4. **Policy-based blocking**: Content matching a block policy is never stored.

## Recommended Implementation Plan

### Phase 1: Core Safety Tools (v0.3.0)

1. `forget(fact_id)` method on MnemoriaStore -- hard deletion with audit
2. `forget_by_content(pattern)` -- pattern-based mass deletion
3. `mcp_umemory_forget` and `mcp_umemory_list` MCP tools
4. `um_deletion_log` table for audit trail

### Phase 2: User Control Surface (v0.3.x)

1. `mcp_umemory_edit` -- correct a memory
2. `mcp_umemory_search` -- semantic search over memories
3. `mcp_umemory_export` -- GDPR data portability
4. `explain_recall()` method with human-readable explanations

### Phase 3: Content Policies (v0.4.0)

1. `um_policies` table with block/forget/flag actions
2. Store-time policy checking
3. `mcp_umemory_policy` tool for managing policies
4. Retroactive policy application (forget all matching existing facts)

### Phase 4: Cascade Forgetting (v0.4.x)

1. `deep_forget()` with link cascade
2. Q-value cleanup for deleted facts
3. Consolidated summary invalidation (if summaries reference deleted facts)

## Integration with Mnemoria Architecture

| Existing Feature | Safety Enhancement |
|---|---|
| Adversarial scoring | Already penalizes injection attempts. Add logging of detected adversarial content. |
| Supersession | Superseded facts should be deletable (currently retain `status='superseded'`). |
| Gauge pressure | Gauge cascade should respect user pins (already does) and deletion policies. |
| Consolidation | Consolidation sweep should check content policies and auto-forget matching facts. |
| Promoter | Pending facts should be policy-checked before promotion. |
| Q-value store | Add `reset_to_default()` for cleanup after deletion. |
| FTS5 index | Rebuild after any deletion to ensure deleted content is unsearchable. |

## Summary

| DeChant Principle | Current Status | Gap | Priority |
|---|---|---|---|
| Transparency | Partial (score components tracked) | No user-facing explanation | P1 |
| User Control | Minimal (retract pending only) | No view/delete/edit/policy | P0 |
| Bounded Influence | Good (IPS, hub dampening, homeostasis) | Contradictions not surfaced | P2 |
| Graceful Degradation | Good (try/except, adversarial scoring) | No confidence threshold | P2 |
| GDPR Compliance | None | No erasure API, no audit trail | P0 |
