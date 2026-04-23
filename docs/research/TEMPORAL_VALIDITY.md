# Temporal Validity Implementation Research

Research date: 2026-04-23
Mnemoria version: v0.2.1 (0.927 internal benchmark)

## Problem Statement

Mnemoria currently tracks creation time (`created_at`) and access history (`um_access_times`) but has no explicit model for when a fact is valid in the real world or when it became stale. Facts like "V[api.url]: https://api.example.com/v2" may become incorrect without Mnemoria knowing. The metabolic decay system approximates staleness via type-based decay rates, but this is a proxy -- a constraint with a slow decay rate (0.3x) might still become invalid if the underlying system changes.

This document researches temporal validity patterns used by competitors and in database theory to inform Mnemoria's implementation.

## How Zep/Graphiti Implements Temporal Validity

### Architecture: Bi-Temporal Knowledge Graph

Zep's Graphiti engine (https://github.com/getzep/graphiti) uses a bi-temporal model on every edge (fact/relationship) in its knowledge graph. The paper is "Zep: A Temporal Knowledge Graph Architecture for Agent Memory" (arXiv:2501.13956).

### Four Temporal Fields per Edge

Every edge in Graphiti carries four timestamps:

| Field | Timeline | Description |
|-------|----------|-------------|
| `created_at` | Transaction (T') | When the fact was first ingested into the system |
| `expired_at` | Transaction (T') | When the fact was invalidated in the system (versioning/logical deletion) |
| `valid_at` | Event (T) | When the fact became true in the real world |
| `invalid_at` | Event (T) | When the fact stopped being true in the real world |

### Dual Timeline Model

**Timeline T (Event Time):** Chronological ordering of events in the real world. Extracted from conversation content using both absolute timestamps ("born on June 23, 1912") and relative dates ("two weeks ago"), resolved against the episode's reference timestamp.

**Timeline T' (Transaction Time):** Transactional order of data ingestion. Records when the system learned about each fact. Enables "as-recorded" queries for auditing.

The separation allows answering both:
- "What was true on March 15?" (event time query)
- "What did the system know on March 15?" (transaction time query)

### Invalidation Process

When new information arrives that contradicts an existing edge:

1. New edges are extracted from the incoming episode
2. The system retrieves semantically related existing edges
3. An LLM compares new edges against existing ones to identify contradictions
4. For temporally overlapping conflicts, the system **invalidates** the previous edge by setting `invalid_at` to the `valid_at` of the new invalidating edge
5. Old edges are **never deleted** -- only marked invalid
6. The system consistently prioritizes new information

### Database Backend

Graphiti uses graph databases (Neo4j, FalkorDB, Kuzu, Amazon Neptune) rather than SQLite. The temporal fields are stored as properties on relationship edges.

### Edge Schema (reconstructed from sources)

```
Edge {
    uuid: str
    source_node_uuid: str
    target_node_uuid: str
    fact: str                 # Natural language fact
    fact_embedding: float[]   # Vector embedding
    name: str                 # Relationship name
    group_id: str
    episodes: list            # Source episodes
    created_at: datetime      # T' - when ingested
    expired_at: datetime      # T' - when system-invalidated
    valid_at: datetime        # T  - when became true
    invalid_at: datetime      # T  - when stopped being true
    attributes: dict          # Additional metadata
}
```

### Relevance to Mnemoria

Graphiti's approach is sophisticated but depends on:
- Graph database infrastructure (Neo4j)
- LLM calls for contradiction detection on every ingestion
- Entity resolution and triplet extraction

Mnemoria already has simpler versions of several components:
- **Contradiction detection** (entity overlap + update language patterns)
- **Supersession** (same type+target replacement)
- **Created_at tracking** (already in `um_facts`)

What Mnemoria lacks: explicit `valid_at` / `invalid_at` fields and the ability to query "what was true at time X."

## Bi-Temporal Data Patterns for SQLite

### Core Concept

A bi-temporal table tracks two independent time dimensions:

1. **Valid time (application time):** When the fact was/is true in the real world
2. **Transaction time (system time):** When the fact was recorded/modified in the database

### SQLite Implementation Pattern

SQLite has no built-in temporal table support (unlike SQL Server or PostgreSQL). Implementation requires manual schema design with triggers.

#### Schema Design

```sql
-- Extended um_facts table with temporal validity
ALTER TABLE um_facts ADD COLUMN valid_from TEXT;      -- ISO 8601, when fact became true
ALTER TABLE um_facts ADD COLUMN valid_until TEXT;      -- ISO 8601, when fact stopped being true (NULL = still valid)
ALTER TABLE um_facts ADD COLUMN sys_from TEXT;         -- when record was created in DB
ALTER TABLE um_facts ADD COLUMN sys_until TEXT;         -- when record was superseded in DB (NULL = current)

-- History table for full audit trail
CREATE TABLE um_facts_history (
    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    type TEXT NOT NULL,
    target TEXT,
    valid_from TEXT,
    valid_until TEXT,
    sys_from TEXT NOT NULL,
    sys_until TEXT NOT NULL,
    operation TEXT NOT NULL,  -- 'INSERT', 'UPDATE', 'INVALIDATE'
    FOREIGN KEY (fact_id) REFERENCES um_facts(id)
);
```

#### Trigger-Based Tracking

```sql
-- Auto-populate sys_from on insert
CREATE TRIGGER um_facts_sys_insert
AFTER INSERT ON um_facts
BEGIN
    UPDATE um_facts
    SET sys_from = datetime('now'),
        valid_from = COALESCE(NEW.valid_from, datetime('now'))
    WHERE id = NEW.id AND sys_from IS NULL;
END;

-- On update: close old version, keep history
CREATE TRIGGER um_facts_sys_update
BEFORE UPDATE ON um_facts
WHEN OLD.content != NEW.content OR OLD.type != NEW.type
BEGIN
    INSERT INTO um_facts_history
        (fact_id, content, type, target, valid_from, valid_until, sys_from, sys_until, operation)
    VALUES
        (OLD.id, OLD.content, OLD.type, OLD.target, OLD.valid_from, OLD.valid_until,
         OLD.sys_from, datetime('now'), 'UPDATE');
END;
```

#### Query Patterns

```sql
-- Current valid facts (default view)
SELECT * FROM um_facts
WHERE (valid_until IS NULL OR valid_until > datetime('now'))
  AND (sys_until IS NULL);

-- What was true at a specific point in time
SELECT * FROM um_facts
WHERE valid_from <= '2026-03-15T00:00:00'
  AND (valid_until IS NULL OR valid_until > '2026-03-15T00:00:00')
  AND (sys_until IS NULL);

-- Full audit: what the system knew at a point in time
SELECT * FROM um_facts
WHERE sys_from <= '2026-03-15T00:00:00'
  AND (sys_until IS NULL OR sys_until > '2026-03-15T00:00:00');
```

### Key Invariants

1. **No gaps**: `valid_until` of superseded version must equal `valid_from` of replacement
2. **No deletions**: Old records get `sys_until` set, never removed
3. **Immutable history**: The `um_facts_history` table is append-only
4. **NULL means current**: Both `valid_until = NULL` and `sys_until = NULL` mean "still active"

## Memory Staleness Detection

### The Staleness Problem

Long-lived agents act on stale data: addresses change, preferences shift, APIs migrate, teams reorganize. Without staleness detection, an agent may confidently use a fact from 2024 in 2026.

### Approaches in the Literature

#### 1. TTL-Based Expiry (Mem0)

Mem0 integrates Time-To-Live (TTL) directly into its storage API:
- Developers assign specific lifespans to memory blocks
- Expired memories are automatically pruned
- Simple but lossy -- expired facts are gone, not just deprioritized

**Mnemoria parallel:** Metabolic decay rates already serve this role (? facts decay at 2.0x, ~ at 5.0x), but decay affects activation score, not validity. A fact with low activation is ranked lower but still "true."

#### 2. Type-Based Decay Heuristics (Current Mnemoria Approach)

| Fact Type | Decay Rate | Rationale |
|-----------|-----------|-----------|
| C (Constraint) | 0.3x | Rules change slowly |
| D (Decision) | 0.7x | Architectural choices are semi-stable |
| V (Value) | 1.0x | Concrete values change at normal rate |
| ? (Unknown) | 2.0x | Open questions resolve quickly |
| Done | 2.5x | Resolved items lose relevance fast |
| ~ (Obsolete) | 5.0x | Already superseded |

This is a good first approximation but lacks fact-specific temporal signals.

#### 3. Temporal Semantic Memory (TSM)

Academic approach that models semantic time explicitly:
- Builds a semantic timeline from conversational cues
- Consolidates temporally continuous information
- Enables retrieval of "temporally appropriate" memories
- More sophisticated than TTL but requires temporal extraction

#### 4. Keyword-Based Staleness Signals

Pattern matching on fact content to detect likely-stale entries:
- "in progress", "TODO:", "just now", "currently" -- signals temporal dependency
- "as of [date]" -- explicit temporal anchor
- "v1.2", "version X" -- version-pinned facts age faster
- Patterns scoped by fact type for precision

#### 5. Temporal Reflection Summaries

Compress old memories into time-bound summaries:
- "User's primary stack changed from React to Svelte in March 2026"
- Preserves the transition while marking old facts as superseded
- Used by some systems for long-term memory compaction

### Recommended Staleness Detection for Mnemoria

A layered approach combining multiple signals:

#### Layer 1: Explicit Temporal Validity (Schema Change)

Add `valid_from` and `valid_until` to `um_facts`:

```sql
ALTER TABLE um_facts ADD COLUMN valid_from TEXT DEFAULT NULL;
ALTER TABLE um_facts ADD COLUMN valid_until TEXT DEFAULT NULL;
```

- `valid_from` set automatically on insertion (default: now)
- `valid_until` set manually via supersession, contradiction detection, or explicit user action
- Facts with `valid_until < now` are excluded from default recall but still queryable for history

#### Layer 2: Content-Based Staleness Signals

Extend the existing observer/extraction pipeline to detect temporal markers:

```python
STALENESS_PATTERNS = {
    'temporal_dependency': [
        r'\bcurrently\b', r'\bright now\b', r'\bin progress\b',
        r'\bas of \d{4}', r'\btoday\b', r'\bthis week\b',
    ],
    'version_pinned': [
        r'\bv\d+\.\d+', r'\bversion \d+', r'\b\d+\.\d+\.\d+\b',
    ],
    'explicitly_temporary': [
        r'\bTODO\b', r'\bFIXME\b', r'\bHACK\b', r'\btemporary\b',
        r'\bworkaround\b', r'\buntil\b',
    ],
}
```

Facts matching these patterns get:
- Increased metabolic decay rate (e.g., 1.5x-2.0x boost)
- A `staleness_risk` field for UI/tool surfacing
- Optional: auto-set `valid_until` based on heuristic (e.g., "as of March" -> valid_until = end of March)

#### Layer 3: Access-Based Staleness Detection

Use existing access history (`um_access_times`) as a signal:
- Facts accessed frequently then suddenly not accessed may be stale
- Facts never accessed after a long period are candidates for staleness review
- Combine with contradiction detection: if a newer fact on the same target exists, mark older fact's `valid_until`

#### Layer 4: Consolidation-Time Review

During `consolidate()`, add a staleness sweep:
- Scan facts with temporal content markers older than threshold
- Check for same-target supersession candidates
- Promote stale facts to `~` (Obsolete) type or set `valid_until`
- Generate staleness report for agent/user review

### Integration with Existing Mnemoria Features

| Existing Feature | Temporal Enhancement |
|---|---|
| Metabolic decay | Staleness patterns increase decay rate |
| Supersession | Auto-sets `valid_until` on superseded fact |
| Contradiction detection | Sets `valid_until` when contradiction confirmed |
| Consolidation | Staleness sweep during lifecycle management |
| Gauge pressure | Stale facts become pressure-pruning candidates |
| Tarjan bridge protection | Stale bridge nodes flagged for review instead of auto-pruning |

### Proposed Schema Changes

```sql
-- Minimal temporal validity extension
ALTER TABLE um_facts ADD COLUMN valid_from TEXT DEFAULT NULL;
ALTER TABLE um_facts ADD COLUMN valid_until TEXT DEFAULT NULL;
ALTER TABLE um_facts ADD COLUMN staleness_risk REAL DEFAULT 0.0;

-- Index for temporal queries
CREATE INDEX idx_um_facts_valid_from ON um_facts(valid_from);
CREATE INDEX idx_um_facts_valid_until ON um_facts(valid_until);

-- View for currently-valid facts only
CREATE VIEW um_facts_current AS
SELECT * FROM um_facts
WHERE (valid_until IS NULL OR valid_until > datetime('now'))
  AND is_active = 1;
```

### Estimated Impact

| Benchmark Category | Current | Expected | Mechanism |
|---|---|---|---|
| temporal_decay | 0.933 | 0.96+ | Explicit validity windows |
| supersession | 0.867 | 0.92+ | valid_until on superseded facts |
| contradictions | 0.950 | 0.97+ | Temporal invalidation tracking |
| topic_shift_recall | 0.833 | 0.86+ | Stale facts deprioritized per-topic |

### Implementation Priority

1. **Phase 1 (v0.3.0):** Add `valid_from`/`valid_until` columns. Auto-populate on store. Set `valid_until` during supersession and contradiction detection. Filter in default recall.
2. **Phase 2 (v0.3.x):** Content-based staleness patterns. `staleness_risk` scoring. Consolidation sweep.
3. **Phase 3 (v0.4.0):** Full bi-temporal with `sys_from`/`sys_until`. History table. Point-in-time queries. Time-travel API.

Phase 1 captures 80% of the value with minimal complexity. Phase 3 is only needed if agents need to reason about "what the system knew at time X" (audit/debugging use case).
