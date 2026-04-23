# Temporal Validity Implementation Spec

Target version: v0.3.0
Spec date: 2026-04-23
Mnemoria version: v0.2.1 (schema version 2)
Prior research: `docs/research/TEMPORAL_VALIDITY.md`

## Overview

Add `valid_from` and `valid_until` columns to `um_facts` so Mnemoria can track
when a fact was true in the real world, not just when it was stored. This enables
filtering out expired facts during recall, automatic invalidation on supersession
and contradiction, and eventual time-context queries.

This spec covers Phase 1 only (explicit validity columns + integration with
existing write/retrieval pipelines). Full bi-temporal tracking (sys_from/sys_until,
history table, point-in-time audit queries) is deferred to Phase 3 per the
research doc.

---

## 1. Schema Migration

### 1.1 New Columns on `um_facts`

```sql
ALTER TABLE um_facts ADD COLUMN valid_from REAL DEFAULT NULL;
ALTER TABLE um_facts ADD COLUMN valid_until REAL DEFAULT NULL;
```

Both use REAL (Unix epoch float) to match the existing `created_at`, `updated_at`,
`last_accessed` columns. This avoids type-mixing and keeps all time comparisons
in the same domain as `self._now()`.

**Semantics:**
- `valid_from`: when the fact became true. Defaults to `created_at` for new
  facts and existing facts during migration.
- `valid_until`: when the fact stopped being true. `NULL` means the fact is
  still considered valid (open-ended validity).

### 1.2 Indexes

```sql
CREATE INDEX IF NOT EXISTS idx_um_facts_valid_until ON um_facts(valid_until);
```

Only `valid_until` needs an index -- it is the column used in the hot-path
filter (`WHERE valid_until IS NULL OR valid_until > ?`). Indexing `valid_from`
is not needed until Phase 3 time-travel queries arrive.

### 1.3 Migration Script

Add to `schema.py` as `_migrate_to_v3`:

```python
def _migrate_to_v3(conn: sqlite3.Connection) -> None:
    """Add temporal validity columns if missing (v0.3.0)."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(um_facts)")}
    if "valid_from" not in cols:
        conn.execute("ALTER TABLE um_facts ADD COLUMN valid_from REAL DEFAULT NULL")
    if "valid_until" not in cols:
        conn.execute("ALTER TABLE um_facts ADD COLUMN valid_until REAL DEFAULT NULL")

    # Backfill: set valid_from = created_at for all existing rows
    conn.execute("UPDATE um_facts SET valid_from = created_at WHERE valid_from IS NULL")

    # Index
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_um_facts_valid_until
        ON um_facts(valid_until)
    """)
```

Call from `init_db()` after `_migrate_to_v2()`. Bump schema version to `'3'`.

### 1.4 Backward Compatibility

- Old databases (schema version 2) get the new columns via ALTER TABLE on first
  connection. All existing facts receive `valid_from = created_at` and
  `valid_until = NULL` (still valid). No data loss.
- The columns are nullable, so code that constructs MemoryFact from rows will
  work even if the columns are absent (the `dict.get()` pattern already used in
  `_row_to_fact` handles this).
- New code must tolerate `valid_from = None` / `valid_until = None` in all
  paths until the backfill runs.

---

## 2. Write Pipeline Changes

### 2.1 `store()` Method (store.py)

In the INSERT statement (line ~276), add `valid_from` and `valid_until`:

```python
self._conn.execute(
    """INSERT INTO um_facts
    (id, content, embedding, type, target, scope_id, status,
     activation, q_value, access_count, metabolic_rate,
     importance, category, layer, pinned,
     created_at, updated_at, last_accessed, source_hash,
     valid_from, valid_until)
    VALUES (?, ?, ?, ?, ?, ?, 'active',
            ?, 0.5, 0, ?,
            ?, ?, 'working', ?,
            ?, ?, ?, ?,
            ?, NULL)""",
    (fact_id, content, embedding_blob, ft_enum.value, target, resolved_scope_id,
     superseded_activation, metabolic_rate,
     importance, category, int(pinned),
     now, now, now, source_hash,
     now)  # valid_from = now, valid_until = NULL
)
```

**Design decision:** `valid_from` always defaults to `now` (the store timestamp).
Callers who know the real-world validity start time can pass it explicitly via a
new optional `valid_from` parameter on `store()`:

```python
def store(
    self,
    content: str,
    ...,
    valid_from: Optional[float] = None,  # NEW
) -> str:
```

If `valid_from` is provided, use it; otherwise use `now`.

### 2.2 `store_pending()` / Promoter (promoter.py)

The pending-fact pipeline does not carry temporal validity -- pending facts are
provisional and their validity starts at promotion time. When
`_promote_by_source` inserts into `um_facts`, set:

```python
valid_from = now   # promotion time = validity start
valid_until = NULL
```

This is consistent: the fact becomes "true" when Mnemoria confirms it.

### 2.3 `get_system_prompt_facts()` (store.py)

System-prompt facts (C/D/pinned/high-importance) should only include currently
valid facts. Add to the WHERE clause:

```sql
AND (valid_until IS NULL OR valid_until > ?)
```

Pass `self._now()` as the parameter.

---

## 3. Supersession Integration

### 3.1 Current Behavior

`_check_supersession()` (store.py, line ~954) marks the old fact as
`status='superseded'` when a new fact with the same (type, target, scope)
arrives. The old fact becomes invisible to recall because `_get_active_facts()`
filters on `status IN ('active', 'cold')`.

### 3.2 Change

When superseding, also set `valid_until` on the old fact:

```python
def _check_supersession(
    self, fact_type: str, target: str, scope: str, now: float,
) -> Optional[str]:
    # ... existing lookup ...
    if row:
        self._conn.execute(
            """UPDATE um_facts
               SET status='superseded', superseded_by=NULL,
                   updated_at=?, valid_until=?
               WHERE id=?""",
            (now, now, row["id"])
        )
        # ...
```

This records **when** the old fact stopped being true, not just that it was
superseded. The `valid_until` timestamp enables future time-travel queries
("what was the API URL on March 15?") even though the fact is already
status='superseded'.

---

## 4. Contradiction Detection Integration

### 4.1 Current Behavior

`check_contradictions()` (retrieval.py, line ~844) marks contradicted facts as
`status='superseded'` with `updated_at=0`.

### 4.2 Change

When a contradiction is confirmed, set `valid_until` on the contradicted fact:

```python
if score >= threshold:
    conn.execute(
        """UPDATE um_facts
           SET superseded_by=NULL, status='superseded',
               updated_at=?, valid_until=?
           WHERE id=?""",
        (now, now, r["id"])
    )
```

This requires threading the current timestamp into `check_contradictions()`.
Add a `now` parameter (currently the function uses `0` for `updated_at`, which
is wrong anyway):

```python
def check_contradictions(
    conn: sqlite3.Connection,
    new_content: str,
    new_embedding: Optional[np.ndarray],
    threshold: float = 0.12,
    category: Optional[str] = None,
    scope_id: Optional[str] = None,
    now: Optional[float] = None,  # NEW
) -> Optional[str]:
```

Update the call site in `store()` (line ~266) to pass `now`:

```python
check_contradictions(
    self._conn, content, embedding, self._config.contradiction_threshold,
    category=category, scope_id=resolved_scope_id, now=now,
)
```

---

## 5. Retrieval Changes

### 5.1 Candidate Filtering

`_get_active_facts()` (store.py, line ~934) is the single entry point for all
candidate retrieval. Add a temporal validity filter:

```python
def _get_active_facts(self, scope_id: Optional[str] = None) -> List[dict]:
    now = self._now()
    if scope_id:
        rows = self._conn.execute(
            """SELECT * FROM um_facts
               WHERE status IN ('active', 'cold')
               AND (scope_id = ? OR scope_id IS NULL)
               AND (valid_until IS NULL OR valid_until > ?)""",
            (scope_id, now)
        ).fetchall()
    else:
        rows = self._conn.execute(
            """SELECT * FROM um_facts
               WHERE status IN ('active', 'cold')
               AND (valid_until IS NULL OR valid_until > ?)""",
            (now,)
        ).fetchall()
    return [dict(r) for r in rows]
```

This filters out expired facts at the candidate stage, before any scoring
happens. This is the right place because:
- It keeps the scoring pipeline unchanged (no new component to tune).
- Expired facts should not appear in results at all by default.
- It is a single change point that covers `recall()`, `explore()`, and any
  future retrieval paths.

### 5.2 Alternative: Soft Penalty (Not Recommended for Phase 1)

An alternative is to keep expired facts as candidates but apply a score penalty
(e.g., `expired_penalty = -5.0` in `score_candidates`). This would allow
expired facts to surface when nothing else matches. This is deferred to Phase 2
because:
- It adds a tunable parameter with unclear optimal value.
- Hard filtering is simpler and matches user expectation ("expired = gone").
- The `explore()` path already includes cold/archived facts, so there is a
  separate path for historical exploration if needed.

### 5.3 Surfacing Validity in Results

Add `valid_from` and `valid_until` to the `MemoryFact` dataclass:

```python
@dataclass(frozen=True)
class MemoryFact:
    # ... existing fields ...
    valid_from:    Optional[float] = None
    valid_until:   Optional[float] = None
```

Update `_row_to_fact()` (retrieval.py, line ~810) to populate them:

```python
def _row_to_fact(row: dict, embedding, access_times) -> MemoryFact:
    # ... existing code ...
    return MemoryFact(
        # ... existing fields ...
        valid_from=row.get("valid_from"),
        valid_until=row.get("valid_until"),
    )
```

This lets consumers (MCP tools, export, UI) display validity windows.

---

## 6. MCP / Tool Layer Changes

Mnemoria does not currently have an MCP server module (the integration lives in
the hermes-a2a plugin layer). The following changes are recommendations for the
consumer layer:

### 6.1 `recall` Tool

Add an optional `time_context` parameter:

```json
{
  "name": "recall",
  "parameters": {
    "query": "string",
    "time_context": "number (optional, Unix epoch)"
  }
}
```

When `time_context` is provided, override the `now` used in
`_get_active_facts()` so the validity filter becomes:
```sql
AND (valid_until IS NULL OR valid_until > :time_context)
AND valid_from <= :time_context
```

This enables "what was true on March 15?" queries. Implementation: pass
`time_context` through `recall()` to `_get_active_facts()`.

### 6.2 Result Formatting

When returning recall results to the agent, include validity metadata:

```
V[api.url]: https://example.com/v2
  (valid since 2026-03-15, still valid)

V[api.url]: https://example.com/v1  [EXPIRED]
  (valid 2025-01-01 to 2026-03-15)
```

### 6.3 `invalidate` Tool

Add a new tool for explicit invalidation:

```json
{
  "name": "invalidate_fact",
  "parameters": {
    "fact_id": "string",
    "valid_until": "number (optional, defaults to now)"
  }
}
```

This sets `valid_until` without changing `status`, allowing the fact to age out
of default recall while remaining queryable for history.

---

## 7. Consolidation: Staleness Sweep

### 7.1 Location

Add to `consolidate()` (store.py, line ~623) after the existing lifecycle
stages (promote/demote/prune).

### 7.2 Content-Based Staleness Detection

Scan active facts for temporal language patterns that suggest the fact may be
stale:

```python
import re

STALENESS_PATTERNS = [
    (r'\bcurrently\b', 0.3),
    (r'\bright now\b', 0.4),
    (r'\bin progress\b', 0.4),
    (r'\bTODO\b', 0.5),
    (r'\btemporary\b', 0.5),
    (r'\bworkaround\b', 0.4),
    (r'\bas of \d{4}', 0.6),          # "as of 2025" -- explicitly time-bound
    (r'\bv\d+\.\d+', 0.2),            # version-pinned
    (r'\bthis week\b', 0.5),
    (r'\btoday\b', 0.5),
]

def _staleness_scan(conn, now, age_threshold_days=30):
    """Find facts with temporal language that are older than threshold."""
    cutoff = now - (age_threshold_days * 86400)
    rows = conn.execute(
        """SELECT id, content, created_at, valid_from
           FROM um_facts
           WHERE status = 'active'
           AND valid_until IS NULL
           AND created_at < ?""",
        (cutoff,)
    ).fetchall()

    candidates = []
    for row in rows:
        risk = 0.0
        for pattern, weight in STALENESS_PATTERNS:
            if re.search(pattern, row["content"], re.IGNORECASE):
                risk = max(risk, weight)
        if risk > 0:
            candidates.append((row["id"], risk))
    return candidates
```

### 7.3 Staleness Action

During consolidation, do NOT auto-expire facts based on content patterns alone.
Instead:
- Log staleness candidates in the consolidation report.
- Boost the metabolic decay rate of flagged facts by the staleness risk score
  (e.g., `new_rate = current_rate * (1 + risk)`). This accelerates natural
  decay without hard-cutting validity.
- Optionally set `valid_until` only when a superseding fact on the same target
  already exists (safe heuristic: the new fact confirms the old one is stale).

```python
# In consolidate():
stale_candidates = _staleness_scan(self._conn, now)
for fact_id, risk in stale_candidates:
    # Check if a newer fact on the same target exists
    row = self._conn.execute(
        """SELECT id FROM um_facts
           WHERE target = (SELECT target FROM um_facts WHERE id = ?)
           AND type = (SELECT type FROM um_facts WHERE id = ?)
           AND id != ?
           AND status = 'active'
           AND created_at > (SELECT created_at FROM um_facts WHERE id = ?)""",
        (fact_id, fact_id, fact_id, fact_id)
    ).fetchone()
    if row:
        # Newer same-target fact exists -- safe to expire the old one
        self._conn.execute(
            "UPDATE um_facts SET valid_until = ? WHERE id = ?",
            (now, fact_id)
        )
        report["stale_expired"] = report.get("stale_expired", 0) + 1
    else:
        # No replacement yet -- just boost decay
        self._conn.execute(
            "UPDATE um_facts SET metabolic_rate = metabolic_rate * ? WHERE id = ?",
            (1.0 + risk, fact_id)
        )
        report["stale_decay_boosted"] = report.get("stale_decay_boosted", 0) + 1
```

---

## 8. Gauge Pressure Integration

The `_gauge_check()` cascade (store.py, line ~1035) at the 70% threshold
merges duplicates. Facts with `valid_until IS NOT NULL` (expired) should be
prioritized for archival:

At the 85% threshold, add before the existing cold-scope archive:

```python
# Archive expired facts first (they have explicit valid_until)
result = self._conn.execute("""
    UPDATE um_facts SET status='archived'
    WHERE status='active'
    AND valid_until IS NOT NULL
    AND valid_until < ?
""", (self._now(),))
if result.rowcount > 0:
    actions.append(f"archived {result.rowcount} expired facts")
    self._conn.commit()
```

This is a zero-risk pressure relief valve: facts that are already known to be
invalid are the first to go under pressure.

---

## 9. Data Model Changes Summary

### 9.1 `MemoryFact` Dataclass (types.py)

```python
@dataclass(frozen=True)
class MemoryFact:
    # ... existing fields through 'pinned' ...
    valid_from:    Optional[float] = None
    valid_until:   Optional[float] = None
```

### 9.2 `_row_to_fact()` (retrieval.py)

Add to the return:
```python
valid_from=row.get("valid_from"),
valid_until=row.get("valid_until"),
```

### 9.3 `get_system_prompt_facts()` (store.py)

The method constructs `MemoryFact` manually from row dicts (line ~878). Add
`valid_from` and `valid_until` to the dict -> dataclass mapping. These fields
have defaults so no special handling needed for old DBs.

---

## 10. Test Plan

### 10.1 Unit Tests

| Test | Description | Assertions |
|------|-------------|------------|
| `test_store_sets_valid_from` | Store a fact, verify `valid_from` matches `created_at` | `valid_from == created_at` |
| `test_store_custom_valid_from` | Store with explicit `valid_from` param | `valid_from == custom_value` |
| `test_valid_until_null_by_default` | New facts have `valid_until = NULL` | `valid_until IS NULL` |
| `test_supersession_sets_valid_until` | Store V[api.url], then store another V[api.url]. Old fact gets `valid_until = new.created_at` | `old.valid_until == new.created_at` |
| `test_contradiction_sets_valid_until` | Store contradicting fact. Old fact gets `valid_until` set | `old.valid_until IS NOT NULL` |
| `test_expired_facts_excluded_from_recall` | Set `valid_until` to past, verify fact does not appear in `recall()` results | `fact not in results` |
| `test_expired_facts_excluded_from_system_prompt` | Set `valid_until` to past, verify `get_system_prompt_facts()` excludes it | `fact not in system_prompt_facts` |
| `test_unexpired_facts_returned` | Facts with `valid_until > now` or `valid_until IS NULL` appear normally | `fact in results` |
| `test_migration_backfill` | Open a schema-v2 DB, apply migration, verify all facts get `valid_from = created_at` | `all facts have valid_from IS NOT NULL` |
| `test_staleness_scan` | Store facts with temporal language ("currently", "TODO"), verify `_staleness_scan` finds them | `len(candidates) > 0` |

### 10.2 Integration Tests

| Test | Description |
|------|-------------|
| `test_supersession_chain_validity` | Store A, supersede with B, supersede with C. Verify A.valid_until == B.created_at, B.valid_until == C.created_at, C.valid_until IS NULL |
| `test_consolidation_expires_stale_with_replacement` | Store fact with "currently" text, store newer fact on same target, run `consolidate()`, verify old fact gets `valid_until` |
| `test_gauge_pressure_archives_expired_first` | Fill store to 85%+, with some expired facts. Verify expired facts are archived before valid ones |
| `test_explore_respects_validity` | Verify `explore()` (which uses `_get_active_facts`) also filters expired facts |

### 10.3 Benchmark Regression

Run the existing 538-scenario benchmark suite. Expected impact:
- `temporal_decay` category: +2-3% (explicit validity windows vs. decay-only)
- `supersession` category: +5% (valid_until tracking prevents ghost recalls)
- No category should regress, since the only behavioral change is filtering
  facts that are already `status='superseded'` in most cases.

---

## 11. Implementation Estimate

### Files Changed

| File | Changes | LOC (approx) |
|------|---------|-------------|
| `schema.py` | `_migrate_to_v3()`, update `init_db()`, bump schema version | +25 |
| `types.py` | Add `valid_from`, `valid_until` to `MemoryFact` | +2 |
| `store.py` | `store()` INSERT + param, `_check_supersession()`, `_get_active_facts()`, `consolidate()` staleness sweep, `_gauge_check()`, `get_system_prompt_facts()` | +60 |
| `retrieval.py` | `check_contradictions()` now param + valid_until, `_row_to_fact()` | +10 |
| `promoter.py` | Add `valid_from`/`valid_until` to promoted fact INSERT | +3 |
| `config.py` | (no changes needed for Phase 1) | 0 |
| **New:** `tests/test_temporal_validity.py` | All unit + integration tests from section 10 | +200 |

**Total: ~300 LOC across 6 existing files + 1 new test file.**

### Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Migration on large DBs is slow (backfill UPDATE) | Low | Backfill is a single UPDATE with no WHERE clause. SQLite handles this in-place without temp tables. |
| `_get_active_facts()` adds a filter clause -- could slow queries | Low | The `valid_until` index makes `IS NULL OR > ?` fast. Most facts will have `NULL`, so the index is highly selective. |
| Breaking frozen MemoryFact by adding fields | None | Dataclass fields have defaults; all existing construction sites use keyword args or `_row_to_fact()`. |
| Staleness scan false positives ("currently" in legitimate content) | Low | Staleness scan only boosts decay rate (reversible). Hard expiry only when a newer same-target fact exists. |

### Dependencies

None. All changes are internal to Mnemoria. No new dependencies.

---

## 12. Non-Goals (Deferred)

These are explicitly out of scope for Phase 1:

- **Full bi-temporal model** (sys_from/sys_until, um_facts_history table,
  trigger-based audit trail) -- Phase 3.
- **LLM-assisted temporal extraction** (parsing "two weeks ago" from content
  to set valid_from) -- Phase 2.
- **staleness_risk column** (persisted staleness score on um_facts) -- Phase 2.
- **Time-travel queries in recall** (time_context parameter on recall()) --
  Phase 2, after the MCP tool layer is built.
- **Soft penalty scoring** (expired facts ranked lower instead of filtered) --
  Phase 2.

---

## 13. Migration Checklist

Before merging:

- [ ] `_migrate_to_v3` runs clean on a fresh DB (no-op)
- [ ] `_migrate_to_v3` runs clean on an existing v2 DB (adds columns, backfills)
- [ ] `_migrate_to_v3` is idempotent (running twice does not error)
- [ ] All existing tests pass without modification
- [ ] New test suite passes
- [ ] Benchmark suite shows no regressions
- [ ] Schema version bumped to 3 in `um_meta`
