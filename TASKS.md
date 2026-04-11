# Mnemoria v0.2.0 — Continuous Memory with False-Memory Prevention

## Goal

Mnemoria currently only stores what the agent explicitly tells it to. This means project constraints, discovered environment quirks, and user preferences accumulated during sessions are lost. Fix this with continuous rule-based extraction that is crash-safe and false-memory-resistant.

## Design principles

1. **Extract only from deterministic signals in v1.** Rule-based observers of tool outputs and explicit user statements. No LLM extraction yet. Fewer facts, near-zero false memories.
2. **Epistemic source tagging.** Every extracted fact records where it came from. Reliable sources promote immediately; weaker sources wait.
3. **Crash safety via append-only pending table.** Extraction writes to SQLite immediately. Promoter moves pending → confirmed asynchronously. Nothing is ever in-memory-only.
4. **Latest-wins within session.** Contradiction handling is just recency within `(type, target, session_id)`. No embeddings, no polarity detection.
5. **TTL-based session detection.** 10 min of no new pending facts for a `session_id` = session ended. No session lifecycle hooks.
6. **Source-dependent promotion latency.** `observed` and `user_stated` facts promote immediately. `agent_inference` waits for TTL.
7. **User control.** Extraction mode config, user-visible pending facts, manual retract/promote.

---

## Wave 1 — Schema foundation

### Task 1.1 — Add `um_pending` table

**File:** `mnemoria/schema.py`

Add after `um_facts` table definition:

```sql
CREATE TABLE IF NOT EXISTS um_pending (
    id              TEXT    PRIMARY KEY,
    content         TEXT    NOT NULL,
    type            TEXT    NOT NULL DEFAULT 'V',
    target          TEXT    NOT NULL DEFAULT 'general',
    scope_id        TEXT    REFERENCES um_scopes(id),
    session_id      TEXT    NOT NULL,
    source          TEXT    NOT NULL,  -- 'observed' | 'user_stated' | 'agent_inference'
    status          TEXT    NOT NULL DEFAULT 'provisional',  -- 'provisional' | 'promoted' | 'retracted'
    retracted_by    TEXT    REFERENCES um_pending(id),
    promoted_to     TEXT    REFERENCES um_facts(id),
    created_at      REAL    NOT NULL,
    updated_at      REAL    NOT NULL,
    provenance      TEXT    -- JSON: extractor name, event type, raw trigger
);

CREATE INDEX IF NOT EXISTS idx_um_pending_session   ON um_pending(session_id);
CREATE INDEX IF NOT EXISTS idx_um_pending_status    ON um_pending(status);
CREATE INDEX IF NOT EXISTS idx_um_pending_source    ON um_pending(source);
CREATE INDEX IF NOT EXISTS idx_um_pending_target    ON um_pending(type, target, session_id);
CREATE INDEX IF NOT EXISTS idx_um_pending_updated   ON um_pending(updated_at DESC);
```

### Task 1.2 — Add `provenance` column to `um_facts`

Same file. Add to `um_facts` CREATE statement:

```sql
provenance      TEXT  -- JSON: source, extractor, original pending id
```

For existing DBs, add migration in `schema.py`:

```python
def _migrate_to_v2(conn):
    """Add provenance column if missing."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(um_facts)")}
    if "provenance" not in cols:
        conn.execute("ALTER TABLE um_facts ADD COLUMN provenance TEXT")
```

Call this from `init_db()` after `SCHEMA_SQL` executes.

### Task 1.3 — Schema version tracking

Add to `schema.py`:

```sql
CREATE TABLE IF NOT EXISTS um_meta (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);
```

`init_db()` sets `('schema_version', '2')` after applying migrations.

**Verification:** Open a fresh DB and an existing DB. Both should show schema_version=2. Existing DB should have `provenance` column added to `um_facts`.

---

## Wave 2 — Fix target population (independent quick win)

### Task 2.1 — Accept and store `target` on store calls

**File:** `mnemoria/store.py`

Audit every write path (`store_fact`, ingestion paths, provider calls). Ensure `target` is:
1. Accepted as a parameter
2. Defaults to `'general'` only if the caller provided nothing
3. Persisted correctly to `um_facts.target`

**File:** `/Users/evinova/Projects/hermes-agent/plugins/memory/mnemoria/provider.py`

The provider's `_handle_store` (or equivalent) must pass `target` through from MCP arguments. Check that `mcp_umemory_store` exposes `target` as a parameter and forwards it to the store.

**File:** Same provider.

Also fix `mcp_umemory_store` to accept an optional `scope_id` (or derive one from a project hint). If not provided, leave as NULL (Mnemoria handles NULL scope_id fine).

### Task 2.2 — Backfill targets for existing facts (optional script)

**File:** `mnemoria/scripts/retag_facts.py` (new)

A one-off CLI script: `python -m mnemoria.scripts.retag_facts --db ~/.hermes/mnemoria.db`

Reads all facts with `target='general'`, prints them interactively, accepts a new target from stdin (or `skip`). Writes back via `UPDATE um_facts SET target = ? WHERE id = ?` inside a transaction with FTS triggers temporarily disabled:

```python
conn.execute("DROP TRIGGER IF EXISTS um_facts_au")
# ... do updates ...
# recreate trigger from schema
```

(The unsafe-virtual-table error we hit earlier is the FTS `au` trigger firing during the update. Dropping and recreating it around the bulk update is the clean workaround.)

**Verification:** After running on the current DB, facts should have meaningful targets like `hermes-agent`, `moonsong-identity`, `git`, etc. Recall quality should measurably improve.

---

## Wave 3 — Rule-based observers

### Task 3.1 — Observer framework

**File:** `mnemoria/observers/__init__.py` (new package)

Define a minimal protocol:

```python
from typing import Protocol, Optional
from dataclasses import dataclass

@dataclass
class PendingFact:
    content: str
    type: str           # 'C' | 'D' | 'V'
    target: str
    source: str         # 'observed' | 'user_stated' | 'agent_inference'
    provenance: dict

class Observer(Protocol):
    name: str
    def observe(self, event: dict) -> list[PendingFact]: ...
```

An `event` is a structured dict describing what just happened. Hermes-agent will emit these via a hook. Minimum fields:

```python
{
    "kind": "tool_call" | "tool_result" | "user_message" | "agent_message",
    "session_id": str,
    "timestamp": float,
    "payload": {...},  # kind-specific
}
```

### Task 3.2 — Tool-output observers

**File:** `mnemoria/observers/tool_output.py` (new)

Three deterministic extractors. Each returns `PendingFact(source='observed')` when triggered.

**PytestObserver**
- Matches `tool_call.payload.tool == 'pytest'` or shell commands matching `pytest[\s]`
- On exit 0 → no fact (pass is the default state)
- On exit non-zero with "FAILED" in output → fact: `"tests failing in {target}: {first failure message}"`, target inferred from cwd or test path
- Also retracts any provisional facts in the same session tagged as "tests passing" or similar

**GitObserver**
- Matches git commands
- `git push` rejected (non-zero with "rejected" in stderr) → fact: `"push to {branch} rejected: {reason}"`, target = repo name
- `git commit` with author different from global → fact: `"commits in {repo} use author {email}"`, target = repo name
- `git status` showing untracked file patterns → no fact (too noisy)

**FileObserver**
- Matches file read/write tool calls
- Path patterns like `.venv`, `__pycache__`, `node_modules` → no fact
- Repeated reads of config files in a session → fact: `"{project} config lives at {path}"`, target = project

Each observer is ~50 lines. They return empty lists when nothing matches.

### Task 3.3 — User-statement observer

**File:** `mnemoria/observers/user_statement.py` (new)

Rule-based pattern matching on user messages (`event.kind == 'user_message'`).

Patterns that produce `user_stated` facts:
- `"I (always|never|prefer to) ..."` → V, target inferred from context
- `"my (email|name|handle) is ..."` → C, target = 'identity'
- `"(use|don't use) X for ..."` → C, target inferred
- `"remember that ..."` → V, target = 'general' (explicit memory request)
- `"(don't|stop|please don't) ..."` → C, target inferred

Inference rules are dumb on purpose: if the message mentions a filename, use its parent directory as target; if it mentions a tool name, use the tool name; otherwise `general`.

False positive mitigation: if the user message is a question (ends in `?` or starts with wh-word), skip extraction entirely. Questions aren't preferences.

---

## Wave 4 — Promoter

### Task 4.1 — Promoter core

**File:** `mnemoria/promoter.py` (new)

A single function, no thread/daemon:

```python
def run_promotion_pass(conn, now: float, session_ttl: float = 600.0):
    """
    One pass over um_pending. Called periodically (every N writes) or on startup.
    
    Rules:
      - observed, user_stated: promote immediately (ignore TTL)
      - agent_inference: promote only if session is TTL-expired (no pending
        activity in session_id for session_ttl seconds) AND not retracted
      - Within same (type, target, session_id), latest provisional retracts earlier ones
    """
```

Implementation:
1. Run retraction pass first: for each `(type, target, session_id)` with multiple provisional rows, mark all but the latest as `retracted`.
2. Promote all `observed` and `user_stated` provisional rows whose `status = 'provisional'`: insert into `um_facts`, set `um_pending.status = 'promoted'` and `promoted_to = new fact id`.
3. For `agent_inference`: only promote if `max(updated_at) for session_id < now - session_ttl`.

All operations in a single transaction. Idempotent — running twice in a row promotes nothing new.

### Task 4.2 — Promotion trigger

**File:** `mnemoria/store.py`

Call `run_promotion_pass` from:
1. `MnemoriaStore.__init__` (drains any pending facts from crashed prior sessions)
2. After every Nth write to `um_pending` (e.g., every 10) — cheap because retraction pass is indexed
3. Expose `MnemoriaStore.flush_pending()` for external callers

### Task 4.3 — Provenance recording

When promoting pending → confirmed, write the provenance JSON to `um_facts.provenance`:

```json
{
  "source": "observed",
  "extractor": "PytestObserver",
  "pending_id": "uuid",
  "session_id": "uuid",
  "promoted_at": 1775600000.0,
  "trigger": {"kind": "tool_result", "tool": "pytest", "exit_code": 1}
}
```

---

## Wave 5 — Hermes-agent integration

### Task 5.1 — Event hook in provider

**File:** `/Users/evinova/Projects/hermes-agent/plugins/memory/mnemoria/provider.py`

Add a new method to `MnemoriaMemoryProvider`:

```python
def observe_event(self, event: dict) -> None:
    """
    Called by hermes-agent for every significant session event.
    Runs registered observers, writes pending facts, occasionally triggers promotion.
    """
```

Implementation:
1. Iterate registered observers
2. Collect `PendingFact` objects
3. Batch-insert into `um_pending` (one transaction)
4. Increment a write counter; every 10 writes, call `run_promotion_pass`

Observers are registered at provider construction:

```python
self._observers = [
    PytestObserver(),
    GitObserver(),
    FileObserver(),
    UserStatementObserver(),
]
```

### Task 5.2 — Hermes-agent hook wiring

**File:** `/Users/evinova/Projects/hermes-agent/plugins/memory/mnemoria/__init__.py`

Ensure the provider is registered so hermes-agent's event dispatch can find it. The agent already calls `system_prompt_block()` and `prefetch()` on memory providers — add a parallel `observe_event` call at the event dispatch site in hermes-agent proper.

**File (hermes-agent proper):** Locate the tool-call lifecycle hooks in hermes-agent. There's likely a `before_tool_call` / `after_tool_call` or similar. Add a call to `memory_provider.observe_event(...)` with the normalized event dict.

If no such hook exists, add one — it's a small change and this is exactly what hooks are for.

### Task 5.3 — Extraction mode config

**File:** `plugin.yaml`

Add config key:

```yaml
HERMES_MEMORY_MNEMORIA_EXTRACT_MODE:
  type: string
  default: observed_only
  description: Continuous extraction mode. off = no extraction. observed_only = tool outputs + user statements only (recommended). full = include agent_inference (v2+).
  choices:
    - "off"
    - "observed_only"
    - "full"
```

`observe_event` checks this flag first. `off` short-circuits. `observed_only` skips any observer that produces `agent_inference` facts.

---

## Wave 6 — User control surfaces

### Task 6.1 — MCP tools

**File:** `/Users/evinova/Projects/hermes-agent/plugins/memory/mnemoria/provider.py`

Add three new MCP tool handlers:

- `mcp_umemory_pending` — list pending facts, optionally filtered by session_id or source. Returns JSON with id, content, type, target, source, status, created_at.
- `mcp_umemory_retract` — takes a pending_id or fact_id. For pending: sets status='retracted'. For confirmed fact: marks as superseded.
- `mcp_umemory_promote` — takes a pending_id, forces immediate promotion (bypasses TTL rules). Useful for `agent_inference` the user has verified.

Register them in the same pattern as existing `mcp_umemory_*` tools.

### Task 6.2 — CLI helpers

**File:** `mnemoria/scripts/pending.py` (new)

CLI inspector: `python -m mnemoria.scripts.pending --db ~/.hermes/mnemoria.db`

Shows pending facts grouped by session_id, color-coded by source, with retract/promote actions. Useful for debugging and for users who want to audit what Mnemoria thinks it learned.

---

## Wave 7 — Telemetry

### Task 7.1 — Metrics table

**File:** `mnemoria/schema.py`

```sql
CREATE TABLE IF NOT EXISTS um_metrics (
    session_id       TEXT    NOT NULL,
    observer         TEXT    NOT NULL,
    event_count      INTEGER NOT NULL DEFAULT 0,
    extract_count    INTEGER NOT NULL DEFAULT 0,
    promote_count    INTEGER NOT NULL DEFAULT 0,
    retract_count    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (session_id, observer)
);
```

### Task 7.2 — Metric emission

Observers and promoter increment counters inside their transactions. Cheap — just `INSERT ... ON CONFLICT DO UPDATE SET count = count + 1`.

### Task 7.3 — Stats MCP tool extension

Extend `mcp_umemory_stats` to include per-observer metrics for the current or last session:

```json
{
  "facts": 18,
  "links": 242,
  "pending": {"provisional": 3, "promoted_last_session": 5, "retracted_last_session": 1},
  "observers": {
    "PytestObserver": {"events": 12, "extracted": 1, "promoted": 1},
    "GitObserver": {"events": 8, "extracted": 2, "promoted": 2},
    "UserStatementObserver": {"events": 34, "extracted": 4, "promoted": 4}
  }
}
```

---

## Wave 8 — Tests

### Task 8.1 — Unit tests

**File:** `tests/test_observers.py` (new)

For each observer, test positive and negative cases:
- PytestObserver: exit 0 produces nothing; exit 1 with "FAILED" produces a fact; unrelated tool calls ignored
- GitObserver: rejected push produces fact; clean push produces nothing
- UserStatementObserver: "I prefer X" produces fact; "do you prefer X?" produces nothing

### Task 8.2 — Promoter tests

**File:** `tests/test_promoter.py` (new)

- `observed` facts promote immediately
- `user_stated` facts promote immediately
- `agent_inference` facts wait for TTL
- Latest-wins retraction within session
- Crash recovery: insert pending rows, close connection without promoting, reopen, verify promotion happens on init
- Idempotency: running promoter twice promotes nothing new

### Task 8.3 — Integration test

**File:** `tests/test_continuous_extraction.py` (new)

End-to-end: feed a synthetic event stream through `observe_event`, verify pending facts accumulate, verify promotion, verify retraction on contradiction. Use a temp SQLite file.

### Task 8.4 — Benchmark regression

Run the existing benchmark suite with the new provider wiring. Confirm no regression on the current 0.910 baseline. New capabilities should only add signal, not subtract.

---

## Wave 9 — Release

### Task 9.1 — CHANGELOG and version bump

**Files:** `pyproject.toml`, `CHANGELOG.md`

Bump to `0.2.0`. CHANGELOG entry covers: continuous extraction, false-memory prevention, crash recovery, target population fix, user control surfaces.

### Task 9.2 — README update

Document:
- The extraction mode config and what each mode does
- How to inspect pending facts
- How to manually retract bad memories
- The epistemic model (why `observed` promotes fast and `agent_inference` waits)

### Task 9.3 — Push to GitHub

Once tests pass and benchmark is clean, push to `github.com/Tranquil-Flow/mnemoria.git`, tag `v0.2.0`.

---

## Out of scope (explicit non-goals for v0.2.0)

- **LLM-based extraction for `agent_inference`.** Deferred to v0.3.0, gated on telemetry from v0.2.0 showing what we're missing.
- **`P` (procedural) fact type.** Procedural knowledge belongs in the skills system, not Mnemoria.
- **`L` (lesson learned) fact type.** C/D/V with proper targeting handles this.
- **Embedding-based contradiction detection.** Recency within `(type, target, session_id)` is sufficient.
- **Explicit session lifecycle hooks from hermes-agent.** TTL-based detection is simpler and works for crashed sessions too.

---

## Verification sequence (run before declaring done)

1. `pytest tests/ -v` — all green including new observer and promoter tests
2. Fresh DB test: delete `~/.hermes/mnemoria.db`, start a session, run a few tool calls, verify `um_pending` grows
3. Crash simulation: populate `um_pending`, kill process, reopen store, verify promoted facts appear in `um_facts`
4. Contradiction test: write two conflicting pending facts in same session, verify only latest survives
5. Benchmark suite: `python -m benchmarks --backend mnemoria --suite all` — no regression from 0.910
6. Manual session test: have Moonsong run a real session, then inspect `um_pending` and `um_facts` to confirm extraction is happening naturally
