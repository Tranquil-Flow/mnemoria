# Changelog

All notable changes to Mnemoria will be documented in this file.

The format is based on Keep a Changelog.

## [0.2.0] - 2026-04-08

### Added

- **Continuous rule-based extraction** — `um_pending` table stores provisional facts extracted from tool outputs and user statements. Crash-safe by construction: extraction writes to SQLite immediately, never held in-memory.
- **`um_pending` table** — Append-only pending facts with source tracking (`observed`, `user_stated`, `agent_inference`), TTL-based session detection (10 min inactivity = session ended), and provenance JSON.
- **Rule-based observers** — Four deterministic extractors:
  - `PytestObserver`: Detects failing pytest runs, emits `V` facts about failures
  - `GitObserver`: Detects rejected pushes and non-default commit authors
  - `FileObserver`: Detects repeated config file reads, emits facts about config locations
  - `UserStatementObserver`: Detects explicit preferences, constraints, and memory requests from user messages
- **Promoter** — `run_promotion_pass()` moves pending → confirmed facts. `observed` and `user_stated` promote immediately. `agent_inference` waits for session TTL expiry. Within same `(type, target, session_id)` group, latest fact retracts earlier ones.
- **Crash recovery** — `MnemoriaStore.__init__` drains pending facts from crashed prior sessions on startup. `flush_pending()` exposed for external callers.
- **Target population fix** — `target` parameter now properly propagates through all write paths (`store()`, provider, backfill script). `scope_id` added to store.
- **Backfill script** — `python -m mnemoria.scripts.retag_facts` interactively reassigns targets to existing `general`-targeted facts.
- **User control surfaces** — Three MCP tools:
  - `mcp_umemory_pending`: List pending facts by session/source/status
  - `mcp_umemory_retract`: Retract pending facts or supersede confirmed facts
  - `mcp_umemory_promote`: Force-promote pending facts (bypass TTL)
- **CLI pending inspector** — `python -m mnemoria.scripts.pending` color-coded pending fact viewer with filter and action support.
- **Hermes-agent event hook** — `observe_event()` wired into tool-result and user-message paths in hermes-agent proper. Fully backward-compatible via `hasattr` guard.
- **Extraction mode config** — `HERMES_MEMORY_MNEMORIA_EXTRACT_MODE`: `off` (no extraction), `observed_only` (default, tool outputs + user statements only), `full` (include `agent_inference`).
- **Telemetry** — `um_metrics` table tracks per-(session, observer) event/extract/promote/retract counts. `mcp_umemory_stats` extended with per-observer breakdown.
- **`provenance` column on `um_facts`** — JSON field recording source, extractor, pending ID, session, and trigger event.
- **`um_meta` table** — Schema version tracking (`schema_version = 2`).

### Changed

- **Promoter idempotency**: Running `run_promotion_pass()` twice in a row promotes nothing new.
- **`get_system_prompt_facts()`**: Now strips unknown `provenance` column before constructing `MemoryFact` (backward compat for pre-v0.2.0 DBs).

### Verified

- **49 tests passing** — Wave 3 observers (15), Wave 4 promoter (13), Wave 8 integration (8), basic (8), migrate (3), benchmark regression (2).
- **No benchmark regression** from v0.1.0 baseline (0.910).

## [0.1.0] - 2026-04-08

### Fixed

- **capacity_stress** (0% → 100%): Increased `w_importance` weight from 0.4 to 0.7 in `config.py`. The importance additive boost was too weak vs ACT-R base_level at scale, causing high-importance facts to be displaced by sheer mass of low-importance noise.
- **supersession** (66.7% → 86.7%): Same fix — higher-importance facts now correctly outrank newer low-importance competitors. Added logic in `_activation_score()` to detect newer facts of the same type and decay older ones, preventing stale facts from competing with current information.

### Added

- **`get_system_prompt_facts()`** — new `MnemoriaStore` method (store.py) returning always-relevant identity facts (C constraints, D decisions, importance ≥ 0.8 V values, identity/self-target V). These are unconditionally injected into every system prompt, fixing identity drift when models switch mid-session.
- **`system_prompt_block()`** — `MnemoriaMemoryProvider` method (provider.py) now returns formatted `[MNEMORIA IDENTITY]` block instead of empty string. Ensures Moonsong identity rules survive Discord model switches.
- **RRF auto-trigger threshold lowered** (0.3 → 0.15): Keyword-heavy queries now trigger Reciprocal Rank Fusion more readily, improving retrieval_ablation keyword and hybrid sub-scores.
- **Smoke tests** for `get_system_prompt_facts()` covering: correct fact-type filtering (`test_get_system_prompt_facts`), `max_facts` cap (`test_get_system_prompt_facts_max`), empty-store guard (`test_get_system_prompt_facts_empty`).

### Changed

- **`w_importance` weight**: 0.4 → 0.7 in `config.py` — importance now carries significantly more weight in activation scoring, fixing both capacity_stress and supersession simultaneously.
- **FactType enum conversion**: Fixed bug in `get_system_prompt_facts()` where DB-stored string types weren't converted to `FactType` enum — now uses `FactType(type_str)` for proper comparison.

### Verified

- Benchmark: 0.910 overall (3-run, no variance) — up from ~0.804 before w_importance fix.
- All suites passing: contradictions (95%), cross_reference (93%), importance_filtering (100%), semantic_recall (82%), temporal_decay (89%).
- Identity block verified rendering correctly in provider.
- 8/8 smoke tests pass (`python3 tests/test_basic.py`).

### Known Limitations

- `topic_shift_recall` (75%) requires topical scoping architecture — overlapping-domain recall remains hard when both topics share dense vocabulary. Not tunable via parameter fix.
- `retrieval_ablation` semantic sub-score failures (ra_s01, ra_s02) are embedding model quality issues, not tunable from Mnemoria config.
- `timestamp_integrity` (87.5%) — verified `created_at` is immutable through consolidation; remaining 12.5% is an open benchmark fixture concern, not a code defect.
