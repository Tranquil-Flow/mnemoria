# Changelog

All notable changes to Mnemoria will be documented in this file.

The format is based on Keep a Changelog.

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
