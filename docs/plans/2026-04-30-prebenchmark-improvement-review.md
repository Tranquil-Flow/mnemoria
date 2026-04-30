# Mnemoria pre-benchmark improvement review

Date: 2026-04-30
Context: review of autonomous-session research/spec docs before running the expanded provider benchmark matrix.

## Current verified baseline

Mnemoria current repo state reviewed from `/workspace/Projects/mnemoria`:

- Package version in `pyproject.toml`: 0.2.2
- Existing result file: `benchmarks/results/v0.2.2.json`
- Existing v0.2.2 all-suite result: 393/424 = 0.9268867924528302
- Weakest old-suite categories from that result:
  - topic_shift_recall: 9/12 = 0.750
  - retrieval_ablation: 7/9 = 0.778
  - notation_parsing: 8/10 = 0.800
  - semantic_recall: 41/50 = 0.820
  - supersession: 13/15 = 0.867
  - timestamp_integrity: 7/8 = 0.875
  - temporal_decay: 40/45 = 0.889

Benchmark harness note: expanded suite now includes P-T categories:

- abstention
- preference_memory
- privacy_forgetting
- multi_hop_exploration
- long_conversation

Mnemoria adapter now declares `forgetting=True` because Mnemoria has a real targeted erasure API (`MnemoriaStore.forget()` and `forget_by_content()`). Earlier notes saying `privacy_forgetting` should be skipped are stale.

## Autonomous-session / research inputs reviewed

Sources reviewed:

- `HANDOVER_mnemoria_provider_benchmarks_2026-04-30.md`
- `docs/research/TOPIC_SHIFT.md`
- `docs/research/DEDUP_IMPROVEMENT.md`
- `docs/research/CONSOLIDATION_IMPROVEMENTS.md`
- `docs/research/MEMORY_SAFETY.md`
- `docs/research/TEMPORAL_VALIDITY.md`
- `docs/research/EMBEDDING_UPGRADE.md`
- `docs/research/PROCEDURAL_MEMORY.md`
- `docs/research/LOCOMO_BENCHMARK.md`
- `docs/specs/LSH_DEDUP_SPEC.md`
- `docs/specs/TEMPORAL_VALIDITY_SPEC.md`
- `docs/specs/EMBEDDING_UPGRADE_SPEC.md`
- `docs/specs/PROCEDURAL_MEMORY_SPEC.md`
- prior session summaries about Mnemoria extraction, v0.2.0 release, benchmark verification, and Astra's compression/inference-acceleration suggestion.

## Current implementation gaps confirmed against source

Source inspected:

- `mnemoria/types.py`
- `mnemoria/schema.py`
- `mnemoria/store.py`
- `mnemoria/config.py`
- `mnemoria/embeddings.py`
- `mnemoria/ingestion.py`

Confirmed gaps:

1. No public `forget(fact_id)` / `forget_by_content(...)` API.
   - `reset()` exists, but it erases everything.
   - Supersession/archive/cold status are not GDPR-style erasure.
   - The new benchmark privacy_forgetting category is therefore skipped for Mnemoria.

2. No temporal-validity columns.
   - `um_facts` lacks `valid_from` / `valid_until`.
   - `MemoryFact` lacks validity metadata.
   - Supersession marks status only; it does not record when the old fact stopped being true.

3. No procedural fact type.
   - `FactType` has C/D/V/?/done/obs only.
   - `NOTATION_PATTERN` does not accept `P[...]`.
   - No `ProceduralObserver`.

4. Embedding upgrade spec is not implemented.
   - Current sentence-transformer default remains `all-MiniLM-L6-v2`.
   - No `embedding_dimension` config.
   - No Nomic task prefixes or Matryoshka truncation path.
   - Recall/store calls do not pass `is_query` / `is_document` flags.

5. Topic segmentation spec is not implemented.
   - No `topic_segment_id` column.
   - No `um_topic_segments` table.
   - No topic-coherence boost/dampening.

6. Tiered dedup spec is not implemented.
   - `find_near_duplicates()` has a single threshold argument and requires word overlap >= 0.4.
   - It does not accept target, hard/soft thresholds, or structural confirmation.
   - Note: current code default threshold is 0.85, while the LSH spec described an older 0.95 call site. The broad conclusion still holds: LSH is not worth adding before benchmarks; if touching dedup, improve decision logic rather than adding an approximate index.

7. Compression/inference acceleration remains design-only.
   - Astra's suggestion was saved as a design direction: store full memories; compress only the injected context.
   - Not relevant for the standalone provider benchmark unless the benchmark measures prompt-injection budget/latency.

## Recommended before provider matrix

### P0: Keep benchmark harness changes; do not run provider matrix until committed/reviewed

Already done in the benchmark worktree:

- schema v2 output
- runtime metadata
- safe credential presence reporting
- `--preflight`
- score views: executed/core/per-track
- skipped categories with reasons
- expanded P-T categories
- report generator support for v2 fields

This makes the moonlit comparison fairer. Run external provider preflights only after these changes are reviewed/committed.

### P1: Add targeted erasure to Mnemoria before the full provider matrix

Why:

- It is a real product/safety requirement, not benchmark gaming.
- It directly addresses `docs/research/MEMORY_SAFETY.md` P0 user-control gap.
- It lets Mnemoria participate in `privacy_forgetting` instead of skipping it.
- It is relatively contained: SQL deletion across facts, links, access history, pending references, qvalues, and FTS rebuild.

Suggested minimal API:

- `MnemoriaStore.forget(fact_id: str, reason: str | None = None) -> dict`
- `MnemoriaStore.forget_by_content(content_pattern: str, reason: str | None = None) -> list[dict]`
- Optional but recommended: `um_deletion_log` with content hash only, not content.

Important: delete hard, do not just mark `status='deleted'`, if claiming forgetting capability.

### P1: Add a simple list/all-active API if needed for control surface

The Hermes benchmark adapter currently works through recall/store, but user-control tooling needs inspectability. If implementing forget, add a safe inspection helper too:

- `MnemoriaStore.list_facts(scope=None, fact_type=None, limit=50, offset=0)`
- or expose a narrower `get_fact(fact_id)` for deletion receipts/tests.

Do not expose raw credentials or private DB internals.

### P2: Implement tiered dedup only if we have time for tests

Why:

- Existing result shows deduplication is not the weakest current category anymore, but duplicate/near-duplicate behavior affects long_conversation, preference_memory, and safety.
- The LSH spec explicitly recommends against LSH now; use tiered thresholds instead.

Suggested minimal change:

- Add config fields `dedup_hard_threshold=0.92`, `dedup_soft_threshold=0.80`.
- Update `find_near_duplicates()` to include target-aware soft tier.
- Add regression tests for paraphrase dedup and non-duplicate same-vocabulary facts.

Risk:

- Too-aggressive soft dedup can collapse distinct facts. Only do this with tests that prove false-positive resistance.

### P2: Add temporal validity if supersession/staleness remains a priority

Why:

- It improves stale-fact safety and timestamp integrity.
- It is a principled product feature.

Risk:

- Schema migration touches core tables and retrieval filters.
- It is more invasive than targeted forget.

Recommendation:

- Good for v0.3.0, but not necessary before the provider matrix unless we specifically want to benchmark time-validity behavior.

### P3: Defer Nomic embedding upgrade for this benchmark run

Why defer:

- The provider matrix should be comparable and reproducible in the current container.
- The handover warns not to install heavy sentence-transformers in the container; large model downloads are brittle here.
- Current benchmark plan uses `--embedding tfidf` for Mnemoria fairness/smoke runs.

When to do it:

- Run as a separate local/host-side experiment with cached models and explicit result labeling.
- Do not mix embedding-model upgrade into provider matrix results unless all backends' embedding choices are documented.

### P3: Defer topic segmentation/procedural memory until after matrix

Why defer:

- Topic segmentation targets topic_shift_recall, but requires schema and retrieval scoring changes.
- Procedural memory is valuable for product differentiation, but not necessary for fair provider matrix unless a procedural-memory track is explicitly added.
- Both are larger feature branches and could confound provider comparisons.

### P3: Defer context compression/inference acceleration

Why defer:

- Compression should happen at context injection time, not storage time.
- It is an agent-runtime feature more than a Mnemoria core benchmark feature.
- It needs separate metrics: token count, latency, lost-critical-fact rate, negation preservation.

## Proposed immediate work order

1. Finish benchmark worktree hygiene and commit-ready diff review.
2. Implement and test Mnemoria targeted erasure API.
3. Update Hermes benchmark Mnemoria adapter capabilities to `forgetting=True` only after real package API exists and tests prove deletion.
4. Run local Mnemoria preflight + privacy_forgetting smoke.
5. Optionally implement tiered dedup if time remains and tests are strong.
6. Then run external provider preflights and full provider matrix.

## Suggested acceptance checks for P1 forgetting

Mnemoria tests:

- Store fact -> recall sees it -> `forget(fact_id)` -> recall no longer sees it.
- Links involving forgotten fact are gone.
- Access times are gone.
- Q-value row is gone if qvalue store exists.
- Pending/promoted references are gone or scrubbed.
- FTS index no longer returns the content.
- Deletion receipt contains fact id, deletion timestamp, traces removed, and content hash only.
- `forget_by_content()` deletes all matching facts and leaves non-matching facts intact.

Benchmark adapter tests:

- Mnemoria backend advertises `forgetting=True` only with the new API.
- `privacy_forgetting` executes, not skips.
- The adapter's `forget()` method calls the real package method, not test-only logic.

## Final recommendation

Before the provider matrix, implement only the safety/control feature that clearly matters for both product integrity and the new benchmark: targeted forgetting. Do not roll in Nomic embeddings, topic segmentation, temporal validity, or procedural memory before this matrix; those are real v0.3+ work, but they would muddy the comparison spell.
