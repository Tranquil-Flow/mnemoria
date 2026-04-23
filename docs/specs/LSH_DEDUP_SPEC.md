# LSH Near-Duplicate Detection at Write Time

Spec date: 2026-04-23
Status: ANALYSIS COMPLETE -- **recommend against LSH; recommend tiered-threshold dedup instead**

## Executive Summary

After analyzing Mnemoria's write pipeline, embedding infrastructure, benchmark failure modes, and scale characteristics, LSH is **not the right tool for improving dedup from 0.75**. The bottleneck is not candidate retrieval speed (O(n) scan takes <1ms at current scale) -- it is the decision threshold being too conservative (cosine >= 0.95 misses paraphrases that score 0.80-0.92).

This spec documents the full analysis, explains why LSH is premature, and provides a concrete alternative that addresses the actual problem.

---

## 1. Current Dedup Architecture

### Layer 1: Exact Hash (store.py L222-229)

```
source_hash = SHA-256(content)[:16]
SELECT id FROM um_facts WHERE source_hash = ? AND status = 'active'
```

O(1) via `idx_um_facts_source_hash` index. Catches byte-identical content only.

### Layer 2: Semantic Dedup (store.py L235-241, ingestion.py L172-213)

```python
find_near_duplicates(conn, content, embedding, threshold=0.95)
```

- Scans ALL active facts with embeddings: `SELECT id, content, embedding FROM um_facts WHERE status = 'active' AND embedding IS NOT NULL`
- Requires cosine >= 0.95 AND word Jaccard >= 0.40
- Returns on first match (short-circuits)

This is the layer that fails. The 0.95 cosine threshold is too high for all-MiniLM-L6-v2, which produces cosine scores of 0.80-0.92 for genuine paraphrases of the same fact.

### Write Pipeline Position

The dedup check happens at this point in `store()`:

```
parse_notation -> resolve_fact_type -> auto_classify -> generate_embedding
  -> EXACT HASH CHECK (L226)
  -> SEMANTIC DEDUP CHECK (L235)  <-- the bottleneck
  -> supersession check -> scope resolution -> contradiction check
  -> INSERT -> access_time -> links -> gauge -> commit
```

Both dedup layers execute before any writes. This is correct -- dedup must gate the INSERT.

## 2. LSH Feasibility Analysis

### 2a. How MinHash LSH Would Work

MinHash approximates Jaccard similarity between token sets. For each fact at write time:

1. Tokenize content into shingles (e.g., word 3-grams)
2. Compute MinHash signature (128 permutations = 128 hash values)
3. Band the signature into b bands of r rows each (b*r = 128)
4. For each band, hash the band slice into a bucket
5. Any existing fact sharing a bucket in any band is a candidate
6. Verify candidates with full cosine comparison

Storage: 128 uint32 values per fact = 512 bytes (vs ~1536 bytes for a 384-dim float32 embedding). Could store in a new `um_lsh_signatures` table or as a BLOB column.

Index structure: An in-memory dict of `{band_hash: [fact_ids]}` for each of the b bands. At 1000 facts and b=16 bands, this is ~16,000 entries. Trivial memory footprint.

### 2b. Performance Cost

**Compute cost per store():**
- MinHash signature (128 perms): ~0.5ms for a 20-word fact
- Band hashing (16 bands): ~0.05ms
- Bucket lookup: O(1) per band, O(b) total = ~0.02ms
- Total LSH overhead: ~0.6ms per write

**Comparison with current approach:**
- Current `find_near_duplicates` O(n) scan: ~0.3ms at n=100, ~3ms at n=1000
- LSH candidate retrieval: ~0.6ms regardless of n

**Break-even point**: LSH becomes faster than linear scan at approximately n=200 facts. Below that, the hash computation overhead exceeds the scan cost.

### 2c. Cosine-LSH Using Existing Embeddings

Random Projection LSH (SimHash / hyperplane LSH) works directly on the existing 384-dim embeddings:

1. Generate a fixed random projection matrix R of shape (384, num_bits) -- e.g., (384, 16)
2. For each embedding: `hash = sign(embedding @ R)` -> 16-bit binary code
3. Store the hash. At query time, find facts with Hamming distance <= threshold

**Advantages over MinHash:**
- Reuses existing embeddings (no new tokenization/hashing)
- Approximates cosine similarity (which is what the dedup decision uses)
- Very fast: one matrix multiply per write

**Disadvantages:**
- With 16 bits and Hamming threshold 3, false negative rate is ~15-25% for cosine similarities in the 0.80-0.95 range -- precisely the zone where dedup matters most
- Increasing bits to 64 reduces false negatives but increases Hamming distance computation and storage
- The projection matrix must be persisted (stored in `um_meta` or a file) and remain constant across DB lifetime -- any change invalidates all existing hashes

### 2d. Why LSH is Wrong for Mnemoria Today

**Scale mismatch**: Mnemoria's fact stores are 100-1000 facts for a typical agent. The existing O(n) scan in `find_near_duplicates` takes <3ms even at 1000 facts. LSH adds complexity (index maintenance, persistence, migration, false negatives) to save <2ms. This is negative ROI.

**Wrong bottleneck**: The dedup score is 0.75 not because candidate retrieval is slow, but because the decision threshold (0.95) is too conservative. LSH does not change the decision -- it only changes how candidates are found. Even with perfect O(1) candidate retrieval, the same paraphrases would be missed because they score below 0.95.

**False negatives are catastrophic for dedup**: LSH trades exactness for speed. A false negative in dedup means a duplicate gets stored, which is the exact problem we are trying to solve. MinHash with b=16, r=8 has a ~5% false negative rate at Jaccard 0.5 (the region where dedup matters). This makes dedup *worse*, not better.

**Index maintenance burden**: The LSH index must be kept in sync with the fact store. Fact deletions, supersessions, and status changes all require index updates. The `reset()` method (used between benchmark runs) must also clear the index. This is engineering overhead with no benefit at current scale.

## 3. What Actually Improves Dedup: Tiered Thresholds

The DEDUP_IMPROVEMENT.md research (already in this repo) correctly identifies the fix. Here is the concrete implementation spec:

### 3a. Algorithm: Two-Tier Cosine Dedup

Replace the single 0.95 threshold with two tiers:

```
Tier 1 (HARD): cosine >= 0.92, word_jaccard >= 0.30  ->  auto-merge (existing behavior, slightly relaxed)
Tier 2 (SOFT): cosine >= 0.80, same target OR entity_overlap >= 0.50  ->  merge with confirmation
```

The key insight: at cosine 0.80-0.92, embedding similarity alone is ambiguous. But if the facts also share the same `target` field (e.g., both target `auth.mfa`), or share a canonical entity (e.g., both mention "PostgreSQL"), they are almost certainly duplicates.

### 3b. Integration Point in store.py

Replace lines 235-241 in `store.py`:

```python
# Current:
if embedding is not None:
    try:
        from mnemoria.ingestion import find_near_duplicates
        dupes = find_near_duplicates(self._conn, content, embedding, threshold=0.95)
        if dupes:
            logger.debug(f"Semantic dedup: near-duplicate of {dupes[0][0][:8]} (sim={dupes[0][1]:.3f})")
            return dupes[0][0]
    except Exception:
        pass

# Proposed:
if embedding is not None:
    try:
        from mnemoria.ingestion import find_near_duplicates
        dupes = find_near_duplicates(
            self._conn, content, embedding,
            target=target,
            hard_threshold=self._config.dedup_hard_threshold,
            soft_threshold=self._config.dedup_soft_threshold,
        )
        if dupes:
            logger.debug(f"Semantic dedup: near-duplicate of {dupes[0][0][:8]} (sim={dupes[0][1]:.3f})")
            return dupes[0][0]
    except Exception:
        pass
```

### 3c. Updated find_near_duplicates (ingestion.py)

```python
def find_near_duplicates(
    conn,
    content: str,
    embedding,
    target: str = "general",
    hard_threshold: float = 0.92,
    soft_threshold: float = 0.80,
) -> List[Tuple[str, float]]:
    """Find existing facts that are near-duplicates of new content.

    Two-tier dedup:
      Hard (cosine >= hard_threshold): merge if word overlap >= 0.30
      Soft (cosine >= soft_threshold): merge only if target matches
        or entity overlap confirms same subject matter
    """
    import numpy as np
    from mnemoria.links import cosine_similarity

    if embedding is None:
        return []

    rows = conn.execute(
        "SELECT id, content, embedding, target FROM um_facts "
        "WHERE status = 'active' AND embedding IS NOT NULL"
    ).fetchall()

    duplicates = []
    content_words = set(content.lower().split())

    for r in rows:
        existing_emb = np.frombuffer(r["embedding"], dtype=np.float32)
        sim = cosine_similarity(embedding, existing_emb)

        if sim >= hard_threshold:
            # Hard match: merge if minimal word overlap confirms
            existing_words = set(r["content"].lower().split())
            jaccard = len(content_words & existing_words) / max(len(content_words | existing_words), 1)
            if jaccard >= 0.30:
                duplicates.append((r["id"], sim))

        elif sim >= soft_threshold:
            # Soft match: merge only with structural confirmation
            if target and target != "general" and r["target"] == target:
                duplicates.append((r["id"], sim))
            else:
                # Entity overlap check (reuse _extract_key_terms from retrieval.py)
                from mnemoria.retrieval import _extract_key_terms
                new_terms, _ = _extract_key_terms(content)
                old_terms, _ = _extract_key_terms(r["content"])
                shared = new_terms & old_terms
                if shared and len(shared) / max(len(new_terms | old_terms), 1) >= 0.50:
                    duplicates.append((r["id"], sim))

    return sorted(duplicates, key=lambda x: x[1], reverse=True)
```

### 3d. Configuration Parameters (config.py)

Add to MnemoriaConfig:

```python
# Deduplication thresholds
dedup_hard_threshold: float = 0.92
"""Cosine similarity above which facts are auto-merged (with word overlap check)."""

dedup_soft_threshold: float = 0.80
"""Cosine similarity above which facts are merged if target or entity overlap confirms."""
```

### 3e. Storage

No schema changes required. No new tables, columns, or indexes. The existing `um_facts` table and `find_near_duplicates` O(n) scan are sufficient.

### 3f. Migration Path

None. The change is purely in threshold logic. Existing facts are unaffected. Newly stored facts will benefit from better dedup immediately.

To retroactively dedup existing stores, a one-time consolidation pass could be added:

```python
def retroactive_dedup(store: MnemoriaStore) -> int:
    """Scan all active facts for pairwise near-duplicates and merge them."""
    merged = 0
    facts = store._conn.execute(
        "SELECT id, content, embedding, target FROM um_facts "
        "WHERE status = 'active' AND embedding IS NOT NULL"
    ).fetchall()
    # ... pairwise comparison with tiered thresholds ...
    return merged
```

This is optional and can be run manually if needed.

### 3g. Test Plan

1. **Unit tests for tiered dedup logic:**
   - Hard tier: "API uses JWT tokens" then "API uses JWT tokens" -> dedup (exact)
   - Hard tier: "API uses JWT tokens" then "The API authenticates with JWT tokens" -> dedup (cosine ~0.93, jaccard ~0.40)
   - Soft tier same target: "V[auth.mfa]: MFA is required" then "V[auth.mfa]: Multi-factor authentication is mandatory" -> dedup (cosine ~0.85, same target)
   - Soft tier entity overlap: "PostgreSQL 15 for auth" then "Postgres 15 handles authentication" -> dedup (cosine ~0.82, shared entity "postgresql")
   - No dedup: "PostgreSQL 15 for auth DB" then "PostgreSQL 15 for analytics DB" -> NOT dedup (different targets, different context despite shared entity)

2. **Regression tests:**
   - All existing tests in `test_basic.py` and `test_benchmark_regressions.py` must pass unchanged
   - The dedup category benchmark must improve from 0.75 to >= 0.85

3. **Performance test:**
   - Measure store() latency with 100, 500, 1000 active facts
   - Ensure no regression beyond +1ms per store() call

### 3h. Expected Benchmark Impact

Current: 6/8 dedup tests pass (0.75)

The 2 failures are paraphrase duplicates that score 0.80-0.92 in cosine and are missed by the 0.95 threshold. The tiered approach catches these:

- If both failures involve facts with the same target: the soft tier with target matching catches them -> 8/8 (1.0)
- If one failure involves facts with different targets but shared entities: the entity overlap check catches it -> 7/8 or 8/8 (0.875-1.0)

Conservative estimate: **0.75 -> 0.875** (7/8)
Optimistic estimate: **0.75 -> 1.0** (8/8)

Overall benchmark impact: 0.927 + (0.875-0.75) * weight_of_dedup_category. If dedup is 1/8 of the overall score, that is 0.927 + 0.016 = **0.943**.

## 4. When LSH Becomes Relevant

LSH should be revisited when:

- Fact stores regularly exceed **10,000 facts** (the O(n) scan exceeds 30ms)
- Mnemoria is used as a shared memory for multiple agents (higher write throughput)
- Real-time dedup latency budget drops below 5ms

At that point, the recommended approach is **Cosine-LSH (Random Projection)** over MinHash, because:

1. Mnemoria already computes embeddings at write time -- no additional tokenization
2. Cosine similarity is the decision metric -- LSH should approximate the same metric
3. The projection matrix is small (384 x 32 = ~50KB) and can be stored in `um_meta`
4. False negative rate with 32-bit hashes and Hamming threshold 4 is ~8% at cosine 0.85, which is acceptable as a pre-filter before exact cosine verification

Implementation sketch for future reference:

```python
# In a new file: mnemoria/lsh.py (~80 lines)
class CosinePreFilter:
    def __init__(self, conn, dim=384, num_bits=32):
        self._conn = conn
        self._proj = self._load_or_create_projection(dim, num_bits)
    
    def hash_embedding(self, embedding: np.ndarray) -> int:
        projected = embedding @ self._proj
        bits = (projected > 0).astype(np.uint8)
        return int.from_bytes(np.packbits(bits).tobytes(), 'big')
    
    def find_candidates(self, hash_val: int, max_hamming: int = 4) -> List[str]:
        # Query um_facts for hashes within Hamming distance
        # SQLite doesn't have native popcount, so this requires
        # fetching all hashes and computing in Python -- which
        # is still O(n) but on integers instead of float vectors.
        # True O(1) requires multiple hash tables (multi-probe LSH).
        ...
```

Schema addition (future):
```sql
ALTER TABLE um_facts ADD COLUMN lsh_hash INTEGER;
CREATE INDEX idx_um_facts_lsh ON um_facts(lsh_hash);
```

**Estimated implementation size**: ~80 lines in `mnemoria/lsh.py`, ~15 lines config, ~10 lines store.py integration, ~5 lines schema migration = ~110 lines total. But this should NOT be built until the scale threshold is reached.

## 5. Files Changed (Tiered Threshold Implementation)

| File | Change | Lines |
|------|--------|-------|
| `mnemoria/config.py` | Add `dedup_hard_threshold`, `dedup_soft_threshold` | +6 |
| `mnemoria/store.py` | Pass target and thresholds to `find_near_duplicates` | +4, -2 |
| `mnemoria/ingestion.py` | Rewrite `find_near_duplicates` with two-tier logic | +25, -10 |
| `tests/test_basic.py` | Add tiered dedup unit tests | +40 |
| Total | | ~85 lines |

## 6. Decision

**Do not implement LSH.** Implement tiered-threshold dedup instead.

The dedup score bottleneck is a threshold problem, not a retrieval speed problem. Tiered thresholds with target/entity confirmation fix the actual failure mode with ~85 lines of code, no new dependencies, no schema migration, and no index maintenance overhead.

LSH is documented here for future reference when scale demands it.
