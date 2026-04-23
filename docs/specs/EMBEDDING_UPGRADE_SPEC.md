# Embedding Upgrade Spec: all-MiniLM-L6-v2 -> nomic-embed-text-v1.5

Spec date: 2026-04-23
Prereq: docs/research/EMBEDDING_UPGRADE.md (research complete)
Target: Mnemoria v0.3.0

## Summary

Replace the all-MiniLM-L6-v2 embedding model (384d, MTEB ~56%) with
nomic-embed-text-v1.5 (768d native / 256d Matryoshka, MTEB 62.28).
This unblocks the semantic_recall 0.800 ceiling and improves every
retrieval-dependent benchmark category.

## Dimension Choice: 256d Matryoshka

Use 256d (Matryoshka truncation), not full 768d.

**Quality tradeoff:**
- 768d: MTEB 62.28
- 512d: MTEB 61.96 (-0.32 points, -0.5%)
- 256d: MTEB 61.04 (-1.24 points, -2.0%)
- 128d: MTEB 59.70 (-2.58 points, -4.1%)

**Storage tradeoff:**

| Config | Bytes/fact | 1000 facts | 10000 facts |
|--------|-----------|------------|-------------|
| Current (384d f32) | 1,536 B | 1.5 MB | 15 MB |
| nomic 768d f32 | 3,072 B | 3.0 MB | 30 MB |
| nomic 256d f32 | 1,024 B | 1.0 MB | 10 MB |

**Decision:** 256d saves 33% storage versus the current 384d model while
delivering +5 MTEB points. The 1.24-point gap to 768d is negligible for
Mnemoria's use case (short facts, <1000 entries typical). The 768d option
doubles storage for a 2% quality gain that will not move benchmark scores.

If a future benchmark shows 256d is insufficient, upgrading to 512d
requires only changing one constant and re-running migration -- no schema
or API changes.

## Exact Code Changes

### 1. `mnemoria/config.py` -- Add embedding_dimension field

**File:** `/Users/evinova/Projects/mnemoria/mnemoria/config.py`
**Location:** After `embedding_model` field (line 111)

Add new field:

```python
embedding_dimension: int = 256
"""Embedding vector dimension. For Matryoshka models, truncates to this dim.
Use 256 for nomic-embed-text-v1.5 (Matryoshka), 384 for MiniLM, 768 for full nomic."""
```

Change the `embedding_model` default from `"auto"` to `"auto"` (no change needed --
the auto-detection chain in EmbeddingProvider will handle the new model name).

### 2. `mnemoria/embeddings.py` -- New model + task prefixes + Matryoshka

**File:** `/Users/evinova/Projects/mnemoria/mnemoria/embeddings.py`

#### 2a. SentenceTransformerEmbedder.__init__ (line 232)

Change default model_name and add dimension + prefix support:

```python
class SentenceTransformerEmbedder:
    """Wraps sentence-transformers for high-quality embeddings."""

    _shared_models: dict = {}

    # Task prefix constants for nomic-embed-text-v1.5
    QUERY_PREFIX = "search_query: "
    DOCUMENT_PREFIX = "search_document: "

    # Models that require task prefixes
    PREFIX_MODELS = frozenset({
        "nomic-ai/nomic-embed-text-v1.5",
        "nomic-ai/nomic-embed-text-v1",
    })

    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        matryoshka_dim: Optional[int] = None,
    ):
        try:
            if model_name in self._shared_models:
                self._model = self._shared_models[model_name]
                logger.info(f"Reusing cached sentence-transformers model: {model_name}")
            else:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(model_name, trust_remote_code=True)
                self._shared_models[model_name] = self._model
                logger.info(f"Loaded sentence-transformers model: {model_name}")

            self._native_dimension = self._model.get_sentence_embedding_dimension()
            self._matryoshka_dim = matryoshka_dim
            self._uses_prefix = model_name in self.PREFIX_MODELS
            self._model_name = model_name

            if matryoshka_dim and matryoshka_dim > self._native_dimension:
                raise ValueError(
                    f"matryoshka_dim={matryoshka_dim} exceeds native "
                    f"dimension={self._native_dimension}"
                )
        except Exception as e:
            raise ImportError(f"sentence-transformers not available: {e}")

    @property
    def dimension(self) -> int:
        return self._matryoshka_dim or self._native_dimension

    def _truncate_and_normalize(self, embedding: np.ndarray) -> np.ndarray:
        """Apply Matryoshka truncation + L2 re-normalization."""
        if self._matryoshka_dim and self._matryoshka_dim < len(embedding):
            embedding = embedding[:self._matryoshka_dim]
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        return embedding

    def encode(self, text: str, is_query: bool = False):
        """Encode text. Set is_query=True for recall queries, False for documents."""
        if self._uses_prefix:
            prefix = self.QUERY_PREFIX if is_query else self.DOCUMENT_PREFIX
            text = prefix + text
        embedding = self._model.encode(text, normalize_embeddings=True)
        embedding = np.asarray(embedding, dtype=np.float32)
        return self._truncate_and_normalize(embedding)

    def encode_batch(self, texts: List[str], is_query: bool = False) -> list:
        if self._uses_prefix:
            prefix = self.QUERY_PREFIX if is_query else self.DOCUMENT_PREFIX
            texts = [prefix + t for t in texts]
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return [
            self._truncate_and_normalize(np.asarray(e, dtype=np.float32))
            for e in embeddings
        ]
```

Key points:
- `trust_remote_code=True` is required for nomic models (custom architecture)
- Task prefixes are applied inside encode/encode_batch, transparent to callers
- Matryoshka truncation is L2-renormalized after slicing (required for cosine similarity correctness)
- `is_query` parameter distinguishes recall-path vs store-path encoding

#### 2b. EmbeddingProvider -- Pass through is_query and dimension (line 275)

```python
class EmbeddingProvider:
    def __init__(self, model: str = "auto", dimension: Optional[int] = None):
        self._backend = None
        self._backend_name = "none"
        self._model_config = model
        self._dimension_config = dimension

        if model == "auto":
            self._try_fallback_chain()
        elif model == "sentence-transformers":
            self._try_sentence_transformers()
        elif model == "tfidf":
            self._init_tfidf()
        else:
            self._try_fallback_chain()

        if self._backend is None:
            self._init_tfidf()

    def _try_sentence_transformers(self) -> None:
        self._backend = SentenceTransformerEmbedder(
            matryoshka_dim=self._dimension_config,
        )
        self._backend_name = "sentence-transformers"
        logger.info("Using sentence-transformers embeddings")

    def encode(self, text: str, is_query: bool = False):
        """Encode text. Set is_query=True for recall queries."""
        if hasattr(self._backend, 'encode') and 'is_query' in self._backend.encode.__code__.co_varnames:
            return self._backend.encode(text, is_query=is_query)
        return self._backend.encode(text)

    def encode_batch(self, texts: List[str], is_query: bool = False) -> list:
        if hasattr(self._backend, 'encode_batch') and 'is_query' in self._backend.encode_batch.__code__.co_varnames:
            return self._backend.encode_batch(texts, is_query=is_query)
        return self._backend.encode_batch(texts)
```

The `is_query` passthrough uses introspection so TfidfEmbedder (no prefix
concept) works without modification.

#### 2c. Update HF cache path detection (line 42)

Change:
```python
_CACHED_MODEL = _PERSISTENT_HF_CACHE / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2"
```
To:
```python
_CACHED_MODEL_LEGACY = _PERSISTENT_HF_CACHE / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2"
_CACHED_MODEL_NOMIC = _PERSISTENT_HF_CACHE / "hub" / "models--nomic-ai--nomic-embed-text-v1.5"
if _CACHED_MODEL_LEGACY.exists() or _CACHED_MODEL_NOMIC.exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
```

### 3. `mnemoria/store.py` -- Pass is_query flag through encode calls

**File:** `/Users/evinova/Projects/mnemoria/mnemoria/store.py`

#### 3a. store() method -- documents get is_query=False (line 218)

Line 218 currently reads:
```python
embedding = embedder.encode(embed_text)
```

Change to:
```python
embedding = embedder.encode(embed_text, is_query=False)
```

This is the default behavior (documents), so functionally identical, but
making it explicit documents the intent and ensures correctness if the
default ever changes.

#### 3b. recall() method -- queries get is_query=True (line 458)

Line 458 currently reads:
```python
query_embedding = embedder.encode(query) if embedder else None
```

Change to:
```python
query_embedding = embedder.encode(query, is_query=True) if embedder else None
```

#### 3c. simulate_access() method -- semantic fallback uses is_query=True (line 602)

Line 602 currently reads:
```python
emb = self._get_embedder().encode(content_substring)
```

Change to:
```python
emb = self._get_embedder().encode(content_substring, is_query=True)
```

#### 3d. _get_embedder() -- Pass dimension config (line 131)

Line 131 currently reads:
```python
self._embedder = EmbeddingProvider(model=self._config.embedding_model)
```

Change to:
```python
self._embedder = EmbeddingProvider(
    model=self._config.embedding_model,
    dimension=self._config.embedding_dimension,
)
```

### 4. `mnemoria/retrieval.py` -- check_contradictions query encoding

**File:** `/Users/evinova/Projects/mnemoria/mnemoria/retrieval.py`

No changes needed. `check_contradictions` receives `new_embedding` as a
parameter from `store()`, which already encodes with `is_query=False`. The
contradiction check compares document embeddings against document embeddings,
which is correct (both sides are `search_document:` prefixed).

### 5. `mnemoria/links.py` -- cosine_similarity dimension handling

**File:** `/Users/evinova/Projects/mnemoria/mnemoria/links.py`

No changes needed. The existing `cosine_similarity` function (line 35)
already handles different-length vectors via zero-padding. During
migration, if any old 384d embeddings coexist with new 256d embeddings,
the padding logic will work correctly (though similarity scores between
mixed-dimension embeddings will be degraded -- this is expected and
handled by the migration process).

### 6. `mnemoria/ingestion.py` -- find_near_duplicates

**File:** `/Users/evinova/Projects/mnemoria/mnemoria/ingestion.py`

No changes needed. `find_near_duplicates` receives pre-computed embeddings
and uses `cosine_similarity` from links.py, which handles dimension
mismatches.

## Migration Script

### `mnemoria/cli/migrate_embeddings.py`

New file. Batch re-embeds all facts in `um_facts` using the new model.

```python
"""
Re-embed all facts in um_facts with the new embedding model.

Usage:
    python -m mnemoria.cli.migrate_embeddings [--db PATH] [--batch-size N] [--dry-run]

Steps:
1. Records old embedding_model + dimension in um_meta (for rollback)
2. Loads new model
3. Iterates um_facts in batches, re-encodes content, updates embedding BLOB
4. Updates um_meta with new model info
5. Rebuilds FTS5 index
"""

import argparse
import sqlite3
import sys
import time
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def migrate(db_path: str, batch_size: int = 100, dry_run: bool = False):
    from mnemoria.schema import get_connection
    from mnemoria.embeddings import EmbeddingProvider

    conn = get_connection(db_path)

    # Record pre-migration state
    total = conn.execute(
        "SELECT COUNT(*) FROM um_facts WHERE status IN ('active', 'cold') AND embedding IS NOT NULL"
    ).fetchone()[0]
    print(f"Facts to re-embed: {total}")

    if dry_run:
        print("Dry run -- no changes made.")
        return

    # Save rollback metadata
    conn.execute(
        "INSERT OR REPLACE INTO um_meta (key, value) VALUES ('pre_migration_model', 'all-MiniLM-L6-v2')"
    )
    conn.execute(
        "INSERT OR REPLACE INTO um_meta (key, value) VALUES ('pre_migration_dim', '384')"
    )
    conn.commit()

    # Initialize new embedder
    provider = EmbeddingProvider(model="auto", dimension=256)
    print(f"Embedding backend: {provider.backend_name}, dimension: {provider.dimension}")

    # Batch process
    offset = 0
    updated = 0
    start = time.time()

    while offset < total:
        rows = conn.execute(
            "SELECT id, content, target, type FROM um_facts "
            "WHERE status IN ('active', 'cold') AND embedding IS NOT NULL "
            "ORDER BY created_at "
            "LIMIT ? OFFSET ?",
            (batch_size, offset),
        ).fetchall()

        if not rows:
            break

        # Build embed texts (same logic as store())
        texts = []
        for r in rows:
            text = r["content"]
            target = r["target"]
            if target and target != "general":
                target_words = target.replace(".", " ").replace("_", " ")
                text = f"{target_words} {text}"
            texts.append(text)

        # Batch encode as documents (not queries)
        embeddings = provider.encode_batch(texts, is_query=False)

        # Update each fact
        for r, emb in zip(rows, embeddings):
            emb_blob = np.asarray(emb, dtype=np.float32).tobytes()
            conn.execute(
                "UPDATE um_facts SET embedding = ? WHERE id = ?",
                (emb_blob, r["id"]),
            )

        conn.commit()
        updated += len(rows)
        offset += batch_size
        elapsed = time.time() - start
        rate = updated / elapsed if elapsed > 0 else 0
        print(f"  {updated}/{total} ({rate:.0f} facts/sec)")

    # Record post-migration state
    conn.execute(
        "INSERT OR REPLACE INTO um_meta (key, value) VALUES ('embedding_model', 'nomic-ai/nomic-embed-text-v1.5')"
    )
    conn.execute(
        "INSERT OR REPLACE INTO um_meta (key, value) VALUES ('embedding_dim', '256')"
    )
    conn.execute(
        "INSERT OR REPLACE INTO um_meta (key, value) VALUES ('migration_timestamp', ?)",
        (str(time.time()),),
    )
    conn.commit()

    elapsed = time.time() - start
    print(f"Migration complete: {updated} facts in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Re-embed all Mnemoria facts")
    parser.add_argument("--db", default=None, help="Database path (default: ~/.hermes/mnemoria.db)")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    db_path = args.db
    if db_path is None:
        import os
        hermes_home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
        db_path = os.path.join(hermes_home, "mnemoria.db")

    migrate(db_path, args.batch_size, args.dry_run)


if __name__ == "__main__":
    main()
```

**Migration performance estimate:**
- nomic-embed-text-v1.5 encodes ~200 texts/sec on CPU (M-series Apple Silicon)
- 1000 facts: ~5 seconds
- Hermes production DB (~200 facts): <1 second

### Migration is mandatory, not optional

Old 384d embeddings and new 256d embeddings cannot coexist usefully.
Cosine similarity between a 384d vector and a 256d vector (zero-padded)
will produce garbage scores. The migration must run before the new model
is used for any recall or store operations.

## Backward Compatibility During Migration

### Can we support both models simultaneously?

**No, not meaningfully.** Here is why:

1. **Dimension mismatch**: 384d vs 256d embeddings produce invalid cosine
   similarity when compared. The zero-padding in `links.py:cosine_similarity`
   handles TF-IDF vocabulary growth (where new dimensions carry semantic
   meaning) but not cross-model comparisons (where dimensions have
   completely different semantic axes).

2. **Prefix asymmetry**: Old embeddings were encoded without task prefixes.
   New embeddings use `search_document:` / `search_query:` prefixes. Even
   at the same dimensionality, comparing prefixed vs unprefixed embeddings
   degrades similarity quality.

### Recommended approach: atomic cutover

1. Stop Moonsong sessions (pause hermes cron)
2. Run migration script: `python -m mnemoria.cli.migrate_embeddings`
3. Deploy updated code
4. Resume Moonsong sessions

Total downtime: <30 seconds for a typical Hermes DB.

### Fallback for large deployments

If atomic cutover is not acceptable (e.g., shared DB with multiple
concurrent writers), add a `embedding_version` column to `um_facts`:

```sql
ALTER TABLE um_facts ADD COLUMN embedding_version TEXT DEFAULT 'minilm-384';
```

Then in recall, filter to only compare facts with matching embedding_version.
This is engineering overhead for a scenario Mnemoria does not currently face,
so it is documented here but not implemented.

## Download Size and First-Load Latency

| Metric | all-MiniLM-L6-v2 | nomic-embed-text-v1.5 |
|--------|-------------------|------------------------|
| Download (HuggingFace) | ~46 MB | ~274 MB |
| First-load time (cold, no cache) | ~3 sec | ~8-12 sec |
| First-load time (cached) | <1 sec | ~2 sec |
| Model load into RAM | <1 sec | ~1-2 sec |

**Mitigation:** The model is loaded lazily (`_get_embedder()` in store.py
line 127) and cached globally (`_shared_models` class variable). First
recall/store in a session pays the load cost; subsequent calls are instant.

For Docker/CI, pre-cache the model at image build time:

```dockerfile
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)"
```

## RAM Usage Comparison

| Metric | all-MiniLM-L6-v2 | nomic-embed-text-v1.5 |
|--------|-------------------|------------------------|
| Model parameters | 22.7M | 137M |
| Model RAM (f32) | ~91 MB | ~548 MB |
| Model RAM (f16, if available) | ~46 MB | ~274 MB |
| Per-embedding RAM (in-flight) | 1.5 KB (384d) | 1.0 KB (256d Matryoshka) |
| SQLite BLOB per fact | 1,536 B | 1,024 B |

The model itself uses ~450 MB more RAM. For Hermes agent sessions on
Apple Silicon (16+ GB RAM), this is negligible. For memory-constrained
environments (<4 GB), consider the `snowflake-arctic-embed-s` fallback
from the research doc (66 MB model, 384d, no migration needed).

Sentence-transformers loads models in f32 by default. To reduce RAM:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
model.half()  # f16 -- halves RAM, negligible quality loss
```

This optimization is optional and can be added later as a config flag
(`embedding_precision: str = "f32"`) if RAM becomes a concern.

## Expected Benchmark Improvement per Category

| Category | Current | Expected | Delta | Reasoning |
|----------|---------|----------|-------|-----------|
| semantic_recall | 0.800 | 0.90-0.95 | +0.10-0.15 | Direct embedding quality upgrade. Currently capped by MiniLM discrimination. |
| retrieval_ablation | 0.889 | 0.93-0.96 | +0.04-0.07 | Semantic sub-scores (ra_s01, ra_s02) currently fail due to embedding quality. |
| cross_reference | 0.956 | 0.97+ | +0.01-0.02 | Better multi-fact association. Marginal -- already high. |
| deduplication | 0.750 | 0.80-0.85 | +0.05-0.10 | Better similarity detection widens the gap between true dupes and related facts. |
| topic_shift_recall | 0.833 | 0.87-0.90 | +0.04-0.07 | Better discrimination between topics at recall time. |
| temporal_decay | (passing) | (no change) | 0.00 | Decay is ACT-R based, not embedding dependent. |
| adversarial | (passing) | (no change) | 0.00 | Adversarial detection is regex/heuristic, not embedding dependent. |
| **Overall** | **0.927** | **0.94-0.96** | **+0.01-0.03** | Conservative. Weighted by category weights. |

The biggest win is semantic_recall (0.800 -> ~0.92), which is currently
the single lowest benchmark score and the primary motivation for this
upgrade.

## Rollback Plan

### If quality regresses after migration:

1. **Check um_meta for pre-migration state:**
   ```sql
   SELECT * FROM um_meta WHERE key LIKE 'pre_migration%';
   ```

2. **Re-run migration with old model:**
   ```bash
   python -m mnemoria.cli.migrate_embeddings --model all-MiniLM-L6-v2 --dim 384
   ```
   (Requires adding a `--model` flag to the migration script, trivial extension.)

3. **Revert code changes:**
   - `embeddings.py`: Change default back to `all-MiniLM-L6-v2`, remove prefix logic
   - `config.py`: Change `embedding_dimension` default back to `384`
   - `store.py`: Remove `is_query=` arguments (or leave them, they are harmless)

4. **Re-run benchmarks** to confirm scores return to baseline.

### Partial rollback (keep model, adjust dimension):

If 256d proves too aggressive, bump to 512d:
1. Change `embedding_dimension` default in config.py from 256 to 512
2. Re-run migration script (re-encodes all facts)
3. No other code changes needed

### Automated rollback gate:

Add to CI/CD (not part of this spec, but recommended):
```bash
# Run benchmark suite; fail if overall < 0.92 (current baseline - margin)
python -m mnemoria.benchmark --min-score 0.92
```

## Implementation Order

1. Add `embedding_dimension` to `MnemoriaConfig` (config.py)
2. Refactor `SentenceTransformerEmbedder` (embeddings.py) -- new model, prefixes, Matryoshka
3. Update `EmbeddingProvider` (embeddings.py) -- pass-through dimension + is_query
4. Add `is_query` to store/recall callsites (store.py)
5. Write migration script (mnemoria/cli/migrate_embeddings.py)
6. Run benchmark suite against new model (no migration -- fresh DB)
7. If benchmarks improve: run migration on production DB
8. If benchmarks regress: diagnose, adjust dimension, or roll back

Steps 1-4 are code changes (~50 lines modified across 3 files).
Step 5 is a new file (~80 lines).
Steps 6-8 are operational.

## Files Changed Summary

| File | Change Type | Lines Modified |
|------|-------------|----------------|
| `mnemoria/config.py` | Add field | +3 |
| `mnemoria/embeddings.py` | Refactor SentenceTransformerEmbedder, update EmbeddingProvider | ~60 |
| `mnemoria/store.py` | Add is_query= to 3 encode calls, pass dimension to EmbeddingProvider | +5 |
| `mnemoria/cli/migrate_embeddings.py` | New file | ~80 |
| `mnemoria/retrieval.py` | No changes | 0 |
| `mnemoria/links.py` | No changes | 0 |
| `mnemoria/ingestion.py` | No changes | 0 |
| `mnemoria/schema.py` | No changes | 0 |
| `mnemoria/types.py` | No changes | 0 |

## Dependencies

The `nomic-embed-text-v1.5` model requires:
- `sentence-transformers >= 2.2.0` (already a dependency)
- `einops` (for rotary position embeddings; `pip install einops`)
- `trust_remote_code=True` in SentenceTransformer constructor

Add to requirements/setup:
```
einops>=0.6.0
```
