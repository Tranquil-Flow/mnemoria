# Embedding Model Upgrade Research

Research date: 2026-04-23
Mnemoria version: v0.2.1 (0.927 internal benchmark)
Current model: `all-MiniLM-L6-v2` (semantic_recall capped at 0.800)

## Problem Statement

Mnemoria's semantic_recall benchmark score is 0.800, tied directly to the quality ceiling of the all-MiniLM-L6-v2 embedding model. This is a 2022-era model with known limitations. Upgrading the embedding model is the single highest-leverage change for improving retrieval quality.

The retrieval_ablation semantic sub-scores (ra_s01, ra_s02) also fail due to embedding quality, as noted in the v0.1.0 changelog.

## Current Model: all-MiniLM-L6-v2

| Property | Value |
|----------|-------|
| Parameters | 22.7M |
| Disk size | ~46 MB |
| Dimensions | 384 |
| Context length | 256 tokens (effective; 512 max with truncation) |
| MTEB overall | ~56% (varies by task) |
| Architecture | 6-layer MiniLM distilled from BERT |
| License | Apache 2.0 |
| Year | 2021 |

**Why it's limiting:**
- 256-token context window -- Mnemoria facts are typically short, but compound facts or context windows get truncated
- 2021 training data -- misses recent vocabulary and semantic patterns
- Low MTEB retrieval score compared to 2024-2026 models
- 384 dimensions is adequate but limits discrimination for large stores (1000+ facts)

## Candidate Models

### Tier 1: Drop-In Replacements (same size class)

#### all-MiniLM-L12-v2

| Property | Value |
|----------|-------|
| Parameters | ~33M |
| Disk size | ~50-60 MB |
| Dimensions | 384 |
| Context length | 256-512 tokens |
| MTEB | ~58-59% (estimated, ~2-3% over L6) |
| Architecture | 12-layer MiniLM |
| License | Apache 2.0 |

**Pros:** Direct upgrade from L6. Same API, same dimensions (384), no code changes needed. 12 layers vs 6 gives better semantic discrimination, especially for paraphrases and subtle intent.
**Cons:** ~2x slower inference (12 vs 6 layers). Same limited context window. Same 2021 training vintage. Marginal improvement may not justify the effort.
**Verdict:** Too small an upgrade to be worth the migration cost.

#### E5-small-v2

| Property | Value |
|----------|-------|
| Parameters | 33M |
| Disk size | ~60 MB |
| Dimensions | 384 |
| Context length | 512 tokens |
| MTEB retrieval (NDCG@10) | ~49.04 |
| Architecture | BERT-based with instruction tuning |
| License | MIT |

**Pros:** Better training methodology (contrastive + instruction tuning). Requires query prefix ("query: " / "passage: ") which aligns with Mnemoria's recall vs store paths.
**Cons:** MTEB retrieval score (49.04) is actually lower than snowflake-arctic-embed-s (51.98). No Matryoshka support. Small context window.
**Verdict:** Not recommended. Snowflake-arctic-embed-s is strictly better at the same size.

#### snowflake-arctic-embed-s

| Property | Value |
|----------|-------|
| Parameters | 33M |
| Disk size | ~66 MB |
| Dimensions | 384 |
| Context length | 512 tokens |
| MTEB retrieval (NDCG@10) | **51.98** |
| Architecture | Based on e5-small-unsupervised, multi-stage contrastive |
| License | Apache 2.0 |

**Pros:** Best retrieval quality in the 33M class. Beats bge-small-en-v1.5 (51.68), Cohere-embed-english-light-v3.0 (51.34), text-embedding-3-small (51.08), and e5-small-v2 (49.04). CLS pooling, L2 normalized -- matches Mnemoria's cosine similarity pipeline. Sentence-transformers compatible. Requires query prefix for optimal performance.
**Cons:** 512-token context limit. No Matryoshka dimensionality reduction.
**Verdict:** Strong choice for minimal-change upgrade. Same 384 dimensions as current model.

### Tier 2: Significant Upgrade (medium size)

#### nomic-embed-text-v1.5

| Property | Value |
|----------|-------|
| Parameters | 137M (~100M effective) |
| Disk size | ~274 MB |
| Dimensions | 768 (supports Matryoshka: 512, 256, 128, 64) |
| Context length | **8,192 tokens** |
| MTEB overall | **62.28** (768d), 61.96 (512d), 61.04 (256d) |
| Architecture | nomic-BERT with rotary position embeddings (RoPE) |
| License | Apache 2.0 (fully open: weights, code, training data) |

**Pros:**
- 8K context window -- handles long facts, compound entries, and larger retrieval windows
- Matryoshka support -- can use 256d for storage efficiency with minimal quality loss (61.04 vs 62.28)
- 62.28 MTEB is a massive jump from MiniLM's ~56%
- Fully open (weights + training data + code) -- auditable
- Sentence-transformers compatible
- Requires task prefixes (`search_query:`, `search_document:`) which naturally maps to Mnemoria's recall vs store paths

**Cons:**
- ~6x larger than current model (274 MB vs 46 MB)
- ~3-4x slower inference
- Requires task prefix strings (minor code change)
- 768d embeddings use 2x storage vs 384d (mitigated by Matryoshka truncation to 256d)

**Verdict:** Best balance of quality and practicality. Recommended primary upgrade target.

### Tier 3: Latest Generation (2025-2026)

#### snowflake-arctic-embed-m-v2.0

| Property | Value |
|----------|-------|
| Parameters | ~110M |
| Disk size | ~220 MB |
| Dimensions | 768 |
| Context length | 512 (m-v2.0), 2048+ (m-long) |
| MTEB retrieval | High (exact score varies by task) |
| Architecture | Multi-stage contrastive with MRL |
| License | Apache 2.0 |

Multilingual support (100+ languages). Released December 2024. Strong MTEB retrieval scores. Matryoshka support down to 256d.

#### EmbeddingGemma-300M (Google DeepMind)

| Property | Value |
|----------|-------|
| Parameters | 300M |
| Disk size | ~200 MB (quantized) |
| Dimensions | configurable |
| Context length | long |
| Architecture | Gemma-based |
| License | Gemma license (restrictive for some uses) |

Optimized for on-device. Can run in under 200MB RAM with quantization. 22ms on EdgeTPU. Strong multilingual. But the Gemma license may be problematic.

## Comparison Matrix

| Model | Params | Disk | Dim | Context | MTEB | Matryoshka | Code Change |
|-------|--------|------|-----|---------|------|------------|-------------|
| all-MiniLM-L6-v2 (current) | 23M | 46MB | 384 | 256 | ~56% | No | -- |
| all-MiniLM-L12-v2 | 33M | 55MB | 384 | 256 | ~58% | No | None |
| E5-small-v2 | 33M | 60MB | 384 | 512 | 49.0 | No | Prefix |
| snowflake-arctic-embed-s | 33M | 66MB | 384 | 512 | 52.0 | No | Prefix |
| **nomic-embed-text-v1.5** | **137M** | **274MB** | **768** | **8192** | **62.3** | **Yes** | **Prefix** |
| snowflake-arctic-embed-m-v2 | 110M | 220MB | 768 | 512+ | ~60 | Yes | Prefix |
| EmbeddingGemma-300M | 300M | 200MB* | var | long | ~61 | Yes | Prefix + license |

*quantized

## Recommendation

### Primary: nomic-embed-text-v1.5

**Why:**
1. **Quality jump**: 62.28 vs ~56% MTEB -- this directly unblocks the semantic_recall 0.800 ceiling
2. **8K context**: Future-proofs for longer fact entries and multi-fact retrieval windows
3. **Matryoshka flexibility**: Use 256d for storage (saves 33% vs 384d current) with only 1.24 MTEB points loss, or full 768d for maximum quality
4. **Fully open**: Only major embedding model with public training data -- important for trust/audit
5. **Sentence-transformers compatible**: Mnemoria's `SentenceTransformerEmbedder` class works with a one-line model name change

### Implementation Plan

```python
# In embeddings.py, change SentenceTransformerEmbedder default:
class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        ...
```

**Required code changes:**

1. **Model name** in `SentenceTransformerEmbedder.__init__` (or make configurable via `MnemoriaConfig`)
2. **Task prefixes**: Add `search_query:` prefix for recall queries, `search_document:` prefix for stored facts. This maps naturally to the existing store/recall split.
3. **Dimension handling**: If using Matryoshka truncation, add post-encode normalization:
   ```python
   import torch.nn.functional as F
   embeddings = F.layer_norm(embeddings, (embeddings.shape[1],))
   embeddings = embeddings[:, :matryoshka_dim]
   embeddings = F.normalize(embeddings, p=2, dim=1)
   ```
4. **Migration**: Existing facts need re-embedding. Add a `reindex()` method or migration script that re-encodes all `um_facts` embeddings with the new model. Running both old and new embeddings simultaneously is not feasible (different dimensionality).
5. **Config**: Add `embedding_model` and `embedding_dimension` to `MnemoriaConfig` for flexibility.

### Fallback: snowflake-arctic-embed-s

If the 274 MB download size is prohibitive (e.g., Docker containers, CI, edge deployment), snowflake-arctic-embed-s gives the best quality-per-byte at 66 MB and same 384 dimensions -- no storage migration needed.

### Storage Impact

| Configuration | Bytes per fact | 1000 facts | 10000 facts |
|---|---|---|---|
| Current (384d float32) | 1,536 B | 1.5 MB | 15 MB |
| nomic 768d float32 | 3,072 B | 3.0 MB | 30 MB |
| nomic 256d float32 (Matryoshka) | 1,024 B | 1.0 MB | 10 MB |

The Matryoshka 256d option actually saves storage while improving quality.

## Expected Impact on Benchmarks

| Benchmark Category | Current | Expected (nomic) | Reasoning |
|---|---|---|---|
| semantic_recall | 0.800 | 0.90-0.95 | Direct quality improvement |
| retrieval_ablation | 0.889 | 0.93-0.96 | Semantic sub-scores unblocked |
| cross_reference | 0.956 | 0.97+ | Better multi-fact association |
| deduplication | 0.750 | 0.80-0.85 | Better similarity detection |
| topic_shift_recall | 0.833 | 0.87-0.90 | Better discrimination between topics |
| **Overall** | **0.927** | **0.94-0.96** | Conservative estimate |
