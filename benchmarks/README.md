# Mnemoria Benchmarks

Benchmark results from the [hermes-agent benchmark suite](https://github.com/NousResearch/hermes-agent).

## Running

Benchmarks run against the standalone `mnemoria` package from a checkout of the
`hermes-agent` benchmark suite:

```bash
cd /path/to/hermes-agent-benchmark-suite
pip install -e ~/Projects/mnemoria   # install local source
python -m benchmarks --backend mnemoria --suite all --runs 1 --seeds 42 --embedding tfidf --output-dir /tmp/bench-out
```

For a multi-seed run (more reliable statistics):

```bash
python -m benchmarks --backend mnemoria --suite all --runs 5 --seeds 42 43 44 45 46 --embedding tfidf --output-dir /tmp/bench-out
```

## Results History

| Version | Overall | Core | Categories | Queries | Seeds | Date |
|---------|---------|------|------------|---------|-------|------|
| v0.1.0  | 0.910   | —    | —          | —       | —     | 2026-04-07 |
| v0.2.0  | 0.910   | —    | —          | —       | —     | 2026-04-08 |
| v0.2.1  | **0.927** | —  | 25         | 424     | 42    | 2026-04-11 |
| v0.2.2  | **0.927** | **0.913** | 25 | 424 | 42 | 2026-04-30 |
| v0.2.3  | **0.907** | **0.873** | 30 | 483 | 42 | 2026-04-30 |

Note: v0.2.3 uses the expanded 30-category benchmark suite with P-T categories
(abstention, preference memory, privacy forgetting, multi-hop exploration, and
long conversation). Compare v0.2.3 against earlier releases by matching category
subsets, not by raw overall score alone.

## v0.2.3 Category Breakdown

| Category | Score |
|---|---|
| abstention | 1.000 |
| capacity_stress | 1.000 |
| compression_survival | 1.000 |
| consolidation | 1.000 |
| conversation_memory | 1.000 |
| deduplication | 1.000 |
| importance_filtering | 1.000 |
| integration | 1.000 |
| long_conversation | 1.000 |
| multi_hop_exploration | 1.000 |
| privacy_forgetting | 1.000 |
| qlearning | 1.000 |
| scale | 1.000 |
| scope_lifecycle | 1.000 |
| scopes | 1.000 |
| typed_decay | 1.000 |
| contradictions | 0.950 |
| adversarial | 0.933 |
| cross_reference | 0.933 |
| compression | 0.900 |
| format_sensitivity | 0.900 |
| notation_parsing | 0.900 |
| retrieval_ablation | 0.889 |
| temporal_decay | 0.889 |
| delegation_memory | 0.875 |
| timestamp_integrity | 0.875 |
| preference_memory | 0.867 |
| supersession | 0.867 |
| topic_shift_recall | 0.750 |
| semantic_recall | 0.560 |

Raw result JSON: `results/v0.2.3.json`

## v0.2.2 Category Breakdown

| Category | Score |
|---|---|
| capacity_stress | 1.000 |
| compression_survival | 1.000 |
| consolidation | 1.000 |
| conversation_memory | 1.000 |
| deduplication | 1.000 |
| delegation_memory | 1.000 |
| importance_filtering | 1.000 |
| integration | 1.000 |
| qlearning | 1.000 |
| scale | 1.000 |
| scope_lifecycle | 1.000 |
| scopes | 1.000 |
| typed_decay | 1.000 |
| contradictions | 0.950 |
| adversarial | 0.933 |
| cross_reference | 0.933 |
| compression | 0.900 |
| format_sensitivity | 0.900 |
| temporal_decay | 0.889 |
| timestamp_integrity | 0.875 |
| supersession | 0.867 |
| semantic_recall | 0.820 |
| notation_parsing | 0.800 |
| retrieval_ablation | 0.778 |
| topic_shift_recall | 0.750 |

Raw result JSON: `results/v0.2.2.json`
