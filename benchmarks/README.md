# Mnemoria Benchmarks

Benchmark results from the [hermes-agent benchmark suite](https://github.com/NousResearch/hermes-agent).

## Running

Benchmarks run against the standalone `mnemoria` package from the `hermes-agent-benchmark-clean` repo:

```bash
cd ~/Projects/hermes-agent-benchmark-clean
pip install -e ~/Projects/mnemoria   # install local source
python -m benchmarks --backend mnemoria --suite all --runs 1 --seeds 42 --output-dir /tmp/bench-out
```

For a multi-seed run (more reliable statistics):

```bash
python -m benchmarks --backend mnemoria --suite all --runs 5 --seeds 42 43 44 45 46 --output-dir /tmp/bench-out
```

## Results History

| Version | Overall | Queries | Seeds | Date |
|---------|---------|---------|-------|------|
| v0.1.0  | 0.910   | —       | —     | 2026-04-07 |
| v0.2.0  | 0.910   | —       | —     | 2026-04-08 |
| v0.2.1  | **0.927** | 424   | 42    | 2026-04-11 |

## v0.2.1 Category Breakdown

| Category | Score |
|---|---|
| capacity_stress | 1.000 |
| compression | 1.000 |
| compression_survival | 1.000 |
| consolidation | 1.000 |
| conversation_memory | 1.000 |
| delegation_memory | 1.000 |
| integration | 1.000 |
| qlearning | 1.000 |
| scale | 1.000 |
| scope_lifecycle | 1.000 |
| scopes | 1.000 |
| timestamp_integrity | 1.000 |
| typed_decay | 1.000 |
| contradictions | 0.950 |
| cross_reference | 0.956 |
| temporal_decay | 0.933 |
| importance_filtering | 0.925 |
| retrieval_ablation | 0.889 |
| format_sensitivity | 0.900 |
| adversarial | 0.867 |
| supersession | 0.867 |
| topic_shift_recall | 0.833 |
| notation_parsing | 0.800 |
| semantic_recall | 0.800 |
| deduplication | 0.750 |

Raw result JSON: `results/v0.2.1.json`
