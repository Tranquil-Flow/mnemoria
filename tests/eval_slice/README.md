# Curated eval slice

Fast verification harness for the v0.3 retrieval-quality work. Runs a fixed
30-question slice in ~5 minutes so we can measure each change's impact
between commits, instead of relying on the full ~1.5h sweep.

## What's in the slice

Questions are chosen deterministically from the 2026-04-30 baseline run, so
diffs across commits are apples-to-apples.

| Benchmark    | Question type        | N  | Why                                                      |
| ------------ | -------------------- | -- | -------------------------------------------------------- |
| LoCoMo       | `multi_hop`          | 10 | Currently 0/91 — pure failure surface for B1/B5/B2       |
| LoCoMo       | `single_hop`         | 10 | Tone-vs-entity failure mode (10/75 baseline)             |
| LongMemEval  | `multi-session`      | 5  | Cross-session synthesis weakness (17/133 baseline)       |
| LongMemEval  | `temporal-reasoning` | 5  | Temporal targeting weakness (30/133 baseline)            |

The question IDs live in [`curated_ids.json`](./curated_ids.json) and are
committed verbatim — do not regenerate without explicit reason.

## Per-change verification protocol

Hard rule: every plan item must show a measured improvement on this slice
*before* it can be committed. Tuning a non-improvement into a positive
number is overfitting; if the change does not help on its first honest run,
either redesign or drop it from the milestone.

```bash
# 1. Record the baseline (only once, then leave alone)
python -m tests.eval_slice.run_slice --label baseline --baseline

# 2. Make the change.

# 3. Re-run with a label naming the change.
python -m tests.eval_slice.run_slice --label A1-rrf-on

# 4. Read the printed Δ vs baseline.json. Commit only if:
#       - target metric improved (positive Δ), AND
#       - no regression metric dropped meaningfully
```

Result JSON files are written to `results/` and overwritten when the same
label is reused. `results/latest.json` always holds the most recent run for
ad-hoc diffs.

## Configuration

The harness imports from the
`hermes-agent-benchmark-fairness` repo (judge, dataset loaders, metric
suite). It expects the repo at `~/Projects/hermes-agent-benchmark-fairness`;
override with `FAIRNESS_ROOT=/path/to/repo`.

Embedding model and profile are pass-through to `MnemoriaConfig`:

```bash
python -m tests.eval_slice.run_slice --embedding sentence-transformers --label A4-st
```

Skip benchmarks for partial runs (e.g. when a change only affects one):

```bash
python -m tests.eval_slice.run_slice --label B1-multihop --skip longmemeval
```
