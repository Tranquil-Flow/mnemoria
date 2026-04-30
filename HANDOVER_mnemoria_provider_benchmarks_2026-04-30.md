# Handover: Mnemoria Provider Benchmark Refresh

Date: 2026-04-30
Scope: Prepare and run fair memory-provider benchmarks for Mnemoria and peer memory plugins.

## Current Repo State

Update after targeted-forgetting implementation:

- Mnemoria now has public targeted erasure APIs:
  - `MnemoriaStore.forget(fact_id, reason=None)`
  - `MnemoriaStore.forget_by_content(pattern, reason=None)`
- Erasure deletes fact rows, FTS entries, links, access history, pending duplicate/reference rows, in-DB qvalues, and external `QValueStore` rows.
- Deletion receipts/logs keep only `fact_id`, SHA-256 content hash, reason, and timestamp; they do not retain deleted content.
- Benchmark `mnemoria` adapter now declares `forgetting=True` and maps benchmark substring deletion to `forget_by_content()`.
- Fresh verification in this container:
  - Mnemoria tests: `95 passed`
  - Benchmark tests: `51 passed`
  - Mnemoria suite `r` privacy smoke: `privacy_forgetting = 1.000` over 10 scenarios

Original handover state before this implementation:

Mnemoria repo:

```bash
cd /Users/evinova/Projects/mnemoria
git branch --show-current
# main
git status --short
```

Current Mnemoria working tree is on `main`, not a scratch branch. The v0.2.2
release edits are present but not committed or pushed:

- `pyproject.toml` version is `0.2.2`
- `mnemoria/__init__.py` reports `__version__ = "0.2.2"`
- `CHANGELOG.md` has a v0.2.2 section
- `benchmarks/README.md` documents the v0.2.2 result
- `benchmarks/results/v0.2.2.json` is untracked and contains the fresh raw result
- `dist/mnemoria-0.2.2.tar.gz` and `dist/mnemoria-0.2.2-py3-none-any.whl` exist locally

Do not assume these changes are committed. Review and commit/tag only after the
user explicitly confirms the release state.

Benchmark harness worktree:

```bash
cd "/Users/evinova/Documents/Playground 2/hermes-agent-benchmark-fairness"
git branch --show-current
# memory-benchmark-fairness
git status --short
```

This is a separate Hermes benchmark worktree created from the clean benchmark
branch. It has local benchmark fairness changes, also not committed or pushed.

## Verification Already Completed

Mnemoria:

```bash
cd /Users/evinova/Projects/mnemoria
python3 -B -m pytest
# 91 passed, 2 warnings

python3 -B -m build
# Successfully built mnemoria-0.2.2.tar.gz and mnemoria-0.2.2-py3-none-any.whl
```

Built wheel verification:

- wheel metadata version: `0.2.2`
- packaged `mnemoria.__version__`: `0.2.2`

Benchmark harness:

```bash
cd "/Users/evinova/Documents/Playground 2/hermes-agent-benchmark-fairness"
python3 -B -m pytest tests/benchmarks -q
# 33 passed, 33 warnings
```

Fresh Mnemoria benchmark run:

```bash
cd "/Users/evinova/Documents/Playground 2/hermes-agent-benchmark-fairness"
python3 -B -m benchmarks \
  --backend mnemoria \
  --suite all \
  --runs 1 \
  --seeds 42 \
  --embedding tfidf \
  --output-dir /tmp/mnemoria-bench-v0.2.2
```

Result:

- overall: `0.9268867924528302` (`0.927`)
- core score: `0.9134199134199135` (`0.913`)
- queries: `424`
- executed categories: `25`
- skipped categories: `0`
- result copied to `/Users/evinova/Projects/mnemoria/benchmarks/results/v0.2.2.json`

Lowest v0.2.2 categories to keep in mind:

- `topic_shift_recall`: `0.750`
- `retrieval_ablation`: `0.778`
- `notation_parsing`: `0.800`
- `semantic_recall`: `0.820`
- `supersession`: `0.867`
- `timestamp_integrity`: `0.875`
- `temporal_decay`: `0.889`

## Benchmark Harness Changes To Preserve

The benchmark worktree contains fixes that should be kept before running the
provider matrix:

- Added first-class `mnemoria` backend adapter at `benchmarks/backends/mnemoria_adapter.py`
- Fixed stale result merging when rerunning partial suites
- Added result metadata:
  - `requested_categories`
  - `executed_categories`
  - `skipped_categories`
  - `score_views`
- Added `score_views.executed`, `score_views.core`, and per-track scores
- Fixed Hindsight reset isolation by rotating `bank_id`
- Fixed `--compare` in `benchmarks.runner` so the comparison backend receives the same suite selection
- Added missing track metadata for all 25 fixture categories
- Fixed integration scoring so capability-skipped scenarios are excluded from totals
- Fixed single-seed significance reporting so it reports `not_enough_runs` instead of scipy `nan`
- Updated benchmark docs and tests

Important: result files now overwrite the target backend JSON in the selected
output directory, with timestamped copies under `history/`. Use a fresh
`--output-dir` for provider runs.

## Provider Client Versions

As of 2026-04-30, local installed versions were stale:

| Provider | Local Version | Latest Seen |
|---|---:|---:|
| mem0ai | 1.0.10 | 2.0.1 |
| honcho-ai | 2.0.1 | 2.1.1 |
| hindsight-client | 0.4.22 | 0.5.6 |
| openviking | 0.3.3 | 0.3.13 |
| byterover-cli | 2.6.0 | 3.10.0 |
| sentence-transformers | 5.3.0 | 5.4.1 |
| mnemoria | 0.2.2 | local release candidate |

Before running the final provider matrix, verify current latest versions again
and record the exact versions in the result bundle. Do not mix old and new
client versions under the same result label.

Recommended version capture:

```bash
python3 -B -c '
import importlib.metadata as md
for name in ["mem0ai", "honcho-ai", "hindsight-client", "openviking", "sentence-transformers", "mnemoria"]:
    try:
        print(name, md.version(name))
    except md.PackageNotFoundError:
        print(name, "not-installed")
'
brv --version
```

Recommended upgrade approach: use an isolated venv in the benchmark worktree so
global Python state is not mutated unnecessarily.

```bash
cd "/Users/evinova/Documents/Playground 2/hermes-agent-benchmark-fairness"
python3 -m venv .venv-bench
source .venv-bench/bin/activate
python -m pip install -U pip
python -m pip install -e /Users/evinova/Projects/mnemoria
python -m pip install -U mem0ai honcho-ai hindsight-client openviking sentence-transformers
python -m pytest tests/benchmarks -q
```

ByteRover is a Node CLI. Current executable was:

```bash
which brv
# /Users/evinova/.nvm/versions/node/v22.21.1/bin/brv
brv --version
# byterover-cli/2.6.0 darwin-arm64 node-v22.21.1
```

Upgrade separately only after confirming the user's preference:

```bash
npm view byterover-cli version
npm install -g byterover-cli@latest
```

## Credential Sources

Do not print secrets in logs, chat, or committed files.

Current shell environment did not have the provider keys loaded:

- `MEM0_API_KEY`: missing
- `RETAINDB_API_KEY`: missing
- `HINDSIGHT_API_KEY`: missing
- `HINDSIGHT_BASE_URL`: missing
- `HONCHO_API_KEY`: missing
- `HONCHO_BASE_URL`: missing
- `OPENVIKING_ENDPOINT`: missing
- `OPENVIKING_API_KEY`: missing
- `BYTEROVER_API_KEY`: missing

Candidate sources found without printing values:

- `~/.honcho/config.json` exists and contains keys `apiKey`, `enabled`, `hosts`
- Honcho host label found: `hermes`
- Hermes sessions contain historical env var names and likely values

Best Hermes session candidate:

```text
/Users/evinova/.hermes/sessions/session_20260406_141314_ad16d2.json
```

It contains names for:

- `MEM0_API_KEY`
- `RETAINDB_API_KEY`
- `HINDSIGHT_API_KEY`
- `HINDSIGHT_BASE_URL`
- `HONCHO_API_KEY`
- `HONCHO_BASE_URL`
- `OPENVIKING_ENDPOINT`
- `OPENVIKING_API_KEY`

Other useful candidates:

- `session_20260407_104427_6c7b24.json`
- `session_20260406_135252_9a4ca8.json`
- `session_20260405_164237_c09e2b.json`
- `session_20260404_172005_a6b279.json`
- `session_20260420_221026_f75e92.json`

Prior safe sighting counts:

| Name | Files With Sightings |
|---|---:|
| `MEM0_API_KEY` | 88 |
| `RETAINDB_API_KEY` | 89 |
| `HINDSIGHT_API_KEY` | 71 |
| `HINDSIGHT_BASE_URL` | 31 |
| `HONCHO_API_KEY` | 123 |
| `HONCHO_BASE_URL` | 89 |
| `OPENVIKING_ENDPOINT` | 42 |
| `OPENVIKING_API_KEY` | 14 |
| `BYTEROVER_API_KEY` | 0 |

Suggested safe workflow:

1. Locate values locally from the candidate files without echoing them.
2. Write exports to a private temp file with `chmod 600`.
3. `source` the file.
4. Verify presence by printing only `set` or `missing`, never values.
5. Delete the temp env file after runs if not needed.

Example presence check:

```bash
for k in MEM0_API_KEY RETAINDB_API_KEY HINDSIGHT_API_KEY HINDSIGHT_BASE_URL HONCHO_API_KEY HONCHO_BASE_URL OPENVIKING_ENDPOINT OPENVIKING_API_KEY; do
  if [ -n "${!k}" ]; then echo "$k=set"; else echo "$k=missing"; fi
done
```

## Provider Readiness Notes

Adapters currently present:

- `baseline-flat`
- `byterover`
- `hindsight`
- `holographic`
- `honcho`
- `mem0`
- `mnemoria`
- `openviking`
- `retaindb`

Isolation semantics:

- Mnemoria: in-memory SQLite, full reset support
- baseline-flat: local reset
- holographic: local temp DB reset
- Mem0: rotates benchmark user ID; no bulk delete
- RetainDB: rotates benchmark user ID
- Hindsight: now rotates benchmark bank ID
- Honcho: deletes session then creates a fresh session
- OpenViking: rotates session and attempts server-side delete
- ByteRover: temp directory reset

Service health:

- OpenViking local endpoint was not running earlier. `curl -s http://127.0.0.1:1933/health` failed with connection refused.
- Honcho has config, but endpoint health was not verified in this pass.
- ByteRover CLI exists, but no `BYTEROVER_API_KEY` was found. The CLI may manage auth separately.

Before full runs, do tiny smoke checks for every external provider:

```bash
python3 -B -m benchmarks --backend mem0 --suite d --runs 1 --seeds 42 --output-dir /tmp/memory-provider-smoke
python3 -B -m benchmarks --backend retaindb --suite d --runs 1 --seeds 42 --output-dir /tmp/memory-provider-smoke
python3 -B -m benchmarks --backend hindsight --suite d --runs 1 --seeds 42 --output-dir /tmp/memory-provider-smoke
python3 -B -m benchmarks --backend honcho --suite d --runs 1 --seeds 42 --output-dir /tmp/memory-provider-smoke
python3 -B -m benchmarks --backend openviking --suite d --runs 1 --seeds 42 --output-dir /tmp/memory-provider-smoke
python3 -B -m benchmarks --backend byterover --suite d --runs 1 --seeds 42 --output-dir /tmp/memory-provider-smoke
```

For Mnemoria, keep `--embedding tfidf` for apples-to-apples comparison with the
v0.2.2 release result unless explicitly running a separate embedding experiment.

## Recommended Benchmark Run Plan

Use a fresh output directory and keep local/service runs separated:

```bash
export BENCH_OUT="/tmp/memory-provider-bench-20260430-latest"
mkdir -p "$BENCH_OUT"
cd "/Users/evinova/Documents/Playground 2/hermes-agent-benchmark-fairness"
```

First, rerun local/reference backends with 5 seeds:

```bash
python3 -B -m benchmarks --backend baseline-flat --suite all --runs 5 --seeds 42 43 44 45 46 --output-dir "$BENCH_OUT"
python3 -B -m benchmarks --backend holographic --suite all --runs 5 --seeds 42 43 44 45 46 --output-dir "$BENCH_OUT"
python3 -B -m benchmarks --backend mnemoria --suite all --runs 5 --seeds 42 43 44 45 46 --embedding tfidf --output-dir "$BENCH_OUT"
```

Then run external providers with one seed first:

```bash
python3 -B -m benchmarks --backend mem0 --suite all --runs 1 --seeds 42 --output-dir "$BENCH_OUT"
python3 -B -m benchmarks --backend retaindb --suite all --runs 1 --seeds 42 --output-dir "$BENCH_OUT"
python3 -B -m benchmarks --backend hindsight --suite all --runs 1 --seeds 42 --output-dir "$BENCH_OUT"
python3 -B -m benchmarks --backend honcho --suite all --runs 1 --seeds 42 --output-dir "$BENCH_OUT"
python3 -B -m benchmarks --backend openviking --suite all --runs 1 --seeds 42 --output-dir "$BENCH_OUT"
python3 -B -m benchmarks --backend byterover --suite all --runs 1 --seeds 42 --output-dir "$BENCH_OUT"
```

If cost, latency, and service limits are acceptable, rerun external providers
with 5 seeds:

```bash
python3 -B -m benchmarks --backend mem0 --suite all --runs 5 --seeds 42 43 44 45 46 --output-dir "$BENCH_OUT"
python3 -B -m benchmarks --backend retaindb --suite all --runs 5 --seeds 42 43 44 45 46 --output-dir "$BENCH_OUT"
python3 -B -m benchmarks --backend hindsight --suite all --runs 5 --seeds 42 43 44 45 46 --output-dir "$BENCH_OUT"
python3 -B -m benchmarks --backend honcho --suite all --runs 5 --seeds 42 43 44 45 46 --output-dir "$BENCH_OUT"
python3 -B -m benchmarks --backend openviking --suite all --runs 5 --seeds 42 43 44 45 46 --output-dir "$BENCH_OUT"
python3 -B -m benchmarks --backend byterover --suite all --runs 5 --seeds 42 43 44 45 46 --output-dir "$BENCH_OUT"
```

## Comparison Rules

Do not rank providers only by raw `mean_score`.

Use these comparison surfaces:

1. `score_views.core.score`
   - Best default cross-provider score.
   - Includes capability-independent core categories.

2. Shared-suite mean
   - Compare only categories executed by every backend in the set.
   - Required when some providers skip temporal or structured categories.

3. Track scores
   - `score_views.tracks.core`
   - `score_views.tracks.temporal`
   - `score_views.tracks.structured`
   - `score_views.tracks.lifecycle`

4. Operational metrics
   - wall time
   - recall tokens/query
   - tokens/correct
   - failure modes and skipped categories

For pairwise result-file comparison:

```bash
python3 -B -m benchmarks --compare "$BENCH_OUT/mnemoria.json" "$BENCH_OUT/mem0.json"
```

For runner-level smoke comparison by backend:

```bash
python3 -B -m benchmarks.runner \
  --backend baseline-flat \
  --suite d \
  --runs 1 \
  --seeds 42 \
  --compare mnemoria \
  --embedding tfidf \
  --output-dir /tmp/memory-bench-compare-smoke
```

The runner-level `--compare` path was smoke-tested after fixes. With one seed it
now reports `Test: not_enough_runs` instead of pretending significance.

## Optional LLM Judge Pass

Default heuristic judging is deterministic and cheap but lexically biased. For
publication-quality claims, rerun semantic-sensitive suites with an LLM judge
and report heuristic and LLM results separately.

Suggested suites/categories to validate:

- Suite A: semantic recall, contradictions, cross-reference
- Suite J: topic shift recall
- Suite M: format sensitivity
- Suite N: retrieval ablation
- Suite D: adversarial

Do not mix LLM-judge and heuristic-judge numbers in one aggregate without
labeling them clearly.

## Expected Deliverables For Next Agent

1. Confirm v0.2.2 release changes are correct on Mnemoria `main`.
2. Commit/tag/publish only if the user explicitly asks.
3. Preserve or commit benchmark harness fairness fixes before final provider runs.
4. Upgrade or isolate provider clients, recording exact versions.
5. Load provider secrets safely without printing them.
6. Run provider smoke tests.
7. Run the full provider matrix into a fresh output directory.
8. Produce a comparison report with:
   - raw overall score
   - core score
   - shared-suite mean
   - per-track scores
   - skipped categories
   - wall time and token/cost efficiency
   - notable failure categories
9. Only then decide which result JSON files are worth committing into Mnemoria or Hermes benchmark docs.

## Cautions

- Do not print or commit API keys.
- Do not run final comparisons from the old broken worktree at `/Users/evinova/Projects/hermes-agent-benchmark-clean`; its git metadata points to missing worktree state.
- Do not use the dirty main Hermes checkout for benchmark changes unless asked.
- Do not overwrite checked-in benchmark result bundles until the run plan is final.
- Do not claim statistical significance from one seed.
- Do not compare Mnemoria's all-category score against a service backend that skipped whole capability tracks without also reporting core/shared-suite scores.
