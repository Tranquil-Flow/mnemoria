# Mnemoria

Cognitive memory system for AI agents — ACT-R activation, typed facts, Hebbian links, RL reranking.

## Installation

```bash
pip install mnemoria
```

## Quick Start

```python
from mnemoria.store import MnemoriaStore
from mnemoria.config import MnemoriaConfig

store = MnemoriaStore(MnemoriaConfig.balanced())
store.store("V[api.url]: https://example.com")
results = store.recall("What is the API URL?")
```

For better semantic recall on a real machine, install the embeddings extra:

```bash
pip install 'mnemoria[embeddings]'
```

Note: in constrained containers, TF-IDF fallback is expected and benchmark results may be lower than on a local machine with real embeddings.

## Continuous Extraction (v0.2.0)

Mnemoria v0.2.0 adds **continuous rule-based extraction**: observers watch tool outputs and user messages, automatically storing facts about what they see — without requiring an explicit `store()` call.

### How it works

Facts extracted from tool outputs and user statements go into the `um_pending` table first, then get promoted to `um_facts` by the **promoter**:

| Source | Promotion rule |
|--------|---------------|
| `observed` (tool outputs) | Promotes immediately |
| `user_stated` (explicit statements) | Promotes immediately |
| `agent_inference` | Waits 10 min after last activity in session |

Within the same `(type, target, session_id)` group, the **latest fact wins** — older provisionals are retracted automatically. This handles contradictions without embeddings.

### Extraction mode

Configure via `HERMES_MEMORY_MNEMORIA_EXTRACT_MODE` (in `plugin.yaml`):

- **`observed_only`** (default): Extracts from tool outputs and user statements. Recommended.
- **`off`**: No extraction. Mnemoria only stores what you explicitly `store()`.
- **`full`**: Also includes `agent_inference` facts (LLM-extracted — not implemented yet in v0.2.0).

### Inspecting pending facts

```bash
# CLI inspector
python -m mnemoria.scripts.pending --db ~/.hermes/mnemoria.db

# Show only user-stated facts
python -m mnemoria.scripts.pending --db ~/.hermes/mnemoria.db --source user_stated

# Promote or retract a pending fact
python -m mnemoria.scripts.pending --db ~/.hermes/mnemoria.db --promote <pending_id>
python -m mnemoria.scripts.pending --db ~/.hermes/mnemoria.db --retract <pending_id>
```

### Backfilling existing facts

If you upgraded from v0.1.0, existing facts may have `target='general'`. Use the backfill script:

```bash
python -m mnemoria.scripts.retag_facts --db ~/.hermes/mnemoria.db
```

### The epistemic model

Mnemoria v0.2.0 uses **source-dependent promotion latency**:

- **Observed tool outputs** are trusted signals — a failed `pytest` run is an unambiguous fact.
- **User statements** are explicit — "I prefer X" is a preference the user owns.
- **Agent inferences** are lower-confidence — the LLM's deduction could be wrong, so it waits for session TTL before becoming a confirmed fact.

This means `observed` and `user_stated` facts appear in recall almost immediately; `agent_inference` facts take 10+ minutes and can be manually promoted or retracted via the MCP tools before then.

## Features

- **ACT-R Activation** — Frequency + recency based activation scoring
- **Typed Facts** — MEMORY_SPEC notation (C/D/V/?/✓/~) with metabolic decay rates
- **Hebbian Links** — NPMI-normalized co-occurrence edges with Ebbinghaus decay
- **RL Reranking** — Q-value UCB-Tuned exploration bonus
- **Self-Optimizing Pipeline** — LinUCB bandits per retrieval stage
- **Scope Lifecycle** — active → cold → closed with gauge pressure management
- **Contradiction Detection** — Entity overlap + update language pattern matching
- **IPS Debiasing** — Inverse propensity scoring to counteract popularity bias
- **PPR Exploration** — Personalized PageRank multi-hop discovery

## Migration

If you are moving from Honcho or another external memory provider, see:
- `MIGRATING_FROM_HONCHO.md`

## Acknowledgements

If you want the provenance of ideas and inspirations, see:
- `ACKNOWLEDGEMENTS.md`

## License

AGPL-3.0-or-later — see LICENSE file or <https://www.gnu.org/licenses/agpl-3.0.html>