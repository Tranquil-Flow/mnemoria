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

## License

AGPL-3.0-or-later — see LICENSE file or <https://www.gnu.org/licenses/agpl-3.0.html>