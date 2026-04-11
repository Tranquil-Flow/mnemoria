# Mnemoria

Cognitive memory for AI agents.

Most agent memory systems are key-value stores with vector search bolted on. Mnemoria models memory the way cognitive science models human recall — facts compete for activation based on frequency, recency, importance, and associative context. The result is memory that naturally strengthens what matters, forgets what doesn't, and discovers connections you didn't explicitly store.

## Features

- **ACT-R activation scoring** — memories strengthen with use and decay with time, following the power-law dynamics of human recall
- **Typed facts with metabolic decay** — constraints decay slowly, open questions decay fast, using MEMORY_SPEC notation (C/D/V/?/~/+)
- **Hebbian links** — co-accessed memories form associative connections that strengthen with repetition and decay via Ebbinghaus forgetting curves
- **RL reranking** — reinforcement learning (UCB-Tuned) reranks retrieval results based on which memories actually proved useful
- **Self-optimizing pipeline** — LinUCB contextual bandits tune retrieval stage weights per query, learning which signals matter for different query types
- **Continuous extraction** — rule-based observers watch tool outputs and user messages, automatically extracting facts with false-memory prevention via a pending/promotion pipeline
- **Personalized PageRank exploration** — discover associatively connected memories that pure similarity search would miss
- **IPS debiasing** — inverse propensity scoring counteracts popularity bias so frequently-accessed memories don't drown out rare but relevant ones

## Research Foundations

| Technique | Foundation |
|-----------|-----------|
| Activation scoring | Anderson & Lebiere (1998). *The Atomic Components of Thought*. Lawrence Erlbaum Associates. |
| Hebbian learning | Hebb (1949). *The Organization of Behavior: A Neuropsychological Theory*. Wiley. |
| Forgetting curves | Ebbinghaus (1885/1913). *Memory: A Contribution to Experimental Psychology*. Teachers College, Columbia University. |
| NPMI normalization | Bouma (2009). "Normalized (Pointwise) Mutual Information in Collocation Extraction." *Proc. GSCL*. |
| Variance-aware exploration | Auer, Cesa-Bianchi & Fischer (2002). "Finite-time Analysis of the Multiarmed Bandit Problem." *Machine Learning*, 47(2-3), 235-256. |
| Contextual bandits | Li, Chu, Langford & Schapire (2010). "A Contextual-Bandit Approach to Personalized News Article Recommendation." *WWW '10*, 661-670. |
| Propensity debiasing | Horvitz & Thompson (1952). *JASA*, 47(260), 663-685; Schnabel et al. (2016). "Recommendations as Treatments." *ICML*. |
| Graph exploration | Page, Brin, Motwani & Winograd (1998). "The PageRank Citation Ranking." Stanford InfoLab. |
| Bridge detection | Tarjan (1972). "Depth-First Search and Linear Graph Algorithms." *SIAM J. Computing*, 1(2), 146-160. |
| Homeostatic scaling | Turrigiano (2008). "The Self-Tuning Neuron: Synaptic Scaling of Excitatory Synapses." *Cell*, 135(3), 422-435. |

## Quick Start

```bash
pip install mnemoria
```

```python
from mnemoria.store import MnemoriaStore
from mnemoria.config import MnemoriaConfig

store = MnemoriaStore(MnemoriaConfig.balanced())
store.store("C[auth]: JWT tokens expire after 7 days")
store.store("V[api.url]: https://api.example.com/v2")

results = store.recall("What is the API URL?")
for r in results:
    print(f"[{r.score:.2f}] {r.fact.content}")
```

For better semantic recall, install the embeddings extra:

```bash
pip install 'mnemoria[embeddings]'
```

Without embeddings, Mnemoria uses TF-IDF fallback — functional but lower quality for semantic queries.

## How It Works

**Write path:** Parse fact notation, auto-classify type and importance, generate embedding, check for duplicates and contradictions, apply supersession rules, store to SQLite with FTS5 index, seed Hebbian and bibliographic links.

**Retrieval pipeline:** Score candidates using ACT-R base-level activation + semantic spreading + importance weighting, fuse with BM25 keyword search, apply dampening (gravity, hub, resolution), correct for popularity bias via IPS, rerank with Q-value UCB-Tuned, and boost by query intent classification.

**Exploration:** Seed Personalized PageRank from recall results, walk the Hebbian link graph to discover associatively connected facts that pure similarity would miss.

**Continuous extraction:** Rule-based observers watch tool outputs and user statements. Extracted facts enter a pending table and promote to confirmed facts based on source reliability — tool outputs promote immediately, agent inferences wait for session TTL.

## Migration

Moving from Honcho or another memory provider? See [MIGRATING_FROM_HONCHO.md](MIGRATING_FROM_HONCHO.md).

## Links

- [CHANGELOG](CHANGELOG.md)
- [CONTRIBUTING](CONTRIBUTING.md)
- [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS.md)
- [Architecture](mnemoria/ARCHITECTURE.md)

## License

AGPL-3.0-or-later — see [LICENSE](LICENSE) or <https://www.gnu.org/licenses/agpl-3.0.html>
