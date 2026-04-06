# Mnemoria cognitive memory system — Architecture

## Overview

The Mnemoria cognitive memory system merges three memory architectures into a single
engine: cognitive science (ACT-R), structured knowledge management (typed facts),
and self-optimizing retrieval (Ori-Mnemos). It exceeds all three individually.

**Benchmark:** 97.2% overall (284 scenarios, 11 categories)  
**vs Cognitive:** +0.4% (97.2% vs 96.8%)  
**vs Structured:** +23.6% (97.2% vs 73.6%)

## Architecture

```
┌─────────────────────────────────────────────────┐
│              UNIFIED MEMORY SYSTEM              │
│                                                 │
│  WRITE PATH                                     │
│  ┌──────────────────────────────────────┐       │
│  │ 1. Parse MEMORY_SPEC notation         │       │
│  │ 2. Auto-classify (category+importance)│       │
│  │ 3. Generate embedding                 │       │
│  │ 4. Exact dedup (source_hash)          │       │
│  │ 5. Semantic dedup (embedding+overlap) │       │
│  │ 6. Supersession (type+target match)   │       │
│  │ 7. Contradiction detection            │       │
│  │ 8. Set metabolic decay rate           │       │
│  │ 9. Store in SQLite + FTS5 index       │       │
│  │ 10. Seed Hebbian + bibliographic links│       │
│  │ 11. Gauge pressure check              │       │
│  │ 12. Session reward tracking           │       │
│  └──────────────────────────────────────┘       │
│                                                 │
│  RETRIEVAL PIPELINE                             │
│  ┌──────────────────────────────────────┐       │
│  │ 1. Score candidates (ACT-R + embed)   │       │
│  │    • base_level × metabolic_rate      │       │
│  │    • spreading (semantic + Hebbian)   │       │
│  │    • importance (saturation-aware)    │       │
│  │    • scope boost + adversarial check  │       │
│  │    • revival spike (new connections)  │       │
│  │ 2. RRF BM25 fusion (intent-gated)    │       │
│  │ 3. Dampening (gravity, hub, resolve)  │       │
│  │ 4. IPS debiasing (propensity correct) │       │
│  │ 5. Q-value reranking (UCB-Tuned)      │       │
│  │ 6. Intent-based type boosting         │       │
│  │ 7. Update access stats + Hebbian      │       │
│  │ 8. Session reward signals             │       │
│  └──────────────────────────────────────┘       │
│                                                 │
│  EXPLORATION (PPR)                              │
│  ┌──────────────────────────────────────┐       │
│  │ 1. Seed from recall() results         │       │
│  │ 2. Build adjacency from Hebbian links │       │
│  │ 3. 20-iteration Personalized PageRank │       │
│  │ 4. Discover associatively connected   │       │
│  │ 5. Merge + re-rank by combined score  │       │
│  └──────────────────────────────────────┘       │
│                                                 │
│  LIFECYCLE                                      │
│  ┌──────────────────────────────────────┐       │
│  │ • Scope lifecycle (active→cold→close) │       │
│  │ • Gauge pressure (5-tier cascade)     │       │
│  │ • Tarjan bridge protection            │       │
│  │ • Consolidation (promote/demote/prune)│       │
│  │ • NPMI normalization on links         │       │
│  │ • LinUCB pipeline self-optimization   │       │
│  │ • Auto-ingestion from conversations   │       │
│  └──────────────────────────────────────┘       │
│                                                 │
│  STORAGE (unified SQLite)                       │
│  ┌──────────────────────────────────────┐       │
│  │ um_facts      content, embedding,     │       │
│  │               type, target, scope,    │       │
│  │               activation, q_value,    │       │
│  │               metabolic_rate, ...     │       │
│  │ um_links      Hebbian (NPMI, co-occ)  │       │
│  │ um_scopes     lifecycle management    │       │
│  │ um_access_times  ACT-R history        │       │
│  │ um_qvalues    RL reward tracking      │       │
│  │ um_facts_fts  FTS5 keyword index      │       │
│  └──────────────────────────────────────┘       │
└─────────────────────────────────────────────────┘
```

## Fact Types (MEMORY_SPEC)

| Notation | Type       | Metabolic Rate | Description |
|----------|------------|----------------|-------------|
| C[t]: x  | Constraint | 0.3x (slow)    | Rules, requirements |
| D[t]: x  | Decision   | 0.7x           | Architectural choices |
| V[t]: x  | Value      | 1.0x (normal)  | Concrete values |
| ?[t]: x  | Unknown    | 2.0x (fast)    | Open questions |
| ✓[t]: x  | Done       | 2.5x           | Resolved items |
| ~[t]: x  | Obsolete   | 5.0x (rapid)   | Superseded facts |

## Scoring Formula

```
ACTIVATION = BASE_LEVEL + SPREADING + IMPORTANCE + SCOPE + ADV + REVIVAL

BASE_LEVEL = ln(Σ tᵢ^(-d × metabolic_rate))
SPREADING  = w_semantic × cosine_sim + hebbian_one_hop
IMPORTANCE = w_imp × importance × (1.5 + sim × (2 + |base|))
             × saturation(access_count)  [if count > 5]
SCOPE      = scope_mult × (0.5 + 0.5 × sim)
REVIVAL    = 0.2 × exp(-0.2 × days_since_new_link)
```

Post-scoring: RRF → Dampening → IPS → Q-value (UCB-Tuned) → Intent boost

## Feature Inventory (27 features)

### From Cognitive Memory
1. ACT-R base-level activation (frequency + recency)
2. Spreading activation via Hebbian links
3. Hebbian link formation (GloVe frequency weighting)
4. Ebbinghaus decay on links
5. Turrigiano homeostatic scaling
6. Q-value reranking with reinforcement learning
7. UCB-Tuned variance-aware exploration bonus
8. Dampening: gravity (cosine ghosts)
9. Dampening: hub (P90 degree penalty)
10. Dampening: resolution boost (actionable categories)
11. Contradiction detection (entity overlap + update language)
12. Adversarial content detection (2-tier pattern matching)

### From Structured Memory (PR #3093)
13. Typed facts (C/D/V/?/✓/~) with MEMORY_SPEC notation
14. Metabolic decay per fact type
15. Automatic supersession (same type+target)
16. Scope lifecycle management (active → cold → closed)
17. Gauge pressure management (5-tier cascade)
18. FTS5 keyword search index
19. Hot-facts system prompt injection

### From Ori-Mnemos
20. NPMI-normalized co-occurrence edges
21. Bibliographic coupling bootstrap (cold-start)
22. Tarjan articulation point protection
23. Revival spike on new connections
24. LinUCB contextual bandit (per-stage optimization)
25. Query intent classification (6 types)

### Novel
26. IPS (Inverse Propensity Scoring) debiasing
27. Semantic deduplication (embedding + word overlap)
28. Conversation auto-ingestion (fact extraction from turns)
29. Access frequency saturation (diminishing returns)
30. Session reward tracking (store-after-recall credit)

## File Structure

```
mnemoria/               4,086 lines
├── __init__.py               Package init
├── types.py                  FactType, MemoryFact, ScoredFact, parse_notation
├── config.py                 MnemoriaConfig (50+ parameters)
├── schema.py                 SQLite DDL (6 tables, FTS5, triggers, views)
├── store.py                  MnemoriaStore (main engine)
├── retrieval.py              Scoring pipeline, dampening, Q-value, IPS
├── links.py                  Hebbian (NPMI, GloVe, Ebbinghaus, homeostasis)
├── lifecycle.py              Tarjan bridge detection + protection
├── intent.py                 6-type query intent classifier (30+ patterns)
├── bandit.py                 LinUCB bandits + session reward tracking
├── ingestion.py              Fact extraction + semantic dedup + memorability
├── migrate.py                Migration from MEMORY.md/USER.md/legacy DBs
├── benchmark_adapter.py      BenchmarkableStore wrapper
└── ARCHITECTURE.md           This file

tools/mnemoria_tool.py  795 lines, 8 agent tools
tests/mnemoria/         2,736 lines, 117 tests
```

## Agent Tools

| Tool | Description |
|------|-------------|
| `mcp_umemory_write` | Store facts (plain text or MEMORY_SPEC) |
| `mcp_umemory_recall` | 4-signal fusion semantic recall |
| `mcp_umemory_search` | FTS5 keyword search (fast) |
| `mcp_umemory_reflect` | Type-grouped topic reflection |
| `mcp_umemory_reward` | Apply RL reward signal |
| `mcp_umemory_explore` | PPR multi-hop exploration |
| `mcp_umemory_stats` | Store statistics |
| `mcp_umemory_consolidate` | Lifecycle management |

## Configuration

All parameters in `MnemoriaConfig`. Key defaults:

| Parameter | Default | Description |
|-----------|---------|-------------|
| d | 0.3 | ACT-R decay rate |
| w_semantic | 0.5 | Embedding similarity weight |
| w_importance | 0.4 | Importance weight |
| enable_typed_decay | True | Metabolic decay by type |
| enable_supersession | True | Auto-supersede same type+target |
| enable_pressure | True | Gauge pressure management |
| enable_dampening | True | Gravity + hub + resolution |
| enable_qvalue_reranking | True | Q-value RL blend |
| enable_intent_classification | True | Query intent detection |
| enable_npmi | True | NPMI on co-occurrence edges |
| enable_tarjan_protection | True | Bridge node protection |
| enable_linucb | True | LinUCB pipeline optimization |
| enable_session_rewards | True | Store-after-recall credits |
| enable_ips | True | Inverse propensity scoring |
| enable_rrf_fusion | False | BM25 fusion (hurts temporal) |

## Migration

```python
from mnemoria.migrate import run_migration
run_migration(
    unified_db_path="~/.hermes/mnemoria.db",
    hermes_home="~/.hermes"
)
```

Migrates from: MEMORY.md, USER.md, structured_memory DB, cognitive_memory DB.

## Activation

Add `- mnemoria` to toolsets in hermes config.yaml.
The agent loop hooks in run_agent.py handle injection and tick automatically.
