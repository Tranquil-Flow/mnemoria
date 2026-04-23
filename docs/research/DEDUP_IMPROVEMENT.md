# Deduplication Improvement Research

Research date: 2026-04-23
Mnemoria version: v0.2.1 (0.927 overall, **deduplication: 0.750** -- weakest category)

## Problem Statement

Deduplication is Mnemoria's weakest benchmark category at 0.750 (6/8 correct). The system currently has two dedup layers:

1. **Exact hash**: `source_hash` (SHA-256 of content) catches identical strings
2. **Semantic dedup**: `find_near_duplicates()` in `ingestion.py` catches near-duplicates via embedding cosine similarity >= 0.95 AND word overlap >= 0.40

Both layers run at write time in `store()`. The problem is that real-world duplicates often share meaning but differ substantially in surface form:

- "The API endpoint is https://api.example.com/v2" vs "Our REST API lives at api.example.com v2"
- "Database uses PostgreSQL 15" vs "We run Postgres 15 for the main DB"
- "Authentication requires MFA" vs "Multi-factor auth is mandatory"

The current 0.95 cosine threshold is conservative -- lowering it increases false positive merges, which is worse than duplicate storage. This is the fundamental tension in dedup: precision vs recall.

## Current Dedup Architecture Analysis

### Layer 1: Source Hash (store.py L225-230)

```python
source_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
existing = conn.execute(
    "SELECT id FROM um_facts WHERE source_hash = ? AND status = 'active'",
    (source_hash,)
).fetchone()
```

Catches only byte-identical content. Good for preventing re-ingestion of the same observer output, but useless for semantic duplicates.

### Layer 2: Semantic Dedup (store.py L235-241, ingestion.py L172-213)

```python
dupes = find_near_duplicates(self._conn, content, embedding, threshold=0.95)
```

The `find_near_duplicates` function:
- Scans ALL active facts with embeddings (O(n) per store)
- Requires cosine >= 0.95 AND word Jaccard >= 0.40
- Returns early on first match

**Why 0.95 is too conservative**: With `all-MiniLM-L6-v2`, paraphrases of the same fact often score 0.80-0.92 in cosine similarity. The model's discrimination between "same meaning, different words" and "related but different meaning" is poor at its 384-dimensional resolution. Lowering the threshold to 0.85 without additional signals would cause false merges between related-but-distinct facts (e.g., "PostgreSQL 15 for auth DB" vs "PostgreSQL 15 for analytics DB").

### What the Benchmark Tests

The deduplication category has 8 tests. Based on the retrieval metrics (recall@1: 0.75, recall@3: 1.0), the failures occur when duplicate facts dilute the top-1 ranking -- the correct fact exists but is pushed to position 2-3 by a duplicate that shouldn't exist.

## Research: Entity Linking for Memory Dedup

### Core Idea

Entity linking resolves surface-form mentions to canonical entities: "Postgres", "PostgreSQL", "pg" all resolve to the same entity. Two facts sharing a canonical entity + overlapping predicate are likely duplicates even at cosine 0.80.

### Approaches

#### Lightweight Entity Resolution (recommended for Mnemoria)

Build a small, domain-adaptive entity table from stored facts:

```sql
CREATE TABLE um_entities (
    id TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL,  -- "postgresql"
    aliases TEXT,                   -- JSON: ["postgres", "pg", "psql"]
    entity_type TEXT,               -- "technology", "person", "service"
    first_seen REAL,
    fact_count INTEGER DEFAULT 0
);

CREATE TABLE um_fact_entities (
    fact_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    role TEXT,  -- "subject", "object", "modifier"
    PRIMARY KEY (fact_id, entity_id)
);
```

At store time:
1. Extract entities from new fact content (regex + NER patterns)
2. Resolve to canonical entities via alias lookup
3. Find existing facts sharing the same canonical entity + same fact type + same target
4. If found, apply a relaxed cosine threshold (0.80 instead of 0.95)

This is what Zep/Graphiti does with Neo4j entity nodes, but adapted for Mnemoria's SQLite architecture.

#### Entity Extraction Patterns

Mnemoria already has `_extract_key_terms()` in `retrieval.py` that does a basic version of this. Extending it:

```python
ENTITY_PATTERNS = {
    'technology': r'\b(?:PostgreSQL?|MySQL|MongoDB|Redis|Docker|K8s|Kubernetes|React|Next\.js|Node|Python|Rust|Go|Java|TypeScript|JavaScript|GraphQL|gRPC)\b',
    'cloud': r'\b(?:AWS|GCP|Azure|Cloudflare|Vercel|Heroku|DigitalOcean|Lambda|S3|EC2|RDS|ECS|EKS)\b',
    'protocol': r'\b(?:HTTP|HTTPS|gRPC|WebSocket|MQTT|AMQP|REST|GraphQL|OAuth|JWT|SAML|OpenID)\b',
    'tool': r'\b(?:Jenkins|GitHub|GitLab|Terraform|Ansible|Prometheus|Grafana|Sentry|DataDog|Splunk|ELK)\b',
}

ALIAS_MAP = {
    'postgres': 'postgresql', 'pg': 'postgresql', 'psql': 'postgresql',
    'mongo': 'mongodb', 'k8s': 'kubernetes',
    'js': 'javascript', 'ts': 'typescript', 'py': 'python',
    'node': 'nodejs', 'react': 'reactjs',
    'gha': 'github_actions', 'gh actions': 'github_actions',
}
```

#### Expected Impact

Entity-aware dedup would catch cases like:
- "PostgreSQL 15 for the auth database" vs "Postgres 15 handles authentication storage" (same entity `postgresql` + same target `auth`)
- "We deploy to AWS us-east-1" vs "Production runs in AWS us-east-1 region" (same entities `aws` + `us-east-1`)

Estimated dedup score improvement: 0.750 -> 0.85-0.90.

### Academic Reference: Entity Linking Literature

- **BLINK** (Facebook Research, 2020): Bi-encoder entity linking using dense retrieval. Maps mentions to Wikipedia entities. Too heavy for Mnemoria but the bi-encoder pattern (encode mention + encode candidate, then compare) is reusable.
- **mGENRE** (2022): Multilingual generative entity linking. Uses seq2seq to generate canonical entity names from context. Interesting but requires a language model.
- **ReFinED** (Amazon, 2022): Fast fine-grained entity linking combining entity type prediction + entity disambiguation. Lightweight enough for edge deployment.

For Mnemoria, a rule-based alias table (no ML model) is sufficient -- the domain is technical facts, not open-domain text.

## Research: LLM-Based Semantic Deduplication

### Core Idea

Use the agent's own LLM (already available via Hermes provider) to compare candidate duplicates at write time. The LLM can understand that "The auth service uses JWT tokens" and "Authentication is handled via JSON Web Tokens" are the same fact.

### How Mem0 Does It

Mem0's dedup pipeline (from their ECAI 2025 paper and open-source code):

1. **Candidate retrieval**: On each new memory, retrieve top-k existing memories by embedding similarity
2. **LLM comparison**: For each candidate above a similarity threshold, prompt the LLM:
   ```
   Compare these two memories. Are they expressing the same information?
   Memory 1: {existing}
   Memory 2: {new}
   Respond with: DUPLICATE, UPDATE, or DIFFERENT
   ```
3. **Action based on LLM verdict**:
   - DUPLICATE: Discard new memory, optionally merge metadata
   - UPDATE: Replace old memory with new (supersession)
   - DIFFERENT: Store both

4. **Batch processing**: Mem0 processes candidates in batches of 5-10 to reduce LLM calls

### Mem0 Code Architecture (from github.com/mem0ai/mem0)

```python
# Simplified from mem0/memory/main.py
class Memory:
    def add(self, data, ...):
        # 1. Extract facts from input
        new_memories = self._extract_memories(data)
        
        # 2. For each new memory, search existing
        for memory in new_memories:
            existing = self.vector_store.search(memory.embedding, top_k=5)
            
            # 3. LLM-based dedup check
            for candidate in existing:
                if candidate.score > 0.7:  # Lower threshold than Mnemoria's 0.95
                    verdict = self.llm.compare_memories(memory, candidate)
                    if verdict == "DUPLICATE":
                        return candidate.id  # Skip storage
                    elif verdict == "UPDATE":
                        self._update_memory(candidate.id, memory)
                        return candidate.id
        
        # 4. No duplicate found, store new
        return self.vector_store.insert(memory)
```

### Adaptation for Mnemoria

Mnemoria has `contradiction_llm_model` config already (config.py L121) but it's only used for contradiction detection, not dedup. The same LLM channel could serve dedup:

```python
# Proposed addition to config.py
dedup_llm_model: Optional[str] = None
"""Model for LLM-assisted deduplication. None = embedding-only dedup."""

dedup_llm_threshold: float = 0.80
"""Minimum cosine similarity to trigger LLM dedup check (lower than pure-embedding threshold)."""

dedup_llm_max_candidates: int = 3
"""Maximum number of candidates to check via LLM per store() call."""
```

**Critical design decision**: LLM dedup adds latency to every `store()` call. For Mnemoria's use case (agent memory, not user-facing API), 200-500ms per store is acceptable. But for high-throughput observer pipelines (PytestObserver, GitObserver), LLM dedup should be skippable.

**Proposed implementation**:
1. At `store()` time, after embedding generation, retrieve top-3 candidates at cosine >= 0.80
2. If any candidates found AND `dedup_llm_model` is set, call the LLM with a structured comparison prompt
3. Parse LLM response: SAME (merge), UPDATE (supersede), DIFFERENT (store both)
4. Fall back to current embedding-only dedup if LLM is unavailable or errors

**Prompt design** (adapted from Mem0, made Mnemoria-specific):

```
You are comparing two memory facts to check for duplication.

Existing fact: {existing.content}
  Type: {existing.fact_type}, Target: {existing.target}

New fact: {new_content}
  Type: {new_fact_type}, Target: {new_target}

Are these facts expressing the same information?

Rules:
- SAME: identical meaning, even if worded differently
- UPDATE: new fact is a newer version of existing (values changed, status changed)
- DIFFERENT: related but distinct facts (e.g., different aspects of the same system)

Respond with exactly one word: SAME, UPDATE, or DIFFERENT.
```

### Cost Analysis

Assuming a 14B parameter local model (qwen3:14b on M4 Pro):
- Prompt: ~150 tokens, response: 1 token
- Latency: ~300ms on M4 Pro
- Per store() with 3 candidates: ~900ms worst case
- Per 100 facts/day: ~90 seconds total

Acceptable for Mnemoria's use case. For cloud models (GPT-4o-mini, Claude Haiku), cost is < $0.001 per dedup check.

## Research: Locality-Sensitive Hashing (LSH)

### Core Idea

LSH provides O(1) approximate nearest-neighbor lookup instead of O(n) linear scan. For dedup, this means finding candidate duplicates without scanning every fact in the store.

### SimHash for Text Dedup

SimHash (Charikar, 2002) produces a fixed-size fingerprint where similar documents have similar hashes. The Hamming distance between hashes approximates cosine distance.

```python
import hashlib
import numpy as np

def simhash(text: str, bits: int = 64) -> int:
    """Compute SimHash fingerprint for text."""
    tokens = text.lower().split()
    v = np.zeros(bits)
    for token in tokens:
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        for i in range(bits):
            if h & (1 << i):
                v[i] += 1
            else:
                v[i] -= 1
    fingerprint = 0
    for i in range(bits):
        if v[i] >= 0:
            fingerprint |= (1 << i)
    return fingerprint

def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count('1')

# Two facts are near-duplicates if hamming_distance <= threshold (e.g., 3)
```

### MinHash + LSH for Jaccard Similarity

MinHash approximates Jaccard similarity using random hash functions. Combined with LSH banding, it enables sub-linear lookup:

```python
from datasketch import MinHash, MinHashLSH

# At initialization
lsh = MinHashLSH(threshold=0.5, num_perm=128)

# At store time
def store_with_lsh(content, fact_id):
    mh = MinHash(num_perm=128)
    for word in content.lower().split():
        mh.update(word.encode())
    
    # Query for near-duplicates
    candidates = lsh.query(mh)
    if candidates:
        # Check candidates more carefully
        ...
    
    # Insert into LSH index
    lsh.insert(fact_id, mh)
```

### Embedding-Space LSH

For Mnemoria's vector-based dedup, Random Projection LSH (RPLSH) works directly on embeddings:

```python
# Random projection matrix (initialized once, persisted)
projection = np.random.randn(embedding_dim, num_hash_bits)  # e.g., 384 x 16

def hash_embedding(embedding, projection):
    """Hash embedding to binary code via random projection."""
    projected = embedding @ projection
    return tuple(int(x > 0) for x in projected)

# Two embeddings with the same hash are candidate duplicates
```

### Applicability to Mnemoria

Mnemoria's fact store is typically small (100-1000 facts for an agent). At this scale, the O(n) scan in `find_near_duplicates()` takes <10ms. LSH overhead (maintaining the index, false negatives from hash collisions) outweighs the benefit.

**Recommendation**: LSH is premature optimization for Mnemoria. Revisit only if fact stores regularly exceed 10,000 facts. The engineering effort is better spent on entity linking and LLM dedup.

However, SimHash is cheap to compute and could serve as a **fast pre-filter** at scale:

```sql
ALTER TABLE um_facts ADD COLUMN simhash INTEGER;
CREATE INDEX idx_um_facts_simhash ON um_facts(simhash);
```

At store time: compute SimHash, query for facts within Hamming distance 3. If any candidates, run full cosine comparison only on those. This reduces the O(n) scan to O(1) lookup + O(k) verification where k is the number of hash neighbors.

## Research: How Zep Handles Deduplication

### Zep/Graphiti Architecture

Zep (Graphiti engine) treats dedup as part of its knowledge graph construction pipeline:

1. **Episode ingestion**: New conversational turns are processed as "episodes"
2. **Entity extraction**: Entities and relationships are extracted from each episode using an LLM
3. **Entity resolution**: Extracted entities are matched against existing graph nodes using embedding similarity + LLM comparison
4. **Edge dedup**: New edges (facts/relationships) are compared against existing edges connecting the same entity pair
5. **Temporal invalidation**: If a new edge contradicts an existing one connecting the same entities, the old edge gets `invalid_at` set (not deleted)

### Key Differences from Mnemoria

| Aspect | Zep/Graphiti | Mnemoria |
|--------|-------------|----------|
| Data model | Knowledge graph (nodes + edges) | Flat fact store + link graph |
| Dedup unit | Entity-pair edges | Individual facts |
| Dedup method | LLM-based edge comparison | Embedding cosine + word overlap |
| Backend | Neo4j/FalkorDB | SQLite |
| LLM dependency | Required for all ingestion | Optional (embedding-only default) |
| Cost per store | High (multiple LLM calls) | Low (embedding only) |

### What Mnemoria Can Adopt

1. **Entity-pair dedup scope**: Instead of comparing a new fact against ALL existing facts, narrow to facts sharing the same (type, target, entity) triple. This is essentially what Mnemoria's supersession already does for (type, target) pairs, but adding entity resolution would catch cases where the target differs slightly ("auth" vs "authentication").

2. **Tiered dedup thresholds**: Zep uses different confidence levels for different dedup decisions. Mnemoria could:
   - Hard dedup (cosine >= 0.95): Auto-merge, no LLM needed (current behavior)
   - Soft dedup (cosine 0.80-0.95): Trigger entity comparison or LLM check
   - No dedup (cosine < 0.80): Store independently

3. **Never delete, always mark**: Zep never removes deduplicated edges -- it marks them with temporal fields. Mnemoria's current approach of returning the existing fact's ID on dedup (store.py L228) is cleaner for the common case, but a `deduplicated_by` field would help debugging.

## Recommended Implementation Plan

### Phase 1: Tiered Threshold Dedup (v0.3.0, no new dependencies)

Lower the semantic dedup threshold from 0.95 to a tiered system:

```python
# In store.py, replace the single threshold:
DEDUP_HARD_THRESHOLD = 0.95   # Auto-merge, no questions asked
DEDUP_SOFT_THRESHOLD = 0.82   # Merge only if entity/target overlap confirms

# In find_near_duplicates:
def find_near_duplicates(conn, content, embedding, target=None, fact_type=None):
    for r in rows:
        sim = cosine_similarity(embedding, existing_emb)
        
        if sim >= DEDUP_HARD_THRESHOLD:
            # Hard match: always dedup (current behavior)
            duplicates.append((r["id"], sim))
        elif sim >= DEDUP_SOFT_THRESHOLD:
            # Soft match: dedup only if target matches or entity overlap is high
            if target and target != "general" and r["target"] == target:
                duplicates.append((r["id"], sim))
            elif _entity_overlap(content, r["content"]) >= 0.5:
                duplicates.append((r["id"], sim))
```

**Expected impact**: dedup 0.750 -> 0.85. Minimal code change, no new dependencies.

### Phase 2: Entity-Aware Dedup (v0.3.x)

Add `um_entities` and `um_fact_entities` tables. Extract entities at store time. Use entity resolution to scope dedup candidates. This narrows the search space and enables lower cosine thresholds safely.

**Expected impact**: dedup 0.85 -> 0.90.

### Phase 3: LLM-Assisted Dedup (v0.4.0)

Add `dedup_llm_model` config. For soft-match candidates (cosine 0.80-0.95), invoke the agent's LLM for SAME/UPDATE/DIFFERENT classification. Use structured prompts with type and target context.

**Expected impact**: dedup 0.90 -> 0.95+.

### Phase 4: SimHash Pre-Filter (v0.5.0, only if scale demands)

Add `simhash` column to `um_facts`. Use as a fast pre-filter before cosine computation. Only worthwhile above 10,000 facts.

## Interaction with Other Improvements

- **Embedding upgrade** (nomic-embed-text-v1.5): Better embeddings will widen the gap between true duplicates and related-but-different facts. This directly improves the soft threshold zone (0.82-0.95) where most dedup decisions are ambiguous. The EMBEDDING_UPGRADE.md estimates dedup 0.750 -> 0.80-0.85 from embedding quality alone.

- **Topic shift recall** (see TOPIC_SHIFT.md): Better topic discrimination reduces false-positive dedup between facts in different topic clusters that share vocabulary.

- **Temporal validity** (see TEMPORAL_VALIDITY.md): `valid_until` enables temporal dedup -- a new fact about the same entity can mark the old one as expired rather than treating it as a duplicate.

## Summary

| Approach | Effort | Expected Impact | Dependencies |
|----------|--------|-----------------|--------------|
| Tiered thresholds + entity overlap | Low | 0.750 -> 0.85 | None |
| Entity resolution tables | Medium | 0.85 -> 0.90 | Schema migration |
| LLM-assisted comparison | Medium | 0.90 -> 0.95+ | LLM provider |
| LSH pre-filter | Medium | Scale only | datasketch or manual |
| Embedding upgrade (separate) | Medium | 0.750 -> 0.80-0.85 | nomic-embed |

Priority order: Tiered thresholds first (highest ROI), then entity resolution, then LLM dedup. LSH is premature.
