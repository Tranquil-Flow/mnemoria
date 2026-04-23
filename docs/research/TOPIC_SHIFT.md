# Topic Shift Recall Research

Research date: 2026-04-23
Mnemoria version: v0.2.1 (0.927 overall, **topic_shift_recall: 0.833** -- 10/12 correct)

## Problem Statement

When a conversation shifts from topic A to topic B and then queries about topic A, Mnemoria sometimes retrieves topic-B facts that share vocabulary with topic A. Score: 0.833 (10/12 correct, recall@1: 0.75).

The root cause is vocabulary overlap between technical domains. For example, "database migration" (ops topic) and "data migration" (ETL topic) share the tokens "migration" and "data", causing the retrieval pipeline to conflate them. Mnemoria's current retrieval has no concept of topic boundaries -- it treats the entire fact store as a flat collection scored by activation + similarity.

From the v0.1.0 changelog: "topic_shift_recall (75%) requires topical scoping architecture -- overlapping-domain recall remains hard when both topics share dense vocabulary. Not tunable via parameter fix."

## Current Architecture Analysis

### Why Topic Shift Is Hard for Mnemoria

1. **Flat retrieval**: `recall()` queries ALL active facts. No topic partitioning.
2. **Scope != topic**: Mnemoria scopes track conversation sessions, not topic clusters. A single scope can span multiple topics.
3. **Embedding overlap**: The embedding model (all-MiniLM-L6-v2, 384d) does not differentiate well between related domains. "Kubernetes deployment" and "application deployment" have high cosine similarity.
4. **FTS5 ambiguity**: BM25 keyword matching treats shared tokens equally regardless of topic context. "migration" matches both "database migration" and "data migration" with similar scores.
5. **Temporal links**: Recent facts from topic B have strong temporal adjacency links, which can boost them above older topic-A facts even when topic A is queried.

### Benchmark Failure Pattern

The 2 failures in topic_shift_recall (12 tests, 10 correct) likely occur when:
- Topic A and topic B share 2+ substantive keywords
- Topic B facts are more recent (higher base-level activation)
- The query uses shared vocabulary without topic-discriminating terms

## Research: Topic Modeling Approaches

### BERTopic

BERTopic (Grootendorst, 2022) is a topic modeling framework that combines:
1. Document embeddings (any embedding model)
2. UMAP dimensionality reduction (to 5-10 dimensions)
3. HDBSCAN density-based clustering
4. c-TF-IDF for topic representation

For Mnemoria, BERTopic could cluster facts into topic groups at consolidation time:

```python
from bertopic import BERTopic

# During consolidation
facts = [f.content for f in all_active_facts]
embeddings = [f.embedding for f in all_active_facts]

topic_model = BERTopic(
    embedding_model=None,  # Use pre-computed embeddings
    umap_model=UMAP(n_components=5, metric='cosine'),
    hdbscan_model=HDBSCAN(min_cluster_size=3, min_samples=2),
)
topics, probs = topic_model.fit_transform(facts, embeddings=np.array(embeddings))
```

**Pros**: State-of-the-art topic quality. Handles dynamic topic count. Good with short texts.
**Cons**: Heavy dependencies (umap-learn, hdbscan, sklearn). Requires re-clustering on every consolidation. 100+ MB of dependencies.
**Verdict**: Too heavy for an agent memory plugin. But the approach (embed -> reduce -> cluster -> label) is sound and can be approximated without the full BERTopic stack.

### Latent Dirichlet Allocation (LDA)

Classic probabilistic topic model. Each document is a mixture of topics, each topic is a distribution over words.

```python
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
tf_matrix = vectorizer.fit_transform(fact_contents)
lda = LatentDirichletAllocation(n_components=10, random_state=42)
topic_distributions = lda.fit_transform(tf_matrix)
```

**Pros**: Fast. Low memory. Only needs sklearn (already a common dependency). Probabilistic: each fact gets a topic distribution, not a hard assignment.
**Cons**: Bag-of-words model -- ignores word order and semantics. Requires pre-specifying number of topics. Poor with short texts (Mnemoria facts are typically 10-50 words).
**Verdict**: Too crude for Mnemoria's short-text facts. Would conflate "database migration" and "data migration" because it only sees word frequencies.

### Lightweight Embedding-Based Clustering (recommended)

A simpler approach that works with Mnemoria's existing embeddings:

1. At consolidation time, run k-means or agglomerative clustering on fact embeddings
2. Assign each fact a `topic_cluster_id`
3. At recall time, identify the query's nearest cluster and boost same-cluster facts

```python
from sklearn.cluster import AgglomerativeClustering
import numpy as np

def cluster_facts(embeddings: np.ndarray, n_clusters: int = None) -> np.ndarray:
    """Cluster fact embeddings into topic groups."""
    if n_clusters is None:
        # Heuristic: sqrt(n) clusters, minimum 3
        n_clusters = max(3, int(np.sqrt(len(embeddings))))
    
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='cosine',
        linkage='average',
    )
    labels = clustering.fit_predict(embeddings)
    return labels
```

**Pros**: Uses existing embeddings. No new dependencies (sklearn is already used by TF-IDF embedder). Fast (sub-second for 1000 facts). Hierarchical clustering naturally handles topic nesting.
**Cons**: Requires choosing n_clusters. Hard assignment (no soft topic membership). Re-clustering needed as facts are added.
**Verdict**: Good fit for Mnemoria. Lightweight, uses existing infrastructure.

## Research: Automatic Topic Segmentation

### Conversation-Level Segmentation

Instead of clustering the entire fact store, detect topic shifts as facts are stored and tag each fact with a topic segment ID. This is more natural for conversational memory.

#### TextTiling (Hearst, 1997)

Classic algorithm for topic segmentation in text:
1. Divide text into pseudo-sentences
2. Compute lexical similarity between adjacent blocks
3. Topic boundaries occur at similarity valleys (local minima)

```python
def detect_topic_shift(recent_facts: list, new_fact: str, threshold: float = 0.3) -> bool:
    """Detect if a new fact represents a topic shift from recent facts."""
    if len(recent_facts) < 3:
        return False
    
    recent_terms = set()
    for f in recent_facts[-3:]:
        recent_terms.update(_normalize_terms(f.content))
    
    new_terms = _normalize_terms(new_fact)
    
    if not recent_terms or not new_terms:
        return False
    
    overlap = len(recent_terms & new_terms) / len(recent_terms | new_terms)
    return overlap < threshold  # Low overlap = topic shift
```

#### Embedding-Based Segmentation

More robust: compute cosine similarity between a new fact's embedding and the centroid of recent facts. A sharp drop indicates topic shift.

```python
def detect_topic_shift_embedding(
    recent_embeddings: list,
    new_embedding: np.ndarray,
    threshold: float = 0.4,
) -> bool:
    """Detect topic shift via embedding centroid distance."""
    if len(recent_embeddings) < 3:
        return False
    
    centroid = np.mean(recent_embeddings[-5:], axis=0)
    centroid_norm = centroid / np.linalg.norm(centroid)
    new_norm = new_embedding / np.linalg.norm(new_embedding)
    similarity = float(np.dot(centroid_norm, new_norm))
    
    return similarity < threshold
```

### Topic Segment Schema

```sql
ALTER TABLE um_facts ADD COLUMN topic_segment_id TEXT;
CREATE INDEX idx_um_facts_topic ON um_facts(topic_segment_id);

CREATE TABLE um_topic_segments (
    id TEXT PRIMARY KEY,
    label TEXT,              -- Auto-generated or LLM-derived topic label
    centroid BLOB,           -- Average embedding of facts in this segment
    fact_count INTEGER DEFAULT 0,
    created_at REAL,
    last_fact_at REAL,
    scope_id TEXT             -- Topic segments live within scopes
);
```

At store time:
1. Compute cosine similarity between new fact embedding and current segment centroid
2. If similarity < threshold, create a new topic segment
3. Assign new fact to current segment
4. Update segment centroid (running average)

At recall time:
1. Find the topic segment(s) most similar to the query embedding
2. Boost facts in matching segments
3. Dampen facts in non-matching segments

## Research: Preventing Vocabulary Overlap Confusion

### The Core Problem

"Database migration to PostgreSQL 15" and "User data migration to new format" share "migration" and "data/database" tokens. BM25 and token-level features cannot distinguish them. Embeddings help but are imperfect, especially with smaller models.

### Approach 1: Discriminative Term Weighting

Instead of treating all query terms equally, weight terms by how discriminative they are within the fact store:

```python
def discriminative_weight(term: str, conn) -> float:
    """Weight a term by how discriminative it is across topic clusters."""
    # IDF-like: terms appearing in many clusters are less discriminative
    clusters_with_term = conn.execute(
        "SELECT COUNT(DISTINCT topic_segment_id) FROM um_facts "
        "WHERE content LIKE ? AND status = 'active'",
        (f'%{term}%',)
    ).fetchone()[0]
    
    total_clusters = conn.execute(
        "SELECT COUNT(DISTINCT topic_segment_id) FROM um_facts WHERE status = 'active'"
    ).fetchone()[0]
    
    if total_clusters == 0:
        return 1.0
    
    return math.log(total_clusters / max(clusters_with_term, 1)) + 1.0
```

This way, "migration" (appears in many clusters) gets low weight, while "PostgreSQL" (appears in one cluster) gets high weight.

### Approach 2: Topic-Conditioned Retrieval

After identifying the query's most likely topic, apply a topic filter or boost:

```python
def recall_with_topic(self, query, scope=None, top_k=8):
    # Standard retrieval
    scored = self._score_candidates(...)
    
    # Find query's topic cluster
    query_embedding = self._embed(query)
    segment_rows = self._conn.execute(
        "SELECT id, centroid FROM um_topic_segments ORDER BY created_at DESC LIMIT 20"
    ).fetchall()
    
    best_segment = None
    best_sim = -1
    for row in segment_rows:
        centroid = np.frombuffer(row["centroid"], dtype=np.float32)
        sim = cosine_similarity(query_embedding, centroid)
        if sim > best_sim:
            best_sim = sim
            best_segment = row["id"]
    
    # Boost same-topic facts, dampen off-topic
    if best_segment and best_sim > 0.5:
        for item in scored:
            if item.fact.topic_segment_id == best_segment:
                item.score *= 1.2  # Topic match boost
            elif item.fact.topic_segment_id is not None:
                item.score *= 0.85  # Off-topic dampening
    
    return sorted(scored, key=lambda s: s.score, reverse=True)[:top_k]
```

### Approach 3: Contrastive Fact Embeddings

Re-embed facts using contrastive learning to maximize distance between same-vocabulary, different-topic facts. This requires fine-tuning the embedding model on Mnemoria's fact pairs, which is impractical for a plugin.

However, the **task prefix** approach (used by nomic-embed-text-v1.5) achieves a similar effect: embedding "search_query: database migration setup" vs "search_document: database migration to PostgreSQL 15" naturally emphasizes the query-document relationship over surface similarity.

## Research: Cluster-Based Retrieval Strategies

### Cluster Hypothesis (van Rijsbergen, 1979)

"Closely associated documents tend to be relevant to the same requests." The cluster hypothesis suggests that retrieving from the best-matching cluster outperforms global retrieval for focused queries.

### Two-Stage Cluster Retrieval

1. **Stage 1 (cluster selection)**: Compute similarity between query and each cluster centroid. Select top-2 clusters.
2. **Stage 2 (within-cluster ranking)**: Run standard activation scoring only on facts within selected clusters.

This dramatically reduces the candidate pool and eliminates off-topic noise. However, it can miss cross-cluster connections (which Mnemoria's Hebbian links are designed to capture).

### Hybrid Approach: Cluster Boost, Not Filter

Rather than hard-filtering by cluster, apply a cluster affinity boost within the existing dampening pipeline:

```python
# In apply_dampening(), add after resolution boost:

# ── 5. TOPIC COHERENCE BOOST ──
if self._config.enable_topic_coherence:
    query_segment = self._find_nearest_segment(query_embedding)
    if query_segment:
        for item in scored:
            if item.fact.topic_segment_id == query_segment:
                item.score *= 1.15
                item.components['topic_coherence_boost'] = 1.15
```

This preserves Mnemoria's global retrieval (no cross-cluster blind spots) while still preferring topic-coherent results.

### Google's Cluster-Based Passage Retrieval

Google Research (2023) showed that clustering passages and retrieving cluster centroids first, then drilling into top clusters, improves retrieval quality by 5-10% on NQ and TriviaQA benchmarks. Their approach:

1. Pre-compute cluster centroids using K-means on passage embeddings
2. At query time, retrieve top-k centroids (fast, k << n)
3. Score all passages within top-k clusters
4. Return best passages across selected clusters

For Mnemoria, this maps to:
1. Cluster facts during consolidation
2. Store centroids in `um_topic_segments`
3. At recall time, find best segment, boost facts in that segment

## Recommended Implementation Plan

### Phase 1: Topic Segment Detection (v0.3.0)

Add topic segmentation at store time:

1. **Schema**: Add `topic_segment_id` to `um_facts`. Create `um_topic_segments` table with centroid embeddings.
2. **Store-time detection**: On each `store()`, compare new fact embedding against current segment centroid. If cosine < 0.45, start new segment.
3. **Segment centroid update**: Running average of fact embeddings in segment.

No retrieval changes yet -- just tag facts with segments.

### Phase 2: Topic-Aware Retrieval Boost (v0.3.x)

Add topic coherence boost to the dampening pipeline:

1. At recall time, find query's nearest topic segment
2. Apply 1.15x boost to same-segment facts
3. Apply 0.90x dampening to facts in distant segments (cosine < 0.3 with query segment)
4. Add `enable_topic_coherence: bool = True` to config

**Expected impact**: topic_shift_recall 0.833 -> 0.90-0.92.

### Phase 3: Consolidation-Time Re-Clustering (v0.4.0)

During `consolidate()`, re-cluster all active facts using agglomerative clustering on embeddings. This handles segment drift (a segment's centroid shifts as facts are added/removed) and merges segments that have converged.

### Phase 4: Discriminative Term Weighting (v0.4.x)

Replace uniform term weighting in `_normalize_terms()` and `fts5_search()` with segment-aware IDF weighting. Terms that appear across many segments are downweighted.

## Integration with Mnemoria Architecture

| Existing Feature | Topic Enhancement |
|---|---|
| Scopes | Scopes = session boundary. Topic segments live within scopes (a scope can have multiple topic segments). |
| Temporal links | Temporal links already connect facts within the same scope+time window. Topic segments add semantic coherence on top. |
| Gravity dampening | Already penalizes cosine ghosts (high similarity, no keyword overlap). Topic dampening adds cluster-level penalization. |
| Hub dampening | Hub nodes that span multiple topic segments are more likely to be generic/noisy. Hub + cross-topic = stronger dampening signal. |
| Intent classification | Query intent (procedural, value, decision) is orthogonal to topic. Both can boost independently. |
| Consolidation | Re-clustering during consolidation naturally handles segment evolution. |
| LinUCB bandit | The pipeline optimizer can learn whether topic_coherence_boost helps for a given user's query patterns. |

## Expected Benchmark Impact

| Category | Current | Expected | Mechanism |
|---|---|---|---|
| topic_shift_recall | 0.833 | 0.90-0.92 | Direct fix via topic-aware retrieval |
| semantic_recall | 0.800 | 0.82 | Better discrimination between related topics |
| cross_reference | 0.956 | 0.96 | Cross-topic links preserved (no hard filtering) |
| contradictions | 0.950 | 0.96 | Same-topic contradictions detected more reliably |
| **Overall** | **0.927** | **0.935-0.94** | Conservative estimate |
