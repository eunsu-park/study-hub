[← Previous: 22. Vector Storage and Indexing](22_Vector_Storage_and_Indexing.md) | [Next: Overview →](00_Overview.md)

# 23. Production Vector Search

## Learning Objectives

1. Design hybrid search systems combining dense vector retrieval with sparse keyword matching
2. Implement metadata filtering strategies and understand pre-filtering vs post-filtering tradeoffs
3. Build reranking pipelines using cross-encoder models to improve precision
4. Scale vector search horizontally via sharding, replication, and load balancing
5. Monitor vector search systems with appropriate metrics and alerting thresholds
6. Optimize costs through dimensionality reduction, quantization, and tiered storage
7. Integrate vector search into production data pipelines with batch embedding updates

---

## Overview

Deploying vector search in production goes far beyond indexing embeddings and running queries. Production systems must handle hybrid retrieval (combining semantic and keyword search), filter results by business metadata, rerank for precision, scale to handle traffic spikes, and do all of this reliably at acceptable cost.

This lesson covers the operational side of vector search — the patterns and practices that data engineers need to turn a prototype into a production service. We cover hybrid search fusion, filtering strategies, reranking pipelines, scaling architectures, monitoring, cost optimization, and integration with broader data pipelines.

> **Why this matters for data engineers**: Vector search is increasingly embedded in production data products — recommendation engines, customer support, knowledge bases, fraud detection. The data engineer owns the pipeline that feeds embeddings into the vector store, monitors its health, and ensures the index stays current as source data changes.

---

## 1. Hybrid Search

### 1.1 Why Hybrid Search

```
Pure vector search vs pure keyword search:

  Query: "error code E-4021 troubleshooting"

  Vector search results:                Keyword search results:
  1. General troubleshooting guide      1. Doc mentioning E-4021 specifically
  2. Common error patterns article      2. E-4021 release notes
  3. System diagnostics overview        3. E-4021 patch instructions

  Vector search missed the exact error code!

  Query: "fast approximate nearest neighbor algorithm"

  Vector search results:                Keyword search results:
  1. HNSW algorithm deep dive           1. (no results — no exact match)
  2. FAISS performance guide            2.
  3. ANN benchmarks                     3.

  Keyword search missed the semantic match!

Hybrid search combines both → gets the best of both worlds
```

### 1.2 Dense + Sparse Retrieval Architecture

```
Hybrid Search Pipeline:

  Query: "E-4021 troubleshooting steps"
       │
       ├──────────────────────┐
       │                      │
       ▼                      ▼
  ┌──────────┐         ┌──────────┐
  │ Embedding │         │ Tokenizer│
  │ Model     │         │ (BM25)   │
  └────┬─────┘         └────┬─────┘
       │                     │
       ▼                     ▼
  ┌──────────┐         ┌──────────┐
  │ Vector   │         │ Inverted │
  │ Index    │         │ Index    │
  │ (HNSW)   │         │ (BM25)   │
  └────┬─────┘         └────┬─────┘
       │                     │
       │  Top-K dense        │  Top-K sparse
       │  results            │  results
       ▼                     ▼
  ┌──────────────────────────────┐
  │      Fusion Algorithm         │
  │  (RRF, linear combination,   │
  │   or learned fusion)          │
  └──────────────┬───────────────┘
                 │
                 ▼
          Merged top-K results
```

### 1.3 Reciprocal Rank Fusion (RRF)

```python
"""
Reciprocal Rank Fusion (RRF) combines ranked lists from
different retrieval systems without requiring score normalization.
"""

def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Fuse multiple ranked lists using RRF.

    Args:
        ranked_lists: List of ranked document ID lists
        k: RRF constant (higher = less weight to top ranks)

    Returns:
        List of (doc_id, rrf_score) sorted by score descending
    """
    scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (k + rank)

    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results

# Example: fuse dense and sparse results
dense_results = ["doc_A", "doc_B", "doc_C", "doc_D", "doc_E"]
sparse_results = ["doc_C", "doc_F", "doc_A", "doc_G", "doc_B"]

fused = reciprocal_rank_fusion([dense_results, sparse_results], k=60)
# doc_A: 1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 = 0.03226 (rank 1)
# doc_C: 1/(60+3) + 1/(60+1) = 0.01587 + 0.01639 = 0.03226 (rank 1 tie)
# doc_B: 1/(60+2) + 1/(60+5) = 0.01613 + 0.01538 = 0.03151 (rank 3)
```

### 1.4 Weighted Linear Combination

```python
"""
Linear combination requires score normalization since dense
and sparse scores live in different ranges.
"""

import numpy as np

def normalize_scores(scores: list[float]) -> list[float]:
    """Min-max normalize scores to [0, 1]."""
    arr = np.array(scores)
    if arr.max() == arr.min():
        return [1.0] * len(scores)
    return ((arr - arr.min()) / (arr.max() - arr.min())).tolist()

def linear_combination(
    dense_results: list[tuple[str, float]],
    sparse_results: list[tuple[str, float]],
    alpha: float = 0.7,  # weight for dense (semantic)
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """
    Combine dense and sparse results with weighted scores.
    alpha=1.0 → pure dense, alpha=0.0 → pure sparse
    """
    # Normalize scores
    dense_ids = [r[0] for r in dense_results]
    dense_scores = normalize_scores([r[1] for r in dense_results])
    sparse_ids = [r[0] for r in sparse_results]
    sparse_scores = normalize_scores([r[1] for r in sparse_results])

    # Merge
    combined: dict[str, float] = {}
    for doc_id, score in zip(dense_ids, dense_scores):
        combined[doc_id] = alpha * score
    for doc_id, score in zip(sparse_ids, sparse_scores):
        combined[doc_id] = combined.get(doc_id, 0.0) + (1 - alpha) * score

    sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]

# Tuning alpha:
#   alpha=0.7 → good default for most use cases
#   alpha=0.9 → when queries are mostly semantic/conversational
#   alpha=0.3 → when queries contain specific codes, IDs, exact terms
```

### 1.5 SPLADE: Learned Sparse Representations

```
SPLADE (Sparse Lexical and Expansion) — Transformer-learned sparse vectors:

  Traditional BM25:       SPLADE:
  "machine learning"  →   "machine learning"  →
  {"machine": 1.2,        {"machine": 0.8,
   "learning": 1.5}        "learning": 1.1,
                            "AI": 0.6,          ← expansion!
                            "neural": 0.3,      ← expansion!
                            "algorithm": 0.2}   ← expansion!

  SPLADE learns to expand queries and documents with related terms,
  combining BM25 efficiency with semantic awareness.

  Pipeline:
    Text → BERT encoder → ReLU + log → Sparse vector (30K dims, ~100 non-zero)
                                          │
                                          ▼
                                    Inverted index (same as BM25)
```

---

## 2. Metadata Filtering

### 2.1 Pre-Filtering vs Post-Filtering

```
Pre-Filtering:
  ① Apply metadata filter → candidate set
  ② Run vector search on filtered candidates only

  ✓ Guarantees exactly N results matching filter
  ✗ If filter is very selective (<1% of data), HNSW graph
    may have few connections within the filtered subset
    → degraded recall

Post-Filtering:
  ① Run vector search on full index → large candidate set
  ② Apply metadata filter on candidates

  ✓ Vector search operates on full graph (good recall)
  ✗ May return fewer than N results if filter removes many
  ✗ Wastes compute searching irrelevant vectors

Hybrid (most modern DBs):
  ① Use index hints to guide search toward filtered regions
  ② Score considers both vector similarity and filter match
  ✗ Complex implementation

  Qdrant approach:
    Uses payload indexes to prune HNSW traversal paths
    that cannot satisfy filter conditions

  Milvus approach:
    Partition pruning (search only relevant partitions)
    + segment-level bloom filters
```

### 2.2 Filter Design Patterns

```python
"""
Metadata filtering patterns for production vector search.
"""

# Pattern 1: Partition-based filtering (Milvus)
# Best when filter divides data into large, stable groups
collection.search(
    data=[query_vector],
    anns_field="embedding",
    param={"metric_type": "L2", "params": {"nprobe": 64}},
    limit=10,
    partition_names=["electronics", "clothing"],  # partition pruning
)

# Pattern 2: Boolean expression filtering (Milvus)
# Best for complex conditions on scalar fields
collection.search(
    data=[query_vector],
    anns_field="embedding",
    param={"metric_type": "L2", "params": {"nprobe": 64}},
    limit=10,
    expr=(
        'category in ["electronics", "clothing"] '
        'and price >= 10.0 and price <= 100.0 '
        'and in_stock == true '
        'and brand != "Acme"'
    ),
)

# Pattern 3: Nested payload filtering (Qdrant)
# Best for hierarchical metadata
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

filter_config = Filter(
    must=[
        FieldCondition(key="category", match=MatchValue(value="electronics")),
        FieldCondition(key="price", range=Range(gte=10.0, lte=100.0)),
        FieldCondition(key="supplier.country", match=MatchValue(value="US")),
    ],
    must_not=[
        FieldCondition(key="status", match=MatchValue(value="discontinued")),
    ],
)

# Pattern 4: Tag-based filtering (Pinecone)
# Best for multi-label classification
results = index.query(
    vector=query_vector,
    top_k=10,
    filter={
        "tags": {"$in": ["sale", "featured"]},
        "rating": {"$gte": 4.0},
    },
)
```

### 2.3 Filtering Performance Tips

```
Optimizing metadata filters:

1. Index your filter fields
   - Qdrant: payload_index (automatically created for filtered fields)
   - Milvus: scalar index (explicit creation recommended)
   - Weaviate: indexFilterable=True on schema properties

2. Use partitions for high-cardinality categorical filters
   - Milvus: create partitions by tenant, region, or category
   - Reduces search space without post-filtering overhead

3. Avoid high-selectivity filters on vector-only indexes
   - If filter selects <1% of data, pre-filtering degrades HNSW recall
   - Solution: over-fetch (search 10x candidates) then post-filter

4. Denormalize metadata into the vector store
   - Avoid join queries between vector DB and relational DB
   - Trade storage for query simplicity
   - Update metadata via upsert when source data changes

5. Use bloom filters for existence checks
   - "Does this document exist in the index?"
   - Faster than point lookup for batch deduplication
```

---

## 3. Reranking Pipelines

### 3.1 Two-Stage Retrieval

```
Why rerank?

  Stage 1 (Retrieval):
    - Fast but approximate
    - Uses bi-encoder (query and doc encoded independently)
    - Retrieve top-100 candidates in ~5ms

  Stage 2 (Reranking):
    - Slow but precise
    - Uses cross-encoder (query and doc encoded together)
    - Rerank top-100 → top-10 in ~50ms

  Bi-encoder (Stage 1):           Cross-encoder (Stage 2):
  ┌─────┐    ┌─────┐             ┌──────────────────┐
  │Query│    │ Doc │             │ [CLS] Query [SEP] │
  └──┬──┘    └──┬──┘             │      Doc [SEP]    │
     │          │                └────────┬─────────┘
     ▼          ▼                         │
  Encoder    Encoder                  Encoder
     │          │                         │
     ▼          ▼                         ▼
  q_vec      d_vec                    score (0-1)
     │          │
  cosine(q, d) = 0.82

  Cross-encoder is ~100x slower but ~10-20% more accurate
  because it can attend to query-document interactions
```

### 3.2 Cross-Encoder Reranking Implementation

```python
"""
Cross-encoder reranking using sentence-transformers.
"""

from sentence_transformers import CrossEncoder
import numpy as np

# Load cross-encoder model
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

def rerank_results(
    query: str,
    documents: list[dict],
    top_k: int = 10,
) -> list[dict]:
    """
    Rerank retrieved documents using a cross-encoder.

    Args:
        query: User query string
        documents: List of {"id": ..., "text": ..., "score": ...}
        top_k: Number of results to return after reranking

    Returns:
        Reranked list of documents with updated scores
    """
    if not documents:
        return []

    # Prepare pairs for cross-encoder
    pairs = [(query, doc["text"]) for doc in documents]

    # Score all pairs
    scores = reranker.predict(pairs, show_progress_bar=False)

    # Attach scores and sort
    for doc, score in zip(documents, scores):
        doc["rerank_score"] = float(score)
        doc["original_score"] = doc.get("score", 0.0)

    reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]


# Usage in a retrieval pipeline
def search_and_rerank(query: str, collection, top_k: int = 10):
    """Full retrieval pipeline: retrieve → rerank."""
    # Stage 1: Retrieve top-100 candidates (fast, approximate)
    candidates = collection.search(
        query_embedding=encode_query(query),
        limit=100,
    )

    # Stage 2: Rerank with cross-encoder (slow, precise)
    reranked = rerank_results(query, candidates, top_k=top_k)

    return reranked
```

### 3.3 Cohere Rerank API (Managed)

```python
"""
Using Cohere's rerank API for production reranking
without running your own model infrastructure.
"""

import cohere

co = cohere.Client("your-api-key")

def rerank_with_cohere(
    query: str,
    documents: list[str],
    top_k: int = 10,
    model: str = "rerank-english-v3.0",
) -> list[dict]:
    """Rerank using Cohere's hosted cross-encoder."""
    response = co.rerank(
        query=query,
        documents=documents,
        top_n=top_k,
        model=model,
    )

    results = []
    for result in response.results:
        results.append({
            "index": result.index,
            "text": documents[result.index],
            "relevance_score": result.relevance_score,
        })
    return results

# Cost: ~$1 per 1000 searches (reranking 100 docs each)
# Latency: ~100-200ms for 100 documents
# Accuracy: state-of-the-art for English text
```

### 3.4 Multi-Stage Pipeline Architecture

```
Production Retrieval Pipeline:

  Query
    │
    ▼
  ┌────────────────────┐
  │ Query Understanding │   Intent classification, query expansion
  └─────────┬──────────┘
            │
    ┌───────┴───────┐
    │               │
    ▼               ▼
  Dense           Sparse
  Retrieval       Retrieval        Stage 1: Retrieve (100 candidates)
  (bi-encoder)    (BM25/SPLADE)
    │               │
    └───────┬───────┘
            │
            ▼
  ┌────────────────────┐
  │   Fusion (RRF)      │            Merge to ~100 unique candidates
  └─────────┬──────────┘
            │
            ▼
  ┌────────────────────┐
  │  Metadata Filter    │            Apply business rules (ACL, freshness)
  └─────────┬──────────┘
            │
            ▼
  ┌────────────────────┐
  │  Cross-Encoder      │            Stage 2: Rerank (100 → 20)
  │  Reranking          │
  └─────────┬──────────┘
            │
            ▼
  ┌────────────────────┐
  │  Business Logic     │            Dedup, boost, diversity, personalize
  └─────────┬──────────┘
            │
            ▼
       Top-10 Results

  Total latency budget:
    Retrieval: ~5ms
    Fusion: ~1ms
    Filter: ~2ms
    Reranking: ~50ms
    Business logic: ~2ms
    ─────────────────
    Total: ~60ms p50
```

---

## 4. Scaling Vector Search

### 4.1 Sharding Strategies

```
Sharding distributes vectors across multiple nodes:

  Strategy 1: Hash-based sharding
  ┌──────────┐
  │ Query    │─── hash(query) % N ──→ Shard K
  └──────────┘                        (search one shard)
  ✗ Cannot search all data (only one shard per query)
  ✗ Only useful for partitioned data (e.g., per-tenant)

  Strategy 2: Scatter-gather (most common)
  ┌──────────┐
  │ Query    │──→ All shards in parallel
  └──────────┘     │    │    │
                   ▼    ▼    ▼
               Shard1 Shard2 Shard3  (each returns top-K)
                   │    │    │
                   └────┼────┘
                        ▼
                   Merge top-K from all shards
                   (return global top-K)

  ✓ Searches all data
  ✗ Latency = slowest shard + merge time
  ✗ More shards = more network overhead

  Strategy 3: Learned routing
  ┌──────────┐
  │ Query    │──→ Router model predicts top-2 shards
  └──────────┘     │    │
                   ▼    ▼
               Shard2 Shard5  (search only relevant shards)
  ✓ Reduces fan-out
  ✗ Requires training a routing model
  ✗ Risk of missing results if routing is wrong
```

### 4.2 Replication for High Availability

```
Replication patterns:

  Single replica (no HA):
  ┌──────────┐
  │ Shard 1  │  ← single point of failure
  └──────────┘

  Read replicas (HA for reads):
  ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ Shard 1  │    │ Replica  │    │ Replica  │
  │ (primary)│───→│ 1a       │───→│ 1b       │
  │ (writes) │    │ (reads)  │    │ (reads)  │
  └──────────┘    └──────────┘    └──────────┘

  Load balancer distributes reads across replicas
  Primary handles writes → async replication to replicas

  Milvus replication:
    - Replica groups: each shard has N replicas
    - Consistency: strong (sync) or eventual (async)
    - Failover: automatic promotion of replica to primary

  Qdrant replication:
    - Raft consensus for write consistency
    - Configurable write_consistency_factor (1 = fast, N = safe)
    - Automatic shard rebalancing on node addition
```

### 4.3 Capacity Planning

```python
"""
Capacity planning calculator for vector search deployments.
"""

def estimate_resources(
    n_vectors: int,
    dim: int,
    index_type: str = "HNSW",
    quantization: str = "none",
    qps_target: int = 1000,
    replication_factor: int = 2,
) -> dict:
    """Estimate compute resources for a vector search deployment."""

    # Vector storage
    bytes_per_element = {
        "none": 4,       # float32
        "float16": 2,
        "int8": 1,
        "pq96": 96 / dim,  # PQ with 96 subquantizers
    }
    bpe = bytes_per_element.get(quantization, 4)
    vector_memory_gb = (n_vectors * dim * bpe) / (1024**3)

    # Index overhead
    index_overhead = {
        "HNSW": n_vectors * 32 * 2 * 8 / (1024**3),  # M=32
        "IVF": 0.1 * vector_memory_gb,                 # ~10% overhead
        "Flat": 0,
    }
    index_gb = index_overhead.get(index_type, 0)

    # Metadata overhead (assume ~200 bytes per vector)
    metadata_gb = (n_vectors * 200) / (1024**3)

    # Total per shard
    total_per_shard_gb = vector_memory_gb + index_gb + metadata_gb

    # QPS estimation (rough: HNSW ~3000 QPS per core for 768d)
    qps_per_core = {"HNSW": 3000, "IVF": 5000, "Flat": 50}
    cores_needed = qps_target / qps_per_core.get(index_type, 1000)

    # Sharding: split if single node can't hold all data
    max_memory_per_node_gb = 64  # typical instance
    n_shards = max(1, int(total_per_shard_gb / (max_memory_per_node_gb * 0.7)) + 1)

    total_nodes = n_shards * replication_factor

    return {
        "vector_memory_gb": round(vector_memory_gb, 2),
        "index_overhead_gb": round(index_gb, 2),
        "metadata_gb": round(metadata_gb, 2),
        "total_per_shard_gb": round(total_per_shard_gb, 2),
        "n_shards": n_shards,
        "replication_factor": replication_factor,
        "total_nodes": total_nodes,
        "cores_per_node": max(4, int(cores_needed / n_shards) + 1),
        "ram_per_node_gb": min(max_memory_per_node_gb,
                               int(total_per_shard_gb / n_shards * 1.3) + 1),
    }

# Example: 50M vectors, 768 dimensions, HNSW, int8 quantization
plan = estimate_resources(
    n_vectors=50_000_000,
    dim=768,
    index_type="HNSW",
    quantization="int8",
    qps_target=5000,
    replication_factor=2,
)
# vector_memory_gb: 35.76
# index_overhead_gb: 4.77
# n_shards: 1
# total_nodes: 2
# cores_per_node: 4
# ram_per_node_gb: 53
```

---

## 5. Monitoring and Observability

### 5.1 Key Metrics

```
Vector search monitoring metrics:

  Latency Metrics:
  ┌─────────────────────────────────────────────────────────┐
  │ search_latency_p50_ms     Target: < 10ms                │
  │ search_latency_p95_ms     Target: < 50ms                │
  │ search_latency_p99_ms     Target: < 100ms               │
  │ rerank_latency_p50_ms     Target: < 100ms               │
  │ embedding_latency_p50_ms  Target: < 20ms                │
  └─────────────────────────────────────────────────────────┘

  Throughput Metrics:
  ┌─────────────────────────────────────────────────────────┐
  │ search_qps                Queries per second             │
  │ upsert_rate               Vectors ingested per second    │
  │ batch_embedding_rate      Embeddings generated per sec   │
  └─────────────────────────────────────────────────────────┘

  Quality Metrics:
  ┌─────────────────────────────────────────────────────────┐
  │ recall@10                 Sample queries vs ground truth │
  │ mrr@10                    Mean reciprocal rank           │
  │ empty_result_rate         % queries with 0 results      │
  │ filter_selectivity        Avg % of data after filtering  │
  └─────────────────────────────────────────────────────────┘

  Resource Metrics:
  ┌─────────────────────────────────────────────────────────┐
  │ memory_usage_pct          Target: < 80%                  │
  │ disk_usage_pct            Target: < 70%                  │
  │ cpu_usage_pct             Target: < 60% sustained        │
  │ index_size_vectors        Total vectors in index         │
  │ segment_count             Number of index segments       │
  └─────────────────────────────────────────────────────────┘
```

### 5.2 Prometheus Metrics Implementation

```python
"""
Instrumenting vector search with Prometheus metrics.
"""

from prometheus_client import Histogram, Counter, Gauge, Summary
import time

# Define metrics
SEARCH_LATENCY = Histogram(
    "vector_search_latency_seconds",
    "Vector search latency",
    ["collection", "index_type"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

SEARCH_RESULTS = Histogram(
    "vector_search_result_count",
    "Number of results returned",
    ["collection"],
    buckets=[0, 1, 5, 10, 20, 50, 100],
)

SEARCH_QPS = Counter(
    "vector_search_total",
    "Total vector searches",
    ["collection", "status"],
)

INDEX_SIZE = Gauge(
    "vector_index_size_total",
    "Total vectors in index",
    ["collection"],
)

EMBEDDING_LATENCY = Histogram(
    "embedding_generation_latency_seconds",
    "Embedding generation latency",
    ["model"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

def instrumented_search(collection, query_vector, top_k=10, filters=None):
    """Search with Prometheus instrumentation."""
    start = time.perf_counter()
    try:
        results = collection.search(
            query_embedding=query_vector,
            limit=top_k,
            filter=filters,
        )
        duration = time.perf_counter() - start

        SEARCH_LATENCY.labels(
            collection=collection.name,
            index_type="HNSW",
        ).observe(duration)
        SEARCH_RESULTS.labels(collection=collection.name).observe(len(results))
        SEARCH_QPS.labels(collection=collection.name, status="success").inc()

        return results
    except Exception as e:
        SEARCH_QPS.labels(collection=collection.name, status="error").inc()
        raise
```

### 5.3 Recall Monitoring

```python
"""
Continuous recall monitoring using ground truth queries.
Run as a periodic job (e.g., every hour via Airflow).
"""

import numpy as np

def compute_recall_at_k(
    predicted: list[list[str]],
    ground_truth: list[list[str]],
    k: int = 10,
) -> float:
    """Compute recall@k averaged over queries."""
    recalls = []
    for pred, truth in zip(predicted, ground_truth):
        pred_set = set(pred[:k])
        truth_set = set(truth[:k])
        if len(truth_set) == 0:
            continue
        recall = len(pred_set & truth_set) / len(truth_set)
        recalls.append(recall)
    return np.mean(recalls) if recalls else 0.0

def recall_monitoring_job(
    collection,
    ground_truth_queries: list[dict],
    alert_threshold: float = 0.90,
):
    """
    Monitor recall by running ground truth queries and alerting on drops.

    ground_truth_queries: [{"vector": [...], "expected_ids": ["a", "b", ...]}]
    """
    predicted = []
    expected = []

    for gt in ground_truth_queries:
        results = collection.search(query_embedding=gt["vector"], limit=10)
        result_ids = [r["id"] for r in results]
        predicted.append(result_ids)
        expected.append(gt["expected_ids"])

    recall = compute_recall_at_k(predicted, expected, k=10)

    if recall < alert_threshold:
        # Send alert via PagerDuty/Slack/email
        send_alert(
            severity="warning",
            message=f"Vector search recall@10 dropped to {recall:.3f} "
                    f"(threshold: {alert_threshold})",
        )

    return recall
```

### 5.4 Grafana Dashboard Layout

```
Recommended Grafana dashboard panels:

  Row 1: Overview
  ┌───────────────┬───────────────┬───────────────┬───────────────┐
  │ Search QPS    │ p50 Latency   │ p99 Latency   │ Error Rate    │
  │ (counter)     │ (gauge)       │ (gauge)       │ (percentage)  │
  └───────────────┴───────────────┴───────────────┴───────────────┘

  Row 2: Quality
  ┌─────────────────────────────┬─────────────────────────────────┐
  │ Recall@10 over time         │ Empty result rate               │
  │ (time series)               │ (time series)                   │
  └─────────────────────────────┴─────────────────────────────────┘

  Row 3: Resources
  ┌───────────────┬───────────────┬───────────────┬───────────────┐
  │ Memory Usage  │ CPU Usage     │ Disk Usage    │ Index Size    │
  │ (per node)    │ (per node)    │ (per node)    │ (vectors)     │
  └───────────────┴───────────────┴───────────────┴───────────────┘

  Row 4: Pipeline
  ┌─────────────────────────────┬─────────────────────────────────┐
  │ Embedding generation rate   │ Upsert throughput               │
  │ (time series)               │ (time series)                   │
  └─────────────────────────────┴─────────────────────────────────┘
```

---

## 6. Cost Optimization

### 6.1 Dimensionality Reduction

```python
"""
Reducing embedding dimensions to save memory and improve speed.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection

# Original: 1536 dimensions (OpenAI text-embedding-3-large)
original_dim = 1536
target_dim = 512
n_vectors = 1_000_000

# Method 1: PCA (best quality, requires training data)
pca = PCA(n_components=target_dim)
training_data = np.random.randn(50_000, original_dim).astype('float32')
pca.fit(training_data)

reduced_vectors = pca.transform(original_vectors)
# Explained variance ratio: typically 90-95% with 512 dims
# Memory savings: 3x reduction

# Method 2: Matryoshka embeddings (if model supports it)
# OpenAI text-embedding-3-* supports truncation
# Just use first N dimensions of the embedding
# response = openai.embeddings.create(
#     model="text-embedding-3-small",
#     input="text",
#     dimensions=512,  # truncate from 1536
# )

# Method 3: Random projection (fast, no training)
rp = GaussianRandomProjection(n_components=target_dim)
reduced_vectors = rp.fit_transform(original_vectors)
# Lower quality than PCA but O(1) training time

# Cost comparison (10M vectors):
#   1536d float32: 57.2 GB RAM → ~$400/month (cloud)
#   512d float32:  19.1 GB RAM → ~$133/month (cloud)
#   512d int8:     4.8 GB RAM  → ~$33/month (cloud)
```

### 6.2 Quantization Strategies

```
Quantization reduces bytes per vector:

  Method          Bytes/dim    Quality Loss    Speed Impact
  ────────────────────────────────────────────────────────────
  float32         4            baseline        baseline
  float16         2            negligible      ~same
  Scalar (int8)   1            1-3% recall     ~1.5x faster
  PQ (m=96)       96/dim       5-10% recall    ~2-3x faster
  Binary          1/8          10-20% recall   ~10x faster

  Recommended progression:
  1. Start with float32 (correctness first)
  2. Switch to int8/SQ8 (easy 4x memory reduction)
  3. Add PQ if memory is still tight (requires tuning)
  4. Binary only for initial candidate screening
```

### 6.3 Tiered Storage

```
Tiered storage reduces cost for large collections:

  Hot tier (RAM):
  ┌────────────────────────────────────┐
  │ Recent/frequently accessed vectors  │
  │ HNSW index fully in memory          │
  │ Cost: $7-12/GB/month               │
  │ Latency: 1-5ms                     │
  └────────────────────────────────────┘
       ↕ promotion/demotion
  Warm tier (SSD/mmap):
  ┌────────────────────────────────────┐
  │ Older vectors, moderate access      │
  │ HNSW graph in RAM, vectors on disk  │
  │ Cost: $0.50-1/GB/month             │
  │ Latency: 5-20ms                    │
  └────────────────────────────────────┘
       ↕ archival
  Cold tier (Object storage):
  ┌────────────────────────────────────┐
  │ Archive, rarely accessed            │
  │ No index (brute-force if needed)    │
  │ Cost: $0.02/GB/month               │
  │ Latency: 100ms-1s                  │
  └────────────────────────────────────┘

  Implementation:
  - Qdrant: on_disk=True for mmap (warm tier)
  - Milvus: tiered storage with MinIO cold tier
  - Pinecone: automatic (serverless pricing handles it)
```

### 6.4 Cost Comparison Table

```
Monthly cost estimate for 10M vectors, 768 dimensions, 1000 QPS:

  Option                    Memory    Compute     Storage    Total/month
  ──────────────────────────────────────────────────────────────────────
  Self-hosted Qdrant        $280      $400        $50        ~$730
  (3x r6i.2xlarge, SQ8)

  Self-hosted Milvus        $400      $600        $100       ~$1,100
  (distributed, 5 nodes)

  Pinecone Serverless       N/A       N/A         N/A        ~$700-1,500
  (pay per read/write)      (depends on read/write patterns)

  Weaviate Cloud            N/A       N/A         N/A        ~$800-1,200
  (managed)

  FAISS on EC2              $280      $200        $20        ~$500
  (single node, no HA)      (cheapest but no built-in HA)
```

---

## 7. Production Deployment Patterns

### 7.1 Blue-Green Index Deployment

```
Blue-Green deployment for zero-downtime index updates:

  ┌──────────────┐
  │ Load Balancer │
  └──────┬───────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
  ┌─────┐  ┌─────┐
  │Blue │  │Green│
  │(v1) │  │(v2) │   ← Build new index on Green
  │ ✓   │  │ ... │
  └─────┘  └─────┘

  Steps:
  1. Blue is serving traffic (index v1)
  2. Build new index (v2) on Green (offline, no impact)
  3. Run validation queries on Green
  4. Switch load balancer to Green
  5. Blue becomes the next build target

  Benefits:
  - Zero downtime during index rebuilds
  - Instant rollback (switch back to Blue)
  - Can run A/B tests between versions
```

### 7.2 Shadow Index Pattern

```
Shadow indexing for safe embedding model migration:

  Query
    │
    ├──────────────────┐
    │                  │ (async, non-blocking)
    ▼                  ▼
  ┌──────────┐   ┌──────────┐
  │ Primary   │   │ Shadow   │
  │ Index     │   │ Index    │
  │ (model A) │   │ (model B)│
  └────┬─────┘   └────┬─────┘
       │               │
       ▼               ▼
  Response to      Log results
  user             for comparison

  After collecting comparison data:
  - If Shadow recall > Primary recall → promote Shadow
  - If Shadow recall < Primary recall → discard Shadow
  - Gradual traffic shift: 0% → 10% → 50% → 100%
```

### 7.3 Embedding Version Management

```python
"""
Managing embedding model versions in production.
When you update the embedding model, all vectors must be re-embedded.
"""

class EmbeddingVersionManager:
    """Track and manage embedding model versions."""

    def __init__(self, vector_db_client, metadata_store):
        self.db = vector_db_client
        self.meta = metadata_store

    def start_migration(self, new_model: str, new_dim: int):
        """Begin migration to a new embedding model."""
        # Create new collection with version suffix
        new_collection = f"documents_v{self._next_version()}"
        self.db.create_collection(
            name=new_collection,
            dimension=new_dim,
        )

        self.meta.record_migration(
            status="in_progress",
            source_collection=self._current_collection(),
            target_collection=new_collection,
            new_model=new_model,
        )
        return new_collection

    def migrate_batch(self, batch_ids: list[str], new_embeddings: list):
        """Upsert a batch of re-embedded vectors."""
        target = self.meta.get_active_migration()["target_collection"]
        self.db.upsert(collection=target, ids=batch_ids, vectors=new_embeddings)

    def complete_migration(self):
        """Switch traffic to new collection."""
        migration = self.meta.get_active_migration()
        # Update alias to point to new collection
        self.db.update_alias(
            alias="documents",
            collection=migration["target_collection"],
        )
        self.meta.record_migration(status="completed")

    def rollback_migration(self):
        """Revert to previous collection."""
        migration = self.meta.get_active_migration()
        self.db.update_alias(
            alias="documents",
            collection=migration["source_collection"],
        )
        self.meta.record_migration(status="rolled_back")
```

---

## 8. Integration with Data Pipelines

### 8.1 Batch Embedding Update Pipeline

```python
"""
Airflow DAG for incremental vector index updates.
Runs daily to embed new/updated documents and upsert to vector DB.
"""

from airflow.decorators import dag, task
from datetime import datetime, timedelta

@dag(
    schedule="0 4 * * *",  # 4 AM daily
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["vector", "embeddings"],
    default_args={"retries": 2, "retry_delay": timedelta(minutes=5)},
)
def incremental_vector_update():

    @task()
    def detect_changes(ds=None):
        """Detect documents changed since last successful run."""
        # Query CDC table or data warehouse for changes
        changes = {
            "new_docs": 1200,
            "updated_docs": 350,
            "deleted_doc_ids": ["doc-old-1", "doc-old-2"],
            "source_path": f"s3://lake/gold/documents/dt={ds}/",
        }
        return changes

    @task()
    def generate_embeddings(changes: dict):
        """Generate embeddings for new/updated documents."""
        # Batch embedding via API or local model
        # Chunk large batches to avoid OOM
        batch_size = 256
        total = changes["new_docs"] + changes["updated_docs"]
        n_batches = (total + batch_size - 1) // batch_size

        return {
            "embeddings_path": "s3://embeddings/incremental/2024-06-15/",
            "total_embedded": total,
            "n_batches": n_batches,
            "model": "text-embedding-3-small",
            "model_version": "v2",
        }

    @task()
    def upsert_vectors(embedding_info: dict, changes: dict):
        """Upsert new embeddings and delete removed documents."""
        # Load embeddings from S3
        # Batch upsert to vector DB
        # Delete removed documents
        return {
            "upserted": embedding_info["total_embedded"],
            "deleted": len(changes["deleted_doc_ids"]),
            "collection": "documents",
        }

    @task()
    def validate_index(upsert_result: dict):
        """Run quality checks on updated index."""
        checks = {
            "total_vectors_after": 1_250_000,
            "recall_at_10": 0.96,
            "p99_latency_ms": 4.2,
            "empty_result_rate": 0.002,
        }
        # Alert if recall drops below threshold
        if checks["recall_at_10"] < 0.90:
            raise ValueError(
                f"Recall dropped to {checks['recall_at_10']}"
            )
        return checks

    changes = detect_changes()
    embeddings = generate_embeddings(changes)
    result = upsert_vectors(embeddings, changes)
    validate_index(result)

incremental_vector_update()
```

### 8.2 Streaming Vector Updates with Kafka

```python
"""
Real-time vector updates via Kafka consumer.
For use cases where embedding freshness matters (e.g., news, support tickets).
"""

from confluent_kafka import Consumer, KafkaError
import json

def vector_update_consumer(
    vector_db_client,
    embedding_model,
    kafka_config: dict,
    topic: str = "document-changes",
):
    """Consume document changes and update vector index in real-time."""
    consumer = Consumer({
        "bootstrap.servers": kafka_config["brokers"],
        "group.id": "vector-updater",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    })
    consumer.subscribe([topic])

    batch = []
    batch_size = 64
    flush_interval_seconds = 5

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                if batch:
                    flush_batch(vector_db_client, embedding_model, batch)
                    consumer.commit()
                    batch = []
                continue
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    raise Exception(msg.error())
                continue

            event = json.loads(msg.value())
            batch.append(event)

            if len(batch) >= batch_size:
                flush_batch(vector_db_client, embedding_model, batch)
                consumer.commit()
                batch = []
    finally:
        consumer.close()


def flush_batch(db_client, model, events: list):
    """Process a batch of document change events."""
    upserts = []
    deletes = []

    for event in events:
        if event["op"] in ("insert", "update"):
            embedding = model.encode(event["text"])
            upserts.append({
                "id": event["doc_id"],
                "vector": embedding.tolist(),
                "metadata": event.get("metadata", {}),
            })
        elif event["op"] == "delete":
            deletes.append(event["doc_id"])

    if upserts:
        db_client.upsert(collection="documents", points=upserts)
    if deletes:
        db_client.delete(collection="documents", ids=deletes)
```

### 8.3 CDC to Vector DB Pattern

```
Change Data Capture → Vector DB integration:

  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ PostgreSQL│───→│ Debezium │───→│  Kafka   │───→│ Vector   │
  │ (source)  │    │ (CDC)    │    │ (buffer) │    │ Updater  │
  └──────────┘    └──────────┘    └──────────┘    └────┬─────┘
                                                       │
                                                       ▼
                                                  ┌──────────┐
                                                  │ Embedding │
                                                  │ Model     │
                                                  └────┬─────┘
                                                       │
                                                       ▼
                                                  ┌──────────┐
                                                  │ Vector DB │
                                                  │ (Qdrant)  │
                                                  └──────────┘

  This pattern ensures vector index stays in sync with source data:
  - Debezium captures every INSERT/UPDATE/DELETE from PostgreSQL WAL
  - Kafka buffers events for reliability
  - Vector Updater embeds text fields and upserts/deletes in vector DB
  - End-to-end latency: typically 1-5 seconds
```

---

## Summary

```
Key takeaways:

1. Hybrid search (dense + sparse) outperforms either alone
   — use RRF or weighted linear combination for fusion

2. Metadata filtering strategy matters: pre-filter for large
   partitions, post-filter for selective queries, or use
   database-native hybrid approaches

3. Cross-encoder reranking improves precision by 10-20%
   but adds ~50ms latency — use for top-K refinement only

4. Scale with scatter-gather sharding + read replicas
   — plan capacity using memory estimation formulas

5. Monitor recall, latency percentiles, and empty result rates
   — recall degradation is the silent killer of search quality

6. Optimize costs progressively: int8 quantization → dimension
   reduction → tiered storage → PQ (as needed)

7. Integrate vector search into data pipelines via batch DAGs
   (Airflow) or streaming consumers (Kafka CDC)
```

---

## Practice Exercises

1. **Hybrid Search Fusion**: Implement both RRF and linear combination fusion. Generate 10 queries where one method outperforms the other.

2. **Filter Benchmark**: Create a dataset with rich metadata. Measure search latency and recall with pre-filtering vs post-filtering at different selectivity levels.

3. **Reranking Pipeline**: Build a two-stage retrieval pipeline using a bi-encoder for retrieval and a cross-encoder for reranking. Measure MRR improvement.

4. **Capacity Planner**: Extend the capacity planning calculator to include cost estimation for AWS, GCP, and Azure instances.

5. **Monitoring Dashboard**: Implement Prometheus metrics for a vector search service and create Grafana dashboard JSON.

---

## Further Reading

- [Reciprocal Rank Fusion paper (Cormack et al., 2009)](https://dl.acm.org/doi/10.1145/1571941.1572114)
- [SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720)
- [Cross-Encoders for Reranking](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- [Qdrant Monitoring Guide](https://qdrant.tech/documentation/guides/monitoring/)
- [Milvus Capacity Planning](https://milvus.io/docs/sizing.md)
- [Vector Database Benchmarks (ANN-Benchmarks)](https://ann-benchmarks.com/)

[← Previous: 22. Vector Storage and Indexing](22_Vector_Storage_and_Indexing.md) | [Next: Overview →](00_Overview.md)
