[← Previous: 21. Data Versioning and Data Contracts](21_Data_Versioning_and_Contracts.md) | [Next: 23. Production Vector Search →](23_Production_Vector_Search.md)

# 22. Vector Storage and Indexing

## Learning Objectives

1. Understand vector storage architectures and tradeoffs between in-memory, disk-based, and distributed approaches
2. Master FAISS index types (Flat, IVF, HNSW, PQ) and composite index construction via factory strings
3. Describe Milvus's distributed architecture including proxy, query, data, and index nodes
4. Explain Weaviate's module system, vectorizer pipeline, and GraphQL API design
5. Use Pinecone's serverless model and Qdrant's advanced metadata filtering
6. Deploy Chroma in embedded and client-server modes for prototyping and production
7. Compare vector databases across performance, scalability, cost, and ecosystem dimensions

---

## Theory & Principles

Vector search looks deceptively simple — "find the K nearest vectors to my query" — until you scale past a million vectors and discover that brute-force comparison is impossibly slow. Every interesting vector database decision is downstream of one fundamental tension: **exact** nearest-neighbor search is provably hard in high dimensions (the "curse of dimensionality"), so production systems use **approximate** nearest-neighbor (ANN) algorithms that trade a tiny bit of recall for orders of magnitude in speed.

- **(A) The curse of dimensionality and why brute force breaks down** — what makes high-dimensional search hard.
- **(B) Distance metrics: cosine, Euclidean, dot product** — what each measures and when each is right.
- **(C) The major ANN algorithm families** — IVF (inverted file), HNSW (hierarchical small-world graphs), PQ (product quantization).
- **(D) The recall-vs-speed-vs-memory triangle** — every index lives somewhere on this surface.
- **(E) Storage architecture for vectors at scale** — partitioning, sharding, the embedding-pipeline as a data engineering problem.

### A. The Curse of Dimensionality

In low dimensions, nearest-neighbor search is straightforward — kd-trees give you O(log N). In high dimensions (D > 30 or so), kd-trees degenerate: nearly every point becomes "approximately equidistant" from the query, and tree pruning fails.

For a typical embedding (D = 768 from BERT, D = 1536 from OpenAI), brute-force comparison is O(N × D) per query. At N = 100M and D = 1536, this is 150 billion floating-point multiplications per query — seconds even on a GPU.

This is not solvable by faster hardware. The math fights back. The solution is to give up on exact answers.

### B. Distance Metrics

What does "nearest" mean?

#### B.1 Euclidean (L2)

`d(a, b) = sqrt(sum((a_i - b_i)^2))`. Geometric distance. Sensitive to vector magnitude.

Use when: vectors are in absolute coordinates (e.g., physical sensor data, raw image pixels).

#### B.2 Cosine similarity

`cos(a, b) = (a · b) / (|a| |b|)`. Measures angle between vectors, ignoring magnitude.

Use when: vectors represent direction in a semantic space (text embeddings, image embeddings). The dominant choice for ML embeddings because two semantically similar texts should be close regardless of length.

#### B.3 Dot product (inner product)

`a · b = sum(a_i × b_i)`. Equivalent to cosine if vectors are normalized to unit length.

Use when: vectors are normalized (most modern embeddings are). Faster than cosine because no magnitude division — and most ANN libraries optimize for it.

#### B.4 The pragmatic rule

For modern text/image embeddings, vectors are usually L2-normalized at production time. Then dot product = cosine similarity, both computed faster than Euclidean. This is what FAISS, Milvus, Pinecone all default to.

### C. ANN Algorithm Families

Three dominant families, three different design philosophies.

#### C.1 IVF (Inverted File Index)

The clustering approach.

1. **Training:** k-means cluster all N vectors into K centroids (K typically 1000-10000).
2. **Indexing:** assign each vector to its nearest centroid; bucket them.
3. **Query:** find the closest M centroids to the query; only search those buckets.

Speed: O(K + M × N/K). Trade-off: increasing M improves recall but slows queries. M=1 might give 70% recall; M=20 might give 95%.

Strengths: simple, fast to build, works well at moderate scale (millions to ~100M vectors). Standard in FAISS as `IndexIVFFlat`.

#### C.2 HNSW (Hierarchical Navigable Small World)

The graph approach.

1. **Indexing:** build a multi-layer graph. Top layer is sparse (few long-range edges); bottom layer is dense (many short edges). Insert each vector with edges to its nearest neighbors at each layer.
2. **Query:** start at the top layer, greedy-walk toward the query (always step to the neighbor closest to the query). Drop down a layer; repeat. At the bottom layer, the K nearest neighbors are the result.

Speed: O(log N) per query. Recall: typically 95-99% with reasonable parameters.

Strengths: best-in-class recall/speed trade-off for moderate-scale workloads. Standard in pgvector, Qdrant, Weaviate, and most modern vector DBs.

Weaknesses: high memory (graph edges); slow to build; updates are tricky.

#### C.3 PQ (Product Quantization)

The compression approach. Not a search algorithm per se, but a way to make IVF or HNSW use less memory.

1. Split each D-dim vector into M sub-vectors of D/M dims each.
2. K-means cluster each sub-space independently into 256 centroids.
3. Store each vector as M bytes (one centroid index per sub-space) instead of D × 4 bytes.

Compression: 32x to 64x. Distance computation: approximate, via cluster-to-cluster lookup tables.

Strengths: enables billion-scale vector search on a single machine. Standard in FAISS as `IndexIVFPQ`.

Weakness: lossy. Recall drops slightly compared to uncompressed.

#### C.4 Combinations

Production systems combine these. `IndexIVFPQ` (FAISS): IVF for partitioning + PQ for compression. Pinecone: HNSW + PQ-style compression. Each tool's index choices are tunings on this design space.

### D. The Recall-Speed-Memory Triangle

Every ANN index lives somewhere on a 3D surface:

- **Recall:** what fraction of the true K-NN does the index return? 100% = exact; lower = approximate.
- **Query speed:** queries per second per node.
- **Memory:** RAM needed to hold the index.

Increasing one usually costs another. `IndexFlat` (brute force) is 100% recall but slow. `IndexHNSWFlat` is fast and ~98% recall but uses 2x memory of raw vectors. `IndexIVFPQ` uses 30x less memory but recall drops to 90% and queries are slower than HNSW.

Picking an index = picking your point on this triangle for your workload. If 95% recall is acceptable and memory is constrained, IVFPQ. If recall is critical and memory is plentiful, HNSW. If query latency is the priority, HNSW with high `ef` parameter.

### E. Storage Architecture for Vectors at Scale

The vector index is just one component; getting vectors *into* and *out of* the system is a data engineering problem.

#### E.1 The embedding pipeline

1. Source content (documents, images, products) lives in a primary store (DB, S3).
2. An embedding service (OpenAI API, local model on GPU) computes vectors.
3. Vectors are written to the vector DB along with metadata (the source ID, schema fields for filtering).
4. On change (new content, content updated), the embedding is recomputed and the vector DB is updated.

This is a CDC + transform + load pipeline (Lessons 18, 12). The vector DB is the sink. Failure modes: stale embeddings (source updated but vector not refreshed), missing embeddings (new content not indexed yet), schema drift between source and vector DB metadata.

#### E.2 Partitioning and sharding

- **Per-tenant partitioning.** Each customer / namespace has its own vector collection. Search is scoped to one partition. Common for multi-tenant SaaS.
- **By-key sharding.** A single logical collection is sharded by key across nodes. Each query scatters to all shards, gathers results. Standard for billion-scale.
- **Hot-cold tiering.** Recent vectors in fast HNSW; older vectors in compressed IVFPQ; archive in object storage. Query accumulates from each tier.

#### E.3 The metadata-filter problem

A common query: "find similar products, but only those in Category=X and Price<100." The naive approach (search top 1000, filter) breaks when filters are selective (out of 1000 results, only 5 match the filter). Solutions:

- **Pre-filter:** apply metadata filter first, then search only matching vectors. Requires metadata index.
- **Post-filter with overshoot:** search top 10000, filter, return top 10. Wastes compute when filter is selective.
- **Filtered-aware index:** indexes that support filters at the graph traversal level (Weaviate's filtered HNSW, Qdrant's filterable HNSW). Best performance when filters are common.

This is one of the deepest design issues in production vector search.

### From Theory to the Code Below

Each section that follows operationalizes one piece of this framework:

- §1 (Vector Search Fundamentals) is §A and §B — distance metrics and the dimensionality problem.
- §2 (FAISS) is §C — the foundational library implementing IVF, HNSW, PQ.
- §3 (Vector Databases) is §E — production systems built around these algorithms.
- §4 (Index Selection) is §D — the recall-speed-memory triangle in practice.
- §5 (Filtering and Hybrid Queries) is §E.3 — pre-filter, post-filter, filtered indexes.
- §6 (Benchmark Comparison) gives concrete recall/QPS/memory numbers across tools.

---

## Overview

Modern data pipelines increasingly deal with high-dimensional vector data — embeddings from language models, image encoders, recommendation systems, and scientific simulations. Storing and searching these vectors efficiently is a data engineering problem that sits at the intersection of indexing theory, distributed systems, and hardware optimization.

This lesson covers the storage layer: how vectors are persisted, indexed, and queried across the major tools in the ecosystem. We start with the foundational library (FAISS), then examine purpose-built vector databases (Milvus, Weaviate, Pinecone, Qdrant, Chroma), and conclude with a benchmark comparison to guide technology selection.

> **Data Engineering perspective**: Vector storage is not just an ML concern. Data engineers must design embedding pipelines, manage index lifecycle, handle schema evolution for metadata, and integrate vector stores into the broader data platform alongside warehouses, lakes, and streaming systems.

---

## 1. Vector Storage Architectures

### 1.1 Storage Model Taxonomy

```
Vector Storage Models:

┌──────────────────────────────────────────────────────────────────────┐
│                        In-Memory                                     │
│  ┌──────────────────────────────────────────────────┐               │
│  │ All vectors + index in RAM                        │               │
│  │ ✓ Lowest latency (~1ms p99)                       │               │
│  │ ✓ Simplest architecture                           │               │
│  │ ✗ Cost: $7-12/GB/month (cloud RAM)                │               │
│  │ ✗ Data loss on crash without WAL                   │               │
│  │ Examples: FAISS (default), Qdrant (default)       │               │
│  └──────────────────────────────────────────────────┘               │
│                                                                      │
│                        Disk-Based                                    │
│  ┌──────────────────────────────────────────────────┐               │
│  │ Index in RAM, vectors on disk (mmap)              │               │
│  │ ✓ 10-50x cheaper storage                          │               │
│  │ ✓ Handles datasets > RAM                          │               │
│  │ ✗ Higher latency (~5-20ms p99)                    │               │
│  │ ✗ Performance depends on OS page cache            │               │
│  │ Examples: FAISS OnDisk, Qdrant mmap, Weaviate     │               │
│  └──────────────────────────────────────────────────┘               │
│                                                                      │
│                        Distributed                                   │
│  ┌──────────────────────────────────────────────────┐               │
│  │ Sharded across nodes, replicated for HA           │               │
│  │ ✓ Scales to billions of vectors                   │               │
│  │ ✓ Fault tolerant                                  │               │
│  │ ✗ Network latency overhead                        │               │
│  │ ✗ Operational complexity                          │               │
│  │ Examples: Milvus, Pinecone, Weaviate cluster      │               │
│  └──────────────────────────────────────────────────┘               │
└──────────────────────────────────────────────────────────────────────┘
```

### 1.2 Memory Estimation

```
Memory formula for vector data:

  Raw vector memory = N × d × bytes_per_element

  Where:
    N = number of vectors
    d = embedding dimension
    bytes_per_element depends on format:
      float32 = 4 bytes
      float16 = 2 bytes
      int8 (SQ8) = 1 byte
      PQ (m subquantizers) = m bytes

  Example: 10 million vectors, dimension 768

  Format        Vector Memory    Index Overhead (HNSW M=16)    Total
  ────────────────────────────────────────────────────────────────────
  float32       28.7 GB          ~2.4 GB                       ~31.1 GB
  float16       14.3 GB          ~2.4 GB                       ~16.7 GB
  int8 (SQ8)    7.2 GB           ~2.4 GB                       ~9.6 GB
  PQ (m=96)     0.9 GB           ~2.4 GB                       ~3.3 GB

  HNSW graph overhead ≈ N × M × 2 × 8 bytes
  With M=16: 10M × 16 × 2 × 8 = 2.4 GB (graph structure only)
```

### 1.3 Memory-Mapped Storage

```python
"""
Memory-mapped vector storage enables datasets larger than RAM.
The OS manages which pages reside in physical memory.
"""

import numpy as np
import os

def create_mmap_vectors(path: str, n_vectors: int, dim: int) -> np.ndarray:
    """Create a memory-mapped vector file."""
    fp = np.memmap(path, dtype='float32', mode='w+', shape=(n_vectors, dim))
    return fp

def load_mmap_vectors(path: str, n_vectors: int, dim: int) -> np.ndarray:
    """Load existing memory-mapped vectors (no RAM copy)."""
    fp = np.memmap(path, dtype='float32', mode='r', shape=(n_vectors, dim))
    return fp

# Usage pattern for large datasets
n_vectors = 10_000_000
dim = 768
path = "/data/vectors.mmap"

# Write phase (batch ETL)
vectors = create_mmap_vectors(path, n_vectors, dim)
batch_size = 100_000
for i in range(0, n_vectors, batch_size):
    end = min(i + batch_size, n_vectors)
    vectors[i:end] = np.random.randn(end - i, dim).astype('float32')
    vectors.flush()  # persist to disk

# Read phase (search service)
vectors = load_mmap_vectors(path, n_vectors, dim)
query = np.random.randn(1, dim).astype('float32')

# OS loads only the pages touched during search
# If RAM is 8 GB and vectors are 28.7 GB, only ~28% is in memory at once
```

### 1.4 Write-Ahead Log (WAL) for Durability

```
WAL ensures no data loss on crash:

  Write request
       │
       ▼
  ┌─────────────┐    ① Append to WAL (sequential I/O, fast)
  │   WAL File   │──────────────────────────────────────────┐
  └──────┬──────┘                                           │
         │ ② ACK to client                                  │
         ▼                                                  │
  ┌─────────────┐    ③ Batch flush to segment (background)  │
  │  In-Memory   │◄─────────────────────────────────────────┘
  │  Buffer      │
  └──────┬──────┘
         │ ④ When buffer full → create sealed segment
         ▼
  ┌─────────────┐
  │  Sealed      │    ⑤ Build index on sealed segment
  │  Segment     │
  └─────────────┘

  On crash: replay WAL entries not yet flushed to segments
  WAL is truncated after successful flush

  Databases using WAL: Milvus, Qdrant, Weaviate
  Libraries without WAL: FAISS (user must handle persistence)
```

---

## 2. FAISS Deep Dive

### 2.1 FAISS in the Ecosystem

```
FAISS position:

  Application Layer:    LangChain, LlamaIndex, custom apps
         │
  Vector DB Layer:      Milvus, Weaviate, Qdrant, Chroma
         │
  Search Engine Layer:  FAISS, ScaNN, Annoy, hnswlib   ← FAISS lives here
         │
  Hardware Layer:       CPU (AVX2/AVX-512), GPU (CUDA)

FAISS is a library, not a database:
  ✓ No server process, no network protocol
  ✓ C++ core with Python bindings
  ✓ Composable index types via factory strings
  ✓ GPU acceleration for training and search
  ✗ No built-in persistence (user saves/loads)
  ✗ No metadata filtering (vectors only)
  ✗ No replication or sharding
```

### 2.2 Core Index Types

```python
import faiss
import numpy as np

dim = 768
n_vectors = 1_000_000
n_query = 100
k = 10

# Generate sample data
xb = np.random.randn(n_vectors, dim).astype('float32')
xq = np.random.randn(n_query, dim).astype('float32')

# ─── IndexFlatL2: Exact brute-force search ───
index_flat = faiss.IndexFlatL2(dim)
index_flat.add(xb)
distances, indices = index_flat.search(xq, k)
# Time: O(N × d) per query → ~200ms for 1M vectors
# Use: ground truth, small datasets (<100K)

# ─── IndexIVFFlat: Inverted file with coarse quantizer ───
nlist = 1024  # number of Voronoi cells
quantizer = faiss.IndexFlatL2(dim)
index_ivf = faiss.IndexIVFFlat(quantizer, dim, nlist)
index_ivf.train(xb)  # learn cluster centroids
index_ivf.add(xb)
index_ivf.nprobe = 32  # search 32 of 1024 cells
distances, indices = index_ivf.search(xq, k)
# Time: O(nprobe/nlist × N × d) → ~6ms
# Recall: ~95% with nprobe=32

# ─── IndexHNSWFlat: Hierarchical Navigable Small World ───
M = 32  # connections per node
index_hnsw = faiss.IndexHNSWFlat(dim, M)
index_hnsw.hnsw.efConstruction = 200  # build quality
index_hnsw.hnsw.efSearch = 64         # search quality
index_hnsw.add(xb)
distances, indices = index_hnsw.search(xq, k)
# Time: ~2ms  |  Recall: ~99%
# Tradeoff: high memory (graph structure)
```

### 2.3 Product Quantization (PQ)

```python
# ─── IndexPQ: Compress vectors for memory efficiency ───
m = 96      # number of subquantizers (must divide dim)
nbits = 8   # bits per subquantizer (256 centroids each)
index_pq = faiss.IndexPQ(dim, m, nbits)
index_pq.train(xb)
index_pq.add(xb)
distances, indices = index_pq.search(xq, k)
# Memory: 96 bytes per vector (vs 3072 for float32)
# Compression ratio: 32x
# Recall: ~85-92% depending on data distribution

# ─── IndexIVFPQ: IVF + PQ (the workhorse for large-scale) ───
nlist = 4096
m = 96
nbits = 8
quantizer = faiss.IndexFlatL2(dim)
index_ivfpq = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
index_ivfpq.train(xb)
index_ivfpq.add(xb)
index_ivfpq.nprobe = 64
distances, indices = index_ivfpq.search(xq, k)
# Memory: ~100 bytes/vector  |  Speed: ~1ms  |  Recall: ~90%
```

### 2.4 Composite Indexes via Factory Strings

```python
"""
FAISS factory strings compose index types declaratively:

  "Flat"           → IndexFlatL2
  "IVF1024,Flat"   → IndexIVFFlat with 1024 cells
  "IVF4096,PQ96"   → IndexIVFPQ with 4096 cells, 96 subquantizers
  "HNSW32"         → IndexHNSWFlat with M=32
  "IVF1024,HNSW32" → IVF with HNSW quantizer (fast coarse search)
  "OPQ96,IVF4096,PQ96" → OPQ rotation + IVF + PQ (best compression)
  "IVF4096,SQ8"    → IVF with scalar quantization (int8)
"""

# Build complex index with one line
index = faiss.index_factory(dim, "OPQ96,IVF4096,PQ96")
index.train(xb)
index.add(xb)

# GPU training (much faster for large datasets)
# res = faiss.StandardGpuResources()
# index_gpu = faiss.index_cpu_to_gpu(res, 0, index)

# Save and load
faiss.write_index(index, "/data/faiss_index.bin")
index_loaded = faiss.read_index("/data/faiss_index.bin")
```

### 2.5 FAISS Index Selection Guide

```
Decision tree for FAISS index selection:

  Dataset size?
  │
  ├── < 100K vectors
  │   └── IndexFlatL2 (exact, no training needed)
  │
  ├── 100K - 1M vectors
  │   ├── Memory OK? → IndexHNSWFlat (best recall)
  │   └── Memory tight? → IndexIVFFlat (nlist=sqrt(N))
  │
  ├── 1M - 100M vectors
  │   ├── Latency priority? → IVF + HNSW quantizer
  │   ├── Memory priority? → IVF + PQ (or OPQ + IVF + PQ)
  │   └── Balanced? → IVF + SQ8
  │
  └── > 100M vectors
      ├── Single machine? → OPQ + IVF + PQ + disk I/O
      └── Multi-GPU? → Sharded IndexIVFPQ across GPUs

  Training data requirement:
    IVF: 30 × nlist to 256 × nlist vectors
    PQ: 10K-100K representative vectors
    OPQ: same as PQ (learns rotation matrix)
```

---

## 3. Milvus Architecture

### 3.1 Distributed Components

```
Milvus Distributed Architecture:

  ┌─────────────────────────────────────────────────────┐
  │                    Clients                           │
  │  (Python SDK, Java SDK, Go SDK, REST, gRPC)         │
  └──────────────────────┬──────────────────────────────┘
                         │
  ┌──────────────────────▼──────────────────────────────┐
  │                  Proxy Layer                          │
  │  (Load balancing, request routing, auth)             │
  │  [Proxy 1] [Proxy 2] [Proxy 3]                      │
  └──────────────────────┬──────────────────────────────┘
                         │
  ┌──────────────────────▼──────────────────────────────┐
  │              Coordinator Layer                        │
  │  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
  │  │Root Coord│ │Query     │ │Data Coord│            │
  │  │(DDL, TSO)│ │Coord     │ │(segments)│            │
  │  └──────────┘ └──────────┘ └──────────┘            │
  │                    ┌──────────┐                      │
  │                    │Index     │                      │
  │                    │Coord     │                      │
  │                    └──────────┘                      │
  └──────────────────────┬──────────────────────────────┘
                         │
  ┌──────────────────────▼──────────────────────────────┐
  │                Worker Layer                           │
  │  [Query Node 1] [Query Node 2] (search execution)   │
  │  [Data Node 1]  [Data Node 2]  (write/flush)        │
  │  [Index Node 1] [Index Node 2] (index building)     │
  └──────────────────────┬──────────────────────────────┘
                         │
  ┌──────────────────────▼──────────────────────────────┐
  │              Storage Layer                            │
  │  ┌──────────┐ ┌───────────┐ ┌──────────┐           │
  │  │ etcd     │ │ MinIO/S3  │ │ Pulsar/  │           │
  │  │(metadata)│ │(segments) │ │ Kafka    │           │
  │  └──────────┘ └───────────┘ └──────────┘           │
  └─────────────────────────────────────────────────────┘
```

### 3.2 Collections, Schemas, and Partitions

```python
from pymilvus import (
    connections, Collection, FieldSchema, CollectionSchema, DataType, utility
)

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Define schema with typed fields
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="published_year", dtype=DataType.INT32),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
]

schema = CollectionSchema(fields, description="Document embeddings")
collection = Collection("documents", schema)

# Create partition for data isolation
collection.create_partition("technical_docs")
collection.create_partition("legal_docs")

# Create index on vector field
index_params = {
    "metric_type": "L2",
    "index_type": "IVF_SQ8",      # IVF with scalar quantization
    "params": {"nlist": 2048},
}
collection.create_index("embedding", index_params)

# Load collection into memory for search
collection.load()

# Insert data
import numpy as np
data = [
    ["Doc A", "Doc B", "Doc C"],               # title
    ["technical", "legal", "technical"],        # category
    [2024, 2023, 2024],                         # published_year
    np.random.randn(3, 768).tolist(),           # embedding
]
collection.insert(data, partition_name="technical_docs")

# Search with metadata filter
search_params = {"metric_type": "L2", "params": {"nprobe": 64}}
results = collection.search(
    data=[np.random.randn(768).tolist()],
    anns_field="embedding",
    param=search_params,
    limit=10,
    expr='category == "technical" and published_year >= 2024',
    output_fields=["title", "category"],
)
```

### 3.3 Consistency Levels

```
Milvus consistency levels:

  Level          Guarantee                          Use Case
  ──────────────────────────────────────────────────────────────
  Strong         Read sees all prior writes         Financial data, exact counts
  Bounded        Read sees writes within T seconds  Analytics (T=5s acceptable)
  Session        Read-your-writes within session    Interactive applications
  Eventually     No ordering guarantee              Batch search, recommendations

  Setting per search:
    results = collection.search(..., consistency_level="Session")

  Default: Bounded Staleness (good balance for most pipelines)
```

---

## 4. Weaviate

### 4.1 Module Architecture

```
Weaviate Module System:

  ┌───────────────────────────────────────────┐
  │            Weaviate Core                   │
  │  (HNSW index, inverted index, GraphQL)    │
  └─────────────────┬─────────────────────────┘
                    │
  ┌─────────────────▼─────────────────────────┐
  │              Module Slots                  │
  │                                            │
  │  Vectorizer Modules:                       │
  │  ├── text2vec-openai (OpenAI embeddings)   │
  │  ├── text2vec-cohere (Cohere embeddings)   │
  │  ├── text2vec-huggingface (local models)   │
  │  ├── img2vec-neural (image embeddings)     │
  │  └── multi2vec-clip (multimodal)           │
  │                                            │
  │  Generative Modules:                       │
  │  ├── generative-openai (GPT generation)    │
  │  ├── generative-cohere                     │
  │  └── generative-anthropic                  │
  │                                            │
  │  Reranker Modules:                         │
  │  ├── reranker-cohere                       │
  │  └── reranker-transformers                 │
  └────────────────────────────────────────────┘
```

### 4.2 Schema and GraphQL API

```python
import weaviate

client = weaviate.Client("http://localhost:8080")

# Define class (collection equivalent)
class_obj = {
    "class": "Article",
    "description": "Technical articles with embeddings",
    "vectorizer": "text2vec-openai",
    "moduleConfig": {
        "text2vec-openai": {
            "model": "text-embedding-3-small",
            "dimensions": 768,
        }
    },
    "properties": [
        {
            "name": "title",
            "dataType": ["text"],
            "moduleConfig": {
                "text2vec-openai": {"skip": False}  # include in vectorization
            }
        },
        {
            "name": "content",
            "dataType": ["text"],
        },
        {
            "name": "category",
            "dataType": ["text"],
            "indexFilterable": True,   # enable filtered search
            "indexSearchable": True,   # enable BM25 search
        },
        {
            "name": "publishedYear",
            "dataType": ["int"],
            "indexFilterable": True,
        },
    ],
}
client.schema.create_class(class_obj)

# Add objects (Weaviate vectorizes automatically)
client.data_object.create(
    class_name="Article",
    data_object={
        "title": "Introduction to Vector Databases",
        "content": "Vector databases store high-dimensional embeddings...",
        "category": "databases",
        "publishedYear": 2024,
    }
)
```

### 4.3 Weaviate GraphQL Queries

```graphql
# Near-text semantic search (Weaviate vectorizes the query)
{
  Get {
    Article(
      nearText: { concepts: ["distributed vector indexing"] }
      where: {
        operator: And
        operands: [
          { path: ["category"], operator: Equal, valueText: "databases" }
          { path: ["publishedYear"], operator: GreaterThan, valueInt: 2023 }
        ]
      }
      limit: 10
    ) {
      title
      content
      category
      _additional {
        distance
        certainty
        id
      }
    }
  }
}

# Hybrid search (BM25 + vector, alpha controls weighting)
{
  Get {
    Article(
      hybrid: { query: "HNSW index performance", alpha: 0.7 }
      limit: 10
    ) {
      title
      _additional { score }
    }
  }
}
```

---

## 5. Pinecone and Qdrant

### 5.1 Pinecone Serverless

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="your-api-key")

# Create serverless index (no infrastructure management)
pc.create_index(
    name="articles",
    dimension=768,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

index = pc.Index("articles")

# Upsert with metadata
vectors = [
    {
        "id": "doc-001",
        "values": [0.1, 0.2, ...],  # 768-dim embedding
        "metadata": {
            "title": "Vector Indexing Guide",
            "category": "technical",
            "year": 2024,
            "tags": ["vectors", "indexing", "HNSW"],
        }
    },
    # ... more vectors
]
index.upsert(vectors=vectors, namespace="technical")

# Query with metadata filter
results = index.query(
    vector=[0.15, 0.22, ...],
    top_k=10,
    namespace="technical",
    filter={
        "$and": [
            {"category": {"$eq": "technical"}},
            {"year": {"$gte": 2024}},
            {"tags": {"$in": ["vectors"]}},
        ]
    },
    include_metadata=True,
)

# Sparse-dense hybrid search (Pinecone native)
results = index.query(
    vector=[0.15, 0.22, ...],               # dense
    sparse_vector={                           # sparse (BM25-like)
        "indices": [102, 4501, 9832],
        "values": [0.8, 0.4, 0.6],
    },
    top_k=10,
)
```

### 5.2 Qdrant Advanced Filtering

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, Range,
    SearchParams, QuantizationSearchParams
)

client = QdrantClient(host="localhost", port=6333)

# Create collection with quantization config
client.create_collection(
    collection_name="articles",
    vectors_config=VectorParams(
        size=768,
        distance=Distance.COSINE,
        on_disk=True,  # mmap storage for large datasets
    ),
    quantization_config={
        "scalar": {
            "type": "int8",
            "quantile": 0.99,
            "always_ram": True,  # keep quantized vectors in RAM
        }
    },
)

# Upsert with rich payload
client.upsert(
    collection_name="articles",
    points=[
        PointStruct(
            id=1,
            vector=[0.1, 0.2, ...],
            payload={
                "title": "Vector Indexing Guide",
                "category": "technical",
                "year": 2024,
                "author": {"name": "Alice", "org": "DataCo"},
                "tags": ["vectors", "indexing"],
            }
        ),
    ]
)

# Advanced filtering with must/should/must_not
results = client.search(
    collection_name="articles",
    query_vector=[0.15, 0.22, ...],
    query_filter=Filter(
        must=[
            FieldCondition(key="category", match=MatchValue(value="technical")),
            FieldCondition(key="year", range=Range(gte=2023)),
        ],
        must_not=[
            FieldCondition(key="tags", match=MatchValue(value="deprecated")),
        ],
        should=[  # at least one should match (boosts score)
            FieldCondition(key="author.org", match=MatchValue(value="DataCo")),
        ],
    ),
    search_params=SearchParams(
        hnsw_ef=128,
        quantization=QuantizationSearchParams(
            rescore=True,       # re-score with original vectors
            oversampling=2.0,   # fetch 2x candidates before rescore
        ),
    ),
    limit=10,
)
```

### 5.3 Pinecone vs Qdrant Comparison

```
Feature               Pinecone                    Qdrant
────────────────────────────────────────────────────────────────
Hosting               Fully managed (SaaS)        Self-hosted or Cloud
Language              Proprietary                 Rust (open-source)
Scaling               Automatic serverless        Manual sharding + replicas
Metadata filter       JSON filter syntax          must/should/must_not
Hybrid search         Native sparse-dense         BM25 + dense (built-in)
Quantization          Automatic                   Scalar, PQ (configurable)
Pricing               Pay per read/write unit     Free (self-hosted) or cloud
Cold start            Scales to zero              Always running
Disk mode             N/A (managed)               mmap support
Multi-tenancy         Namespaces                  Collection + payload filter
Max dimension         20,000                      65,535
```

---

## 6. Chroma

### 6.1 Embedded Mode (Prototyping)

```python
import chromadb

# Embedded mode — runs in-process, persists to local directory
client = chromadb.PersistentClient(path="/data/chroma_db")

# Create collection with custom embedding function
from chromadb.utils import embedding_functions

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="your-key",
    model_name="text-embedding-3-small"
)

collection = client.get_or_create_collection(
    name="articles",
    embedding_function=openai_ef,
    metadata={"hnsw:space": "cosine"},  # distance metric
)

# Add documents (Chroma auto-generates embeddings)
collection.add(
    documents=[
        "Introduction to vector databases and similarity search",
        "Advanced indexing techniques for high-dimensional data",
        "Production deployment patterns for vector search systems",
    ],
    metadatas=[
        {"category": "intro", "year": 2024},
        {"category": "advanced", "year": 2024},
        {"category": "production", "year": 2025},
    ],
    ids=["doc-001", "doc-002", "doc-003"],
)

# Query by text (auto-embedded)
results = collection.query(
    query_texts=["how to scale vector search"],
    n_results=5,
    where={"year": {"$gte": 2024}},
    include=["documents", "metadatas", "distances"],
)

# Query by embedding directly
results = collection.query(
    query_embeddings=[[0.1, 0.2, ...]],
    n_results=5,
)
```

### 6.2 Client-Server Mode (Production)

```python
# Server: run as separate process
# chroma run --host 0.0.0.0 --port 8000 --path /data/chroma_db

# Client: connect via HTTP
client = chromadb.HttpClient(host="chroma-server", port=8000)

# Same API as embedded mode
collection = client.get_or_create_collection("articles")
collection.add(documents=["..."], ids=["doc-004"])
```

```
Chroma deployment modes:

  Embedded (development):
  ┌──────────────────────────┐
  │ Application Process       │
  │  ├── App Code             │
  │  └── Chroma Library       │
  │       └── SQLite + HNSW   │
  │            └── /data/     │
  └──────────────────────────┘

  Client-Server (production):
  ┌──────────────┐    HTTP    ┌──────────────────┐
  │ App Process   │──────────→│ Chroma Server     │
  │ (thin client) │           │  ├── HNSW index   │
  └──────────────┘            │  ├── SQLite meta   │
                              │  └── /data/        │
                              └──────────────────┘

  Distributed (Chroma Cloud / upcoming):
  ┌──────────────┐    gRPC    ┌──────────────────┐
  │ App Process   │──────────→│ Coordinator       │
  └──────────────┘            │  ├── Shard 1       │
                              │  ├── Shard 2       │
                              │  └── Shard 3       │
                              └──────────────────┘
```

---

## 7. Storage Persistence Patterns

### 7.1 Index Lifecycle in Data Pipelines

```
Embedding Pipeline Integration:

  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ Raw Data  │───→│ Embedding│───→│ Index    │───→│ Serving  │
  │ (S3/Lake) │    │ Service  │    │ Builder  │    │ Layer    │
  └──────────┘    └──────────┘    └──────────┘    └──────────┘
       │               │               │               │
  Batch: hourly    GPU cluster     FAISS train     Load balanced
  Stream: real-    or API calls    + quantize      query nodes
  time CDC                        + serialize

  Persistence strategy by tool:

  Tool        Primary Storage    Backup Strategy
  ─────────────────────────────────────────────────────
  FAISS       faiss.write_index  S3/GCS upload + version tag
  Milvus      MinIO/S3 segments  Built-in snapshot API
  Weaviate    Disk (LSMT)        Backup API → S3
  Qdrant      Disk (segments)    Snapshot API → S3
  Pinecone    Managed            Managed (SLA-backed)
  Chroma      SQLite + files     File-level backup
```

### 7.2 Snapshot and Backup

```python
"""
Qdrant snapshot management for production deployments.
"""
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

# Create snapshot (consistent point-in-time)
snapshot = client.create_snapshot(collection_name="articles")
# Returns: SnapshotDescription(name='articles-2024-06-15-12-00-00.snapshot')

# List snapshots
snapshots = client.list_snapshots(collection_name="articles")

# Download snapshot for off-site backup
client.download_snapshot(
    collection_name="articles",
    snapshot_name=snapshot.name,
    path="/backups/articles-latest.snapshot",
)

# Restore from snapshot (disaster recovery)
# client.recover_snapshot(
#     collection_name="articles",
#     location="/backups/articles-latest.snapshot",
# )
```

```python
"""
FAISS index versioning for data pipelines.
"""
import faiss
import json
from datetime import datetime
from pathlib import Path

def save_versioned_index(
    index: faiss.Index,
    base_path: str,
    metadata: dict,
) -> str:
    """Save FAISS index with version metadata."""
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(base_path) / version

    path.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path / "index.faiss"))

    metadata["version"] = version
    metadata["ntotal"] = index.ntotal
    metadata["d"] = index.d
    metadata["timestamp"] = datetime.now().isoformat()

    with open(path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Update symlink for latest
    latest = Path(base_path) / "latest"
    if latest.exists():
        latest.unlink()
    latest.symlink_to(path)

    return version

def load_latest_index(base_path: str) -> tuple:
    """Load the latest versioned index."""
    path = Path(base_path) / "latest"
    index = faiss.read_index(str(path / "index.faiss"))
    with open(path / "metadata.json") as f:
        metadata = json.load(f)
    return index, metadata
```

---

## 8. Benchmark Comparison

### 8.1 Performance Comparison Table

```
Benchmark: 1M vectors, 768 dimensions, top-10, single node

Database       Index Type    QPS (p50)  Latency p99  Recall@10  Memory
────────────────────────────────────────────────────────────────────────
FAISS          IVF4096,PQ96  8,500      1.2ms        0.92       1.2 GB
FAISS          HNSW32        3,200      2.8ms        0.99       35 GB
Milvus         IVF_SQ8       2,800      4.1ms        0.95       12 GB
Milvus         HNSW          2,100      5.5ms        0.98       36 GB
Weaviate       HNSW+PQ       1,800      6.2ms        0.94       8 GB
Qdrant         HNSW+SQ8      3,100      3.5ms        0.97       14 GB
Pinecone       Managed       2,500      8.0ms        0.96       N/A
Chroma         HNSW          900        12.0ms       0.97       35 GB

Notes:
- FAISS numbers exclude network overhead (library, not server)
- Pinecone latency includes network round-trip
- All databases tuned for ~95%+ recall target
- Memory includes index + vectors + metadata overhead
```

### 8.2 Feature Comparison Matrix

```
Feature               FAISS   Milvus   Weaviate  Qdrant   Pinecone  Chroma
──────────────────────────────────────────────────────────────────────────────
Type                  Lib     DB       DB        DB       SaaS      DB
Managed hosting       ✗       Zilliz   WCS       Qdrant   ✓ only    ✗
Open source           ✓       ✓        ✓         ✓        ✗         ✓
Metadata filtering    ✗       ✓        ✓         ✓        ✓         ✓
Hybrid search         ✗       ✓        ✓         ✓        ✓         ✗
Multi-tenancy         ✗       ✓        ✓         ✓        ✓         ✓
Auto-vectorization    ✗       ✗        ✓         ✗        ✓(infer)  ✓
GraphQL API           ✗       ✗        ✓         ✗        ✗         ✗
GPU support           ✓       ✓        ✗         ✗        N/A       ✗
Max tested scale      1B+     10B+     100M      100M     1B+       1M
Disk-based index      ✓       ✓        ✓         ✓        N/A       ✗
Replication           ✗       ✓        ✓         ✓        ✓         ✗
Snapshot/backup       ✗       ✓        ✓         ✓        ✓         Manual
```

### 8.3 Selection Guide

```
When to use what:

  FAISS:
    - You need maximum throughput and control
    - Building a custom search engine or embedding into another system
    - GPU acceleration is required
    - No metadata filtering needed

  Milvus:
    - Billion-scale datasets
    - Need distributed, fault-tolerant vector search
    - Complex filtering + vector search
    - Enterprise deployment with Zilliz Cloud option

  Weaviate:
    - Want auto-vectorization (no separate embedding pipeline)
    - GraphQL API fits your stack
    - Need generative search (RAG built-in)
    - Module ecosystem matters

  Qdrant:
    - Want the best single-node performance
    - Need advanced filtering (must/should/must_not)
    - Rust performance with Python simplicity
    - Self-hosted with low operational overhead

  Pinecone:
    - Zero infrastructure management
    - Need serverless scaling (scale-to-zero)
    - Budget allows managed pricing
    - Fast time-to-production

  Chroma:
    - Prototyping and development
    - Small-scale production (<1M vectors)
    - Want embedded mode (no server)
    - LangChain/LlamaIndex integration
```

---

## 9. Integration with Data Engineering Pipelines

### 9.1 Batch Embedding Pipeline with Airflow

```python
"""
Airflow DAG for batch vector index updates.
Demonstrates how vector storage fits into a data engineering pipeline.
"""

from airflow.decorators import dag, task
from datetime import datetime

@dag(
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["vector", "embeddings"],
)
def vector_index_update():

    @task()
    def extract_new_documents(ds=None):
        """Extract documents modified since last run."""
        # Query data warehouse for new/updated documents
        # In practice: Spark SQL, dbt model, or direct DB query
        return {
            "count": 5000,
            "source": f"s3://data-lake/gold/documents/dt={ds}/",
        }

    @task()
    def generate_embeddings(doc_info: dict):
        """Generate embeddings using a model API or local model."""
        # Batch embedding generation
        # In practice: call OpenAI API, run local model on GPU
        return {
            "embeddings_path": "s3://embeddings/batch/2024-06-15/",
            "count": doc_info["count"],
            "model": "text-embedding-3-small",
            "dimension": 768,
        }

    @task()
    def upsert_to_vector_db(embedding_info: dict):
        """Upsert embeddings to vector database."""
        # In practice: batch upsert to Milvus/Qdrant/Pinecone
        return {
            "upserted": embedding_info["count"],
            "collection": "documents",
        }

    @task()
    def validate_index(upsert_result: dict):
        """Run sanity checks on updated index."""
        # Check: total count, sample query recall, latency
        checks = {
            "total_vectors": 1_250_000,
            "sample_recall": 0.96,
            "p99_latency_ms": 4.2,
            "status": "passed",
        }
        return checks

    docs = extract_new_documents()
    embeddings = generate_embeddings(docs)
    result = upsert_to_vector_db(embeddings)
    validate_index(result)

vector_index_update()
```

---

## Summary

```
Key takeaways:

1. Storage architecture choice (in-memory / disk / distributed) depends on
   dataset size, latency requirements, and budget

2. FAISS is the foundational library — understand its index types
   (Flat, IVF, HNSW, PQ) and factory strings for composition

3. Milvus excels at billion-scale distributed deployments with
   strong consistency options

4. Weaviate's module system enables auto-vectorization and
   built-in RAG capabilities

5. Pinecone offers zero-ops serverless; Qdrant offers the best
   self-hosted single-node performance

6. Chroma is ideal for prototyping but has scaling limitations

7. Vector stores must integrate with data pipelines — batch
   embedding updates, index versioning, and monitoring are
   data engineering responsibilities
```

---

## Practice Exercises

1. **FAISS Index Comparison**: Build Flat, IVF, HNSW, and IVFPQ indexes on 100K random vectors. Measure search time, recall, and memory for each.

2. **Milvus Collection Design**: Design a Milvus schema for an e-commerce product search system with category filtering and price range queries.

3. **Qdrant vs Chroma**: Implement the same search application using both Qdrant and Chroma. Compare API ergonomics and performance.

4. **Index Versioning Pipeline**: Build a Python script that saves FAISS indexes with version metadata and implements rollback.

5. **Benchmark Runner**: Write a benchmark script that measures QPS, latency percentiles, and recall for a given index configuration.

---

## Further Reading

- [FAISS Wiki — Guidelines to choose an index](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)
- [Milvus Documentation](https://milvus.io/docs)
- [Weaviate Documentation](https://weaviate.io/developers/weaviate)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Chroma Documentation](https://docs.trychroma.com/)

[← Previous: 21. Data Versioning and Data Contracts](21_Data_Versioning_and_Contracts.md) | [Next: 23. Production Vector Search →](23_Production_Vector_Search.md)
