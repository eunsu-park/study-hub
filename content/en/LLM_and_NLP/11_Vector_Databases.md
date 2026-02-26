# 11. Vector Databases

## Learning Objectives

- Vector database concepts
- Using Chroma, FAISS, Pinecone
- Indexing and search optimization
- Practical usage patterns

---

## 1. Vector Database Overview

### Why Vector DB?

```
Traditional DB:
    SELECT * FROM docs WHERE text LIKE '%machine learning%'
    → Keyword matching only

Vector DB:
    query_vector = embed("What is AI?")
    SELECT * FROM docs ORDER BY similarity(vector, query_vector)
    → Semantic similarity search
```

### Major Vector DBs

| Name | Type | Features |
|------|------|----------|
| Chroma | Local/Embedded | Simple, for development |
| FAISS | Library | Fast, large-scale |
| Pinecone | Cloud | Managed, scalable |
| Weaviate | Open source | Hybrid search |
| Qdrant | Open source | Strong filtering |
| Milvus | Open source | Large-scale, distributed |

---

## 2. Chroma

### Installation and Basic Usage

```python
pip install chromadb
```

```python
import chromadb
from chromadb.utils import embedding_functions

# Create client
client = chromadb.Client()  # In-memory
# client = chromadb.PersistentClient(path="./chroma_db")  # Persistent

# Embedding function
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create collection
collection = client.create_collection(
    name="my_collection",
    embedding_function=embedding_fn
)
```

### Adding Documents

```python
# Add documents
collection.add(
    documents=["Document 1 text", "Document 2 text", "Document 3 text"],
    metadatas=[{"source": "a"}, {"source": "b"}, {"source": "a"}],
    ids=["doc1", "doc2", "doc3"]
)

# Provide embeddings directly
collection.add(
    embeddings=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    documents=["Doc 1", "Doc 2"],
    ids=["id1", "id2"]
)
```

### Search

```python
# Query search
results = collection.query(
    query_texts=["What is machine learning?"],
    n_results=3
)

print(results['documents'])  # Document content
print(results['distances'])  # Distances
print(results['metadatas'])  # Metadata

# Metadata filtering
results = collection.query(
    query_texts=["query"],
    n_results=5,
    where={"source": "a"}  # Only source "a"
)

# Complex filters
results = collection.query(
    query_texts=["query"],
    where={"$and": [{"source": "a"}, {"year": {"$gt": 2020}}]}
)
```

### LangChain Integration

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# Create
vectorstore = Chroma.from_texts(
    texts=["text1", "text2", "text3"],
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Search
docs = vectorstore.similarity_search("query", k=3)

# Use as Retriever
retriever = vectorstore.as_retriever()
```

---

## 3. FAISS

### Installation and Basic Usage

```python
pip install faiss-cpu  # CPU version
# pip install faiss-gpu  # GPU version
```

```python
import faiss
import numpy as np

# Create index
dimension = 384
index = faiss.IndexFlatL2(dimension)  # L2 distance

# Add vectors
vectors = np.random.random((1000, dimension)).astype('float32')
index.add(vectors)

print(f"Total vectors: {index.ntotal}")
```

### Search

```python
# Search
query = np.random.random((1, dimension)).astype('float32')
distances, indices = index.search(query, k=5)

print(f"Indices: {indices}")
print(f"Distances: {distances}")
```

### Index Types

```python
# Flat (accurate, slow)
index = faiss.IndexFlatL2(dimension)

# IVF (approximate, fast)
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist=100)
index.train(vectors)  # Training required
index.add(vectors)
index.nprobe = 10  # Number of clusters to search

# HNSW (very fast)
index = faiss.IndexHNSWFlat(dimension, 32)  # 32 = M parameter
index.add(vectors)

# PQ (memory efficient)
index = faiss.IndexPQ(dimension, 8, 8)  # M=8, nbits=8
index.train(vectors)
index.add(vectors)
```

### Save/Load

```python
# Save
faiss.write_index(index, "index.faiss")

# Load
index = faiss.read_index("index.faiss")
```

### LangChain Integration

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# Create
vectorstore = FAISS.from_texts(
    texts=["text1", "text2"],
    embedding=embeddings
)

# Save/Load
vectorstore.save_local("faiss_index")
vectorstore = FAISS.load_local("faiss_index", embeddings)
```

---

## 4. Pinecone

### Installation and Setup

```python
pip install pinecone-client
```

```python
from pinecone import Pinecone, ServerlessSpec

# Create client
pc = Pinecone(api_key="your-api-key")

# Create index
pc.create_index(
    name="my-index",
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

# Connect to index
index = pc.Index("my-index")
```

### Adding Documents

```python
# Upsert (add/update)
index.upsert(
    vectors=[
        {"id": "vec1", "values": [0.1, 0.2, ...], "metadata": {"source": "a"}},
        {"id": "vec2", "values": [0.3, 0.4, ...], "metadata": {"source": "b"}},
    ]
)

# Batch upsert
from itertools import islice

def chunks(iterable, batch_size=100):
    it = iter(iterable)
    chunk = list(islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = list(islice(it, batch_size))

for batch in chunks(vectors, batch_size=100):
    index.upsert(vectors=batch)
```

### Search

```python
# Query
results = index.query(
    vector=[0.1, 0.2, ...],
    top_k=5,
    include_metadata=True
)

for match in results['matches']:
    print(f"ID: {match['id']}, Score: {match['score']}")
    print(f"Metadata: {match['metadata']}")

# Metadata filtering
results = index.query(
    vector=[0.1, 0.2, ...],
    top_k=5,
    filter={"source": {"$eq": "a"}}
)
```

### LangChain Integration

```python
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

vectorstore = PineconeVectorStore.from_texts(
    texts=["text1", "text2"],
    embedding=embeddings,
    index_name="my-index"
)

# Search
docs = vectorstore.similarity_search("query", k=3)
```

---

## 5. Indexing Strategies

### Index Type Comparison

| Type | Accuracy | Speed | Memory | When to Use |
|------|----------|-------|--------|-------------|
| Flat | 100% | Slow | High | Small-scale (<100K) |
| IVF | 95%+ | Fast | Medium | Medium-scale |
| HNSW | 98%+ | Very fast | High | Large-scale, real-time |
| PQ | 90%+ | Fast | Low | Memory limited |

### Hybrid Index

```python
# IVF + PQ
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFPQ(
    quantizer,
    dimension,
    nlist=100,   # Number of clusters
    m=8,         # PQ segments
    nbits=8      # PQ bits
)
index.train(vectors)
index.add(vectors)
```

---

## 6. Using Metadata

### Filtering Patterns

```python
# Chroma filter syntax
results = collection.query(
    query_texts=["query"],
    where={
        "$and": [
            {"category": "tech"},
            {"year": {"$gte": 2023}},
            {"author": {"$in": ["Alice", "Bob"]}}
        ]
    }
)

# Supported operators
# $eq, $ne: equal, not equal
# $gt, $gte, $lt, $lte: comparison
# $in, $nin: in, not in
# $and, $or: logical operations
```

### Metadata Updates

```python
# Chroma
collection.update(
    ids=["doc1"],
    metadatas=[{"source": "updated"}]
)

# Pinecone
index.update(
    id="vec1",
    set_metadata={"source": "updated"}
)
```

---

## 7. Practical Patterns

### Document Management Class

```python
class VectorStore:
    def __init__(self, persist_dir="./db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name="documents",
            embedding_function=self.embedding_fn
        )

    def add_documents(self, texts, metadatas=None, ids=None):
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        return ids

    def search(self, query, k=5, where=None):
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=where
        )
        return results

    def delete(self, ids):
        self.collection.delete(ids=ids)
```

### Incremental Updates

```python
import hashlib

def get_doc_id(text):
    return hashlib.md5(text.encode()).hexdigest()

def upsert_documents(texts, collection):
    """Upsert with deduplication"""
    ids = [get_doc_id(t) for t in texts]

    # Check existing documents
    existing = collection.get(ids=ids)
    existing_ids = set(existing['ids'])

    # Add only new documents
    new_texts = []
    new_ids = []
    for text, doc_id in zip(texts, ids):
        if doc_id not in existing_ids:
            new_texts.append(text)
            new_ids.append(doc_id)

    if new_texts:
        collection.add(documents=new_texts, ids=new_ids)

    return len(new_texts)
```

### Batch Processing

```python
def batch_add(collection, texts, batch_size=100):
    """Add large number of documents in batches"""
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        ids = [str(uuid.uuid4()) for _ in batch]
        collection.add(documents=batch, ids=ids)
        print(f"Added {min(i + batch_size, total)}/{total}")
```

---

## 8. Performance Optimization

### Embedding Caching

```python
import pickle
import os

class CachedEmbeddings:
    def __init__(self, model, cache_dir="./embed_cache"):
        self.model = model
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def embed(self, text):
        cache_key = hashlib.md5(text.encode()).hexdigest()
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")

        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        embedding = self.model.encode(text)

        with open(cache_path, 'wb') as f:
            pickle.dump(embedding, f)

        return embedding
```

### Index Optimization

```python
# FAISS search parameter tuning
index.nprobe = 20  # Search more clusters (accuracy ↑, speed ↓)

# Parallel search
faiss.omp_set_num_threads(4)  # Set number of threads
```

---

## Summary

### Selection Guide

| Situation | Recommendation |
|-----------|----------------|
| Development/Prototype | Chroma |
| Large-scale local | FAISS |
| Production managed | Pinecone |
| Open source self-hosted | Qdrant, Milvus |

### Key Code

```python
# Chroma
collection = client.create_collection("name")
collection.add(documents=texts, ids=ids)
results = collection.query(query_texts=["query"], n_results=5)

# FAISS
index = faiss.IndexFlatL2(dimension)
index.add(vectors)
distances, indices = index.search(query, k=5)

# LangChain
vectorstore = Chroma.from_texts(texts, embeddings)
docs = vectorstore.similarity_search("query", k=3)
```

---

## Exercises

### Exercise 1: FAISS Index Type Selection

You are building a vector search system and have the following constraints. For each scenario, choose the appropriate FAISS index type and explain why.

| Scenario | Dataset Size | Constraints | Best Index Type? |
|----------|-------------|-------------|-----------------|
| A. Medical record search | 50,000 vectors | 100% accuracy required | ? |
| B. Real-time product search | 10 million vectors | <50ms latency required | ? |
| C. Mobile app embedding search | 500,000 vectors | 500MB memory limit | ? |
| D. Nightly batch recommendation | 2 million vectors | Accuracy >95%, training time OK | ? |

<details>
<summary>Show Answer</summary>

| Scenario | Best Index | Reasoning |
|----------|-----------|-----------|
| A. Medical records (50K, 100% accuracy) | **IndexFlatL2** | Exact search, no approximation. 50K vectors × 384 dims × 4 bytes ≈ 73MB — perfectly manageable. Medical decisions require precision. |
| B. Real-time product search (10M, <50ms) | **IndexHNSWFlat** | 98%+ accuracy at millisecond latency. No training required. Best recall/latency trade-off for real-time serving. |
| C. Mobile app (500K, 500MB limit) | **IndexIVFPQ** | PQ compresses vectors from 1536 bytes to ~64 bytes (24x compression). 500K × 64 bytes ≈ 32MB, well within budget. |
| D. Batch recommendation (2M, >95%) | **IndexIVFFlat** | Good accuracy with fast approximate search. Training is a one-time cost. `nprobe` tunable for accuracy/speed trade-off. |

```python
import faiss
import numpy as np

dimension = 384
vectors = np.random.random((50000, dimension)).astype('float32')

# Scenario A: Exact flat index
index_a = faiss.IndexFlatL2(dimension)
index_a.add(vectors)

# Scenario B: HNSW (high connectivity for fast graph traversal)
index_b = faiss.IndexHNSWFlat(dimension, 32)  # M=32: higher = better recall, more memory
index_b.add(vectors[:50000])  # No training needed

# Scenario C: IVF + PQ for memory efficiency
quantizer = faiss.IndexFlatL2(dimension)
index_c = faiss.IndexIVFPQ(quantizer, dimension, nlist=1000, m=8, nbits=8)
# m=8: 8 sub-quantizers, nbits=8: 256 centroids each → 8 bytes/vector
index_c.train(vectors)
index_c.add(vectors)
print(f"Scenario C memory: ~{50000 * 8 / 1e6:.1f}MB (vs {50000 * dimension * 4 / 1e6:.0f}MB flat)")

# Scenario D: IVF for tunable accuracy
quantizer_d = faiss.IndexFlatL2(dimension)
index_d = faiss.IndexIVFFlat(quantizer_d, dimension, nlist=2000)
index_d.train(vectors)
index_d.add(vectors)
index_d.nprobe = 50  # Search 50 of 2000 clusters (~2.5%); increase for higher recall
```
</details>

---

### Exercise 2: Chroma Metadata Filtering

A document collection stores research papers with metadata: `year` (int), `category` (str: "ml", "nlp", "cv"), and `citations` (int). Write Chroma queries for the following requirements:

1. Find papers from 2022 or later in the "nlp" category
2. Find papers from "ml" or "cv" categories with more than 100 citations
3. Find papers from the last 3 years (2023-2025) that are NOT in the "cv" category

<details>
<summary>Show Answer</summary>

```python
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.Client()
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = client.create_collection("papers", embedding_function=embedding_fn)

# Sample data
collection.add(
    documents=["Attention is all you need", "BERT pre-training", "ResNet deep residual"],
    metadatas=[
        {"year": 2017, "category": "nlp", "citations": 50000},
        {"year": 2018, "category": "nlp", "citations": 30000},
        {"year": 2016, "category": "cv", "citations": 80000},
    ],
    ids=["p1", "p2", "p3"]
)

# Query 1: NLP papers from 2022+
results_1 = collection.query(
    query_texts=["transformer architecture"],
    n_results=5,
    where={
        "$and": [
            {"year": {"$gte": 2022}},
            {"category": {"$eq": "nlp"}}
        ]
    }
)

# Query 2: ML or CV with >100 citations
results_2 = collection.query(
    query_texts=["neural network"],
    n_results=5,
    where={
        "$and": [
            {"category": {"$in": ["ml", "cv"]}},
            {"citations": {"$gt": 100}}
        ]
    }
)

# Query 3: 2023-2025, not CV
results_3 = collection.query(
    query_texts=["deep learning"],
    n_results=10,
    where={
        "$and": [
            {"year": {"$gte": 2023}},
            {"year": {"$lte": 2025}},
            {"category": {"$ne": "cv"}}
        ]
    }
)

# Chroma filter operators reference:
# $eq, $ne: equal, not equal
# $gt, $gte, $lt, $lte: numeric comparisons
# $in, $nin: membership in list
# $and, $or: logical combinations
```

**Common pitfall:** Chroma's `$and`/`$or` take a list, and ALL conditions must be at the same nesting level. You cannot mix list-level and dict-level operators in the same `where` clause.
</details>

---

### Exercise 3: Deduplication with Content Hashing

Extend the `upsert_documents` function to also handle document updates: if a document with the same ID already exists but has different content, it should be updated; if it's identical, skip it.

```python
import hashlib

def get_doc_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

# Current implementation (add-only deduplication)
def upsert_documents(texts, collection):
    ids = [get_doc_id(t) for t in texts]
    existing = collection.get(ids=ids)
    existing_ids = set(existing['ids'])

    new_texts = [t for t, id_ in zip(texts, ids) if id_ not in existing_ids]
    new_ids = [id_ for id_, t in zip(ids, texts) if id_ not in existing_ids]

    if new_texts:
        collection.add(documents=new_texts, ids=new_ids)
    return len(new_texts)
```

<details>
<summary>Show Answer</summary>

The key insight is that content-based IDs (MD5 hashes) make identical documents always have the same ID. So "updating" a changed document means: the old content has ID `old_hash`, the new content has ID `new_hash` — they are different entries.

```python
import hashlib
from typing import Optional

def get_content_hash(text: str) -> str:
    """Generate a stable ID from content."""
    return hashlib.md5(text.encode()).hexdigest()

def smart_upsert(
    texts: list[str],
    doc_keys: list[str],  # Logical IDs (e.g. "doc_001", "doc_002")
    collection,
    metadatas: Optional[list[dict]] = None
) -> dict:
    """
    Upsert documents with change detection.

    Strategy: store both the logical key AND content hash in metadata.
    When re-indexing, check if the content hash changed.

    Returns: {"added": N, "updated": N, "skipped": N}
    """
    stats = {"added": 0, "updated": 0, "skipped": 0}

    for i, (text, key) in enumerate(zip(texts, doc_keys)):
        new_hash = get_content_hash(text)
        meta = metadatas[i] if metadatas else {}
        meta["doc_key"] = key
        meta["content_hash"] = new_hash

        # Check if this logical key already exists (query by metadata)
        existing = collection.get(where={"doc_key": {"$eq": key}})

        if not existing["ids"]:
            # New document
            collection.add(
                documents=[text],
                ids=[new_hash],
                metadatas=[meta]
            )
            stats["added"] += 1

        elif existing["metadatas"][0]["content_hash"] == new_hash:
            # Identical content — skip
            stats["skipped"] += 1

        else:
            # Content changed — delete old, add new
            collection.delete(ids=existing["ids"])
            collection.add(
                documents=[text],
                ids=[new_hash],
                metadatas=[meta]
            )
            stats["updated"] += 1

    return stats

# Test
import chromadb
client = chromadb.Client()
coll = client.create_collection("docs")

result = smart_upsert(
    texts=["Version 1 of doc A", "Version 1 of doc B"],
    doc_keys=["doc_A", "doc_B"],
    collection=coll
)
print(result)  # {"added": 2, "updated": 0, "skipped": 0}

result = smart_upsert(
    texts=["Version 2 of doc A", "Version 1 of doc B"],  # A changed, B same
    doc_keys=["doc_A", "doc_B"],
    collection=coll
)
print(result)  # {"added": 0, "updated": 1, "skipped": 1}
```

**Why content-hash IDs matter:** If you used sequential IDs, you'd have to compare document text on every upsert. With content hashes, unchanged documents always generate the same ID — no text comparison needed.
</details>

---

## Next Steps

Build a conversational AI system in [12_Practical_Chatbot.md](./12_Practical_Chatbot.md).
