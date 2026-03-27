"""
11. Vector Database Example

Vector search using Chroma and FAISS
"""

import numpy as np

print("=" * 60)
print("Vector Database")
print("=" * 60)


# ============================================
# 1. Basic Vector Search (NumPy)
# ============================================
print("\n[1] NumPy Vector Search")
print("-" * 40)

def cosine_similarity(query, vectors):
    """Compute cosine similarity"""
    query_norm = query / np.linalg.norm(query)
    vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.dot(vectors_norm, query_norm)

# Sample data
documents = [
    "Python is a programming language",
    "Machine learning uses algorithms",
    "Deep learning is a subset of ML",
    "JavaScript is for web development",
    "Data science involves statistics"
]

# Simulated embeddings
np.random.seed(42)
embeddings = np.random.randn(len(documents), 128)

# Search
query_embedding = np.random.randn(128)
similarities = cosine_similarity(query_embedding, embeddings)

# Top results
top_k = 3
top_indices = np.argsort(similarities)[-top_k:][::-1]

print("Search results:")
for idx in top_indices:
    print(f"  [{similarities[idx]:.4f}] {documents[idx]}")


# ============================================
# 2. Chroma DB
# ============================================
print("\n[2] Chroma DB")
print("-" * 40)

try:
    import chromadb

    # Client (in-memory)
    client = chromadb.Client()

    # Create collection
    collection = client.create_collection(
        name="demo_collection",
        metadata={"description": "Demo collection"}
    )

    # Add documents
    collection.add(
        documents=documents,
        ids=[f"doc_{i}" for i in range(len(documents))],
        metadatas=[{"source": "demo"} for _ in documents]
    )

    print(f"Collection created: {collection.name}")
    print(f"Number of documents: {collection.count()}")

    # Search
    results = collection.query(
        query_texts=["What is Python?"],
        n_results=3
    )

    print("\nChroma search results:")
    for doc, dist in zip(results['documents'][0], results['distances'][0]):
        print(f"  [{dist:.4f}] {doc}")

    # Metadata filtering
    filtered = collection.query(
        query_texts=["programming"],
        n_results=2,
        where={"source": "demo"}
    )
    print(f"\nFiltered results: {len(filtered['documents'][0])} items")

except ImportError:
    print("chromadb not installed (pip install chromadb)")


# ============================================
# 3. FAISS
# ============================================
print("\n[3] FAISS")
print("-" * 40)

try:
    import faiss

    # Create index
    dimension = 128
    index = faiss.IndexFlatL2(dimension)  # L2 distance

    # Add vectors
    vectors = np.random.randn(1000, dimension).astype('float32')
    index.add(vectors)

    print(f"Index created: {index.ntotal} vectors")

    # Search
    query = np.random.randn(1, dimension).astype('float32')
    distances, indices = index.search(query, k=5)

    print(f"Search results (top 5):")
    print(f"  Indices: {indices[0]}")
    print(f"  Distances: {distances[0]}")

    # IVF index (for large-scale)
    nlist = 10  # Number of clusters
    quantizer = faiss.IndexFlatL2(dimension)
    ivf_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

    # Train and add
    ivf_index.train(vectors)
    ivf_index.add(vectors)
    ivf_index.nprobe = 3  # Number of clusters to search

    print(f"\nIVF index: {ivf_index.ntotal} vectors, {nlist} clusters")

    # Save/Load
    faiss.write_index(index, "demo_index.faiss")
    loaded_index = faiss.read_index("demo_index.faiss")
    print(f"Index save/load complete")

    import os
    os.remove("demo_index.faiss")

except ImportError:
    print("faiss not installed (pip install faiss-cpu)")


# ============================================
# 4. Sentence Transformers + Chroma
# ============================================
print("\n[4] Sentence Transformers + Chroma")
print("-" * 40)

try:
    import chromadb
    from chromadb.utils import embedding_functions

    # Sentence Transformer embedding function
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Client
    client = chromadb.Client()

    # Collection (with embedding function)
    collection = client.create_collection(
        name="semantic_search",
        embedding_function=embedding_fn
    )

    # Add documents (embeddings auto-generated)
    collection.add(
        documents=documents,
        ids=[f"doc_{i}" for i in range(len(documents))]
    )

    # Semantic search
    results = collection.query(
        query_texts=["How to learn programming?"],
        n_results=3
    )

    print("Semantic search results:")
    for doc, dist in zip(results['documents'][0], results['distances'][0]):
        print(f"  [{dist:.4f}] {doc}")

except ImportError as e:
    print(f"Required packages not installed: {e}")


# ============================================
# 5. LangChain + Chroma
# ============================================
print("\n[5] LangChain + Chroma (code)")
print("-" * 40)

langchain_chroma = '''
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Embeddings
embeddings = OpenAIEmbeddings()

# Create vector store
vectorstore = Chroma.from_texts(
    texts=documents,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Search
docs = vectorstore.similarity_search("What is Python?", k=3)

# Convert to retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
results = retriever.invoke("programming languages")

# Create with metadata
from langchain.schema import Document

docs_with_meta = [
    Document(page_content=text, metadata={"source": f"doc_{i}"})
    for i, text in enumerate(texts)
]

vectorstore = Chroma.from_documents(
    documents=docs_with_meta,
    embedding=embeddings
)
'''
print(langchain_chroma)


# ============================================
# 6. Index Type Comparison
# ============================================
print("\n[6] FAISS Index Type Comparison")
print("-" * 40)

index_comparison = """
| Index Type  | Accuracy | Speed    | Memory | Use Case           |
|-------------|----------|----------|--------|--------------------|
| IndexFlatL2 | 100%     | Slow     | High   | Small (<100K)      |
| IndexIVF    | 95%+     | Fast     | Medium | Medium-scale       |
| IndexHNSW   | 98%+     | Very fast| High   | Large, real-time   |
| IndexPQ     | 90%+     | Fast     | Low    | Memory-constrained |
"""
print(index_comparison)

faiss_indexes = '''
import faiss

# Flat (exact)
index = faiss.IndexFlatL2(dim)

# IVF (clustering)
quantizer = faiss.IndexFlatL2(dim)
index = faiss.IndexIVFFlat(quantizer, dim, nlist=100)
index.train(vectors)

# HNSW (graph-based)
index = faiss.IndexHNSWFlat(dim, 32)

# PQ (compression)
index = faiss.IndexPQ(dim, m=8, nbits=8)
index.train(vectors)
'''
print(faiss_indexes)


# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("Vector DB Summary")
print("=" * 60)

summary = """
Selection Guide:
    - Development/Prototype: Chroma
    - Large-scale local: FAISS
    - Production managed: Pinecone

Key Code:
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
    retriever = vectorstore.as_retriever()
"""
print(summary)
