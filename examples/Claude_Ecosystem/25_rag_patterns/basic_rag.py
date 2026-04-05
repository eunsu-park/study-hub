"""
25_rag_patterns — RAG (Retrieval-Augmented Generation) Patterns
Demonstrates chunking, retrieval, and citation patterns.

This example uses simple in-memory implementations to illustrate
RAG concepts without requiring external dependencies.
"""

import math
from collections import Counter


# === 1. Document Chunking ===

def chunk_by_tokens(text: str, max_tokens: int = 200, overlap: int = 50) -> list[dict]:
    """Split text into overlapping chunks by approximate token count."""
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunk_words = words[start:end]
        chunks.append({
            "text": " ".join(chunk_words),
            "start_word": start,
            "end_word": end,
            "token_estimate": len(chunk_words),
        })
        start += max_tokens - overlap

    return chunks


def chunk_by_sections(text: str) -> list[dict]:
    """Split Markdown text by headers."""
    chunks = []
    current_header = "Introduction"
    current_lines = []

    for line in text.split("\n"):
        if line.startswith("# ") or line.startswith("## "):
            if current_lines:
                chunks.append({
                    "header": current_header,
                    "text": "\n".join(current_lines).strip(),
                })
            current_header = line.lstrip("#").strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        chunks.append({"header": current_header, "text": "\n".join(current_lines).strip()})

    return chunks


# === 2. Simple TF-IDF Retrieval ===

def compute_tf(words: list[str]) -> dict[str, float]:
    """Term frequency: count / total words."""
    counter = Counter(words)
    total = len(words)
    return {word: count / total for word, count in counter.items()}


def compute_idf(documents: list[list[str]]) -> dict[str, float]:
    """Inverse document frequency: log(N / df)."""
    n = len(documents)
    df = Counter()
    for doc in documents:
        df.update(set(doc))
    return {word: math.log(n / freq) for word, freq in df.items()}


class SimpleRetriever:
    """BM25-like retriever using TF-IDF scoring."""

    def __init__(self):
        self.chunks: list[dict] = []
        self.tokenized: list[list[str]] = []
        self.idf: dict[str, float] = {}

    def index(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        self.tokenized = [c["text"].lower().split() for c in chunks]
        self.idf = compute_idf(self.tokenized)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        query_words = query.lower().split()
        scores = []

        for i, doc_words in enumerate(self.tokenized):
            tf = compute_tf(doc_words)
            score = sum(tf.get(w, 0) * self.idf.get(w, 0) for w in query_words)
            scores.append((score, i))

        scores.sort(reverse=True)
        return [
            {**self.chunks[idx], "score": round(score, 4)}
            for score, idx in scores[:top_k]
            if score > 0
        ]


# === 3. RAG Prompt Construction ===

def build_rag_prompt(query: str, retrieved: list[dict]) -> str:
    """Construct a RAG prompt with retrieved context and citation instructions."""
    context_parts = []
    for i, chunk in enumerate(retrieved, 1):
        header = chunk.get("header", f"Chunk {i}")
        context_parts.append(f"[Source {i}: {header}]\n{chunk['text']}")

    context = "\n\n---\n\n".join(context_parts)

    return f"""Answer the user's question based ONLY on the provided context.
If the context doesn't contain enough information, say so explicitly.
Cite sources using [Source N] notation.

## Context

{context}

## Question

{query}

## Answer"""


# === 4. Hybrid Retrieval ===

def keyword_search(chunks: list[dict], query: str) -> list[dict]:
    """Simple keyword matching (simulates BM25)."""
    query_words = set(query.lower().split())
    results = []
    for chunk in chunks:
        chunk_words = set(chunk["text"].lower().split())
        overlap = len(query_words & chunk_words)
        if overlap > 0:
            results.append({**chunk, "keyword_score": overlap})
    results.sort(key=lambda x: x["keyword_score"], reverse=True)
    return results


def hybrid_search(retriever: SimpleRetriever, chunks: list[dict], query: str, top_k: int = 3) -> list[dict]:
    """Combine TF-IDF and keyword search with reciprocal rank fusion."""
    tfidf_results = retriever.search(query, top_k=top_k * 2)
    kw_results = keyword_search(chunks, query)[:top_k * 2]

    # Reciprocal Rank Fusion
    scores: dict[int, float] = {}
    k = 60  # RRF constant

    for rank, result in enumerate(tfidf_results):
        idx = chunks.index(next(c for c in chunks if c["text"] == result["text"]))
        scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank + 1)

    for rank, result in enumerate(kw_results):
        idx = chunks.index(next(c for c in chunks if c["text"] == result["text"]))
        scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank + 1)

    sorted_indices = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [{**chunks[i], "rrf_score": round(scores[i], 4)} for i in sorted_indices]


# === Demo ===

if __name__ == "__main__":
    # Sample document
    doc = """# Python Variables
Variables in Python are created when you assign a value. Python uses dynamic typing.

## Data Types
Python has several built-in data types: int, float, str, bool, list, dict, tuple, set.
Use type() to check a variable's type at runtime.

## Type Hints
Since Python 3.5, you can add type hints for better code documentation.
Type hints are not enforced at runtime but help with IDE support and static analysis.

## Mutability
Lists, dicts, and sets are mutable. Strings, tuples, and frozensets are immutable.
Understanding mutability is crucial for avoiding bugs in Python programs.

## Scope
Python uses LEGB rule: Local, Enclosing, Global, Built-in scope resolution.
Use the global keyword to modify global variables from within a function."""

    # Step 1: Chunk
    chunks = chunk_by_sections(doc)
    print(f"Chunked into {len(chunks)} sections:")
    for c in chunks:
        print(f"  [{c['header']}] {len(c['text'])} chars")

    # Step 2: Index
    retriever = SimpleRetriever()
    retriever.index(chunks)

    # Step 3: Search
    query = "What are type hints in Python?"
    results = retriever.search(query, top_k=2)
    print(f"\nQuery: '{query}'")
    print(f"Top results:")
    for r in results:
        print(f"  [{r.get('header', '?')}] score={r['score']}")

    # Step 4: Build RAG prompt
    prompt = build_rag_prompt(query, results)
    print(f"\nRAG prompt ({len(prompt)} chars):")
    print(prompt[:300] + "...")

    # Step 5: Hybrid search
    print(f"\nHybrid search results:")
    hybrid = hybrid_search(retriever, chunks, query, top_k=2)
    for r in hybrid:
        print(f"  [{r.get('header', '?')}] rrf_score={r['rrf_score']}")
