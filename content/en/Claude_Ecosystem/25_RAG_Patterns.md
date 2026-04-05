# RAG Patterns

**Previous**: [24. Prompt Caching and Batch API](./24_Prompt_Caching_and_Batch_API.md)

---

Retrieval-Augmented Generation (RAG) extends Claude's capabilities beyond its training data by dynamically retrieving relevant information at query time. This lesson covers the full spectrum of RAG patterns — from basic document retrieval to advanced multi-step architectures — with a focus on practical implementation using Claude's unique strengths: a 200K context window, contextual retrieval, and seamless integration with MCP servers.

**Difficulty**: ⭐⭐⭐⭐

**Prerequisites**:
- Claude API fundamentals ([Lesson 15](./15_Claude_API_Fundamentals.md))
- Tool use and function calling ([Lesson 16](./16_Tool_Use_and_Function_Calling.md))
- Model Context Protocol basics ([Lesson 12](./12_Model_Context_Protocol.md))
- Prompt caching ([Lesson 24](./24_Prompt_Caching_and_Batch_API.md))

## Learning Objectives

After completing this lesson, you will be able to:

1. Design and implement end-to-end RAG pipelines with Claude
2. Choose appropriate document chunking strategies for different content types
3. Implement hybrid retrieval using embeddings and BM25
4. Make informed decisions between long-context and RAG approaches
5. Build contextual retrieval pipelines that improve retrieval accuracy
6. Implement citation and grounding patterns for verifiable outputs
7. Design multi-step RAG architectures for complex queries
8. Evaluate RAG systems with appropriate metrics
9. Build production RAG systems with MCP integration

---

## Table of Contents

1. [RAG Fundamentals](#1-rag-fundamentals)
2. [Document Chunking Strategies](#2-document-chunking-strategies)
3. [Embedding Models and Vector Databases](#3-embedding-models-and-vector-databases)
4. [Long Context vs RAG Trade-offs](#4-long-context-vs-rag-trade-offs)
5. [Contextual Retrieval](#5-contextual-retrieval)
6. [Citation and Grounding Patterns](#6-citation-and-grounding-patterns)
7. [Multi-Step RAG](#7-multi-step-rag)
8. [RAG Evaluation Metrics](#8-rag-evaluation-metrics)
9. [Production RAG with MCP](#9-production-rag-with-mcp)
10. [Exercises](#10-exercises)

---

## 1. RAG Fundamentals

### 1.1 What Is RAG?

RAG is a pattern that augments an LLM's generation with dynamically retrieved context. Instead of relying solely on training data, the model receives relevant documents at query time:

```
User Query → Retriever → Relevant Documents → Claude + Documents → Answer
```

### 1.2 Why RAG?

| Challenge | How RAG Helps |
|---|---|
| Knowledge cutoff | Retrieves up-to-date information |
| Hallucination | Grounds answers in source documents |
| Domain specificity | Injects domain knowledge without fine-tuning |
| Data privacy | Keeps sensitive data in your own infrastructure |
| Scalability | Works with any size knowledge base |

### 1.3 Basic RAG Pipeline

```python
import anthropic
import numpy as np
from dataclasses import dataclass


@dataclass
class Document:
    id: str
    content: str
    metadata: dict
    embedding: list[float] | None = None


class SimpleRAG:
    """A minimal RAG pipeline for demonstration."""

    def __init__(self, embedding_fn):
        self.client = anthropic.Anthropic()
        self.embedding_fn = embedding_fn
        self.documents: list[Document] = []

    def index(self, documents: list[Document]):
        """Index documents by computing embeddings."""
        for doc in documents:
            doc.embedding = self.embedding_fn(doc.content)
            self.documents.append(doc)

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """Retrieve the most relevant documents for a query."""
        query_embedding = self.embedding_fn(query)

        scored = []
        for doc in self.documents:
            score = np.dot(query_embedding, doc.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc.embedding)
            )
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]

    def generate(self, query: str, top_k: int = 5) -> str:
        """Retrieve relevant docs and generate an answer."""
        docs = self.retrieve(query, top_k)

        context = "\n\n---\n\n".join(
            f"[Source: {doc.id}]\n{doc.content}" for doc in docs
        )

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=(
                "Answer the user's question based on the provided context. "
                "If the context doesn't contain enough information, say so. "
                "Always cite your sources using [Source: id] format."
            ),
            messages=[
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}",
                }
            ],
        )
        return response.content[0].text
```

---

## 2. Document Chunking Strategies

How you split documents into chunks significantly affects retrieval quality. There is no one-size-fits-all strategy.

### 2.1 Fixed-Size Chunking

The simplest approach: split by character or token count with overlap.

```python
def fixed_size_chunks(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Split text into fixed-size chunks with overlap."""
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap

    return chunks
```

**Pros**: Simple, predictable chunk sizes.
**Cons**: May split sentences, paragraphs, or logical sections mid-thought.

### 2.2 Semantic Chunking

Split on natural boundaries: paragraphs, sections, or sentences.

```python
import re


def semantic_chunks(
    text: str,
    max_chunk_size: int = 1000,
    min_chunk_size: int = 100,
) -> list[str]:
    """Split text on semantic boundaries (paragraphs, then sections)."""
    # Try splitting by sections (## headings in Markdown)
    sections = re.split(r"\n(?=##\s)", text)

    chunks = []
    for section in sections:
        if len(section.split()) <= max_chunk_size:
            if len(section.split()) >= min_chunk_size:
                chunks.append(section.strip())
            elif chunks:
                # Merge small sections with previous chunk
                chunks[-1] += "\n\n" + section.strip()
            else:
                chunks.append(section.strip())
        else:
            # Section too large, split by paragraphs
            paragraphs = section.split("\n\n")
            current_chunk = ""
            for para in paragraphs:
                if len((current_chunk + para).split()) <= max_chunk_size:
                    current_chunk += ("\n\n" if current_chunk else "") + para
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para
            if current_chunk:
                chunks.append(current_chunk.strip())

    return chunks
```

**Pros**: Preserves logical structure, better for retrieval.
**Cons**: Variable chunk sizes, more complex implementation.

### 2.3 Recursive Character Splitting

Split by hierarchical separators: first by sections, then paragraphs, then sentences, then words.

```python
def recursive_split(
    text: str,
    max_size: int = 500,
    separators: list[str] | None = None,
) -> list[str]:
    """Recursively split text using a hierarchy of separators."""
    if separators is None:
        separators = ["\n\n## ", "\n\n", "\n", ". ", " "]

    if len(text.split()) <= max_size:
        return [text]

    # Try each separator level
    for sep in separators:
        parts = text.split(sep)
        if len(parts) > 1:
            chunks = []
            current = ""
            for part in parts:
                candidate = current + sep + part if current else part
                if len(candidate.split()) <= max_size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current.strip())
                    current = part
            if current:
                chunks.append(current.strip())

            # Recursively split any chunks that are still too large
            result = []
            for chunk in chunks:
                if len(chunk.split()) > max_size:
                    result.extend(recursive_split(chunk, max_size, separators[1:]))
                else:
                    result.append(chunk)
            return result

    # Fallback: just split by words
    return fixed_size_chunks(text, max_size, overlap=50)
```

### 2.4 Choosing a Strategy

| Content Type | Recommended Strategy |
|---|---|
| Structured docs (Markdown, HTML) | Semantic (by headings) |
| Legal/scientific papers | Semantic (by sections/paragraphs) |
| Unstructured text (logs, transcripts) | Fixed-size with overlap |
| Code files | Semantic (by functions/classes) |
| Mixed content | Recursive character splitting |

---

## 3. Embedding Models and Vector Databases

### 3.1 Embedding with Voyage AI

Anthropic recommends Voyage AI embeddings for use with Claude:

```python
import voyageai


voyage_client = voyageai.Client()  # Uses VOYAGE_API_KEY env var


def get_embeddings(texts: list[str], input_type: str = "document") -> list[list[float]]:
    """Get embeddings from Voyage AI."""
    result = voyage_client.embed(
        texts,
        model="voyage-3",
        input_type=input_type,  # "document" for indexing, "query" for search
    )
    return result.embeddings


def embed_query(query: str) -> list[float]:
    """Embed a search query."""
    return get_embeddings([query], input_type="query")[0]


def embed_documents(docs: list[str]) -> list[list[float]]:
    """Embed documents for indexing."""
    # Process in batches of 128
    all_embeddings = []
    for i in range(0, len(docs), 128):
        batch = docs[i : i + 128]
        embeddings = get_embeddings(batch, input_type="document")
        all_embeddings.extend(embeddings)
    return all_embeddings
```

### 3.2 Vector Database Integration

Example with ChromaDB (lightweight, embedded):

```python
import chromadb


def create_rag_collection(
    collection_name: str,
    documents: list[dict],
) -> chromadb.Collection:
    """Create a ChromaDB collection with embedded documents."""
    client = chromadb.PersistentClient(path="./chroma_db")

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Batch add documents
    ids = [doc["id"] for doc in documents]
    texts = [doc["content"] for doc in documents]
    metadatas = [doc.get("metadata", {}) for doc in documents]
    embeddings = embed_documents(texts)

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return collection


def search_collection(
    collection: chromadb.Collection,
    query: str,
    top_k: int = 5,
    where: dict | None = None,
) -> list[dict]:
    """Search a collection with optional metadata filtering."""
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
    )

    return [
        {
            "id": results["ids"][0][i],
            "content": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        }
        for i in range(len(results["ids"][0]))
    ]
```

### 3.3 Hybrid Search: Embeddings + BM25

Combining dense (embedding) and sparse (BM25) retrieval significantly improves accuracy:

```python
import math
from collections import Counter


class BM25:
    """Simple BM25 implementation for sparse retrieval."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = {}
        self.doc_lengths = []
        self.avg_dl = 0
        self.corpus_size = 0
        self.tokenized_docs = []

    def fit(self, documents: list[str]):
        """Index documents for BM25 scoring."""
        self.tokenized_docs = [doc.lower().split() for doc in documents]
        self.corpus_size = len(documents)
        self.doc_lengths = [len(doc) for doc in self.tokenized_docs]
        self.avg_dl = sum(self.doc_lengths) / self.corpus_size

        # Calculate document frequencies
        for doc in self.tokenized_docs:
            unique_terms = set(doc)
            for term in unique_terms:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

    def score(self, query: str) -> list[float]:
        """Score all documents against a query."""
        query_terms = query.lower().split()
        scores = [0.0] * self.corpus_size

        for term in query_terms:
            if term not in self.doc_freqs:
                continue

            df = self.doc_freqs[term]
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1)

            for i, doc in enumerate(self.tokenized_docs):
                tf = doc.count(term)
                dl = self.doc_lengths[i]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
                scores[i] += idf * numerator / denominator

        return scores


class HybridRetriever:
    """Combine embedding similarity with BM25 for hybrid retrieval."""

    def __init__(self, alpha: float = 0.7):
        """
        Args:
            alpha: Weight for embedding similarity (1-alpha for BM25).
        """
        self.alpha = alpha
        self.bm25 = BM25()
        self.documents = []
        self.collection = None

    def index(self, documents: list[dict], collection_name: str = "hybrid"):
        """Index documents for both dense and sparse retrieval."""
        self.documents = documents
        texts = [doc["content"] for doc in documents]

        # Sparse index
        self.bm25.fit(texts)

        # Dense index
        self.collection = create_rag_collection(collection_name, documents)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Hybrid search combining embedding and BM25 scores."""
        # Dense scores (convert distance to similarity)
        dense_results = search_collection(self.collection, query, top_k=len(self.documents))
        dense_scores = {r["id"]: 1 - r["distance"] for r in dense_results}

        # Sparse scores
        bm25_scores = self.bm25.score(query)
        sparse_scores = {
            self.documents[i]["id"]: score
            for i, score in enumerate(bm25_scores)
        }

        # Normalize scores to [0, 1]
        def normalize(scores: dict) -> dict:
            values = list(scores.values())
            if not values:
                return scores
            min_v, max_v = min(values), max(values)
            if max_v == min_v:
                return {k: 0.5 for k in scores}
            return {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}

        dense_norm = normalize(dense_scores)
        sparse_norm = normalize(sparse_scores)

        # Combine scores
        combined = {}
        all_ids = set(dense_norm.keys()) | set(sparse_norm.keys())
        for doc_id in all_ids:
            d_score = dense_norm.get(doc_id, 0)
            s_score = sparse_norm.get(doc_id, 0)
            combined[doc_id] = self.alpha * d_score + (1 - self.alpha) * s_score

        # Sort and return top_k
        sorted_ids = sorted(combined, key=combined.get, reverse=True)[:top_k]
        id_to_doc = {doc["id"]: doc for doc in self.documents}

        return [
            {**id_to_doc[doc_id], "score": combined[doc_id]}
            for doc_id in sorted_ids
            if doc_id in id_to_doc
        ]
```

---

## 4. Long Context vs RAG Trade-offs

Claude's 200K token context window creates an important design decision: when to use RAG versus stuffing everything into context.

### 4.1 Decision Framework

| Factor | Long Context | RAG |
|---|---|---|
| **Knowledge base size** | < 200K tokens | Any size |
| **Update frequency** | Rarely changes | Frequently updated |
| **Query types** | Need holistic understanding | Need specific facts |
| **Latency requirements** | Higher latency OK | Needs fast retrieval |
| **Cost sensitivity** | Lower volume | High volume |
| **Accuracy needs** | All info visible to model | Depends on retrieval quality |

### 4.2 When to Use Long Context

```python
def stuff_context_approach(documents: list[str], query: str) -> str:
    """For smaller knowledge bases, just put everything in context."""
    full_context = "\n\n---\n\n".join(documents)

    # Check if it fits in context window
    # Rough estimate: 1 token ≈ 4 characters
    estimated_tokens = len(full_context) / 4
    if estimated_tokens > 180_000:  # Leave room for query + response
        raise ValueError(f"Context too large: ~{estimated_tokens:.0f} tokens")

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=[
            {
                "type": "text",
                "text": f"Reference documents:\n\n{full_context}",
                "cache_control": {"type": "ephemeral"},  # Cache the documents!
            }
        ],
        messages=[{"role": "user", "content": query}],
    )
    return response.content[0].text
```

### 4.3 When to Use RAG

- Knowledge base > 200K tokens
- Documents are updated frequently (avoid re-caching)
- Only a few documents are relevant per query
- Cost is a concern (processing 200K tokens per query is expensive)
- Need to scale to millions of documents

### 4.4 Hybrid: RAG + Long Context

```python
def hybrid_approach(query: str, top_k: int = 20) -> str:
    """Retrieve more documents than typical RAG and use long context."""
    # Retrieve a generous set of candidates
    candidates = retriever.search(query, top_k=top_k)

    # Stuff them all into context (works because 20 docs usually < 200K tokens)
    context = "\n\n---\n\n".join(
        f"[Document {doc['id']}]\n{doc['content']}" for doc in candidates
    )

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=(
            "You have been given relevant documents to answer the user's question. "
            "Read ALL documents carefully before answering. "
            "Cite specific documents using [Document ID] format."
        ),
        messages=[
            {
                "role": "user",
                "content": f"Documents:\n\n{context}\n\nQuestion: {query}",
            }
        ],
    )
    return response.content[0].text
```

---

## 5. Contextual Retrieval

Anthropic's contextual retrieval technique improves chunk retrieval accuracy by prepending context to each chunk before embedding it.

### 5.1 The Problem with Naive Chunking

When you chunk a document, individual chunks often lose context:

```
Original: "In Q3 2024, revenue grew 15% YoY to $2.3B..."
Chunk:    "Revenue grew 15% YoY to $2.3B"
Problem:  Which company? Which quarter? Which year?
```

### 5.2 Adding Contextual Descriptions

Use Claude to generate a short context prefix for each chunk:

```python
def add_chunk_context(
    chunk: str,
    full_document: str,
    doc_title: str,
) -> str:
    """Use Claude to generate a contextual prefix for a chunk."""
    response = client.messages.create(
        model="claude-haiku-4-20250514",  # Use Haiku for cost efficiency
        max_tokens=200,
        system=[
            {
                "type": "text",
                "text": (
                    "You will be given a document and a chunk from that document. "
                    "Generate a SHORT (1-2 sentence) context that situates the chunk "
                    "within the full document. Include key identifiers like company name, "
                    "date, section title. Return ONLY the context, nothing else."
                ),
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Document title: {doc_title}\n\nFull document:\n{full_document}",
                        "cache_control": {"type": "ephemeral"},  # Cache the full doc
                    },
                    {
                        "type": "text",
                        "text": f"Chunk:\n{chunk}",
                    },
                ],
            }
        ],
    )
    context = response.content[0].text.strip()
    return f"{context}\n\n{chunk}"


def build_contextual_index(
    document: str,
    doc_title: str,
    chunk_fn,
) -> list[dict]:
    """Build a contextual retrieval index for a document."""
    chunks = chunk_fn(document)

    contextualized = []
    for i, chunk in enumerate(chunks):
        contextualized_chunk = add_chunk_context(chunk, document, doc_title)
        contextualized.append({
            "id": f"{doc_title}-chunk-{i}",
            "content": contextualized_chunk,
            "metadata": {
                "source": doc_title,
                "chunk_index": i,
                "original_content": chunk,  # Keep original for display
            },
        })

    return contextualized
```

### 5.3 Contextual Retrieval with Hybrid Search

Combining contextual embeddings with BM25 yields the best results:

```python
class ContextualRAG:
    """RAG pipeline with contextual retrieval and hybrid search."""

    def __init__(self, alpha: float = 0.7):
        self.client = anthropic.Anthropic()
        self.retriever = HybridRetriever(alpha=alpha)

    def ingest(self, documents: list[dict]):
        """Ingest documents with contextual chunk processing."""
        all_chunks = []
        for doc in documents:
            chunks = build_contextual_index(
                doc["content"],
                doc["title"],
                chunk_fn=lambda t: semantic_chunks(t, max_chunk_size=500),
            )
            all_chunks.extend(chunks)

        self.retriever.index(all_chunks, collection_name="contextual_rag")

    def query(self, question: str, top_k: int = 5) -> str:
        """Query with contextual retrieval."""
        results = self.retriever.search(question, top_k=top_k)

        # Use original content for display, but contextual content was used for retrieval
        context = "\n\n---\n\n".join(
            f"[Source: {r['metadata']['source']}]\n{r['metadata'].get('original_content', r['content'])}"
            for r in results
        )

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=(
                "Answer based on the provided sources. "
                "Cite sources using [Source: name] format. "
                "If information is insufficient, say so explicitly."
            ),
            messages=[
                {
                    "role": "user",
                    "content": f"Sources:\n{context}\n\nQuestion: {question}",
                }
            ],
        )
        return response.content[0].text
```

---

## 6. Citation and Grounding Patterns

Reliable citations are essential for trustworthy RAG systems.

### 6.1 Inline Citation Pattern

```python
CITATION_SYSTEM_PROMPT = """Answer the user's question based on the provided sources.

CITATION RULES:
1. Every factual claim MUST include a citation in [Source N] format
2. If multiple sources support a claim, cite all: [Source 1][Source 3]
3. Direct quotes must use quotation marks with citation: "exact text" [Source 2]
4. If no source supports a claim, explicitly state it is your general knowledge
5. End your answer with a "Sources Used" section listing all cited sources

Example:
The company reported $2.3B in revenue [Source 1], representing a 15% increase
year-over-year [Source 1][Source 3]. The CEO noted this was "driven primarily by
cloud services" [Source 2].

Sources Used:
- [Source 1]: Q3 2024 Earnings Report
- [Source 2]: CEO Earnings Call Transcript
- [Source 3]: Annual Report 2024"""


def generate_with_citations(query: str, sources: list[dict]) -> str:
    """Generate an answer with inline citations."""
    context = "\n\n".join(
        f"[Source {i+1}]: {src['title']}\n{src['content']}"
        for i, src in enumerate(sources)
    )

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=CITATION_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"{context}\n\nQuestion: {query}",
            }
        ],
    )
    return response.content[0].text
```

### 6.2 Structured Citation with Tool Use

For programmatic citation extraction, use tool use:

```python
citation_tool = {
    "name": "submit_answer",
    "description": "Submit an answer with structured citations.",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The answer text with [N] citation markers.",
            },
            "citations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "marker": {"type": "string", "description": "Citation marker, e.g., [1]"},
                        "source_id": {"type": "string"},
                        "quote": {"type": "string", "description": "Exact quoted text from source"},
                        "relevance": {"type": "string", "enum": ["direct", "supporting", "background"]},
                    },
                    "required": ["marker", "source_id", "quote"],
                },
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "Confidence that sources fully support the answer.",
            },
            "unsupported_claims": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Any claims in the answer not supported by sources.",
            },
        },
        "required": ["answer", "citations", "confidence"],
    },
}
```

### 6.3 Grounding Verification

```python
def verify_grounding(answer: str, sources: list[str]) -> dict:
    """Verify that an answer is grounded in the provided sources."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system=(
            "You are a fact-checking assistant. Verify each claim in the answer "
            "against the provided sources. For each claim, determine if it is:\n"
            "- SUPPORTED: Directly stated or clearly implied by a source\n"
            "- PARTIALLY_SUPPORTED: Related info exists but claim goes beyond sources\n"
            "- UNSUPPORTED: No source evidence for this claim\n"
            "- CONTRADICTED: Sources say the opposite\n\n"
            "Return JSON: {\"claims\": [{\"text\": str, \"status\": str, \"source\": str|null}]}"
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    f"Sources:\n" + "\n---\n".join(sources) +
                    f"\n\nAnswer to verify:\n{answer}"
                ),
            }
        ],
    )
    return json.loads(response.content[0].text)
```

---

## 7. Multi-Step RAG

Complex queries often cannot be answered with a single retrieval step. Multi-step RAG decomposes queries and iteratively retrieves information.

### 7.1 Query Decomposition

```python
def decompose_query(query: str) -> list[str]:
    """Break a complex query into simpler sub-queries."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=(
            "Break the user's complex question into 2-5 simpler sub-questions "
            "that, when answered together, fully address the original question. "
            "Return a JSON array of strings. No explanation, just the array."
        ),
        messages=[{"role": "user", "content": query}],
    )
    return json.loads(response.content[0].text)


def multi_step_rag(query: str, retriever, top_k: int = 5) -> str:
    """Answer a complex query by decomposing and retrieving iteratively."""
    sub_queries = decompose_query(query)
    all_context = []

    for sub_query in sub_queries:
        results = retriever.search(sub_query, top_k=top_k)
        for r in results:
            if r["content"] not in [c["content"] for c in all_context]:
                all_context.append(r)

    # Synthesize final answer from all retrieved context
    context_text = "\n\n---\n\n".join(
        f"[{doc['id']}]\n{doc['content']}" for doc in all_context
    )

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=(
            "Synthesize a comprehensive answer from the provided sources. "
            "The user asked a complex question that was broken into sub-questions. "
            "Address all aspects of the original question."
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    f"Original question: {query}\n\n"
                    f"Sub-questions investigated: {json.dumps(sub_queries)}\n\n"
                    f"Retrieved sources:\n{context_text}"
                ),
            }
        ],
    )
    return response.content[0].text
```

### 7.2 Hypothetical Document Embeddings (HyDE)

Generate a hypothetical answer first, then use it to retrieve better documents:

```python
def hyde_retrieval(query: str, retriever, top_k: int = 5) -> list[dict]:
    """Use HyDE to improve retrieval quality."""
    # Step 1: Generate a hypothetical answer
    response = client.messages.create(
        model="claude-haiku-4-20250514",
        max_tokens=512,
        system=(
            "Write a short, factual answer to the question as if you had access "
            "to a comprehensive knowledge base. This will be used for search, "
            "so include specific terms and details."
        ),
        messages=[{"role": "user", "content": query}],
    )
    hypothetical_answer = response.content[0].text

    # Step 2: Retrieve using the hypothetical answer as the query
    results = retriever.search(hypothetical_answer, top_k=top_k)

    return results
```

### 7.3 Agentic RAG with Tool Use

Let Claude decide when and how to retrieve:

```python
def agentic_rag(query: str, retriever) -> str:
    """Let Claude autonomously decide when to search and what to look for."""
    search_tool = {
        "name": "search_knowledge_base",
        "description": "Search the knowledge base for relevant documents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "top_k": {"type": "integer", "description": "Number of results (1-10)", "default": 5},
            },
            "required": ["query"],
        },
    }

    messages = [{"role": "user", "content": query}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=(
                "You are a research assistant with access to a knowledge base. "
                "Use the search tool to find relevant information before answering. "
                "You may search multiple times with different queries. "
                "When you have enough information, provide a comprehensive answer with citations."
            ),
            tools=[search_tool],
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            return next(b.text for b in response.content if b.type == "text")

        # Execute search tool calls
        tool_results = []
        for block in response.content:
            if block.type == "tool_use" and block.name == "search_knowledge_base":
                results = retriever.search(
                    block.input["query"],
                    top_k=block.input.get("top_k", 5),
                )
                formatted = "\n\n".join(
                    f"[{r['id']}] (score: {r.get('score', 'N/A')})\n{r['content']}"
                    for r in results
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": formatted,
                })

        messages.append({"role": "user", "content": tool_results})
```

---

## 8. RAG Evaluation Metrics

### 8.1 Key Metrics

| Metric | What It Measures | Range |
|---|---|---|
| **Context Precision** | Are retrieved docs relevant? | 0-1 |
| **Context Recall** | Were all needed docs retrieved? | 0-1 |
| **Faithfulness** | Is the answer supported by context? | 0-1 |
| **Answer Relevance** | Does the answer address the query? | 0-1 |

### 8.2 Evaluation with Claude as Judge

```python
class RAGEvaluator:
    """Evaluate RAG pipeline quality using Claude as a judge."""

    def __init__(self):
        self.client = anthropic.Anthropic()

    def evaluate_faithfulness(
        self,
        answer: str,
        context: list[str],
    ) -> dict:
        """Evaluate whether the answer is faithful to the provided context."""
        context_text = "\n\n---\n\n".join(context)

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=(
                "Evaluate the faithfulness of the answer to the provided context.\n\n"
                "1. Extract each factual claim from the answer\n"
                "2. Check if each claim is supported by the context\n"
                "3. Calculate faithfulness = supported_claims / total_claims\n\n"
                "Return JSON:\n"
                '{"claims": [{"text": str, "supported": bool}], '
                '"faithfulness_score": float, "reasoning": str}'
            ),
            messages=[
                {
                    "role": "user",
                    "content": f"Context:\n{context_text}\n\nAnswer:\n{answer}",
                }
            ],
        )
        return json.loads(response.content[0].text)

    def evaluate_relevance(
        self,
        query: str,
        answer: str,
    ) -> dict:
        """Evaluate whether the answer is relevant to the query."""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=(
                "Evaluate answer relevance to the query.\n\n"
                "Consider:\n"
                "1. Does the answer address the question directly?\n"
                "2. Is the answer complete (covers all aspects)?\n"
                "3. Is the answer concise (no unnecessary information)?\n\n"
                "Return JSON:\n"
                '{"relevance_score": float (0-1), "completeness": float (0-1), '
                '"conciseness": float (0-1), "reasoning": str}'
            ),
            messages=[
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nAnswer: {answer}",
                }
            ],
        )
        return json.loads(response.content[0].text)

    def evaluate_context_precision(
        self,
        query: str,
        contexts: list[str],
    ) -> dict:
        """Evaluate whether retrieved contexts are relevant to the query."""
        context_list = "\n\n".join(
            f"[Context {i+1}]:\n{ctx}" for i, ctx in enumerate(contexts)
        )

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=(
                "For each retrieved context, determine if it is relevant to answering "
                "the query. Return JSON:\n"
                '{"contexts": [{"id": int, "relevant": bool}], '
                '"precision": float (relevant / total)}'
            ),
            messages=[
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nRetrieved contexts:\n{context_list}",
                }
            ],
        )
        return json.loads(response.content[0].text)

    def full_evaluation(
        self,
        query: str,
        answer: str,
        contexts: list[str],
    ) -> dict:
        """Run all evaluation metrics."""
        faithfulness = self.evaluate_faithfulness(answer, contexts)
        relevance = self.evaluate_relevance(query, answer)
        precision = self.evaluate_context_precision(query, contexts)

        return {
            "faithfulness": faithfulness["faithfulness_score"],
            "answer_relevance": relevance["relevance_score"],
            "context_precision": precision["precision"],
            "details": {
                "faithfulness": faithfulness,
                "relevance": relevance,
                "precision": precision,
            },
        }
```

---

## 9. Production RAG with MCP

MCP servers provide a clean abstraction for RAG components, making your architecture modular and reusable.

### 9.1 RAG as an MCP Server

```python
"""MCP server that exposes a RAG pipeline as tools."""
from mcp.server import Server
from mcp.types import Tool, TextContent

app = Server("rag-server")

# Initialize RAG components (would load from persistent storage in production)
retriever = None  # Initialized at startup


@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="search",
            description="Search the knowledge base for relevant documents.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                    "filter_source": {"type": "string", "description": "Optional: filter by source name"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_document",
            description="Retrieve a full document by ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {"type": "string"},
                },
                "required": ["document_id"],
            },
        ),
        Tool(
            name="ingest",
            description="Add a new document to the knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                    "metadata": {"type": "object"},
                },
                "required": ["title", "content"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "search":
        results = retriever.search(
            arguments["query"],
            top_k=arguments.get("top_k", 5),
        )
        formatted = "\n\n---\n\n".join(
            f"**[{r['id']}]** (relevance: {r.get('score', 'N/A'):.3f})\n{r['content']}"
            for r in results
        )
        return [TextContent(type="text", text=formatted)]

    elif name == "get_document":
        doc = retriever.get_by_id(arguments["document_id"])
        if doc:
            return [TextContent(type="text", text=doc["content"])]
        return [TextContent(type="text", text="Document not found.")]

    elif name == "ingest":
        doc = {
            "id": arguments["title"].lower().replace(" ", "-"),
            "content": arguments["content"],
            "metadata": arguments.get("metadata", {}),
        }
        retriever.add_document(doc)
        return [TextContent(type="text", text=f"Ingested: {doc['id']}")]
```

### 9.2 Multi-Source RAG Architecture

```python
"""Claude Code configuration for multi-source RAG."""

# .claude/settings.json
MCP_CONFIG = {
    "mcpServers": {
        "docs-rag": {
            "command": "python",
            "args": ["-m", "rag_server", "--source", "documentation"],
        },
        "tickets-rag": {
            "command": "python",
            "args": ["-m", "rag_server", "--source", "jira"],
        },
        "code-search": {
            "command": "python",
            "args": ["-m", "rag_server", "--source", "codebase"],
        },
    }
}

# Claude can now search across all sources:
# "Search docs-rag for API reference, tickets-rag for related bugs,
#  and code-search for implementation examples"
```

### 9.3 Production Checklist

- [ ] **Chunking**: Choose strategy based on content type; test chunk sizes
- [ ] **Embedding**: Use Voyage AI or equivalent; batch embed for efficiency
- [ ] **Retrieval**: Implement hybrid search (embeddings + BM25)
- [ ] **Contextual retrieval**: Add context prefixes to chunks
- [ ] **Caching**: Cache system prompts and repeated context
- [ ] **Evaluation**: Set up automated faithfulness and relevance checks
- [ ] **Monitoring**: Track retrieval hit rates, answer quality, latency
- [ ] **Updates**: Implement incremental indexing for new/updated documents
- [ ] **Fallback**: Handle cases where no relevant documents are found
- [ ] **Rate limiting**: Respect API limits during batch ingestion

---

## 10. Exercises

### Exercise 1: Basic RAG Pipeline

Build a complete RAG pipeline for a set of Markdown documentation files:

```python
"""
Exercise 1 starter code — build a Markdown documentation RAG pipeline.
"""
from pathlib import Path


class MarkdownRAG:
    """RAG pipeline for Markdown documentation."""

    def __init__(self, docs_dir: str):
        self.docs_dir = Path(docs_dir)
        self.client = anthropic.Anthropic()
        # TODO: Initialize embedding function and retriever

    def ingest(self):
        """Load and index all Markdown files from docs_dir."""
        # TODO: Read all .md files
        # TODO: Split into chunks using semantic chunking
        # TODO: Embed and index chunks
        pass

    def query(self, question: str, top_k: int = 5) -> str:
        """Answer a question using the indexed documentation."""
        # TODO: Retrieve relevant chunks
        # TODO: Generate answer with citations
        pass


# Test
rag = MarkdownRAG("./docs")
rag.ingest()
answer = rag.query("How do I configure authentication?")
print(answer)
```

### Exercise 2: Contextual Retrieval Implementation

Implement contextual retrieval that adds context to each chunk:

```python
"""
Exercise 2 starter code — implement contextual retrieval.
"""


class ContextualRetriever:
    """Retriever that uses contextual chunk descriptions."""

    def __init__(self):
        self.client = anthropic.Anthropic()
        # TODO: Initialize storage

    def ingest_document(self, title: str, content: str):
        """
        Ingest a document with contextual chunk processing.

        Steps:
        1. Chunk the document
        2. For each chunk, generate a contextual description using Claude
        3. Embed the contextualized chunks
        4. Store in vector database
        """
        # TODO: Implement
        pass

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search with hybrid retrieval (embedding + BM25)."""
        # TODO: Implement hybrid search
        pass
```

### Exercise 3: RAG Evaluation Suite

Build an evaluation suite that measures your RAG pipeline's quality:

```python
"""
Exercise 3 starter code — RAG evaluation suite.
"""


class RAGTestSuite:
    """Evaluate a RAG pipeline on a test dataset."""

    def __init__(self, rag_pipeline):
        self.rag = rag_pipeline
        self.evaluator = RAGEvaluator()

    def load_test_cases(self, path: str) -> list[dict]:
        """
        Load test cases from a JSON file.

        Each test case: {
            "query": str,
            "expected_answer": str,  # Reference answer
            "required_sources": [str],  # Doc IDs that should be retrieved
        }
        """
        # TODO: Load and validate test cases
        pass

    def run(self, test_cases: list[dict]) -> dict:
        """
        Run evaluation on all test cases.

        Returns aggregate metrics:
        - Average faithfulness
        - Average relevance
        - Average context precision
        - Average context recall
        """
        # TODO: For each test case:
        #   1. Run RAG query
        #   2. Evaluate with all metrics
        #   3. Aggregate results
        pass

    def report(self, results: dict) -> str:
        """Generate a formatted evaluation report."""
        # TODO: Format results as a readable report
        pass
```

### Exercise 4: Multi-Step RAG Agent

Build a RAG agent that can decompose complex queries and iteratively retrieve:

```python
"""
Exercise 4 starter code — multi-step RAG agent.
"""


class ResearchAgent:
    """Agent that performs multi-step research using RAG."""

    def __init__(self, retriever):
        self.client = anthropic.Anthropic()
        self.retriever = retriever

    def research(self, question: str, max_steps: int = 5) -> dict:
        """
        Perform multi-step research to answer a complex question.

        Returns:
            {
                "answer": str,
                "steps": [
                    {"query": str, "findings": str}
                ],
                "sources": [str],
                "confidence": float,
            }
        """
        # TODO: Decompose the question into sub-queries
        # TODO: For each sub-query, retrieve and analyze
        # TODO: Synthesize findings into a comprehensive answer
        # TODO: Verify answer is grounded in sources
        pass
```

---

**Previous**: [24. Prompt Caching and Batch API](./24_Prompt_Caching_and_Batch_API.md)
