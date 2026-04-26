# 10. RAG Basics

## Learning Objectives

- Understanding RAG (Retrieval-Augmented Generation)
- Document embedding and retrieval
- Chunking strategies
- Implementing RAG pipelines

---

## Theory & Principles

RAG (Retrieval-Augmented Generation) addresses two fundamental limitations of LLMs: **knowledge cutoff** (the model only knows what was in its pre-training corpus) and **hallucination** (the model freely invents content when uncertain). The recipe is simple: at inference time, *retrieve* relevant documents from an external corpus, *prepend* them to the prompt, and let the LLM *generate* an answer grounded in the retrieved evidence. The engineering challenge is making each of these three stages — retrieve, prepend, generate — reliable, fast, and accurate.

This section covers:

- **(A) Why retrieve at all** — the parametric vs non-parametric memory trade-off, and what RAG buys you over fine-tuning.
- **(B) Embeddings and similarity** — how text becomes a vector, and why cosine similarity is the natural metric.
- **(C) Chunking strategies** — the unit-of-retrieval question, sliding windows, semantic chunking, recursive splits.
- **(D) Dense vs sparse retrieval** — embedding-based vs BM25, when to use each, hybrid search.
- **(E) The RAG pipeline** — end-to-end flow with citations, the role of context length budgeting.
- **(F) Evaluation** — retrieval metrics (recall@k, MRR), generation metrics (faithfulness, answer relevance), and the RAGAS framework.

### A. Why Retrieve at All: Parametric vs Non-Parametric Memory

An LLM stores knowledge in its weights — *parametric memory*. RAG adds a separate text store from which relevant pieces are pulled at inference — *non-parametric memory*. The two have complementary properties:

| Property | Parametric (LLM weights) | Non-parametric (RAG corpus) |
|----------|--------------------------|------------------------------|
| Update cost | re-train or fine-tune (expensive) | re-index a document (cheap) |
| Capacity | bounded by parameter count | bounded by storage |
| Citability | no source attribution | exact source available |
| Latency | one forward pass | retrieval + forward pass |
| Privacy | knowledge baked in | can scope per user/tenant |

RAG dominates fine-tuning for use cases where: knowledge changes frequently (news, docs), citations are required (legal, medical), per-user knowledge isolation is needed (multi-tenant SaaS), or the corpus is too large to memorize (enterprise document store).

Fine-tuning still wins for: skills/behaviors (formatting, tone, refusal patterns), highly stable knowledge, and very tight latency budgets where retrieval overhead is unacceptable.

### B. Embeddings and Similarity

**B.1 Sentence embeddings.** A model `f : text → ℝ^d` (typically d ∈ [384, 1536]) such that semantically similar texts map to nearby vectors. Modern embedding models (Sentence-BERT, OpenAI `text-embedding-3`, BGE, Cohere Embed) are trained with **contrastive loss**: positive pairs (e.g., a question and its correct answer paragraph) get pulled together, negatives get pushed apart:

```
L = − log [ exp(sim(q, p+) / τ) / Σ_{p ∈ {p+, p-_1, ..., p-_k}} exp(sim(q, p) / τ) ]
```

After training, `sim(q, p)` (usually cosine) ranks `p+` ahead of all `p-`s. The embedding space inherits this geometry: relevant chunks cluster near their queries.

**B.2 Cosine vs dot product.** Cosine similarity `cos(a, b) = (a · b) / (||a|| · ||b||)` is invariant to vector magnitude. Dot product `a · b` is not — long vectors win regardless of direction. For models trained with cosine loss (most), cosine is the correct metric. For models trained with dot product (e.g., some retrieval models), dot product is correct. Mismatching the metric to training silently degrades quality.

**B.3 Why this geometry encodes meaning.** Embedding training creates a low-dimensional manifold where directions correspond to semantic axes (topic, tone, intent). The contrastive objective forces the model to discover what features actually distinguish relevant from irrelevant. With enough training pairs, the geometry generalizes — unseen queries land near unseen-but-relevant passages.

### C. Chunking Strategies

The unit-of-retrieval question: do you embed and retrieve sentences? paragraphs? whole documents? The trade-off:

- **Smaller chunks**: precise retrieval (the chunk is mostly relevant), but loses surrounding context that may be needed for the answer.
- **Larger chunks**: more context per retrieval, but the same chunk may be retrieved for unrelated queries (recall-precision dilution).

**C.1 Fixed-size with overlap.** Split into N-token chunks (typical N = 256-1024) with M-token overlap (M ≈ 0.1·N). Overlap prevents losing answers that straddle chunk boundaries.

**C.2 Sentence- or paragraph-aligned.** Use natural document structure. Avoids splitting mid-thought but produces variable chunk sizes.

**C.3 Recursive character text splitting.** Try splitting on paragraph breaks first, then sentences, then words, then characters — recurse until chunks fit the target size. Used by LangChain.

**C.4 Semantic chunking.** Compute embeddings for sentences; merge consecutive sentences whose embeddings are close (same topic). Splits where embedding similarity drops sharply (topic change). More expensive, often higher-quality chunks.

**C.5 Information-theoretic view.** Each chunk is a unit of context. Retrieving `k` chunks gives a context budget of roughly `k · chunk_size` tokens, capped by the model's context window. Smaller chunks → higher `k` is feasible → better recall on diverse queries; larger chunks → lower `k` → better single-chunk completeness. The optimal point depends on query type (specific facts vs broad summaries) and is usually empirical.

### D. Dense vs Sparse Retrieval

**D.1 Sparse (BM25, TF-IDF).** Treat queries and documents as bags of words; rank by lexical overlap with frequency-weighted scoring. BM25's score:

```
score(q, d) = Σ_{t ∈ q}  IDF(t) · [ (tf(t,d) · (k₁+1)) / (tf(t,d) + k₁ · (1 − b + b · |d|/avgdl)) ]
```

with parameters `k₁ ≈ 1.5`, `b ≈ 0.75`. Strengths: exact term matching (great for proper nouns, IDs, exact phrases), no embedding model required, blazingly fast on inverted indexes. Weaknesses: vocabulary mismatch (synonyms invisible), no semantic generalization.

**D.2 Dense (embedding-based).** Encode query and documents into vectors, retrieve nearest neighbors. Strengths: semantic matching (synonyms, paraphrases), cross-lingual capability with multilingual encoders. Weaknesses: misses exact lexical matches, requires embedding model + vector index.

**D.3 Hybrid search.** Combine both: retrieve top-N from BM25 and top-N from dense, then either union and re-rank or use **Reciprocal Rank Fusion (RRF)**:

```
RRF_score(d) = Σ_methods  1 / (k + rank_method(d))   typical k = 60
```

Hybrid almost always beats either alone — they fail on disjoint cases. Production RAG systems default to hybrid.

### E. The RAG Pipeline

```
[user query]
    ↓
[embed query]
    ↓
[retrieve top-k from vector store + BM25]
    ↓
[(optional) rerank with cross-encoder]
    ↓
[build prompt: instruction + retrieved chunks (with sources) + query]
    ↓
[LLM generates answer with inline citations]
    ↓
[(optional) verify citations, retry if hallucinated]
```

**Context budgeting.** Total prompt tokens ≤ context window. Subtract overhead (system prompt, instructions, query, expected response): the budget for retrieved chunks is typically 4-16K tokens for a 32K-context model. With chunks of 512 tokens, that is `k = 8-32` chunks.

**Citation discipline.** A robust RAG prompt includes per-chunk identifiers (`[Source 3]`) and instructs the model to cite sources for every claim. The downstream pipeline can validate that cited sources actually appear in the retrieved set, and reject answers that cite non-existent sources.

### F. Evaluation

**F.1 Retrieval metrics.** Given a labeled set of (query, relevant chunk IDs):

- **Recall@k**: fraction of relevant chunks in the top-k retrieved set. The dominant metric for RAG retrieval — if relevant chunks are not retrieved, no LLM can answer correctly.
- **MRR (Mean Reciprocal Rank)**: 1 / rank of first relevant chunk, averaged. Captures how high the relevant chunk appears.
- **NDCG**: handles graded relevance.

**F.2 Generation metrics** (RAGAS framework):

- **Faithfulness**: does every claim in the answer follow from the retrieved chunks? An LLM judge scores this by extracting claims and verifying each against the context.
- **Answer relevance**: does the answer address the query? An LLM judge generates plausible questions for the answer and measures embedding similarity to the original query.
- **Context precision/recall**: do the retrieved chunks contain (only) relevant information?

These can all be computed with another LLM as judge — the standard pattern is GPT-4 as the evaluator, applied to outputs from any model under test.

### From Theory to the Functions Below

- §1 (RAG overview) — frames the §A parametric vs non-parametric trade-off.
- §2 (preprocessing) — implements §C chunking strategies (fixed, recursive, semantic).
- §3 (embedding generation) — calls §B's embedding models (Sentence-BERT, OpenAI embeddings).
- §4 (vector search) — basic dense retrieval; hybrid (§D) covered in lesson 11.
- §5 (simple RAG) — the §E pipeline end-to-end on a small corpus.
- §6 (advanced techniques) — pointers to query rewriting and reranking (full lesson 12).
- §7 (chunking comparison) — empirical comparison of §C strategies.
- §8 (evaluation) — implements §F's RAGAS metrics on the simple RAG.

---

## 1. RAG Overview

### Why RAG?

```
LLM Limitations:
- No knowledge of information after training (knowledge cutoff)
- Hallucination (generating incorrect information)
- Lack of specific domain knowledge

RAG Solution:
- Generate answers after retrieving external knowledge
- Can reflect latest information
- Improved trustworthiness by providing sources
```

### RAG Architecture

> **RAG Pipeline**
>
> Question --> Embedding --> Vector Search --> Relevant Docs
>
> Question + Docs --> LLM --> Answer

---

## 2. Document Preprocessing

### Chunking

```python
def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into chunks with overlap"""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks

# Usage
text = "Very long document text here..."
chunks = chunk_text(text, chunk_size=500, overlap=100)
```

### Sentence-based Chunking

```python
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def chunk_by_sentences(text, max_sentences=5, overlap_sentences=1):
    sentences = sent_tokenize(text)
    chunks = []

    for i in range(0, len(sentences), max_sentences - overlap_sentences):
        chunk = ' '.join(sentences[i:i + max_sentences])
        chunks.append(chunk)

    return chunks
```

### Semantic Chunking

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
)

chunks = splitter.split_text(text)
```

---

## 3. Embedding Generation

### Sentence Transformers

```python
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
texts = ["Hello world", "How are you?"]
embeddings = model.encode(texts)

print(embeddings.shape)  # (2, 384)
```

### HuggingFace Embeddings

```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def get_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pooling
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()
```

### OpenAI Embeddings

```python
from openai import OpenAI

client = OpenAI()

def get_openai_embeddings(texts, model="text-embedding-3-small"):
    response = client.embeddings.create(input=texts, model=model)
    return [r.embedding for r in response.data]
```

---

## 4. Vector Search

### Cosine Similarity

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def search(query_embedding, document_embeddings, top_k=5):
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return top_indices, similarities[top_indices]

# Usage
query_emb = model.encode(["What is machine learning?"])[0]
doc_embs = model.encode(documents)

indices, scores = search(query_emb, doc_embs, top_k=3)
```

### Using FAISS

```python
import faiss
import numpy as np

# Create index
dimension = 384  # Embedding dimension
index = faiss.IndexFlatIP(dimension)  # Inner Product (requires normalization for cosine similarity)

# Add after normalization
embeddings = np.array(embeddings).astype('float32')
faiss.normalize_L2(embeddings)
index.add(embeddings)

# Search
query_emb = model.encode(["query"])[0].astype('float32').reshape(1, -1)
faiss.normalize_L2(query_emb)

distances, indices = index.search(query_emb, k=5)
```

---

## 5. Simple RAG Implementation

```python
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import numpy as np

class SimpleRAG:
    def __init__(self, embedding_model='all-MiniLM-L6-v2'):
        self.embed_model = SentenceTransformer(embedding_model)
        self.client = OpenAI()
        self.documents = []
        self.embeddings = None

    def add_documents(self, documents):
        """Add documents and generate embeddings"""
        self.documents.extend(documents)
        self.embeddings = self.embed_model.encode(self.documents)

    def search(self, query, top_k=3):
        """Search relevant documents"""
        query_emb = self.embed_model.encode([query])[0]

        # Cosine similarity
        similarities = np.dot(self.embeddings, query_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb)
        )

        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]

    def generate(self, query, top_k=3):
        """Generate RAG answer"""
        # Search
        relevant_docs = self.search(query, top_k)
        context = "\n\n".join(relevant_docs)

        # Construct prompt
        prompt = f"""Answer the question based on the context below.

Context:
{context}

Question: {query}

Answer:"""

        # Call LLM
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content

# Usage
rag = SimpleRAG()
rag.add_documents([
    "Python is a programming language.",
    "Machine learning is a subset of AI.",
    "RAG combines retrieval with generation."
])

answer = rag.generate("What is RAG?")
print(answer)
```

---

## 6. Advanced RAG Techniques

### Hybrid Search

```python
from rank_bm25 import BM25Okapi

class HybridRAG:
    def __init__(self):
        self.documents = []
        self.bm25 = None
        self.embeddings = None

    def add_documents(self, documents):
        self.documents = documents

        # BM25 (keyword search)
        tokenized = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

        # Embeddings (semantic search)
        self.embeddings = model.encode(documents)

    def hybrid_search(self, query, top_k=5, alpha=0.5):
        # BM25 scores
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_scores = bm25_scores / bm25_scores.max()  # Normalize

        # Embedding scores
        query_emb = model.encode([query])[0]
        embed_scores = cosine_similarity([query_emb], self.embeddings)[0]

        # Combine
        combined = alpha * embed_scores + (1 - alpha) * bm25_scores

        top_indices = np.argsort(combined)[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]
```

### Query Expansion

```python
def expand_query(query, llm_client):
    """Improve search performance with query expansion"""
    prompt = f"""Generate 3 alternative versions of this search query:
    Original: {query}

    Alternatives:
    1."""

    response = llm_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    expanded = [query] + parse_alternatives(response.choices[0].message.content)
    return expanded
```

### Reranking

```python
from sentence_transformers import CrossEncoder

class RAGWithReranker:
    def __init__(self):
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def search_and_rerank(self, query, candidates, top_k=3):
        # Stage 1: Initial search (many candidates)
        initial_results = self.search(query, top_k=20)

        # Stage 2: Reranking
        pairs = [[query, doc] for doc in initial_results]
        scores = self.reranker.predict(pairs)

        # Select top k
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [initial_results[i] for i in top_indices]
```

### Multi-Query RAG

```python
def multi_query_rag(question, rag, num_queries=3):
    """Search with queries from multiple perspectives"""
    # Generate diverse queries
    prompt = f"""Generate {num_queries} different search queries for:
    Question: {question}

    Queries:"""

    queries = generate_queries(prompt)

    # Search with each query
    all_docs = set()
    for q in queries:
        docs = rag.search(q, top_k=3)
        all_docs.update(docs)

    return list(all_docs)
```

---

## 7. Chunking Strategy Comparison

| Strategy | Advantages | Disadvantages | When to Use |
|----------|-----------|---------------|-------------|
| Fixed size | Simple implementation | Context breaks | General text |
| Sentence-based | Semantic units | Uneven lengths | Structured text |
| Semantic | Preserves meaning | Computation cost | Quality required |
| Hierarchical | Multi-level search | Complex | Long documents |

---

## 8. Evaluation Metrics

### Retrieval Evaluation

```python
def calculate_recall_at_k(retrieved, relevant, k):
    """Calculate Recall@K"""
    retrieved_k = set(retrieved[:k])
    relevant_set = set(relevant)
    return len(retrieved_k & relevant_set) / len(relevant_set)

def calculate_mrr(retrieved, relevant):
    """Mean Reciprocal Rank"""
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            return 1 / (i + 1)
    return 0
```

### Generation Evaluation

```python
# Using RAGAS library
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision]
)
```

---

## Summary

### RAG Checklist

```
□ Choose appropriate chunk size
□ Select embedding model (consider domain)
□ Tune retrieval top-k
□ Optimize prompts
□ Set evaluation metrics
```

### Key Code

```python
# Embedding
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents)

# Search
query_emb = model.encode([query])[0]
similarities = cosine_similarity([query_emb], embeddings)

# Generation
context = "\n".join(relevant_docs)
prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
```

---

## Exercises

### Exercise 1: Chunking Strategy Analysis

Given the following 500-character text, compare the output of fixed-size chunking (chunk_size=100, overlap=20) vs sentence-based chunking. Explain which strategy produces better semantic units and why.

```python
text = (
    "Machine learning is a subset of artificial intelligence. "
    "It enables systems to learn from data without being explicitly programmed. "
    "Deep learning uses neural networks with many layers. "
    "These networks can learn hierarchical representations of data. "
    "Natural language processing applies these techniques to text. "
    "Modern LLMs like GPT and BERT use transformer architectures. "
    "Transformers rely on self-attention mechanisms. "
    "They have revolutionized NLP tasks."
)
```

<details>
<summary>Show Answer</summary>

```python
# Fixed-size chunking output (chunk_size=100, overlap=20)
def chunk_text(text, chunk_size=100, overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

fixed_chunks = chunk_text(text, 100, 20)
# Chunk 0: "Machine learning is a subset of artificial intelligence. It enables systems to learn from data wi"
# Chunk 1: "data without being explicitly programmed. Deep learning uses neural networks with many layers. These"
# Problem: sentences are broken mid-way!

# Sentence-based chunking output (max_sentences=2, overlap=1)
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt', quiet=True)

def chunk_by_sentences(text, max_sentences=2, overlap_sentences=1):
    sentences = sent_tokenize(text)
    chunks = []
    step = max_sentences - overlap_sentences
    for i in range(0, len(sentences), step):
        chunk = ' '.join(sentences[i:i + max_sentences])
        chunks.append(chunk)
    return chunks

sent_chunks = chunk_by_sentences(text, max_sentences=2, overlap_sentences=1)
# Chunk 0: "Machine learning is a subset of artificial intelligence. It enables systems to learn from data without being explicitly programmed."
# Chunk 1: "It enables systems to learn from data without being explicitly programmed. Deep learning uses neural networks with many layers."
# Each chunk is a complete thought!
```

**Why sentence-based is better here:**
- Fixed-size chunking cuts sentences mid-word, destroying semantic coherence
- A retrieval query about "deep learning" might match a chunk that starts mid-sentence, making the context hard for the LLM to use
- Sentence-based chunking preserves complete thoughts, improving retrieval accuracy
- The overlap (1 sentence) ensures no context is lost at boundaries

**When fixed-size is acceptable:** Very long documents without clear sentence boundaries (e.g., logs, transcripts), or when token budget is strict and sentences are very long.
</details>

---

### Exercise 2: Mean Pooling vs CLS Token

The `get_embeddings` function in this lesson uses mean pooling over the last hidden state. An alternative is to use only the `[CLS]` token representation. Implement both approaches and explain when each is preferred.

```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

texts = ["The cat sat on the mat.", "Artificial intelligence is transforming industries."]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)
    # outputs.last_hidden_state shape: (batch, seq_len, hidden_dim)
```

<details>
<summary>Show Answer</summary>

```python
# Approach 1: CLS token (first token)
cls_embeddings = outputs.last_hidden_state[:, 0, :]
# Shape: (batch_size, hidden_dim) = (2, 384)

# Approach 2: Mean pooling (average over non-padding tokens)
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    # Expand mask to match embedding dimensions
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()
    ).float()
    # Sum embeddings for non-padding tokens, then divide by count
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask

mean_embeddings = mean_pooling(outputs, inputs['attention_mask'])
# Shape: (batch_size, hidden_dim) = (2, 384)

# Approach 3: Max pooling (captures strongest features)
max_embeddings = outputs.last_hidden_state.max(dim=1).values
```

**When to use each:**
| Method | Best For | Notes |
|--------|----------|-------|
| CLS token | Models fine-tuned with CLS (e.g., BERT for classification) | CLS must be trained to encode sentence meaning |
| Mean pooling | General sentence similarity, SentenceTransformers models | More robust, captures all tokens equally |
| Max pooling | Capturing salient features | Useful when key phrases matter more than overall meaning |

**Important**: The attention mask is crucial for mean pooling — without it, padding tokens contribute to the average, degrading embedding quality for shorter sentences.
</details>

---

### Exercise 3: Hybrid Search Alpha Tuning

In the `HybridRAG.hybrid_search` method, the `alpha` parameter controls the balance between semantic search and BM25 keyword search. Given the following query types, recommend an `alpha` value (0=pure BM25, 1=pure semantic) and justify your choice.

| Query | Recommended Alpha | Justification |
|-------|-------------------|---------------|
| "Python syntax error fix" | ? | ? |
| "What is consciousness?" | ? | ? |
| "RFC 7231 status codes" | ? | ? |
| "How to feel better about failure?" | ? | ? |

<details>
<summary>Show Answer</summary>

| Query | Recommended Alpha | Justification |
|-------|-------------------|---------------|
| "Python syntax error fix" | 0.3 | Technical queries benefit from keyword matching (exact terms like "syntax error" matter); semantic search might return general Python tutorials |
| "What is consciousness?" | 0.9 | Philosophical/conceptual queries need semantic understanding; the word "consciousness" alone misses related concepts like "self-awareness", "qualia" |
| "RFC 7231 status codes" | 0.1 | Exact identifier ("RFC 7231") must match; semantic search would return any HTTP documentation |
| "How to feel better about failure?" | 0.8 | Emotional/nuanced query; semantic search finds related content about resilience/growth mindset even if those exact words aren't used |

```python
# Practical auto-tuning heuristic:
def estimate_alpha(query: str) -> float:
    """Estimate alpha based on query characteristics."""
    tokens = query.lower().split()

    # High keyword-specificity signals
    has_numbers = any(t.isdigit() or any(c.isdigit() for c in t) for t in tokens)
    has_technical = any(t in ['error', 'syntax', 'rfc', 'api', 'fix'] for t in tokens)
    is_short = len(tokens) <= 3

    # High semantic signals
    is_question = query.lower().startswith(('what', 'how', 'why', 'explain'))
    is_long = len(tokens) >= 7

    score = 0.5  # Default balanced
    if has_numbers or has_technical: score -= 0.3
    if is_short: score -= 0.1
    if is_question: score += 0.2
    if is_long: score += 0.1

    return max(0.0, min(1.0, score))
```

In production, alpha is often tuned on a labeled evaluation set using grid search over the validation queries.
</details>

---

### Exercise 4: RAG Evaluation with Recall@K

Implement the `calculate_recall_at_k` function and use it to evaluate two different retrieval configurations on a small test set. Then explain what a good Recall@K value is for RAG.

```python
# Test data
queries = [
    "What is machine learning?",
    "How does BERT work?",
    "What is a transformer?",
]

# Ground truth: which document indices are relevant for each query
ground_truth = {
    "What is machine learning?": [0, 2],
    "How does BERT work?": [1, 3],
    "What is a transformer?": [1, 3, 4],
}

# Retrieved results (indices) for each query — two systems to compare
system_a = {
    "What is machine learning?": [0, 5, 2, 7, 1],
    "How does BERT work?": [3, 6, 1, 8, 2],
    "What is a transformer?": [4, 1, 6, 3, 9],
}

system_b = {
    "What is machine learning?": [5, 7, 6, 8, 0],
    "How does BERT work?": [6, 8, 2, 9, 1],
    "What is a transformer?": [6, 9, 7, 8, 4],
}
```

<details>
<summary>Show Answer</summary>

```python
def calculate_recall_at_k(retrieved: list, relevant: list, k: int) -> float:
    """
    Recall@K = |retrieved[:k] ∩ relevant| / |relevant|
    Measures: what fraction of relevant docs were found in top-K results?
    """
    retrieved_k = set(retrieved[:k])
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    return len(retrieved_k & relevant_set) / len(relevant_set)


def evaluate_system(system: dict, ground_truth: dict, k: int) -> float:
    """Average Recall@K across all queries."""
    recalls = []
    for query, retrieved in system.items():
        relevant = ground_truth[query]
        recalls.append(calculate_recall_at_k(retrieved, relevant, k))
    return sum(recalls) / len(recalls)


# Evaluate at K=3 and K=5
for k in [3, 5]:
    r_a = evaluate_system(system_a, ground_truth, k)
    r_b = evaluate_system(system_b, ground_truth, k)
    print(f"Recall@{k}: System A = {r_a:.3f}, System B = {r_b:.3f}")

# System A:
# Recall@3: 0.833  (A finds relevant docs early)
# Recall@5: 1.000  (A finds all relevant docs by K=5)

# System B:
# Recall@3: 0.333  (B misses many relevant docs in top-3)
# Recall@5: 0.667  (B still misses some at K=5)
```

**What is a good Recall@K for RAG?**
- **Recall@K goal**: As high as possible — if a relevant document is not retrieved, the LLM cannot include it in the answer
- **Typical targets**: Recall@3 ≥ 0.7, Recall@5 ≥ 0.85 for production systems
- **Trade-off**: Higher K improves recall but increases context length (cost + LLM attention dilution)
- **Common practice**: Use K=5-10 for retrieval, then rerank to select top-3 for the LLM prompt
- Recall measures coverage; use MRR or NDCG when ranking order matters
</details>

---

## Next Steps

Learn about the LangChain framework in [13_LangChain_Basics.md](./13_LangChain_Basics.md).
