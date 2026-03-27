"""
Exercises for Lesson 12: Advanced RAG
Topic: NLP_and_LLM

Practice problems for advanced retrieval-augmented generation techniques.
"""

import math
import re
from typing import List, Tuple, Dict


# === Exercise 1: Chunking Strategies ===
# Problem: Implement multiple text chunking strategies and compare them.
# 1. Fixed-size chunking
# 2. Sentence-based chunking
# 3. Recursive chunking with overlap

def exercise_1():
    """Implement and compare chunking strategies."""
    print("=" * 60)
    print("Exercise 1: Chunking Strategies")
    print("=" * 60)

    text = (
        "Machine learning is a subset of artificial intelligence. "
        "It uses statistical methods to learn from data. "
        "Deep learning is a subset of machine learning. "
        "It uses neural networks with many layers. "
        "Transformers are a type of deep learning architecture. "
        "They use self-attention mechanisms. "
        "BERT and GPT are based on transformers. "
        "They have revolutionized NLP tasks. "
        "RAG combines retrieval with generation. "
        "It improves factual accuracy of LLMs."
    )

    # TODO: Implement fixed-size chunking (by character count with overlap)
    def fixed_size_chunks(text: str, chunk_size: int = 100, overlap: int = 20) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end].strip())
            start = end - overlap
        return [c for c in chunks if c]

    # TODO: Implement sentence-based chunking (max N sentences per chunk)
    def sentence_chunks(text: str, max_sentences: int = 3) -> list[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks = []
        for i in range(0, len(sentences), max_sentences):
            chunk = " ".join(sentences[i:i + max_sentences])
            chunks.append(chunk)
        return chunks

    # TODO: Implement recursive chunking (try to split at paragraph, then sentence, then word)
    def recursive_chunks(text: str, max_size: int = 150) -> list[str]:
        if len(text) <= max_size:
            return [text.strip()] if text.strip() else []

        # Try splitting at sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) + 1 <= max_size:
                current = (current + " " + sent).strip()
            else:
                if current:
                    chunks.append(current)
                current = sent
        if current:
            chunks.append(current)
        return chunks

    print("Fixed-size chunks (100 chars, 20 overlap):")
    for i, c in enumerate(fixed_size_chunks(text)):
        print(f"  [{i}] ({len(c)} chars) {c[:60]}...")

    print("\nSentence-based chunks (3 sentences each):")
    for i, c in enumerate(sentence_chunks(text)):
        print(f"  [{i}] ({len(c)} chars) {c[:60]}...")

    print("\nRecursive chunks (max 150 chars):")
    for i, c in enumerate(recursive_chunks(text)):
        print(f"  [{i}] ({len(c)} chars) {c[:60]}...")


# === Exercise 2: Relevance Scoring ===
# Problem: Implement BM25 scoring for document retrieval.

def exercise_2():
    """Implement BM25 scoring."""
    print("\n" + "=" * 60)
    print("Exercise 2: BM25 Scoring")
    print("=" * 60)

    documents = [
        "Machine learning uses statistical methods to learn patterns from data",
        "Deep learning neural networks have multiple hidden layers",
        "Natural language processing deals with text and speech data",
        "Transformer architecture uses self-attention mechanisms",
        "Retrieval augmented generation combines search with LLMs",
    ]

    def tokenize(text: str) -> list[str]:
        return re.findall(r'\w+', text.lower())

    # TODO: Implement BM25 scoring
    # BM25(q, d) = sum over terms in q of: IDF(t) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * |d|/avgdl))
    def bm25_score(query: str, documents: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
        query_terms = tokenize(query)
        doc_tokens = [tokenize(d) for d in documents]
        N = len(documents)
        avgdl = sum(len(d) for d in doc_tokens) / N

        # Calculate IDF for each query term
        scores = []
        for doc_tok in doc_tokens:
            score = 0.0
            dl = len(doc_tok)
            for term in query_terms:
                # Document frequency
                df = sum(1 for dt in doc_tokens if term in dt)
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                # Term frequency in current doc
                tf = doc_tok.count(term)
                # BM25 formula
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * dl / avgdl)
                score += idf * numerator / denominator
            scores.append(round(score, 4))
        return scores

    query = "neural network learning"
    scores = bm25_score(query, documents)
    ranked = sorted(enumerate(scores), key=lambda x: -x[1])

    print(f"Query: '{query}'")
    print("\nRanked results:")
    for idx, score in ranked:
        print(f"  Score {score:.4f}: {documents[idx][:60]}...")


# === Exercise 3: Query Expansion ===
# Problem: Implement simple query expansion using synonyms.

def exercise_3():
    """Implement query expansion with synonyms."""
    print("\n" + "=" * 60)
    print("Exercise 3: Query Expansion")
    print("=" * 60)

    # Simple synonym dictionary
    SYNONYMS = {
        "fast": ["quick", "rapid", "speedy"],
        "big": ["large", "huge", "enormous"],
        "learn": ["study", "understand", "acquire"],
        "model": ["architecture", "system", "framework"],
        "data": ["information", "dataset", "records"],
    }

    # TODO: Expand the query by adding synonyms for matching words
    def expand_query(query: str, synonyms: dict, max_expansions: int = 2) -> list[str]:
        words = query.lower().split()
        expanded_queries = [query]

        for word in words:
            if word in synonyms:
                for syn in synonyms[word][:max_expansions]:
                    new_query = query.lower().replace(word, syn)
                    expanded_queries.append(new_query)

        return expanded_queries

    queries = [
        "fast model for learning",
        "big data processing",
        "learn a new model",
    ]

    for q in queries:
        expanded = expand_query(q, SYNONYMS)
        print(f"Original: '{q}'")
        print(f"Expanded: {expanded}")
        print()


# === Exercise 4: Reranking ===
# Problem: Implement a simple cross-encoder style reranker
# that scores query-document pairs based on token overlap.

def exercise_4():
    """Implement simple reranking."""
    print("\n" + "=" * 60)
    print("Exercise 4: Reranking")
    print("=" * 60)

    query = "How do transformers handle long sequences?"
    candidates = [
        "Transformers use self-attention to process sequences in parallel.",
        "RNNs process sequences one token at a time, which is slow.",
        "Long sequences are challenging for transformers due to quadratic attention cost.",
        "The transformer architecture was introduced in 2017 by Vaswani et al.",
        "Flash attention and sparse attention help transformers handle long inputs.",
    ]

    def tokenize(text: str) -> set[str]:
        return set(re.findall(r'\w+', text.lower()))

    # TODO: Score based on query-document token overlap, weighted by term rarity
    def rerank_score(query: str, document: str, all_docs: list[str]) -> float:
        q_tokens = tokenize(query)
        d_tokens = tokenize(document)

        # IDF-weighted overlap
        all_tokens = [tokenize(d) for d in all_docs]
        N = len(all_docs)

        score = 0.0
        for token in q_tokens & d_tokens:
            df = sum(1 for dt in all_tokens if token in dt)
            idf = math.log(N / (df + 1)) + 1
            score += idf

        # Normalize by query length
        return round(score / max(len(q_tokens), 1), 4)

    # Initial ranking (by position)
    print(f"Query: '{query}'")
    print("\nInitial order:")
    for i, doc in enumerate(candidates):
        print(f"  [{i}] {doc[:70]}")

    # Rerank
    scores = [(i, rerank_score(query, doc, candidates)) for i, doc in enumerate(candidates)]
    scores.sort(key=lambda x: -x[1])

    print("\nReranked:")
    for idx, score in scores:
        print(f"  Score {score:.4f}: {candidates[idx][:70]}")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
