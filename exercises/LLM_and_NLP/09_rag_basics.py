"""
Exercises for Lesson 09: RAG Basics
Topic: LLM_and_NLP

Solutions to practice problems from the lesson.
"""

import numpy as np
import hashlib


# === Exercise 1: Chunking Strategy Analysis ===
# Problem: Compare fixed-size chunking (chunk_size=100, overlap=20) vs
# sentence-based chunking. Explain which produces better semantic units.

def exercise_1():
    """Chunking strategy analysis: fixed-size vs sentence-based."""

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

    # --- Fixed-size chunking ---
    def chunk_fixed(text, chunk_size=100, overlap=20):
        """Split text into fixed-size character chunks with overlap."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
        return chunks

    fixed_chunks = chunk_fixed(text, chunk_size=100, overlap=20)

    # --- Sentence-based chunking ---
    def split_sentences(text):
        """Simple sentence splitter using period + space heuristic."""
        sentences = []
        current = []
        for char in text:
            current.append(char)
            if char == '.' and len(current) > 1:
                sentence = ''.join(current).strip()
                if sentence:
                    sentences.append(sentence)
                current = []
        remainder = ''.join(current).strip()
        if remainder:
            sentences.append(remainder)
        return sentences

    def chunk_by_sentences(text, max_sentences=2, overlap_sentences=1):
        """Chunk text by sentence boundaries with overlap."""
        sentences = split_sentences(text)
        chunks = []
        step = max(1, max_sentences - overlap_sentences)
        for i in range(0, len(sentences), step):
            chunk = ' '.join(sentences[i:i + max_sentences])
            chunks.append(chunk)
        return chunks

    sentence_chunks = chunk_by_sentences(text, max_sentences=2, overlap_sentences=1)

    print("Exercise 1: Chunking Strategy Analysis")
    print("=" * 70)

    print(f"\nOriginal text length: {len(text)} characters")
    print(f"Number of sentences: {len(split_sentences(text))}")

    print(f"\n--- Fixed-size chunks (size=100, overlap=20) ---")
    for i, chunk in enumerate(fixed_chunks):
        # Check if chunk breaks a word
        breaks_word = not chunk.endswith((' ', '.')) and i < len(fixed_chunks) - 1
        marker = " <-- BREAKS MID-WORD" if breaks_word else ""
        print(f"  Chunk {i}: [{len(chunk):3d} chars] \"{chunk}\"{marker}")

    print(f"\n--- Sentence-based chunks (max=2 sentences, overlap=1) ---")
    for i, chunk in enumerate(sentence_chunks):
        print(f"  Chunk {i}: [{len(chunk):3d} chars] \"{chunk}\"")

    print(f"\n--- Comparison ---")
    print(f"  Fixed chunks: {len(fixed_chunks)} chunks, may break sentences mid-word")
    print(f"  Sentence chunks: {len(sentence_chunks)} chunks, preserves complete thoughts")

    print(f"\nWhy sentence-based is generally better for RAG:")
    print(f"  - Fixed-size splits sentences mid-word, destroying semantic coherence")
    print(f"  - A query about 'deep learning' might match a chunk starting mid-sentence")
    print(f"  - Sentence-based preserves complete thoughts, improving retrieval accuracy")
    print(f"  - Overlap (1 sentence) ensures no context is lost at boundaries")
    print(f"\nWhen fixed-size is acceptable:")
    print(f"  - Very long documents without clear sentence boundaries (logs, transcripts)")
    print(f"  - When token budget is strict and sentences are very long")


# === Exercise 2: Mean Pooling vs CLS Token ===
# Problem: Implement both mean pooling and CLS token approaches for
# sentence embeddings and explain when each is preferred.

def exercise_2():
    """Mean pooling vs CLS token embedding approaches."""

    # Simulated transformer output for 2 sentences
    # Sentence 1: "The cat sat" (3 tokens + CLS + SEP = 5 tokens)
    # Sentence 2: "AI is transforming" (3 tokens + CLS + SEP = 5 tokens, padded to 5)
    np.random.seed(42)
    hidden_dim = 8  # Small dimension for demonstration
    batch_size = 2
    seq_len = 5

    # Simulated last_hidden_state: (batch, seq_len, hidden_dim)
    last_hidden_state = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)

    # Attention masks: 1 for real tokens, 0 for padding
    # Sentence 1 has 5 real tokens, Sentence 2 has 4 real tokens + 1 padding
    attention_mask = np.array([
        [1, 1, 1, 1, 1],  # All real tokens
        [1, 1, 1, 1, 0],  # Last position is padding
    ], dtype=np.float32)

    # Zero out the padded position's hidden state (as a transformer would)
    last_hidden_state[1, 4, :] = 0.0

    # --- Approach 1: CLS token (first token) ---
    cls_embeddings = last_hidden_state[:, 0, :]
    # Shape: (batch_size, hidden_dim) = (2, 8)

    # --- Approach 2: Mean pooling (average over non-padding tokens) ---
    def mean_pooling(hidden_states, attention_mask):
        """Average hidden states, excluding padding tokens."""
        # Expand mask: (batch, seq_len) -> (batch, seq_len, hidden_dim)
        mask_expanded = attention_mask[:, :, np.newaxis]  # (2, 5, 1)
        mask_expanded = np.broadcast_to(mask_expanded, hidden_states.shape)

        # Sum embeddings for non-padding tokens
        sum_embeddings = (hidden_states * mask_expanded).sum(axis=1)

        # Count non-padding tokens per sample
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)

        return sum_embeddings / sum_mask

    mean_embeddings = mean_pooling(last_hidden_state, attention_mask)

    # --- Approach 3: Max pooling ---
    def max_pooling(hidden_states, attention_mask):
        """Take max over non-padding tokens."""
        # Replace padding positions with large negative value
        mask_expanded = attention_mask[:, :, np.newaxis]
        mask_expanded = np.broadcast_to(mask_expanded, hidden_states.shape)
        masked = np.where(mask_expanded > 0, hidden_states, -1e9)
        return masked.max(axis=1)

    max_embeddings = max_pooling(last_hidden_state, attention_mask)

    print("Exercise 2: Mean Pooling vs CLS Token")
    print("=" * 60)

    print(f"\nSimulated data:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Attention mask: {attention_mask.tolist()}")

    print(f"\nCLS token embeddings (shape {cls_embeddings.shape}):")
    for i in range(batch_size):
        print(f"  Sentence {i}: {np.round(cls_embeddings[i], 3).tolist()}")

    print(f"\nMean pooling embeddings (shape {mean_embeddings.shape}):")
    for i in range(batch_size):
        print(f"  Sentence {i}: {np.round(mean_embeddings[i], 3).tolist()}")

    print(f"\nMax pooling embeddings (shape {max_embeddings.shape}):")
    for i in range(batch_size):
        print(f"  Sentence {i}: {np.round(max_embeddings[i], 3).tolist()}")

    # Demonstrate the padding issue
    # Without mask, mean pooling gives wrong result for sentence 2
    wrong_mean = last_hidden_state.mean(axis=1)
    diff = np.abs(mean_embeddings[1] - wrong_mean[1]).mean()
    print(f"\n--- Why attention mask matters ---")
    print(f"  Mean difference (masked vs unmasked) for sentence 2: {diff:.4f}")
    print(f"  Without mask, padding tokens dilute the embedding!")

    print(f"\n--- When to use each method ---")
    table = [
        ("CLS token", "BERT for classification", "CLS must be trained to encode sentence meaning"),
        ("Mean pooling", "Sentence similarity (SentenceTransformers)", "More robust, captures all tokens equally"),
        ("Max pooling", "Capturing salient features", "Key phrases matter more than overall meaning"),
    ]
    print(f"  {'Method':<15} {'Best for':<40} {'Notes'}")
    print(f"  {'-'*15} {'-'*40} {'-'*45}")
    for method, best, notes in table:
        print(f"  {method:<15} {best:<40} {notes}")


# === Exercise 3: Hybrid Search Alpha Tuning ===
# Problem: Given different query types, recommend alpha values
# (0=pure BM25, 1=pure semantic) for hybrid search.

def exercise_3():
    """Hybrid search alpha tuning for different query types."""

    def estimate_alpha(query):
        """Estimate optimal alpha based on query characteristics."""
        tokens = query.lower().split()

        # High keyword-specificity signals (favor BM25)
        has_numbers = any(
            t.isdigit() or any(c.isdigit() for c in t) for t in tokens
        )
        has_technical = any(
            t in ['error', 'syntax', 'rfc', 'api', 'fix', 'bug', 'code']
            for t in tokens
        )
        is_short = len(tokens) <= 3

        # High semantic signals (favor semantic search)
        is_question = query.lower().startswith(('what', 'how', 'why', 'explain'))
        is_long = len(tokens) >= 7

        score = 0.5  # Default balanced
        if has_numbers or has_technical:
            score -= 0.3
        if is_short:
            score -= 0.1
        if is_question:
            score += 0.2
        if is_long:
            score += 0.1

        return max(0.0, min(1.0, score))

    # Test queries with expected alpha recommendations
    queries = [
        {
            "query": "Python syntax error fix",
            "recommended_alpha": 0.3,
            "justification": (
                "Technical queries benefit from keyword matching (exact terms like "
                "'syntax error' matter); semantic search might return general Python tutorials"
            ),
        },
        {
            "query": "What is consciousness?",
            "recommended_alpha": 0.9,
            "justification": (
                "Philosophical/conceptual query needs semantic understanding; "
                "the word 'consciousness' alone misses related concepts like "
                "'self-awareness', 'qualia'"
            ),
        },
        {
            "query": "RFC 7231 status codes",
            "recommended_alpha": 0.1,
            "justification": (
                "Exact identifier ('RFC 7231') must match; semantic search "
                "would return any HTTP documentation"
            ),
        },
        {
            "query": "How to feel better about failure?",
            "recommended_alpha": 0.8,
            "justification": (
                "Emotional/nuanced query; semantic search finds related content "
                "about resilience/growth mindset even if exact words aren't used"
            ),
        },
    ]

    print("Exercise 3: Hybrid Search Alpha Tuning")
    print("=" * 70)
    print("(alpha: 0 = pure BM25 keyword, 1 = pure semantic)")

    print(f"\n{'Query':<45} {'Recommended':<14} {'Auto-estimated':<16} {'Match'}")
    print(f"{'-'*45} {'-'*14} {'-'*16} {'-'*7}")

    for q in queries:
        estimated = estimate_alpha(q["query"])
        recommended = q["recommended_alpha"]
        # Consider it a match if within 0.3 of the recommendation
        match = "~" if abs(estimated - recommended) <= 0.3 else "X"
        print(f"{q['query']:<45} {recommended:<14.1f} {estimated:<16.2f} {match}")

    print(f"\n--- Detailed justifications ---")
    for q in queries:
        print(f"\n  Query: \"{q['query']}\"")
        print(f"  Alpha: {q['recommended_alpha']}")
        print(f"  Why: {q['justification']}")

    print(f"\n--- Auto-tuning heuristic explanation ---")
    print(f"  The estimate_alpha function uses surface-level query features:")
    print(f"  - Numbers/technical terms -> lower alpha (favor BM25)")
    print(f"  - Question words (what/how/why) -> higher alpha (favor semantic)")
    print(f"  - Short queries -> lower alpha (limited context for embeddings)")
    print(f"  - Long queries -> higher alpha (more context for semantic matching)")
    print(f"\n  In production, alpha is tuned on a labeled evaluation set")
    print(f"  using grid search over validation queries.")


# === Exercise 4: RAG Evaluation with Recall@K ===
# Problem: Implement calculate_recall_at_k and evaluate two retrieval
# systems on a small test set.

def exercise_4():
    """RAG evaluation with Recall@K metric."""

    def calculate_recall_at_k(retrieved, relevant, k):
        """
        Recall@K = |retrieved[:k] intersection relevant| / |relevant|
        Measures: what fraction of relevant docs were found in top-K results?
        """
        retrieved_k = set(retrieved[:k])
        relevant_set = set(relevant)
        if not relevant_set:
            return 0.0
        return len(retrieved_k & relevant_set) / len(relevant_set)

    def evaluate_system(system, ground_truth, k):
        """Average Recall@K across all queries."""
        recalls = []
        for query, retrieved in system.items():
            relevant = ground_truth[query]
            recalls.append(calculate_recall_at_k(retrieved, relevant, k))
        return sum(recalls) / len(recalls)

    # Ground truth: which document indices are relevant for each query
    ground_truth = {
        "What is machine learning?": [0, 2],
        "How does BERT work?": [1, 3],
        "What is a transformer?": [1, 3, 4],
    }

    # Retrieved results (indices) for each query -- two systems to compare
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

    print("Exercise 4: RAG Evaluation with Recall@K")
    print("=" * 60)

    # Per-query analysis
    print("\n--- Per-query Recall@K ---")
    for query in ground_truth:
        relevant = ground_truth[query]
        print(f"\n  Query: \"{query}\"")
        print(f"  Relevant docs: {relevant}")
        print(f"  System A retrieved: {system_a[query]}")
        print(f"  System B retrieved: {system_b[query]}")
        for k in [3, 5]:
            ra = calculate_recall_at_k(system_a[query], relevant, k)
            rb = calculate_recall_at_k(system_b[query], relevant, k)
            print(f"    Recall@{k}: A={ra:.3f}, B={rb:.3f}")

    # System-level comparison
    print(f"\n--- System-level Comparison ---")
    print(f"  {'Metric':<12} {'System A':<12} {'System B':<12} {'Winner'}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*10}")

    for k in [3, 5]:
        r_a = evaluate_system(system_a, ground_truth, k)
        r_b = evaluate_system(system_b, ground_truth, k)
        winner = "A" if r_a > r_b else ("B" if r_b > r_a else "Tie")
        print(f"  Recall@{k:<5} {r_a:<12.3f} {r_b:<12.3f} {winner}")

    print(f"\n--- What is a good Recall@K for RAG? ---")
    targets = [
        ("Recall@3", ">= 0.7", "Minimum for usable RAG"),
        ("Recall@5", ">= 0.85", "Production target"),
        ("Recall@10", ">= 0.95", "With reranking pipeline"),
    ]
    for metric, target, desc in targets:
        print(f"  {metric}: {target} ({desc})")

    print(f"\n  Trade-off: Higher K improves recall but increases context length")
    print(f"  (more cost + LLM attention dilution).")
    print(f"  Common practice: Use K=5-10 for retrieval, then rerank to select")
    print(f"  top-3 for the LLM prompt.")
    print(f"  Recall measures coverage; use MRR or NDCG when ranking order matters.")


if __name__ == "__main__":
    print("=== Exercise 1: Chunking Strategy Analysis ===")
    exercise_1()
    print("\n=== Exercise 2: Mean Pooling vs CLS Token ===")
    exercise_2()
    print("\n=== Exercise 3: Hybrid Search Alpha Tuning ===")
    exercise_3()
    print("\n=== Exercise 4: RAG Evaluation with Recall@K ===")
    exercise_4()
    print("\nAll exercises completed!")
