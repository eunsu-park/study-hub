"""
02. Word2Vec and GloVe - Word Embedding Examples

Word embedding training and usage
"""

import numpy as np

print("=" * 60)
print("Word Embeddings")
print("=" * 60)


# ============================================
# 1. Cosine Similarity
# ============================================
print("\n[1] Cosine Similarity")
print("-" * 40)

def cosine_similarity(v1, v2):
    """Cosine similarity between two vectors"""
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return dot / norm if norm > 0 else 0

# Example vectors
vec_king = np.array([0.5, 0.3, 0.8, 0.1])
vec_queen = np.array([0.5, 0.4, 0.7, 0.2])
vec_apple = np.array([-0.2, 0.9, 0.1, 0.5])

print(f"king-queen similarity: {cosine_similarity(vec_king, vec_queen):.4f}")
print(f"king-apple similarity: {cosine_similarity(vec_king, vec_apple):.4f}")


# ============================================
# 2. Simple Embedding Layer (PyTorch)
# ============================================
print("\n[2] PyTorch Embedding Layer")
print("-" * 40)

try:
    import torch
    import torch.nn as nn

    # Embedding layer
    vocab_size = 100
    embed_dim = 64
    embedding = nn.Embedding(vocab_size, embed_dim)

    # Input: word indices
    input_ids = torch.tensor([1, 5, 10, 20])
    embedded = embedding(input_ids)

    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {embedded.shape}")
    print(f"Embedding weight shape: {embedding.weight.shape}")

except ImportError:
    print("PyTorch not installed")


# ============================================
# 3. Gensim Word2Vec
# ============================================
print("\n[3] Gensim Word2Vec")
print("-" * 40)

try:
    from gensim.models import Word2Vec

    # Sample corpus
    sentences = [
        ["i", "love", "machine", "learning"],
        ["machine", "learning", "is", "fun"],
        ["deep", "learning", "is", "great"],
        ["i", "love", "deep", "learning"],
        ["neural", "networks", "are", "powerful"],
        ["deep", "neural", "networks", "learn", "features"],
    ]

    # Train Word2Vec
    model = Word2Vec(
        sentences,
        vector_size=50,    # Embedding dimension
        window=3,          # Context window
        min_count=1,       # Minimum frequency
        sg=1,              # Skip-gram (0=CBOW)
        epochs=100
    )

    # Similar words
    print("Words similar to 'learning':")
    similar = model.wv.most_similar("learning", topn=3)
    for word, score in similar:
        print(f"  {word}: {score:.4f}")

    # Get vector
    vec = model.wv["learning"]
    print(f"\n'learning' vector shape: {vec.shape}")

    # Save/Load
    model.save("word2vec_demo.model")
    loaded = Word2Vec.load("word2vec_demo.model")
    print("Model save/load complete")

    # Cleanup
    import os
    os.remove("word2vec_demo.model")

except ImportError:
    print("gensim not installed (pip install gensim)")


# ============================================
# 4. Using Pretrained Embeddings
# ============================================
print("\n[4] Applying Pretrained Embeddings")
print("-" * 40)

try:
    import torch
    import torch.nn as nn

    # Simulated pretrained embeddings (in practice, load GloVe etc.)
    pretrained_embeddings = torch.randn(1000, 100)  # vocab_size=1000, dim=100

    # Apply to embedding layer
    embedding = nn.Embedding.from_pretrained(
        pretrained_embeddings,
        freeze=False,  # True = no training
        padding_idx=0
    )

    print(f"Pretrained embedding shape: {pretrained_embeddings.shape}")
    print(f"freeze=False: fine-tuning enabled")

    # Apply to classification model
    class TextClassifier(nn.Module):
        def __init__(self, pretrained_emb, num_classes):
            super().__init__()
            self.embedding = nn.Embedding.from_pretrained(pretrained_emb, freeze=False)
            self.fc = nn.Linear(pretrained_emb.shape[1], num_classes)

        def forward(self, x):
            embedded = self.embedding(x)  # (batch, seq, embed)
            pooled = embedded.mean(dim=1)  # Average pooling
            return self.fc(pooled)

    model = TextClassifier(pretrained_embeddings, num_classes=2)
    print(f"Classification model created")

except ImportError:
    print("PyTorch not installed")


# ============================================
# 5. Word Analogy
# ============================================
print("\n[5] Word Analogy")
print("-" * 40)

def word_analogy(word_a, word_b, word_c, embeddings, word2idx, idx2word, topk=3):
    """
    a : b = c : ?
    Example: king : queen = man : woman
    """
    # Get vectors
    vec_a = embeddings[word2idx[word_a]]
    vec_b = embeddings[word2idx[word_b]]
    vec_c = embeddings[word2idx[word_c]]

    # Analogy vector: b - a + c
    target = vec_b - vec_a + vec_c

    # Compute similarities
    similarities = np.dot(embeddings, target) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(target)
    )

    # Top k (excluding a, b, c)
    exclude = {word2idx[word_a], word2idx[word_b], word2idx[word_c]}
    results = []
    for idx in np.argsort(similarities)[::-1]:
        if idx not in exclude:
            results.append((idx2word[idx], similarities[idx]))
        if len(results) >= topk:
            break

    return results

# Example (simulated data)
vocab = ["king", "queen", "man", "woman", "prince", "princess", "boy", "girl"]
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for i, w in enumerate(vocab)}

# Simulated embeddings (in practice, use trained embeddings)
np.random.seed(42)
embeddings = np.random.randn(len(vocab), 50)
# Simulate semantic relationships
embeddings[word2idx["queen"]] = embeddings[word2idx["king"]] + np.array([0.1] * 50)
embeddings[word2idx["woman"]] = embeddings[word2idx["man"]] + np.array([0.1] * 50)

result = word_analogy("king", "queen", "man", embeddings, word2idx, idx2word)
print(f"king : queen = man : ?")
for word, score in result:
    print(f"  {word}: {score:.4f}")


# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("Word Embeddings Summary")
print("=" * 60)

summary = """
Key Concepts:
    - Distributed representation: Represent words as dense vectors
    - Word2Vec: Skip-gram, CBOW
    - GloVe: Co-occurrence statistics based

Usage:
    # Gensim Word2Vec
    model = Word2Vec(sentences, vector_size=100, window=5)
    similar = model.wv.most_similar("word", topn=5)

    # PyTorch Embedding
    embedding = nn.Embedding.from_pretrained(vectors, freeze=False)

Word Arithmetic:
    king - queen + man ~ woman
"""
print(summary)
