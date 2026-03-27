"""
Exercises for Lesson 02: Word2Vec and GloVe
Topic: LLM_and_NLP

Solutions to practice problems from the lesson.
"""

import numpy as np


# === Exercise 1: Skip-gram vs CBOW Trade-offs ===
# Problem: Given the sentence "The quick brown fox jumps over the lazy dog"
# and a window size of 2, list the training pairs that Skip-gram would generate
# for the center word "fox". Then list CBOW's input/output. Explain when each
# model performs better.

def exercise_1():
    """Skip-gram vs CBOW training pair generation."""
    sentence = "The quick brown fox jumps over the lazy dog".split()
    window_size = 2
    center_idx = 3  # "fox"

    # Skip-gram: one center -> multiple contexts
    center_word = sentence[center_idx]
    print(f"Center word: '{center_word}' (index {center_idx})")
    print(f"Window size: {window_size}")

    print("\n--- Skip-gram pairs ---")
    context_words = []
    for offset in range(-window_size, window_size + 1):
        if offset != 0:
            ctx_idx = center_idx + offset
            if 0 <= ctx_idx < len(sentence):
                context_words.append(sentence[ctx_idx])
                print(f"  ('{center_word}', '{sentence[ctx_idx]}')")

    # CBOW: multiple contexts -> one center
    print(f"\n--- CBOW ---")
    print(f"  Input (context):  {context_words}")
    print(f"  Output (center):  '{center_word}'")
    print(f"  CBOW averages the context embeddings, then predicts the center word.")

    print("\n--- When each performs better ---")
    print("Skip-gram: Better for rare words and smaller datasets.")
    print("  By generating more training pairs per word, it provides")
    print("  more gradient updates for infrequent words.")
    print("CBOW: Faster to train (fewer forward passes), works better")
    print("  on large datasets. Averaging context embeddings makes it")
    print("  more robust to noise.")


# === Exercise 2: The Word Analogy Task ===
# Problem: Explain why vector("king") - vector("man") + vector("woman")
# should yield a vector close to vector("queen"). Identify one limitation.

def exercise_2():
    """Word analogy task: geometric interpretation of word embeddings."""

    # Simulated embeddings illustrating the analogy concept
    # Each dimension represents a conceptual axis: [royalty, gender, human, ...]
    embeddings = {
        "king":  np.array([1.0, 0.9, 1.0, 0.3]),
        "queen": np.array([1.0, 0.1, 1.0, 0.3]),
        "man":   np.array([0.1, 0.9, 1.0, 0.3]),
        "woman": np.array([0.1, 0.1, 1.0, 0.3]),
    }

    # The analogy computation
    result = embeddings["king"] - embeddings["man"] + embeddings["woman"]

    print("Conceptual embeddings [royalty, gender, human, other]:")
    for word, vec in embeddings.items():
        print(f"  {word:6s} -> {vec}")

    print(f"\nComputation: king - man + woman")
    print(f"  = {embeddings['king']} - {embeddings['man']} + {embeddings['woman']}")
    print(f"  = {result}")
    print(f"  queen  = {embeddings['queen']}")

    # Cosine similarity between result and queen
    cos_sim = np.dot(result, embeddings["queen"]) / (
        np.linalg.norm(result) * np.linalg.norm(embeddings["queen"])
    )
    print(f"\nCosine similarity(result, queen) = {cos_sim:.4f}")

    print("\n--- Why the arithmetic works ---")
    print("The gender direction:")
    gender_dir = embeddings["woman"] - embeddings["man"]
    print(f"  woman - man   = {gender_dir}")
    queen_king_dir = embeddings["queen"] - embeddings["king"]
    print(f"  queen - king  = {queen_king_dir}")
    print("These two directions are approximately equal!")

    print("\n--- Limitations ---")
    print("1. Polysemy: 'bank' (financial vs riverbank) gets a single")
    print("   averaged vector that blends both meanings.")
    print("2. Analogies can fail: 'Tokyo - Japan + France' should give")
    print("   'Paris', but may give unexpected results.")
    print("3. Cultural bias: Embeddings absorb biases present in")
    print("   training text (e.g., gender stereotypes).")


# === Exercise 3: GloVe Loss Function Analysis ===
# Problem: Explain the purpose of the weighting function f(X_ij) in GloVe.
# What problem occurs if all co-occurrence counts are weighted equally?

def exercise_3():
    """GloVe weighting function analysis."""

    def glove_weight(X_ij, x_max=100, alpha=0.75):
        """GloVe weighting function."""
        return min(X_ij / x_max, 1.0) ** alpha

    # Visualize for different co-occurrence counts
    counts = [1, 5, 10, 25, 50, 75, 100, 150, 200, 500]

    print("GloVe Weighting Function: f(X_ij) = min(X_ij / x_max, 1)^alpha")
    print(f"  x_max = 100, alpha = 0.75")
    print(f"\n{'X_ij':>8}  {'alpha=0.50':>12}  {'alpha=0.75':>12}  {'alpha=1.00':>12}")
    print("-" * 50)

    for c in counts:
        w50 = min(c / 100, 1.0) ** 0.50
        w75 = min(c / 100, 1.0) ** 0.75
        w10 = min(c / 100, 1.0) ** 1.00
        print(f"{c:>8d}  {w50:>12.4f}  {w75:>12.4f}  {w10:>12.4f}")

    print("\n--- Purpose of the weighting function ---")
    print("Without weighting, very frequent co-occurrences (like 'the'")
    print("with almost every word) would dominate the loss function.")
    print("These high-frequency pairs contain less semantic information.")
    print("\nThe weighting function achieves two things:")
    print("1. Caps the weight at 1.0 for pairs exceeding x_max")
    print("   - prevents stopword pairs from dominating")
    print("2. Gives lower weight to very rare pairs (X_ij near 0)")
    print("   - rare co-occurrences may be noisy or accidental")

    print("\n--- Effect of alpha ---")
    print("alpha = 1.0: Linear scaling. All sub-threshold pairs weighted")
    print("  proportionally.")
    print("alpha < 1.0 (e.g., 0.75): Concave curve - words with moderate")
    print("  frequency get relatively higher weight. Best empirically.")
    print("alpha -> 0: All non-zero pairs get nearly equal weight.")
    print("\nThe original GloVe paper found alpha = 0.75 works best.")


# === Exercise 4: Pre-trained Embedding Initialization ===
# Problem: Compare random initialization vs pre-trained GloVe initialization
# for a domain-specific classification task.

def exercise_4():
    """Pre-trained embedding initialization strategies comparison."""

    vocab_size = 100
    embed_dim = 50
    num_classes = 3

    # Strategy 1: Random initialization
    np.random.seed(42)
    embedding_random = np.random.normal(0, 0.1, (vocab_size, embed_dim))
    embedding_random[0] = 0  # Padding token stays zero

    # Strategy 2: Pre-trained initialization (simulated GloVe)
    # In practice, you'd load from glove.6B.50d.txt
    embedding_pretrained = np.random.normal(0, 0.3, (vocab_size, embed_dim))
    # Simulate that 70% of vocab words are found in GloVe
    found_count = int(0.7 * vocab_size)
    oov_count = vocab_size - found_count
    # OOV words get random initialization
    embedding_pretrained[found_count:] = np.random.normal(0, 0.1, (oov_count, embed_dim))
    embedding_pretrained[0] = 0  # Padding

    print("Embedding Initialization Strategies:")
    print("=" * 60)

    print(f"\nStrategy 1: Random Initialization")
    print(f"  Shape: ({vocab_size}, {embed_dim})")
    print(f"  Mean: {embedding_random.mean():.6f}")
    print(f"  Std:  {embedding_random.std():.6f}")
    print(f"  All parameters trainable from scratch")

    print(f"\nStrategy 2: Pre-trained GloVe + Random for OOV")
    print(f"  Shape: ({vocab_size}, {embed_dim})")
    print(f"  Pre-trained words: {found_count}/{vocab_size} ({found_count/vocab_size*100:.0f}%)")
    print(f"  OOV words (random init): {oov_count}/{vocab_size}")
    print(f"  Mean: {embedding_pretrained.mean():.6f}")
    print(f"  Std:  {embedding_pretrained.std():.6f}")

    # Cosine similarity between random embeddings (should be ~0)
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    random_sim = cosine_sim(embedding_random[1], embedding_random[2])
    pretrained_sim = cosine_sim(embedding_pretrained[1], embedding_pretrained[2])

    print(f"\nSimilarity between word 1 and word 2:")
    print(f"  Random:      {random_sim:.4f} (essentially random)")
    print(f"  Pre-trained: {pretrained_sim:.4f} (may capture semantic relations)")

    print("\n--- When to use each strategy ---")
    print(f"{'Scenario':<45} {'Recommended'}")
    print("-" * 70)
    strategies = [
        ("Large general-domain dataset (>100k)", "Random init, train from scratch"),
        ("Small dataset (<10k samples)", "Pre-trained GloVe, fine-tune"),
        ("Domain-specific vocab (medical, legal)", "Pre-trained + random for OOV"),
        ("Very small dataset (<1k samples)", "Pre-trained with freeze=True"),
        ("Enough compute for transformer", "Use BERT/RoBERTa instead"),
    ]
    for scenario, strategy in strategies:
        print(f"  {scenario:<45} {strategy}")


if __name__ == "__main__":
    print("=== Exercise 1: Skip-gram vs CBOW Trade-offs ===")
    exercise_1()
    print("\n=== Exercise 2: The Word Analogy Task ===")
    exercise_2()
    print("\n=== Exercise 3: GloVe Loss Function Analysis ===")
    exercise_3()
    print("\n=== Exercise 4: Pre-trained Embedding Initialization ===")
    exercise_4()
    print("\nAll exercises completed!")
