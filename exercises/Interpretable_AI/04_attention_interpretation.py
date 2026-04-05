"""
Exercises for Lesson 04: Attention Interpretation
Topic: Interpretable_AI

Solutions to practice problems from the lesson.
"""

import numpy as np


# === Exercise 1: Attention Rollout for a 2-Layer Toy Example ===
# Problem: Given attention matrices from two transformer layers,
# compute attention rollout to find effective input-to-output attention.

def exercise_1():
    """Compute attention rollout for a 2-layer toy transformer."""
    print("=" * 60)
    print("Exercise 1: Attention Rollout (2-Layer Toy Example)")
    print("=" * 60)

    # 4-token sequence: [CLS, A, B, C]
    tokens = ["[CLS]", "A", "B", "C"]
    n = len(tokens)

    # Layer 1 attention matrix (each row sums to 1)
    # Row i = how much token i attends to each token
    attn_layer1 = np.array([
        [0.3, 0.4, 0.2, 0.1],   # CLS attends mostly to A
        [0.1, 0.5, 0.3, 0.1],   # A attends mostly to itself and B
        [0.1, 0.2, 0.4, 0.3],   # B attends to itself and C
        [0.2, 0.1, 0.3, 0.4],   # C attends mostly to itself and B
    ])

    # Layer 2 attention matrix
    attn_layer2 = np.array([
        [0.5, 0.2, 0.1, 0.2],   # CLS mostly self-attends
        [0.3, 0.3, 0.2, 0.2],   # A distributes evenly
        [0.1, 0.1, 0.6, 0.2],   # B strongly self-attends
        [0.2, 0.3, 0.1, 0.4],   # C attends to A and itself
    ])

    print(f"\n  Tokens: {tokens}")
    print(f"\n  Layer 1 Attention:")
    for i, row in enumerate(attn_layer1):
        print(f"    {tokens[i]:>5} -> [{', '.join(f'{v:.2f}' for v in row)}]")

    print(f"\n  Layer 2 Attention:")
    for i, row in enumerate(attn_layer2):
        print(f"    {tokens[i]:>5} -> [{', '.join(f'{v:.2f}' for v in row)}]")

    # Attention Rollout (Abnar & Zuidema, 2020):
    # 1. Add residual connection: A_hat = 0.5 * A + 0.5 * I
    # 2. Multiply layers: Rollout = A_hat_L2 @ A_hat_L1
    # 3. Re-normalize rows

    I = np.eye(n)
    attn_hat_l1 = 0.5 * attn_layer1 + 0.5 * I
    attn_hat_l2 = 0.5 * attn_layer2 + 0.5 * I

    print(f"\n  Step 1: Add residual connections (0.5*A + 0.5*I)")
    print(f"  A_hat_L1:")
    for i, row in enumerate(attn_hat_l1):
        print(f"    {tokens[i]:>5} -> [{', '.join(f'{v:.3f}' for v in row)}]")

    # Rollout = product of adjusted attention matrices
    rollout = attn_hat_l2 @ attn_hat_l1

    # Re-normalize rows to sum to 1
    rollout = rollout / rollout.sum(axis=1, keepdims=True)

    print(f"\n  Step 2: Rollout = A_hat_L2 @ A_hat_L1 (re-normalized):")
    for i, row in enumerate(rollout):
        print(f"    {tokens[i]:>5} -> [{', '.join(f'{v:.3f}' for v in row)}]")

    # Focus on CLS token's attention (used for classification)
    cls_rollout = rollout[0]
    print(f"\n  [CLS] rollout attention over inputs:")
    for j, tok in enumerate(tokens):
        bar = "#" * int(cls_rollout[j] * 40)
        print(f"    {tok:>5}: {cls_rollout[j]:.3f}  {bar}")

    print(f"\n  Interpretation: After rollout, [CLS] effectively attends most to")
    most_attended = tokens[np.argmax(cls_rollout[1:])+1]
    print(f"  '{most_attended}' among input tokens (excluding self-attention).")


# === Exercise 2: Head Specialization via Entropy Analysis ===
# Problem: Analyze attention head specialization by computing the
# entropy of each head's attention distribution. Low entropy = focused,
# high entropy = diffuse.

def exercise_2():
    """Analyze head specialization from attention entropy."""
    print("\n" + "=" * 60)
    print("Exercise 2: Head Specialization via Entropy")
    print("=" * 60)

    np.random.seed(42)
    n_tokens = 6
    n_heads = 4

    # Create attention patterns for different head types
    # Each head: attention matrix of shape (n_tokens, n_tokens)
    heads = {}

    # Head 0: Position head (attends to adjacent tokens)
    attn_0 = np.zeros((n_tokens, n_tokens))
    for i in range(n_tokens):
        for j in range(n_tokens):
            attn_0[i, j] = np.exp(-abs(i - j))
    attn_0 /= attn_0.sum(axis=1, keepdims=True)
    heads["Position (adjacent)"] = attn_0

    # Head 1: BOS/delimiter head (all tokens attend to position 0)
    attn_1 = np.full((n_tokens, n_tokens), 0.02)
    attn_1[:, 0] = 0.9
    attn_1 /= attn_1.sum(axis=1, keepdims=True)
    heads["BOS/delimiter"] = attn_1

    # Head 2: Uniform head (no specialization)
    attn_2 = np.ones((n_tokens, n_tokens)) / n_tokens
    heads["Uniform (no specialization)"] = attn_2

    # Head 3: Syntactic head (specific dependency pattern)
    attn_3 = np.full((n_tokens, n_tokens), 0.01)
    attn_3[0, 2] = 0.95   # token 0 -> token 2
    attn_3[1, 0] = 0.95   # token 1 -> token 0
    attn_3[2, 4] = 0.95   # token 2 -> token 4
    attn_3[3, 1] = 0.95   # token 3 -> token 1
    attn_3[4, 3] = 0.95   # token 4 -> token 3
    attn_3[5, 2] = 0.95   # token 5 -> token 2
    attn_3 /= attn_3.sum(axis=1, keepdims=True)
    heads["Syntactic (specific deps)"] = attn_3

    def attention_entropy(attn_matrix):
        """Compute mean entropy across all query positions."""
        entropies = []
        for row in attn_matrix:
            row_clipped = np.clip(row, 1e-10, 1.0)
            h = -np.sum(row_clipped * np.log2(row_clipped))
            entropies.append(h)
        return np.mean(entropies), np.std(entropies)

    max_entropy = np.log2(n_tokens)

    print(f"\n  Sequence length: {n_tokens} tokens")
    print(f"  Max possible entropy: {max_entropy:.3f} bits")
    print(f"\n  {'Head Type':<30} {'Mean Entropy':<15} {'Std':<10} "
          f"{'Normalized':<12} {'Specialization':<15}")
    print("  " + "-" * 82)

    for name, attn in heads.items():
        mean_h, std_h = attention_entropy(attn)
        normalized = mean_h / max_entropy
        if normalized < 0.4:
            spec = "High (focused)"
        elif normalized < 0.7:
            spec = "Medium"
        else:
            spec = "Low (diffuse)"

        print(f"  {name:<30} {mean_h:<15.3f} {std_h:<10.3f} "
              f"{normalized:<12.3f} {spec:<15}")

    print(f"\n  Interpretation:")
    print(f"  - Low entropy heads are 'specialists': they attend to specific")
    print(f"    positions or patterns (e.g., syntactic dependencies, BOS token).")
    print(f"  - High entropy heads are 'generalists': they distribute attention")
    print(f"    broadly, which may indicate redundancy or context aggregation.")
    print(f"  - Pruning candidates: heads near max entropy contribute little")
    print(f"    discriminative information and may be safely pruned.")


# === Exercise 3: Raw Attention vs Gradient-Weighted Attention ===
# Problem: Compare raw attention weights with gradient-weighted
# attention for a simple classification example.

def exercise_3():
    """Compare raw attention vs gradient-weighted attention."""
    print("\n" + "=" * 60)
    print("Exercise 3: Raw vs Gradient-Weighted Attention")
    print("=" * 60)

    # Simulated scenario: sentiment classification
    # Sentence: "The movie was not great" (5 tokens)
    tokens = ["The", "movie", "was", "not", "great"]
    n = len(tokens)

    # Raw attention from CLS to each token (last layer, head average)
    raw_attention = np.array([0.10, 0.25, 0.15, 0.20, 0.30])

    # Gradient of class score w.r.t. attention weights
    # Captures which attention connections actually matter for the decision
    attn_gradients = np.array([0.02, 0.15, 0.05, 0.60, 0.35])

    # Gradient-weighted attention = attention * |gradient|
    grad_weighted = raw_attention * np.abs(attn_gradients)
    grad_weighted /= grad_weighted.sum()  # normalize

    print(f"\n  Sentence: '{' '.join(tokens)}'")
    print(f"  True sentiment: Negative (due to negation 'not great')")
    print(f"\n  {'Token':<10} {'Raw Attn':<12} {'Attn Grad':<12} "
          f"{'Grad-Weighted':<15}")
    print("  " + "-" * 50)

    for i, tok in enumerate(tokens):
        print(f"  {tok:<10} {raw_attention[i]:<12.3f} "
              f"{attn_gradients[i]:<12.3f} {grad_weighted[i]:<15.3f}")

    print(f"\n  Analysis:")
    print(f"  - Raw attention: 'great' gets highest weight ({raw_attention[4]:.2f}),")
    print(f"    suggesting positive sentiment. This is MISLEADING.")
    print(f"  - Gradient-weighted: 'not' becomes most important because the")
    print(f"    gradient indicates the model's decision is most sensitive to")
    print(f"    the negation word. This better explains the negative prediction.")
    print(f"\n  Key takeaway: Raw attention shows WHERE the model looks, but")
    print(f"  gradient-weighted attention shows what actually INFLUENCES the output.")


# === Exercise 4: Evaluating "Attention is Not Explanation" ===
# Problem: Implement the Jain & Wallace (2019) argument by showing
# that alternative attention distributions can produce similar outputs.

def exercise_4():
    """Evaluate the 'attention is not explanation' argument empirically."""
    print("\n" + "=" * 60)
    print("Exercise 4: Attention is Not Explanation")
    print("=" * 60)

    np.random.seed(42)

    # Simulate: value vectors for 5 tokens
    tokens = ["The", "movie", "was", "not", "great"]
    n = len(tokens)
    d = 4  # value dimension

    value_vectors = np.array([
        [0.1, -0.2,  0.3,  0.1],   # The
        [0.8,  0.5, -0.3,  0.2],   # movie
        [0.0,  0.1,  0.0,  0.1],   # was
        [-0.6, 0.4,  0.7, -0.3],   # not
        [0.9,  0.3, -0.5,  0.4],   # great
    ])

    # Original attention distribution
    original_attn = np.array([0.10, 0.25, 0.15, 0.20, 0.30])

    # Output = weighted sum of value vectors
    original_output = original_attn @ value_vectors

    print(f"\n  Tokens: {tokens}")
    print(f"  Original attention: {original_attn}")
    print(f"  Original output: {original_output}")

    # Generate many random attention distributions and check which
    # produce similar outputs (demonstrating non-uniqueness)
    n_trials = 10000
    similar_count = 0
    best_alternative = None
    best_jsd = float("inf")

    def kl_divergence(p, q):
        p_safe = np.clip(p, 1e-10, 1.0)
        q_safe = np.clip(q, 1e-10, 1.0)
        return np.sum(p_safe * np.log(p_safe / q_safe))

    def js_divergence(p, q):
        m = 0.5 * (p + q)
        return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

    output_threshold = 0.1  # cosine distance threshold

    for _ in range(n_trials):
        # Random attention (Dirichlet distribution for valid probabilities)
        alt_attn = np.random.dirichlet(np.ones(n))
        alt_output = alt_attn @ value_vectors

        # Check output similarity (cosine similarity)
        cos_sim = (np.dot(original_output, alt_output) /
                   (np.linalg.norm(original_output) *
                    np.linalg.norm(alt_output) + 1e-10))

        if cos_sim > 0.95:  # very similar output
            similar_count += 1
            jsd = js_divergence(original_attn, alt_attn)
            if jsd > best_jsd * 0.5:  # find diverse alternative
                if jsd < best_jsd or best_alternative is None:
                    pass
                best_alternative = alt_attn.copy()
                best_jsd = jsd

    # Find the most different attention that gives similar output
    best_alt = None
    max_jsd_found = 0
    for _ in range(50000):
        alt_attn = np.random.dirichlet(np.ones(n))
        alt_output = alt_attn @ value_vectors
        cos_sim = (np.dot(original_output, alt_output) /
                   (np.linalg.norm(original_output) *
                    np.linalg.norm(alt_output) + 1e-10))
        if cos_sim > 0.95:
            jsd = js_divergence(original_attn, alt_attn)
            if jsd > max_jsd_found:
                max_jsd_found = jsd
                best_alt = alt_attn.copy()

    print(f"\n  Adversarial attention search ({n_trials + 50000} random trials):")
    print(f"  Alternatives with output cosine sim > 0.95: {similar_count}")
    print(f"  Fraction: {similar_count / n_trials:.2%}")

    if best_alt is not None:
        alt_output = best_alt @ value_vectors
        cos_sim = (np.dot(original_output, alt_output) /
                   (np.linalg.norm(original_output) *
                    np.linalg.norm(alt_output) + 1e-10))
        print(f"\n  Most different alternative attention found:")
        print(f"  {'Token':<10} {'Original':<12} {'Alternative':<12}")
        print("  " + "-" * 34)
        for i, tok in enumerate(tokens):
            print(f"  {tok:<10} {original_attn[i]:<12.3f} {best_alt[i]:<12.3f}")
        print(f"\n  JSD between distributions: {max_jsd_found:.4f}")
        print(f"  Output cosine similarity: {cos_sim:.4f}")

    print(f"\n  Conclusion (Jain & Wallace, 2019):")
    print(f"  Different attention distributions can produce nearly identical")
    print(f"  outputs, meaning attention weights are NOT uniquely determined")
    print(f"  by the model's decision. Therefore, attention weights alone")
    print(f"  cannot be reliable explanations of model behavior.")
    print(f"\n  Counterargument (Wiegreffe & Pinter, 2019):")
    print(f"  Attention may still be a useful (if imperfect) explanation")
    print(f"  signal, and the existence of alternatives does not mean")
    print(f"  the original attention is meaningless.")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
