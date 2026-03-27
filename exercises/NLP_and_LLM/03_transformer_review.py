"""
Exercises for Lesson 03: Transformer Review
Topic: LLM_and_NLP

Solutions to practice problems from the lesson.
"""

import torch
import torch.nn.functional as F
import math
import numpy as np


# === Exercise 1: Causal Mask Behavior ===
# Problem: For a sequence of length 5, write out the full causal mask matrix.
# Explain which tokens the 3rd token (index 2) can attend to and why.

def exercise_1():
    """Causal mask behavior analysis."""
    seq_len = 5

    # Create the causal mask (lower triangular)
    mask = torch.tril(torch.ones(seq_len, seq_len))
    print(f"Causal mask (seq_len={seq_len}):")
    print(mask)

    # Analyze which tokens each position can attend to
    print("\nAttendable tokens per position:")
    for pos in range(seq_len):
        can_attend = [j for j in range(seq_len) if mask[pos, j] == 1]
        print(f"  Token {pos}: can attend to positions {can_attend}")

    # Focus on the 3rd token (index 2)
    print(f"\nFocus: Token at index 2 (3rd token):")
    print(f"  Row 2 of mask: {mask[2].tolist()}")
    print(f"  Can attend to: positions 0, 1, 2 (itself and all previous)")
    print(f"  Cannot attend to: positions 3, 4 (future tokens)")

    # Demonstrate the effect on attention scores
    print("\nEffect in attention computation:")
    scores = torch.randn(seq_len, seq_len)
    print(f"  Raw scores (row 2): {scores[2].tolist()}")

    scores_masked = scores.clone()
    scores_masked[mask == 0] = -1e9
    print(f"  After masking:      {scores_masked[2].tolist()}")

    attn_weights = F.softmax(scores_masked, dim=-1)
    print(f"  After softmax:      {attn_weights[2].tolist()}")
    print(f"  Sum of weights:     {attn_weights[2].sum().item():.4f}")
    print(f"  Weights at pos 3,4: ~0 (masked positions ignored)")

    print("\nWithout the causal mask:")
    print("  During training: model could 'cheat' by looking at future tokens")
    print("  During inference: future tokens don't exist yet, so output would")
    print("  be incoherent or require all tokens to be known upfront.")


# === Exercise 2: Encoder vs Decoder Architecture Differences ===
# Problem: Fill in the comparison table, and explain why BERT cannot be used
# for text generation while GPT cannot be used for bidirectional tasks.

def exercise_2():
    """Encoder vs Decoder architecture comparison."""
    comparison = [
        ("Attention type", "Bidirectional self-attention",
         "Causal (unidirectional) self-attention"),
        ("Training objective", "MLM + NSP",
         "Next token prediction (CLM)"),
        ("Typical use cases", "Classification, NER, QA, similarity",
         "Text generation, dialogue, completion"),
        ("Can see future tokens?", "Yes (full sequence visible)",
         "No (only past tokens visible)"),
    ]

    print("Encoder vs Decoder Comparison:")
    print(f"{'Feature':<25} {'BERT (Encoder)':<35} {'GPT (Decoder)'}")
    print("-" * 95)
    for feature, bert, gpt in comparison:
        print(f"{feature:<25} {bert:<35} {gpt}")

    print("\nWhy BERT cannot generate text:")
    print("  BERT is trained to fill in masked tokens given both left AND right")
    print("  context. At inference time, generation requires predicting token t")
    print("  before token t+1 exists. BERT has no mechanism for autoregressive")
    print("  generation -- it expects a complete (masked) input, not a partial")
    print("  one to extend.")

    print("\nWhy GPT cannot do bidirectional tasks well:")
    print("  GPT's causal masking means each token's representation is computed")
    print("  only from past tokens. For tasks like NER or QA where a token's")
    print("  label may depend on future context (e.g., determining if 'Washington'")
    print("  is a person or place requires seeing subsequent words), GPT's")
    print("  unidirectional attention is fundamentally limited.")


# === Exercise 3: Positional Encoding Properties ===
# Problem: Implement sinusoidal positional encoding and verify two key
# properties: (1) different positions produce different encodings, and
# (2) the relative offset between positions is consistent.

def exercise_3():
    """Positional encoding properties verification."""

    def sinusoidal_encoding(max_len, d_model):
        """Compute sinusoidal positional encodings."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)   # Odd dimensions
        return pe

    pe = sinusoidal_encoding(max_len=100, d_model=64)
    print(f"Positional encoding shape: {pe.shape}")

    # Property 1: Different positions produce different encodings
    print("\n--- Property 1: Uniqueness ---")
    positions_to_check = [(5, 10), (5, 50), (10, 50), (0, 99)]
    for p1, p2 in positions_to_check:
        sim = F.cosine_similarity(pe[p1].unsqueeze(0), pe[p2].unsqueeze(0)).item()
        print(f"  Cosine similarity pos {p1:2d} vs {p2:2d}: {sim:.4f} (< 1.0 = distinct)")

    # Property 2: Consistent relative offset
    print("\n--- Property 2: Consistent relative offset ---")
    offset = 5
    dots = []
    print(f"  Dot products for offset={offset}:")
    for pos in [0, 10, 20, 30, 50, 70]:
        dot = (pe[pos] * pe[pos + offset]).sum().item()
        dots.append(dot)
        print(f"    pe[{pos:2d}] . pe[{pos + offset:2d}] = {dot:.4f}")

    std_dots = torch.tensor(dots).std().item()
    print(f"\n  Std of dot products: {std_dots:.4f} (should be small)")
    print(f"  Mean of dot products: {torch.tensor(dots).mean().item():.4f}")

    # Verify with different offsets
    print("\n  Consistency across different offsets:")
    for k in [1, 3, 5, 10]:
        dots_k = []
        for pos in range(0, 50, 10):
            dot = (pe[pos] * pe[pos + k]).sum().item()
            dots_k.append(dot)
        std_k = np.std(dots_k)
        print(f"    Offset {k:2d}: mean={np.mean(dots_k):.4f}, std={std_k:.4f}")

    print("\n  Key properties verified:")
    print("  1. Each position gets a unique encoding vector")
    print("  2. The dot product pe[pos] . pe[pos+k] is approximately constant")
    print("     for fixed offset k, regardless of absolute position")


# === Exercise 4: Weight Tying in Language Models ===
# Problem: Explain what "weight tying" means, why it is done, and its
# practical benefits.

def exercise_4():
    """Weight tying in language models demonstration."""
    import torch.nn as nn

    class TiedModel(nn.Module):
        def __init__(self, vocab_size=1000, d_model=64):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.head.weight = self.embedding.weight  # Tie weights

    class UntiedModel(nn.Module):
        def __init__(self, vocab_size=1000, d_model=64):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

    vocab_size = 1000
    d_model = 64

    tied = TiedModel(vocab_size, d_model)
    untied = UntiedModel(vocab_size, d_model)

    tied_params = sum(p.numel() for p in tied.parameters())
    untied_params = sum(p.numel() for p in untied.parameters())

    print("Weight Tying in Language Models")
    print("=" * 50)

    print(f"\nModel comparison (vocab_size={vocab_size}, d_model={d_model}):")
    print(f"  Tied model parameters:   {tied_params:,}")
    print(f"  Untied model parameters: {untied_params:,}")
    print(f"  Parameter savings:       {untied_params - tied_params:,} "
          f"({(untied_params - tied_params) / untied_params * 100:.1f}%)")

    # Verify the weights are truly shared
    tied.embedding.weight.data[0] = torch.ones(d_model) * 42.0
    assert torch.equal(tied.embedding.weight[0], tied.head.weight[0]), \
        "Weights should be shared"
    print(f"\n  Weight sharing verified: embedding[0] == head[0] after modification")

    # GPT-2 scale example
    gpt2_vocab = 50257
    gpt2_d = 768
    savings = gpt2_vocab * gpt2_d
    print(f"\n  GPT-2 scale savings: {savings:,} params "
          f"(~{savings * 4 / 1024**2:.1f} MB in FP32)")

    print("\nWhat weight tying means:")
    print("  The output projection layer (head) and the token embedding layer")
    print("  share the same weight matrix. Setting self.head.weight =")
    print("  self.embedding.weight makes them literally the same object.")

    print("\nWhy it is done:")
    print("  There is an elegant symmetry: if a word's embedding vector is")
    print("  close to the hidden state h, the model should assign high")
    print("  probability to that word as the next token.")

    print("\nPractical benefits:")
    print("  1. Parameter reduction: eliminates one vocab_size x d_model matrix")
    print("  2. Regularization: shared matrix adds implicit constraint")
    print("  3. Better embedding quality: shared matrix receives gradient")
    print("     updates from both input and output directions")


if __name__ == "__main__":
    print("=== Exercise 1: Causal Mask Behavior ===")
    exercise_1()
    print("\n=== Exercise 2: Encoder vs Decoder Architecture ===")
    exercise_2()
    print("\n=== Exercise 3: Positional Encoding Properties ===")
    exercise_3()
    print("\n=== Exercise 4: Weight Tying in Language Models ===")
    exercise_4()
    print("\nAll exercises completed!")
