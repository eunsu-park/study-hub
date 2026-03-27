"""
Exercises for Lesson 04: Pretraining Objectives
Topic: Foundation_Models

Solutions to practice problems from the lesson.
"""

import numpy as np


# === Exercise 1: Objective Function Comparison ===
# Problem: For each scenario, identify the most appropriate pre-training
# objective (CLM, MLM, or Span Corruption) and justify.

def exercise_1():
    """Solution: Objective function comparison"""
    scenarios = [
        {
            "scenario": "Building a chatbot for multi-turn conversations",
            "best_objective": "Causal LM (CLM)",
            "justification": (
                "Generation requires predicting the next token "
                "autoregressively. CLM is the natural objective since the "
                "model learns to generate fluent continuations. GPT-style."
            ),
        },
        {
            "scenario": "Fine-tuning for sentiment classification on reviews",
            "best_objective": "Masked LM (MLM)",
            "justification": (
                "For understanding-focused tasks with fine-tuning, "
                "bidirectional context (MLM) produces richer token "
                "representations. BERT-style models excel here."
            ),
        },
        {
            "scenario": "QA system: document + question -> short answer",
            "best_objective": "Span Corruption (Encoder-Decoder)",
            "justification": (
                "T5-style span corruption teaches reconstruction from "
                "context, which mimics QA structure. Encoder-decoder "
                "allows flexible-length output."
            ),
        },
        {
            "scenario": "Single model for both classification and generation",
            "best_objective": "UL2 / Mixture of Denoisers",
            "justification": (
                "UL2 trains with multiple denoising objectives (R, S, X) "
                "covering both short-span MLM-like and long-span generation. "
                "Mode prefix allows task-type specification at inference."
            ),
        },
    ]

    for s in scenarios:
        print(f"  Scenario: {s['scenario']}")
        print(f"  Best objective: {s['best_objective']}")
        print(f"  Justification: {s['justification']}")
        print()


# === Exercise 2: BERT Masking Strategy ===
# Problem: BERT masks 15% of tokens: 80% [MASK], 10% random, 10% unchanged.
# Explain the rationale for each choice.

def exercise_2():
    """Solution: BERT masking strategy analysis"""
    print("  BERT masking breakdown: 15% selected, of which:")
    print("    80% -> [MASK] token")
    print("    10% -> random token")
    print("    10% -> keep unchanged")
    print()

    print("  Q1: Why keep 10% unchanged?")
    print("    Without this, the model never learns good representations for")
    print("    actual tokens (only for [MASK]). During fine-tuning, [MASK]")
    print("    doesn't appear -- the model must produce useful representations")
    print("    for real tokens at every position.")
    print()

    print("  Q2: Why 10% random tokens?")
    print("    Forces the model to consider every token as potentially 'wrong'")
    print("    and produce contextually grounded predictions for each position.")
    print("    Prevents the model from learning a shortcut that only [MASK]")
    print("    positions need prediction.")
    print()

    print("  Q3: Problem with 50% masking?")
    print("    Too much context destroyed: not enough surrounding tokens to")
    print("    correctly infer masked positions. Training signal becomes noisy.")
    print("    Resulting representations degrade. 15% was empirically optimal.")

    # Demonstration: simulate masking
    np.random.seed(42)
    sentence = "The quick brown fox jumps over the lazy dog".split()
    n_tokens = len(sentence)
    n_masked = max(1, int(n_tokens * 0.15))

    mask_indices = np.random.choice(n_tokens, n_masked, replace=False)
    masked_sentence = list(sentence)

    for idx in mask_indices:
        r = np.random.random()
        if r < 0.8:
            masked_sentence[idx] = "[MASK]"
        elif r < 0.9:
            masked_sentence[idx] = np.random.choice(sentence)  # random
        # else: keep unchanged

    print()
    print(f"  Example: Original: {' '.join(sentence)}")
    print(f"           Masked:   {' '.join(masked_sentence)}")
    print(f"           Indices:  {sorted(mask_indices.tolist())}")


# === Exercise 3: Causal Mask Implementation ===
# Problem: Trace through causal mask applied to a 4-token attention matrix.

def exercise_3():
    """Solution: Causal mask trace-through"""
    # Given scores matrix
    scores = np.array([
        [0.9, 0.3, 0.7, 0.5],
        [0.2, 0.8, 0.1, 0.4],
        [0.6, 0.5, 0.9, 0.3],
        [0.4, 0.7, 0.8, 0.6],
    ])

    print("  Original scores:")
    for row in scores:
        print(f"    [{', '.join(f'{v:.1f}' for v in row)}]")
    print()

    # Step 1: Create causal mask (True = block future positions)
    n = 4
    mask = np.triu(np.ones((n, n)), k=1).astype(bool)
    print("  Causal mask (True = block):")
    for row in mask:
        print(f"    [{', '.join(str(bool(v)).ljust(5) for v in row)}]")
    print()

    # Step 2: Apply mask (set masked positions to -infinity)
    masked_scores = scores.copy()
    masked_scores[mask] = -np.inf

    print("  Masked scores (-inf for future positions):")
    for row in masked_scores:
        vals = []
        for v in row:
            if v == -np.inf:
                vals.append(" -inf")
            else:
                vals.append(f" {v:.1f}")
        print(f"    [{', '.join(vals)}]")
    print()

    # Step 3: Softmax row-wise
    def softmax(x):
        # Handle -inf properly
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    attention_weights = softmax(masked_scores)

    print("  Attention weights after softmax:")
    for i, row in enumerate(attention_weights):
        vals = [f"{v:.3f}" for v in row]
        attended = [j for j, v in enumerate(row) if v > 0.001]
        print(f"    Token {i+1}: [{', '.join(vals)}]  -> attends to positions {attended}")

    print()
    print("  Key insight: position t can only attend to positions <= t")
    print("  preventing information leakage from future tokens.")


# === Exercise 4: Span Corruption vs MLM Signal Density ===
# Problem: Calculate training signal density for BERT MLM vs T5 span corruption
# on 512-token sequence.

def exercise_4():
    """Solution: Signal density comparison"""
    seq_len = 512
    mask_rate = 0.15
    avg_span_len = 3

    # BERT MLM
    bert_masked = int(seq_len * mask_rate)
    bert_density = bert_masked / seq_len

    print("  BERT MLM:")
    print(f"    Input tokens: {seq_len}")
    print(f"    Masked tokens: {seq_len} * {mask_rate} = {bert_masked}")
    print(f"    Signal density: {bert_masked} / {seq_len} = {bert_density:.0%}")
    print()

    # T5 Span Corruption
    t5_corrupted = int(seq_len * mask_rate)
    t5_num_spans = t5_corrupted // avg_span_len
    t5_output_tokens = t5_num_spans + t5_corrupted  # sentinels + original tokens
    t5_density = t5_output_tokens / seq_len

    print("  T5 Span Corruption:")
    print(f"    Input tokens: {seq_len}")
    print(f"    Corrupted tokens: {t5_corrupted}")
    print(f"    Number of spans (avg {avg_span_len} tokens): {t5_num_spans}")
    print(f"    Output sequence: {t5_num_spans} sentinels + {t5_corrupted} original = {t5_output_tokens}")
    print(f"    Signal density: {t5_output_tokens} / {seq_len} = {t5_density:.0%}")
    print()

    print("  Key difference: T5 decoder predicts entire target sequence")
    print("  autoregressively (learns to generate coherent multi-token spans),")
    print("  while BERT predicts all masked positions in parallel.")
    print("  Span corruption better prepares for generative tasks.")


if __name__ == "__main__":
    print("=== Exercise 1: Objective Function Comparison ===")
    exercise_1()
    print("\n=== Exercise 2: BERT Masking Strategy ===")
    exercise_2()
    print("\n=== Exercise 3: Causal Mask Implementation ===")
    exercise_3()
    print("\n=== Exercise 4: Span Corruption vs MLM Signal Density ===")
    exercise_4()
    print("\nAll exercises completed!")
