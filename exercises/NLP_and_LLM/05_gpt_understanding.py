"""
Exercises for Lesson 05: GPT Understanding
Topic: LLM_and_NLP

Solutions to practice problems from the lesson.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# === Exercise 1: Generation Strategy Comparison ===
# Problem: Compare four generation strategies (greedy, temperature, top-k,
# top-p) using simulated logits. Explain when to use each.

def exercise_1():
    """Generation strategy comparison with simulated logits."""

    # Simulated vocabulary and logits (as if from a small language model)
    vocab = ["the", "cat", "sat", "on", "mat", "dog", "ran", "happy", ".", "is"]
    vocab_size = len(vocab)

    # Simulated logits for next token prediction
    torch.manual_seed(42)
    logits = torch.tensor([3.0, 1.5, 0.8, 0.5, 0.3, 1.2, 0.1, -0.5, 2.0, 0.7])

    print("Vocabulary:", vocab)
    print(f"Raw logits: {logits.tolist()}")

    # 1. Greedy decoding
    greedy_idx = logits.argmax().item()
    print(f"\n1. Greedy: '{vocab[greedy_idx]}' (always picks highest probability)")

    # 2. Temperature sampling (T=0.5 - sharper)
    temp = 0.5
    probs_temp = F.softmax(logits / temp, dim=-1)
    print(f"\n2. Temperature (T={temp}):")
    for i, (word, p) in enumerate(zip(vocab, probs_temp)):
        bar = "#" * int(p * 50)
        print(f"   {word:>6s}: {p:.4f} {bar}")

    # 3. Top-k sampling (k=3)
    k = 3
    top_k_logits, top_k_indices = logits.topk(k)
    top_k_probs = F.softmax(top_k_logits, dim=-1)
    print(f"\n3. Top-k (k={k}):")
    for idx, prob in zip(top_k_indices, top_k_probs):
        print(f"   {vocab[idx]:>6s}: {prob:.4f}")
    print(f"   (other {vocab_size - k} tokens excluded)")

    # 4. Top-p (nucleus) sampling (p=0.9)
    p_threshold = 0.9
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = probs.sort(descending=True)
    cumsum = sorted_probs.cumsum(dim=-1)

    # Find cutoff
    mask = cumsum - sorted_probs > p_threshold
    sorted_probs_filtered = sorted_probs.clone()
    sorted_probs_filtered[mask] = 0
    sorted_probs_filtered = sorted_probs_filtered / sorted_probs_filtered.sum()

    print(f"\n4. Top-p (p={p_threshold}):")
    for idx, prob, cum, filt in zip(sorted_indices, sorted_probs, cumsum,
                                     sorted_probs_filtered):
        included = "included" if filt > 0 else "excluded"
        print(f"   {vocab[idx]:>6s}: {prob:.4f} (cumulative: {cum:.4f}) [{included}]")

    # Summary table
    print("\n--- When to use each strategy ---")
    strategies = [
        ("Greedy", "Translation, factual QA",
         "Maximizes likelihood, consistent and reproducible"),
        ("Temp (low)", "Code generation, formal text",
         "Controlled creativity, near-deterministic"),
        ("Temp (high)", "Brainstorming, poetry",
         "High diversity, may sacrifice coherence"),
        ("Top-k", "Dialogue, chatbots",
         "Prevents rare artifacts while allowing variety"),
        ("Top-p", "Creative writing, storytelling",
         "Adapts vocabulary size to context complexity"),
    ]
    print(f"  {'Strategy':<12} {'Best for':<30} {'Why'}")
    print(f"  {'-'*12} {'-'*30} {'-'*45}")
    for strat, best, why in strategies:
        print(f"  {strat:<12} {best:<30} {why}")


# === Exercise 2: KV Cache Memory Savings ===
# Problem: Calculate KV computation savings when generating 100 new tokens
# given a 50-token prompt, assuming 12 attention layers.

def exercise_2():
    """KV Cache computational savings calculation."""
    prompt_len = 50
    new_tokens = 100
    num_layers = 12

    # Without KV Cache: recompute K and V for all previous tokens at each step
    total_kv_without_cache = 0
    for t in range(new_tokens):
        seq_len = prompt_len + t + 1  # Current sequence length
        total_kv_without_cache += seq_len * num_layers

    # With KV Cache: compute K and V only for the NEW token
    total_kv_with_cache = new_tokens * num_layers

    # Prompt processing (same for both)
    prompt_processing = prompt_len * num_layers

    speedup = total_kv_without_cache / total_kv_with_cache

    print("KV Cache Memory Savings")
    print("=" * 50)
    print(f"Prompt length:   {prompt_len} tokens")
    print(f"New tokens:      {new_tokens}")
    print(f"Attention layers: {num_layers}")

    print(f"\nWithout KV Cache:")
    print(f"  Total KV computations: {total_kv_without_cache:,}")
    print(f"  = sum(seq_len * layers) for each generation step")
    print(f"  = sum({prompt_len + 1} to {prompt_len + new_tokens}) * {num_layers}")

    print(f"\nWith KV Cache:")
    print(f"  Total KV computations: {total_kv_with_cache:,}")
    print(f"  = {new_tokens} new tokens * {num_layers} layers")

    print(f"\nSpeedup: {speedup:.1f}x in KV computation")

    # Memory trade-off
    d_model = 768
    d_k = d_model // 12  # 64 per head
    num_heads = 12
    bytes_per_param = 2  # FP16

    cache_size_per_layer = 2 * prompt_len * num_heads * d_k * bytes_per_param
    total_cache = cache_size_per_layer * num_layers

    print(f"\nMemory trade-off (BERT-base scale):")
    print(f"  d_model={d_model}, num_heads={num_heads}, d_k={d_k}")
    print(f"  Cache per layer: {cache_size_per_layer / 1024:.1f} KB")
    print(f"  Total KV cache:  {total_cache / 1024:.1f} KB")
    print(f"\n  KV Cache trades computation for memory -- stores K, V")
    print(f"  for all previous tokens so they don't need recomputation.")


# === Exercise 3: In-Context Learning Prompt Design ===
# Problem: Design zero-shot, few-shot, and chain-of-thought prompts for
# a text classification task (movie review sentiment).

def exercise_3():
    """In-context learning prompt design: zero-shot, few-shot, and CoT."""

    # Version 1: Zero-shot
    zero_shot_prompt = (
        'Classify the following movie review as Positive or Negative.\n\n'
        'Review: "The acting was superb and the story kept me engaged throughout."\n'
        'Sentiment:'
    )

    # Version 2: Few-shot (3 examples)
    few_shot_prompt = (
        'Classify the following movie review as Positive or Negative.\n\n'
        'Review: "Absolutely terrible. I walked out after 30 minutes."\n'
        'Sentiment: Negative\n\n'
        'Review: "One of the best films I\'ve seen this decade. Masterpiece!"\n'
        'Sentiment: Positive\n\n'
        'Review: "Mediocre plot but the cinematography saved it somewhat."\n'
        'Sentiment: Negative\n\n'
        'Review: "The acting was superb and the story kept me engaged throughout."\n'
        'Sentiment:'
    )

    # Version 3: Chain-of-Thought
    cot_prompt = (
        'Classify the following movie review as Positive or Negative.\n'
        'Think step by step before giving your final answer.\n\n'
        'Review: "Absolutely terrible. I walked out after 30 minutes."\n'
        'Reasoning: The reviewer says "absolutely terrible" which is very negative, '
        'and they left early, showing they couldn\'t finish watching.\n'
        'Sentiment: Negative\n\n'
        'Review: "The acting was superb and the story kept me engaged throughout."\n'
        'Reasoning:'
    )

    prompts = {
        "Zero-shot": zero_shot_prompt,
        "Few-shot": few_shot_prompt,
        "Chain-of-Thought": cot_prompt,
    }

    for name, prompt in prompts.items():
        print(f"\n{'=' * 50}")
        print(f"--- {name} Prompt ---")
        print(f"{'=' * 50}")
        print(prompt)

    print("\n--- Why each approach progressively improves ---")
    print("\nZero-shot: Relies entirely on patterns from pre-training.")
    print("  Works for simple tasks where the model has seen similar formats.")

    print("\nFew-shot: Provides concrete input-output examples that:")
    print("  - Disambiguate the task format")
    print("  - Demonstrate the output vocabulary ('Positive', not 'pos')")
    print("  - Calibrate the model's decision boundary")

    print("\nChain-of-Thought: Forces the model to:")
    print("  - Identify relevant evidence in the text")
    print("  - Reason explicitly before committing to an answer")
    print("  - Reduce errors from 'jumping to conclusions'")


# === Exercise 4: Autoregressive Training Setup ===
# Problem: Write a complete training loop for a small character-level GPT model.

def exercise_4():
    """Character-level GPT training loop."""

    # Character-level dataset preparation
    text = "Hello, World! This is a training example for our tiny GPT model. " \
           "The model learns to predict the next character in a sequence."
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    vocab_size = len(chars)

    print(f"Character-level GPT Training")
    print(f"=" * 50)
    print(f"Text length: {len(text)} characters")
    print(f"Vocabulary size: {vocab_size} unique characters")
    print(f"Characters: {''.join(chars)}")

    # Encode text
    data = torch.tensor([stoi[c] for c in text])

    # Simple GPT-like model using PyTorch components
    class TinyCharGPT(nn.Module):
        def __init__(self, vocab_size, d_model=32, num_heads=2, num_layers=2, max_len=64):
            super().__init__()
            self.token_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(max_len, d_model)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 4,
                batch_first=True, dropout=0.1
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, input_ids):
            seq_len = input_ids.size(1)
            # Create causal mask
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1
            ).bool()

            pos = torch.arange(seq_len, device=input_ids.device)
            x = self.token_emb(input_ids) + self.pos_emb(pos)
            x = self.transformer(x, mask=causal_mask)
            return self.head(self.ln_f(x))

    def get_batch(data, block_size=32, batch_size=4):
        """Create input/target pairs for CLM training."""
        max_start = len(data) - block_size - 1
        if max_start <= 0:
            max_start = 1
        starts = torch.randint(0, max_start, (batch_size,))
        x = torch.stack([data[s:s + block_size] for s in starts])
        # Target is input shifted by 1: predict next character
        y = torch.stack([data[s + 1:s + block_size + 1] for s in starts])
        return x, y

    # Training
    model = TinyCharGPT(vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    model.train()
    block_size = min(32, len(data) - 2)
    losses = []

    print(f"\nTraining (block_size={block_size}):")
    for step in range(200):
        x, y = get_batch(data, block_size=block_size, batch_size=4)
        logits = model(x)

        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            y.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())

        if step % 50 == 0:
            perplexity = torch.exp(loss).item()
            print(f"  Step {step:3d}: loss = {loss.item():.4f}, "
                  f"perplexity = {perplexity:.2f}")

    # Generation
    model.eval()
    with torch.no_grad():
        start = torch.tensor([[stoi['H']]])
        generated_ids = start.clone()
        for _ in range(40):
            logits = model(generated_ids)
            next_char_logits = logits[:, -1, :]
            probs = F.softmax(next_char_logits / 0.8, dim=-1)
            next_char = torch.multinomial(probs, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_char], dim=1)
            # Truncate to max_len if needed
            if generated_ids.size(1) > 60:
                generated_ids = generated_ids[:, -60:]

    generated_text = ''.join([itos[i.item()] for i in generated_ids[0]])
    print(f"\nGenerated text (starting with 'H'):")
    print(f"  '{generated_text}'")

    print(f"\nFinal loss: {losses[-1]:.4f}")
    print(f"Loss reduction: {losses[0]:.4f} -> {losses[-1]:.4f}")


if __name__ == "__main__":
    print("=== Exercise 1: Generation Strategy Comparison ===")
    exercise_1()
    print("\n=== Exercise 2: KV Cache Memory Savings ===")
    exercise_2()
    print("\n=== Exercise 3: In-Context Learning Prompt Design ===")
    exercise_3()
    print("\n=== Exercise 4: Autoregressive Training Setup ===")
    exercise_4()
    print("\nAll exercises completed!")
