# 05. GPT Understanding

## Learning Objectives

- Understanding GPT architecture
- Autoregressive language modeling
- Text generation techniques
- Evolution of GPT series

---

## Theory & Principles

GPT (Generative Pre-trained Transformer) is a Transformer **decoder** stack trained on a single objective — **next-token prediction** — and used for everything by autoregressive generation. Where BERT optimizes for understanding via bidirectional context, GPT trades that for the ability to *generate* and to be steered by *prompts*. Every behavior we associate with modern LLMs — chat, in-context learning, code completion — falls out of next-token prediction at sufficient scale.

This section covers:

- **(A) Autoregressive language modeling** — the objective, why it suffices, the chain-rule decomposition.
- **(B) Causal masking** — implementation, geometric meaning, and why it makes parallel training possible.
- **(C) Decoding strategies** — greedy, beam, sampling, top-k, top-p (nucleus), temperature; the mathematics of each.
- **(D) KV-cache** — autoregressive inference is `O(L²)` naively; the cache makes it `O(L)`.
- **(E) In-context learning** — why a sufficiently large LM can pattern-match few-shot examples without any gradient updates.
- **(F) Scaling laws** — Kaplan / Chinchilla compute–data–parameter trade-offs.

### A. Autoregressive Language Modeling

**A.1 The objective.** The joint probability of a sequence factorizes by the chain rule:

```
p(w_1, w_2, ..., w_L) = ∏_{t=1..L}  p(w_t | w_1, ..., w_{t-1})
```

GPT models each conditional `p(w_t | w_{<t})` and is trained by maximizing the log-likelihood of the corpus:

```
L_LM = − Σ_t  log p(w_t | w_{<t})
```

**A.2 Why this suffices for everything.** Any task that can be expressed as continuing a text prompt — classification ("Sentiment: ___"), translation ("English: ... French: ___"), QA ("Question: ... Answer: ___") — collapses to next-token prediction once the right prompt is chosen. This is the foundation of zero-shot and in-context learning.

**A.3 Loss is per-token.** Every position in the sequence contributes a loss term, so a sequence of length `L` provides `L` training signals. This is much more efficient than NSP-style sentence-level objectives, which provide one signal per pair.

### B. Causal Masking

To compute `p(w_t | w_{<t})` for *all* `t` in parallel, we need the hidden state at position `t` to depend only on positions `1...t-1`. The mechanism is **causal masking**: in self-attention, set `S[i, j] = -∞` for all `j > i` before softmax:

```
mask = torch.triu(torch.ones(L, L), diagonal=1).bool()
S.masked_fill_(mask, float('-inf'))
attn = softmax(S / sqrt(d_k)) @ V
```

After softmax, the masked entries become 0, so each row only mixes values from the corresponding row index and earlier. Crucially, this lets us train on a length-`L` sequence with **a single forward pass** and compute `L` next-token predictions and their losses simultaneously. Without causal masking we would need `L` separate forward passes, each with one fewer token visible.

The same masking is applied at every layer — bidirectional information cannot leak even through deep stacking.

### C. Decoding Strategies

The model outputs a categorical distribution over the next token. Sampling from that distribution is its own design space.

**C.1 Greedy.** `w_t = argmax p(w_t | w_{<t})`. Deterministic, fast, but myopic — locally optimal tokens often lead to globally degenerate text (repetition loops).

**C.2 Beam search.** Maintain `B` candidate sequences; at each step, expand each by every vocabulary token, keep the top `B` by joint log-probability. Better than greedy for tasks with a single correct answer (translation, summarization), but produces unnaturally "average" text for open-ended generation.

**C.3 Temperature.** Scale logits before softmax: `p(w) ∝ exp(logit(w) / T)`. `T < 1` sharpens (more deterministic), `T > 1` flattens (more random), `T = 0` reduces to greedy. A core knob for the determinism vs creativity trade-off.

**C.4 Top-k sampling.** Restrict sampling to the `k` highest-probability tokens, renormalize, sample. Avoids the long tail of nonsense tokens. Choice of `k` is awkward — too small at confident steps wastes capacity, too large at unconfident steps reintroduces noise.

**C.5 Top-p (nucleus) sampling.** Restrict to the smallest set of tokens whose cumulative probability ≥ `p` (typical `p = 0.9` or `0.95`). Adapts dynamically: at confident steps the set is small (1-3 tokens), at uncertain steps it expands. Standard choice for chat models.

**C.6 Combined: temperature + top-p.** Modern API default. `temperature ≈ 0.7`, `top_p ≈ 0.9` yields fluent yet diverse generations.

The trade-off is fundamental: high randomness produces creative but error-prone text; low randomness produces accurate but repetitive text.

### D. KV-Cache

At inference, generating token `t` requires running self-attention over the entire prefix `1...t-1`. Naïvely this is `O(L²)` per generated token, `O(L³)` total — quadratic in the worst case.

But: at step `t`, the keys and values for positions `1...t-1` are *identical* to those computed at step `t-1`. Only the new token's `K, V` need to be computed, and only the new token's `Q` needs to attend to the cached `K`s.

**KV-cache:**

```
At step t:
  q_t, k_t, v_t = compute Q, K, V for position t only
  K_cache.append(k_t)
  V_cache.append(v_t)
  output_t = softmax(q_t · K_cache^T / sqrt(d_k)) · V_cache
```

Per-token cost drops to `O(L · d)`, total generation cost `O(L² · d)` — quadratic in sequence length but linear in model dim (instead of cubic in length). The memory cost is `2 · L · d_model · n_layers` (keys + values per layer per token) and dominates large-context LLM serving. Tricks like multi-query attention (MQA), grouped-query attention (GQA), and PagedAttention exist specifically to compress or virtualize this cache.

### E. In-Context Learning

GPT-3 (Brown et al., 2020) showed that a sufficiently large LM can perform new tasks given only natural-language examples in the prompt — *no gradient updates*:

```
Prompt:
  "English: hello / French: bonjour
   English: thank you / French: merci
   English: goodbye / French:"
Output: "au revoir"
```

**Why does this work?** A large LM has implicitly seen many "task description + examples + completion" patterns in its training data. The forward pass at test time pattern-matches the prompt to similar structures and continues accordingly. Mechanistically, recent work (induction heads in Olsson et al., 2022) shows that two attention heads in series can learn to perform "find a previous occurrence of `[A]` followed by `[B]`, and when `[A]` appears again, predict `[B]`." This is a primitive form of meta-learning baked into attention.

In-context learning *requires* scale — small LMs cannot pattern-match novel tasks this way. The emergence of the capability around the GPT-3 scale (175B parameters, 300B tokens) is one of the most studied "phase transitions" in deep learning.

### F. Scaling Laws

Kaplan et al. (2020) and Hoffmann et al. (Chinchilla, 2022) studied how loss `L` depends on model parameters `N`, dataset size `D`, and compute `C`:

```
L(N, D) ≈ L_∞ + (A / N^α) + (B / D^β)
```

Two empirical conclusions:

1. **Loss is power-law in each axis.** Doubling parameters predictably reduces loss; doubling data does too.
2. **Compute-optimal trade-off (Chinchilla).** For a fixed compute budget, the optimal allocation has roughly **20 training tokens per parameter**. Earlier models (GPT-3) were *under-trained* — the same compute split toward more data and fewer parameters would have done better. LLaMA-1 (65B at 1.4T tokens) and LLaMA-2 (7B at 2T tokens) follow the Chinchilla recipe.

### From Theory to the Functions Below

- §1 (overview) — situates GPT among Transformer variants from the previous lesson.
- §2 (autoregressive LM) — implements §A's chain-rule loss with `cross_entropy` over shifted logits.
- §3 (architecture) — assembles the decoder stack with the §B causal mask.
- §4 (text generation) — codes §C's decoding strategies (greedy, beam, top-k, top-p, temperature).
- §5 (GPT series) — surveys the §F scaling-law journey from GPT-1 to GPT-4.
- §6 (HuggingFace GPT-2) — wires §A-§D to `GPT2LMHeadModel.generate()`.
- §7 (in-context learning) — demonstrates §E's zero-shot / few-shot prompting.
- §8 (KV cache) — implements the §D cache and benchmarks the speedup.

---

## 1. GPT Overview

### Generative Pre-trained Transformer

```
GPT = Stack of Transformer decoders

Features:
- Unidirectional (left→right)
- Autoregressive generation
- Trained via next token prediction
```

### BERT vs GPT

| Item | BERT | GPT |
|------|------|-----|
| Architecture | Encoder | Decoder |
| Direction | Bidirectional | Unidirectional |
| Training | MLM | Next token prediction |
| Use Cases | Understanding (classification, QA) | Generation (dialogue, writing) |

---

## 2. Autoregressive Language Modeling

### Training Objective

```
P(x) = P(x₁) × P(x₂|x₁) × P(x₃|x₁,x₂) × ...

Sentence: "I love NLP"
P("I") × P("love"|"I") × P("NLP"|"I love") × P("<eos>"|"I love NLP")

Loss: -log P(next token | previous tokens)
```

### Causal Language Modeling

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def causal_lm_loss(logits, targets):
    """
    logits: (batch, seq, vocab_size)
    targets: (batch, seq) - next token

    Input: [BOS, I, love, NLP]
    Target: [I, love, NLP, EOS]
    """
    batch_size, seq_len, vocab_size = logits.shape

    # (batch*seq, vocab) vs (batch*seq,)
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        targets.view(-1),
        ignore_index=-100  # Ignore padding
    )
    return loss
```

---

## 3. GPT Architecture

### Structure

> 1. Input tokens
> 2. Token Embedding + Position Embedding
> 3. **Transformer Block** (x N layers):
>    - Masked Multi-Head Attention
>    - Add & LayerNorm
>    - Feed Forward
>    - Add & LayerNorm
> 4. LayerNorm
> 5. Linear (vocab_size)
> 6. Softmax --> Next token probability

### Implementation

```python
class GPTBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # Pre-LayerNorm (GPT-2 style)
        ln_x = self.ln1(x)
        attn_out, _ = self.attn(ln_x, ln_x, ln_x, attn_mask=attn_mask)
        x = x + self.dropout(attn_out)

        ln_x = self.ln2(x)
        x = x + self.ffn(ln_x)

        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12,
                 num_layers=12, d_ff=3072, max_len=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            GPTBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.token_emb.weight

        # Register causal mask
        mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        self.register_buffer('causal_mask', mask)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        assert seq_len <= self.max_len

        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.drop(x)

        # Causal mask
        mask = self.causal_mask[:seq_len, :seq_len]

        # Transformer blocks
        for block in self.blocks:
            x = block(x, attn_mask=mask)

        x = self.ln_f(x)
        logits = self.head(x)  # (batch, seq, vocab)

        return logits
```

---

## 4. Text Generation

### Greedy Decoding

```python
def generate_greedy(model, input_ids, max_new_tokens):
    """Always select highest probability token"""
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
    return input_ids
```

### Temperature Sampling

```python
def generate_with_temperature(model, input_ids, max_new_tokens, temperature=1.0):
    """Control distribution with temperature"""
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
    return input_ids

# temperature < 1: more deterministic (prefer high probability tokens)
# temperature > 1: more random (increase diversity)
```

### Top-k Sampling

```python
def generate_top_k(model, input_ids, max_new_tokens, k=50, temperature=1.0):
    """Sample only from top k tokens"""
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)[:, -1, :] / temperature

            # Top-k filtering
            top_k_logits, top_k_indices = logits.topk(k, dim=-1)
            probs = F.softmax(top_k_logits, dim=-1)

            # Sampling
            idx = torch.multinomial(probs, num_samples=1)
            next_token = top_k_indices.gather(-1, idx)

            input_ids = torch.cat([input_ids, next_token], dim=1)
    return input_ids
```

### Top-p (Nucleus) Sampling

```python
def generate_top_p(model, input_ids, max_new_tokens, p=0.9, temperature=1.0):
    """Sample from tokens with cumulative probability up to p"""
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)

            # Sort probabilities in descending order
            sorted_probs, sorted_indices = probs.sort(descending=True)
            cumsum = sorted_probs.cumsum(dim=-1)

            # Mask tokens after p
            mask = cumsum - sorted_probs > p
            sorted_probs[mask] = 0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

            # Sampling
            idx = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_indices.gather(-1, idx)

            input_ids = torch.cat([input_ids, next_token], dim=1)
    return input_ids
```

---

## 5. GPT Series

### GPT-1 (2018)

```
- 12 layers, 768 dim, 117M parameters
- Trained on BooksCorpus
- Introduced fine-tuning paradigm
```

### GPT-2 (2019)

```
- Up to 48 layers, 1.5B parameters
- Trained on WebText (40GB)
- Discovered zero-shot capabilities
- "Too dangerous to release"

Size variants:
- Small: 117M (same as GPT-1)
- Medium: 345M
- Large: 762M
- XL: 1.5B
```

### GPT-3 (2020)

```
- 96 layers, 175B parameters
- Few-shot / In-context Learning
- Available only via API

Key findings:
- Perform various tasks with prompts alone
- Scaling laws: model size ↑ = performance ↑
```

### GPT-4 (2023)

```
- Multimodal (text + images)
- Longer context (8K, 32K, 128K)
- Improved reasoning capabilities
- Aligned with RLHF
```

---

## 6. HuggingFace GPT-2

### Basic Usage

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Text generation
input_text = "The quick brown fox"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Generate
output = model.generate(
    input_ids,
    max_length=50,
    num_return_sequences=1,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

### Generation Parameters

```python
output = model.generate(
    input_ids,
    max_length=100,           # Maximum length
    min_length=10,            # Minimum length
    do_sample=True,           # Use sampling
    temperature=0.8,          # Temperature
    top_k=50,                 # Top-k
    top_p=0.95,               # Top-p
    num_return_sequences=3,   # Number of sequences
    no_repeat_ngram_size=2,   # Prevent n-gram repetition
    repetition_penalty=1.2,   # Repetition penalty
    pad_token_id=tokenizer.eos_token_id
)
```

### Conditional Generation

```python
# Prompt-based generation
prompt = """
Q: What is the capital of France?
A:"""

input_ids = tokenizer.encode(prompt, return_tensors='pt')
output = model.generate(
    input_ids,
    max_new_tokens=20,
    do_sample=False  # Greedy
)
print(tokenizer.decode(output[0]))
```

---

## 7. In-Context Learning

### Zero-shot

```
Perform task with prompt alone:

"Translate English to French:
Hello, how are you? →"
```

### Few-shot

```
Include examples in prompt:

"Translate English to French:
Hello → Bonjour
Thank you → Merci
Good morning → Bonjour
How are you? →"
```

### Chain-of-Thought (CoT)

```
Guide step-by-step reasoning:

"Q: Roger has 5 tennis balls. He buys 2 more cans of 3 balls each.
How many balls does he have now?
A: Let's think step by step.
Roger started with 5 balls.
2 cans of 3 balls each = 6 balls.
5 + 6 = 11 balls.
The answer is 11."
```

---

## 8. KV Cache

### Efficient Generation

```python
class GPTWithKVCache(nn.Module):
    def forward(self, input_ids, past_key_values=None):
        """
        past_key_values: K, V cache from previous tokens
        Compute only for new token and update cache
        """
        if past_key_values is None:
            # Compute entire sequence
            ...
        else:
            # Compute only last token
            ...

        return logits, new_past_key_values

# During generation
past = None
for _ in range(max_new_tokens):
    logits, past = model(new_token, past_key_values=past)
    # O(1) complexity instead of O(n)
```

### HuggingFace KV Cache

```python
output = model.generate(
    input_ids,
    max_new_tokens=50,
    use_cache=True  # Enable KV Cache (default)
)
```

---

## Summary

### Generation Strategy Comparison

| Method | Advantages | Disadvantages | Use Cases |
|--------|-----------|---------------|-----------|
| Greedy | Fast, consistent | Repetitive, boring | Translation, QA |
| Temperature | Control diversity | Requires tuning | General generation |
| Top-k | Stable | Fixed k | General generation |
| Top-p | Adaptive | Slightly slower | Creative, dialogue |

### Key Code

```python
# HuggingFace GPT-2
from transformers import GPT2Tokenizer, GPT2LMHeadModel
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Generation
output = model.generate(input_ids, max_length=50, do_sample=True,
                        temperature=0.7, top_p=0.9)
```

---

## Exercises

### Exercise 1: Generation Strategy Comparison

Using HuggingFace's GPT-2, generate text from the same prompt with four different strategies: greedy decoding, temperature sampling (T=0.5), top-k sampling (k=50), and top-p sampling (p=0.9). Compare the outputs and explain when you would choose each strategy in a real application.

<details>
<summary>Show Answer</summary>

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

prompt = "The future of artificial intelligence is"
input_ids = tokenizer.encode(prompt, return_tensors='pt')

def decode(output):
    return tokenizer.decode(output[0], skip_special_tokens=True)

# 1. Greedy decoding - always picks the most likely token
greedy = model.generate(input_ids, max_new_tokens=30, do_sample=False)
print("GREEDY:", decode(greedy))
# Deterministic, often repetitive

# 2. Temperature sampling (T=0.5) - sharper distribution, less random
low_temp = model.generate(
    input_ids, max_new_tokens=30, do_sample=True, temperature=0.5
)
print("\nTEMP=0.5:", decode(low_temp))
# More focused, less diverse but still varied

# 3. Top-k sampling (k=50)
top_k = model.generate(
    input_ids, max_new_tokens=30, do_sample=True, top_k=50
)
print("\nTOP-K=50:", decode(top_k))
# Excludes very unlikely tokens, stable quality

# 4. Top-p (nucleus) sampling (p=0.9)
top_p = model.generate(
    input_ids, max_new_tokens=30, do_sample=True, top_p=0.9, temperature=1.0
)
print("\nTOP-P=0.9:", decode(top_p))
# Adaptive vocabulary size based on cumulative probability
```

**When to use each strategy**:

| Strategy | Best for | Why |
|----------|----------|-----|
| Greedy | Translation, factual QA | Maximizes likelihood, consistent and reproducible |
| Temperature (low) | Code generation, formal text | Controlled creativity, near-deterministic |
| Temperature (high) | Brainstorming, poetry | High diversity, may sacrifice coherence |
| Top-k | Dialogue, chatbots | Prevents rare artifacts while allowing variety |
| Top-p | Creative writing, storytelling | Adapts vocabulary size to context complexity |

In practice, **top-p combined with temperature** (e.g., `p=0.9, temperature=0.8`) is the most commonly used strategy for general-purpose generation as it combines both forms of control.

</details>

### Exercise 2: KV Cache Memory Savings

Explain the computational benefit of KV Cache (Key-Value Cache) during autoregressive generation. Calculate how many times the key and value matrices are recomputed (without cache) vs computed (with cache) when generating 100 new tokens given a 50-token prompt, assuming 12 attention layers.

<details>
<summary>Show Answer</summary>

**Without KV Cache**:

At each generation step `t`, the model computes K and V for the entire sequence seen so far (prompt + generated tokens). So at step `t`, it processes `50 + t` tokens through all 12 layers.

```python
# Without KV Cache: total KV computations
prompt_len = 50
new_tokens = 100
num_layers = 12

# For each new token, recompute K and V for all previous tokens
total_kv_without_cache = 0
for t in range(new_tokens):
    seq_len = prompt_len + t + 1  # Current sequence length
    total_kv_without_cache += seq_len * num_layers

print(f"Total KV computations without cache: {total_kv_without_cache}")
# = sum(51 to 150) * 12 = 10050 * 12 = 120,600

# With KV Cache: compute K and V only for the NEW token
total_kv_with_cache = new_tokens * num_layers
print(f"Total KV computations with cache: {total_kv_with_cache}")
# = 100 * 12 = 1,200

speedup = total_kv_without_cache / total_kv_with_cache
print(f"Speedup: {speedup:.1f}x")
# ≈ 100.5x speedup in KV computation
```

**How KV Cache works**:

```python
# Conceptual KV Cache mechanism
class AttentionWithCache:
    def forward(self, x, past_kv=None):
        # Compute Q, K, V for current token only
        q = self.W_q(x)  # Only for new token: (batch, 1, d_k)
        k = self.W_k(x)  # Only for new token: (batch, 1, d_k)
        v = self.W_v(x)  # Only for new token: (batch, 1, d_k)

        if past_kv is not None:
            past_k, past_v = past_kv
            # Concatenate with cached K, V from previous steps
            k = torch.cat([past_k, k], dim=1)  # (batch, seq+1, d_k)
            v = torch.cat([past_v, v], dim=1)

        # Attend using full K, V but only new Q
        attn = softmax(q @ k.T / sqrt(d_k)) @ v  # (batch, 1, d_k)

        return attn, (k, v)  # Return updated cache
```

**Memory trade-off**: KV Cache trades computation for memory — it must store K and V for all previous tokens. For GPT-3 with 96 layers, 175B parameters, and context length 4096: each K and V matrix is `(batch, seq, 128, d_k)`, requiring ~10GB of GPU memory just for the cache. This is why LLM inference requires careful memory management.

</details>

### Exercise 3: In-Context Learning Prompt Design

Design three versions of a prompt for a text classification task (classifying movie reviews as positive/negative): zero-shot, few-shot (3 examples), and chain-of-thought. Explain why each progressively improves model performance.

<details>
<summary>Show Answer</summary>

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Version 1: Zero-shot
zero_shot_prompt = """Classify the following movie review as Positive or Negative.

Review: "The acting was superb and the story kept me engaged throughout."
Sentiment:"""

# Version 2: Few-shot (3 examples)
few_shot_prompt = """Classify the following movie review as Positive or Negative.

Review: "Absolutely terrible. I walked out after 30 minutes."
Sentiment: Negative

Review: "One of the best films I've seen this decade. Masterpiece!"
Sentiment: Positive

Review: "Mediocre plot but the cinematography saved it somewhat."
Sentiment: Negative

Review: "The acting was superb and the story kept me engaged throughout."
Sentiment:"""

# Version 3: Chain-of-Thought
cot_prompt = """Classify the following movie review as Positive or Negative.
Think step by step before giving your final answer.

Review: "Absolutely terrible. I walked out after 30 minutes."
Reasoning: The reviewer says "absolutely terrible" which is very negative, and they
left early (walked out after 30 minutes), showing they couldn't finish watching.
Sentiment: Negative

Review: "The acting was superb and the story kept me engaged throughout."
Reasoning:"""
```

**Why each approach progressively improves performance**:

**Zero-shot**: Relies entirely on patterns learned during pre-training. The model must infer the task from the format alone. Works for simple tasks where the model has seen similar formats during training.

**Few-shot**: Provides concrete input-output examples that:
- Disambiguate the task format (what "Sentiment:" should look like)
- Demonstrate the output vocabulary ("Positive", "Negative" — not "pos", "neg", or "good")
- Calibrate the model's decision boundary with real examples

GPT-3's paper showed few-shot performance often matches fine-tuned models on standard benchmarks.

**Chain-of-Thought**: Forces the model to:
- Identify relevant evidence in the text
- Reason explicitly before committing to an answer
- Reduce errors from "jumping to conclusions"

CoT is particularly valuable for nuanced reviews where sentiment isn't immediately obvious (e.g., mixed reviews, sarcasm). The intermediate reasoning steps also make the model's decisions more interpretable.

</details>

### Exercise 4: Autoregressive Training Setup

Write a complete training loop for a small character-level GPT model. The model should learn to generate sequences character by character. Show how the input and target sequences are constructed, how the causal language modeling loss is computed, and how to monitor training progress.

<details>
<summary>Show Answer</summary>

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_heads=4, num_layers=2, max_len=128):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model*4,
            batch_first=True, dropout=0.1
        )
        # Use TransformerDecoder with causal mask for autoregressive behavior
        self.blocks = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model, num_heads, d_model*4, batch_first=True)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight  # Weight tying

    def forward(self, input_ids, causal_mask=None):
        seq_len = input_ids.size(1)
        if causal_mask is None:
            # Create causal mask: True = masked (cannot attend)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1
            ).bool()

        pos = torch.arange(seq_len, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(pos)

        for block in self.blocks:
            x = block(x, x, tgt_mask=causal_mask, memory_mask=causal_mask)

        return self.head(self.ln_f(x))

# Character-level dataset preparation
text = "Hello, World! This is a training example for our tiny GPT model."
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
vocab_size = len(chars)

# Encode text
data = torch.tensor([stoi[c] for c in text])

def get_batch(data, block_size=32, batch_size=4):
    """Create input/target pairs for CLM training"""
    starts = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[s:s+block_size] for s in starts])
    # Target is input shifted by 1: predict next character
    y = torch.stack([data[s+1:s+block_size+1] for s in starts])
    return x, y

# Training loop
model = TinyGPT(vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
for step in range(200):
    x, y = get_batch(data)
    logits = model(x)  # (batch, seq, vocab_size)

    # Causal LM loss: predict each next token
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),  # (batch*seq, vocab)
        y.view(-1)                    # (batch*seq,)
    )

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step}: loss = {loss.item():.4f}, "
              f"perplexity = {torch.exp(loss).item():.2f}")

# Generation
model.eval()
with torch.no_grad():
    start = torch.tensor([[stoi['H']]])  # Start with 'H'
    for _ in range(30):
        logits = model(start)
        next_char = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        start = torch.cat([start, next_char], dim=1)
    print("Generated:", ''.join([itos[i.item()] for i in start[0]]))
```

**Key design decisions explained**:
- **Input vs target offset**: `x = data[t:t+L]`, `y = data[t+1:t+L+1]` — this means for position `i` in `x`, the model predicts `y[i] = x[i+1]`. All positions are trained simultaneously in one forward pass.
- **Gradient clipping**: `clip_grad_norm_(..., 1.0)` prevents exploding gradients, critical for transformer training.
- **Perplexity**: `exp(loss)` is a more interpretable metric — a perplexity of 2 means the model is as uncertain as a fair coin toss between 2 tokens on average.

</details>

## Next Steps

Learn about the HuggingFace Transformers library in [06_HuggingFace_Basics.md](./06_HuggingFace_Basics.md).
