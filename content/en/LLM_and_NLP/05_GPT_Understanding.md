# 05. GPT Understanding

## Learning Objectives

- Understanding GPT architecture
- Autoregressive language modeling
- Text generation techniques
- Evolution of GPT series

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

```
Input tokens
    ↓
Token Embedding + Position Embedding
    ↓
┌─────────────────────────────────┐
│  Masked Multi-Head Attention    │
│           ↓                     │
│  Add & LayerNorm                │
│           ↓                     │
│  Feed Forward                   │
│           ↓                     │
│  Add & LayerNorm                │
└─────────────────────────────────┘
            × N layers
    ↓
LayerNorm
    ↓
Linear (vocab_size)
    ↓
Softmax → Next token probability
```

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
