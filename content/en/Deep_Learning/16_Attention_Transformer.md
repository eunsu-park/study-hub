# 16. Attention and Transformer

[Previous: LSTM & GRU Implementation](./15_Impl_LSTM_GRU.md) | [Next: Attention Deep Dive](./17_Attention_Deep_Dive.md)

---

## Learning Objectives

- Understand the principles of Attention mechanism
- Learn Self-Attention
- Understand Transformer architecture
- Implement with PyTorch

---

## 1. Need for Attention

### Seq2Seq Limitations

```
Encoder: "I go to school" → Fixed-size vector
                              ↓
Decoder: Fixed vector → "나는 학교에 간다"

Problem: Information loss when long sentences compressed
```

### Attention Solution

```
When decoder generates each output word,
it can "attend" to all encoder words

Generating "I" → High attention on "나는"
Generating "school" → High attention on "학교"
```

---

## 2. Attention Mechanism

### Formula

```python
# Query, Key, Value
Q = Current decoder state
K = All encoder states
V = All encoder states (usually same as K)

# Attention Score
score = Q @ K.T  # (query_len, key_len)

# Attention Weight (softmax)
weight = softmax(score / sqrt(d_k))  # Scaling

# Context
context = weight @ V  # Weighted sum
```

### Scaled Dot-Product Attention

```python
def attention(Q, K, V, mask=None):
    d_k = K.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    weights = F.softmax(scores, dim=-1)
    return weights @ V, weights
```

---

## 3. Self-Attention

### Concept

```
Each word attends to all other words in the same sequence

"The cat sat on the mat because it was tired"
"it" has high attention on "cat" → Pronoun resolution
```

### Formula

```python
# Generate Q, K, V from input X
Q = X @ W_Q
K = X @ W_K
V = X @ W_V

# Self-Attention
output = attention(Q, K, V)
```

---

## 4. Multi-Head Attention

### Idea

```
Multiple attention heads learn different relationships

Head 1: Grammatical relationships
Head 2: Semantic relationships
Head 3: Positional relationships
...
```

### Formula

```python
def multi_head_attention(Q, K, V, num_heads):
    d_model = Q.size(-1)
    d_k = d_model // num_heads

    # Split heads
    Q = Q.view(batch, seq, num_heads, d_k).transpose(1, 2)
    K = K.view(batch, seq, num_heads, d_k).transpose(1, 2)
    V = V.view(batch, seq, num_heads, d_k).transpose(1, 2)

    # Attention for each head
    attn_output, _ = attention(Q, K, V)

    # Combine heads
    output = attn_output.transpose(1, 2).contiguous().view(batch, seq, d_model)
    return output
```

---

## 5. Transformer Architecture

### Structure

```
Input → Embedding → Positional Encoding
                      ↓
┌─────────────────────────────────────┐
│  Multi-Head Self-Attention          │
│           ↓                         │
│  Add & LayerNorm                    │
│           ↓                         │
│  Feed Forward Network               │
│           ↓                         │
│  Add & LayerNorm                    │
└─────────────────────────────────────┘
            × N layers
                ↓
             Output
```

### Key Components

1. **Multi-Head Attention**
2. **Position-wise Feed Forward**
3. **Residual Connection**
4. **Layer Normalization**
5. **Positional Encoding**

---

## 6. Positional Encoding

### Necessity

```
Attention has no order information
→ Explicitly add position information
```

### Sinusoidal Encoding

```python
def positional_encoding(seq_len, d_model):
    PE = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000) / d_model))

    PE[:, 0::2] = torch.sin(position * div_term)
    PE[:, 1::2] = torch.cos(position * div_term)
    return PE
```

---

## 7. PyTorch Transformer

### Basic Usage

```python
import torch.nn as nn

# Transformer encoder
encoder_layer = nn.TransformerEncoderLayer(
    d_model=512,
    nhead=8,
    dim_feedforward=2048,
    dropout=0.1
)
encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

# Forward pass
x = torch.randn(10, 32, 512)  # (seq, batch, d_model)
output = encoder(x)
```

### Classification Model

```python
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (batch, seq)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (seq, batch, d_model)
        x = self.transformer(x)
        x = x.mean(dim=0)  # Mean pooling
        return self.fc(x)
```

---

## 8. Vision Transformer (ViT)

### Idea

```
Split image into patches → Process as sequence

Image (224×224) → 16×16 patches (196 patches) → Transformer
```

### Structure

```python
class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, num_classes, d_model, nhead, num_layers):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # Extract and embed patches
        patches = extract_patches(x)
        x = self.patch_embed(patches)

        # Add CLS token
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Position embedding
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x.transpose(0, 1))

        # Classification (use CLS token)
        return self.fc(x[0])
```

---

## 9. Attention vs RNN Comparison

| Item | RNN/LSTM | Transformer |
|------|----------|-------------|
| Parallelization | Difficult | Easy |
| Long-range Dependencies | Difficult | Easy |
| Training Speed | Slow | Fast |
| Memory | O(n) | O(n²) |
| Position Information | Implicit | Explicit |

---

## 10. Practical Applications

### NLP

- BERT: Bidirectional encoder
- GPT: Decoder-based generation
- T5: Encoder-decoder

### Vision

- ViT: Image classification
- DETR: Object detection
- Swin Transformer: Hierarchical structure

---

## 11. Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)

Standard Multi-Head Attention (MHA) gives each head its own Key and Value projections. While this is expressive, it becomes a memory bottleneck during autoregressive inference because we must cache K and V for every head at every past position. MQA and GQA are two important optimizations that dramatically reduce this overhead.

### 11.1 The KV Cache Problem

During autoregressive generation (e.g., GPT generating one token at a time), we cache the K and V tensors from all previous positions to avoid recomputation. This **KV cache** grows as:

```
KV cache memory = 2 × n_layers × n_kv_heads × seq_len × head_dim × bytes_per_param

For a standard MHA model (e.g., LLaMA 1 65B):
  n_layers=80, n_heads=64, head_dim=128, seq_len=2048, FP16
  = 2 × 80 × 64 × 2048 × 128 × 2 bytes ≈ 5.2 GB per sequence!

This limits batch size and maximum sequence length at inference time.
```

### 11.2 Multi-Query Attention (MQA)

**MQA** (Shazeer, 2019) uses a single shared K and V across all query heads:

```
Standard MHA:
  Q: (batch, n_heads, seq, head_dim)   ← separate per head
  K: (batch, n_heads, seq, head_dim)   ← separate per head
  V: (batch, n_heads, seq, head_dim)   ← separate per head

MQA:
  Q: (batch, n_heads, seq, head_dim)   ← separate per head
  K: (batch, 1, seq, head_dim)         ← ONE shared K
  V: (batch, 1, seq, head_dim)         ← ONE shared V

  K and V are broadcast across all query heads.

KV cache reduced by factor of n_heads (e.g., 64× less memory).
```

### 11.3 Grouped-Query Attention (GQA)

**GQA** (Ainslie et al., 2023) is a middle ground: instead of 1 shared KV (MQA) or n_heads separate KVs (MHA), we use **G groups**, where each group of query heads shares one K and one V.

```
MHA (n_kv_heads = n_heads):      Each head has its own K, V
GQA (1 < n_kv_heads < n_heads):  Groups of heads share K, V
MQA (n_kv_heads = 1):            All heads share one K, V

Example with 8 query heads, 2 KV groups:
  Q heads: [Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7]
  KV groups: [KV_0, KV_1]

  Q0, Q1, Q2, Q3 → attend using KV_0
  Q4, Q5, Q6, Q7 → attend using KV_1

LLaMA 2 70B uses GQA with 64 query heads and 8 KV heads.
Mistral 7B uses GQA with 32 query heads and 8 KV heads.
```

### 11.4 Memory and Latency Comparison

```
┌──────────────────┬───────────┬───────────┬───────────┐
│ Metric           │ MHA       │ GQA       │ MQA       │
├──────────────────┼───────────┼───────────┼───────────┤
│ KV heads         │ n_heads   │ n_groups  │ 1         │
│ KV cache size    │ 1×        │ G/H ×     │ 1/H ×     │
│ KV params        │ 1×        │ G/H ×     │ 1/H ×     │
│ Quality          │ Best      │ Near MHA  │ Slight ↓  │
│ Inference speed  │ Slowest   │ Fast      │ Fastest   │
│ Decoding latency │ Highest   │ Low       │ Lowest    │
├──────────────────┼───────────┼───────────┼───────────┤
│ Example          │ GPT-3     │ LLaMA 2   │ PaLM      │
│ (H=heads, G=grp) │ H=96,G=96│ H=64,G=8  │ H=16,G=1  │
│ KV cache savings │ 0%        │ 87.5%     │ 93.75%    │
└──────────────────┴───────────┴───────────┴───────────┘

H = total query heads, G = number of KV groups
KV cache savings = (1 - G/H) × 100%
```

### 11.5 PyTorch Implementation of GQA

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA).

    Why GQA over MHA: At inference time, the KV cache is the primary
    memory bottleneck. By sharing K/V across groups of query heads,
    GQA reduces cache size by (n_heads / n_kv_heads)× while retaining
    most of MHA's quality. This is the approach used by LLaMA 2 70B,
    Mistral 7B, and many modern LLMs.
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads  # How many Q heads per KV group
        self.head_dim = d_model // n_heads

        # Q has n_heads projections; K, V have only n_kv_heads projections
        self.W_q = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.W_o = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: optional attention mask

        Returns:
            output: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # Project Q, K, V
        Q = self.W_q(x).view(batch, seq_len, self.n_heads, self.head_dim)
        K = self.W_k(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        V = self.W_v(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)

        # Transpose to (batch, heads, seq, head_dim)
        Q = Q.transpose(1, 2)  # (batch, n_heads, seq, head_dim)
        K = K.transpose(1, 2)  # (batch, n_kv_heads, seq, head_dim)
        V = V.transpose(1, 2)  # (batch, n_kv_heads, seq, head_dim)

        # Expand K, V to match Q's head count by repeating within groups
        # Each KV head is shared by (n_heads // n_kv_heads) query heads
        K = K.repeat_interleave(self.n_groups, dim=1)  # (batch, n_heads, seq, head_dim)
        V = V.repeat_interleave(self.n_groups, dim=1)  # (batch, n_heads, seq, head_dim)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        # Recombine heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)

        return self.W_o(attn_output)


# Example: LLaMA 2 70B style GQA
gqa = GroupedQueryAttention(d_model=512, n_heads=8, n_kv_heads=2)
x = torch.randn(2, 10, 512)  # batch=2, seq=10
output = gqa(x)
print(f"Output shape: {output.shape}")  # (2, 10, 512)

# Compare parameter counts
mha_kv_params = 2 * 512 * 512  # MHA: W_k + W_v, both (512, 512)
gqa_kv_params = 2 * 512 * 128  # GQA: W_k + W_v, both (512, 128) for 2 KV heads
print(f"MHA KV params: {mha_kv_params:,}")    # 524,288
print(f"GQA KV params: {gqa_kv_params:,}")    # 131,072
print(f"KV param reduction: {(1 - gqa_kv_params/mha_kv_params)*100:.0f}%")  # 75%
```

### 11.6 GQA in Modern LLMs

```
┌─────────────────────┬──────────┬───────────┬───────────────────────┐
│ Model               │ Q Heads  │ KV Heads  │ Attention Type        │
├─────────────────────┼──────────┼───────────┼───────────────────────┤
│ GPT-3 175B          │ 96       │ 96        │ MHA                   │
│ PaLM 540B           │ 16       │ 1         │ MQA                   │
│ LLaMA 1 65B         │ 64       │ 64        │ MHA                   │
│ LLaMA 2 70B         │ 64       │ 8         │ GQA (8 groups)        │
│ Mistral 7B          │ 32       │ 8         │ GQA (4 groups)        │
│ Gemma 7B            │ 16       │ 16        │ MHA                   │
│ Falcon 40B          │ 64       │ 1         │ MQA                   │
│ Qwen2 72B           │ 64       │ 8         │ GQA                   │
└─────────────────────┴──────────┴───────────┴───────────────────────┘

Trend: GQA has become the default for models > 7B parameters
because it provides the best quality-efficiency trade-off.

Key insight: MQA can degrade quality noticeably for large models.
GQA with 8 KV heads retains nearly all of MHA's quality while
providing most of MQA's speed benefit.
```

---

## Summary

### Core Concepts

1. **Attention**: Calculate relevance with Query-Key-Value
2. **Self-Attention**: Reference all positions within sequence
3. **Multi-Head**: Learn various relationships simultaneously
4. **Positional Encoding**: Add order information

### Key Code

```python
# Scaled Dot-Product Attention
scores = Q @ K.T / sqrt(d_k)
weights = softmax(scores)
output = weights @ V

# PyTorch Transformer
encoder = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(d_model=512, nhead=8),
    num_layers=6
)
```

---

## Exercises

### Exercise 1: Scaled Dot-Product Attention by Hand

Implement scaled dot-product attention in pure NumPy (no PyTorch).

1. Create random Q, K, V matrices of shape `(4, 8)` (seq_len=4, d_k=8).
2. Compute the attention scores `Q @ K.T`, apply the `1/sqrt(d_k)` scaling factor, then softmax.
3. Multiply the resulting weights by V to get the context matrix.
4. Verify your result matches `F.scaled_dot_product_attention` in PyTorch (with the same values).
5. Explain why we divide by `sqrt(d_k)`: what goes wrong if we omit the scaling?

### Exercise 2: Self-Attention Visualization

Visualize self-attention weights on a short sentence to see which words attend to which.

1. Build the `MultiHeadAttention` module from the lesson (or use `nn.MultiheadAttention`).
2. Tokenize a short sentence (e.g., "The cat sat on the mat") at the word level and embed each word with `nn.Embedding`.
3. Run a forward pass with `return_attention=True` and extract the attention weights.
4. Plot the attention weight matrix as a heatmap with word labels on both axes.
5. Identify at least one interesting attention pattern (e.g., which word "sat" attends to most strongly).

### Exercise 3: Build a Transformer Encoder for Classification

Use PyTorch's `nn.TransformerEncoder` to classify sequences.

1. Create synthetic data: sequences of 20 integers, labeled by whether their mean is above or below 50.
2. Build `TransformerClassifier` with `d_model=64`, `nhead=4`, `num_layers=2`.
3. Add a mean-pooling step after the encoder to aggregate the sequence into a single vector.
4. Train for 30 epochs and report accuracy.
5. Compare against a simple LSTM classifier with similar parameter count — which converges faster?

### Exercise 4: GQA vs MHA Parameter and Memory Analysis

Quantify the memory savings of Grouped-Query Attention.

1. Instantiate `GroupedQueryAttention` with configurations: `(d_model=512, n_heads=8, n_kv_heads=8)` (MHA), `(d_model=512, n_heads=8, n_kv_heads=4)` (GQA-4), `(d_model=512, n_heads=8, n_kv_heads=1)` (MQA).
2. Count the KV projection parameters for each configuration.
3. Compute the theoretical KV cache size for `seq_len=2048` in MB (float16), assuming 1 layer.
4. Run a forward pass with a batch of shape `(2, 32, 512)` and confirm all outputs have the same shape.
5. Summarize the trade-off: at what point does the quality loss from MQA outweigh its memory savings?

### Exercise 5: Causal (Autoregressive) Attention Mask

Implement a causal mask and observe how it prevents attending to future tokens.

1. Create a causal mask of shape `(seq_len, seq_len)` where position `i` can only attend to positions `0..i` (upper triangle is `-inf`).
2. Apply this mask inside `scaled_dot_product_attention`.
3. Verify the mask works by checking that the softmax weights for future positions are exactly 0.
4. Build a small causal Transformer (2 layers, `d_model=64`) and train it to predict the next token in a sine wave discretized into 50 bins.
5. Report the next-token prediction accuracy on a held-out 10% of the data.

---

## Next Steps

In [23_Training_Optimization.md](./23_Training_Optimization.md), we'll learn advanced training techniques.
