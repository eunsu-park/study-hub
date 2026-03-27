# 03. Transformer Review

## Learning Objectives

- Understanding Transformer from NLP perspective
- Encoder and Decoder structures
- Attention in language modeling context
- Understanding BERT/GPT-based architectures

---

## 1. Transformer Overview

### Architecture Summary

```
Encoder (BERT-style):
    Input → [Embedding + Positional] → [Self-Attention + FFN] × N → Output

Decoder (GPT-style):
    Input → [Embedding + Positional] → [Masked Self-Attention + FFN] × N → Output

Encoder-Decoder (T5-style):
    Input → Encoder → [Cross-Attention] → Decoder → Output
```

### Role in NLP

| Model | Architecture | Use Cases |
|-------|-------------|-----------|
| BERT | Encoder only | Classification, QA, NER |
| GPT | Decoder only | Text generation |
| T5, BART | Encoder-Decoder | Translation, summarization |

---

## 2. Self-Attention (NLP Perspective)

### Learning Intra-sentence Relationships

```
"The cat sat on the mat because it was tired"

"it" → Attention → "cat" (high weight)
                → "mat" (low weight)

Model learns that pronoun "it" refers to "cat"
```

### Query, Key, Value

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        Q = self.W_q(x)  # (batch, seq, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # Split into multi-heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # (batch, num_heads, seq, d_k)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)

        # Combine heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)

        return output, attention_weights
```

---

## 3. Causal Masking (GPT-style)

### Autoregressive Language Model

```
Training "I love NLP":
    Input: [I]         → Predict: love
    Input: [I, love]   → Predict: NLP
    Input: [I, love, NLP] → Predict: <eos>

Cannot see future tokens → Need Causal Mask
```

### Causal Mask Implementation

```python
def create_causal_mask(seq_len):
    """Create lower triangular mask (block future tokens)"""
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask  # 1 = can attend, 0 = masked

# Example (seq_len=4)
# [[1, 0, 0, 0],
#  [1, 1, 0, 0],
#  [1, 1, 1, 0],
#  [1, 1, 1, 1]]

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_len=512):
        super().__init__()
        self.attention = SelfAttention(d_model, num_heads)
        # Register pre-computed mask
        mask = torch.tril(torch.ones(max_len, max_len))
        self.register_buffer('mask', mask)

    def forward(self, x):
        seq_len = x.size(1)
        mask = self.mask[:seq_len, :seq_len]
        return self.attention(x, mask)
```

---

## 4. Encoder vs Decoder

### Encoder (Bidirectional)

```python
class TransformerEncoderBlock(nn.Module):
    """BERT-style encoder block"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = SelfAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        # Self-Attention (bidirectional)
        attn_out, _ = self.self_attn(x, padding_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed Forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x
```

### Decoder (Unidirectional)

```python
class TransformerDecoderBlock(nn.Module):
    """GPT-style decoder block"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = CausalSelfAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Masked Self-Attention (unidirectional)
        attn_out, _ = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed Forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x
```

---

## 5. Positional Encoding

### Sinusoidal (Original Transformer)

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
```

### Learnable (BERT, GPT)

```python
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return x + self.pos_embedding(positions)
```

---

## 6. Complete Transformer Model

### GPT-style Language Model

```python
class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12,
                 num_layers=12, d_ff=3072, max_len=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Token + positional embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)

        # Decoder blocks
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying (optional)
        self.head.weight = self.token_embedding.weight

    def forward(self, x):
        # x: (batch, seq_len)
        batch_size, seq_len = x.shape

        # Embeddings
        tok_emb = self.token_embedding(x)
        pos = torch.arange(seq_len, device=x.device)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)  # (batch, seq, vocab_size)

        return logits

    def generate(self, idx, max_new_tokens, temperature=1.0):
        """Autoregressive text generation"""
        for _ in range(max_new_tokens):
            # Logits at last position
            logits = self(idx)[:, -1, :]  # (batch, vocab)
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx
```

### BERT-style Encoder

```python
class BERTModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12,
                 num_layers=12, d_ff=3072, max_len=512, dropout=0.1):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.segment_embedding = nn.Embedding(2, d_model)  # Sentence separation

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, input_ids, segment_ids=None, attention_mask=None):
        batch_size, seq_len = input_ids.shape

        # Combine embeddings
        tok_emb = self.token_embedding(input_ids)
        pos = torch.arange(seq_len, device=input_ids.device)
        pos_emb = self.position_embedding(pos)

        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids)
        seg_emb = self.segment_embedding(segment_ids)

        x = tok_emb + pos_emb + seg_emb

        # Transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)

        return self.ln_f(x)
```

---

## 7. Comparison by Training Objective

### Masked Language Modeling (BERT)

```
Input: "The [MASK] sat on the mat"
Predict: [MASK] → "cat"

Mask 15% of tokens and predict
Utilize bidirectional context
```

### Causal Language Modeling (GPT)

```
Input: "The cat sat on"
Predict: "the" "cat" "sat" "on" "the" "mat"

Predict next token
Unidirectional (left→right)
```

### Seq2Seq (T5, BART)

```
Input: "translate English to French: Hello"
Output: "Bonjour"

Encoder: Understand input
Decoder: Generate output
```

---

## 8. PyTorch Built-in Transformer

```python
import torch.nn as nn

# Encoder
encoder_layer = nn.TransformerEncoderLayer(
    d_model=512,
    nhead=8,
    dim_feedforward=2048,
    dropout=0.1,
    batch_first=True
)
encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

# Decoder
decoder_layer = nn.TransformerDecoderLayer(
    d_model=512,
    nhead=8,
    dim_feedforward=2048,
    dropout=0.1,
    batch_first=True
)
decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

# Usage
x = torch.randn(32, 100, 512)  # (batch, seq, d_model)
encoded = encoder(x)
decoded = decoder(x, encoded)
```

---

## Summary

### Model Comparison

| Item | BERT (Encoder) | GPT (Decoder) | T5 (Enc-Dec) |
|------|----------------|---------------|--------------|
| Attention | Bidirectional | Unidirectional (Causal) | Bidirectional + Unidirectional |
| Training | MLM + NSP | Next token prediction | Denoising |
| Output | Context vector | Generation | Generation |
| Use Cases | Classification, QA | Generation, dialogue | Translation, summarization |

### Key Code

```python
# Causal Mask
mask = torch.tril(torch.ones(seq_len, seq_len))
scores = scores.masked_fill(mask == 0, -1e9)

# Multi-Head Attention split
Q = Q.view(batch, seq, num_heads, d_k).transpose(1, 2)

# Scaled Dot-Product
scores = Q @ K.T / sqrt(d_k)
attn = softmax(scores) @ V
```

---

## Exercises

### Exercise 1: Causal Mask Behavior

For a sequence of length 5, write out the full causal mask matrix (by hand or code). Then explain: given the 3rd token (index 2) in the sequence, which tokens can it attend to and why? What would happen if this mask were absent during autoregressive generation?

<details>
<summary>Show Answer</summary>

```python
import torch

def create_causal_mask(seq_len):
    """Lower triangular matrix: 1 = attend, 0 = blocked"""
    return torch.tril(torch.ones(seq_len, seq_len))

mask = create_causal_mask(5)
print(mask)
# tensor([[1., 0., 0., 0., 0.],
#         [1., 1., 0., 0., 0.],
#         [1., 1., 1., 0., 0.],
#         [1., 1., 1., 1., 0.],
#         [1., 1., 1., 1., 1.]])
```

**Reading the mask for token at index 2 (3rd token)**:
- Row 2 is `[1, 1, 1, 0, 0]`
- It can attend to tokens at positions 0, 1, and 2 (itself and all previous tokens)
- It **cannot** attend to tokens at positions 3 and 4 (future tokens)

**Effect in the attention computation**:
```python
# Positions with mask=0 get -1e9 score before softmax
scores[mask == 0] = -1e9
# After softmax, these positions have attention weight ≈ 0
# So the output vector is a weighted sum of only past+current tokens
```

**Without the causal mask**:
- During training: the model could "cheat" by looking at future tokens to predict the next word, making the task trivially easy but the model useless for actual generation.
- During inference: future tokens don't exist yet (they're being generated one by one), so the model would produce incoherent output or require all tokens to be known upfront.

The causal mask enforces the **autoregressive property**: the prediction of position `t` depends only on positions `0` through `t-1`.

</details>

### Exercise 2: Encoder vs Decoder Architecture Differences

Fill in the following comparison table, and then write one sentence explaining why BERT cannot be used directly for open-ended text generation while GPT cannot be used directly for tasks requiring full bidirectional understanding.

| Feature | BERT (Encoder) | GPT (Decoder) |
|---------|----------------|---------------|
| Attention type | ? | ? |
| Training objective | ? | ? |
| Typical use cases | ? | ? |
| Can see future tokens? | ? | ? |

<details>
<summary>Show Answer</summary>

| Feature | BERT (Encoder) | GPT (Decoder) |
|---------|----------------|---------------|
| Attention type | Bidirectional self-attention | Causal (unidirectional) self-attention |
| Training objective | MLM + NSP | Next token prediction (CLM) |
| Typical use cases | Classification, NER, QA, similarity | Text generation, dialogue, completion |
| Can see future tokens? | Yes (full sequence visible) | No (only past tokens visible) |

**Why BERT cannot generate text**:
BERT is trained to fill in masked tokens given both left and right context. At inference time, generation requires predicting token `t` before token `t+1` exists, but BERT has no mechanism for autoregressive generation — it expects a complete (masked) input, not a partial one to extend.

**Why GPT cannot do bidirectional tasks well**:
GPT's causal masking means each token's representation is computed only from past tokens. For tasks like NER or QA where a token's label may depend on future context (e.g., determining if "Washington" is a person or place requires seeing subsequent words), GPT's unidirectional attention is fundamentally limited.

</details>

### Exercise 3: Positional Encoding Properties

The sinusoidal positional encoding uses frequencies based on the formula `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`. Implement a function to compute this encoding and verify two key properties: (1) different positions produce different encodings, and (2) the relative offset between positions is consistent regardless of the absolute position.

<details>
<summary>Show Answer</summary>

```python
import torch
import torch.nn.functional as F
import math

def sinusoidal_encoding(max_len, d_model):
    """Compute sinusoidal positional encodings"""
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                         (-math.log(10000.0) / d_model))

    pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
    pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions
    return pe

pe = sinusoidal_encoding(max_len=100, d_model=64)

# Property 1: Different positions produce different encodings
pos_5 = pe[5]
pos_10 = pe[10]
pos_50 = pe[50]

sim_5_10 = F.cosine_similarity(pos_5.unsqueeze(0), pos_10.unsqueeze(0)).item()
sim_5_50 = F.cosine_similarity(pos_5.unsqueeze(0), pos_50.unsqueeze(0)).item()
print(f"Similarity pos 5 vs 10: {sim_5_10:.4f}")   # < 1.0 (distinct)
print(f"Similarity pos 5 vs 50: {sim_5_50:.4f}")   # Even less similar (more distant)

# Property 2: Consistent relative offset
# The dot product pe[pos] · pe[pos+k] should be similar for any pos, given same k
offset = 5
dots = []
for pos in [0, 10, 20, 50]:
    dot = (pe[pos] * pe[pos + offset]).sum().item()
    dots.append(dot)
    print(f"Dot product pe[{pos}] · pe[{pos+offset}] = {dot:.4f}")

# All values should be approximately the same
print(f"Std of dot products: {torch.tensor(dots).std():.4f}")  # Should be small
```

**Key properties**:
1. Each position gets a unique encoding vector (verified by cosine similarities < 1.0).
2. The dot product `pe[pos] · pe[pos+k]` is approximately constant for fixed offset `k`, regardless of absolute `pos`. This allows the model to learn relative position patterns that generalize across positions.

These properties make sinusoidal encodings suitable for sequences of any length, even lengths not seen during training — a key advantage over learned positional embeddings.

</details>

### Exercise 4: Weight Tying in Language Models

In the `GPTModel` implementation, there is the line `self.head.weight = self.token_embedding.weight`. Explain what "weight tying" means in this context, why it is done, and what the practical benefits are.

<details>
<summary>Show Answer</summary>

**What weight tying means**:

The output projection layer (`self.head`) maps from the hidden dimension back to vocabulary size to produce logits for next-token prediction. The token embedding matrix maps from vocabulary indices to the hidden dimension. Weight tying sets these two matrices to be literally the same object in memory:

```python
# Without weight tying: two separate matrices
# embedding: (vocab_size, d_model)  →  maps token_id → vector
# head:      (d_model, vocab_size)  →  maps vector → logit per token

# With weight tying: they share the same data
self.head.weight = self.token_embedding.weight
# head.weight is embedding.weight.T effectively
# (PyTorch linear uses W @ x, so the weight shape is (out, in) = (vocab, d_model))
# This means: input embedding and output embedding are the same matrix
```

**Why it is done**:

There is an elegant symmetry: if a word's embedding vector is close to the hidden state `h`, then the model should assign high probability to that word as the next token. The input embedding and the output projection matrix are performing inverse operations on the same semantic space.

**Practical benefits**:

1. **Parameter reduction**: Eliminates one `vocab_size × d_model` matrix. For GPT-2 with vocab_size=50,257 and d_model=768, this saves ~38.6M parameters.

2. **Regularization**: Forcing the input and output embeddings to be the same adds an implicit constraint that helps prevent overfitting on the language modeling objective.

3. **Better embedding quality**: The shared matrix receives gradient updates from both the embedding (input) direction and the prediction (output) direction, often resulting in higher-quality word representations.

```python
# Verify weight tying in practice
import torch.nn as nn

class TiedModel(nn.Module):
    def __init__(self, vocab_size=1000, d_model=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embedding.weight  # Tie weights

# Count parameters
model = TiedModel()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")  # 64,000 (only one copy of the matrix)
# Without tying: 128,000 parameters (two separate matrices)
```

</details>

## Next Steps

Learn BERT's architecture and training methods in detail in [04_BERT_Understanding.md](./04_BERT_Understanding.md).
