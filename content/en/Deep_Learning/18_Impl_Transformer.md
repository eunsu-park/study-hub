# 18. Transformer

[Previous: Attention Deep Dive](./17_Attention_Deep_Dive.md) | [Next: BERT](./19_Impl_BERT.md)

---

## Overview

Transformer is the architecture proposed in "Attention Is All You Need" (Vaswani et al., 2017) and is the core of modern deep learning. It processes sequences using only **Self-Attention** without RNNs.

## Learning Objectives

1. **Self-Attention**: Understanding Query, Key, Value operations
2. **Multi-Head Attention**: Parallel processing of multiple attention heads
3. **Positional Encoding**: Injecting position information
4. **Encoder-Decoder**: Overall architecture structure

---

## Mathematical Background

### 1. Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

Where:
- Q (Query): what to look for
- K (Key): matching target
- V (Value): actual value to retrieve
- d_k: dimension of Key (scaling factor)

Formula breakdown:
1. QK^T: compute similarity between Query and Key → (seq_len, seq_len)
2. / √d_k: prevent large values (softmax stability)
3. softmax: convert to probability distribution
4. × V: weighted average
```

### 2. Multi-Head Attention

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O

where head_i = Attention(Q W^Q_i, K W^K_i, V W^V_i)

Features:
- Learn attention from multiple "perspectives"
- Each head captures different patterns
- Can be parallelized
```

### 3. Positional Encoding

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Purpose:
- Transformer has no order information
- Explicitly inject position information
- Sinusoidal: generated without training, can extrapolate
```

---

## File Structure

```
07_Transformer/
├── README.md
├── pytorch_lowlevel/
│   ├── attention_lowlevel.py      # Basic Attention implementation
│   ├── multihead_attention.py     # Multi-Head Attention
│   ├── positional_encoding.py     # Positional encoding
│   └── transformer_lowlevel.py    # Complete Transformer
├── paper/
│   ├── transformer_paper.py       # Paper reproduction
│   └── transformer_xl.py          # Transformer-XL variant
└── exercises/
    ├── 01_flash_attention.md      # Flash Attention implementation
    ├── 02_rotary_embeddings.md    # RoPE implementation
    └── 03_kv_cache.md             # KV Cache implementation
```

---

## Core Concepts

### 1. Self-Attention vs Cross-Attention

```
Self-Attention:
- Q, K, V all from same sequence
- Used inside Encoder, Decoder

Cross-Attention:
- Q from Decoder, K, V from Encoder
- Connects Encoder-Decoder
```

### 2. Masking

```python
# Padding mask: ignore padding tokens
padding_mask = (input_ids == pad_token_id)  # (batch, seq_len)

# Causal mask: prevent seeing future tokens
causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
# Set upper triangular matrix to -inf
```

### 3. Feed-Forward Network

```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2

Or (using GELU):
FFN(x) = GELU(xW_1)W_2

Features:
- Position-wise: applied independently to each position
- Expansion: usually 4x expansion (d_model → 4*d_model → d_model)
```

---

## Practice Problems

### Basic
1. Directly implement Scaled Dot-Product Attention
2. Visualize Positional Encoding
3. Visualize Self-Attention patterns

### Intermediate
1. Implement Multi-Head Attention
2. Complete Encoder block
3. Complete Decoder block (including causal mask)

### Advanced
1. Optimize autoregressive generation with KV Cache
2. Implement Flash Attention (memory efficient)
3. Implement Rotary Position Embedding (RoPE)

---

## References

- Vaswani et al. (2017). "Attention Is All You Need"
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
