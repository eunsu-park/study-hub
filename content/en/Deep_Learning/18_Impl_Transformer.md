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

## Theory & Principles

Implementing a Transformer from scratch makes the data flow concrete: tensor shapes, mask construction, the encoder/decoder split, and the autoregressive training trick. The math is identical to the previous lesson; this section emphasizes the implementation-level details that determine whether your implementation actually works.

This section covers:

- **A.** Encoder vs decoder vs encoder-decoder
- **B.** Causal masks and the upper-triangular mechanic
- **C.** Teacher forcing and the parallel-training shortcut
- **D.** Token / position embeddings and shape bookkeeping

### A. Three Transformer Variants

The original "Transformer" actually came in three architectural patterns, each suited to a class of tasks:

- **Encoder-only** (BERT-like): bidirectional self-attention over the entire input. Used for classification, NER, anything where the whole input is available at once. No mask.
- **Decoder-only** (GPT-like): unidirectional (causal) self-attention. Each token only attends to itself and earlier tokens. Used for autoregressive generation.
- **Encoder-decoder** (original Vaswani 2017, T5): encoder processes the source sequence (no mask), decoder generates the target while attending both to its own past tokens (causal mask) and to the encoder's output (cross-attention, no mask). Used for translation, summarization.

Modern LLMs converged on decoder-only — it is simpler, scales well, and a sufficiently large decoder can do most encoder tasks via prompting.

### B. Causal Masks

For a decoder, position `t` must not attend to positions `> t` (otherwise generation cheats by seeing the future). Implemented by adding a mask to the attention scores before softmax:

```
mask[i, j] = -inf  if j > i  else  0
attn = softmax((Q K^T / sqrt(d_k)) + mask)
```

Adding `-inf` makes those positions exactly zero after softmax. The mask is the same upper-triangular pattern for every example in the batch and every head, which is why it is constructed once via `torch.triu(...)` and broadcast.

For padded sequences, an additional mask masks out padding tokens regardless of position. Combined: the effective mask is `causal_mask | padding_mask`.

### C. Teacher Forcing

Naive autoregressive training would generate token `t` from the model's own output at step `t-1`, but this serializes training (each step depends on the previous). **Teacher forcing** replaces the model's prediction with the *ground-truth* token at training time:

```
input  = [<bos>, y_1, y_2, ..., y_{T-1}]
target = [y_1,  y_2, y_3, ..., y_T]
```

The model predicts `y_t` from `[<bos>, y_1, ..., y_{t-1}]` for every `t` *in parallel*. This works because of the causal mask — position `t` cannot see positions `> t`, so feeding the ground truth as input does not leak information beyond what would be available at inference. Training cost drops from `O(T)` sequential steps to `O(1)` parallel forward passes.

The catch is **exposure bias**: at inference, the model conditions on its own (possibly wrong) outputs, but at training it conditioned on perfect ground truth. This mismatch is largely tolerated in practice but motivates techniques like scheduled sampling and reinforcement-learning-from-feedback for high-stakes generation.

### D. Embeddings and Shape Bookkeeping

A Transformer input passes through:

```
x: token IDs                  shape (B, T)
emb = TokenEmb(x)             shape (B, T, d_model)
emb = emb + PosEnc(positions) shape (B, T, d_model)         # broadcast over batch
out = TransformerStack(emb)   shape (B, T, d_model)
logits = LMHead(out)          shape (B, T, vocab_size)
```

Token embedding is `nn.Embedding(vocab, d_model)`, just a lookup table. Positional encoding is added (not concatenated) — this works because the network can in principle learn to "ignore" positions in some dimensions and only use the token info there. A common parameter-saving trick is to **tie the LM head's weight matrix to the token embedding** (`LMHead.weight = TokenEmb.weight`), which halves the parameters in the input/output and is justified by symmetry: the embedding maps tokens to vectors, the LM head maps vectors back to tokens.

### From Theory to the Code Below

| Theory concept | Code construct in this lesson |
|----------------|-------------------------------|
| Encoder vs decoder | `nn.TransformerEncoderLayer` vs `nn.TransformerDecoderLayer` |
| Causal mask | `torch.triu(torch.ones(T, T), diagonal=1).bool()` |
| Teacher forcing | Feeding shifted target as decoder input |
| Embedding tying | `model.lm_head.weight = model.token_emb.weight` |

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
