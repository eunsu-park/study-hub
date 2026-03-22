# 20. GPT

[Previous: BERT](./19_Impl_BERT.md) | [Next: Vision Transformer](./21_Vision_Transformer.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the autoregressive (causal) language modeling objective and describe how GPT differs from BERT in architecture, pretraining, and use cases.
2. Describe the causal self-attention mechanism and explain how the causal mask enforces the autoregressive property during training.
3. Implement the GPT decoder architecture from scratch in PyTorch, including causal masked multi-head attention, layer normalization, and positional embeddings.
4. Generate text using autoregressive sampling strategies (greedy, top-k, nucleus sampling) and explain the trade-offs between generation quality and diversity.
5. Fine-tune a GPT model on a downstream generation task (e.g., text summarization, dialogue) and evaluate output quality.
6. Trace the evolution from GPT-1 to GPT-2 and GPT-3, identifying the scaling decisions (model size, data, compute) that drove capability improvements.

---

## Overview

GPT (Generative Pre-trained Transformer) is an autoregressive language model developed by OpenAI. It generates text **left-to-right** and became the foundation of modern LLMs.

---

## Mathematical Background

### 1. Causal Language Modeling

```
Objective function:
L = -Σ log P(x_t | x_<t)

Autoregressive model:
P(x_1, x_2, ..., x_n) = Π P(x_t | x_1, ..., x_{t-1})

Features:
- Cannot reference future tokens (causal mask)
- All tokens are training signals
- Natural for text generation
```

### 2. Causal Self-Attention

```
Standard Attention:
Attention(Q, K, V) = softmax(QK^T / √d) V

Causal Attention (future masking):
mask = upper_triangular(-∞)
Attention(Q, K, V) = softmax((QK^T + mask) / √d) V

Matrix visualization:
Q\K  | t1  t2  t3  t4
---------------------
t1   |  ✓   ×   ×   ×
t2   |  ✓   ✓   ×   ×
t3   |  ✓   ✓   ✓   ×
t4   |  ✓   ✓   ✓   ✓
```

### 3. GPT vs BERT

```
BERT (Bidirectional):
- Masked LM: 15% masking
- Bidirectional context
- Strong at classification/understanding tasks

GPT (Autoregressive):
- Causal LM: predict next token
- Left context only
- Strong at generation tasks
```

---

## GPT-2 Architecture

```
GPT-2 Small (117M):
- Hidden size: 768
- Layers: 12
- Attention heads: 12

GPT-2 Medium (345M):
- Hidden size: 1024
- Layers: 24
- Attention heads: 16

GPT-2 Large (774M):
- Hidden size: 1280
- Layers: 36
- Attention heads: 20

GPT-2 XL (1.5B):
- Hidden size: 1600
- Layers: 48
- Attention heads: 25

Structure:
Token Embedding + Position Embedding
  ↓
Transformer Decoder × L layers (Pre-LN)
  ↓
Layer Norm
  ↓
LM Head (shared with embedding)
```

---

## File Structure

```
09_GPT/
├── README.md
├── pytorch_lowlevel/
│   └── gpt_lowlevel.py         # Direct GPT Decoder implementation
├── paper/
│   └── gpt2_paper.py           # GPT-2 paper reproduction
└── exercises/
    ├── 01_text_generation.md   # Text generation practice
    └── 02_kv_cache.md          # KV Cache implementation
```

---

## Core Concepts

### 1. Pre-LN vs Post-LN

```
Post-LN (original Transformer):
x → Attention → Add → LayerNorm → FFN → Add → LayerNorm

Pre-LN (GPT-2):
x → LayerNorm → Attention → Add → LayerNorm → FFN → Add

Pre-LN advantages:
- Improved training stability
- Enables deeper networks
```

### 2. Weight Tying

```
Share weights between Embedding and LM Head:

E = Embedding matrix (vocab_size × hidden_size)
LM_head = E.T (or shared)

Advantages:
- Saves parameters
- Learns consistent representations
```

### 3. Generation Strategies

```
Greedy: argmax(P(x_t | x_<t))
- Deterministic, repetition problems

Sampling: x_t ~ P(x_t | x_<t)
- Diversity, potential quality degradation

Top-K: sample from top K
- Balance quality and diversity

Top-P (Nucleus): up to cumulative probability P
- Dynamic candidate size

Temperature: softmax(logits / T)
- T < 1: more deterministic
- T > 1: more diverse
```

---

## Implementation Levels

### Level 2: PyTorch Low-Level (pytorch_lowlevel/)
- Direct Causal Attention implementation
- Pre-LN structure
- Text generation function

### Level 3: Paper Implementation (paper/)
- Exact GPT-2 specifications
- WebText style training
- Various generation strategies

### Level 4: Code Analysis (separate document)
- Analyze HuggingFace GPT2
- Analyze nanoGPT code

---

## Learning Checklist

- [ ] Implement causal mask
- [ ] Understand Pre-LN structure
- [ ] Understand weight tying
- [ ] Implement various generation strategies
- [ ] KV Cache optimization
- [ ] Differences between GPT vs BERT

---

## References

- Radford et al. (2018). "Improving Language Understanding by Generative Pre-Training" (GPT-1)
- Radford et al. (2019). "Language Models are Unsupervised Multitask Learners" (GPT-2)
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [../LLM_and_NLP/03_BERT_GPT_Architecture.md](../LLM_and_NLP/03_BERT_GPT_Architecture.md)
