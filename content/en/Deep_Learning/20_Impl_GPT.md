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

## Overview

GPT (Generative Pre-trained Transformer) is an autoregressive language model developed by OpenAI. It generates text **left-to-right** and became the foundation of modern LLMs.

---

## Mathematical Background

### Theory: Cross-Entropy and Perplexity

The per-token loss is exactly cross-entropy between the predicted distribution and the one-hot truth:

```
L_t = -log p(x_t | x_{<t}) = -log softmax(logits)[x_t]
```

This is what `nn.CrossEntropyLoss` computes. Two convenient interpretations:

- **Information-theoretic**: `L_t` is the number of bits (with `log_2`) or nats (with `log_e`) needed to encode `x_t` given the model's distribution. Lower = better compression = better model.
- **Perplexity**: `PPL = exp(mean L)`. Roughly the "effective number of equally-likely choices" the model considers per token. Vocabulary of 50k means random has `PPL = 50000`; well-trained GPT-2 reaches ~20 on Wikipedia.

Cross-entropy and perplexity are the only meaningful evaluation metrics for an LM as such; downstream task performance is a separate (and often more interesting) question.


### Theory: Autoregressive Factorization

A language model assigns probability to a sequence `x_1, ..., x_T`. By the chain rule of probability:

```
p(x_1, ..., x_T) = prod_{t=1}^{T} p(x_t | x_1, ..., x_{t-1})
```

This factorization is *exact* — no approximation. A neural language model parameterizes each conditional `p(x_t | x_{<t}) = softmax(f_\theta(x_{<t}))` where `f_\theta` is the Transformer producing logits over the vocabulary. Training maximizes log-likelihood:

```
log p(x_1, ..., x_T) = sum_t log p(x_t | x_{<t})
```

Negating gives the loss; averaging over sequences gives the per-token loss. Because of teacher forcing + causal mask, all `T` conditional probabilities are computed in one forward pass.


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

### Theory: Scaling Laws and Chinchilla

Kaplan et al. (2020) and Hoffmann et al. (2022, "Chinchilla") showed that LM loss follows a predictable power law in **compute (C), parameters (N), and training tokens (D)**:

```
L(N, D) ≈ L_inf + A / N^\alpha + B / D^\beta
```

with `\alpha, \beta ≈ 0.3-0.4`. For a fixed compute budget `C ≈ 6 N D`, the loss-minimizing allocation is **N proportional to D** — roughly equal scaling of parameters and tokens. GPT-3 (175B params, 300B tokens) was *under-trained*; Chinchilla (70B params, 1.4T tokens) outperformed it with 2.5x fewer parameters.

The takeaway is that LM quality is not "bigger is better" but "bigger AND more data, in balance, is better." Modern LLMs (LLaMA-2, GPT-4) follow Chinchilla-optimal recipes much more closely.


### Theory: Sampling Strategies

Once trained, the model produces `p(x_t | x_{<t})` — but how to choose an actual next token? Four strategies:

- **Greedy / argmax**: `x_t = argmax_v p(v)`. Deterministic, often boring or repetitive (the model gets stuck in loops).
- **Sampling from full distribution**: `x_t ~ p`. Diverse but sometimes incoherent (low-prob tokens get picked).
- **Top-k**: keep only the `k` highest-probability tokens, renormalize, sample. Bounds the worst-case bad sample.
- **Top-p (nucleus)**: keep the smallest set of tokens whose cumulative probability `>= p` (e.g., 0.9), renormalize, sample. Adapts to the distribution's shape — sharp distributions keep few tokens, flat ones keep many.
- **Temperature**: rescale logits by `1/T` before softmax. `T < 1` sharpens (closer to argmax), `T > 1` flattens (more diverse), `T = 1` is the natural distribution.

Modern LLMs typically combine top-p + temperature for generation, with `T = 0.7-0.9` being a common practical default.


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
