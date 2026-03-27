# 34. Cross-Entropy Loss for Language Model Training

**Previous**: [Multimodal CLIP-Style Learning](./33_Multimodal_CLIP_Style.md) | **Next**: [Optimizers](./35_Optimizers.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Derive log-softmax + NLL as the numerically stable form of cross-entropy
2. Implement logsumexp with the max-subtraction trick
3. Implement a fused softmax-CE backward pass that avoids materializing the full softmax matrix
4. Apply cross-entropy to next-token prediction (shifted label convention)
5. Compute perplexity as the exponentiated mean cross-entropy loss

---

## 1. Cross-Entropy: Derivation

For a classification task with V classes, the raw model output is a logits vector `z ∈ ℝᵛ`. The predicted probability for class `c` is:

```
p_c = exp(z_c) / Σ_k exp(z_k)   (softmax)
```

The cross-entropy loss for a true class `y` is:

```
L = -log(p_y)
  = -log(exp(z_y) / Σ_k exp(z_k))
  = -z_y + log(Σ_k exp(z_k))
  = -z_y + logsumexp(z)
```

This is exactly `-z_y + logsumexp(z)`. No need to compute softmax explicitly — we just need the raw logit of the correct class and the logsumexp over all classes.

### Why Naive Softmax Overflows

```
exp(1000.0f) = +inf   (IEEE 754 overflow)
exp(-1000.0f) = 0.0f  (underflow)
```

Language model logits can be large in magnitude, especially early in training.

---

## 2. Logsumexp: Numerically Stable

```c
#include <math.h>
#include <float.h>

/*
 * logsumexp — compute log(Σ exp(x[i])) stably.
 *
 * Subtract max before exponentiation:
 *   log(Σ exp(x_i)) = max_x + log(Σ exp(x_i - max_x))
 *
 * x  : input array of length n
 * n  : number of elements
 */
float logsumexp(const float *x, int n) {
    /* Find maximum */
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    /* Stable sum */
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += expf(x[i] - mx);
    return mx + logf(sum);
}
```

Proof of stability: after subtracting `max_x`, all exponents are ≤ 0, so `exp(x_i - max_x) ∈ (0, 1]`. No overflow, no underflow (unless all logits are `-∞`, which is degenerate).

---

## 3. Log-Softmax

```c
/*
 * log_softmax — compute log(softmax(x)) for each element.
 *
 * Numerically equivalent to:
 *   log_softmax(x)[i] = x[i] - logsumexp(x)
 *
 * x   : input logits  [n]
 * out : output        [n]
 */
void log_softmax(const float *x, float *out, int n) {
    float lse = logsumexp(x, n);
    for (int i = 0; i < n; i++) out[i] = x[i] - lse;
}

/*
 * nll_loss — negative log-likelihood for a single example.
 *
 * log_probs : output of log_softmax [V]
 * target    : correct class index
 */
float nll_loss(const float *log_probs, int target) {
    return -log_probs[target];
}
```

---

## 4. Batch Cross-Entropy (Naïve — For Reference)

```c
/*
 * cross_entropy_forward_naive — computes mean CE loss over a batch.
 *
 * logits   : [N, V]  raw model output
 * targets  : [N]     correct class indices
 * losses   : [N]     per-sample CE loss (output)
 * N        : batch size
 * V        : vocabulary size
 *
 * Materializes log_probs [N, V] — O(N*V) memory.
 */
float cross_entropy_forward_naive(const float *logits, const int *targets,
                                  float *losses, int N, int V)
{
    float total = 0.0f;
    for (int i = 0; i < N; i++) {
        const float *row = logits + i * V;
        float lse = logsumexp(row, V);
        losses[i] = -(row[targets[i]] - lse);
        total += losses[i];
    }
    return total / (float)N;
}
```

The problem: for a language model with V=50257 (GPT-2 vocabulary) and N=B×T=16×1024=16384 tokens per batch, the logit matrix is `16384 × 50257 × 4 bytes ≈ 3.3 GB`. This is the most memory-intensive operation in LLM training.

---

## 5. Fused Softmax-CE Forward + Backward

The key insight: **the backward pass of cross-entropy w.r.t. logits is just `softmax(z) - one_hot(y)`**, scaled by `1/N`. We never need to store log_probs — we compute the loss in the forward pass and compute the gradient directly.

### 5.1 Derivation of the Fused Backward

```
L = (1/N) Σ_i [ -z_{i,y_i} + logsumexp(z_i) ]

∂L/∂z_{i,j} = (1/N) * [ -1_{j=y_i} + exp(z_{i,j}) / Σ_k exp(z_{i,k}) ]
             = (1/N) * [ softmax(z_i)[j] - 1_{j=y_i} ]
```

So `dlogits = (softmax(z) - one_hot(y)) / N`.

No explicit log_probs matrix needed in the backward pass — only the softmax probabilities (which we compute row-by-row) and the target indices.

### 5.2 Fused Implementation

```c
#include <string.h>
#include <stdlib.h>

/*
 * fused_ce_forward_backward — compute CE loss and gradient in one pass.
 *
 * logits   : [N, V]  input logits (read-only)
 * targets  : [N]     correct class indices
 * dlogits  : [N, V]  output gradient (written here)
 * N        : number of tokens in batch
 * V        : vocabulary size
 *
 * Returns: scalar mean cross-entropy loss.
 *
 * Memory: only allocates one row [V] of floats at a time — O(V) not O(N*V).
 */
float fused_ce_forward_backward(const float *logits, const int *targets,
                                float *dlogits, int N, int V)
{
    float *probs = (float *)malloc((size_t)V * sizeof(float));
    float total_loss = 0.0f;
    float scale = 1.0f / (float)N;

    for (int i = 0; i < N; i++) {
        const float *row   = logits  + i * V;
        float       *drow  = dlogits + i * V;

        /* Stable softmax: find max */
        float mx = row[0];
        for (int j = 1; j < V; j++) if (row[j] > mx) mx = row[j];

        /* Compute unnormalized exp */
        float sum = 0.0f;
        for (int j = 0; j < V; j++) {
            probs[j] = expf(row[j] - mx);
            sum += probs[j];
        }
        /* Normalize to get softmax probabilities */
        float inv_sum = 1.0f / sum;
        for (int j = 0; j < V; j++) probs[j] *= inv_sum;

        /* Cross-entropy loss for this token */
        total_loss += -logf(probs[targets[i]] + 1e-10f);

        /* Gradient: dlogits = (softmax - one_hot) / N */
        for (int j = 0; j < V; j++) {
            drow[j] = probs[j] * scale;
        }
        drow[targets[i]] -= scale;   /* subtract 1/N at the correct class */
    }

    free(probs);
    return total_loss / (float)N;
}
```

This implementation is O(N·V) time and O(V) memory (per row), which is optimal. Compare with the naïve version that required O(N·V) memory.

### 5.3 Why Fusing Avoids Materializing the Softmax Matrix

In the naïve approach:
1. Forward: compute log_probs [N, V] → save to memory → read targets → compute loss
2. Backward: re-read log_probs → compute grad

In the fused approach:
1. Compute softmax row-by-row (one row of size V at a time)
2. Immediately compute loss contribution and gradient for that row
3. Write gradient to dlogits → discard the softmax row

Peak memory usage: O(V) instead of O(N·V). For V=50257 and N=16384 this is a 16384× reduction in memory for this operation.

---

## 6. Cross-Entropy for Next-Token Prediction

Language models are trained with next-token prediction: given tokens `[t_0, t_1, ..., t_{T-1}]`, predict `[t_1, t_2, ..., t_T]`.

### 6.1 Label Shifting

```c
/*
 * shift_labels — extract (inputs, targets) for next-token prediction.
 *
 * tokens  : [B, T+1]  raw token sequence (including the final target)
 * inputs  : [B, T]    tokens 0..T-1 (model input)
 * targets : [B, T]    tokens 1..T   (prediction targets)
 * B       : batch size
 * T       : sequence length (the model sees T tokens, predicts T tokens)
 */
void shift_labels(const int *tokens, int *inputs, int *targets, int B, int T) {
    for (int b = 0; b < B; b++) {
        const int *seq = tokens + b * (T + 1);
        int *inp = inputs  + b * T;
        int *tgt = targets + b * T;
        for (int t = 0; t < T; t++) {
            inp[t] = seq[t];
            tgt[t] = seq[t + 1];
        }
    }
}

/*
 * Training cross-entropy for an LM:
 *
 * logits  : [B, T, V]   model output
 * targets : [B, T]      next-token labels (shifted by 1)
 *
 * Flatten to [B*T, V] and [B*T] → call fused_ce_forward_backward.
 */
float lm_cross_entropy(const float *logits, const int *targets,
                       float *dlogits, int B, int T, int V)
{
    int N = B * T;
    return fused_ce_forward_backward(logits, targets, dlogits, N, V);
}
```

### 6.2 Ignoring Padding Tokens

When batching sequences of different lengths with padding, ignore padding positions in the loss:

```c
/*
 * lm_cross_entropy_masked — ignore positions where target == pad_id.
 *
 * Returns: mean loss over non-padding positions.
 */
float lm_cross_entropy_masked(const float *logits, const int *targets,
                               float *dlogits, int B, int T, int V,
                               int pad_id)
{
    float *probs  = (float *)malloc((size_t)V * sizeof(float));
    float total_loss = 0.0f;
    int   count      = 0;

    /* First pass: count non-padding tokens for normalization */
    int N = B * T;
    for (int i = 0; i < N; i++) if (targets[i] != pad_id) count++;

    float scale = (count > 0) ? 1.0f / (float)count : 0.0f;

    for (int i = 0; i < N; i++) {
        const float *row  = logits  + i * V;
        float       *drow = dlogits + i * V;

        if (targets[i] == pad_id) {
            /* Zero gradient at padding positions */
            memset(drow, 0, (size_t)V * sizeof(float));
            continue;
        }

        /* Stable softmax */
        float mx = row[0];
        for (int j = 1; j < V; j++) if (row[j] > mx) mx = row[j];
        float sum = 0.0f;
        for (int j = 0; j < V; j++) { probs[j] = expf(row[j] - mx); sum += probs[j]; }
        float inv = 1.0f / sum;
        for (int j = 0; j < V; j++) probs[j] *= inv;

        total_loss += -logf(probs[targets[i]] + 1e-10f);
        for (int j = 0; j < V; j++) drow[j] = probs[j] * scale;
        drow[targets[i]] -= scale;
    }

    free(probs);
    return (count > 0) ? total_loss / (float)count : 0.0f;
}
```

---

## 7. Perplexity

Perplexity is the standard evaluation metric for language models:

```
Perplexity = exp(mean_CE_loss)
```

For a model with mean CE loss of 3.0, perplexity = exp(3.0) ≈ 20.1. This means the model is (on average) as confused as if it had to choose uniformly among 20 options.

GPT-2 achieves ~18.3 perplexity on WikiText-103. A character-level model on Shakespeare typically reaches ~1.4-1.6 bits/char ≈ perplexity 2.6-3.0.

```c
#include <math.h>

/*
 * compute_perplexity — given mean CE loss, return perplexity.
 * loss: mean cross-entropy (nats, base e)
 */
float compute_perplexity(float loss) {
    return expf(loss);
}

/*
 * eval_perplexity — run model on eval set, return perplexity.
 *
 * eval_tokens   : [num_eval_tokens]  tokenized validation set
 * model_forward : function pointer for the model forward pass
 * B, T, V       : batch, seq len, vocab size
 */
float eval_perplexity(const int *eval_tokens, int num_eval_tokens,
                      /* model forward fn: tokens[B,T] → logits[B,T,V] */
                      void (*model_forward)(const int *tokens, float *logits,
                                           int B, int T),
                      int B, int T, int V)
{
    float *logits  = (float *)malloc((size_t)B * T * V * sizeof(float));
    float *dlogits = (float *)malloc((size_t)B * T * V * sizeof(float));
    int   *inputs  = (int   *)malloc((size_t)B * T     * sizeof(int));
    int   *targets = (int   *)malloc((size_t)B * T     * sizeof(int));

    float total_loss = 0.0f;
    int   num_batches = (num_eval_tokens / (B * (T + 1)));

    for (int b = 0; b < num_batches; b++) {
        const int *batch_tokens = eval_tokens + b * B * (T + 1);
        shift_labels(batch_tokens, inputs, targets, B, T);
        model_forward(inputs, logits, B, T);
        float loss = lm_cross_entropy(logits, targets, dlogits, B, T, V);
        total_loss += loss;
    }

    free(logits); free(dlogits); free(inputs); free(targets);

    float mean_loss = (num_batches > 0) ? total_loss / (float)num_batches : 0.0f;
    return compute_perplexity(mean_loss);
}
```

---

## Key Takeaways

- **Cross-entropy = -z_y + logsumexp(z)**: no need to compute softmax explicitly for the forward pass. All you need is the correct-class logit and the logsumexp.
- **Logsumexp stability**: always subtract the row maximum before exponentiation. This converts potential overflow/underflow into a harmless `log(1) = 0` term.
- **Fused backward**: `dlogits[i][j] = (softmax[i][j] - one_hot[i][y_i]) / N`. This avoids storing a full [N, V] log-probability matrix — critical when V=50257 and N=16384.
- **Label shifting**: for next-token prediction, inputs are `tokens[0..T-1]` and targets are `tokens[1..T]`. Off-by-one errors here are a common source of training bugs.
- **Perplexity** = `exp(mean_CE_loss)`. It is the standard LM evaluation metric. A drop from 3.0 to 2.85 nats corresponds to perplexity dropping from 20.1 to 17.3 — a meaningful improvement.
- **Padding masking**: when mixing sequences of different lengths, exclude padding positions from both the loss and the gradient to avoid training on meaningless tokens.

---

**Previous**: [Multimodal CLIP-Style Learning](./33_Multimodal_CLIP_Style.md) | **Next**: [Optimizers](./35_Optimizers.md)

> Next lesson implements SGD with momentum, Adam, AdamW, gradient clipping, and LR scheduling using function pointers.
