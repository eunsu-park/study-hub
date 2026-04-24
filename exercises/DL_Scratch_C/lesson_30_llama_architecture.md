# Lesson 30 — Llama Architecture Differences (per-lesson exercise)

Prerequisites: L29 (GPT-2 forward), L24 (LayerNorm), L27 (FFN activations).

Compile: `gcc -std=c11 -Wall -Wextra -O2 -o ex ex.c -lm`

Llama (and nearly every post-2023 open LLM) differs from GPT-2 in four concrete ways. This exercise implements each difference so you can see them in isolation.

1. **RMSNorm** replaces LayerNorm.
2. **Rotary Position Embedding (RoPE)** replaces additive positional embeddings.
3. **SwiGLU** replaces GELU in the FFN.
4. **Grouped-Query Attention (GQA)** replaces full multi-head attention in most sizes.

---

## Exercise 30.1 — RMSNorm

**Difficulty**: ★

### Problem

Implement `void rmsnorm(const float *x, const float *gamma, float *y, int D, float eps)`:

$$y_i = \gamma_i \cdot \frac{x_i}{\sqrt{\frac{1}{D}\sum_j x_j^2 + \epsilon}}$$

Note the differences vs LayerNorm:

- No mean subtraction
- No bias term ($\beta$)
- The normalizer is RMS, not standard deviation

```c
void rmsnorm(const float *x, const float *gamma, float *y, int D, float eps) {
    /* 1. ms = sum(x_i^2) / D */
    /* 2. rstd = 1 / sqrtf(ms + eps) */
    /* 3. y[i] = gamma[i] * x[i] * rstd */
    /* TODO */
    (void)x; (void)gamma; (void)y; (void)D; (void)eps;
}
```

Why RMSNorm? It is ~25% faster than LayerNorm (no mean pass) with empirically equivalent quality for large transformers. The LLM community switched almost uniformly.

---

## Exercise 30.2 — RoPE for a Single Query/Key

**Difficulty**: ★★★

RoPE rotates pairs of query and key dimensions by a position-dependent angle:

$$\text{For pair } (2i, 2i+1): \quad \theta_i = 10000^{-2i/D}, \quad \text{rotate by } m \cdot \theta_i$$

where $m$ is the token position. Implement:

```c
void rope_rotate_inplace(float *x, int D, int position) {
    /* For k in 0, 1, ..., D/2 - 1:
         angle = position * powf(10000.0f, -2.0f * k / D)
         c = cosf(angle);  s = sinf(angle)
         x0 = x[2k]; x1 = x[2k + 1]
         x[2k]     = x0 * c - x1 * s
         x[2k + 1] = x0 * s + x1 * c
    */
    /* TODO */
    (void)x; (void)D; (void)position;
}
```

Verify: rotating a vector at position 0 should be a no-op (all angles are 0). Rotating then un-rotating (`rope_rotate_inplace(..., -position)`) should return the original to within `1e-5` per element.

---

## Exercise 30.3 — SwiGLU FFN

**Difficulty**: ★★

SwiGLU replaces `FFN(x) = W2 · GELU(W1 · x)` with:

$$\text{FFN}(x) = W_2 \cdot (\text{SiLU}(W_1 x) \odot W_3 x)$$

where SiLU is $x \cdot \sigma(x)$. Implement `void swiglu_ffn(const float *x, ...)` with three weight matrices. The hidden dimension is usually $\frac{2}{3} \cdot 4d$ to keep parameter count matched to the GELU variant.

---

## Exercise 30.4 — GQA — Bonus

**Difficulty**: ★★★★

Grouped-Query Attention uses fewer K/V heads than Q heads. If there are $H$ query heads and $G$ key/value groups with $H/G$ queries sharing each K/V, the KV cache shrinks by $H/G$. Typical: Llama-70B uses $H = 64$, $G = 8$, for an 8× KV-cache reduction.

Extend your attention routine to accept `n_query_heads` and `n_kv_heads` separately, broadcasting K and V across the query groups. Verify numerical equality when $G = H$ (pure MHA).
