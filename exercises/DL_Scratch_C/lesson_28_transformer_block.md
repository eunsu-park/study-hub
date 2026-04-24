# Lesson 28 — Transformer Block (per-lesson exercise)

Prerequisites: L24 (LayerNorm), L25 (attention), L26 (KV cache), L27 (FFN).

Compile: `gcc -std=c11 -Wall -Wextra -O2 -o ex ex.c -lm`

A transformer block is the unit that gets stacked $N$ times in any modern decoder. Pre-norm form (used by every modern LLM):

```
x → LN1 → MHA → +residual → LN2 → FFN → +residual → out
```

The structure is small but every detail matters: residual connections, the order of normalize/attention/residual, and which weights have which shape.

---

## Exercise 28.1 — Pre-Norm Block Forward Pass

**Difficulty**: ★★★

### Problem

Implement `transformer_block(const float *x, ..., float *y, int d_model, int n_heads)` that performs:

```
h1 = layernorm(x, ln1_g, ln1_b)
attn_out = multihead_attention(h1, Wq, Wk, Wv, Wo, n_heads)
x_mid = x + attn_out               # residual

h2 = layernorm(x_mid, ln2_g, ln2_b)
ffn_out = ffn(h2, W1, b1, W2, b2)
y = x_mid + ffn_out                # residual
```

Reuse your routines from L24 (layernorm), L25 (attention), L27 (ffn). The new code is just the wiring + residuals.

### Starter

```c
#include <stdio.h>
#include <string.h>

void transformer_block(const float *x,
                       /* attention weights */
                       const float *Wq, const float *Wk, const float *Wv, const float *Wo,
                       /* FFN weights */
                       const float *W1, const float *b1, const float *W2, const float *b2,
                       /* layernorm parameters */
                       const float *ln1_g, const float *ln1_b,
                       const float *ln2_g, const float *ln2_b,
                       float *y,
                       int d_model, int d_ff, int n_heads) {
    float *h1      = malloc(d_model * sizeof(float));
    float *attn    = malloc(d_model * sizeof(float));
    float *x_mid   = malloc(d_model * sizeof(float));
    float *h2      = malloc(d_model * sizeof(float));
    float *ffn_out = malloc(d_model * sizeof(float));

    /* TODO: fill in using your previous lesson functions */
    layernorm_forward(x, ln1_g, ln1_b, h1, /* ... */);
    multihead_attention(h1, Wq, Wk, Wv, Wo, attn, d_model, n_heads);
    for (int i = 0; i < d_model; i++) x_mid[i] = x[i] + attn[i];

    layernorm_forward(x_mid, ln2_g, ln2_b, h2, /* ... */);
    ffn_forward(h2, W1, b1, W2, b2, ffn_out, d_model, d_ff, d_model);
    for (int i = 0; i < d_model; i++) y[i] = x_mid[i] + ffn_out[i];

    free(h1); free(attn); free(x_mid); free(h2); free(ffn_out);
}
```

---

## Exercise 28.2 — Pre-Norm vs. Post-Norm

**Difficulty**: ★★

Implement both block variants:

- **Post-norm** (original 2017): `y = LN(x + sublayer(x))` — applied after the residual sum.
- **Pre-norm** (modern): `y = x + sublayer(LN(x))` — applied to the input of the sublayer.

Pre-norm is used by every model since 2020 because:

1. It allows much deeper stacks without learning-rate warmup tricks.
2. The residual stream's L2 norm grows monotonically with depth (predictable signal scale).
3. Gradient flow through the residual path is "clean" — derivatives do not pass through layernorm.

Reproduce both variants and stack 24 of them on a random input. Compare the L2 norms of the per-layer outputs. You should see post-norm growing erratically while pre-norm grows monotonically.

---

## Exercise 28.3 — Sequence-Level Forward — Bonus

**Difficulty**: ★★★

Generalize the block from a single token (shape `[d_model]`) to a sequence (shape `[T, d_model]`). The attention sublayer must receive all tokens at once and use causal masking; the FFN runs independently on each position.

Verify against the per-token version: feeding tokens one-by-one through the per-token block (with KV cache) should produce identical output (up to floating-point error) to the sequence version with no cache.
