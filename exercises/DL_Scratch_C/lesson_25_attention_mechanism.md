# Lesson 25 — Attention Mechanism (per-lesson exercise)

Prerequisites: L24 (LayerNorm), L03 (BLAS), basic linear algebra.

Compile: `gcc -std=c11 -Wall -Wextra -O2 -o ex ex.c -lm`

Scaled dot-product attention — the centerpiece of every transformer — is two matmuls and a softmax:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

This exercise implements it from scratch in three increasingly realistic forms.

---

## Exercise 25.1 — Single-Head Attention Forward

**Difficulty**: ★★

### Problem

Implement `attention(Q, K, V, out, T_q, T_k, d)` for a single head:

- `Q`: `[T_q, d]`
- `K`: `[T_k, d]`
- `V`: `[T_k, d]`
- `out`: `[T_q, d]`

Pseudocode:

```
1. scores = Q @ K^T            shape [T_q, T_k]
2. scores /= sqrt(d)
3. weights = softmax(scores, dim=-1)   shape [T_q, T_k]
4. out = weights @ V             shape [T_q, d]
```

### Starter

```c
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

void attention(const float *Q, const float *K, const float *V,
               float *out, int T_q, int T_k, int d) {
    float scale = 1.0f / sqrtf((float)d);
    float *scores = malloc(T_q * T_k * sizeof(float));

    /* 1. scores = Q @ K^T * scale */
    for (int i = 0; i < T_q; i++)
        for (int j = 0; j < T_k; j++) {
            float acc = 0;
            for (int k = 0; k < d; k++) acc += Q[i*d+k] * K[j*d+k];
            scores[i*T_k + j] = acc * scale;
        }

    /* 2. softmax along T_k axis (per row of scores) — numerically stable */
    for (int i = 0; i < T_q; i++) {
        float row_max = scores[i*T_k];
        for (int j = 1; j < T_k; j++)
            if (scores[i*T_k + j] > row_max) row_max = scores[i*T_k + j];

        float row_sum = 0;
        for (int j = 0; j < T_k; j++) {
            scores[i*T_k + j] = expf(scores[i*T_k + j] - row_max);
            row_sum += scores[i*T_k + j];
        }
        for (int j = 0; j < T_k; j++) scores[i*T_k + j] /= row_sum;
    }

    /* 3. out = scores @ V */
    for (int i = 0; i < T_q; i++)
        for (int k = 0; k < d; k++) {
            float acc = 0;
            for (int j = 0; j < T_k; j++) acc += scores[i*T_k + j] * V[j*d + k];
            out[i*d + k] = acc;
        }

    free(scores);
}

int main(void) {
    /* Tiny test: 2 queries, 3 keys/values, d=4 */
    float Q[8]  = {1,0,0,0, 0,1,0,0};
    float K[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
    float V[12] = {10,11,12,13, 20,21,22,23, 30,31,32,33};
    float out[8];

    attention(Q, K, V, out, 2, 3, 4);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) printf("%.4f ", out[i*4 + j]);
        printf("\n");
    }
    /* Each query best-matches the corresponding key, so output ≈ that V row,
       softened by the attention weights from the other keys. */
    return 0;
}
```

---

## Exercise 25.2 — Causal Mask

**Difficulty**: ★★

For autoregressive models, query $i$ may not attend to keys $j > i$. Add a causal mask:

```c
/* Apply mask AFTER computing scores, BEFORE softmax */
for (int i = 0; i < T_q; i++)
    for (int j = i + 1; j < T_k; j++)
        scores[i*T_k + j] = -INFINITY;
```

The `-INFINITY` ensures `expf` produces 0, and the future positions get exactly 0 attention weight. Verify by passing different upper-triangular values for $V$ and confirming they do not affect early outputs.

---

## Exercise 25.3 — Multi-Head Attention

**Difficulty**: ★★★

In multi-head attention, the model dimension $d$ is split into $H$ heads each of size $d_h = d / H$. Each head runs the standard attention independently, then the outputs are concatenated and linearly projected:

```
For head h in 0..H:
    Q_h = Q @ W_q[h]            # [T_q, d_h]
    K_h = K @ W_k[h]
    V_h = V @ W_v[h]
    head_out[h] = attention(Q_h, K_h, V_h)
out = concat(head_outs) @ W_o   # [T_q, d]
```

In practice the per-head projections are bundled into one large `W_q` matrix and split via reshape — but for this exercise, run the heads in a Python-style loop and concatenate.

Verify against your single-head code by setting $H = 1$.

---

## Exercise 25.4 — Numerical Stability Drill — Bonus

**Difficulty**: ★

Without the row-max subtraction, `expf(scores[i*T_k+j])` overflows when scores are large. Construct a test where the un-subtracted softmax produces NaN, then verify that the stable version (subtracting `row_max`) returns finite values. This is the same lesson as L34 cross-entropy and L33 (CUDA) — softmax stability is a transferable habit.
