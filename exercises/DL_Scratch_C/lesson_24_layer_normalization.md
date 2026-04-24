# Lesson 24 — Layer Normalization (per-lesson exercise)

Prerequisites: L23 (positional encodings), basic BLAS (L03), autograd for affine ops (L06).

Compile: `gcc -std=c11 -Wall -Wextra -O2 -o ex ex.c -lm`

LayerNorm is applied on the last dimension of activations in a transformer block. For a vector $x \in \mathbb{R}^D$, gain $\gamma \in \mathbb{R}^D$, bias $\beta \in \mathbb{R}^D$:

$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y_i = \gamma_i \hat{x}_i + \beta_i$$

where $\mu = \frac{1}{D}\sum x_i$ and $\sigma^2 = \frac{1}{D}\sum (x_i - \mu)^2$.

---

## Exercise 24.1 — Forward Pass on a Batch

**Difficulty**: ★★

### Problem

Implement `layernorm_forward(const float *X, const float *gamma, const float *beta, float *Y, int B, int D, float eps)` that applies LayerNorm to each of the `B` rows of `X` (shape `[B, D]`) independently. Also fill `mean_out[B]` and `rstd_out[B]` arrays (reciprocal standard deviation) for later use in the backward pass.

Rules:
- Row-major storage: element `(b, i)` is at `X[b*D + i]`.
- Compute the per-row mean and variance in one pass through the row (Welford's algorithm is fine but a two-pass implementation is acceptable for this exercise).
- Use `rstd = 1 / sqrtf(var + eps)` and cache it.

### Starter Code

```c
#include <stdio.h>
#include <math.h>
#include <string.h>

void layernorm_forward(const float *X, const float *gamma, const float *beta,
                       float *Y, float *mean_out, float *rstd_out,
                       int B, int D, float eps) {
    /* For each row b:
         1. compute mean
         2. compute variance
         3. compute rstd = 1/sqrt(var + eps)
         4. Y[b*D + i] = (X[b*D + i] - mean) * rstd * gamma[i] + beta[i]
         5. store mean_out[b] and rstd_out[b]
    */
    /* TODO */
    (void)X; (void)gamma; (void)beta; (void)Y; (void)mean_out; (void)rstd_out;
    (void)B; (void)D; (void)eps;
}

int main(void) {
    /* Tiny example: B=2, D=4 */
    float X[]     = { 1,  2,  3,  4,
                      5,  5,  5,  5};
    float gamma[] = { 1,  1,  1,  1};
    float beta[]  = { 0,  0,  0,  0};
    float Y[8]       = {0};
    float mean[2]    = {0};
    float rstd[2]    = {0};

    layernorm_forward(X, gamma, beta, Y, mean, rstd, 2, 4, 1e-5f);

    /* Row 1 should be centered with small std-close-to-1 (standard-normal-ish).
       Row 2 is constant; after rstd explosion is prevented by eps, the output
       is all zeros plus beta. */
    printf("Row 0: ");
    for (int i = 0; i < 4; i++) printf("%8.4f ", Y[i]);
    printf("\nRow 1: ");
    for (int i = 4; i < 8; i++) printf("%8.4f ", Y[i]);
    printf("\nmean = [%.2f %.2f]  rstd = [%.2f %.2f]\n", mean[0], mean[1], rstd[0], rstd[1]);
    return 0;
}
```

### Expected

Row 0 approx `[-1.3416 -0.4472  0.4472  1.3416]` (classic standard-normalized 1,2,3,4).
Row 1 approx `[0.0000 0.0000 0.0000 0.0000]` (constant input → every $x_i = \mu$).
`mean = [2.50 5.00]`, `rstd = [approx 0.89 approx 316.23]` (the second is `1/sqrt(eps)`).

---

## Exercise 24.2 — Numerical Stability Check

**Difficulty**: ★

Explain (in code or comments) why `layernorm_forward` should NOT compute `rstd = 1/sqrt(var)` without `eps`. Create a row of length `D=4` where every element is `1.0e8` and show what happens to `Y` without `eps`. Then re-run with `eps=1e-5` and compare.

Record the two outputs in your submission and state, in one sentence, what `eps` prevents.
