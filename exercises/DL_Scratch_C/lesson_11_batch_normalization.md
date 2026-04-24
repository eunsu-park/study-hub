# Lesson 11 — Batch Normalization (per-lesson exercise)

Prerequisites: L05 (autograd basics), familiarity with mini-batch tensor layout.

Compile: `gcc -std=c11 -Wall -Wextra -O2 -o ex ex.c -lm`

Batch norm normalizes each feature across a mini-batch. For input `X` of shape `[N, C]`, with learnable scale `γ ∈ ℝ^C` and bias `β ∈ ℝ^C`:

$$\mu_c = \frac{1}{N}\sum_n X_{n,c}, \qquad \sigma_c^2 = \frac{1}{N}\sum_n (X_{n,c} - \mu_c)^2$$

$$Y_{n,c} = \gamma_c \cdot \frac{X_{n,c} - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}} + \beta_c$$

The trick (vs. LayerNorm in L24) is that statistics are taken over the batch dimension, not the feature dimension.

---

## Exercise 11.1 — Forward Pass

**Difficulty**: ★★

### Problem

Implement `bn_forward(const float *X, const float *gamma, const float *beta, float *Y, float *mean, float *var, int N, int C, float eps)`.

Cache `mean[c]` and `var[c]` so the backward pass can reuse them.

### Starter

```c
#include <stdio.h>
#include <math.h>

void bn_forward(const float *X, const float *gamma, const float *beta,
                float *Y, float *mean, float *var,
                int N, int C, float eps) {
    /* For each channel c:
         1. mean[c]   = (1/N) sum_n X[n*C + c]
         2. var[c]    = (1/N) sum_n (X[n*C + c] - mean[c])^2
         3. rstd      = 1 / sqrtf(var[c] + eps)
         4. Y[n*C + c] = gamma[c] * (X[n*C + c] - mean[c]) * rstd + beta[c]
    */
    /* TODO */
    (void)X; (void)gamma; (void)beta; (void)Y; (void)mean; (void)var;
    (void)N; (void)C; (void)eps;
}

int main(void) {
    float X[]      = { 1, 2, 3, 4,
                       5, 5, 5, 5,
                      -1, 0, 1, 2 };          /* N=3, C=4 */
    float gamma[]  = {1, 1, 1, 1};
    float beta[]   = {0, 0, 0, 0};
    float Y[12], mean[4], var[4];

    bn_forward(X, gamma, beta, Y, mean, var, 3, 4, 1e-5f);

    for (int n = 0; n < 3; n++) {
        for (int c = 0; c < 4; c++) printf("%7.4f ", Y[n*4 + c]);
        printf("\n");
    }
    /* Each column should have mean ≈ 0 and std ≈ 1. */
    return 0;
}
```

---

## Exercise 11.2 — Inference Mode (Running Statistics)

**Difficulty**: ★★

At inference time, batch norm uses **running** mean and variance, not the per-batch statistics. Implement an exponential moving average:

```c
/* During training */
running_mean[c] = momentum * running_mean[c] + (1 - momentum) * mean[c];
running_var[c]  = momentum * running_var[c]  + (1 - momentum) * var[c];

/* During inference: use running stats instead of fresh ones */
```

`momentum = 0.9` is the typical value. Demonstrate that running with `N = 1` (single inference sample) using running stats produces a sensible output, while using per-batch stats would normalize to zero (variance with one sample is zero).

---

## Exercise 11.3 — Backward Pass — Bonus

**Difficulty**: ★★★★

The backward pass for BN is famously fiddly because the gradient flows through both the per-channel mean AND variance, which themselves depend on every element. Derive (or look up) the formula:

$$\frac{\partial L}{\partial X_{n,c}} = \frac{\gamma_c \cdot \text{rstd}_c}{N}\left[N \frac{\partial L}{\partial Y_{n,c}} - \sum_m \frac{\partial L}{\partial Y_{m,c}} - \hat{X}_{n,c}\sum_m \frac{\partial L}{\partial Y_{m,c}} \hat{X}_{m,c}\right]$$

Implement and verify against finite differences on a tiny case ($N=4, C=2$).
