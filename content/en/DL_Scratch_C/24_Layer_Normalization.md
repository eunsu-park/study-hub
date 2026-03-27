# 24. Layer Normalization

**Previous**: [Positional Encodings](./23_Positional_Encodings.md) | **Next**: [Attention Mechanism](./25_Attention_Mechanism.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement LayerNorm forward pass (normalizes over the last dimension, per sample)
2. Implement RMSNorm (simpler variant used in Llama)
3. Derive and implement the LayerNorm backward pass through mean and variance
4. Explain why LayerNorm is preferred over BatchNorm for sequence models
5. Verify LayerNorm output matches PyTorch on a test input

---

## 1. LayerNorm vs BatchNorm

```
BatchNorm: normalize over (N, H, W) per channel → requires batch; fails for batch=1
LayerNorm: normalize over d_model per token → independent of batch size; works for any N

For sequences [N, T, d_model]:
  BN normalizes across N (and T if used) per feature → batch-dependent
  LN normalizes across d_model per (n, t) position   → batch-independent

Why LN for Transformers:
  - Each token position is normalized independently
  - Batch size can be 1 (inference) or vary between batches
  - The normalized direction is the "feature space" that makes semantic sense
```

---

## 2. LayerNorm Forward

```
For input x ∈ ℝ^d:
  μ = (1/d) Σ x_i
  σ² = (1/d) Σ (x_i - μ)²
  x̂ = (x - μ) / √(σ² + ε)
  y = γ ⊙ x̂ + β
```

```c
#define LN_EPS 1e-5f

// layernorm_forward: normalize last dimension independently per (n,t)
// input:  [N, T, C]  (C = d_model)
// gamma:  [C]
// beta:   [C]
// output: [C] per position
// Saves mean, rstd, x_hat for backward
void layernorm_forward(
    const float *X,      // [N*T, C]
    const float *gamma,  // [C]
    const float *beta,   // [C]
    float       *Y,      // [N*T, C]
    float       *mean,   // [N*T] — saved for backward
    float       *rstd,   // [N*T] — 1/std, saved for backward
    int M, int C) {       // M = N*T

    for (int m = 0; m < M; m++) {
        const float *x = X + (long)m * C;
        float       *y = Y + (long)m * C;

        // Compute mean
        float mu = 0.0f;
        for (int i = 0; i < C; i++) mu += x[i];
        mu /= C;

        // Compute variance
        float var = 0.0f;
        for (int i = 0; i < C; i++) {
            float d = x[i] - mu;
            var += d * d;
        }
        var /= C;

        float rs = 1.0f / sqrtf(var + LN_EPS);
        mean[m] = mu;
        rstd[m] = rs;

        for (int i = 0; i < C; i++)
            y[i] = gamma[i] * (x[i] - mu) * rs + beta[i];
    }
}
```

---

## 3. LayerNorm Backward

The backward through LN must account for the fact that μ and σ depend on all x:

```
Let x_hat_i = (x_i - μ) / σ,  y_i = γ_i * x_hat_i + β_i

∂L/∂γ_i = Σ_m dY[m,i] * x_hat[m,i]    (sum over positions)
∂L/∂β_i = Σ_m dY[m,i]

∂L/∂x[m,i] = (γ_i / σ_m) * [ dY[m,i]
              - (1/C) Σ_j dY[m,j]
              - (x_hat[m,i] / C) Σ_j dY[m,j] * x_hat[m,j] ]
```

```c
// layernorm_backward: compute dX, dgamma, dbeta
void layernorm_backward(
    const float *dY,     // [M, C]
    const float *X,      // [M, C] — original input
    const float *gamma,  // [C]
    const float *mean,   // [M] — saved from forward
    const float *rstd,   // [M] — 1/std, saved from forward
    float       *dX,     // [M, C]
    float       *dgamma, // [C] — accumulated
    float       *dbeta,  // [C] — accumulated
    int M, int C) {

    for (int m = 0; m < M; m++) {
        const float *dy   = dY    + (long)m * C;
        const float *x    = X     + (long)m * C;
        float       *dx   = dX    + (long)m * C;
        float        mu   = mean[m];
        float        rs   = rstd[m];  // = 1/σ

        // Compute x_hat and accumulate dgamma, dbeta
        float sum1 = 0.0f, sum2 = 0.0f;
        for (int i = 0; i < C; i++) {
            float xhat_i = (x[i] - mu) * rs;
            dgamma[i] += dy[i] * xhat_i;
            dbeta[i]  += dy[i];
            // dx_hat_i = dy_i * gamma_i
            float dx_hat_i = dy[i] * gamma[i];
            sum1 += dx_hat_i;
            sum2 += dx_hat_i * xhat_i;
        }

        // dX = rs/C * [C*dx_hat - sum1 - x_hat*sum2]
        float inv_C = 1.0f / C;
        for (int i = 0; i < C; i++) {
            float xhat_i   = (x[i] - mu) * rs;
            float dx_hat_i = dy[i] * gamma[i];
            dx[i] = rs * (dx_hat_i - inv_C * sum1 - inv_C * xhat_i * sum2);
        }
    }
}
```

---

## 4. RMSNorm (Llama / Mistral)

RMSNorm omits the mean subtraction — simpler and slightly faster:

```
RMS(x) = √((1/d) Σ x_i²)
x̂ = x / RMS(x)
y = γ ⊙ x̂           (no β offset)
```

```c
// rmsnorm_forward: normalize by RMS, no mean subtraction
void rmsnorm_forward(
    const float *X,      // [M, C]
    const float *gamma,  // [C]
    float       *Y,      // [M, C]
    float       *rrms,   // [M] — 1/RMS, saved for backward
    int M, int C) {

    for (int m = 0; m < M; m++) {
        const float *x = X + (long)m * C;
        float       *y = Y + (long)m * C;

        float ss = 0.0f;
        for (int i = 0; i < C; i++) ss += x[i] * x[i];
        float rms = 1.0f / sqrtf(ss / C + LN_EPS);
        rrms[m] = rms;

        for (int i = 0; i < C; i++)
            y[i] = gamma[i] * x[i] * rms;
    }
}

// rmsnorm_backward
void rmsnorm_backward(
    const float *dY,    // [M, C]
    const float *X,     // [M, C]
    const float *gamma, // [C]
    const float *rrms,  // [M]
    float       *dX,    // [M, C]
    float       *dgamma,// [C]
    int M, int C) {

    for (int m = 0; m < M; m++) {
        const float *dy  = dY   + (long)m * C;
        const float *x   = X    + (long)m * C;
        float       *dx  = dX   + (long)m * C;
        float        rms = rrms[m];

        // dgamma
        for (int i = 0; i < C; i++)
            dgamma[i] += dy[i] * x[i] * rms;

        // dx = rms * (dy*gamma - x * (1/C) * Σ(dy*gamma*x) * rms²)
        float dot = 0.0f;
        for (int i = 0; i < C; i++)
            dot += dy[i] * gamma[i] * x[i];
        dot *= rms * rms / C;

        for (int i = 0; i < C; i++)
            dx[i] = rms * (dy[i] * gamma[i] - x[i] * dot);
    }
}
```

---

## 5. Pre-norm vs Post-norm

```
Original Transformer (post-norm):
  y = LN(x + sublayer(x))

Modern Transformer (pre-norm, GPT-2, Llama):
  y = x + sublayer(LN(x))

Why pre-norm is preferred:
  - Gradient flows directly through the residual path (no LN in the gradient highway)
  - More stable training — allows higher learning rates
  - GPT-2, Llama, PaLM, Falcon all use pre-norm

Code pattern:
  // Pre-norm attention block
  float *normed = layernorm(x, gamma, beta);        // LN first
  float *attn   = attention(normed);                // then attention
  x = x + attn;                                     // then residual add
```

---

## 6. Numerical Verification

```c
static void test_layernorm(void) {
    int M = 2, C = 4;
    float X[] = {1,2,3,4, 2,3,4,5};
    float gamma[] = {1,1,1,1};
    float beta[]  = {0,0,0,0};
    float Y[8], mean[2], rstd[2];

    layernorm_forward(X, gamma, beta, Y, mean, rstd, M, C);

    printf("LayerNorm output (identity gamma/beta):\n");
    for (int m = 0; m < M; m++) {
        printf("  row %d: ", m);
        for (int i = 0; i < C; i++) printf("%.4f ", Y[m*C+i]);
        printf("\n");
    }
    // Expected (row 0: mean=2.5, std=1.118):
    //   [−1.3416, −0.4472, 0.4472, 1.3416]
    // Expected (row 1: mean=3.5, std=1.118):
    //   [−1.3416, −0.4472, 0.4472, 1.3416]

    // Verify mean≈0, std≈1 for each row
    for (int m = 0; m < M; m++) {
        float s = 0, s2 = 0;
        for (int i = 0; i < C; i++) { s += Y[m*C+i]; s2 += Y[m*C+i]*Y[m*C+i]; }
        printf("  row %d: mean=%.6f  var=%.6f\n", m, s/C, s2/C - (s/C)*(s/C));
    }
}
```

---

## Key Takeaways

- **LayerNorm** normalizes over `d_model` per token independently — no batch dependency, works for any batch size
- **RMSNorm** omits mean subtraction: `y = γ × x / RMS(x)` — slightly simpler, used in Llama/Mistral
- LN backward follows the same pattern as BN backward: gradient through mean and variance requires two correction terms
- **Pre-norm** (normalize before sublayer, then add residual) is more stable than post-norm for deep Transformers — used in all modern LLMs
- RMSNorm backward: `dx = rms × (dy×γ − x × dot(dy×γ, x) × rms² / C)`

---

**Next**: [25. Attention Mechanism](./25_Attention_Mechanism.md) — Multi-head self-attention: Q/K/V projections, scaled dot-product, causal masking, and output projection.
