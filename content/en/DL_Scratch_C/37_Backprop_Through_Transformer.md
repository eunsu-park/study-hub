# 37. Backprop Through the Transformer

**Previous**: [Training Loop](./36_Training_Loop.md) | **Next**: [Training GPT-2 Small](./38_Training_GPT2_Small.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Derive the full attention backward pass: dV, dK, dQ from upstream gradients
2. Implement layernorm backward through mean and variance (see also Lesson 24)
3. Assemble the backward pass for a complete GPT-2 stack (12 blocks)
4. Verify gradients with finite differences on a 2-layer model
5. Identify and fix common backward pass bugs: sign errors, missing /N, incorrect transpose

---

## 1. Autodiff vs. Manual Backprop

For production code, use an autodiff framework. But implementing backprop manually:
- Forces you to understand exactly what each gradient means
- Reveals subtle bugs invisible at the API level (wrong transpose, missing scale)
- Produces faster code when you fuse operations (no intermediate tensors)

This lesson builds the complete backward pass for a single Transformer block, then chains 12 blocks for GPT-2.

---

## 2. Attention Backward Pass

### 2.1 Forward Pass Recap

```
Q = X @ W_Q  [B, T, H]        (H = d_head = d_model / n_heads)
K = X @ W_K  [B, T, H]
V = X @ W_V  [B, T, H]
A = softmax(Q @ K^T / sqrt(H) + mask)  [B, T, T]
out = A @ V  [B, T, H]
```

Saved for backward: `A` (attention weights), `Q`, `K`, `V`, `X`.

### 2.2 Gradient Through A @ V

Given `dout` (gradient of loss w.r.t. `out`), we need:

```
out = A @ V
dV  = A^T @ dout          [B, T, H]
dA  = dout @ V^T          [B, T, T]
```

```c
/*
 * grad_attn_av — backward through out = A @ V.
 *
 * dout : [B, T, H]   gradient from upstream
 * A    : [B, T, T]   saved attention weights (post-softmax)
 * V    : [B, T, H]   saved value matrix
 * dA   : [B, T, T]   output gradient w.r.t. A
 * dV   : [B, T, H]   output gradient w.r.t. V
 */
void grad_attn_av(const float *dout, const float *A, const float *V,
                  float *dA, float *dV,
                  int B, int T, int H)
{
    for (int b = 0; b < B; b++) {
        /* dV = A^T @ dout */
        /* A  [T, T], dout [T, H] → dV [T, H] */
        for (int t2 = 0; t2 < T; t2++) {         /* row of A^T = col of A */
            for (int h = 0; h < H; h++) {
                float acc = 0.0f;
                for (int t1 = 0; t1 < T; t1++) {
                    acc += A[b*T*T + t1*T + t2] * dout[b*T*H + t1*H + h];
                }
                dV[b*T*H + t2*H + h] += acc;
            }
        }

        /* dA = dout @ V^T */
        /* dout [T, H], V [T, H] → dA [T, T] */
        for (int t1 = 0; t1 < T; t1++) {
            for (int t2 = 0; t2 < T; t2++) {
                float acc = 0.0f;
                for (int h = 0; h < H; h++) {
                    acc += dout[b*T*H + t1*H + h] * V[b*T*H + t2*H + h];
                }
                dA[b*T*T + t1*T + t2] += acc;
            }
        }
    }
}
```

### 2.3 Gradient Through Softmax

Given `dA` (gradient w.r.t. the softmax output), we need `dS` (gradient w.r.t. the pre-softmax scores):

```
Softmax Jacobian: dS[i][j] = A[i][j] * (dA[i][j] - sum_k(dA[i][k] * A[i][k]))
                            = A[i][j] * (dA[i][j] - dot(dA[i], A[i]))
```

This is the standard softmax backward formula: for each query position `t1`, it is a row-wise operation.

```c
/*
 * grad_softmax_rows — backward through row-wise softmax.
 *
 * dA : [B, T, T]   gradient w.r.t. softmax output
 * A  : [B, T, T]   saved softmax output
 * dS : [B, T, T]   output gradient w.r.t. pre-softmax scores
 *
 * For causal attention, masked positions (A=0) automatically have dS=0.
 */
void grad_softmax_rows(const float *dA, const float *A, float *dS,
                       int B, int T)
{
    for (int b = 0; b < B; b++) {
        for (int t1 = 0; t1 < T; t1++) {
            const float *a_row  = A  + b*T*T + t1*T;
            const float *da_row = dA + b*T*T + t1*T;
            float       *ds_row = dS + b*T*T + t1*T;

            /* dot(dA[t1], A[t1]) — sum over all t2 */
            float dot = 0.0f;
            for (int t2 = 0; t2 < T; t2++) dot += da_row[t2] * a_row[t2];

            /* dS[t1][t2] = A[t1][t2] * (dA[t1][t2] - dot) */
            for (int t2 = 0; t2 < T; t2++) {
                ds_row[t2] += a_row[t2] * (da_row[t2] - dot);
            }
        }
    }
}
```

### 2.4 Gradient Through Q @ K^T / sqrt(H)

```
S = Q @ K^T / sqrt(H)
dQ = dS @ K / sqrt(H)
dK = dS^T @ Q / sqrt(H)
```

```c
/*
 * grad_qk — backward through S = Q @ K^T / sqrt(H).
 *
 * dS : [B, T, T]   gradient w.r.t. S
 * Q  : [B, T, H]   saved Q
 * K  : [B, T, H]   saved K
 * dQ : [B, T, H]   accumulated gradient w.r.t. Q
 * dK : [B, T, H]   accumulated gradient w.r.t. K
 */
void grad_qk(const float *dS, const float *Q, const float *K,
             float *dQ, float *dK, int B, int T, int H)
{
    float scale = 1.0f / sqrtf((float)H);
    for (int b = 0; b < B; b++) {
        /* dQ = dS @ K / sqrt(H)  :  [T,T] @ [T,H] → [T,H] */
        for (int t1 = 0; t1 < T; t1++) {
            for (int h = 0; h < H; h++) {
                float acc = 0.0f;
                for (int t2 = 0; t2 < T; t2++) {
                    acc += dS[b*T*T + t1*T + t2] * K[b*T*H + t2*H + h];
                }
                dQ[b*T*H + t1*H + h] += acc * scale;
            }
        }
        /* dK = dS^T @ Q / sqrt(H)  :  [T,T]^T @ [T,H] → [T,H] */
        for (int t2 = 0; t2 < T; t2++) {
            for (int h = 0; h < H; h++) {
                float acc = 0.0f;
                for (int t1 = 0; t1 < T; t1++) {
                    acc += dS[b*T*T + t1*T + t2] * Q[b*T*H + t1*H + h];
                }
                dK[b*T*H + t2*H + h] += acc * scale;
            }
        }
    }
}
```

### 2.5 Common Bug: Missing /sqrt(H)

The scale factor `1/sqrt(H)` must appear in **both** dQ and dK. A common mistake is to apply it only in one direction, which silently produces wrong gradients and slower convergence.

---

## 3. LayerNorm Backward

(Full derivation in Lesson 24. Summary here for completeness.)

```
Forward:  y = (x - mean) / sqrt(var + eps) * gamma + beta
          where mean = mean(x), var = mean((x-mean)^2)

Backward (for one row of [T, D]):
  dvar   = sum(dhat_x * (x - mean) * -0.5 * (var+eps)^(-3/2))
  dmean  = sum(dhat_x * -1/sqrt(var+eps)) + dvar * mean(-2*(x-mean))
  dx     = dhat_x / sqrt(var+eps) + dvar * 2*(x-mean)/D + dmean/D
  dgamma = sum(dhat_x * x_hat)
  dbeta  = sum(dhat_x)
```

```c
/*
 * layernorm_backward — gradient through LayerNorm.
 * Reference implementation; see Lesson 24 for the detailed derivation.
 *
 * dout  : [N, D]  gradient from upstream
 * x     : [N, D]  saved input
 * xhat  : [N, D]  saved normalized input (x - mean)/std
 * gamma : [D]     scale parameter
 * dx    : [N, D]  output gradient w.r.t. x
 * dgamma: [D]     gradient w.r.t. gamma (accumulated)
 * dbeta : [D]     gradient w.r.t. beta  (accumulated)
 * N     : number of rows
 * D     : row dimension
 * eps   : stabilizer (must match forward eps)
 */
void layernorm_backward(const float *dout, const float *x, const float *xhat,
                        const float *gamma, float *dx, float *dgamma,
                        float *dbeta, int N, int D, float eps)
{
    for (int n = 0; n < N; n++) {
        const float *dout_row = dout  + n * D;
        const float *x_row    = x     + n * D;
        const float *xhat_row = xhat  + n * D;
        float       *dx_row   = dx    + n * D;

        /* Compute mean and var (re-use from saved xhat or recompute) */
        float mean = 0.0f, var = 0.0f;
        for (int d = 0; d < D; d++) mean += x_row[d];
        mean /= D;
        for (int d = 0; d < D; d++) { float diff = x_row[d]-mean; var += diff*diff; }
        var /= D;
        float std = sqrtf(var + eps);
        float inv_std = 1.0f / std;

        /* dhat_x = dout * gamma */
        float dvar = 0.0f, dmean = 0.0f;
        for (int d = 0; d < D; d++) {
            float dhat_x = dout_row[d] * gamma[d];
            dvar  += dhat_x * (x_row[d] - mean) * -0.5f * inv_std * inv_std * inv_std;
            dmean += dhat_x * (-inv_std);
            dgamma[d] += dout_row[d] * xhat_row[d];
            dbeta[d]  += dout_row[d];
        }
        dmean += dvar * (-2.0f / D);   /* correction from dvar's dependence on mean */

        /* dx */
        for (int d = 0; d < D; d++) {
            float dhat_x = dout_row[d] * gamma[d];
            dx_row[d] += dhat_x * inv_std
                       + dvar * 2.0f * (x_row[d] - mean) / (float)D
                       + dmean / (float)D;
        }
    }
}
```

---

## 4. Full Transformer Block Backward

A block forward:
```
x1 = x0 + Attn(LayerNorm1(x0))
x2 = x1 + FFN(LayerNorm2(x1))
```

Backward (chain rule in reverse):
```
dx1  = dFFN_residual(dx2)
dx1 += dLN2_attn(dx1_from_FFN)
dx0  = dAttn_residual(dx1)
dx0 += dLN1(dx0_from_Attn)
```

```c
/*
 * transformer_block_backward — backward pass through one Transformer block.
 *
 * Saved from forward: x0, x1, ln1_out, ln1_xhat, attn_A, attn_Q, attn_K,
 *                     attn_V, attn_out, ln2_out, ln2_xhat, ffn_hidden
 * All intermediate tensors [B, T, D] unless noted.
 * grads : accumulation target for weight gradients (W_Q, W_K, ..., W1, W2)
 */
void transformer_block_backward(
    /* upstream gradient */
    const float *dx2,
    /* saved activations */
    const float *x0, const float *x1,
    const float *ln1_xhat, const float *ln2_xhat,
    const float *attn_A, const float *Q, const float *K, const float *V,
    /* parameters */
    const float *gamma1, const float *gamma2,
    const float *W_Q, const float *W_K, const float *W_V, const float *W_O,
    const float *W1, const float *W2,
    /* output gradients (accumulated) */
    float *dx0_out,
    float *dW_Q, float *dW_K, float *dW_V, float *dW_O,
    float *dW1, float *dW2,
    float *dgamma1, float *dbeta1, float *dgamma2, float *dbeta2,
    /* scratch buffers (B*T*D each) */
    float *scratch1, float *scratch2,
    int B, int T, int D, int n_heads, float eps)
{
    int H = D / n_heads;

    /* --- FFN branch backward --- */
    /* dx2 flows through: FFN residual (x2 = x1 + FFN(LN2(x1))) */
    /* dx1_from_ffn_residual = dx2 (residual just copies gradient) */

    /* 1. Backward through FFN output linear W2 and GELU */
    /* (details omitted for brevity — standard matmul backward) */
    ffn_backward(dx2, W2, W1, scratch1 /* dffn_input */, dW2, dW1,
                 B, T, D, 4*D);

    /* 2. Backward through LayerNorm2 */
    layernorm_backward(scratch1, x1, ln2_xhat, gamma2,
                       scratch2 /* dx1_from_ffn */, dgamma2, dbeta2,
                       B*T, D, eps);

    /* dx1 = dx2 (residual) + dx1_from_ffn */
    for (int i = 0; i < B*T*D; i++) scratch2[i] += dx2[i];

    /* --- Attention branch backward --- */
    /* 3. Backward through attention output proj W_O */
    /* attn_combined_out = attn_heads_out @ W_O */
    float *dattn_out = scratch1;   /* reuse scratch */
    proj_backward(scratch2, W_O, dattn_out, dW_O, B*T, D, D);

    /* 4. Backward through A @ V, softmax, Q @ K^T */
    float *dA  = (float *)calloc((size_t)B*T*T, sizeof(float));
    float *dQ  = (float *)calloc((size_t)B*T*H*n_heads, sizeof(float));
    float *dK  = (float *)calloc((size_t)B*T*H*n_heads, sizeof(float));
    float *dV  = (float *)calloc((size_t)B*T*H*n_heads, sizeof(float));
    float *dS  = (float *)calloc((size_t)B*T*T, sizeof(float));

    grad_attn_av(dattn_out, attn_A, V, dA, dV, B, T, H*n_heads);
    grad_softmax_rows(dA, attn_A, dS, B, T);
    grad_qk(dS, Q, K, dQ, dK, B, T, H);

    /* 5. Backward through Q/K/V linear projections */
    float *dx1_from_attn = scratch1;
    qkv_proj_backward(dQ, dK, dV, x0, W_Q, W_K, W_V,
                      dx1_from_attn, dW_Q, dW_K, dW_V,
                      B*T, D, D);

    free(dA); free(dQ); free(dK); free(dV); free(dS);

    /* 6. Backward through LayerNorm1 */
    layernorm_backward(dx1_from_attn, x0, ln1_xhat, gamma1,
                       dx0_out, dgamma1, dbeta1,
                       B*T, D, eps);

    /* dx0 += dx1 (attention residual) */
    for (int i = 0; i < B*T*D; i++) dx0_out[i] += scratch2[i];
}
```

---

## 5. Numerical Gradient Check

The most reliable way to verify a backward pass implementation is finite differences:

```
∂L/∂θ_i ≈ (L(θ + ε*e_i) - L(θ - ε*e_i)) / (2ε)
```

Compare this numeric estimate with the analytic gradient from backprop. Relative error < 1e-4 is acceptable for float32.

```c
#include <math.h>
#include <stdio.h>

/*
 * grad_check — verify analytic gradient against finite differences.
 *
 * forward  : function pointer (params → loss)
 * params   : parameter array [n_params]
 * analytic : analytic gradient from backprop [n_params]
 * n_params : number of parameters to check
 * n_check  : number of random parameters to spot-check (full check is slow)
 * eps      : finite difference step size (e.g. 1e-3)
 */
void grad_check(float (*forward)(const float *params, int n),
                const float *params, const float *analytic,
                int n_params, int n_check, float eps)
{
    float *params_plus  = (float *)malloc((size_t)n_params * sizeof(float));
    float *params_minus = (float *)malloc((size_t)n_params * sizeof(float));

    float max_rel_err = 0.0f;
    int   n_bad = 0;

    for (int c = 0; c < n_check; c++) {
        /* Pick a random parameter index */
        int i = rand() % n_params;

        /* Copy params */
        memcpy(params_plus,  params, (size_t)n_params * sizeof(float));
        memcpy(params_minus, params, (size_t)n_params * sizeof(float));
        params_plus[i]  += eps;
        params_minus[i] -= eps;

        float L_plus  = forward(params_plus,  n_params);
        float L_minus = forward(params_minus, n_params);
        float numeric = (L_plus - L_minus) / (2.0f * eps);

        float an  = analytic[i];
        float diff = fabsf(numeric - an);
        float norm = fabsf(numeric) + fabsf(an) + 1e-8f;
        float rel  = diff / norm;

        if (rel > 1e-3f) {
            printf("  FAIL param[%d]: numeric=%.6f analytic=%.6f rel=%.2e\n",
                   i, numeric, an, rel);
            n_bad++;
        }
        if (rel > max_rel_err) max_rel_err = rel;
    }

    printf("Gradient check: max_rel_err=%.2e, %d/%d passed\n",
           max_rel_err, n_check - n_bad, n_check);
    free(params_plus); free(params_minus);
}

/*
 * Example: 2-layer GPT gradient check.
 *
 * Build a tiny 2-layer model with B=2, T=8, D=32, n_heads=2.
 * Compute forward + backward. Then run grad_check on 100 random params.
 * If max_rel_err < 1e-3, the backward pass is correct.
 */
void run_gradient_check(void) {
    int B=2, T=8, D=32, n_heads=2, V=64, n_layers=2;
    int n_params = estimate_param_count(n_layers, D, V);   /* user-defined */

    float *params   = (float *)malloc((size_t)n_params * sizeof(float));
    float *grads    = (float *)calloc((size_t)n_params, sizeof(float));
    int   *tokens   = (int   *)malloc((size_t)B * T * sizeof(int));

    /* Random init */
    for (int i = 0; i < n_params; i++) params[i] = (float)rand()/RAND_MAX * 0.02f;
    for (int i = 0; i < B*T; i++) tokens[i] = rand() % V;

    /* Analytic backward */
    two_layer_forward_backward(params, tokens, grads, B, T, D, n_heads, V);

    /* Finite difference check */
    grad_check(
        /* forward only wrapper: */
        (float (*)(const float *, int))two_layer_forward_only,
        params, grads, n_params, 100, 1e-3f
    );

    free(params); free(grads); free(tokens);
}
```

---

## 6. Common Backward Pass Bugs

| Bug | Symptom | Fix |
|---|---|---|
| Missing `1/sqrt(H)` in dQ or dK | Gradients 5-30× too large for Q or K | Apply scale to both dQ and dK |
| `dS = dA^T @ ...` (transposed) | Gradient of K and Q swapped | Check: dQ uses dS @ K, dK uses dS^T @ Q |
| `+=` vs `=` for gradient accumulation | Gradients from later blocks overwrite earlier ones | All gradient outputs must use `+=` |
| Missing `1/N` normalization in CE backward | Gradient scales with batch size | Divide by N in fused_ce_backward |
| Wrong softmax backward | Loss decreases then spikes | Use `A * (dA - dot(dA, A))` formula exactly |
| LayerNorm dmean correction | Slow convergence, numerically unstable | Include `dvar * (-2/D)` term in dmean |
| Residual bypass not included in dx | Gradients "stop" at residual branch | `dx0 += dx1` after the branch backward |

---

## Key Takeaways

- **Attention backward** decomposes into three sequential operations: `A@V` backward (dV, dA), softmax backward (dS), `Q@K^T/sqrt(H)` backward (dQ, dK). Each is a straightforward matrix derivative.
- **Softmax Jacobian** is `A * (dA - dot(dA, A))` per row. This is derived from `d(softmax)/dx = diag(p) - pp^T` where `p` is the softmax output vector.
- **LayerNorm backward** requires careful accounting of dvar and dmean — they both depend on all inputs, so their gradients flow to every element of dx.
- **Residual connections** have trivially simple gradients: `dx0 = dx1 + dblock(dx1)`. The gradient flows unchanged through the skip connection.
- **Finite difference gradient check** is the gold standard for debugging backprop. Use `eps=1e-3` for float32 and check relative error `<1e-3`.
- **+=  discipline**: every backward output must accumulate (`+=`), never overwrite (`=`). Overwriting discards gradients from earlier paths through the computation graph.
- **GPT-2 backward** chains 12 blocks from output to input. Intermediate activations from all 12 blocks must be saved during the forward pass — this is the dominant memory cost of training.

---

**Previous**: [Training Loop](./36_Training_Loop.md) | **Next**: [Training GPT-2 Small](./38_Training_GPT2_Small.md)

> Next lesson assembles everything into a full GPT-2 124M training run, matching the llm.c benchmark on FineWebEdu.
