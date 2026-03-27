# 27. FFN and Activations

**Previous**: [KV Cache](./26_KV_Cache.md) | **Next**: [Transformer Block](./28_Transformer_Block.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement the GPT-2 FFN (two FC layers with GELU activation)
2. Derive and implement the GELU backward pass
3. Implement the SwiGLU FFN used in Llama (gated architecture)
4. Explain why SwiGLU outperforms standard FFN
5. Count FFN parameters and FLOPs as a fraction of total model compute

---

## 1. Feed-Forward Network Structure

The FFN in a Transformer is a two-layer MLP applied independently to each token:

```
GPT-2 FFN (non-gated):
  x → FC(d_model → 4*d_model) → GELU → FC(4*d_model → d_model)

Llama FFN (gated / SwiGLU):
  x → [FC_gate(d_model → d_ffn) × SiLU]  ⊙  FC_up(d_model → d_ffn)
    → FC_down(d_ffn → d_model)

where d_ffn ≈ 2/3 × 4 × d_model  (rounded to multiple of 64 in Llama)
```

---

## 2. GELU Activation (GPT-2)

GELU (Gaussian Error Linear Unit) approximation used in GPT-2:

```
GELU(x) ≈ 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))

Exact: GELU(x) = x × Φ(x)  where Φ is the CDF of N(0,1)
```

```c
#include <math.h>

#define SQRT_2_OVER_PI 0.7978845608f  // √(2/π)
#define GELU_COEF      0.044715f

// GELU forward (fast tanh approximation)
static inline float gelu(float x) {
    float inner = SQRT_2_OVER_PI * (x + GELU_COEF * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

void gelu_forward(float *X, int size) {
    for (int i = 0; i < size; i++) X[i] = gelu(X[i]);
}

// GELU backward: d(GELU)/dx
// Using saved pre-activation x (not post-GELU output)
static inline float gelu_grad(float x) {
    float inner = SQRT_2_OVER_PI * (x + GELU_COEF * x * x * x);
    float tanh_v = tanhf(inner);
    float sech2  = 1.0f - tanh_v * tanh_v;
    float dtanh  = SQRT_2_OVER_PI * (1.0f + 3.0f * GELU_COEF * x * x);
    return 0.5f * (1.0f + tanh_v) + 0.5f * x * sech2 * dtanh;
}

void gelu_backward(float *dX, const float *X_pre, int size) {
    for (int i = 0; i < size; i++)
        dX[i] *= gelu_grad(X_pre[i]);
}
```

---

## 3. GPT-2 FFN Forward Pass

```c
// gpt2_ffn_forward: two-layer MLP with GELU
// input:  [M, d_model]   (M = N*T)
// fc1_w: [4*d, d]  fc1_b: [4*d]
// fc2_w: [d, 4*d]  fc2_b: [d]
// buf:    [M, 4*d] — intermediate (saved for backward)
void gpt2_ffn_forward(
    const float *input,   // [M, d_model]
    const float *fc1_w,   // [4*d, d]
    const float *fc1_b,   // [4*d]
    const float *fc2_w,   // [d, 4*d]
    const float *fc2_b,   // [d]
    float       *buf,     // [M, 4*d] — saved for backward
    float       *output,  // [M, d]
    int M, int d) {

    int d4 = 4 * d;

    // FC1: [M, d] × [d, 4d]^T → [M, 4d] + b1
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, d4, d,
                1.0f, input, d,
                       fc1_w, d,
                0.0f, buf, d4);
    for (int m = 0; m < M; m++)
    for (int j = 0; j < d4; j++)
        buf[m * d4 + j] += fc1_b[j];

    // GELU in-place
    gelu_forward(buf, M * d4);

    // FC2: [M, 4d] × [4d, d]^T → [M, d] + b2
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, d, d4,
                1.0f, buf,   d4,
                       fc2_w, d4,
                0.0f, output, d);
    for (int m = 0; m < M; m++)
    for (int j = 0; j < d; j++)
        output[m * d + j] += fc2_b[j];
}

// FFN parameter count: (d × 4d + 4d) + (4d × d + d) = 8d² + 5d ≈ 8d²
// For d=768: 8 × 768² ≈ 4.7M params per layer
// For d=4096 (Llama 7B): 8 × 4096² ≈ 134M params per layer
```

---

## 4. SiLU Activation

SiLU (Sigmoid Linear Unit) = Swish activation, used in Llama's SwiGLU:

```
SiLU(x) = x × σ(x) = x / (1 + e^{-x})
```

```c
static inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

static inline float silu_grad(float x) {
    float sig = 1.0f / (1.0f + expf(-x));
    return sig + x * sig * (1.0f - sig);
}

void silu_forward(float *X, int size) {
    for (int i = 0; i < size; i++) X[i] = silu(X[i]);
}

void silu_backward(float *dX, const float *X_pre, int size) {
    for (int i = 0; i < size; i++)
        dX[i] *= silu_grad(X_pre[i]);
}
```

---

## 5. SwiGLU FFN (Llama / Mistral)

```
SwiGLU(x) = SiLU(W_gate × x) ⊙ (W_up × x)
output = W_down × SwiGLU(x)

vs GPT-2:
  GPT-2: GELU(W1 × x) → W2
  Llama: SiLU(W_gate × x) ⊙ (W_up × x) → W_down  (two separate up-projections)
```

```c
// llama_ffn_forward: SwiGLU gated FFN
// gate_w: [d_ffn, d]  up_w: [d_ffn, d]  down_w: [d, d_ffn]
void llama_ffn_forward(
    const float *input,    // [M, d]
    const float *gate_w,   // [d_ffn, d]
    const float *up_w,     // [d_ffn, d]
    const float *down_w,   // [d, d_ffn]
    float       *gate_buf, // [M, d_ffn] — saved for backward
    float       *up_buf,   // [M, d_ffn] — saved for backward
    float       *output,   // [M, d]
    int M, int d, int d_ffn) {

    // Gate branch: W_gate × x
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, d_ffn, d,
                1.0f, input,  d,
                       gate_w, d,
                0.0f, gate_buf, d_ffn);

    // Up branch: W_up × x
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, d_ffn, d,
                1.0f, input, d,
                       up_w,  d,
                0.0f, up_buf, d_ffn);

    // SwiGLU: gate_buf = SiLU(gate_buf) ⊙ up_buf
    for (int i = 0; i < M * d_ffn; i++)
        gate_buf[i] = silu(gate_buf[i]) * up_buf[i];

    // Down projection: W_down × SwiGLU_out
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, d, d_ffn,
                1.0f, gate_buf, d_ffn,
                       down_w,  d_ffn,
                0.0f, output,   d);
}

// Llama d_ffn formula: round up to multiple of 256
int llama_ffn_dim(int d_model, int multiple_of) {
    int ffn = (int)(d_model * 8.0 / 3.0);  // ≈ 2.67 × d_model
    return ((ffn + multiple_of - 1) / multiple_of) * multiple_of;
}
// Llama 7B: d=4096, d_ffn=11008 (≈ 2.69 × 4096)
```

---

## 6. SwiGLU Backward

```c
// llama_ffn_backward: backprop through SwiGLU
void llama_ffn_backward(
    const float *doutput,   // [M, d]
    const float *input,     // [M, d]  — original input
    const float *gate_pre,  // [M, d_ffn] — pre-SiLU gate values
    const float *up_buf,    // [M, d_ffn] — up projection output
    const float *gate_silu, // [M, d_ffn] — SiLU(gate) ⊙ up
    const float *gate_w, const float *up_w, const float *down_w,
    float *dinput,    // [M, d]
    float *dgate_w, float *dup_w, float *ddown_w,
    int M, int d, int d_ffn) {

    // 1. dgate_silu = doutput × W_down^T    [M, d_ffn]
    float *dg_silu = calloc(M * d_ffn, sizeof(float));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, d_ffn, d,
                1.0f, doutput, d, down_w, d_ffn,
                0.0f, dg_silu, d_ffn);

    // dW_down += gate_silu^T × doutput    [d_ffn, d]
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                d_ffn, d, M,
                1.0f, gate_silu, d_ffn, doutput, d,
                1.0f, ddown_w, d);

    // 2. Backward through SwiGLU gating
    // gate_silu = SiLU(gate_pre) ⊙ up_buf
    // d_gate_pre = dg_silu ⊙ up_buf ⊙ SiLU'(gate_pre)
    // d_up       = dg_silu ⊙ SiLU(gate_pre)
    float *d_gate_pre = malloc(M * d_ffn * sizeof(float));
    float *d_up       = malloc(M * d_ffn * sizeof(float));
    for (int i = 0; i < M * d_ffn; i++) {
        float g = gate_pre[i];
        float silu_g = silu(g);
        d_gate_pre[i] = dg_silu[i] * up_buf[i] * silu_grad(g);
        d_up[i]       = dg_silu[i] * silu_g;
    }
    free(dg_silu);

    // 3. dW_gate, dW_up, dinput
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                d_ffn, d, M,
                1.0f, d_gate_pre, d_ffn, input, d,
                1.0f, dgate_w, d);
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                d_ffn, d, M,
                1.0f, d_up, d_ffn, input, d,
                1.0f, dup_w, d);

    // dinput from gate branch + up branch
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, d, d_ffn,
                1.0f, d_gate_pre, d_ffn, gate_w, d,
                1.0f, dinput, d);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, d, d_ffn,
                1.0f, d_up, d_ffn, up_w, d,
                1.0f, dinput, d);

    free(d_gate_pre); free(d_up);
}
```

---

## 7. FFN Compute Share

```
Transformer block compute breakdown:
  Attention QKV:    2 × T × d² × 3    (Q,K,V projections)
  Attention scores: 2 × T² × d_head × h  (QK^T and softmax×V)
  Attention out:    2 × T × d²          (output projection)
  FFN (GPT-2):      2 × T × d × 4d × 2 = 16 × T × d²

For T << d (typical for batched training, T=1024, d=768):
  FFN dominates: 16d² vs 6d² (attn) + 2T × d (QK scores)

Compute split at T=1024, d=768:
  Attention projections: 6 × 768² = 3.5M FLOPs/token/layer
  FFN:                  16 × 768² = 9.4M FLOPs/token/layer
  → FFN is ~73% of total compute
```

---

## Key Takeaways

- GPT-2 FFN: `GELU(x × W1^T + b1) × W2^T + b2` — standard two-layer MLP with GELU
- **SwiGLU** (Llama): `(SiLU(W_gate × x) ⊙ W_up × x) × W_down` — gated architecture with two parallel up-projections
- GELU backward: `dX *= 0.5 × (1 + tanh(inner)) + 0.5 × x × sech²(inner) × dtanh`
- SwiGLU backward: gradient splits at the ⊙ operator — each branch receives `dout × other_branch`
- FFN accounts for ~73% of Transformer compute (at typical T/d ratios) — it is the primary compute bottleneck

---

**Next**: [28. Transformer Block](./28_Transformer_Block.md) — Assemble the full pre-norm residual block: LN → attention → residual → LN → FFN → residual, and verify numerically against PyTorch.
