# 25. Attention Mechanism

**Previous**: [Layer Normalization](./24_Layer_Normalization.md) | **Next**: [KV Cache](./26_KV_Cache.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement multi-head self-attention (MHA) forward pass in C
2. Apply causal masking for autoregressive language modeling
3. Implement scaled dot-product attention with numerical stability
4. Implement the attention backward pass using saved Q, K, V, and attention weights
5. Verify MHA output numerically against PyTorch on a sample sequence

---

## 1. Multi-Head Attention Overview

```
Input:  X [N, T, d_model]

Step 1: Project to Q, K, V
  Q = X × W_Q  [N, T, d_model]
  K = X × W_K  [N, T, d_model]
  V = X × W_V  [N, T, d_model]

Step 2: Split into h heads, each of size d_head = d_model / h
  Q_i = Q[:, :, i*d_head:(i+1)*d_head]   [N, T, d_head]  for i = 0..h-1

Step 3: Scaled dot-product attention per head
  A_i = softmax(Q_i × K_i^T / √d_head + mask) × V_i
  mask[t, s] = -inf if s > t (causal: can't attend to future)

Step 4: Concatenate heads
  concat(A_0, ..., A_{h-1}) → [N, T, d_model]

Step 5: Output projection
  output = concat × W_O + b_O
```

---

## 2. QKV Projection

```c
// qkv_forward: project X to Q, K, V in a single fused matmul
// W_qkv: [3*d_model, d_model]  (fused weight matrix)
// b_qkv: [3*d_model]
// qkv_out: [N*T, 3*d_model]
void qkv_forward(
    const float *X,      // [N*T, d_model]  (M = N*T)
    const float *W_qkv,  // [3*d_model, d_model]
    const float *b_qkv,  // [3*d_model]
    float       *qkv,    // [N*T, 3*d_model]
    int M, int d_model) {

    // qkv = X × W_qkv^T + b_qkv
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, 3 * d_model, d_model,
                1.0f, X,     d_model,
                       W_qkv, d_model,
                0.0f, qkv, 3 * d_model);
    // Add bias
    for (int m = 0; m < M; m++)
    for (int j = 0; j < 3 * d_model; j++)
        qkv[m * 3 * d_model + j] += b_qkv[j];
}
```

---

## 3. Scaled Dot-Product Attention

```c
// attention_forward: single-head attention
// Q, K, V: [N, T, d_head]
// attn_w:  [N, T, T] — saved softmax weights for backward
// output:  [N, T, d_head]
void attention_forward(
    const float *Q,      // [N, T, d_head]
    const float *K,      // [N, T, d_head]
    const float *V,      // [N, T, d_head]
    float       *attn_w, // [N, T, T] — saved for backward
    float       *output, // [N, T, d_head]
    int N, int T, int d_head,
    int causal) {        // 1 = causal mask

    float scale = 1.0f / sqrtf((float)d_head);

    for (int n = 0; n < N; n++) {
        const float *Qn = Q + (long)n * T * d_head;
        const float *Kn = K + (long)n * T * d_head;
        const float *Vn = V + (long)n * T * d_head;
        float       *An = attn_w + (long)n * T * T;
        float       *On = output  + (long)n * T * d_head;

        // scores[T, T] = Q[T, d_head] × K^T[d_head, T] × scale
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    T, T, d_head,
                    scale, Qn, d_head,
                           Kn, d_head,
                    0.0f, An, T);

        // Causal mask: set upper triangle to -inf
        if (causal) {
            for (int t = 0; t < T; t++)
            for (int s = t + 1; s < T; s++)
                An[t * T + s] = -1e9f;
        }

        // Softmax over last dim (per query position t)
        for (int t = 0; t < T; t++) {
            float *row = An + t * T;
            float max_v = row[0];
            for (int s = 1; s <= t || (!causal && s < T); s++)
                if (row[s] > max_v) max_v = row[s];
            float sum = 0.0f;
            int lim = causal ? t + 1 : T;
            for (int s = 0; s < lim; s++) {
                row[s] = expf(row[s] - max_v);
                sum += row[s];
            }
            for (int s = 0; s < lim; s++) row[s] /= sum;
            if (causal)
                for (int s = lim; s < T; s++) row[s] = 0.0f;
        }

        // output[T, d_head] = attn_w[T, T] × V[T, d_head]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    T, d_head, T,
                    1.0f, An,  T,
                          Vn, d_head,
                    0.0f, On, d_head);
    }
}
```

---

## 4. Full Multi-Head Attention Forward

```c
// mha_forward: full multi-head self-attention
// Fuses QKV projection + per-head attention + output projection
void mha_forward(
    const float *X,      // [N, T, d_model]
    const float *W_qkv,  // [3*d_model, d_model]
    const float *b_qkv,  // [3*d_model]
    const float *W_o,    // [d_model, d_model]
    const float *b_o,    // [d_model]
    float       *output, // [N, T, d_model]
    // saved for backward:
    float       *qkv_buf,   // [N*T, 3*d_model]
    float       *attn_w,    // [N, n_heads, T, T]
    float       *head_out,  // [N, n_heads, T, d_head]
    int N, int T, int d_model, int n_heads, int causal) {

    int M = N * T;
    int d_head = d_model / n_heads;

    // 1. Fused QKV projection
    qkv_forward(X, W_qkv, b_qkv, qkv_buf, M, d_model);

    // 2. Per-head attention
    for (int h = 0; h < n_heads; h++) {
        // Extract Q, K, V for head h (interleaved in qkv_buf)
        // qkv_buf: [M, 3*d_model] = [M, n_heads*d_head * 3]
        // Q_h is at offset h*d_head within each row's first d_model elements

        // For simplicity, copy Q/K/V for this head into contiguous buffers
        float *Qh = malloc(M * d_head * sizeof(float));
        float *Kh = malloc(M * d_head * sizeof(float));
        float *Vh = malloc(M * d_head * sizeof(float));

        for (int m = 0; m < M; m++) {
            const float *row = qkv_buf + m * 3 * d_model;
            memcpy(Qh + m * d_head, row + h * d_head,           d_head * sizeof(float));
            memcpy(Kh + m * d_head, row + d_model + h * d_head, d_head * sizeof(float));
            memcpy(Vh + m * d_head, row + 2*d_model + h * d_head, d_head * sizeof(float));
        }

        // Reshape to [N, T, d_head]
        float *attn_wh = attn_w   + (long)h * N * T * T;
        float *head_oh = head_out + (long)h * N * T * d_head;

        attention_forward(Qh, Kh, Vh, attn_wh, head_oh,
                          N, T, d_head, causal);
        free(Qh); free(Kh); free(Vh);
    }

    // 3. Concatenate heads: [N, n_heads, T, d_head] → [N, T, d_model]
    float *concat = malloc(M * d_model * sizeof(float));
    for (int m = 0; m < M; m++)
    for (int h = 0; h < n_heads; h++)
    for (int j = 0; j < d_head; j++) {
        int n = m / T, t = m % T;
        concat[m * d_model + h * d_head + j]
            = head_out[((long)h * N + n) * T * d_head + t * d_head + j];
    }

    // 4. Output projection: [M, d_model] × W_o^T + b_o
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, d_model, d_model,
                1.0f, concat, d_model,
                       W_o,   d_model,
                0.0f, output, d_model);
    for (int m = 0; m < M; m++)
    for (int j = 0; j < d_model; j++)
        output[m * d_model + j] += b_o[j];

    free(concat);
}
```

---

## 5. Causal Mask Visualization

```
T=4 sequence, causal mask:

Score matrix (before mask):        After causal mask:
  pos  0    1    2    3              0    1    2    3
  0  [s00  s01  s02  s03]    0  [s00  -∞   -∞   -∞ ]
  1  [s10  s11  s12  s13]    1  [s10  s11  -∞   -∞ ]
  2  [s20  s21  s22  s23]    2  [s20  s21  s22  -∞ ]
  3  [s30  s31  s32  s33]    3  [s30  s31  s32  s33]

Row 0 (token 0): only attends to itself
Row 1 (token 1): attends to tokens 0 and 1
Row 3 (token 3): attends to all previous tokens
```

---

## 6. Attention Backward (High Level)

The attention backward requires backpropagating through softmax and two matmuls:

```c
// attention_backward: backprop through scaled dot-product attention
// Given: dO [N, T, d_head], saved: Q, K, V, attn_w
// Produces: dQ, dK, dV [N, T, d_head]
void attention_backward(
    const float *dO,     // [N, T, d_head]
    const float *Q, const float *K, const float *V,
    const float *attn_w, // [N, T, T] — saved softmax output
    float *dQ, float *dK, float *dV,
    int N, int T, int d_head, int causal) {

    float scale = 1.0f / sqrtf((float)d_head);

    for (int n = 0; n < N; n++) {
        const float *An  = attn_w + (long)n * T * T;
        const float *Qn  = Q  + (long)n * T * d_head;
        const float *Kn  = K  + (long)n * T * d_head;
        const float *Vn  = V  + (long)n * T * d_head;
        const float *dOn = dO + (long)n * T * d_head;
        float       *dQn = dQ + (long)n * T * d_head;
        float       *dKn = dK + (long)n * T * d_head;
        float       *dVn = dV + (long)n * T * d_head;

        // dV = A^T × dO   [T, d_head]
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    T, d_head, T,
                    1.0f, An,  T,
                          dOn, d_head,
                    1.0f, dVn, d_head);  // accumulate

        // dA = dO × V^T   [T, T]
        float *dA = calloc(T * T, sizeof(float));
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    T, T, d_head,
                    1.0f, dOn, d_head,
                          Vn,  d_head,
                    0.0f, dA, T);

        // Softmax backward: dS = A ⊙ (dA - Σ_j A_j dA_j)
        float *dS = malloc(T * T * sizeof(float));
        for (int t = 0; t < T; t++) {
            float dot = 0.0f;
            int lim = causal ? t + 1 : T;
            for (int s = 0; s < lim; s++) dot += An[t*T+s] * dA[t*T+s];
            for (int s = 0; s < lim; s++)
                dS[t*T+s] = An[t*T+s] * (dA[t*T+s] - dot);
            for (int s = lim; s < T; s++) dS[t*T+s] = 0.0f;
        }

        // dQ = dS × K × scale   [T, d_head]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    T, d_head, T,
                    scale, dS, T, Kn, d_head,
                    1.0f, dQn, d_head);

        // dK = dS^T × Q × scale  [T, d_head]
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    T, d_head, T,
                    scale, dS, T, Qn, d_head,
                    1.0f, dKn, d_head);

        free(dA); free(dS);
    }
}
```

---

## Key Takeaways

- **QKV projection**: single fused matmul `X × W_qkv^T` produces Q, K, V simultaneously
- **Scaled dot-product**: divide by √d_head to prevent vanishing gradients in softmax when d_head is large
- **Causal mask**: set upper-triangular entries to -∞ before softmax → autoregressive generation
- Attention backward requires saving `Q, K, V, attn_w` — memory cost grows as O(N × T²)
- FlashAttention (Lesson 41) reduces this memory cost to O(N × T) by fusing the attention computation

---

**Next**: [26. KV Cache](./26_KV_Cache.md) — Pre-allocate KV cache for efficient autoregressive generation; analyze memory usage per token; implement cache append logic.
