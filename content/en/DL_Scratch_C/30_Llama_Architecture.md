# 30. Llama Architecture

**Previous**: [GPT-2 Forward Pass](./29_GPT2_Forward_Pass.md) | **Next**: [Vision Transformer](./31_Vision_Transformer_ViT.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. List the four architectural differences between GPT-2 and Llama 2/3
2. Implement Grouped Query Attention (GQA) with configurable n_kv_heads
3. Integrate RoPE into the attention computation
4. Assemble a Llama forward pass using RMSNorm + SwiGLU + GQA + RoPE
5. Verify Llama forward outputs against a reference implementation

---

## 1. Llama vs GPT-2: Four Key Differences

```
Component         GPT-2                   Llama 2/3
──────────────────────────────────────────────────────────
Normalization     LayerNorm (w/ bias)     RMSNorm (no bias, no mean)
FFN activation    GELU                    SwiGLU (gated)
Position encoding Learned absolute (wpe)  RoPE (applied to Q,K in attention)
Attention heads   MHA: n_kv = n_heads     GQA: n_kv_heads < n_heads

Llama 3 8B specifics:
  n_layers:   32
  n_heads:    32
  n_kv_heads: 8       ← GQA: 4 Q heads share 1 KV head
  d_model:    4096
  d_head:     128     (4096 / 32)
  d_ffn:      14336   (≈ 3.5 × d_model, SwiGLU with 2/3×8/3 factor)
  T_max:      8192    (Llama 3 base), 128K (Llama 3 Instruct with rope scaling)
  V:          128,256
```

---

## 2. Grouped Query Attention (GQA)

Standard MHA: each attention head has its own K, V → expensive KV cache.

GQA: groups of Q heads share a single K, V:

```
n_heads = 32, n_kv_heads = 8:
  Group 0:  Q[0], Q[1], Q[2], Q[3]  → share K[0], V[0]
  Group 1:  Q[4], Q[5], Q[6], Q[7]  → share K[1], V[1]
  ...
  Group 7:  Q[28]..Q[31]            → share K[7], V[7]

KV cache memory: 8/32 = 25% of full MHA
Accuracy impact: minimal (Llama 2 70B with GQA ≈ MHA)
```

```c
typedef struct {
    int d_model, n_heads, n_kv_heads, d_head;
    float *q_w;   // [n_heads * d_head, d_model]
    float *k_w;   // [n_kv_heads * d_head, d_model]
    float *v_w;   // [n_kv_heads * d_head, d_model]
    float *o_w;   // [d_model, n_heads * d_head]
} GQAWeights;

// gqa_forward: Grouped Query Attention
void gqa_forward(
    const float *X,       // [N, T, d_model]
    GQAWeights  *w,
    const float *cos_t,   // [T, d_head/2]  RoPE cosines
    const float *sin_t,   // [T, d_head/2]  RoPE sines
    float       *output,  // [N, T, d_model]
    int N, int T,
    KVCache *cache, int cache_layer) {

    int d   = w->d_model;
    int nh  = w->n_heads;
    int nkv = w->n_kv_heads;
    int dh  = w->d_head;
    int M   = N * T;
    int gqa_factor = nh / nkv;

    // Project Q [M, nh*dh]
    float *Q = malloc(M * nh * dh * sizeof(float));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, nh * dh, d,
                1.0f, X,    d,
                       w->q_w, d,
                0.0f, Q, nh * dh);

    // Project K, V [M, nkv*dh]
    float *K = malloc(M * nkv * dh * sizeof(float));
    float *V = malloc(M * nkv * dh * sizeof(float));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, nkv * dh, d, 1.0f, X, d, w->k_w, d, 0.0f, K, nkv * dh);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, nkv * dh, d, 1.0f, X, d, w->v_w, d, 0.0f, V, nkv * dh);

    // Apply RoPE to Q [M, nh, dh] and K [M, nkv, dh]
    rope_apply(Q, cos_t, sin_t, N, nh, T, dh);
    rope_apply(K, cos_t, sin_t, N, nkv, T, dh);

    // Append K, V to KV cache
    if (cache) {
        KVLayer *kl = &cache->layers[cache_layer];
        int pos = kl->pos;
        // Append new tokens (assume T new tokens starting at pos)
        for (int t = 0; t < T; t++) {
            for (int n = 0; n < N; n++) {
                kvcache_append(kl,
                    K + (long)(n * T + t) * nkv * dh,
                    V + (long)(n * T + t) * nkv * dh,
                    pos + t);
            }
        }
        kl->pos += T;
    }

    // Compute attention per head (with GQA grouping)
    float *head_out = malloc(M * nh * dh * sizeof(float));
    float scale = 1.0f / sqrtf((float)dh);
    float *scores = malloc(T * sizeof(float));

    for (int n = 0; n < N; n++)
    for (int h = 0; h < nh; h++) {
        int kv_h = h / gqa_factor;  // which KV head to use

        for (int t_q = 0; t_q < T; t_q++) {
            const float *q = Q + (long)(n * T + t_q) * nh * dh + h * dh;
            float       *o = head_out + (long)(n * T + t_q) * nh * dh + h * dh;

            int T_kv = cache ? cache->layers[cache_layer].pos : T;
            float *sc = malloc(T_kv * sizeof(float));

            // Attention scores: q · k_t for each k in cache
            for (int t_k = 0; t_k <= t_q || (cache && t_k < T_kv); t_k++) {
                const float *k;
                if (cache) {
                    k = cache->layers[cache_layer].k
                        + (long)t_k * nkv * dh + kv_h * dh;
                } else {
                    k = K + (long)(n * T + t_k) * nkv * dh + kv_h * dh;
                }
                float dot = 0.0f;
                for (int j = 0; j < dh; j++) dot += q[j] * k[j];
                sc[t_k] = dot * scale;
            }

            // Softmax
            int T_att = cache ? T_kv : t_q + 1;
            float max_s = sc[0];
            for (int t = 1; t < T_att; t++) if (sc[t] > max_s) max_s = sc[t];
            float sum = 0.0f;
            for (int t = 0; t < T_att; t++) { sc[t] = expf(sc[t]-max_s); sum += sc[t]; }
            for (int t = 0; t < T_att; t++) sc[t] /= sum;

            // Output: sc × V
            memset(o, 0, dh * sizeof(float));
            for (int t = 0; t < T_att; t++) {
                const float *v;
                if (cache) {
                    v = cache->layers[cache_layer].v
                        + (long)t * nkv * dh + kv_h * dh;
                } else {
                    v = V + (long)(n * T + t) * nkv * dh + kv_h * dh;
                }
                for (int j = 0; j < dh; j++) o[j] += sc[t] * v[j];
            }
            free(sc);
        }
    }
    free(scores); free(K); free(V); free(Q);

    // Output projection: [M, nh*dh] → [M, d]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, d, nh * dh,
                1.0f, head_out, nh * dh,
                       w->o_w, nh * dh,
                0.0f, output, d);
    free(head_out);
}
```

---

## 3. Llama Block

```c
// Llama block (vs GPT-2 block):
//   RMSNorm instead of LayerNorm
//   GQA + RoPE instead of standard MHA
//   SwiGLU instead of GELU FFN
//   No bias terms in attention or FFN

void llama_block_forward(
    const float *X,        // [M, d]
    // RMSNorm 1 (before attn)
    const float *rn1_w,    // [d]
    // GQA weights
    GQAWeights  *attn_w,
    const float *cos_t, const float *sin_t,
    // RMSNorm 2 (before FFN)
    const float *rn2_w,    // [d]
    // SwiGLU FFN
    const float *gate_w, const float *up_w, const float *down_w,
    // output
    float       *Y,        // [M, d]
    // buffers (saved for backward)
    float *rn1_out, float *rn1_rrms,
    float *attn_out,
    float *rn2_out, float *rn2_rrms,
    float *ffn_gate_buf, float *ffn_up_buf,
    int N, int T, int d, int d_ffn,
    KVCache *cache, int layer_idx) {

    int M = N * T;

    // 1. RMSNorm 1
    rmsnorm_forward(X, rn1_w, rn1_out, rn1_rrms, M, d);

    // 2. GQA + RoPE attention
    gqa_forward(rn1_out, attn_w, cos_t, sin_t, attn_out,
                N, T, cache, layer_idx);

    // 3. Residual add 1
    float *x1 = malloc(M * d * sizeof(float));
    for (int i = 0; i < M * d; i++) x1[i] = X[i] + attn_out[i];

    // 4. RMSNorm 2
    rmsnorm_forward(x1, rn2_w, rn2_out, rn2_rrms, M, d);

    // 5. SwiGLU FFN
    float *ffn_out = malloc(M * d * sizeof(float));
    llama_ffn_forward(rn2_out, gate_w, up_w, down_w,
                      ffn_gate_buf, ffn_up_buf, ffn_out,
                      M, d, d_ffn);

    // 6. Residual add 2
    for (int i = 0; i < M * d; i++) Y[i] = x1[i] + ffn_out[i];
    free(x1); free(ffn_out);
}
```

---

## 4. Llama Parameter Count

```
Llama 3 8B:
  n_layers=32, n_heads=32, n_kv_heads=8, d=4096, d_ffn=14336, V=128256

Per layer:
  Q:    n_heads  × d_head × d = 32 × 128 × 4096  = 16.8M
  K:    n_kv_heads × d_head × d = 8 × 128 × 4096 =  4.2M
  V:    same as K                                  =  4.2M
  O:    d_model × n_heads × d_head = 4096 × 4096  = 16.8M
  FFN gate: d_ffn × d = 14336 × 4096              = 58.7M
  FFN up:   same                                   = 58.7M
  FFN down: d × d_ffn                              = 58.7M
  RMSNorm: 2 × d = 8K
  Total per layer: ~218M

32 layers:  ~7.0B
Embeddings: 128256 × 4096 = 525M
Total:      ~8B params  ✓

KV cache reduction from GQA:
  MHA (32 KV heads): 32 × 128 × 2 = 8192 bytes/token/layer
  GQA ( 8 KV heads):  8 × 128 × 2 = 2048 bytes/token/layer  (4× smaller)
```

---

## Key Takeaways

- **Llama vs GPT-2**: RMSNorm, SwiGLU, RoPE, GQA — each change improves efficiency or quality
- **GQA**: `n_kv_heads` Q-groups share one K/V pair — 4× KV cache reduction in Llama 3 8B with minimal accuracy loss
- **RoPE integration**: apply `rope_apply(Q, cos, sin)` and `rope_apply(K, cos, sin)` before computing attention scores — not to V or the output
- Llama has no bias terms in attention or FFN — simplifies backward and reduces parameters
- The architecture is modular: each component (RMSNorm, GQA, SwiGLU) is independently testable against a reference

---

**Next**: [31. Vision Transformer (ViT)](./31_Vision_Transformer_ViT.md) — Apply self-attention to image patches: patch embedding, [CLS] token, 2D positional encoding, and ViT-Base forward pass.
