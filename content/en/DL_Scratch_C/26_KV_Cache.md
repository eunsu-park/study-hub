# 26. KV Cache

**Previous**: [Attention Mechanism](./25_Attention_Mechanism.md) | **Next**: [FFN and Activations](./27_FFN_and_Activations.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why KV cache is essential for efficient autoregressive generation
2. Pre-allocate a KV cache and implement the append-only write pattern
3. Calculate memory usage per token per layer for a given model size
4. Implement attention with KV cache (query only the current token, attend to full cache)
5. Handle the cache fill and cache eviction (when context exceeds T_max)

---

## 1. The Problem with Naive Autoregressive Generation

Without KV cache, generating token `t` requires:

```
Decode step t:
  Input: all tokens 0..t
  Forward pass: [1, t+1, d_model] through all L layers
  Attention at layer l: Q,K,V ∈ [1, t+1, d_head×h]
  Attention score: [t+1, t+1] matrix

Cost per token:  O(t × L × d_model²)  → grows quadratically with sequence length!

Generating 2K tokens, L=32, d=4096:
  Step 1:    1 token processed
  Step 2000: 2000 tokens reprocessed
  Total: Σ_{t=1}^{2000} t × 32 × 4096² ≈ 10^14 FLOPs  ← completely impractical
```

---

## 2. KV Cache: Store and Reuse

The insight: **K and V for past tokens don't change** — only Q changes at each step.

```
Prefill phase (process prompt tokens 0..P):
  - Compute K, V for all prompt tokens
  - Store in cache[l][0..P-1]

Decode phase (generate token t = P, P+1, ...):
  - Compute Q, K, V for NEW token t only (not the whole sequence!)
  - Append K[t], V[t] to cache[l][t] (one new row per layer)
  - Attend: Q_t [1, d_head] × K_cache[0..t, d_head]^T → scores [1, t+1]
  - Apply softmax → output [1, d_head]

Cost per decode step: O(t × L × d_head × h) + O(L × d_model²)
                                 ↑ linear in sequence length (not quadratic in t²)
```

---

## 3. KV Cache Data Structure

```c
typedef struct {
    float *k;     // [T_max, n_kv_heads, d_head]  key cache per layer
    float *v;     // [T_max, n_kv_heads, d_head]  value cache per layer
    int   pos;    // current position (number of tokens in cache)
    int   T_max;  // maximum context length
    int   n_kv_heads;
    int   d_head;
} KVLayer;

typedef struct {
    KVLayer *layers;   // [n_layers]
    int      n_layers;
} KVCache;

// Allocate KV cache for one forward pass
KVCache *kvcache_create(int n_layers, int T_max, int n_kv_heads, int d_head) {
    KVCache *cache = malloc(sizeof(KVCache));
    cache->n_layers = n_layers;
    cache->layers   = malloc(n_layers * sizeof(KVLayer));

    for (int l = 0; l < n_layers; l++) {
        KVLayer *kl = &cache->layers[l];
        kl->pos      = 0;
        kl->T_max    = T_max;
        kl->n_kv_heads = n_kv_heads;
        kl->d_head   = d_head;
        size_t sz = (size_t)T_max * n_kv_heads * d_head * sizeof(float);
        kl->k = malloc(sz);
        kl->v = malloc(sz);
    }
    return cache;
}

void kvcache_free(KVCache *cache) {
    for (int l = 0; l < cache->n_layers; l++) {
        free(cache->layers[l].k);
        free(cache->layers[l].v);
    }
    free(cache->layers);
    free(cache);
}

void kvcache_reset(KVCache *cache) {
    for (int l = 0; l < cache->n_layers; l++)
        cache->layers[l].pos = 0;
}
```

---

## 4. Append K/V to Cache

```c
// Append new K and V for the current token at position `pos`
void kvcache_append(
    KVLayer     *kl,
    const float *k_new,  // [n_kv_heads, d_head]
    const float *v_new,  // [n_kv_heads, d_head]
    int pos) {

    assert(pos < kl->T_max);
    int stride = kl->n_kv_heads * kl->d_head;
    memcpy(kl->k + (long)pos * stride, k_new, stride * sizeof(float));
    memcpy(kl->v + (long)pos * stride, v_new, stride * sizeof(float));
}
```

---

## 5. Attention with KV Cache

During decode, Q has shape [1, d_head] per head (only the new token):

```c
// cached_attention_forward: attention for ONE new token against full KV cache
// q_new:  [n_heads, d_head]   — query for current token only
// cache:  KVLayer with pos tokens already stored
// out:    [n_heads, d_head]
void cached_attention_forward(
    const float *q_new,  // [n_heads, d_head]
    KVLayer     *kl,
    float       *out,    // [n_heads, d_head]
    int n_heads, int n_kv_heads, int d_head) {

    int T = kl->pos;  // number of cached tokens
    int kv_stride = n_kv_heads * d_head;
    float scale = 1.0f / sqrtf((float)d_head);

    // For GQA: n_queries_per_kv = n_heads / n_kv_heads
    int gqa_factor = n_heads / n_kv_heads;

    float *scores = malloc(T * sizeof(float));
    float *attn   = malloc(T * sizeof(float));

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / gqa_factor;  // which KV head to use (GQA)

        const float *q = q_new + h * d_head;
        float       *o = out   + h * d_head;

        // scores[t] = Q · K[t] × scale
        for (int t = 0; t < T; t++) {
            const float *k = kl->k + (long)t * kv_stride + kv_h * d_head;
            float dot = 0.0f;
            for (int j = 0; j < d_head; j++) dot += q[j] * k[j];
            scores[t] = dot * scale;
        }

        // Softmax over T positions
        float max_s = scores[0];
        for (int t = 1; t < T; t++) if (scores[t] > max_s) max_s = scores[t];
        float sum = 0.0f;
        for (int t = 0; t < T; t++) { attn[t] = expf(scores[t] - max_s); sum += attn[t]; }
        for (int t = 0; t < T; t++) attn[t] /= sum;

        // out = Σ_t attn[t] × V[t]
        memset(o, 0, d_head * sizeof(float));
        for (int t = 0; t < T; t++) {
            const float *v = kl->v + (long)t * kv_stride + kv_h * d_head;
            float a = attn[t];
            for (int j = 0; j < d_head; j++) o[j] += a * v[j];
        }
    }
    free(scores); free(attn);
}
```

---

## 6. Memory Analysis

```
KV cache memory per token per layer:
  K: n_kv_heads × d_head × 4 bytes (FP32)
  V: n_kv_heads × d_head × 4 bytes

Total per token: 2 × n_kv_heads × d_head × 4 bytes

Model examples (FP16 = 2 bytes):
  GPT-2 small  (L=12, h=12, d_head=64):
    per token = 12 × 12 × 64 × 2 × 2 bytes = 36,864 bytes ≈ 36 KB
    for 1K context: 36 KB × 12 layers × 1024 = 442 MB

  Llama 3 8B  (L=32, n_kv_heads=8, d_head=128):
    per token = 2 × 8 × 128 × 2 = 4096 bytes = 4 KB
    for 128K context: 4 KB × 32 layers × 131072 = 16 GB

  Llama 3 8B uses GQA (n_kv_heads=8 vs n_heads=32):
    vs full MHA: 4× less KV memory (32→8 KV heads)
```

```c
void print_kvcache_memory(int n_layers, int n_kv_heads, int d_head,
                          int T_max, int dtype_bytes) {
    long per_token = 2L * n_kv_heads * d_head * dtype_bytes;
    long per_layer  = per_token * T_max;
    long total      = per_layer * n_layers;
    printf("KV cache memory:\n");
    printf("  Per token:  %ld bytes\n", per_token * n_layers);
    printf("  Total (%d tokens): %.1f MB\n", T_max, total / 1048576.0);
}
// Usage: print_kvcache_memory(32, 8, 128, 131072, 2);
// → Per token: 4096 bytes; Total (131072 tokens): 16384.0 MB
```

---

## 7. Sliding Window and Cache Eviction

When the cache is full (pos == T_max), options:

```c
// Option 1: Sliding window — discard oldest tokens, shift cache
void kvcache_slide(KVLayer *kl, int evict_n) {
    int remaining = kl->pos - evict_n;
    int stride = kl->n_kv_heads * kl->d_head;
    memmove(kl->k, kl->k + (long)evict_n * stride,
            remaining * stride * sizeof(float));
    memmove(kl->v, kl->v + (long)evict_n * stride,
            remaining * stride * sizeof(float));
    kl->pos = remaining;
}
// Mistral uses sliding window attention — local window of 4K tokens

// Option 2: Truncate context (simple)
void kvcache_truncate(KVLayer *kl, int new_pos) {
    kl->pos = new_pos < kl->T_max ? new_pos : kl->T_max - 1;
}
```

---

## Key Takeaways

- Without KV cache, generating T tokens costs O(T²) — unusable for long generation
- **KV cache**: store K and V for all past tokens; each new decode step appends one row and does O(T) attention — linear cost per token
- Memory: `2 × n_kv_heads × d_head × n_layers × T_max × dtype_bytes` — Llama 3 8B needs ~16GB for 128K context in FP16
- **GQA (Grouped Query Attention)**: share K/V across multiple Q heads — Llama 3 uses n_kv_heads=8 vs n_heads=32, 4× memory reduction
- Cache reset is required at the start of each new generation sequence

---

**Next**: [27. FFN and Activations](./27_FFN_and_Activations.md) — Implement GELU (GPT-2) and SwiGLU (Llama) feed-forward networks; compare gated vs non-gated architectures.
