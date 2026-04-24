# Lesson 26 — KV Cache (per-lesson exercise)

Prerequisites: L25 (attention mechanism), basic C memory management.

Compile: `gcc -std=c11 -Wall -Wextra -O2 -o ex ex.c -lm`

Autoregressive transformers spend most of their inference compute recomputing the same $K$ and $V$ tensors for tokens they already emitted. The **KV cache** stores past $K$ and $V$ across a sequence so each new token only needs to compute its own row. This single optimization typically moves inference from $O(L^2)$ to $O(L)$ per new token.

---

## Exercise 26.1 — Fixed-Capacity KV Cache

**Difficulty**: ★★

### Problem

Implement a struct `KVCache` and three functions:

- `kvcache_init(KVCache *c, int max_len, int head_dim)` — allocate internal `K` and `V` arrays of shape `[max_len, head_dim]`.
- `kvcache_append(KVCache *c, const float *k, const float *v)` — append one token's $K$/$V$ row. Must fail cleanly when `c->len >= c->max_len`.
- `kvcache_free(KVCache *c)` — release memory.

### Starter

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int max_len;
    int head_dim;
    int len;         // current number of cached rows
    float *K;        // [max_len, head_dim], row-major
    float *V;        // [max_len, head_dim]
} KVCache;

int  kvcache_init  (KVCache *c, int max_len, int head_dim);
int  kvcache_append(KVCache *c, const float *k, const float *v);
void kvcache_free  (KVCache *c);

/* Expected usage pattern:

   KVCache c;
   kvcache_init(&c, 2048, 64);
   for (each new token) {
       float k_new[64], v_new[64];
       compute_kv(&k_new, &v_new, ...);
       kvcache_append(&c, k_new, v_new);     // O(1)
       attend_to_cache(&c, query, output);    // O(c.len * head_dim)
   }
   kvcache_free(&c);
*/
```

---

## Exercise 26.2 — Attention Over the Cache

**Difficulty**: ★★★

Given the cache plus a freshly-computed query $q \in \mathbb{R}^d$, compute the attention output $o \in \mathbb{R}^d$:

```c
void attend_to_cache(const KVCache *c, const float *q, float *o);
```

Steps: compute scores `s[i] = (q · K[i]) / sqrt(d)`, softmax them, then compute `o = sum_i softmax(s)[i] * V[i]`.

The key implementation lesson: process ONE row at a time and accumulate the output incrementally. This lets the cache grow beyond L1-cache size without losing performance — memory bandwidth on K,V becomes the limit, not re-reading Q.

---

## Exercise 26.3 — Memory Footprint — Bonus

**Difficulty**: ★★

Compute the total KV-cache bytes per layer for:

- Llama 3 8B: 32 attention heads, head_dim 128, `fp16`, context 8192.
- GPT-4 class hypothetical: 96 heads, head_dim 128, `fp16`, context 128k.

Express both in MiB and GiB. This is the number that drives GPU memory planning for long-context inference — the KV cache often dwarfs the model weights.
