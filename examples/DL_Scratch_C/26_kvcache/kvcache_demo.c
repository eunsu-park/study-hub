/*
 * kvcache_demo.c - KV cache for autoregressive generation
 *
 * Demonstrates:
 *   - KV cache data structure and allocation
 *   - Append-only write pattern during decode
 *   - Attention with KV cache (query single token against full cache)
 *   - Performance comparison: with cache vs without cache
 *   - Memory usage analysis
 *
 * Build:  gcc -std=c11 -Wall -Wextra -O2 -o kvcache_demo kvcache_demo.c -lm
 * Run:    ./kvcache_demo
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static float randn(void) {
    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    float u2 = (float)rand() / ((float)RAND_MAX + 1.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

/* ---- KV Cache for one layer ---- */
typedef struct {
    float *k;        /* [T_max, n_heads, d_head] */
    float *v;        /* [T_max, n_heads, d_head] */
    int   pos;       /* current position (number of cached tokens) */
    int   T_max;
    int   n_heads;
    int   d_head;
} KVCache;

static KVCache *kvcache_create(int T_max, int n_heads, int d_head) {
    KVCache *c = malloc(sizeof(KVCache));
    c->T_max   = T_max;
    c->n_heads = n_heads;
    c->d_head  = d_head;
    c->pos     = 0;
    size_t sz  = (size_t)T_max * n_heads * d_head * sizeof(float);
    c->k = calloc(1, sz);
    c->v = calloc(1, sz);
    return c;
}

static void kvcache_free(KVCache *c) {
    free(c->k); free(c->v); free(c);
}

/* Append K and V for a new token */
static void kvcache_append(KVCache *c, const float *k_new, const float *v_new) {
    int stride = c->n_heads * c->d_head;
    memcpy(c->k + (long)c->pos * stride, k_new, (size_t)stride * sizeof(float));
    memcpy(c->v + (long)c->pos * stride, v_new, (size_t)stride * sizeof(float));
    c->pos++;
}

/* ---- Softmax ---- */
static void softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

/* ---- Attention with KV cache (decode one token) ---- */
static void cached_attention(const float *q_new, KVCache *c,
                               float *out, int n_heads, int d_head) {
    int T = c->pos;
    int stride = n_heads * d_head;
    float scale = 1.0f / sqrtf((float)d_head);

    float *scores = malloc((size_t)T * sizeof(float));

    for (int h = 0; h < n_heads; h++) {
        const float *q = q_new + h * d_head;
        float       *o = out   + h * d_head;

        /* Compute scores: q dot K[t] for all cached tokens */
        for (int t = 0; t < T; t++) {
            const float *k = c->k + (long)t * stride + h * d_head;
            float dot = 0.0f;
            for (int j = 0; j < d_head; j++) dot += q[j] * k[j];
            scores[t] = dot * scale;
        }

        /* Softmax */
        softmax(scores, T);

        /* Weighted sum of V */
        memset(o, 0, (size_t)d_head * sizeof(float));
        for (int t = 0; t < T; t++) {
            const float *v = c->v + (long)t * stride + h * d_head;
            float a = scores[t];
            for (int j = 0; j < d_head; j++) o[j] += a * v[j];
        }
    }
    free(scores);
}

/* ---- Full attention without cache (recompute all) ---- */
static void full_attention(const float *Q_all, const float *K_all,
                             const float *V_all,
                             float *out, int T, int n_heads, int d_head) {
    int stride = n_heads * d_head;
    float scale = 1.0f / sqrtf((float)d_head);
    float *scores = malloc((size_t)T * sizeof(float));

    /* Only compute output for the last token (position T-1) */
    for (int h = 0; h < n_heads; h++) {
        int t = T - 1;
        const float *q = Q_all + (long)t * stride + h * d_head;
        float       *o = out   + h * d_head;

        /* Scores against all positions 0..T-1 (causal: last token sees all) */
        for (int s = 0; s <= t; s++) {
            const float *k = K_all + (long)s * stride + h * d_head;
            float dot = 0.0f;
            for (int j = 0; j < d_head; j++) dot += q[j] * k[j];
            scores[s] = dot * scale;
        }
        softmax(scores, t + 1);

        memset(o, 0, (size_t)d_head * sizeof(float));
        for (int s = 0; s <= t; s++) {
            const float *v = V_all + (long)s * stride + h * d_head;
            float a = scores[s];
            for (int j = 0; j < d_head; j++) o[j] += a * v[j];
        }
    }
    free(scores);
}

/* ---- Linear projection (simple) ---- */
static void project(const float *x, const float *W, float *out,
                      int d_in, int d_out) {
    for (int o = 0; o < d_out; o++) {
        float sum = 0.0f;
        for (int i = 0; i < d_in; i++) sum += x[i] * W[o * d_in + i];
        out[o] = sum;
    }
}

int main(void) {
    srand(42);

    int T_max   = 64;
    int n_heads = 4;
    int d_head  = 8;
    int d_model = n_heads * d_head;  /* 32 */
    int T_seq   = 10;  /* number of tokens to process */

    printf("=== KV Cache Demo ===\n\n");
    printf("Config: T_max=%d, n_heads=%d, d_head=%d, d_model=%d\n", T_max, n_heads, d_head, d_model);
    printf("Sequence length: %d tokens\n\n", T_seq);

    /* Weights for Q, K, V projections */
    float *W_q = malloc((size_t)d_model * d_model * sizeof(float));
    float *W_k = malloc((size_t)d_model * d_model * sizeof(float));
    float *W_v = malloc((size_t)d_model * d_model * sizeof(float));
    for (int i = 0; i < d_model * d_model; i++) {
        W_q[i] = randn() * 0.02f;
        W_k[i] = randn() * 0.02f;
        W_v[i] = randn() * 0.02f;
    }

    /* Generate random token embeddings */
    float *embeddings = malloc((size_t)T_seq * d_model * sizeof(float));
    for (int i = 0; i < T_seq * d_model; i++) embeddings[i] = randn() * 0.5f;

    /* ---- Method 1: With KV cache ---- */
    printf("=== Method 1: With KV Cache (incremental) ===\n\n");

    KVCache *cache = kvcache_create(T_max, n_heads, d_head);
    float *q_new = malloc((size_t)d_model * sizeof(float));
    float *k_new = malloc((size_t)d_model * sizeof(float));
    float *v_new = malloc((size_t)d_model * sizeof(float));
    float *out_cached = malloc((size_t)d_model * sizeof(float));

    clock_t t0 = clock();

    for (int t = 0; t < T_seq; t++) {
        float *x_t = embeddings + t * d_model;

        /* Project current token to Q, K, V */
        project(x_t, W_q, q_new, d_model, d_model);
        project(x_t, W_k, k_new, d_model, d_model);
        project(x_t, W_v, v_new, d_model, d_model);

        /* Append K, V to cache */
        kvcache_append(cache, k_new, v_new);

        /* Attend: q_new against all cached K, V */
        cached_attention(q_new, cache, out_cached, n_heads, d_head);

        printf("  Step %d: cache_pos=%d, out=[%.4f, %.4f, %.4f, ...]\n",
               t, cache->pos, out_cached[0], out_cached[1], out_cached[2]);
    }

    clock_t t1 = clock();
    double cached_time = (double)(t1 - t0) / CLOCKS_PER_SEC * 1000.0;

    /* ---- Method 2: Without cache (recompute from scratch) ---- */
    printf("\n=== Method 2: Without Cache (recompute all) ===\n\n");

    float *Q_all = malloc((size_t)T_seq * d_model * sizeof(float));
    float *K_all = malloc((size_t)T_seq * d_model * sizeof(float));
    float *V_all = malloc((size_t)T_seq * d_model * sizeof(float));
    float *out_nocache = malloc((size_t)d_model * sizeof(float));

    t0 = clock();

    for (int t = 0; t < T_seq; t++) {
        /* Recompute ALL Q, K, V from scratch for positions 0..t */
        for (int s = 0; s <= t; s++) {
            float *x_s = embeddings + s * d_model;
            project(x_s, W_q, Q_all + s * d_model, d_model, d_model);
            project(x_s, W_k, K_all + s * d_model, d_model, d_model);
            project(x_s, W_v, V_all + s * d_model, d_model, d_model);
        }

        /* Full attention for last token */
        full_attention(Q_all, K_all, V_all, out_nocache, t + 1, n_heads, d_head);

        printf("  Step %d: recomputed %d tokens, out=[%.4f, %.4f, %.4f, ...]\n",
               t, t + 1, out_nocache[0], out_nocache[1], out_nocache[2]);
    }

    t1 = clock();
    double nocache_time = (double)(t1 - t0) / CLOCKS_PER_SEC * 1000.0;

    /* ---- Compare outputs ---- */
    printf("\n=== Output Comparison (last step) ===\n");
    float max_diff = 0.0f;
    for (int i = 0; i < d_model; i++) {
        float diff = fabsf(out_cached[i] - out_nocache[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("  Max difference: %.2e %s\n", max_diff,
           max_diff < 1e-5f ? "(MATCH)" : "(MISMATCH)");

    /* ---- Performance comparison ---- */
    printf("\n=== Performance Comparison ===\n");
    printf("  With cache:    %.3f ms\n", cached_time);
    printf("  Without cache: %.3f ms\n", nocache_time);
    if (nocache_time > 0.001)
        printf("  Speedup:       %.1fx\n", nocache_time / cached_time);

    /* ---- Complexity analysis ---- */
    printf("\n=== Complexity Analysis ===\n");
    printf("  Without cache (generating T tokens):\n");
    printf("    QKV projection: O(T^2 * d^2)  (recompute all at each step)\n");
    printf("    Attention:      O(T^3 * d)    (T^2 score matrix at each step)\n");
    printf("  With cache:\n");
    printf("    QKV projection: O(T * d^2)    (only new token)\n");
    printf("    Attention:      O(T^2 * d)    (T scores at each step, T steps)\n");

    /* ---- Memory analysis ---- */
    printf("\n=== KV Cache Memory Analysis ===\n");
    printf("  Per token per layer: 2 * n_heads * d_head * 4 bytes\n");

    struct { const char *name; int L; int nh; int dh; int T; } models[] = {
        {"GPT-2 small",  12, 12,  64,  1024},
        {"Llama 3 8B",   32,  8, 128, 131072},
        {"GPT-4 (est.)", 96, 96, 128,   8192},
    };

    for (int i = 0; i < 3; i++) {
        long per_tok = 2L * models[i].nh * models[i].dh * 4;
        long total = per_tok * models[i].L * models[i].T;
        printf("  %s (L=%d, nh=%d, dh=%d, T=%d):\n",
               models[i].name, models[i].L, models[i].nh, models[i].dh, models[i].T);
        printf("    Per token:  %ld bytes x %d layers = %.1f KB\n",
               per_tok, models[i].L, per_tok * models[i].L / 1024.0);
        printf("    Full cache: %.1f MB\n", total / (1024.0 * 1024.0));
    }

    /* Cleanup */
    kvcache_free(cache);
    free(W_q); free(W_k); free(W_v);
    free(embeddings); free(q_new); free(k_new); free(v_new);
    free(out_cached); free(Q_all); free(K_all); free(V_all); free(out_nocache);

    printf("\nDone.\n");
    return 0;
}
