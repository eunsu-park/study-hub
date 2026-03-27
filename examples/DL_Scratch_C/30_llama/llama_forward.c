/*
 * llama_forward.c -- Llama-style forward pass demo
 *
 * Demonstrates: RMSNorm, RoPE, Grouped Query Attention (GQA), SwiGLU FFN.
 * Uses 1 block with small dimensions for clarity.
 *
 * Compile: gcc -std=c11 -Wall -Wextra -O2 -o llama_forward llama_forward.c -lm
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- helpers ---- */

static float randf(void) {
    return (float)rand() / (float)RAND_MAX - 0.5f;
}

static void rand_init(float *a, int n) {
    for (int i = 0; i < n; i++) a[i] = randf() * 0.5f;
}

/* ---- RMSNorm ---- */

static void rmsnorm(const float *x, const float *w, float *out, int d) {
    float ss = 0.0f;
    for (int i = 0; i < d; i++) ss += x[i] * x[i];
    float rrms = 1.0f / sqrtf(ss / (float)d + 1e-6f);
    for (int i = 0; i < d; i++) out[i] = x[i] * rrms * w[i];
}

/* ---- RoPE ---- */

static void rope_precompute(float *cos_t, float *sin_t, int T, int half_d,
                            float base) {
    for (int t = 0; t < T; t++) {
        for (int i = 0; i < half_d; i++) {
            float freq = 1.0f / powf(base, (float)(2 * i) / (float)(2 * half_d));
            float angle = (float)t * freq;
            cos_t[t * half_d + i] = cosf(angle);
            sin_t[t * half_d + i] = sinf(angle);
        }
    }
}

static void rope_apply(float *vec, const float *cos_t, const float *sin_t,
                       int half_d) {
    /* vec has dimension 2*half_d; apply rotation to pairs (i, i+half_d) */
    for (int i = 0; i < half_d; i++) {
        float x0 = vec[i];
        float x1 = vec[i + half_d];
        vec[i]          = x0 * cos_t[i] - x1 * sin_t[i];
        vec[i + half_d] = x0 * sin_t[i] + x1 * cos_t[i];
    }
}

/* ---- naive matmul: out[M,N] = A[M,K] x B[N,K]^T ---- */

static void matmul(float *out, const float *A, const float *B,
                   int M, int N, int K) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++)
                s += A[m * K + k] * B[n * K + k];
            out[m * N + n] = s;
        }
}

/* ---- SwiGLU FFN ---- */

static float silu(float x) { return x / (1.0f + expf(-x)); }

static void swiglu_ffn(const float *x, const float *gate_w, const float *up_w,
                       const float *down_w, float *out, int d, int d_ffn) {
    /* gate = x @ gate_w^T, up = x @ up_w^T  (both [d_ffn]) */
    float *gate = (float *)malloc((size_t)d_ffn * sizeof(float));
    float *up   = (float *)malloc((size_t)d_ffn * sizeof(float));

    for (int j = 0; j < d_ffn; j++) {
        float g = 0.0f, u = 0.0f;
        for (int k = 0; k < d; k++) {
            g += x[k] * gate_w[j * d + k];
            u += x[k] * up_w[j * d + k];
        }
        gate[j] = silu(g) * u;
    }
    /* down = gate @ down_w^T  → [d] */
    for (int j = 0; j < d; j++) {
        float s = 0.0f;
        for (int k = 0; k < d_ffn; k++)
            s += gate[k] * down_w[j * d_ffn + k];
        out[j] = s;
    }
    free(gate);
    free(up);
}

/* ---- GQA (Grouped Query Attention) ---- */

static void gqa_forward(const float *X, int T, int d,
                        const float *q_w, const float *k_w, const float *v_w,
                        const float *o_w,
                        int n_heads, int n_kv_heads, int d_head,
                        const float *cos_t, const float *sin_t,
                        float *output) {
    int half = d_head / 2;
    int gqa_factor = n_heads / n_kv_heads;

    /* project Q [T, n_heads*d_head] */
    float *Q = (float *)malloc((size_t)T * n_heads * d_head * sizeof(float));
    matmul(Q, X, q_w, T, n_heads * d_head, d);

    /* project K, V [T, n_kv_heads*d_head] */
    float *K = (float *)malloc((size_t)T * n_kv_heads * d_head * sizeof(float));
    float *V = (float *)malloc((size_t)T * n_kv_heads * d_head * sizeof(float));
    matmul(K, X, k_w, T, n_kv_heads * d_head, d);
    matmul(V, X, v_w, T, n_kv_heads * d_head, d);

    /* apply RoPE to Q and K */
    for (int t = 0; t < T; t++) {
        for (int h = 0; h < n_heads; h++)
            rope_apply(Q + t * n_heads * d_head + h * d_head,
                       cos_t + t * half, sin_t + t * half, half);
        for (int h = 0; h < n_kv_heads; h++)
            rope_apply(K + t * n_kv_heads * d_head + h * d_head,
                       cos_t + t * half, sin_t + t * half, half);
    }

    /* attention per head with GQA grouping (causal) */
    float *head_out = (float *)calloc((size_t)T * n_heads * d_head, sizeof(float));
    float scale = 1.0f / sqrtf((float)d_head);

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / gqa_factor;
        for (int tq = 0; tq < T; tq++) {
            const float *q = Q + tq * n_heads * d_head + h * d_head;
            float *o       = head_out + tq * n_heads * d_head + h * d_head;

            /* compute scores for causal positions [0..tq] */
            int T_att = tq + 1;
            float *sc = (float *)malloc((size_t)T_att * sizeof(float));
            for (int tk = 0; tk < T_att; tk++) {
                const float *k = K + tk * n_kv_heads * d_head + kv_h * d_head;
                float dot = 0.0f;
                for (int j = 0; j < d_head; j++) dot += q[j] * k[j];
                sc[tk] = dot * scale;
            }
            /* softmax */
            float mx = sc[0];
            for (int t = 1; t < T_att; t++) if (sc[t] > mx) mx = sc[t];
            float sum = 0.0f;
            for (int t = 0; t < T_att; t++) { sc[t] = expf(sc[t] - mx); sum += sc[t]; }
            for (int t = 0; t < T_att; t++) sc[t] /= sum;
            /* weighted sum of V */
            for (int t = 0; t < T_att; t++) {
                const float *v = V + t * n_kv_heads * d_head + kv_h * d_head;
                for (int j = 0; j < d_head; j++) o[j] += sc[t] * v[j];
            }
            free(sc);
        }
    }

    /* output projection */
    matmul(output, head_out, o_w, T, d, n_heads * d_head);

    free(Q); free(K); free(V); free(head_out);
}

/* ---- Llama Block ---- */

static void llama_block(const float *X, float *Y, int T, int d, int d_ffn,
                        const float *rn1_w, const float *rn2_w,
                        const float *q_w, const float *k_w,
                        const float *v_w, const float *o_w,
                        int n_heads, int n_kv_heads, int d_head,
                        const float *cos_t, const float *sin_t,
                        const float *gate_w, const float *up_w,
                        const float *down_w) {
    float *rn1 = (float *)malloc((size_t)T * d * sizeof(float));
    float *attn = (float *)malloc((size_t)T * d * sizeof(float));
    float *x1  = (float *)malloc((size_t)T * d * sizeof(float));
    float *rn2 = (float *)malloc((size_t)T * d * sizeof(float));
    float *ffn = (float *)malloc((size_t)T * d * sizeof(float));

    /* 1. RMSNorm + GQA attention */
    for (int t = 0; t < T; t++)
        rmsnorm(X + t * d, rn1_w, rn1 + t * d, d);
    gqa_forward(rn1, T, d, q_w, k_w, v_w, o_w,
                n_heads, n_kv_heads, d_head, cos_t, sin_t, attn);

    /* 2. Residual */
    for (int i = 0; i < T * d; i++) x1[i] = X[i] + attn[i];

    /* 3. RMSNorm + SwiGLU FFN */
    for (int t = 0; t < T; t++) {
        rmsnorm(x1 + t * d, rn2_w, rn2 + t * d, d);
        swiglu_ffn(rn2 + t * d, gate_w, up_w, down_w, ffn + t * d, d, d_ffn);
    }

    /* 4. Residual */
    for (int i = 0; i < T * d; i++) Y[i] = x1[i] + ffn[i];

    free(rn1); free(attn); free(x1); free(rn2); free(ffn);
}

/* ---- main ---- */

int main(void) {
    srand(42);

    /* Small Llama config */
    const int T         = 4;    /* sequence length */
    const int d         = 32;   /* model dim */
    const int n_heads   = 4;
    const int n_kv_heads = 2;   /* GQA: 2 Q heads per KV head */
    const int d_head    = d / n_heads;  /* 8 */
    const int d_ffn     = d * 3;        /* 96 */
    const int V         = 16;   /* vocab */

    printf("=== Llama Forward Pass Demo ===\n");
    printf("T=%d, d=%d, n_heads=%d, n_kv_heads=%d, d_head=%d, d_ffn=%d, V=%d\n\n",
           T, d, n_heads, n_kv_heads, d_head, d_ffn, V);

    /* Allocate and init weights */
    float *rn1_w = (float *)malloc((size_t)d * sizeof(float));
    float *rn2_w = (float *)malloc((size_t)d * sizeof(float));
    for (int i = 0; i < d; i++) { rn1_w[i] = 1.0f; rn2_w[i] = 1.0f; }

    float *q_w = (float *)malloc((size_t)n_heads * d_head * d * sizeof(float));
    float *k_w = (float *)malloc((size_t)n_kv_heads * d_head * d * sizeof(float));
    float *v_w = (float *)malloc((size_t)n_kv_heads * d_head * d * sizeof(float));
    float *o_w = (float *)malloc((size_t)d * n_heads * d_head * sizeof(float));
    rand_init(q_w, n_heads * d_head * d);
    rand_init(k_w, n_kv_heads * d_head * d);
    rand_init(v_w, n_kv_heads * d_head * d);
    rand_init(o_w, d * n_heads * d_head);

    float *gate_w = (float *)malloc((size_t)d_ffn * d * sizeof(float));
    float *up_w   = (float *)malloc((size_t)d_ffn * d * sizeof(float));
    float *down_w = (float *)malloc((size_t)d * d_ffn * sizeof(float));
    rand_init(gate_w, d_ffn * d);
    rand_init(up_w,   d_ffn * d);
    rand_init(down_w, d * d_ffn);

    /* Token embedding (random) */
    float *wte = (float *)malloc((size_t)V * d * sizeof(float));
    rand_init(wte, V * d);

    /* Output head (tied with wte) */
    float *final_rn_w = (float *)malloc((size_t)d * sizeof(float));
    for (int i = 0; i < d; i++) final_rn_w[i] = 1.0f;

    /* RoPE tables */
    int half = d_head / 2;
    float *cos_t = (float *)malloc((size_t)T * half * sizeof(float));
    float *sin_t = (float *)malloc((size_t)T * half * sizeof(float));
    rope_precompute(cos_t, sin_t, T, half, 10000.0f);

    /* Input tokens */
    int tokens[4] = {3, 7, 1, 12};

    /* Embed */
    float *X = (float *)malloc((size_t)T * d * sizeof(float));
    for (int t = 0; t < T; t++)
        memcpy(X + t * d, wte + tokens[t] * d, (size_t)d * sizeof(float));

    /* One Llama block */
    float *Y = (float *)malloc((size_t)T * d * sizeof(float));
    llama_block(X, Y, T, d, d_ffn, rn1_w, rn2_w,
                q_w, k_w, v_w, o_w,
                n_heads, n_kv_heads, d_head,
                cos_t, sin_t, gate_w, up_w, down_w);

    /* Final RMSNorm */
    float *final_out = (float *)malloc((size_t)T * d * sizeof(float));
    for (int t = 0; t < T; t++)
        rmsnorm(Y + t * d, final_rn_w, final_out + t * d, d);

    /* Compute logits = final_out @ wte^T  → [T, V] */
    float *logits = (float *)malloc((size_t)T * V * sizeof(float));
    matmul(logits, final_out, wte, T, V, d);

    /* Print logits for each position */
    printf("Logits (last position, first 8 of %d vocab):\n", V);
    for (int v = 0; v < 8; v++)
        printf("  logit[%d] = %8.4f\n", v, logits[(T - 1) * V + v]);

    /* Softmax and predicted token for last position */
    float *probs = (float *)malloc((size_t)V * sizeof(float));
    float mx = logits[(T - 1) * V];
    for (int v = 1; v < V; v++)
        if (logits[(T - 1) * V + v] > mx) mx = logits[(T - 1) * V + v];
    float sum = 0.0f;
    for (int v = 0; v < V; v++) {
        probs[v] = expf(logits[(T - 1) * V + v] - mx);
        sum += probs[v];
    }
    int best = 0;
    for (int v = 0; v < V; v++) {
        probs[v] /= sum;
        if (probs[v] > probs[best]) best = v;
    }
    printf("\nPredicted next token: %d (prob=%.4f)\n", best, probs[best]);

    printf("\n--- Architecture highlights ---\n");
    printf("  RMSNorm (no mean, no bias) instead of LayerNorm\n");
    printf("  RoPE applied to Q and K (not V)\n");
    printf("  GQA: %d Q heads share %d KV heads (factor=%d)\n",
           n_heads, n_kv_heads, n_heads / n_kv_heads);
    printf("  SwiGLU FFN with gate + up projections\n");
    printf("  No bias terms anywhere\n");

    /* Cleanup */
    free(X); free(Y); free(final_out); free(logits); free(probs);
    free(rn1_w); free(rn2_w); free(final_rn_w);
    free(q_w); free(k_w); free(v_w); free(o_w);
    free(gate_w); free(up_w); free(down_w);
    free(wte); free(cos_t); free(sin_t);

    return 0;
}
