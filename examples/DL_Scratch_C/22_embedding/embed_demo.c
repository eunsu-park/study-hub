/*
 * embed_demo.c - Token embedding lookup table with forward/backward
 *
 * Demonstrates:
 *   - Embedding table: integer token IDs -> dense vectors
 *   - Forward pass: simple row lookup
 *   - Backward pass: scatter-add gradient accumulation
 *   - Weight tying: same table for embedding and unembedding (logits)
 *   - Embedding initialization (N(0, 0.02))
 *
 * Build:  gcc -std=c11 -Wall -Wextra -O2 -o embed_demo embed_demo.c -lm
 * Run:    ./embed_demo
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float randn(void) {
    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    float u2 = (float)rand() / ((float)RAND_MAX + 1.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

/* ---- Embedding forward: lookup rows from table ---- */
static void embedding_forward(const int *tokens, const float *table,
                                float *output, int n_tokens, int d_model) {
    for (int i = 0; i < n_tokens; i++) {
        int id = tokens[i];
        memcpy(output + (long)i * d_model,
               table  + (long)id * d_model,
               (size_t)d_model * sizeof(float));
    }
}

/* ---- Embedding backward: scatter-add gradients to table ---- */
static void embedding_backward(const int *tokens, const float *doutput,
                                 float *dtable, int n_tokens, int d_model) {
    for (int i = 0; i < n_tokens; i++) {
        int id = tokens[i];
        float       *dst = dtable  + (long)id * d_model;
        const float *src = doutput + (long)i  * d_model;
        for (int j = 0; j < d_model; j++)
            dst[j] += src[j];
    }
}

/* ---- Unembedding: project hidden -> logits using the same table ---- */
static void unembed_forward(const float *hidden, const float *table,
                              float *logits, int n_tokens, int d_model, int V) {
    /* logits[i, v] = dot(hidden[i], table[v]) */
    for (int i = 0; i < n_tokens; i++)
    for (int v = 0; v < V; v++) {
        float dot = 0.0f;
        for (int j = 0; j < d_model; j++)
            dot += hidden[i * d_model + j] * table[v * d_model + j];
        logits[i * V + v] = dot;
    }
}

/* ---- Print a vector ---- */
static void print_vec(const char *label, const float *v, int n) {
    printf("  %s: [", label);
    int show = n < 8 ? n : 8;
    for (int i = 0; i < show; i++) {
        printf("%.4f", v[i]);
        if (i < show - 1) printf(", ");
    }
    if (n > 8) printf(", ... (%d more)", n - 8);
    printf("]\n");
}

int main(void) {
    srand(42);

    /* Configuration */
    int V = 20;        /* vocabulary size (small for demo) */
    int d_model = 8;   /* embedding dimension */
    int T = 5;         /* sequence length */

    printf("=== Token Embedding Demo ===\n\n");
    printf("Config: V=%d, d_model=%d, T=%d\n\n", V, d_model, T);

    /* Initialize embedding table: N(0, 0.02) */
    float *table = calloc((size_t)V * d_model, sizeof(float));
    for (int i = 0; i < V * d_model; i++)
        table[i] = randn() * 0.02f;

    printf("=== Embedding Table (first 5 rows) ===\n");
    for (int v = 0; v < 5; v++) {
        char label[32];
        snprintf(label, sizeof(label), "token %2d", v);
        print_vec(label, table + v * d_model, d_model);
    }
    printf("\n");

    /* Token sequence */
    int tokens[] = {3, 7, 1, 7, 15};
    printf("=== Forward Pass ===\n");
    printf("Token IDs: [");
    for (int i = 0; i < T; i++) {
        printf("%d", tokens[i]);
        if (i < T - 1) printf(", ");
    }
    printf("]\n\n");

    /* Forward: lookup */
    float *embeddings = calloc((size_t)T * d_model, sizeof(float));
    embedding_forward(tokens, table, embeddings, T, d_model);

    printf("Embedded vectors:\n");
    for (int i = 0; i < T; i++) {
        char label[32];
        snprintf(label, sizeof(label), "pos %d (tok=%d)", i, tokens[i]);
        print_vec(label, embeddings + i * d_model, d_model);
    }
    printf("\n");

    /* Verify: tokens[1] and tokens[3] are both ID 7 -> same embedding */
    printf("Verification: tokens[1] == tokens[3] == 7\n");
    int match = 1;
    for (int j = 0; j < d_model; j++) {
        if (embeddings[1 * d_model + j] != embeddings[3 * d_model + j]) {
            match = 0;
            break;
        }
    }
    printf("  Embeddings identical: %s\n\n", match ? "YES (correct)" : "NO (bug!)");

    /* ---- Backward pass ---- */
    printf("=== Backward Pass ===\n");

    /* Simulate upstream gradient */
    float *doutput = calloc((size_t)T * d_model, sizeof(float));
    for (int i = 0; i < T * d_model; i++)
        doutput[i] = randn() * 0.1f;

    printf("Upstream gradient (doutput) for each position:\n");
    for (int i = 0; i < T; i++) {
        char label[32];
        snprintf(label, sizeof(label), "dout pos %d", i);
        print_vec(label, doutput + i * d_model, d_model);
    }
    printf("\n");

    /* Backward: scatter-add */
    float *dtable = calloc((size_t)V * d_model, sizeof(float));
    embedding_backward(tokens, doutput, dtable, T, d_model);

    printf("Gradient table (non-zero rows only):\n");
    for (int v = 0; v < V; v++) {
        float norm = 0.0f;
        for (int j = 0; j < d_model; j++)
            norm += dtable[v * d_model + j] * dtable[v * d_model + j];
        if (norm > 1e-12f) {
            char label[32];
            snprintf(label, sizeof(label), "dtable[%2d]", v);
            print_vec(label, dtable + v * d_model, d_model);
        }
    }
    printf("\n");

    /* Token 7 appears at positions 1 and 3 -> gradient is sum of both */
    printf("Token 7 gradient = dout[pos1] + dout[pos3]:\n");
    printf("  dout[1]: [");
    for (int j = 0; j < d_model; j++) printf("%.4f%s", doutput[1 * d_model + j], j < d_model - 1 ? ", " : "");
    printf("]\n  dout[3]: [");
    for (int j = 0; j < d_model; j++) printf("%.4f%s", doutput[3 * d_model + j], j < d_model - 1 ? ", " : "");
    printf("]\n  sum:     [");
    for (int j = 0; j < d_model; j++) printf("%.4f%s",
        doutput[1 * d_model + j] + doutput[3 * d_model + j], j < d_model - 1 ? ", " : "");
    printf("]\n  dtable:  [");
    for (int j = 0; j < d_model; j++) printf("%.4f%s", dtable[7 * d_model + j], j < d_model - 1 ? ", " : "");
    printf("]\n");

    float err = 0.0f;
    for (int j = 0; j < d_model; j++) {
        float expected = doutput[1 * d_model + j] + doutput[3 * d_model + j];
        err += fabsf(dtable[7 * d_model + j] - expected);
    }
    printf("  Match error: %.2e %s\n\n", err, err < 1e-6f ? "(PASS)" : "(FAIL)");

    /* ---- Weight tying: unembed ---- */
    printf("=== Weight Tying (Unembedding) ===\n");
    printf("Using same table as output projection (weight tying).\n\n");

    /* Simulate hidden states (just use the embeddings) */
    float *logits = calloc((size_t)T * V, sizeof(float));
    unembed_forward(embeddings, table, logits, T, d_model, V);

    printf("Logits for each position (top-3 tokens):\n");
    for (int i = 0; i < T; i++) {
        float *row = logits + i * V;
        /* Find top 3 */
        int top[3] = {0, 0, 0};
        for (int iter = 0; iter < 3; iter++) {
            float best = -1e30f;
            for (int v = 0; v < V; v++) {
                int skip = 0;
                for (int k = 0; k < iter; k++)
                    if (top[k] == v) { skip = 1; break; }
                if (!skip && row[v] > best) { best = row[v]; top[iter] = v; }
            }
        }
        printf("  pos %d (input tok=%d): top=[%d(%.3f), %d(%.3f), %d(%.3f)]\n",
               i, tokens[i],
               top[0], row[top[0]], top[1], row[top[1]], top[2], row[top[2]]);
    }
    printf("\n");

    /* Parameter count comparison */
    printf("=== Parameter Count (GPT-2 scale) ===\n");
    printf("  GPT-2 small: V=50257, d_model=768\n");
    printf("  Embedding params:    50257 x 768 = %.1fM\n", 50257.0 * 768 / 1e6);
    printf("  With weight tying:   no extra params for output projection\n");
    printf("  Without tying:       +%.1fM params\n", 50257.0 * 768 / 1e6);

    /* Cleanup */
    free(table); free(embeddings); free(doutput); free(dtable); free(logits);

    printf("\nDone.\n");
    return 0;
}
