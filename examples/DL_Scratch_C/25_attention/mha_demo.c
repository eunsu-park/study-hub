/*
 * mha_demo.c - Scaled dot-product and multi-head attention
 *
 * Demonstrates:
 *   - Q, K, V linear projections from input
 *   - Scaled dot-product attention with causal mask
 *   - Multi-head splitting and concatenation
 *   - Output projection
 *   - Attention weight visualization
 *
 * Build:  gcc -std=c11 -Wall -Wextra -O2 -o mha_demo mha_demo.c -lm
 * Run:    ./mha_demo
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

/* ---- Linear layer: Y[M, out] = X[M, in] * W[in, out] + b[out] ---- */
static void linear_forward(const float *X, const float *W, const float *b,
                             float *Y, int M, int in_d, int out_d) {
    for (int m = 0; m < M; m++)
    for (int o = 0; o < out_d; o++) {
        float sum = (b != NULL) ? b[o] : 0.0f;
        for (int i = 0; i < in_d; i++)
            sum += X[m * in_d + i] * W[o * in_d + i];  /* W stored as [out, in] */
        Y[m * out_d + o] = sum;
    }
}

/* ---- Softmax over last dimension of a row ---- */
static void softmax_row(float *row, int n) {
    float max_v = row[0];
    for (int i = 1; i < n; i++)
        if (row[i] > max_v) max_v = row[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        row[i] = expf(row[i] - max_v);
        sum += row[i];
    }
    for (int i = 0; i < n; i++) row[i] /= sum;
}

/* ---- Scaled dot-product attention (single head) ---- */
static void attention_forward(const float *Q, const float *K, const float *V,
                                float *attn_w, float *output,
                                int T, int d_head, int causal) {
    float scale = 1.0f / sqrtf((float)d_head);

    /* scores[T, T] = Q[T, d_head] * K^T[d_head, T] * scale */
    for (int t = 0; t < T; t++)
    for (int s = 0; s < T; s++) {
        float dot = 0.0f;
        for (int j = 0; j < d_head; j++)
            dot += Q[t * d_head + j] * K[s * d_head + j];
        attn_w[t * T + s] = dot * scale;
    }

    /* Causal mask: upper triangle -> -inf */
    if (causal) {
        for (int t = 0; t < T; t++)
        for (int s = t + 1; s < T; s++)
            attn_w[t * T + s] = -1e9f;
    }

    /* Softmax per row */
    for (int t = 0; t < T; t++) {
        int lim = causal ? t + 1 : T;
        softmax_row(attn_w + t * T, lim);
        if (causal)
            for (int s = lim; s < T; s++) attn_w[t * T + s] = 0.0f;
    }

    /* output[T, d_head] = attn_w[T, T] * V[T, d_head] */
    for (int t = 0; t < T; t++)
    for (int j = 0; j < d_head; j++) {
        float sum = 0.0f;
        for (int s = 0; s < T; s++)
            sum += attn_w[t * T + s] * V[s * d_head + j];
        output[t * d_head + j] = sum;
    }
}

/* ---- Print matrix ---- */
static void print_mat(const char *label, const float *M_arr, int rows, int cols,
                       int max_cols) {
    printf("  %s [%d x %d]:\n", label, rows, cols);
    int show_c = cols < max_cols ? cols : max_cols;
    for (int r = 0; r < rows; r++) {
        printf("    [");
        for (int c = 0; c < show_c; c++) {
            printf("%7.4f", M_arr[r * cols + c]);
            if (c < show_c - 1) printf(", ");
        }
        if (cols > max_cols) printf(", ...");
        printf("]\n");
    }
}

int main(void) {
    srand(42);

    int T = 5;          /* sequence length */
    int d_model = 16;   /* model dimension */
    int n_heads = 4;    /* number of attention heads */
    int d_head = d_model / n_heads;  /* 4 */

    printf("=== Multi-Head Attention Demo ===\n\n");
    printf("Config: T=%d, d_model=%d, n_heads=%d, d_head=%d\n\n", T, d_model, n_heads, d_head);

    /* Input: [T, d_model] */
    float *X = malloc((size_t)T * d_model * sizeof(float));
    for (int i = 0; i < T * d_model; i++) X[i] = randn() * 0.5f;

    printf("=== Input ===\n");
    print_mat("X", X, T, d_model, 8);
    printf("\n");

    /* QKV weights: W_q, W_k, W_v each [d_model, d_model], stored as [d_model, d_model] */
    float *W_q = malloc((size_t)d_model * d_model * sizeof(float));
    float *W_k = malloc((size_t)d_model * d_model * sizeof(float));
    float *W_v = malloc((size_t)d_model * d_model * sizeof(float));
    float *W_o = malloc((size_t)d_model * d_model * sizeof(float));
    float *b_o = calloc((size_t)d_model, sizeof(float));

    float init_std = 0.02f;
    for (int i = 0; i < d_model * d_model; i++) {
        W_q[i] = randn() * init_std;
        W_k[i] = randn() * init_std;
        W_v[i] = randn() * init_std;
        W_o[i] = randn() * init_std;
    }

    /* Step 1: Project to Q, K, V */
    float *Q = malloc((size_t)T * d_model * sizeof(float));
    float *K = malloc((size_t)T * d_model * sizeof(float));
    float *V = malloc((size_t)T * d_model * sizeof(float));

    linear_forward(X, W_q, NULL, Q, T, d_model, d_model);
    linear_forward(X, W_k, NULL, K, T, d_model, d_model);
    linear_forward(X, W_v, NULL, V, T, d_model, d_model);

    printf("=== QKV Projections ===\n");
    print_mat("Q", Q, T, d_model, 8);
    print_mat("K", K, T, d_model, 8);
    print_mat("V", V, T, d_model, 8);
    printf("\n");

    /* Step 2: Split into heads and run attention per head */
    printf("=== Per-Head Attention (causal) ===\n\n");

    float *head_outputs = malloc((size_t)n_heads * T * d_head * sizeof(float));
    float *all_attn_w   = malloc((size_t)n_heads * T * T * sizeof(float));

    for (int h = 0; h < n_heads; h++) {
        /* Extract head h from Q, K, V */
        float *Qh = malloc((size_t)T * d_head * sizeof(float));
        float *Kh = malloc((size_t)T * d_head * sizeof(float));
        float *Vh = malloc((size_t)T * d_head * sizeof(float));

        for (int t = 0; t < T; t++) {
            memcpy(Qh + t * d_head, Q + t * d_model + h * d_head, (size_t)d_head * sizeof(float));
            memcpy(Kh + t * d_head, K + t * d_model + h * d_head, (size_t)d_head * sizeof(float));
            memcpy(Vh + t * d_head, V + t * d_model + h * d_head, (size_t)d_head * sizeof(float));
        }

        float *attn_wh = all_attn_w + (long)h * T * T;
        float *out_h   = head_outputs + (long)h * T * d_head;

        attention_forward(Qh, Kh, Vh, attn_wh, out_h, T, d_head, 1);

        printf("  --- Head %d ---\n", h);
        printf("  Attention weights (causal masked, after softmax):\n");
        for (int t = 0; t < T; t++) {
            printf("    pos %d: [", t);
            for (int s = 0; s < T; s++) {
                printf("%.3f", attn_wh[t * T + s]);
                if (s < T - 1) printf(", ");
            }
            printf("]\n");
        }
        printf("\n");

        free(Qh); free(Kh); free(Vh);
    }

    /* Step 3: Concatenate heads -> [T, d_model] */
    float *concat = malloc((size_t)T * d_model * sizeof(float));
    for (int t = 0; t < T; t++)
    for (int h = 0; h < n_heads; h++)
    for (int j = 0; j < d_head; j++)
        concat[t * d_model + h * d_head + j] = head_outputs[h * T * d_head + t * d_head + j];

    printf("=== Concatenated Head Output ===\n");
    print_mat("concat", concat, T, d_model, 8);
    printf("\n");

    /* Step 4: Output projection */
    float *output = malloc((size_t)T * d_model * sizeof(float));
    linear_forward(concat, W_o, b_o, output, T, d_model, d_model);

    printf("=== Final MHA Output ===\n");
    print_mat("MHA(X)", output, T, d_model, 8);
    printf("\n");

    /* Verify attention weight properties */
    printf("=== Attention Weight Properties ===\n");
    for (int h = 0; h < n_heads; h++) {
        printf("  Head %d:\n", h);
        float *aw = all_attn_w + (long)h * T * T;
        for (int t = 0; t < T; t++) {
            float row_sum = 0.0f;
            for (int s = 0; s < T; s++) row_sum += aw[t * T + s];
            int causal_ok = 1;
            for (int s = t + 1; s < T; s++)
                if (aw[t * T + s] > 1e-6f) causal_ok = 0;
            printf("    pos %d: sum=%.6f  causal=%s\n",
                   t, row_sum, causal_ok ? "OK" : "FAIL");
        }
    }

    /* Parameter count */
    printf("\n=== Parameter Count ===\n");
    int total_params = 4 * d_model * d_model + d_model;  /* W_q,W_k,W_v,W_o + b_o */
    printf("  W_q, W_k, W_v: 3 x %d x %d = %d\n",
           d_model, d_model, 3 * d_model * d_model);
    printf("  W_o + b_o:      %d x %d + %d = %d\n",
           d_model, d_model, d_model, d_model * d_model + d_model);
    printf("  Total MHA params: %d\n", total_params);
    printf("\n  GPT-2 small: 4 x 768 x 768 + 768 = %.1fM params per layer\n",
           (4.0 * 768 * 768 + 768) / 1e6);

    /* Cleanup */
    free(X); free(W_q); free(W_k); free(W_v); free(W_o); free(b_o);
    free(Q); free(K); free(V);
    free(head_outputs); free(all_attn_w); free(concat); free(output);

    printf("\nDone.\n");
    return 0;
}
