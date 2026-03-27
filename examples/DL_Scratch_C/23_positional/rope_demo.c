/*
 * rope_demo.c - Rotary Position Embedding (RoPE)
 *
 * Demonstrates:
 *   - Precomputing cos/sin tables for RoPE frequencies
 *   - Applying RoPE rotation to query and key vectors
 *   - Verifying relative position property: dot(q_m, k_n) depends on (m-n)
 *   - Comparison with sinusoidal positional encoding
 *   - Showing how embeddings change with position
 *
 * Build:  gcc -std=c11 -Wall -Wextra -O2 -o rope_demo rope_demo.c -lm
 * Run:    ./rope_demo
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

/* ---- RoPE: precompute cos/sin tables ---- */
static void rope_precompute(float *cos_tab, float *sin_tab,
                              int T_max, int d_head) {
    int half = d_head / 2;
    for (int t = 0; t < T_max; t++)
    for (int i = 0; i < half; i++) {
        float theta = (float)t / powf(10000.0f, 2.0f * i / d_head);
        cos_tab[t * half + i] = cosf(theta);
        sin_tab[t * half + i] = sinf(theta);
    }
}

/* ---- Apply RoPE to a single vector at a given position ---- */
static void rope_apply_vec(float *vec, const float *cos_tab,
                             const float *sin_tab, int pos, int d_head) {
    int half = d_head / 2;
    const float *c = cos_tab + pos * half;
    const float *s = sin_tab + pos * half;
    for (int i = 0; i < half; i++) {
        float x0 = vec[2 * i];
        float x1 = vec[2 * i + 1];
        vec[2 * i]     = x0 * c[i] - x1 * s[i];
        vec[2 * i + 1] = x0 * s[i] + x1 * c[i];
    }
}

/* ---- Apply RoPE to [n_heads, T, d_head] ---- */
static void rope_apply(float *x, const float *cos_tab,
                         const float *sin_tab,
                         int n_heads, int T, int d_head) {
    for (int h = 0; h < n_heads; h++)
    for (int t = 0; t < T; t++) {
        float *vec = x + ((long)h * T + t) * d_head;
        rope_apply_vec(vec, cos_tab, sin_tab, t, d_head);
    }
}

/* ---- Sinusoidal PE for comparison ---- */
static void sinusoidal_pe(float *pe, int T, int d_model) {
    for (int pos = 0; pos < T; pos++)
    for (int i = 0; i < d_model / 2; i++) {
        float freq = 1.0f / powf(10000.0f, 2.0f * i / d_model);
        pe[pos * d_model + 2 * i]     = sinf(pos * freq);
        pe[pos * d_model + 2 * i + 1] = cosf(pos * freq);
    }
}

/* ---- Dot product ---- */
static float dot(const float *a, const float *b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

/* ---- Print vector ---- */
static void print_vec(const char *label, const float *v, int n) {
    printf("  %s: [", label);
    int show = n < 6 ? n : 6;
    for (int i = 0; i < show; i++) {
        printf("%7.4f", v[i]);
        if (i < show - 1) printf(", ");
    }
    if (n > 6) printf(", ...");
    printf("]\n");
}

int main(void) {
    srand(42);

    int d_head = 8;
    int n_heads = 2;
    int T = 8;
    int half = d_head / 2;

    printf("=== Rotary Position Embedding (RoPE) Demo ===\n\n");
    printf("Config: d_head=%d, n_heads=%d, T=%d\n\n", d_head, n_heads, T);

    /* Precompute tables */
    float *cos_tab = malloc((size_t)T * half * sizeof(float));
    float *sin_tab = malloc((size_t)T * half * sizeof(float));
    rope_precompute(cos_tab, sin_tab, T, d_head);

    printf("=== RoPE Frequency Table ===\n");
    printf("  theta_i = 1 / 10000^(2i/d_head)\n");
    for (int i = 0; i < half; i++) {
        float freq = 1.0f / powf(10000.0f, 2.0f * i / d_head);
        printf("  dim pair (%d,%d): theta = %.6f  period = %.1f positions\n",
               2 * i, 2 * i + 1, freq, 2.0f * (float)M_PI / freq);
    }
    printf("\n");

    /* Show cos/sin values at different positions */
    printf("=== cos/sin at positions 0, 1, 2, 7 ===\n");
    int show_pos[] = {0, 1, 2, 7};
    for (int p = 0; p < 4; p++) {
        int pos = show_pos[p];
        printf("  pos %d: cos=[", pos);
        for (int i = 0; i < half; i++) printf("%.4f%s", cos_tab[pos * half + i], i < half - 1 ? ", " : "");
        printf("]  sin=[");
        for (int i = 0; i < half; i++) printf("%.4f%s", sin_tab[pos * half + i], i < half - 1 ? ", " : "");
        printf("]\n");
    }
    printf("\n");

    /* Demo: apply RoPE to a single query vector */
    printf("=== Applying RoPE to Query Vector ===\n");
    float q_orig[8];
    for (int i = 0; i < d_head; i++) q_orig[i] = randn() * 0.5f;

    printf("Original q:");
    print_vec("", q_orig, d_head);

    for (int pos = 0; pos < 4; pos++) {
        float q_rot[8];
        memcpy(q_rot, q_orig, (size_t)d_head * sizeof(float));
        rope_apply_vec(q_rot, cos_tab, sin_tab, pos, d_head);
        char label[32];
        snprintf(label, sizeof(label), "pos %d", pos);
        print_vec(label, q_rot, d_head);
    }
    printf("\n");

    /* Verify: vector norm is preserved (rotation is orthogonal) */
    printf("=== Norm Preservation (RoPE is orthogonal) ===\n");
    float q_test[8];
    for (int i = 0; i < d_head; i++) q_test[i] = randn();
    float norm_before = sqrtf(dot(q_test, q_test, d_head));
    rope_apply_vec(q_test, cos_tab, sin_tab, 5, d_head);
    float norm_after = sqrtf(dot(q_test, q_test, d_head));
    printf("  Norm before RoPE: %.6f\n", norm_before);
    printf("  Norm after  RoPE: %.6f\n", norm_after);
    printf("  Difference:       %.2e %s\n\n",
           fabsf(norm_before - norm_after),
           fabsf(norm_before - norm_after) < 1e-5f ? "(PASS)" : "(FAIL)");

    /* Key property: dot(RoPE(q,m), RoPE(k,n)) depends on (m-n) */
    printf("=== Relative Position Property ===\n");
    printf("  dot(RoPE(q,m), RoPE(k,n)) should depend only on (m-n)\n\n");

    float q_base[8], k_base[8];
    for (int i = 0; i < d_head; i++) q_base[i] = randn() * 0.5f;
    for (int i = 0; i < d_head; i++) k_base[i] = randn() * 0.5f;

    printf("  %-8s %-8s %-8s  %-12s\n", "m", "n", "m-n", "dot(q_m, k_n)");
    printf("  %-8s %-8s %-8s  %-12s\n", "---", "---", "---", "-------------");

    /* For each relative distance, show dots from different absolute positions */
    for (int delta = 0; delta <= 3; delta++) {
        printf("  --- delta = %d ---\n", delta);
        for (int m = 0; m < 4; m++) {
            int n = m + delta;
            if (n >= T) continue;
            float qm[8], kn[8];
            memcpy(qm, q_base, (size_t)d_head * sizeof(float));
            memcpy(kn, k_base, (size_t)d_head * sizeof(float));
            rope_apply_vec(qm, cos_tab, sin_tab, m, d_head);
            rope_apply_vec(kn, cos_tab, sin_tab, n, d_head);
            float d = dot(qm, kn, d_head);
            printf("  m=%-6d n=%-6d %-8d  %.6f\n", m, n, m - n, d);
        }
    }
    printf("\n  Note: dots with same (m-n) are identical (relative position property).\n\n");

    /* Compare with sinusoidal PE */
    printf("=== Comparison: Sinusoidal PE vs RoPE ===\n\n");
    printf("  Sinusoidal PE (additive):\n");
    float *spe = calloc((size_t)T * d_head, sizeof(float));
    sinusoidal_pe(spe, T, d_head);
    for (int t = 0; t < 4; t++) {
        char label[32];
        snprintf(label, sizeof(label), "pos %d", t);
        print_vec(label, spe + t * d_head, d_head);
    }

    printf("\n  RoPE (multiplicative rotation):\n");
    printf("  - No extra parameters (frequencies are deterministic)\n");
    printf("  - Better length extrapolation\n");
    printf("  - Exact relative position in dot product\n");
    printf("  - Used in: Llama, Falcon, Mistral, GPT-NeoX\n");

    /* Apply RoPE to multi-head Q and K */
    printf("\n=== Multi-Head RoPE Application ===\n");
    float *Q = malloc((size_t)n_heads * T * d_head * sizeof(float));
    float *K = malloc((size_t)n_heads * T * d_head * sizeof(float));
    for (int i = 0; i < n_heads * T * d_head; i++) {
        Q[i] = randn() * 0.5f;
        K[i] = randn() * 0.5f;
    }

    printf("Before RoPE (head 0, pos 0):\n");
    print_vec("Q", Q, d_head);
    print_vec("K", K, d_head);

    rope_apply(Q, cos_tab, sin_tab, n_heads, T, d_head);
    rope_apply(K, cos_tab, sin_tab, n_heads, T, d_head);

    printf("After RoPE (head 0, pos 0):\n");
    print_vec("Q", Q, d_head);
    print_vec("K", K, d_head);
    printf("After RoPE (head 0, pos 1):\n");
    print_vec("Q", Q + d_head, d_head);
    print_vec("K", K + d_head, d_head);

    /* Cleanup */
    free(cos_tab); free(sin_tab); free(spe); free(Q); free(K);

    printf("\nDone.\n");
    return 0;
}
