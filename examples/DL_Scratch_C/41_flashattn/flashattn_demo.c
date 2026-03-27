/*
 * flashattn_demo.c -- Tiled attention (FlashAttention concept) demo
 *
 * Implements both naive attention and FlashAttention-style tiled attention
 * with online softmax. Compares outputs to verify correctness and shows
 * memory usage differences.
 *
 * Compile: gcc -std=c11 -Wall -Wextra -O2 -o flashattn_demo flashattn_demo.c -lm
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- Timer ---- */

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---- Naive attention: materializes full T x T matrix ---- */

static void naive_attention(float *out,
                            const float *Q, const float *K, const float *V,
                            int T, int d) {
    float *S = (float *)malloc((size_t)T * T * sizeof(float));
    float scale = 1.0f / sqrtf((float)d);

    /* S = Q K^T / sqrt(d) */
    for (int i = 0; i < T; i++)
        for (int j = 0; j < T; j++) {
            float dot = 0.0f;
            for (int k = 0; k < d; k++)
                dot += Q[i * d + k] * K[j * d + k];
            S[i * T + j] = dot * scale;
        }

    /* Softmax each row */
    for (int i = 0; i < T; i++) {
        float mx = S[i * T];
        for (int j = 1; j < T; j++) if (S[i * T + j] > mx) mx = S[i * T + j];
        float sum = 0.0f;
        for (int j = 0; j < T; j++) { S[i * T + j] = expf(S[i * T + j] - mx); sum += S[i * T + j]; }
        for (int j = 0; j < T; j++) S[i * T + j] /= sum;
    }

    /* O = S V */
    for (int i = 0; i < T; i++)
        for (int k = 0; k < d; k++) {
            float acc = 0.0f;
            for (int j = 0; j < T; j++)
                acc += S[i * T + j] * V[j * d + k];
            out[i * d + k] = acc;
        }

    free(S);
}

/* ---- FlashAttention: tiled with online softmax ---- */

static void flash_attention(float *out,
                            const float *Q, const float *K, const float *V,
                            int T, int d, int Br, int Bc) {
    float scale = 1.0f / sqrtf((float)d);

    /* Tile working buffers */
    float *O_tile = (float *)malloc((size_t)Br * d * sizeof(float));
    float *m_tile = (float *)malloc((size_t)Br * sizeof(float));
    float *l_tile = (float *)malloc((size_t)Br * sizeof(float));
    float *s_blk  = (float *)malloc((size_t)Br * Bc * sizeof(float));

    /* Iterate over Q tiles */
    for (int q_start = 0; q_start < T; q_start += Br) {
        int q_end = q_start + Br;
        if (q_end > T) q_end = T;
        int cur_Br = q_end - q_start;

        /* Initialize accumulators */
        for (int i = 0; i < cur_Br; i++) {
            m_tile[i] = -1e30f;
            l_tile[i] = 0.0f;
            for (int k = 0; k < d; k++)
                O_tile[i * d + k] = 0.0f;
        }

        /* Iterate over K/V tiles */
        for (int kv_start = 0; kv_start < T; kv_start += Bc) {
            int kv_end = kv_start + Bc;
            if (kv_end > T) kv_end = T;
            int cur_Bc = kv_end - kv_start;

            /* Compute score tile: S = Q_tile @ K_tile^T * scale */
            for (int i = 0; i < cur_Br; i++) {
                int qi = q_start + i;
                for (int j = 0; j < cur_Bc; j++) {
                    int kj = kv_start + j;
                    float dot = 0.0f;
                    for (int dd = 0; dd < d; dd++)
                        dot += Q[qi * d + dd] * K[kj * d + dd];
                    s_blk[i * cur_Bc + j] = dot * scale;
                }
            }

            /* Online softmax update for each Q row */
            for (int i = 0; i < cur_Br; i++) {
                const float *s_row = s_blk + i * cur_Bc;

                /* Find local max */
                float local_max = s_row[0];
                for (int j = 1; j < cur_Bc; j++)
                    if (s_row[j] > local_max) local_max = s_row[j];

                float m_new = fmaxf(m_tile[i], local_max);

                /* Rescale existing accumulator */
                float alpha = expf(m_tile[i] - m_new);
                for (int dd = 0; dd < d; dd++)
                    O_tile[i * d + dd] *= alpha;

                /* Add contribution from this KV block */
                float local_sum = 0.0f;
                for (int j = 0; j < cur_Bc; j++) {
                    float e = expf(s_row[j] - m_new);
                    local_sum += e;
                    for (int dd = 0; dd < d; dd++)
                        O_tile[i * d + dd] += e * V[(kv_start + j) * d + dd];
                }

                /* Update running stats */
                l_tile[i] = alpha * l_tile[i] + local_sum;
                m_tile[i] = m_new;
            }
        }

        /* Normalize and write to output */
        for (int i = 0; i < cur_Br; i++) {
            int qi = q_start + i;
            float inv_l = 1.0f / l_tile[i];
            for (int k = 0; k < d; k++)
                out[qi * d + k] = O_tile[i * d + k] * inv_l;
        }
    }

    free(O_tile); free(m_tile); free(l_tile); free(s_blk);
}

/* ---- Memory usage analysis ---- */

static void compare_memory(int T, int d, int Br, int Bc) {
    long naive_bytes = 2L * T * T * (long)sizeof(float);  /* S and A matrices */
    long flash_bytes = (long)(Br * Bc) * (long)sizeof(float)  /* s_blk */
                     + (long)(Br * d) * (long)sizeof(float)    /* O_tile */
                     + (long)Br * 2 * (long)sizeof(float);     /* m_tile, l_tile */
    long input_bytes = 3L * T * d * (long)sizeof(float);       /* Q, K, V */

    printf("  T=%d, d=%d, Br=%d, Bc=%d:\n", T, d, Br, Bc);
    if (naive_bytes > 1024 * 1024)
        printf("    Naive extra memory:   %.2f MB\n", naive_bytes / (1024.0 * 1024.0));
    else
        printf("    Naive extra memory:   %ld KB\n", naive_bytes / 1024);
    printf("    Flash working set:    %ld KB\n", flash_bytes / 1024);
    printf("    Input (Q,K,V):        %.2f MB\n", input_bytes / (1024.0 * 1024.0));
    printf("    Flash memory savings: %.0fx\n",
           (double)naive_bytes / (double)flash_bytes);
}

/* ---- main ---- */

int main(void) {
    srand(123);

    printf("=== FlashAttention Demo ===\n\n");

    /* Test with small T for correctness verification */
    const int T = 64;
    const int d = 32;
    const int Br = 16;
    const int Bc = 16;

    /* Allocate Q, K, V with random data */
    float *Q = (float *)malloc((size_t)T * d * sizeof(float));
    float *K = (float *)malloc((size_t)T * d * sizeof(float));
    float *V = (float *)malloc((size_t)T * d * sizeof(float));
    float *out_naive = (float *)malloc((size_t)T * d * sizeof(float));
    float *out_flash = (float *)malloc((size_t)T * d * sizeof(float));

    for (int i = 0; i < T * d; i++) {
        Q[i] = (float)rand() / (float)RAND_MAX - 0.5f;
        K[i] = (float)rand() / (float)RAND_MAX - 0.5f;
        V[i] = (float)rand() / (float)RAND_MAX - 0.5f;
    }

    printf("--- Part 1: Correctness Verification ---\n");
    printf("  T=%d, d=%d, Br=%d, Bc=%d\n\n", T, d, Br, Bc);

    naive_attention(out_naive, Q, K, V, T, d);
    flash_attention(out_flash, Q, K, V, T, d, Br, Bc);

    /* Compare outputs */
    float max_diff = 0.0f;
    double sum_diff = 0.0;
    for (int i = 0; i < T * d; i++) {
        float diff = fabsf(out_naive[i] - out_flash[i]);
        if (diff > max_diff) max_diff = diff;
        sum_diff += diff;
    }
    float mean_diff = (float)(sum_diff / (T * d));

    printf("  Max absolute difference:  %.2e\n", max_diff);
    printf("  Mean absolute difference: %.2e\n", mean_diff);
    printf("  %s (threshold: 1e-5)\n\n",
           max_diff < 1e-5f ? "PASSED" : "FAILED");

    /* Show a few output values for comparison */
    printf("  Sample outputs (first 4 positions, first 4 dims):\n");
    printf("  %-6s  %-12s  %-12s  %-12s\n", "Pos", "Naive", "Flash", "Diff");
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            printf("  [%d,%d]  %+10.6f  %+10.6f  %+.2e\n",
                   i, j, out_naive[i * d + j], out_flash[i * d + j],
                   out_naive[i * d + j] - out_flash[i * d + j]);

    /* --- Part 2: Timing comparison --- */
    printf("\n--- Part 2: Timing Comparison ---\n");

    int reps = 50;
    double t0 = now_sec();
    for (int r = 0; r < reps; r++)
        naive_attention(out_naive, Q, K, V, T, d);
    double t_naive = (now_sec() - t0) / reps;

    t0 = now_sec();
    for (int r = 0; r < reps; r++)
        flash_attention(out_flash, Q, K, V, T, d, Br, Bc);
    double t_flash = (now_sec() - t0) / reps;

    printf("  Naive attention:  %.3f ms\n", t_naive * 1000.0);
    printf("  Flash attention:  %.3f ms\n", t_flash * 1000.0);
    printf("  Ratio:            %.2fx\n\n", t_naive / t_flash);

    /* --- Part 3: Memory usage analysis --- */
    printf("--- Part 3: Memory Usage Analysis ---\n\n");
    compare_memory(64, 32, 16, 16);
    printf("\n");
    compare_memory(512, 64, 32, 32);
    printf("\n");
    compare_memory(2048, 128, 64, 64);
    printf("\n");
    compare_memory(8192, 128, 64, 64);

    /* --- Part 4: Online softmax explanation --- */
    printf("\n--- Part 4: Online Softmax Algorithm ---\n\n");
    printf("  Standard softmax requires all scores in memory:\n");
    printf("    1. Compute all scores S[T, T]\n");
    printf("    2. Find row max\n");
    printf("    3. Subtract max, exponentiate, normalize\n\n");
    printf("  Online softmax processes blocks incrementally:\n");
    printf("    For each KV block:\n");
    printf("      1. Compute local scores (Br x Bc tile)\n");
    printf("      2. Find local max\n");
    printf("      3. Update running max: m_new = max(m_old, local_max)\n");
    printf("      4. Rescale old accumulator: O *= exp(m_old - m_new)\n");
    printf("      5. Add new contribution: O += exp(s - m_new) * V\n");
    printf("      6. Update running sum: l = alpha * l + local_sum\n");
    printf("    Final: O /= l\n\n");
    printf("  This is EXACT (not approximate).\n");
    printf("  Peak memory: O(Br*Bc) instead of O(T^2)\n");

    free(Q); free(K); free(V);
    free(out_naive); free(out_flash);

    return 0;
}
