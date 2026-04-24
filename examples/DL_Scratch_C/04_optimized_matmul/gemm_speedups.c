/*
 * gemm_speedups.c - Three GEMM implementations of increasing sophistication
 *
 * Demonstrates:
 *   - Naive ijk loop (cache-unfriendly inner)
 *   - Loop-reordered ikj (sequential access on B and C)
 *   - Tiled / blocked (each working set fits in L1)
 *
 * Build:  gcc -std=c11 -Wall -Wextra -O3 -march=native -o gemm_speedups gemm_speedups.c -lm
 * Run:    ./gemm_speedups
 *
 * Expected on a modern laptop at N=512:
 *   naive:  ~1-2 GFLOPS
 *   ikj:    ~3-6 GFLOPS  (2-4x speedup)
 *   tiled:  ~5-10 GFLOPS (1.5-2x further speedup)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N      512
#define BLOCK  64

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---- Naive: ijk ordering. Inner loop has stride-N read on B. ---- */
static void gemm_naive(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float acc = 0;
            for (int k = 0; k < n; k++)
                acc += A[i * n + k] * B[k * n + j];
            C[i * n + j] = acc;
        }
}

/* ---- ikj: inner loop is sequential over B[k][j] and C[i][j]. ---- */
static void gemm_ikj(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n * n; i++) C[i] = 0;
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) {
            float a = A[i * n + k];
            for (int j = 0; j < n; j++)
                C[i * n + j] += a * B[k * n + j];
        }
}

/* ---- Tiled: process BLOCK x BLOCK output tiles; each tile fits in L1. ---- */
static void gemm_tiled(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n * n; i++) C[i] = 0;
    for (int ii = 0; ii < n; ii += BLOCK)
        for (int kk = 0; kk < n; kk += BLOCK)
            for (int jj = 0; jj < n; jj += BLOCK)
                for (int i = ii; i < ii + BLOCK; i++)
                    for (int k = kk; k < kk + BLOCK; k++) {
                        float a = A[i * n + k];
                        for (int j = jj; j < jj + BLOCK; j++)
                            C[i * n + j] += a * B[k * n + j];
                    }
}

static void fill_random(float *p, int n) {
    for (int i = 0; i < n; i++) p[i] = (float)rand() / (float)RAND_MAX;
}

static double frob_diff(const float *A, const float *B, int n) {
    double s = 0;
    for (int i = 0; i < n; i++) { double d = A[i] - B[i]; s += d * d; }
    return s;
}

int main(void) {
    srand(0);
    float *A = malloc((size_t)N * N * sizeof(float));
    float *B = malloc((size_t)N * N * sizeof(float));
    float *C_naive = malloc((size_t)N * N * sizeof(float));
    float *C_ikj   = malloc((size_t)N * N * sizeof(float));
    float *C_tiled = malloc((size_t)N * N * sizeof(float));
    fill_random(A, N * N);
    fill_random(B, N * N);

    double flops = 2.0 * N * N * N;     /* GEMM: 2 ops per (i,j,k) */

    double t0 = now_sec(); gemm_naive(A, B, C_naive, N); double t_naive = now_sec() - t0;
    t0 = now_sec();        gemm_ikj  (A, B, C_ikj,   N); double t_ikj   = now_sec() - t0;
    t0 = now_sec();        gemm_tiled(A, B, C_tiled, N); double t_tiled = now_sec() - t0;

    printf("Matrix size: %d x %d\n\n", N, N);
    printf("%-10s %10s %12s %10s\n", "kernel", "time (s)", "GFLOPS", "speedup");
    printf("%-10s %10.3f %12.2f %10.2fx\n", "naive", t_naive, flops / t_naive / 1e9, 1.0);
    printf("%-10s %10.3f %12.2f %10.2fx\n", "ikj",   t_ikj,   flops / t_ikj   / 1e9, t_naive / t_ikj);
    printf("%-10s %10.3f %12.2f %10.2fx\n", "tiled", t_tiled, flops / t_tiled / 1e9, t_naive / t_tiled);

    /* Sanity: all three should produce numerically identical (within rounding) results. */
    printf("\nFrobenius diff (vs naive):\n");
    printf("  ikj   = %.6e\n", frob_diff(C_naive, C_ikj,   N * N));
    printf("  tiled = %.6e\n", frob_diff(C_naive, C_tiled, N * N));
    printf("(Both should be < 1e-3 — minor rounding from different summation order.)\n");

    free(A); free(B); free(C_naive); free(C_ikj); free(C_tiled);
    return 0;
}
