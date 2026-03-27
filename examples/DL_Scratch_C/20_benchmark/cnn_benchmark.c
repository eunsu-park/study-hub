/*
 * cnn_benchmark.c - Benchmark naive vs im2col convolution
 *
 * Demonstrates:
 *   - Naive nested-loop convolution
 *   - im2col + GEMM convolution approach
 *   - Timing both methods using clock()
 *   - Comparison table for varying input sizes
 *   - FLOP counting for convolution layers
 *
 * Build:  gcc -std=c11 -Wall -Wextra -O2 -o cnn_benchmark cnn_benchmark.c -lm
 * Run:    ./cnn_benchmark
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ---- Naive convolution: [C_in, H, W] * [C_out, C_in, K, K] -> [C_out, OH, OW] ---- */
static void conv2d_naive(const float *X, const float *W, float *Y,
                          int C_in, int C_out, int H, int W_in,
                          int K, int stride, int pad) {
    int OH = (H + 2 * pad - K) / stride + 1;
    int OW = (W_in + 2 * pad - K) / stride + 1;

    for (int co = 0; co < C_out; co++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        float sum = 0.0f;
        for (int ci = 0; ci < C_in; ci++)
        for (int kh = 0; kh < K; kh++)
        for (int kw = 0; kw < K; kw++) {
            int ih = oh * stride - pad + kh;
            int iw = ow * stride - pad + kw;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W_in)
                sum += X[ci * H * W_in + ih * W_in + iw]
                     * W[co * C_in * K * K + ci * K * K + kh * K + kw];
        }
        Y[co * OH * OW + oh * OW + ow] = sum;
    }
}

/* ---- im2col: rearrange input patches into columns ---- */
static void im2col(const float *X, float *col,
                    int C_in, int H, int W_in,
                    int K, int stride, int pad) {
    int OH = (H + 2 * pad - K) / stride + 1;
    int OW = (W_in + 2 * pad - K) / stride + 1;
    int col_row = C_in * K * K;

    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        int col_idx = oh * OW + ow;
        for (int ci = 0; ci < C_in; ci++)
        for (int kh = 0; kh < K; kh++)
        for (int kw = 0; kw < K; kw++) {
            int ih = oh * stride - pad + kh;
            int iw = ow * stride - pad + kw;
            int row = ci * K * K + kh * K + kw;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W_in)
                col[row * OH * OW + col_idx] = X[ci * H * W_in + ih * W_in + iw];
            else
                col[row * OH * OW + col_idx] = 0.0f;
        }
    }
    (void)col_row;
}

/* ---- Naive matmul: C[M,N] = A[M,K] x B[K,N] ---- */
static void matmul(const float *A, const float *B, float *C_out,
                    int M, int K, int N) {
    for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += A[m * K + k] * B[k * N + n];
        C_out[m * N + n] = sum;
    }
}

/* ---- im2col convolution: im2col + GEMM ---- */
static void conv2d_im2col(const float *X, const float *W, float *Y,
                            float *col_buf,
                            int C_in, int C_out, int H, int W_in,
                            int K, int stride, int pad) {
    int OH = (H + 2 * pad - K) / stride + 1;
    int OW = (W_in + 2 * pad - K) / stride + 1;

    /* im2col: [C_in*K*K, OH*OW] */
    im2col(X, col_buf, C_in, H, W_in, K, stride, pad);

    /* GEMM: W[C_out, C_in*K*K] x col[C_in*K*K, OH*OW] -> Y[C_out, OH*OW] */
    matmul(W, col_buf, Y, C_out, C_in * K * K, OH * OW);
}

/* ---- FLOP counting ---- */
static long conv_flops(int C_out, int OH, int OW, int C_in, int K) {
    return 2L * C_out * OH * OW * C_in * K * K;
}

/* ---- Benchmark entry ---- */
typedef struct {
    int C_in, C_out, H, W, K, stride, pad;
} ConvConfig;

static void run_benchmark(ConvConfig cfg, int n_iters) {
    int OH = (cfg.H + 2 * cfg.pad - cfg.K) / cfg.stride + 1;
    int OW = (cfg.W + 2 * cfg.pad - cfg.K) / cfg.stride + 1;
    int in_sz  = cfg.C_in * cfg.H * cfg.W;
    int wt_sz  = cfg.C_out * cfg.C_in * cfg.K * cfg.K;
    int out_sz = cfg.C_out * OH * OW;
    int col_sz = cfg.C_in * cfg.K * cfg.K * OH * OW;

    float *X   = malloc((size_t)in_sz * sizeof(float));
    float *W   = malloc((size_t)wt_sz * sizeof(float));
    float *Y1  = malloc((size_t)out_sz * sizeof(float));
    float *Y2  = malloc((size_t)out_sz * sizeof(float));
    float *col = malloc((size_t)col_sz * sizeof(float));

    for (int i = 0; i < in_sz; i++) X[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < wt_sz; i++) W[i] = (float)rand() / RAND_MAX - 0.5f;

    /* Warmup */
    conv2d_naive(X, W, Y1, cfg.C_in, cfg.C_out, cfg.H, cfg.W, cfg.K, cfg.stride, cfg.pad);
    conv2d_im2col(X, W, Y2, col, cfg.C_in, cfg.C_out, cfg.H, cfg.W, cfg.K, cfg.stride, cfg.pad);

    /* Verify outputs match */
    float max_diff = 0.0f;
    for (int i = 0; i < out_sz; i++) {
        float d = fabsf(Y1[i] - Y2[i]);
        if (d > max_diff) max_diff = d;
    }

    /* Time naive */
    clock_t t0 = clock();
    for (int iter = 0; iter < n_iters; iter++)
        conv2d_naive(X, W, Y1, cfg.C_in, cfg.C_out, cfg.H, cfg.W, cfg.K, cfg.stride, cfg.pad);
    clock_t t1 = clock();
    double naive_ms = (double)(t1 - t0) / CLOCKS_PER_SEC * 1000.0 / n_iters;

    /* Time im2col */
    t0 = clock();
    for (int iter = 0; iter < n_iters; iter++)
        conv2d_im2col(X, W, Y2, col, cfg.C_in, cfg.C_out, cfg.H, cfg.W, cfg.K, cfg.stride, cfg.pad);
    t1 = clock();
    double im2col_ms = (double)(t1 - t0) / CLOCKS_PER_SEC * 1000.0 / n_iters;

    long flops = conv_flops(cfg.C_out, OH, OW, cfg.C_in, cfg.K);
    double speedup = naive_ms / (im2col_ms > 0.001 ? im2col_ms : 0.001);

    printf("  %3dx%-3d  Cin=%-3d Cout=%-3d K=%d  | %8.3f ms  %8.3f ms  %5.2fx  | %6.2f MFLOPs  maxdiff=%.1e\n",
           cfg.H, cfg.W, cfg.C_in, cfg.C_out, cfg.K,
           naive_ms, im2col_ms, speedup,
           flops / 1e6, max_diff);

    free(X); free(W); free(Y1); free(Y2); free(col);
}

int main(void) {
    srand(42);

    printf("=== CNN Convolution Benchmark: Naive vs im2col ===\n\n");

    printf("  %-11s %-20s | %-10s %-10s %-7s | %-14s\n",
           "Size", "Config", "Naive", "im2col", "Speedup", "FLOPs");
    printf("  %-11s %-20s | %-10s %-10s %-7s | %-14s\n",
           "----", "------", "-----", "------", "-------", "-----");

    /* Small spatial, few channels (LeNet-style) */
    ConvConfig configs[] = {
        { 1,  6,  16, 16, 3, 1, 1},
        { 3, 16,  32, 32, 3, 1, 1},
        { 3, 16,  32, 32, 5, 1, 2},
        {16, 32,  16, 16, 3, 1, 1},
        {16, 32,  32, 32, 3, 1, 1},
        {32, 64,  16, 16, 3, 1, 1},
        {32, 64,  32, 32, 3, 1, 1},
        {64, 128,  8,  8, 3, 1, 1},
        {64, 128, 16, 16, 3, 1, 1},
    };
    int n_configs = (int)(sizeof(configs) / sizeof(configs[0]));

    for (int i = 0; i < n_configs; i++) {
        int n_iters = 5;
        /* Fewer iterations for large configs */
        long flops = conv_flops(configs[i].C_out,
            (configs[i].H + 2 * configs[i].pad - configs[i].K) / configs[i].stride + 1,
            (configs[i].W + 2 * configs[i].pad - configs[i].K) / configs[i].stride + 1,
            configs[i].C_in, configs[i].K);
        if (flops > 100000000L) n_iters = 2;
        run_benchmark(configs[i], n_iters);
    }

    /* FLOP breakdown comparison */
    printf("\n=== Architecture FLOP Comparison (batch=1, CIFAR-10 32x32) ===\n");
    printf("  %-20s %10s %10s %10s\n", "Architecture", "Params", "FLOPs", "Approx Acc");
    printf("  %-20s %10s %10s %10s\n", "------------", "------", "-----", "----------");
    printf("  %-20s %10s %10s %10s\n", "LeNet-5",      "62K",   "1.0M",  "~68%");
    printf("  %-20s %10s %10s %10s\n", "AlexNet(small)","2.3M", "118M",  "~85%");
    printf("  %-20s %10s %10s %10s\n", "VGG-11(small)", "9.2M", "153M",  "~91%");
    printf("  %-20s %10s %10s %10s\n", "ResNet-20",     "270K", "41M",   "91.25%");
    printf("  %-20s %10s %10s %10s\n", "ResNet-56",     "860K", "127M",  "93.03%");
    printf("  %-20s %10s %10s %10s\n", "EfficientNet-B0","5.3M","390M",  "~93%");

    printf("\n=== Observations ===\n");
    printf("  - im2col converts convolution to a single GEMM call\n");
    printf("  - Naive has better cache behavior for small inputs\n");
    printf("  - im2col advantage grows with larger spatial dimensions\n");
    printf("  - im2col requires extra memory for the column buffer\n");
    printf("  - With optimized BLAS, im2col is typically 5-50x faster\n");

    printf("\nDone.\n");
    return 0;
}
