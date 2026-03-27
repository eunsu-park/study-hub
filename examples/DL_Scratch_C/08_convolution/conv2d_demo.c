/*
 * conv2d_demo.c - 2D Convolution: naive and im2col+GEMM approaches
 *
 * Demonstrates:
 *   - Naive 6-loop 2D convolution with stride, padding, dilation
 *   - im2col transformation that converts convolution to matrix multiply
 *   - Naive GEMM to replace BLAS dependency
 *   - Edge-detection kernel on a synthetic 6x6 single-channel image
 *   - Numerical verification: naive vs im2col outputs match
 *
 * Build:  gcc -std=c11 -Wall -Wextra -O2 -o conv2d_demo conv2d_demo.c -lm
 * Run:    ./conv2d_demo
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

/* NCHW tensor indexing */
#define NCHW(ptr, N, C, H, W, n, c, h, w) \
    ((ptr)[(n)*(C)*(H)*(W) + (c)*(H)*(W) + (h)*(W) + (w)])

/* Compute convolution output size */
static int conv_out_size(int in_size, int kernel, int stride, int pad, int dilation) {
    return (in_size + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
}

/* ---------- Naive 2D Convolution ---------- */
static void conv2d_naive(
    const float *input,  int N, int C_in,  int H,  int W,
    const float *weight, int C_out, int KH, int KW,
    float       *output, int OH, int OW,
    int stride, int pad, int dilation) {

    for (int n  = 0; n  < N;     n++)
    for (int oc = 0; oc < C_out; oc++)
    for (int oh = 0; oh < OH;    oh++)
    for (int ow = 0; ow < OW;    ow++) {
        float sum = 0.0f;
        for (int ic = 0; ic < C_in; ic++)
        for (int kh = 0; kh < KH;   kh++)
        for (int kw = 0; kw < KW;   kw++) {
            int ih = oh * stride + kh * dilation - pad;
            int iw = ow * stride + kw * dilation - pad;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                float x = NCHW(input,  N, C_in, H,  W,  n,  ic, ih, iw);
                float w = NCHW(weight, C_out, C_in, KH, KW, oc, ic, kh, kw);
                sum += x * w;
            }
        }
        NCHW(output, N, C_out, OH, OW, n, oc, oh, ow) = sum;
    }
}

/* ---------- im2col ---------- */
static void im2col(
    const float *input, int N, int C_in, int H, int W,
    int KH, int KW, int OH, int OW,
    int stride, int pad, int dilation,
    float *col) {

    int col_w = C_in * KH * KW;

    for (int n  = 0; n  < N;  n++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        int row = n * OH * OW + oh * OW + ow;
        for (int ic = 0; ic < C_in; ic++)
        for (int kh = 0; kh < KH;   kh++)
        for (int kw = 0; kw < KW;   kw++) {
            int col_idx = ic * KH * KW + kh * KW + kw;
            int ih = oh * stride + kh * dilation - pad;
            int iw = ow * stride + kw * dilation - pad;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                col[row * col_w + col_idx] = NCHW(input, N, C_in, H, W, n, ic, ih, iw);
            else
                col[row * col_w + col_idx] = 0.0f;
        }
    }
}

/* Naive GEMM: C[M,N2] = A[M,K] x B^T[N2,K] */
static void gemm_nt(const float *A, const float *B, float *C_,
                    int M, int N2, int K) {
    for (int i = 0; i < M; i++)
    for (int j = 0; j < N2; j++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += A[i * K + k] * B[j * K + k];
        C_[i * N2 + j] = sum;
    }
}

/* Convolution via im2col + GEMM */
static void conv2d_im2col(
    const float *input,  int N, int C_in, int H, int W,
    const float *weight, int C_out, int KH, int KW,
    float       *output, int OH, int OW,
    int stride, int pad, int dilation) {

    int M = N * OH * OW;
    int K = C_in * KH * KW;

    float *col = (float *)malloc((size_t)M * K * sizeof(float));
    im2col(input, N, C_in, H, W, KH, KW, OH, OW, stride, pad, dilation, col);

    /* output[M, C_out] = col[M, K] x weight^T[C_out, K] */
    gemm_nt(col, weight, output, M, C_out, K);

    free(col);
}

/* ---------- Printing helpers ---------- */
static void print_matrix(const char *name, const float *data, int rows, int cols) {
    printf("%s (%dx%d):\n", name, rows, cols);
    for (int r = 0; r < rows; r++) {
        printf("  ");
        for (int c = 0; c < cols; c++)
            printf("%7.2f ", data[r * cols + c]);
        printf("\n");
    }
}

/* ---------- Demo 1: Edge Detection ---------- */
static void demo_edge_detection(void) {
    printf("=== Demo 1: Edge Detection (Horizontal Sobel) ===\n\n");

    /* 1x1x6x6 input: synthetic gradient image */
    int N = 1, C_in = 1, H = 6, W = 6;
    float input[36];
    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++)
            input[r * W + c] = (float)(r * 10 + c);

    print_matrix("Input (6x6)", input, H, W);

    /* Horizontal Sobel edge-detection kernel (3x3) */
    int C_out = 1, KH = 3, KW = 3;
    float kernel[] = {
        -1, -2, -1,
         0,  0,  0,
         1,  2,  1
    };
    print_matrix("Kernel (Sobel-H)", kernel, KH, KW);

    /* Valid convolution (no padding) */
    int stride = 1, pad = 0, dil = 1;
    int OH = conv_out_size(H, KH, stride, pad, dil);
    int OW = conv_out_size(W, KW, stride, pad, dil);
    printf("Output size: %dx%d (valid, stride=%d)\n\n", OH, OW, stride);

    float out_naive[16], out_im2col[16];
    conv2d_naive(input, N, C_in, H, W, kernel, C_out, KH, KW,
                 out_naive, OH, OW, stride, pad, dil);
    conv2d_im2col(input, N, C_in, H, W, kernel, C_out, KH, KW,
                  out_im2col, OH, OW, stride, pad, dil);

    print_matrix("Output (naive)", out_naive, OH, OW);
    print_matrix("Output (im2col+GEMM)", out_im2col, OH, OW);

    /* Verify outputs match */
    float max_diff = 0.0f;
    for (int i = 0; i < OH * OW; i++) {
        float d = fabsf(out_naive[i] - out_im2col[i]);
        if (d > max_diff) max_diff = d;
    }
    printf("Max |naive - im2col| = %.8f  %s\n\n",
           max_diff, max_diff < 1e-4f ? "PASS" : "FAIL");
}

/* ---------- Demo 2: Multi-channel with padding ---------- */
static void demo_multichannel(void) {
    printf("=== Demo 2: Multi-channel Conv (2 in, 3 out, same-pad) ===\n\n");

    int N = 1, C_in = 2, H = 4, W = 4;
    float input[32];
    for (int i = 0; i < 32; i++)
        input[i] = (float)(i % 7) - 3.0f;

    int C_out = 3, KH = 3, KW = 3;
    float weight[3 * 2 * 3 * 3];
    for (int i = 0; i < 3 * 2 * 9; i++)
        weight[i] = ((float)(i % 5) - 2.0f) * 0.1f;

    int stride = 1, pad = 1, dil = 1;
    int OH = conv_out_size(H, KH, stride, pad, dil);
    int OW = conv_out_size(W, KW, stride, pad, dil);
    printf("Input: [%d,%d,%d,%d]  Weight: [%d,%d,%d,%d]  Output: [%d,%d,%d,%d]\n",
           N, C_in, H, W, C_out, C_in, KH, KW, N, C_out, OH, OW);

    float out_naive[48], out_im2col[48];
    conv2d_naive(input, N, C_in, H, W, weight, C_out, KH, KW,
                 out_naive, OH, OW, stride, pad, dil);
    conv2d_im2col(input, N, C_in, H, W, weight, C_out, KH, KW,
                  out_im2col, OH, OW, stride, pad, dil);

    for (int oc = 0; oc < C_out; oc++) {
        char name[64];
        snprintf(name, sizeof(name), "Output channel %d (naive)", oc);
        print_matrix(name, out_naive + oc * OH * OW, OH, OW);
    }

    float max_diff = 0.0f;
    for (int i = 0; i < C_out * OH * OW; i++) {
        float d = fabsf(out_naive[i] - out_im2col[i]);
        if (d > max_diff) max_diff = d;
    }
    printf("Max |naive - im2col| = %.8f  %s\n\n",
           max_diff, max_diff < 1e-4f ? "PASS" : "FAIL");
}

/* ---------- Demo 3: Stride-2 convolution ---------- */
static void demo_stride2(void) {
    printf("=== Demo 3: Stride-2 Convolution ===\n\n");

    int N = 1, C_in = 1, H = 8, W = 8;
    float input[64];
    for (int i = 0; i < 64; i++)
        input[i] = sinf((float)i * 0.5f);

    int C_out = 1, KH = 3, KW = 3;
    float kernel[] = {1, 1, 1, 1, -8, 1, 1, 1, 1};  /* Laplacian */

    int stride = 2, pad = 1, dil = 1;
    int OH = conv_out_size(H, KH, stride, pad, dil);
    int OW = conv_out_size(W, KW, stride, pad, dil);
    printf("Input: [1,1,8,8]  Stride: 2  Pad: 1  -> Output: [1,1,%d,%d]\n\n", OH, OW);

    float out_naive[16], out_im2col[16];
    conv2d_naive(input, N, C_in, H, W, kernel, C_out, KH, KW,
                 out_naive, OH, OW, stride, pad, dil);
    conv2d_im2col(input, N, C_in, H, W, kernel, C_out, KH, KW,
                  out_im2col, OH, OW, stride, pad, dil);

    print_matrix("Output (Laplacian, stride=2)", out_naive, OH, OW);

    float max_diff = 0.0f;
    for (int i = 0; i < OH * OW; i++) {
        float d = fabsf(out_naive[i] - out_im2col[i]);
        if (d > max_diff) max_diff = d;
    }
    printf("Max |naive - im2col| = %.8f  %s\n\n",
           max_diff, max_diff < 1e-4f ? "PASS" : "FAIL");
}

/* ---------- FLOP Analysis ---------- */
static void demo_flop_analysis(void) {
    printf("=== FLOP Analysis ===\n");
    int C_in = 64, C_out = 64, H = 56, W = 56, K = 3, N2 = 1;
    int OH = H, OW = W;  /* same-padding */
    long long flops = 2LL * N2 * C_out * OH * OW * C_in * K * K;
    long long im2col_bytes = (long long)OH * OW * C_in * K * K * 4;
    printf("Conv 64->64, 3x3, 56x56:  FLOPs = %.1f M\n", flops / 1e6);
    printf("im2col buffer: %.2f MB\n", im2col_bytes / 1e6);
    printf("Arithmetic intensity: %.1f FLOP/byte\n\n",
           (double)flops / im2col_bytes);
}

int main(void) {
    demo_edge_detection();
    demo_multichannel();
    demo_stride2();
    demo_flop_analysis();
    printf("All convolution demos completed.\n");
    return 0;
}
