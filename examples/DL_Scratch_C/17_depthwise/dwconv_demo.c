/*
 * dwconv_demo.c - Depthwise separable convolution
 *
 * Demonstrates:
 *   - Depthwise convolution: each channel filtered independently
 *   - Pointwise convolution (1x1 conv): channel mixing via matmul
 *   - Depthwise separable = depthwise + pointwise
 *   - FLOP and parameter count comparison vs standard convolution
 *   - Forward pass with synthetic data
 *   - MobileNet-style inverted residual block structure
 *
 * Build:  gcc -std=c11 -Wall -Wextra -O2 -o dwconv_demo dwconv_demo.c -lm
 * Run:    ./dwconv_demo
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NCHW(ptr, N, C, H, W, n, c, h, w) \
    ((ptr)[(n)*(C)*(H)*(W) + (c)*(H)*(W) + (h)*(W) + (w)])

static float randn(void) {
    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    float u2 = (float)rand() / ((float)RAND_MAX + 1.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

/* ---- Standard Convolution ---- */
static void conv2d_standard(
    const float *X, int N, int Ci, int H, int W,
    const float *Wt, int Co, int K,
    float *Y, int OH, int OW, int stride, int pad) {

    for (int n = 0; n < N; n++)
    for (int oc = 0; oc < Co; oc++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        float sum = 0.0f;
        for (int ic = 0; ic < Ci; ic++)
        for (int kh = 0; kh < K; kh++)
        for (int kw = 0; kw < K; kw++) {
            int ih = oh * stride + kh - pad;
            int iw = ow * stride + kw - pad;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                sum += NCHW(X, N, Ci, H, W, n, ic, ih, iw)
                     * Wt[oc * Ci * K * K + ic * K * K + kh * K + kw];
        }
        NCHW(Y, N, Co, OH, OW, n, oc, oh, ow) = sum;
    }
}

/* ---- Depthwise Convolution ---- */
static void depthwise_conv2d(
    const float *X, int N, int C, int H, int W,
    const float *Wt, int K,
    float *Y, int OH, int OW, int stride, int pad) {

    for (int n = 0; n < N; n++)
    for (int c = 0; c < C; c++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        float sum = 0.0f;
        for (int kh = 0; kh < K; kh++)
        for (int kw = 0; kw < K; kw++) {
            int ih = oh * stride + kh - pad;
            int iw = ow * stride + kw - pad;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                sum += NCHW(X, N, C, H, W, n, c, ih, iw)
                     * Wt[c * K * K + kh * K + kw];
        }
        NCHW(Y, N, C, OH, OW, n, c, oh, ow) = sum;
    }
}

/* ---- Pointwise Convolution (1x1 conv via naive loops) ---- */
static void pointwise_conv2d(
    const float *X, int N, int Ci, int H, int W,
    const float *Wt, int Co,
    float *Y) {

    for (int n = 0; n < N; n++)
    for (int co = 0; co < Co; co++)
    for (int h = 0; h < H; h++)
    for (int w = 0; w < W; w++) {
        float sum = 0.0f;
        for (int ci = 0; ci < Ci; ci++)
            sum += NCHW(X, N, Ci, H, W, n, ci, h, w)
                 * Wt[co * Ci + ci];
        NCHW(Y, N, Co, H, W, n, co, h, w) = sum;
    }
}

/* ---- Depthwise Separable Convolution ---- */
static void dwsep_conv2d(
    const float *X, int N, int Ci, int H, int W,
    const float *dw_w, int K,
    const float *pw_w, int Co,
    float *dw_out, float *Y,
    int OH, int OW, int stride, int pad) {

    /* Step 1: Depthwise */
    depthwise_conv2d(X, N, Ci, H, W, dw_w, K, dw_out, OH, OW, stride, pad);

    /* Step 2: Pointwise */
    pointwise_conv2d(dw_out, N, Ci, OH, OW, pw_w, Co, Y);
}

/* ---- ReLU6 ---- */
static void relu6(float *X, int size) {
    for (int i = 0; i < size; i++)
        X[i] = fmaxf(0.0f, fminf(6.0f, X[i]));
}

/* ---- Activation statistics ---- */
static void print_stats(const char *name, const float *data, int size) {
    float sum = 0, mn = data[0], mx = data[0];
    for (int i = 0; i < size; i++) {
        sum += data[i];
        if (data[i] < mn) mn = data[i];
        if (data[i] > mx) mx = data[i];
    }
    printf("  %-25s mean=%7.4f  min=%7.4f  max=%7.4f\n",
           name, sum / size, mn, mx);
}

int main(void) {
    srand(42);
    printf("=== Depthwise Separable Convolution Demo ===\n\n");

    int N = 1, C_in = 32, H = 8, W = 8, K = 3, C_out = 64;
    int pad = 1, stride = 1;
    int OH = (H + 2 * pad - K) / stride + 1;
    int OW = (W + 2 * pad - K) / stride + 1;

    /* Allocate input */
    int x_size = N * C_in * H * W;
    float *X = (float *)malloc((size_t)x_size * sizeof(float));
    for (int i = 0; i < x_size; i++) X[i] = randn() * 0.5f;

    /* ---- Demo 1: Standard conv ---- */
    printf("--- Standard Convolution ---\n");
    int std_w_sz = C_out * C_in * K * K;
    float *std_w = (float *)malloc((size_t)std_w_sz * sizeof(float));
    float std_s = sqrtf(2.0f / (C_in * K * K));
    for (int i = 0; i < std_w_sz; i++) std_w[i] = randn() * std_s;

    int out_sz = N * C_out * OH * OW;
    float *std_out = (float *)malloc((size_t)out_sz * sizeof(float));
    conv2d_standard(X, N, C_in, H, W, std_w, C_out, K,
                    std_out, OH, OW, stride, pad);

    printf("Input:  [%d, %d, %d, %d]\n", N, C_in, H, W);
    printf("Weight: [%d, %d, %d, %d]  params=%d\n", C_out, C_in, K, K, std_w_sz);
    printf("Output: [%d, %d, %d, %d]\n", N, C_out, OH, OW);
    print_stats("Standard conv output", std_out, out_sz);

    /* ---- Demo 2: Depthwise separable ---- */
    printf("\n--- Depthwise Separable Convolution ---\n");
    int dw_w_sz = C_in * K * K;
    int pw_w_sz = C_out * C_in;
    float *dw_w = (float *)malloc((size_t)dw_w_sz * sizeof(float));
    float *pw_w = (float *)malloc((size_t)pw_w_sz * sizeof(float));
    float dw_s = sqrtf(2.0f / (K * K));
    for (int i = 0; i < dw_w_sz; i++) dw_w[i] = randn() * dw_s;
    float pw_s = sqrtf(2.0f / C_in);
    for (int i = 0; i < pw_w_sz; i++) pw_w[i] = randn() * pw_s;

    float *dw_out = (float *)malloc((size_t)N * C_in * OH * OW * sizeof(float));
    float *dws_out = (float *)malloc((size_t)out_sz * sizeof(float));
    dwsep_conv2d(X, N, C_in, H, W, dw_w, K, pw_w, C_out,
                 dw_out, dws_out, OH, OW, stride, pad);

    printf("Step 1 - Depthwise:\n");
    printf("  Weight: [%d, 1, %d, %d]  params=%d\n", C_in, K, K, dw_w_sz);
    printf("  Output: [%d, %d, %d, %d]\n", N, C_in, OH, OW);
    print_stats("Depthwise output", dw_out, N * C_in * OH * OW);

    printf("Step 2 - Pointwise:\n");
    printf("  Weight: [%d, %d, 1, 1]  params=%d\n", C_out, C_in, pw_w_sz);
    printf("  Output: [%d, %d, %d, %d]\n", N, C_out, OH, OW);
    print_stats("Pointwise output", dws_out, out_sz);

    /* ---- Parameter comparison ---- */
    printf("\n--- Parameter Count Comparison ---\n");
    int dws_total = dw_w_sz + pw_w_sz;
    printf("Standard conv:  %d params\n", std_w_sz);
    printf("DWS conv:       %d params  (DW=%d + PW=%d)\n",
           dws_total, dw_w_sz, pw_w_sz);
    printf("Reduction:      %.1fx fewer params\n",
           (float)std_w_sz / dws_total);

    /* ---- FLOP comparison ---- */
    printf("\n--- FLOP Comparison ---\n");
    long long std_flops = 2LL * N * C_out * OH * OW * C_in * K * K;
    long long dw_flops  = 2LL * N * C_in * OH * OW * K * K;
    long long pw_flops  = 2LL * N * C_out * OH * OW * C_in;
    long long dws_flops = dw_flops + pw_flops;
    printf("Standard conv:  %lld FLOPs\n", std_flops);
    printf("DWS conv:       %lld FLOPs  (DW=%lld + PW=%lld)\n",
           dws_flops, dw_flops, pw_flops);
    printf("Reduction:      %.1fx fewer FLOPs\n",
           (double)std_flops / dws_flops);
    printf("Theory: 1/C_out + 1/K^2 = 1/%d + 1/%d = %.4f  ->  %.1fx reduction\n",
           C_out, K * K, 1.0f / C_out + 1.0f / (K * K),
           1.0f / (1.0f / C_out + 1.0f / (K * K)));

    /* ---- Demo 3: Stride-2 depthwise (spatial downsampling) ---- */
    printf("\n--- Stride-2 Depthwise Separable ---\n");
    int s2_stride = 2;
    int s2_OH = (H + 2 * pad - K) / s2_stride + 1;
    int s2_OW = (W + 2 * pad - K) / s2_stride + 1;
    float *s2_dw_out = (float *)malloc((size_t)N * C_in * s2_OH * s2_OW * sizeof(float));
    float *s2_out = (float *)malloc((size_t)N * C_out * s2_OH * s2_OW * sizeof(float));
    depthwise_conv2d(X, N, C_in, H, W, dw_w, K, s2_dw_out, s2_OH, s2_OW, s2_stride, pad);
    pointwise_conv2d(s2_dw_out, N, C_in, s2_OH, s2_OW, pw_w, C_out, s2_out);
    printf("Input:  [%d, %d, %d, %d]\n", N, C_in, H, W);
    printf("Output: [%d, %d, %d, %d]  (spatial halved by stride-2 DW)\n",
           N, C_out, s2_OH, s2_OW);

    /* ---- Demo 4: MobileNet inverted residual structure ---- */
    printf("\n--- MobileNetV2 Inverted Residual (conceptual) ---\n");
    int t = 6;  /* expansion factor */
    int C_mid = C_in * t;
    printf("Structure: PW(%d->%d) -> DW(3x3) -> PW(%d->%d)\n",
           C_in, C_mid, C_mid, C_in);
    long long ir_pw1 = 2LL * C_in * C_mid * H * W;
    long long ir_dw  = 2LL * C_mid * H * W * K * K;
    long long ir_pw2 = 2LL * C_mid * C_in * H * W;
    long long ir_total = ir_pw1 + ir_dw + ir_pw2;
    printf("FLOPs: PW1=%lld + DW=%lld + PW2=%lld = %lld\n",
           ir_pw1, ir_dw, ir_pw2, ir_total);
    printf("Params: PW1=%d + DW=%d + PW2=%d = %d\n",
           C_in * C_mid, C_mid * K * K, C_mid * C_in,
           C_in * C_mid + C_mid * K * K + C_mid * C_in);
    printf("With residual connection (stride=1, same channels)\n");

    /* ---- Demo 5: ReLU6 effect ---- */
    printf("\n--- ReLU6 Activation ---\n");
    float test_vals[] = {-2.0f, -0.5f, 0.0f, 1.0f, 3.0f, 5.0f, 7.0f, 10.0f};
    printf("Input:  ");
    for (int i = 0; i < 8; i++) printf("%5.1f ", test_vals[i]);
    printf("\n");
    relu6(test_vals, 8);
    printf("ReLU6:  ");
    for (int i = 0; i < 8; i++) printf("%5.1f ", test_vals[i]);
    printf("\n");

    free(X); free(std_w); free(std_out);
    free(dw_w); free(pw_w); free(dw_out); free(dws_out);
    free(s2_dw_out); free(s2_out);

    printf("\n=== Depthwise Separable Conv Demo Complete ===\n");
    return 0;
}
