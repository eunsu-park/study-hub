/*
 * pooling_demo.c - Max pooling and average pooling with forward/backward
 *
 * Demonstrates:
 *   - Max pooling forward (with argmax) and backward (argmax masking)
 *   - Average pooling forward and backward (uniform gradient distribution)
 *   - Global average pooling (GAP) forward and backward
 *   - Stride and padding effects on output size
 *   - Numerical gradient verification for both pooling types
 *
 * Build:  gcc -std=c11 -Wall -Wextra -O2 -o pooling_demo pooling_demo.c -lm
 * Run:    ./pooling_demo
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#define NCHW(ptr, N, C, H, W, n, c, h, w) \
    ((ptr)[(n)*(C)*(H)*(W) + (c)*(H)*(W) + (h)*(W) + (w)])

/* ---- Max Pooling Forward ---- */
static void max_pool2d_forward(
    const float *input, float *output, int *argmax,
    int N, int C, int H, int W,
    int KH, int KW, int OH, int OW,
    int stride, int pad) {

    for (int n = 0; n < N; n++)
    for (int c = 0; c < C; c++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        float max_val = -FLT_MAX;
        int max_idx = -1;
        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            int ih = oh * stride + kh - pad;
            int iw = ow * stride + kw - pad;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                float val = NCHW(input, N, C, H, W, n, c, ih, iw);
                if (val > max_val) {
                    max_val = val;
                    max_idx = ih * W + iw;
                }
            }
        }
        NCHW(output, N, C, OH, OW, n, c, oh, ow) = max_val;
        NCHW(argmax, N, C, OH, OW, n, c, oh, ow) = max_idx;
    }
}

/* ---- Max Pooling Backward ---- */
static void max_pool2d_backward(
    const float *dY, const int *argmax, float *dX,
    int N, int C, int H, int W, int OH, int OW) {

    memset(dX, 0, (size_t)N * C * H * W * sizeof(float));
    for (int n = 0; n < N; n++)
    for (int c = 0; c < C; c++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        float grad = NCHW(dY, N, C, OH, OW, n, c, oh, ow);
        int idx = NCHW(argmax, N, C, OH, OW, n, c, oh, ow);
        if (idx >= 0) {
            int ih = idx / W;
            int iw = idx % W;
            NCHW(dX, N, C, H, W, n, c, ih, iw) += grad;
        }
    }
}

/* ---- Average Pooling Forward ---- */
static void avg_pool2d_forward(
    const float *input, float *output,
    int N, int C, int H, int W,
    int KH, int KW, int OH, int OW,
    int stride, int pad) {

    for (int n = 0; n < N; n++)
    for (int c = 0; c < C; c++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        float sum = 0.0f;
        int cnt = 0;
        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            int ih = oh * stride + kh - pad;
            int iw = ow * stride + kw - pad;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                sum += NCHW(input, N, C, H, W, n, c, ih, iw);
                cnt++;
            }
        }
        NCHW(output, N, C, OH, OW, n, c, oh, ow) = (cnt > 0) ? sum / cnt : 0.0f;
    }
}

/* ---- Average Pooling Backward ---- */
static void avg_pool2d_backward(
    const float *dY, float *dX,
    int N, int C, int H, int W,
    int KH, int KW, int OH, int OW,
    int stride, int pad) {

    memset(dX, 0, (size_t)N * C * H * W * sizeof(float));
    for (int n = 0; n < N; n++)
    for (int c = 0; c < C; c++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        int cnt = 0;
        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            int ih = oh * stride + kh - pad;
            int iw = ow * stride + kw - pad;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) cnt++;
        }
        float grad_per = NCHW(dY, N, C, OH, OW, n, c, oh, ow) / cnt;
        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            int ih = oh * stride + kh - pad;
            int iw = ow * stride + kw - pad;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                NCHW(dX, N, C, H, W, n, c, ih, iw) += grad_per;
        }
    }
}

/* ---- Global Average Pooling ---- */
static void gap_forward(const float *input, float *output,
                         int N, int C, int H, int W) {
    int spatial = H * W;
    for (int n = 0; n < N; n++)
    for (int c = 0; c < C; c++) {
        float sum = 0.0f;
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++)
            sum += NCHW(input, N, C, H, W, n, c, h, w);
        output[n * C + c] = sum / spatial;
    }
}

static void gap_backward(const float *dOut, float *dX,
                          int N, int C, int H, int W) {
    float inv = 1.0f / (H * W);
    memset(dX, 0, (size_t)N * C * H * W * sizeof(float));
    for (int n = 0; n < N; n++)
    for (int c = 0; c < C; c++) {
        float grad = dOut[n * C + c] * inv;
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++)
            NCHW(dX, N, C, H, W, n, c, h, w) = grad;
    }
}

/* ---- Print helpers ---- */
static void print_mat(const char *name, const float *d, int R, int C_) {
    printf("%s (%dx%d):\n", name, R, C_);
    for (int r = 0; r < R; r++) {
        printf("  ");
        for (int c = 0; c < C_; c++) printf("%7.2f ", d[r * C_ + c]);
        printf("\n");
    }
}

static void print_imat(const char *name, const int *d, int R, int C_) {
    printf("%s (%dx%d):\n", name, R, C_);
    for (int r = 0; r < R; r++) {
        printf("  ");
        for (int c = 0; c < C_; c++) printf("%4d ", d[r * C_ + c]);
        printf("\n");
    }
}

/* ---- Numerical gradient check ---- */
static int numgrad_check_pool(
    const char *name,
    const float *X, int N, int C, int H, int W,
    int KH, int KW, int stride, int pad,
    const float *dY, const float *dX_ana,
    int is_max) {

    int OH = (H + 2 * pad - KH) / stride + 1;
    int OW = (W + 2 * pad - KW) / stride + 1;
    int x_size = N * C * H * W;
    int y_size = N * C * OH * OW;

    float *X_tmp  = (float *)malloc((size_t)x_size * sizeof(float));
    float *Y_p    = (float *)malloc((size_t)y_size * sizeof(float));
    float *Y_m    = (float *)malloc((size_t)y_size * sizeof(float));
    int   *am_tmp = (int *)malloc((size_t)y_size * sizeof(int));

    int errors = 0;
    float max_rel = 0.0f;
    float eps = 1e-4f;

    for (int i = 0; i < x_size; i++) {
        memcpy(X_tmp, X, (size_t)x_size * sizeof(float));
        X_tmp[i] += eps;
        if (is_max)
            max_pool2d_forward(X_tmp, Y_p, am_tmp, N, C, H, W, KH, KW, OH, OW, stride, pad);
        else
            avg_pool2d_forward(X_tmp, Y_p, N, C, H, W, KH, KW, OH, OW, stride, pad);

        memcpy(X_tmp, X, (size_t)x_size * sizeof(float));
        X_tmp[i] -= eps;
        if (is_max)
            max_pool2d_forward(X_tmp, Y_m, am_tmp, N, C, H, W, KH, KW, OH, OW, stride, pad);
        else
            avg_pool2d_forward(X_tmp, Y_m, N, C, H, W, KH, KW, OH, OW, stride, pad);

        float num = 0.0f;
        for (int j = 0; j < y_size; j++)
            num += dY[j] * (Y_p[j] - Y_m[j]) / (2.0f * eps);

        float rel = fabsf(dX_ana[i] - num) / (fabsf(num) + 1e-8f);
        if (rel > max_rel) max_rel = rel;
        if (rel > 1e-2f) errors++;
    }

    printf("  %s backward: max_rel=%.6f  errors=%d/%d  %s\n",
           name, max_rel, errors, x_size, errors == 0 ? "PASS" : "FAIL");

    free(X_tmp); free(Y_p); free(Y_m); free(am_tmp);
    return errors;
}

int main(void) {
    printf("=== Pooling Layers Demo ===\n\n");

    /* ---- Demo 1: Max Pooling ---- */
    printf("--- Max Pooling (2x2, stride=2) ---\n");
    int N = 1, C = 1, H = 4, W = 4;
    float X1[] = {
        3, 1, 4, 2,
        1, 5, 9, 6,
        2, 7, 8, 3,
        0, 4, 6, 1
    };
    int KH = 2, KW = 2, stride = 2, pad = 0;
    int OH = (H - KH) / stride + 1;
    int OW = (W - KW) / stride + 1;

    float Y1[4];
    int argmax1[4];
    max_pool2d_forward(X1, Y1, argmax1, N, C, H, W, KH, KW, OH, OW, stride, pad);

    print_mat("Input", X1, H, W);
    print_mat("MaxPool output", Y1, OH, OW);
    print_imat("Argmax indices", argmax1, OH, OW);

    float dY1[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float dX1[16];
    max_pool2d_backward(dY1, argmax1, dX1, N, C, H, W, OH, OW);
    print_mat("dX (max pool backward)", dX1, H, W);
    printf("\n");

    /* ---- Demo 2: Average Pooling ---- */
    printf("--- Average Pooling (2x2, stride=2) ---\n");
    float Y2[4], dX2[16];
    avg_pool2d_forward(X1, Y2, N, C, H, W, KH, KW, OH, OW, stride, pad);
    print_mat("AvgPool output", Y2, OH, OW);

    avg_pool2d_backward(dY1, dX2, N, C, H, W, KH, KW, OH, OW, stride, pad);
    print_mat("dX (avg pool backward)", dX2, H, W);
    printf("\n");

    /* ---- Demo 3: Average Pooling with padding ---- */
    printf("--- Average Pooling (3x3, stride=1, pad=1) ---\n");
    int KH3 = 3, KW3 = 3, stride3 = 1, pad3 = 1;
    int OH3 = (H + 2 * pad3 - KH3) / stride3 + 1;
    int OW3 = (W + 2 * pad3 - KW3) / stride3 + 1;
    float Y3[16], dX3[16];
    avg_pool2d_forward(X1, Y3, N, C, H, W, KH3, KW3, OH3, OW3, stride3, pad3);
    print_mat("AvgPool output (3x3,s=1,p=1)", Y3, OH3, OW3);
    printf("\n");

    float dY3[16];
    for (int i = 0; i < 16; i++) dY3[i] = 1.0f;
    avg_pool2d_backward(dY3, dX3, N, C, H, W, KH3, KW3, OH3, OW3, stride3, pad3);
    print_mat("dX (avg 3x3 backward)", dX3, H, W);
    printf("\n");

    /* ---- Demo 4: Global Average Pooling ---- */
    printf("--- Global Average Pooling ---\n");
    int N4 = 1, C4 = 3, H4 = 4, W4 = 4;
    float X4[48];
    for (int c = 0; c < C4; c++)
        for (int i = 0; i < H4 * W4; i++)
            X4[c * H4 * W4 + i] = (float)(c * 10 + i);

    float gap_out[3], dOut4[] = {1.0f, 2.0f, 3.0f};
    float dX4[48];
    gap_forward(X4, gap_out, N4, C4, H4, W4);
    printf("GAP input channels 0,1,2 means: ");
    for (int c = 0; c < C4; c++) printf("%.2f  ", gap_out[c]);
    printf("\n");

    gap_backward(dOut4, dX4, N4, C4, H4, W4);
    printf("GAP backward: dX[ch0][0,0]=%.4f  dX[ch2][0,0]=%.4f\n\n",
           dX4[0], dX4[2 * H4 * W4]);

    /* ---- Gradient Checks ---- */
    printf("--- Numerical Gradient Checks ---\n");
    int errs = 0;
    errs += numgrad_check_pool("MaxPool(2x2)", X1, N, C, H, W,
                                KH, KW, stride, pad, dY1, dX1, 1);
    errs += numgrad_check_pool("AvgPool(2x2)", X1, N, C, H, W,
                                KH, KW, stride, pad, dY1, dX2, 0);

    printf("\n=== Result: %s ===\n",
           errs == 0 ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
    return errs != 0;
}
