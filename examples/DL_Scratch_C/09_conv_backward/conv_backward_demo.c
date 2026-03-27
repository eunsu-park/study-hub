/*
 * conv_backward_demo.c - Forward and backward pass for 2D convolution
 *
 * Demonstrates:
 *   - Naive forward convolution
 *   - Backward pass: dX (input gradient), dW (weight gradient), db (bias gradient)
 *   - col2im for input gradient scatter
 *   - Numerical gradient checking via finite differences
 *
 * Build:  gcc -std=c11 -Wall -Wextra -O2 -o conv_backward_demo conv_backward_demo.c -lm
 * Run:    ./conv_backward_demo
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NCHW(ptr, N, C, H, W, n, c, h, w) \
    ((ptr)[(n)*(C)*(H)*(W) + (c)*(H)*(W) + (h)*(W) + (w)])

#define EPS 1e-4f

static int conv_out_size(int in_sz, int k, int s, int p, int d) {
    return (in_sz + 2 * p - d * (k - 1) - 1) / s + 1;
}

/* ---- Forward (naive) ---- */
static void conv2d_forward(
    const float *X, int N, int Ci, int H, int W,
    const float *Wt, int Co, int KH, int KW,
    const float *bias,
    float *Y, int OH, int OW,
    int stride, int pad) {

    for (int n = 0; n < N; n++)
    for (int oc = 0; oc < Co; oc++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        float sum = bias ? bias[oc] : 0.0f;
        for (int ic = 0; ic < Ci; ic++)
        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            int ih = oh * stride + kh - pad;
            int iw = ow * stride + kw - pad;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                sum += NCHW(X, N, Ci, H, W, n, ic, ih, iw)
                     * NCHW(Wt, Co, Ci, KH, KW, oc, ic, kh, kw);
        }
        NCHW(Y, N, Co, OH, OW, n, oc, oh, ow) = sum;
    }
}

/* ---- Backward: bias gradient ---- */
static void bias_backward(const float *dY, float *db,
                           int N, int Co, int OH, int OW) {
    memset(db, 0, (size_t)Co * sizeof(float));
    for (int n = 0; n < N; n++)
    for (int oc = 0; oc < Co; oc++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++)
        db[oc] += NCHW(dY, N, Co, OH, OW, n, oc, oh, ow);
}

/* ---- Backward: weight gradient (naive) ---- */
static void weight_backward(
    const float *X, int N, int Ci, int H, int W,
    const float *dY, int Co, int KH, int KW,
    float *dW, int OH, int OW,
    int stride, int pad) {

    memset(dW, 0, (size_t)Co * Ci * KH * KW * sizeof(float));
    for (int n = 0; n < N; n++)
    for (int oc = 0; oc < Co; oc++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        float dy = NCHW(dY, N, Co, OH, OW, n, oc, oh, ow);
        for (int ic = 0; ic < Ci; ic++)
        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            int ih = oh * stride + kh - pad;
            int iw = ow * stride + kw - pad;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                NCHW(dW, Co, Ci, KH, KW, oc, ic, kh, kw)
                    += dy * NCHW(X, N, Ci, H, W, n, ic, ih, iw);
        }
    }
}

/* ---- Backward: input gradient (col2im-style scatter) ---- */
static void input_backward(
    const float *dY, int N, int Co, int OH, int OW,
    const float *Wt, int Ci, int KH, int KW,
    float *dX, int H, int W,
    int stride, int pad) {

    memset(dX, 0, (size_t)N * Ci * H * W * sizeof(float));
    for (int n = 0; n < N; n++)
    for (int oc = 0; oc < Co; oc++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        float dy = NCHW(dY, N, Co, OH, OW, n, oc, oh, ow);
        for (int ic = 0; ic < Ci; ic++)
        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            int ih = oh * stride + kh - pad;
            int iw = ow * stride + kw - pad;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                NCHW(dX, N, Ci, H, W, n, ic, ih, iw)
                    += dy * NCHW(Wt, Co, Ci, KH, KW, oc, ic, kh, kw);
        }
    }
}

/* ---- Compute scalar loss = sum(dY_ref * Y) for gradient checking ---- */
static float compute_loss(const float *Y, const float *dY_ref, int size) {
    float loss = 0.0f;
    for (int i = 0; i < size; i++)
        loss += Y[i] * dY_ref[i];
    return loss;
}

/* ---- Numerical gradient check for a parameter tensor ---- */
static int check_gradient(
    const char *name,
    float *param, int param_size,
    const float *X, int N, int Ci, int H, int W,
    const float *Wt, int Co, int KH, int KW,
    const float *bias,
    const float *dY_ref, int OH, int OW,
    int stride, int pad,
    const float *analytical_grad,
    int is_weight, int is_bias) {

    int out_size = N * Co * OH * OW;
    float *Y_plus  = (float *)malloc((size_t)out_size * sizeof(float));
    float *Y_minus = (float *)malloc((size_t)out_size * sizeof(float));

    int errors = 0;
    float max_rel = 0.0f;
    int check_count = param_size > 50 ? 50 : param_size;

    for (int i = 0; i < check_count; i++) {
        float orig = param[i];

        param[i] = orig + EPS;
        if (is_weight)
            conv2d_forward(X, N, Ci, H, W, param, Co, KH, KW, bias,
                           Y_plus, OH, OW, stride, pad);
        else if (is_bias)
            conv2d_forward(X, N, Ci, H, W, Wt, Co, KH, KW, param,
                           Y_plus, OH, OW, stride, pad);
        else
            conv2d_forward(param, N, Ci, H, W, Wt, Co, KH, KW, bias,
                           Y_plus, OH, OW, stride, pad);

        param[i] = orig - EPS;
        if (is_weight)
            conv2d_forward(X, N, Ci, H, W, param, Co, KH, KW, bias,
                           Y_minus, OH, OW, stride, pad);
        else if (is_bias)
            conv2d_forward(X, N, Ci, H, W, Wt, Co, KH, KW, param,
                           Y_minus, OH, OW, stride, pad);
        else
            conv2d_forward(param, N, Ci, H, W, Wt, Co, KH, KW, bias,
                           Y_minus, OH, OW, stride, pad);

        param[i] = orig;

        float num_grad = (compute_loss(Y_plus, dY_ref, out_size)
                        - compute_loss(Y_minus, dY_ref, out_size)) / (2.0f * EPS);
        float ana_grad = analytical_grad[i];

        float rel = fabsf(ana_grad - num_grad) / (fabsf(num_grad) + 1e-8f);
        if (rel > max_rel) max_rel = rel;
        if (rel > 5e-2f) {
            if (errors < 3)
                printf("  MISMATCH %s[%d]: ana=%.6f num=%.6f rel=%.4f\n",
                       name, i, ana_grad, num_grad, rel);
            errors++;
        }
    }

    printf("  %-6s gradient check: max_rel=%.6f  checked=%d  errors=%d  %s\n",
           name, max_rel, check_count, errors, errors == 0 ? "PASS" : "FAIL");

    free(Y_plus);
    free(Y_minus);
    return errors;
}

/* ---- Main ---- */
int main(void) {
    printf("=== Convolution Backward Demo ===\n\n");

    /* Setup: small conv */
    int N = 1, Ci = 2, H = 5, W = 5;
    int Co = 2, KH = 3, KW = 3;
    int stride = 1, pad = 1;
    int OH = conv_out_size(H, KH, stride, pad, 1);
    int OW = conv_out_size(W, KW, stride, pad, 1);

    printf("Input:  [%d, %d, %d, %d]\n", N, Ci, H, W);
    printf("Weight: [%d, %d, %d, %d]\n", Co, Ci, KH, KW);
    printf("Output: [%d, %d, %d, %d]\n", N, Co, OH, OW);
    printf("Stride=%d, Pad=%d\n\n", stride, pad);

    int x_size = N * Ci * H * W;
    int w_size = Co * Ci * KH * KW;
    int y_size = N * Co * OH * OW;

    float *X    = (float *)malloc((size_t)x_size * sizeof(float));
    float *Wt   = (float *)malloc((size_t)w_size * sizeof(float));
    float *bias  = (float *)malloc((size_t)Co * sizeof(float));
    float *Y    = (float *)malloc((size_t)y_size * sizeof(float));
    float *dY   = (float *)malloc((size_t)y_size * sizeof(float));
    float *dX   = (float *)malloc((size_t)x_size * sizeof(float));
    float *dW   = (float *)malloc((size_t)w_size * sizeof(float));
    float *db   = (float *)malloc((size_t)Co * sizeof(float));

    /* Initialize with small values */
    srand(42);
    for (int i = 0; i < x_size; i++) X[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    for (int i = 0; i < w_size; i++) Wt[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
    for (int i = 0; i < Co; i++) bias[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    for (int i = 0; i < y_size; i++) dY[i] = ((float)rand() / RAND_MAX - 0.5f) * 1.0f;

    /* Forward pass */
    conv2d_forward(X, N, Ci, H, W, Wt, Co, KH, KW, bias, Y, OH, OW, stride, pad);
    printf("Forward pass done. Sample outputs:\n");
    for (int i = 0; i < 5 && i < y_size; i++)
        printf("  Y[%d] = %.6f\n", i, Y[i]);
    printf("\n");

    /* Backward pass */
    printf("--- Analytical Backward ---\n");
    bias_backward(dY, db, N, Co, OH, OW);
    weight_backward(X, N, Ci, H, W, dY, Co, KH, KW, dW, OH, OW, stride, pad);
    input_backward(dY, N, Co, OH, OW, Wt, Ci, KH, KW, dX, H, W, stride, pad);

    printf("db = [%.4f, %.4f]\n", db[0], db[1]);
    printf("dW sample: dW[0]=%.6f  dW[1]=%.6f\n", dW[0], dW[1]);
    printf("dX sample: dX[0]=%.6f  dX[1]=%.6f\n\n", dX[0], dX[1]);

    /* Numerical gradient verification */
    printf("--- Numerical Gradient Verification (eps=%.0e) ---\n", (double)EPS);

    int total_errors = 0;

    /* Check dX */
    total_errors += check_gradient("dX", X, x_size,
        X, N, Ci, H, W, Wt, Co, KH, KW, bias,
        dY, OH, OW, stride, pad, dX, 0, 0);

    /* Check dW */
    total_errors += check_gradient("dW", Wt, w_size,
        X, N, Ci, H, W, Wt, Co, KH, KW, bias,
        dY, OH, OW, stride, pad, dW, 1, 0);

    /* Check db */
    total_errors += check_gradient("db", bias, Co,
        X, N, Ci, H, W, Wt, Co, KH, KW, bias,
        dY, OH, OW, stride, pad, db, 0, 1);

    printf("\n=== Result: %s ===\n",
           total_errors == 0 ? "ALL GRADIENT CHECKS PASSED" : "SOME CHECKS FAILED");
    if (total_errors > 0)
        printf("(Small relative errors < 5%% are expected with float32 finite differences)\n");

    free(X); free(Wt); free(bias); free(Y);
    free(dY); free(dX); free(dW); free(db);
    return 0;
}
