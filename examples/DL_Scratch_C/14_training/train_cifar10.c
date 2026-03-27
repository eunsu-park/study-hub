/*
 * train_cifar10.c - Minimal CNN training loop on synthetic data
 *
 * Demonstrates:
 *   - Small CNN: Conv->ReLU->Pool -> Conv->ReLU->Pool -> FC
 *   - Cross-entropy loss with numerically stable softmax
 *   - Full backward pass: dLogits->dFC->dPool->dConv
 *   - SGD weight update with learning rate
 *   - Training loss decreasing over iterations
 *
 * Build:  gcc -std=c11 -Wall -Wextra -O2 -o train_cifar10 train_cifar10.c -lm
 * Run:    ./train_cifar10
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define NCHW(ptr, N, C, H, W, n, c, h, w) \
    ((ptr)[(n)*(C)*(H)*(W) + (c)*(H)*(W) + (h)*(W) + (w)])

/* ---- Random normal ---- */
static float randn(void) {
    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    float u2 = (float)rand() / ((float)RAND_MAX + 1.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

/* ---- Conv2d forward (naive, with bias) ---- */
static void conv2d_fwd(
    const float *X, int N, int Ci, int H, int W,
    const float *Wt, int Co, int KH, int KW,
    const float *bias, float *Y, int OH, int OW,
    int stride, int pad) {

    for (int n = 0; n < N; n++)
    for (int oc = 0; oc < Co; oc++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        float sum = bias[oc];
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

/* ---- Conv2d backward (naive) ---- */
static void conv2d_bwd(
    const float *X, int N, int Ci, int H, int W,
    const float *Wt, int Co, int KH, int KW,
    const float *dY, int OH, int OW,
    float *dX, float *dW, float *db,
    int stride, int pad) {

    /* dX */
    if (dX) {
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

    /* dW */
    for (int n = 0; n < N; n++)
    for (int oc = 0; oc < Co; oc++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        float dy = NCHW(dY, N, Co, OH, OW, n, oc, oh, ow);
        db[oc] += dy;
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

/* ---- ReLU forward (saves mask) ---- */
static void relu_fwd(float *X, uint8_t *mask, int size) {
    for (int i = 0; i < size; i++) {
        mask[i] = X[i] > 0.0f ? 1 : 0;
        if (X[i] < 0.0f) X[i] = 0.0f;
    }
}

static void relu_bwd(float *dX, const uint8_t *mask, int size) {
    for (int i = 0; i < size; i++)
        dX[i] *= mask[i];
}

/* ---- Max Pool 2x2, stride 2 ---- */
static void maxpool_fwd(const float *X, float *Y, int *argmax,
                         int N, int C, int H, int W, int OH, int OW) {
    for (int n = 0; n < N; n++)
    for (int c = 0; c < C; c++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        float mx = -1e30f;
        int mi = -1;
        for (int kh = 0; kh < 2; kh++)
        for (int kw = 0; kw < 2; kw++) {
            int ih = oh * 2 + kh, iw = ow * 2 + kw;
            float v = NCHW(X, N, C, H, W, n, c, ih, iw);
            if (v > mx) { mx = v; mi = ih * W + iw; }
        }
        NCHW(Y, N, C, OH, OW, n, c, oh, ow) = mx;
        NCHW(argmax, N, C, OH, OW, n, c, oh, ow) = mi;
    }
}

static void maxpool_bwd(const float *dY, const int *argmax, float *dX,
                         int N, int C, int H, int W, int OH, int OW) {
    memset(dX, 0, (size_t)N * C * H * W * sizeof(float));
    for (int n = 0; n < N; n++)
    for (int c = 0; c < C; c++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        int idx = NCHW(argmax, N, C, OH, OW, n, c, oh, ow);
        int ih = idx / W, iw = idx % W;
        NCHW(dX, N, C, H, W, n, c, ih, iw) += NCHW(dY, N, C, OH, OW, n, c, oh, ow);
    }
}

/* ---- FC forward ---- */
static void fc_fwd(const float *X, const float *W, const float *b,
                    float *Y, int N, int in_d, int out_d) {
    for (int n = 0; n < N; n++)
    for (int o = 0; o < out_d; o++) {
        float s = b[o];
        for (int i = 0; i < in_d; i++)
            s += X[n * in_d + i] * W[i * out_d + o];
        Y[n * out_d + o] = s;
    }
}

static void fc_bwd(const float *X, const float *W, const float *dY,
                    float *dX, float *dW, float *db,
                    int N, int in_d, int out_d) {
    /* dX */
    if (dX) {
        memset(dX, 0, (size_t)N * in_d * sizeof(float));
        for (int n = 0; n < N; n++)
        for (int o = 0; o < out_d; o++)
        for (int i = 0; i < in_d; i++)
            dX[n * in_d + i] += dY[n * out_d + o] * W[i * out_d + o];
    }
    /* dW, db */
    for (int n = 0; n < N; n++)
    for (int o = 0; o < out_d; o++) {
        db[o] += dY[n * out_d + o];
        for (int i = 0; i < in_d; i++)
            dW[i * out_d + o] += X[n * in_d + i] * dY[n * out_d + o];
    }
}

/* ---- Softmax cross-entropy ---- */
static float softmax_ce_fwd(const float *logits, const uint8_t *labels,
                              float *probs, int N, int C) {
    float loss = 0.0f;
    for (int n = 0; n < N; n++) {
        const float *row = logits + n * C;
        float *p = probs + n * C;
        float mx = row[0];
        for (int c = 1; c < C; c++) if (row[c] > mx) mx = row[c];
        float sum = 0.0f;
        for (int c = 0; c < C; c++) { p[c] = expf(row[c] - mx); sum += p[c]; }
        for (int c = 0; c < C; c++) p[c] /= sum;
        loss += -logf(p[labels[n]] + 1e-9f);
    }
    return loss / N;
}

static void softmax_ce_bwd(const float *probs, const uint8_t *labels,
                             float *dlogits, int N, int C) {
    memcpy(dlogits, probs, (size_t)N * C * sizeof(float));
    for (int n = 0; n < N; n++)
        dlogits[n * C + labels[n]] -= 1.0f;
    float inv = 1.0f / N;
    for (int i = 0; i < N * C; i++) dlogits[i] *= inv;
}

/* ---- SGD update ---- */
static void sgd_update(float *param, float *grad, int size, float lr) {
    for (int i = 0; i < size; i++) {
        param[i] -= lr * grad[i];
        grad[i] = 0.0f;  /* zero grad */
    }
}

/* ---- Network definition ----
 * Input: [N, 3, 8, 8]  (small for speed)
 * Conv1: 3->8, 3x3, p=1 -> [N,8,8,8] -> ReLU -> MaxPool -> [N,8,4,4]
 * Conv2: 8->16, 3x3, p=1 -> [N,16,4,4] -> ReLU -> MaxPool -> [N,16,2,2]
 * FC: 16*2*2=64 -> 10
 */
#define BS    16
#define IN_C   3
#define IN_H   8
#define IN_W   8
#define C1    8
#define C2   16
#define FC_IN (C2*2*2)
#define NCLS  10

int main(void) {
    srand(42);
    printf("=== Minimal CNN Training Demo ===\n\n");

    /* Allocate weights */
    int w1_sz = C1*IN_C*3*3, w2_sz = C2*C1*3*3, wfc_sz = FC_IN*NCLS;
    float *w1 = (float*)malloc((size_t)w1_sz*sizeof(float));
    float *b1 = (float*)calloc(C1, sizeof(float));
    float *w2 = (float*)malloc((size_t)w2_sz*sizeof(float));
    float *b2 = (float*)calloc(C2, sizeof(float));
    float *wfc = (float*)malloc((size_t)wfc_sz*sizeof(float));
    float *bfc = (float*)calloc(NCLS, sizeof(float));

    /* He init */
    float std1 = sqrtf(2.0f / (IN_C*9));
    for (int i = 0; i < w1_sz; i++) w1[i] = randn() * std1;
    float std2 = sqrtf(2.0f / (C1*9));
    for (int i = 0; i < w2_sz; i++) w2[i] = randn() * std2;
    float stdfc = sqrtf(2.0f / FC_IN);
    for (int i = 0; i < wfc_sz; i++) wfc[i] = randn() * stdfc;

    /* Gradient buffers */
    float *dw1 = (float*)calloc(w1_sz, sizeof(float));
    float *db1 = (float*)calloc(C1, sizeof(float));
    float *dw2 = (float*)calloc(w2_sz, sizeof(float));
    float *db2 = (float*)calloc(C2, sizeof(float));
    float *dwfc = (float*)calloc(wfc_sz, sizeof(float));
    float *dbfc = (float*)calloc(NCLS, sizeof(float));

    /* Activation buffers */
    float *X     = (float*)malloc((size_t)BS*IN_C*IN_H*IN_W*sizeof(float));
    uint8_t *y   = (uint8_t*)malloc(BS);
    float *a1    = (float*)malloc((size_t)BS*C1*IN_H*IN_W*sizeof(float));
    uint8_t *m1  = (uint8_t*)malloc((size_t)BS*C1*IN_H*IN_W);
    float *p1    = (float*)malloc((size_t)BS*C1*4*4*sizeof(float));
    int   *am1   = (int*)malloc((size_t)BS*C1*4*4*sizeof(int));
    float *a2    = (float*)malloc((size_t)BS*C2*4*4*sizeof(float));
    uint8_t *m2  = (uint8_t*)malloc((size_t)BS*C2*4*4);
    float *p2    = (float*)malloc((size_t)BS*C2*2*2*sizeof(float));
    int   *am2   = (int*)malloc((size_t)BS*C2*2*2*sizeof(int));
    float *logits= (float*)malloc((size_t)BS*NCLS*sizeof(float));
    float *probs = (float*)malloc((size_t)BS*NCLS*sizeof(float));
    float *dlog  = (float*)malloc((size_t)BS*NCLS*sizeof(float));
    float *dp2   = (float*)malloc((size_t)BS*C2*2*2*sizeof(float));
    float *da2   = (float*)malloc((size_t)BS*C2*4*4*sizeof(float));
    float *dp1   = (float*)malloc((size_t)BS*C1*4*4*sizeof(float));
    float *da1   = (float*)malloc((size_t)BS*C1*IN_H*IN_W*sizeof(float));

    printf("Network: Conv(3->%d,3x3)->ReLU->Pool -> Conv(%d->%d,3x3)->ReLU->Pool -> FC(%d->%d)\n",
           C1, C1, C2, FC_IN, NCLS);
    printf("Input: [%d, %d, %d, %d]  Classes: %d\n\n", BS, IN_C, IN_H, IN_W, NCLS);

    float lr = 0.01f;
    int n_iters = 50;

    printf("Training for %d iterations (lr=%.3f):\n", n_iters, lr);
    printf("%-6s %-12s %-10s\n", "Iter", "Loss", "Accuracy");

    for (int iter = 0; iter < n_iters; iter++) {
        /* Generate random batch */
        for (int i = 0; i < BS * IN_C * IN_H * IN_W; i++)
            X[i] = randn() * 0.5f;
        for (int i = 0; i < BS; i++) y[i] = (uint8_t)(rand() % NCLS);

        /* Forward */
        conv2d_fwd(X, BS, IN_C, IN_H, IN_W, w1, C1, 3, 3, b1, a1, IN_H, IN_W, 1, 1);
        relu_fwd(a1, m1, BS*C1*IN_H*IN_W);
        maxpool_fwd(a1, p1, am1, BS, C1, IN_H, IN_W, 4, 4);

        conv2d_fwd(p1, BS, C1, 4, 4, w2, C2, 3, 3, b2, a2, 4, 4, 1, 1);
        relu_fwd(a2, m2, BS*C2*4*4);
        maxpool_fwd(a2, p2, am2, BS, C2, 4, 4, 2, 2);

        fc_fwd(p2, wfc, bfc, logits, BS, FC_IN, NCLS);

        float loss = softmax_ce_fwd(logits, y, probs, BS, NCLS);

        /* Accuracy */
        int correct = 0;
        for (int n = 0; n < BS; n++) {
            int pred = 0;
            for (int c = 1; c < NCLS; c++)
                if (probs[n*NCLS+c] > probs[n*NCLS+pred]) pred = c;
            if (pred == y[n]) correct++;
        }

        if (iter % 5 == 0 || iter == n_iters - 1)
            printf("%-6d %-12.4f %.1f%%\n", iter, loss, 100.0f * correct / BS);

        /* Backward */
        softmax_ce_bwd(probs, y, dlog, BS, NCLS);

        fc_bwd(p2, wfc, dlog, dp2, dwfc, dbfc, BS, FC_IN, NCLS);

        maxpool_bwd(dp2, am2, da2, BS, C2, 4, 4, 2, 2);
        relu_bwd(da2, m2, BS*C2*4*4);
        conv2d_bwd(p1, BS, C1, 4, 4, w2, C2, 3, 3, da2, 4, 4,
                   dp1, dw2, db2, 1, 1);

        maxpool_bwd(dp1, am1, da1, BS, C1, IN_H, IN_W, 4, 4);
        relu_bwd(da1, m1, BS*C1*IN_H*IN_W);
        conv2d_bwd(X, BS, IN_C, IN_H, IN_W, w1, C1, 3, 3, da1, IN_H, IN_W,
                   NULL, dw1, db1, 1, 1);

        /* SGD update */
        sgd_update(w1, dw1, w1_sz, lr);
        sgd_update(b1, db1, C1, lr);
        sgd_update(w2, dw2, w2_sz, lr);
        sgd_update(b2, db2, C2, lr);
        sgd_update(wfc, dwfc, wfc_sz, lr);
        sgd_update(bfc, dbfc, NCLS, lr);
    }

    printf("\nTraining complete. Loss should decrease over iterations.\n");
    printf("Note: accuracy on random data is limited; this demonstrates the training loop.\n");

    free(w1); free(b1); free(w2); free(b2); free(wfc); free(bfc);
    free(dw1); free(db1); free(dw2); free(db2); free(dwfc); free(dbfc);
    free(X); free(y); free(a1); free(m1); free(p1); free(am1);
    free(a2); free(m2); free(p2); free(am2);
    free(logits); free(probs); free(dlog);
    free(dp2); free(da2); free(dp1); free(da1);

    printf("\n=== Training Demo Complete ===\n");
    return 0;
}
