/*
 * lenet_demo.c - Simplified LeNet-5 forward pass
 *
 * Demonstrates:
 *   - LeNet-5 architecture: Conv->Pool->Conv->Pool->FC->FC->FC
 *   - Naive convolution, average pooling, tanh activation, dense layers
 *   - Xavier weight initialization for tanh networks
 *   - Shape tracing through each layer
 *   - Output logits for a synthetic 32x32 grayscale input
 *
 * Build:  gcc -std=c11 -Wall -Wextra -O2 -o lenet_demo lenet_demo.c -lm
 * Run:    ./lenet_demo
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NCHW(ptr, N, C, H, W, n, c, h, w) \
    ((ptr)[(n)*(C)*(H)*(W) + (c)*(H)*(W) + (h)*(W) + (w)])

/* Box-Muller standard normal */
static float randn(void) {
    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    float u2 = (float)rand() / ((float)RAND_MAX + 1.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

/* Xavier initialization: std = sqrt(2 / (fan_in + fan_out)) */
static void xavier_init(float *W, int fan_in, int fan_out, int total) {
    float std = sqrtf(2.0f / (float)(fan_in + fan_out));
    for (int i = 0; i < total; i++) W[i] = randn() * std;
}

/* ---- Naive 2D Convolution ---- */
static void conv2d(
    const float *X, int N, int Ci, int H, int W,
    const float *Wt, int Co, int KH, int KW,
    const float *bias,
    float *Y, int OH, int OW,
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

/* ---- Average Pooling ---- */
static void avg_pool2d(
    const float *X, int N, int C, int H, int W,
    float *Y, int OH, int OW, int K, int stride) {

    for (int n = 0; n < N; n++)
    for (int c = 0; c < C; c++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        float sum = 0.0f;
        for (int kh = 0; kh < K; kh++)
        for (int kw = 0; kw < K; kw++)
            sum += NCHW(X, N, C, H, W, n, c, oh * stride + kh, ow * stride + kw);
        NCHW(Y, N, C, OH, OW, n, c, oh, ow) = sum / (K * K);
    }
}

/* ---- Activations ---- */
static void apply_tanh(float *X, int size) {
    for (int i = 0; i < size; i++) X[i] = tanhf(X[i]);
}

/* ---- Fully connected: Y[N, out] = X[N, in] * W[in, out] + bias[out] ---- */
static void fc_forward(const float *X, const float *W, const float *bias,
                        float *Y, int N, int in_dim, int out_dim) {
    for (int n = 0; n < N; n++)
    for (int o = 0; o < out_dim; o++) {
        float sum = bias[o];
        for (int i = 0; i < in_dim; i++)
            sum += X[n * in_dim + i] * W[i * out_dim + o];
        Y[n * out_dim + o] = sum;
    }
}

/* ---- Softmax ---- */
static void softmax(const float *logits, float *probs, int N, int C) {
    for (int n = 0; n < N; n++) {
        const float *row = logits + n * C;
        float *p = probs + n * C;
        float mx = row[0];
        for (int c = 1; c < C; c++)
            if (row[c] > mx) mx = row[c];
        float sum = 0.0f;
        for (int c = 0; c < C; c++) {
            p[c] = expf(row[c] - mx);
            sum += p[c];
        }
        for (int c = 0; c < C; c++) p[c] /= sum;
    }
}

/* ---- LeNet-5 Weights ---- */
typedef struct {
    float *c1_w, *c1_b;     /* Conv1: 1->6, 5x5 */
    float *c3_w, *c3_b;     /* Conv3: 6->16, 5x5 */
    float *c5_w, *c5_b;     /* Conv5: 16->120, 5x5 */
    float *f6_w, *f6_b;     /* FC6: 120->84 */
    float *out_w, *out_b;   /* Out: 84->10 */
} LeNet5;

static LeNet5 *lenet_create(void) {
    LeNet5 *net = (LeNet5 *)malloc(sizeof(LeNet5));
    net->c1_w = (float *)malloc(6 * 1 * 5 * 5 * sizeof(float));
    net->c1_b = (float *)calloc(6, sizeof(float));
    net->c3_w = (float *)malloc(16 * 6 * 5 * 5 * sizeof(float));
    net->c3_b = (float *)calloc(16, sizeof(float));
    net->c5_w = (float *)malloc(120 * 16 * 5 * 5 * sizeof(float));
    net->c5_b = (float *)calloc(120, sizeof(float));
    net->f6_w = (float *)malloc(120 * 84 * sizeof(float));
    net->f6_b = (float *)calloc(84, sizeof(float));
    net->out_w = (float *)malloc(84 * 10 * sizeof(float));
    net->out_b = (float *)calloc(10, sizeof(float));

    /* Xavier init */
    xavier_init(net->c1_w, 1 * 5 * 5, 6, 6 * 1 * 5 * 5);
    xavier_init(net->c3_w, 6 * 5 * 5, 16, 16 * 6 * 5 * 5);
    xavier_init(net->c5_w, 16 * 5 * 5, 120, 120 * 16 * 5 * 5);
    xavier_init(net->f6_w, 120, 84, 120 * 84);
    xavier_init(net->out_w, 84, 10, 84 * 10);
    return net;
}

static void lenet_free(LeNet5 *net) {
    free(net->c1_w); free(net->c1_b);
    free(net->c3_w); free(net->c3_b);
    free(net->c5_w); free(net->c5_b);
    free(net->f6_w); free(net->f6_b);
    free(net->out_w); free(net->out_b);
    free(net);
}

static int count_params(void) {
    int c1 = (5*5*1 + 1) * 6;
    int c3 = (5*5*6 + 1) * 16;
    int c5 = (5*5*16 + 1) * 120;
    int f6 = (120 + 1) * 84;
    int out = (84 + 1) * 10;
    return c1 + c3 + c5 + f6 + out;
}

int main(void) {
    srand(42);
    printf("=== LeNet-5 Forward Pass Demo ===\n\n");

    int N = 2;  /* batch size */
    LeNet5 *net = lenet_create();

    /* Generate synthetic 32x32 grayscale input */
    int in_size = N * 1 * 32 * 32;
    float *input = (float *)malloc((size_t)in_size * sizeof(float));
    for (int i = 0; i < in_size; i++)
        input[i] = randn() * 0.3f;

    printf("Architecture: LeNet-5\n");
    printf("Total parameters: %d\n\n", count_params());

    /* C1: Conv(1->6, 5x5, s=1, p=0) -> [N,6,28,28] + Tanh */
    int c1_sz = N * 6 * 28 * 28;
    float *c1_out = (float *)malloc((size_t)c1_sz * sizeof(float));
    conv2d(input, N, 1, 32, 32, net->c1_w, 6, 5, 5, net->c1_b,
           c1_out, 28, 28, 1, 0);
    apply_tanh(c1_out, c1_sz);
    printf("C1: [%d,1,32,32] -> Conv(1->6,5x5) -> Tanh -> [%d,6,28,28]\n", N, N);

    /* S2: AvgPool(2x2, s=2) -> [N,6,14,14] */
    int s2_sz = N * 6 * 14 * 14;
    float *s2_out = (float *)malloc((size_t)s2_sz * sizeof(float));
    avg_pool2d(c1_out, N, 6, 28, 28, s2_out, 14, 14, 2, 2);
    printf("S2: [%d,6,28,28] -> AvgPool(2x2)   -> [%d,6,14,14]\n", N, N);

    /* C3: Conv(6->16, 5x5, s=1, p=0) -> [N,16,10,10] + Tanh */
    int c3_sz = N * 16 * 10 * 10;
    float *c3_out = (float *)malloc((size_t)c3_sz * sizeof(float));
    conv2d(s2_out, N, 6, 14, 14, net->c3_w, 16, 5, 5, net->c3_b,
           c3_out, 10, 10, 1, 0);
    apply_tanh(c3_out, c3_sz);
    printf("C3: [%d,6,14,14] -> Conv(6->16,5x5) -> Tanh -> [%d,16,10,10]\n", N, N);

    /* S4: AvgPool(2x2, s=2) -> [N,16,5,5] */
    int s4_sz = N * 16 * 5 * 5;
    float *s4_out = (float *)malloc((size_t)s4_sz * sizeof(float));
    avg_pool2d(c3_out, N, 16, 10, 10, s4_out, 5, 5, 2, 2);
    printf("S4: [%d,16,10,10]-> AvgPool(2x2)   -> [%d,16,5,5]\n", N, N);

    /* C5: Conv(16->120, 5x5, s=1, p=0) -> [N,120,1,1] + Tanh */
    int c5_sz = N * 120 * 1 * 1;
    float *c5_out = (float *)malloc((size_t)c5_sz * sizeof(float));
    conv2d(s4_out, N, 16, 5, 5, net->c5_w, 120, 5, 5, net->c5_b,
           c5_out, 1, 1, 1, 0);
    apply_tanh(c5_out, c5_sz);
    printf("C5: [%d,16,5,5]  -> Conv(16->120,5x5)-> Tanh -> [%d,120,1,1]\n", N, N);

    /* F6: FC(120->84) + Tanh */
    float *f6_out = (float *)malloc((size_t)N * 84 * sizeof(float));
    fc_forward(c5_out, net->f6_w, net->f6_b, f6_out, N, 120, 84);
    apply_tanh(f6_out, N * 84);
    printf("F6: [%d,120]     -> FC(120->84)     -> Tanh -> [%d,84]\n", N, N);

    /* Output: FC(84->10) */
    float *logits = (float *)malloc((size_t)N * 10 * sizeof(float));
    fc_forward(f6_out, net->out_w, net->out_b, logits, N, 84, 10);
    printf("OUT:[%d,84]      -> FC(84->10)      -> [%d,10]\n\n", N, N);

    /* Print raw logits and softmax probabilities */
    float *probs = (float *)malloc((size_t)N * 10 * sizeof(float));
    softmax(logits, probs, N, 10);

    for (int n = 0; n < N; n++) {
        printf("Sample %d logits:  ", n);
        for (int c = 0; c < 10; c++) printf("%7.3f ", logits[n * 10 + c]);
        printf("\n");

        printf("Sample %d softmax: ", n);
        for (int c = 0; c < 10; c++) printf("%7.4f ", probs[n * 10 + c]);
        printf("\n");

        int pred = 0;
        for (int c = 1; c < 10; c++)
            if (probs[n * 10 + c] > probs[n * 10 + pred]) pred = c;
        printf("Sample %d prediction: class %d (prob=%.4f)\n\n", n, pred, probs[n * 10 + pred]);
    }

    /* Parameter count per layer */
    printf("--- Parameter Count ---\n");
    printf("C1: %5d  (5x5x1+1)x6\n", (5*5*1+1)*6);
    printf("C3: %5d  (5x5x6+1)x16\n", (5*5*6+1)*16);
    printf("C5: %5d  (5x5x16+1)x120\n", (5*5*16+1)*120);
    printf("F6: %5d  (120+1)x84\n", (120+1)*84);
    printf("OUT:%5d  (84+1)x10\n", (84+1)*10);
    printf("Total: %d\n", count_params());

    free(input); free(c1_out); free(s2_out);
    free(c3_out); free(s4_out); free(c5_out);
    free(f6_out); free(logits); free(probs);
    lenet_free(net);

    printf("\n=== LeNet-5 Demo Complete ===\n");
    return 0;
}
