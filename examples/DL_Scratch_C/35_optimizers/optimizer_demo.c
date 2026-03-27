/*
 * optimizer_demo.c -- SGD with momentum and Adam optimizer comparison
 *
 * Trains a 2-layer neural network on the XOR problem using both
 * SGD (with momentum) and Adam, comparing convergence.
 *
 * Compile: gcc -std=c11 -Wall -Wextra -O2 -o optimizer_demo optimizer_demo.c -lm
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- XOR dataset ---- */

static const float XOR_X[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
static const float XOR_Y[4]    = {0, 1, 1, 0};

/* ---- 2-layer network: 2 -> hidden -> 1 ---- */

typedef struct {
    int hidden;
    float *w1;   /* [hidden, 2] */
    float *b1;   /* [hidden]    */
    float *w2;   /* [1, hidden] */
    float *b2;   /* [1]         */
    int n_params;
} Net;

static void net_init(Net *net, int hidden) {
    net->hidden = hidden;
    int n = hidden * 2 + hidden + hidden + 1;
    net->n_params = n;
    net->w1 = (float *)malloc((size_t)hidden * 2 * sizeof(float));
    net->b1 = (float *)calloc((size_t)hidden, sizeof(float));
    net->w2 = (float *)malloc((size_t)hidden * sizeof(float));
    net->b2 = (float *)calloc(1, sizeof(float));

    /* Xavier-like init */
    for (int i = 0; i < hidden * 2; i++)
        net->w1[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f;
    for (int i = 0; i < hidden; i++)
        net->w2[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f;
}

static void net_free(Net *net) {
    free(net->w1); free(net->b1); free(net->w2); free(net->b2);
}

/* Flatten params into a single array */
static void net_flatten(const Net *net, float *params) {
    int h = net->hidden;
    memcpy(params,              net->w1, (size_t)h * 2 * sizeof(float));
    memcpy(params + h * 2,      net->b1, (size_t)h * sizeof(float));
    memcpy(params + h * 3,      net->w2, (size_t)h * sizeof(float));
    memcpy(params + h * 4,      net->b2, sizeof(float));
}

/* Unflatten params back to net */
static void net_unflatten(Net *net, const float *params) {
    int h = net->hidden;
    memcpy(net->w1, params,              (size_t)h * 2 * sizeof(float));
    memcpy(net->b1, params + h * 2,      (size_t)h * sizeof(float));
    memcpy(net->w2, params + h * 3,      (size_t)h * sizeof(float));
    memcpy(net->b2, params + h * 4,      sizeof(float));
}

static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

/* Forward + backward for one sample, returns MSE loss */
static float net_train_step(Net *net, const float x[2], float y,
                            float *grads) {
    int h = net->hidden;
    float *hidden = (float *)malloc((size_t)h * sizeof(float));
    float *dh     = (float *)malloc((size_t)h * sizeof(float));

    /* Forward: hidden = sigmoid(W1 @ x + b1) */
    for (int i = 0; i < h; i++) {
        hidden[i] = net->b1[i];
        for (int j = 0; j < 2; j++)
            hidden[i] += net->w1[i * 2 + j] * x[j];
        hidden[i] = sigmoid(hidden[i]);
    }

    /* Output = sigmoid(W2 @ hidden + b2) */
    float out = net->b2[0];
    for (int i = 0; i < h; i++)
        out += net->w2[i] * hidden[i];
    out = sigmoid(out);

    /* MSE loss */
    float loss = (out - y) * (out - y);

    /* Backward */
    float dout = 2.0f * (out - y) * out * (1.0f - out);  /* d(MSE)/d(pre_sigmoid) * sigmoid' */

    /* Gradients for w2, b2 */
    float *dw1 = grads;
    float *db1 = grads + h * 2;
    float *dw2 = grads + h * 3;
    float *db2 = grads + h * 4;

    *db2 = dout;
    for (int i = 0; i < h; i++) {
        dw2[i] = dout * hidden[i];
        dh[i] = dout * net->w2[i] * hidden[i] * (1.0f - hidden[i]);
    }

    for (int i = 0; i < h; i++) {
        db1[i] = dh[i];
        for (int j = 0; j < 2; j++)
            dw1[i * 2 + j] = dh[i] * x[j];
    }

    free(hidden);
    free(dh);
    return loss;
}

/* ---- SGD with Momentum ---- */

typedef struct {
    float *v;
    float lr;
    float momentum;
    int n;
} SGDState;

static SGDState *sgd_new(int n, float lr, float momentum) {
    SGDState *s = (SGDState *)calloc(1, sizeof(SGDState));
    s->n = n;
    s->lr = lr;
    s->momentum = momentum;
    s->v = (float *)calloc((size_t)n, sizeof(float));
    return s;
}

static void sgd_free(SGDState *s) { free(s->v); free(s); }

static void sgd_update(float *params, const float *grads, SGDState *s) {
    for (int i = 0; i < s->n; i++) {
        s->v[i] = s->momentum * s->v[i] + grads[i];
        params[i] -= s->lr * s->v[i];
    }
}

/* ---- Adam ---- */

typedef struct {
    float *m1, *m2;
    float beta1, beta2, eps, lr;
    int step, n;
} AdamState;

static AdamState *adam_new(int n, float lr, float beta1, float beta2, float eps) {
    AdamState *s = (AdamState *)calloc(1, sizeof(AdamState));
    s->n = n;
    s->lr = lr;
    s->beta1 = beta1;
    s->beta2 = beta2;
    s->eps = eps;
    s->step = 0;
    s->m1 = (float *)calloc((size_t)n, sizeof(float));
    s->m2 = (float *)calloc((size_t)n, sizeof(float));
    return s;
}

static void adam_free(AdamState *s) { free(s->m1); free(s->m2); free(s); }

static void adam_update(float *params, const float *grads, AdamState *s) {
    s->step++;
    float bc1 = 1.0f - powf(s->beta1, (float)s->step);
    float bc2 = 1.0f - powf(s->beta2, (float)s->step);
    float lr_corr = s->lr * sqrtf(bc2) / bc1;

    for (int i = 0; i < s->n; i++) {
        float g = grads[i];
        s->m1[i] = s->beta1 * s->m1[i] + (1.0f - s->beta1) * g;
        s->m2[i] = s->beta2 * s->m2[i] + (1.0f - s->beta2) * g * g;
        params[i] -= lr_corr * s->m1[i] / (sqrtf(s->m2[i]) + s->eps);
    }
}

/* ---- Training loop ---- */

static void train_xor(const char *name, int use_adam) {
    srand(12345);  /* Same seed for fair comparison */

    Net net;
    net_init(&net, 8);
    int np = net.n_params;

    float *params = (float *)malloc((size_t)np * sizeof(float));
    float *grads  = (float *)malloc((size_t)np * sizeof(float));
    net_flatten(&net, params);

    SGDState  *sgd  = NULL;
    AdamState *adam  = NULL;
    if (use_adam) {
        adam = adam_new(np, 0.01f, 0.9f, 0.999f, 1e-8f);
    } else {
        sgd = sgd_new(np, 0.1f, 0.9f);
    }

    int max_steps = 2000;
    printf("\n--- Training XOR with %s ---\n", name);
    printf("  %5s  %10s\n", "Step", "Loss");

    for (int step = 0; step < max_steps; step++) {
        memset(grads, 0, (size_t)np * sizeof(float));
        float total_loss = 0.0f;

        /* Accumulate gradients over all 4 samples */
        float *sample_grads = (float *)malloc((size_t)np * sizeof(float));
        for (int s = 0; s < 4; s++) {
            net_unflatten(&net, params);
            float loss = net_train_step(&net, XOR_X[s], XOR_Y[s], sample_grads);
            total_loss += loss;
            for (int i = 0; i < np; i++) grads[i] += sample_grads[i];
        }
        free(sample_grads);

        /* Average gradients */
        for (int i = 0; i < np; i++) grads[i] /= 4.0f;
        total_loss /= 4.0f;

        /* Optimizer step */
        if (use_adam) adam_update(params, grads, adam);
        else          sgd_update(params, grads, sgd);

        if (step % 200 == 0 || step == max_steps - 1)
            printf("  %5d  %10.6f\n", step, total_loss);
    }

    /* Final predictions */
    net_unflatten(&net, params);
    printf("\n  Final predictions:\n");
    for (int s = 0; s < 4; s++) {
        int h = net.hidden;
        float *hidden = (float *)malloc((size_t)h * sizeof(float));
        for (int i = 0; i < h; i++) {
            hidden[i] = net.b1[i];
            for (int j = 0; j < 2; j++)
                hidden[i] += net.w1[i * 2 + j] * XOR_X[s][j];
            hidden[i] = sigmoid(hidden[i]);
        }
        float out = net.b2[0];
        for (int i = 0; i < h; i++) out += net.w2[i] * hidden[i];
        out = sigmoid(out);
        printf("    [%.0f, %.0f] -> %.4f (target: %.0f)\n",
               XOR_X[s][0], XOR_X[s][1], out, XOR_Y[s]);
        free(hidden);
    }

    net_free(&net);
    free(params); free(grads);
    if (sgd)  sgd_free(sgd);
    if (adam) adam_free(adam);
}

/* ---- main ---- */

int main(void) {
    printf("=== Optimizer Comparison Demo ===\n");
    printf("Task: XOR problem with 2-layer network (2 -> 8 -> 1)\n");
    printf("SGD: lr=0.1, momentum=0.9\n");
    printf("Adam: lr=0.01, beta1=0.9, beta2=0.999\n");

    train_xor("SGD (momentum=0.9)", 0);
    train_xor("Adam", 1);

    printf("\n--- Optimizer Summary ---\n");
    printf("  SGD with momentum:\n");
    printf("    v_t = beta * v_{t-1} + g_t\n");
    printf("    theta -= lr * v_t\n");
    printf("    Pros: simple, good generalization, low memory\n");
    printf("    Cons: sensitive to LR, slower convergence\n\n");
    printf("  Adam:\n");
    printf("    m1_t = beta1 * m1_{t-1} + (1-beta1) * g_t\n");
    printf("    m2_t = beta2 * m2_{t-1} + (1-beta2) * g_t^2\n");
    printf("    theta -= lr_corrected * m1_hat / (sqrt(m2_hat) + eps)\n");
    printf("    Pros: adaptive LR per parameter, fast convergence\n");
    printf("    Cons: 2x memory for m1/m2, may generalize worse\n");

    return 0;
}
