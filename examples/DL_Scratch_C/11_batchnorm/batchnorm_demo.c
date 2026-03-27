/*
 * batchnorm_demo.c - Batch Normalization forward (train/eval) and backward
 *
 * Demonstrates:
 *   - BN training forward: compute batch mean/var, normalize, scale/shift
 *   - Running EMA statistics for inference
 *   - BN eval forward: use frozen running statistics
 *   - BN backward: gradients for dX, dgamma, dbeta
 *   - Effect of BN on synthetic data distribution
 *   - Numerical gradient verification
 *
 * Build:  gcc -std=c11 -Wall -Wextra -O2 -o batchnorm_demo batchnorm_demo.c -lm
 * Run:    ./batchnorm_demo
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NCHW(ptr, N, C, H, W, n, c, h, w) \
    ((ptr)[(n)*(C)*(H)*(W) + (c)*(H)*(W) + (h)*(W) + (w)])

#define BN_EPS 1e-5f

/* ---- BN Training Forward ---- */
static void bn_forward_train(
    const float *X, const float *gamma, const float *beta,
    float *Y, float *mean, float *var, float *X_hat,
    float *run_mean, float *run_var, float momentum,
    int N, int C, int H, int W) {

    int M = N * H * W;

    for (int c = 0; c < C; c++) {
        /* Compute batch mean */
        float m = 0.0f;
        for (int n = 0; n < N; n++)
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++)
            m += NCHW(X, N, C, H, W, n, c, h, w);
        m /= M;
        mean[c] = m;

        /* Compute batch variance */
        float v = 0.0f;
        for (int n = 0; n < N; n++)
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++) {
            float diff = NCHW(X, N, C, H, W, n, c, h, w) - m;
            v += diff * diff;
        }
        v /= M;
        var[c] = v;

        float inv_std = 1.0f / sqrtf(v + BN_EPS);

        /* Normalize and scale */
        for (int n = 0; n < N; n++)
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++) {
            float x_norm = (NCHW(X, N, C, H, W, n, c, h, w) - m) * inv_std;
            NCHW(X_hat, N, C, H, W, n, c, h, w) = x_norm;
            NCHW(Y, N, C, H, W, n, c, h, w) = gamma[c] * x_norm + beta[c];
        }

        /* Update running statistics (EMA) */
        run_mean[c] = (1.0f - momentum) * run_mean[c] + momentum * m;
        run_var[c]  = (1.0f - momentum) * run_var[c]  + momentum * v;
    }
}

/* ---- BN Eval Forward ---- */
static void bn_forward_eval(
    const float *X, const float *gamma, const float *beta,
    float *Y, const float *run_mean, const float *run_var,
    int N, int C, int H, int W) {

    for (int c = 0; c < C; c++) {
        float inv_std = 1.0f / sqrtf(run_var[c] + BN_EPS);
        for (int n = 0; n < N; n++)
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++) {
            float x_norm = (NCHW(X, N, C, H, W, n, c, h, w) - run_mean[c]) * inv_std;
            NCHW(Y, N, C, H, W, n, c, h, w) = gamma[c] * x_norm + beta[c];
        }
    }
}

/* ---- BN Backward ---- */
static void bn_backward(
    const float *dY, const float *X_hat, const float *gamma,
    const float *var,
    float *dX, float *dgamma, float *dbeta,
    int N, int C, int H, int W) {

    int M = N * H * W;

    for (int c = 0; c < C; c++) {
        float inv_std = 1.0f / sqrtf(var[c] + BN_EPS);

        /* dgamma, dbeta */
        float dg = 0.0f, db_val = 0.0f;
        for (int n = 0; n < N; n++)
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++) {
            float dy   = NCHW(dY,    N, C, H, W, n, c, h, w);
            float xhat = NCHW(X_hat, N, C, H, W, n, c, h, w);
            dg     += dy * xhat;
            db_val += dy;
        }
        dgamma[c] += dg;
        dbeta[c]  += db_val;

        /* Compute sum1, sum2 for dX */
        float sum1 = 0.0f, sum2 = 0.0f;
        for (int n = 0; n < N; n++)
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++) {
            float dxhat = NCHW(dY, N, C, H, W, n, c, h, w) * gamma[c];
            float xhat  = NCHW(X_hat, N, C, H, W, n, c, h, w);
            sum1 += dxhat;
            sum2 += dxhat * xhat;
        }

        /* dX */
        float scale = inv_std / M;
        for (int n = 0; n < N; n++)
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++) {
            float dxhat = NCHW(dY, N, C, H, W, n, c, h, w) * gamma[c];
            float xhat  = NCHW(X_hat, N, C, H, W, n, c, h, w);
            NCHW(dX, N, C, H, W, n, c, h, w) =
                scale * ((float)M * dxhat - sum1 - xhat * sum2);
        }
    }
}

/* ---- Statistics helper ---- */
static void compute_stats(const float *data, int size, float *mean, float *std) {
    float sum = 0.0f, sum2 = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += data[i];
        sum2 += data[i] * data[i];
    }
    *mean = sum / size;
    *std  = sqrtf(sum2 / size - (*mean) * (*mean) + 1e-8f);
}

/* ---- Simple randn (Box-Muller) ---- */
static float randn(void) {
    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    float u2 = (float)rand() / ((float)RAND_MAX + 1.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

int main(void) {
    srand(42);
    printf("=== Batch Normalization Demo ===\n\n");

    int N = 4, C = 3, H = 4, W = 4;
    int total = N * C * H * W;
    int spatial = N * H * W;

    float *X      = (float *)malloc((size_t)total * sizeof(float));
    float *Y      = (float *)malloc((size_t)total * sizeof(float));
    float *X_hat  = (float *)malloc((size_t)total * sizeof(float));
    float *gamma  = (float *)malloc((size_t)C * sizeof(float));
    float *beta   = (float *)malloc((size_t)C * sizeof(float));
    float *mean   = (float *)malloc((size_t)C * sizeof(float));
    float *var    = (float *)malloc((size_t)C * sizeof(float));
    float *run_mean = (float *)calloc((size_t)C, sizeof(float));
    float *run_var  = (float *)malloc((size_t)C * sizeof(float));

    /* Initialize: channels have different distributions */
    printf("--- Generating synthetic data ---\n");
    float ch_means[] = {10.0f, -5.0f, 100.0f};
    float ch_stds[]  = {2.0f,   0.5f,  20.0f};
    for (int c = 0; c < C; c++) {
        for (int n = 0; n < N; n++)
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++)
            NCHW(X, N, C, H, W, n, c, h, w) = ch_means[c] + ch_stds[c] * randn();
    }

    /* Compute and print pre-BN stats */
    for (int c = 0; c < C; c++) {
        float m, s;
        float ch_data[64];
        for (int n = 0; n < N; n++)
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++)
            ch_data[n * H * W + h * W + w] = NCHW(X, N, C, H, W, n, c, h, w);
        compute_stats(ch_data, spatial, &m, &s);
        printf("  Channel %d before BN: mean=%7.2f  std=%6.2f\n", c, m, s);
    }
    printf("\n");

    /* Initialize BN parameters */
    for (int c = 0; c < C; c++) {
        gamma[c] = 1.0f;
        beta[c]  = 0.0f;
        run_var[c] = 1.0f;
    }

    /* ---- Training forward ---- */
    printf("--- BN Training Forward ---\n");
    bn_forward_train(X, gamma, beta, Y, mean, var, X_hat,
                     run_mean, run_var, 0.1f, N, C, H, W);

    for (int c = 0; c < C; c++) {
        float m, s;
        float ch_data[64];
        for (int n = 0; n < N; n++)
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++)
            ch_data[n * H * W + h * W + w] = NCHW(Y, N, C, H, W, n, c, h, w);
        compute_stats(ch_data, spatial, &m, &s);
        printf("  Channel %d after  BN: mean=%7.4f  std=%6.4f  (batch_mean=%7.2f var=%6.2f)\n",
               c, m, s, mean[c], var[c]);
    }
    printf("\n");

    /* ---- Eval forward (using running stats) ---- */
    printf("--- BN Eval Forward (EMA stats after 1 step) ---\n");
    float *Y_eval = (float *)malloc((size_t)total * sizeof(float));
    bn_forward_eval(X, gamma, beta, Y_eval, run_mean, run_var, N, C, H, W);

    for (int c = 0; c < C; c++) {
        float m, s;
        float ch_data[64];
        for (int n = 0; n < N; n++)
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++)
            ch_data[n * H * W + h * W + w] = NCHW(Y_eval, N, C, H, W, n, c, h, w);
        compute_stats(ch_data, spatial, &m, &s);
        printf("  Channel %d eval mode: mean=%7.4f  std=%6.4f  (run_mean=%7.2f run_var=%6.2f)\n",
               c, m, s, run_mean[c], run_var[c]);
    }
    printf("  Note: eval stats differ because EMA has only 1 update (momentum=0.1)\n\n");

    /* ---- Backward ---- */
    printf("--- BN Backward ---\n");
    float *dY     = (float *)malloc((size_t)total * sizeof(float));
    float *dX     = (float *)malloc((size_t)total * sizeof(float));
    float *dgamma = (float *)calloc((size_t)C, sizeof(float));
    float *dbeta  = (float *)calloc((size_t)C, sizeof(float));

    for (int i = 0; i < total; i++)
        dY[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;

    bn_backward(dY, X_hat, gamma, var, dX, dgamma, dbeta, N, C, H, W);

    for (int c = 0; c < C; c++)
        printf("  Channel %d: dgamma=%8.5f  dbeta=%8.5f\n", c, dgamma[c], dbeta[c]);
    printf("\n");

    /* ---- Numerical gradient check ---- */
    printf("--- Numerical Gradient Check ---\n");
    float eps = 1e-4f;
    int errors = 0;
    float max_rel = 0.0f;

    /* Check a subset of dX elements */
    float *Y_p = (float *)malloc((size_t)total * sizeof(float));
    float *Y_m = (float *)malloc((size_t)total * sizeof(float));
    float *Xhat_tmp = (float *)malloc((size_t)total * sizeof(float));
    float *m_tmp = (float *)malloc((size_t)C * sizeof(float));
    float *v_tmp = (float *)malloc((size_t)C * sizeof(float));
    float *rm_tmp = (float *)calloc((size_t)C, sizeof(float));
    float *rv_tmp = (float *)malloc((size_t)C * sizeof(float));

    int check_n = total < 30 ? total : 30;
    for (int i = 0; i < check_n; i++) {
        float orig = X[i];

        for (int c = 0; c < C; c++) { rm_tmp[c] = 0; rv_tmp[c] = 1.0f; }
        X[i] = orig + eps;
        bn_forward_train(X, gamma, beta, Y_p, m_tmp, v_tmp, Xhat_tmp,
                         rm_tmp, rv_tmp, 0.1f, N, C, H, W);

        for (int c = 0; c < C; c++) { rm_tmp[c] = 0; rv_tmp[c] = 1.0f; }
        X[i] = orig - eps;
        bn_forward_train(X, gamma, beta, Y_m, m_tmp, v_tmp, Xhat_tmp,
                         rm_tmp, rv_tmp, 0.1f, N, C, H, W);

        X[i] = orig;

        float num_grad = 0.0f;
        for (int j = 0; j < total; j++)
            num_grad += dY[j] * (Y_p[j] - Y_m[j]) / (2.0f * eps);

        float rel = fabsf(dX[i] - num_grad) / (fabsf(num_grad) + 1e-8f);
        if (rel > max_rel) max_rel = rel;
        if (rel > 5e-2f) errors++;
    }
    printf("  dX check: max_rel=%.6f  errors=%d/%d  %s\n",
           max_rel, errors, check_n, errors == 0 ? "PASS" : "FAIL");

    /* Check dgamma */
    float dgamma_num[3] = {0};
    for (int c = 0; c < C; c++) {
        float orig = gamma[c];

        for (int cc = 0; cc < C; cc++) { rm_tmp[cc] = 0; rv_tmp[cc] = 1.0f; }
        gamma[c] = orig + eps;
        bn_forward_train(X, gamma, beta, Y_p, m_tmp, v_tmp, Xhat_tmp,
                         rm_tmp, rv_tmp, 0.1f, N, C, H, W);

        for (int cc = 0; cc < C; cc++) { rm_tmp[cc] = 0; rv_tmp[cc] = 1.0f; }
        gamma[c] = orig - eps;
        bn_forward_train(X, gamma, beta, Y_m, m_tmp, v_tmp, Xhat_tmp,
                         rm_tmp, rv_tmp, 0.1f, N, C, H, W);

        gamma[c] = orig;

        for (int j = 0; j < total; j++)
            dgamma_num[c] += dY[j] * (Y_p[j] - Y_m[j]) / (2.0f * eps);
    }

    float dg_max_rel = 0.0f;
    for (int c = 0; c < C; c++) {
        float rel = fabsf(dgamma[c] - dgamma_num[c]) / (fabsf(dgamma_num[c]) + 1e-8f);
        if (rel > dg_max_rel) dg_max_rel = rel;
    }
    printf("  dgamma check: max_rel=%.6f  %s\n",
           dg_max_rel, dg_max_rel < 1e-2f ? "PASS" : "FAIL");

    printf("\n=== Batch Normalization Demo Complete ===\n");

    free(X); free(Y); free(Y_eval); free(X_hat);
    free(gamma); free(beta); free(mean); free(var);
    free(run_mean); free(run_var);
    free(dY); free(dX); free(dgamma); free(dbeta);
    free(Y_p); free(Y_m); free(Xhat_tmp);
    free(m_tmp); free(v_tmp); free(rm_tmp); free(rv_tmp);
    return 0;
}
