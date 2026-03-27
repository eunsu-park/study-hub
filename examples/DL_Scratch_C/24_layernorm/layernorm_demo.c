/*
 * layernorm_demo.c - Layer Normalization forward and backward
 *
 * Demonstrates:
 *   - LayerNorm forward: normalize over last dimension per token
 *   - LayerNorm backward: gradient through mean and variance
 *   - RMSNorm forward (Llama-style, no mean subtraction)
 *   - Comparison: LayerNorm vs BatchNorm behavior on same input
 *   - Numerical gradient check for backward correctness
 *
 * Build:  gcc -std=c11 -Wall -Wextra -O2 -o layernorm_demo layernorm_demo.c -lm
 * Run:    ./layernorm_demo
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define LN_EPS 1e-5f

static float randn(void) {
    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    float u2 = (float)rand() / ((float)RAND_MAX + 1.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

/* ---- LayerNorm forward ---- */
static void layernorm_forward(const float *X, const float *gamma,
                                const float *beta, float *Y,
                                float *mean, float *rstd,
                                int M, int C) {
    for (int m = 0; m < M; m++) {
        const float *x = X + (long)m * C;
        float       *y = Y + (long)m * C;

        float mu = 0.0f;
        for (int i = 0; i < C; i++) mu += x[i];
        mu /= C;

        float var = 0.0f;
        for (int i = 0; i < C; i++) {
            float d = x[i] - mu;
            var += d * d;
        }
        var /= C;

        float rs = 1.0f / sqrtf(var + LN_EPS);
        mean[m] = mu;
        rstd[m] = rs;

        for (int i = 0; i < C; i++)
            y[i] = gamma[i] * (x[i] - mu) * rs + beta[i];
    }
}

/* ---- LayerNorm backward ---- */
static void layernorm_backward(const float *dY, const float *X,
                                 const float *gamma, const float *mean_arr,
                                 const float *rstd_arr,
                                 float *dX, float *dgamma, float *dbeta,
                                 int M, int C) {
    for (int m = 0; m < M; m++) {
        const float *dy = dY + (long)m * C;
        const float *x  = X  + (long)m * C;
        float       *dx = dX + (long)m * C;
        float mu = mean_arr[m];
        float rs = rstd_arr[m];

        float sum1 = 0.0f, sum2 = 0.0f;
        for (int i = 0; i < C; i++) {
            float xhat_i   = (x[i] - mu) * rs;
            dgamma[i] += dy[i] * xhat_i;
            dbeta[i]  += dy[i];
            float dx_hat_i = dy[i] * gamma[i];
            sum1 += dx_hat_i;
            sum2 += dx_hat_i * xhat_i;
        }

        float inv_C = 1.0f / C;
        for (int i = 0; i < C; i++) {
            float xhat_i   = (x[i] - mu) * rs;
            float dx_hat_i = dy[i] * gamma[i];
            dx[i] = rs * (dx_hat_i - inv_C * sum1 - inv_C * xhat_i * sum2);
        }
    }
}

/* ---- RMSNorm forward (Llama/Mistral style) ---- */
static void rmsnorm_forward(const float *X, const float *gamma,
                              float *Y, float *rrms,
                              int M, int C) {
    for (int m = 0; m < M; m++) {
        const float *x = X + (long)m * C;
        float       *y = Y + (long)m * C;

        float ss = 0.0f;
        for (int i = 0; i < C; i++) ss += x[i] * x[i];
        float rms = 1.0f / sqrtf(ss / C + LN_EPS);
        rrms[m] = rms;

        for (int i = 0; i < C; i++)
            y[i] = gamma[i] * x[i] * rms;
    }
}

/* ---- BatchNorm-style normalization (for comparison) ---- */
static void batchnorm_like(const float *X, float *Y,
                             int M, int C) {
    /* Normalize over M (batch/token dimension) per feature */
    for (int c = 0; c < C; c++) {
        float mu = 0.0f;
        for (int m = 0; m < M; m++) mu += X[m * C + c];
        mu /= M;

        float var = 0.0f;
        for (int m = 0; m < M; m++) {
            float d = X[m * C + c] - mu;
            var += d * d;
        }
        var /= M;

        float rs = 1.0f / sqrtf(var + LN_EPS);
        for (int m = 0; m < M; m++)
            Y[m * C + c] = (X[m * C + c] - mu) * rs;
    }
}

static void print_mat(const char *label, const float *M_arr, int rows, int cols) {
    printf("  %s:\n", label);
    for (int r = 0; r < rows; r++) {
        printf("    [");
        for (int c = 0; c < cols; c++) {
            printf("%8.4f", M_arr[r * cols + c]);
            if (c < cols - 1) printf(", ");
        }
        printf("]\n");
    }
}

int main(void) {
    srand(42);

    int M = 4;  /* number of tokens */
    int C = 6;  /* feature dimension (d_model) */

    printf("=== Layer Normalization Demo ===\n\n");
    printf("Config: M=%d tokens, C=%d features\n\n", M, C);

    /* Create input with varying scales per token */
    float X[] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,     /* token 0: range 1-6 */
        10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, /* token 1: range 10-60 */
        -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f,      /* token 2: range -1 to 4 */
        0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,        /* token 3: range 0.1-0.6 */
    };

    float gamma[6], beta[6];
    for (int i = 0; i < C; i++) { gamma[i] = 1.0f; beta[i] = 0.0f; }

    printf("=== Input ===\n");
    print_mat("X", X, M, C);
    printf("\n");

    /* LayerNorm */
    float Y_ln[24], mean_ln[4], rstd_ln[4];
    layernorm_forward(X, gamma, beta, Y_ln, mean_ln, rstd_ln, M, C);

    printf("=== LayerNorm Output (gamma=1, beta=0) ===\n");
    print_mat("LN(X)", Y_ln, M, C);

    printf("\n  Per-token statistics after LayerNorm:\n");
    for (int m = 0; m < M; m++) {
        float s = 0.0f, s2 = 0.0f;
        for (int i = 0; i < C; i++) {
            s  += Y_ln[m * C + i];
            s2 += Y_ln[m * C + i] * Y_ln[m * C + i];
        }
        printf("    token %d: mean=%.6f  var=%.6f  (input mean=%.2f)\n",
               m, s / C, s2 / C - (s / C) * (s / C), mean_ln[m]);
    }
    printf("    -> Each token independently normalized to mean~0, var~1\n\n");

    /* BatchNorm comparison */
    float Y_bn[24];
    batchnorm_like(X, Y_bn, M, C);

    printf("=== BatchNorm-style Output (for comparison) ===\n");
    print_mat("BN(X)", Y_bn, M, C);

    printf("\n  Per-feature statistics after BatchNorm:\n");
    for (int c = 0; c < C; c++) {
        float s = 0.0f, s2 = 0.0f;
        for (int m = 0; m < M; m++) {
            s  += Y_bn[m * C + c];
            s2 += Y_bn[m * C + c] * Y_bn[m * C + c];
        }
        printf("    feature %d: mean=%.6f  var=%.6f\n",
               c, s / M, s2 / M - (s / M) * (s / M));
    }
    printf("    -> Each feature normalized across tokens (batch-dependent!)\n\n");

    printf("=== Key Difference ===\n");
    printf("  LayerNorm: normalizes over features (d_model) per token\n");
    printf("             -> each token independently, batch-size agnostic\n");
    printf("  BatchNorm: normalizes over batch per feature\n");
    printf("             -> depends on batch statistics, fails for batch=1\n\n");

    /* RMSNorm */
    float Y_rms[24], rrms[4];
    float gamma_rms[6];
    for (int i = 0; i < C; i++) gamma_rms[i] = 1.0f;
    rmsnorm_forward(X, gamma_rms, Y_rms, rrms, M, C);

    printf("=== RMSNorm Output (Llama-style) ===\n");
    print_mat("RMSNorm(X)", Y_rms, M, C);
    printf("  -> No mean subtraction, normalizes by RMS only\n\n");

    /* ---- Backward pass verification ---- */
    printf("=== LayerNorm Backward Pass ===\n\n");

    /* Use random gamma for a more interesting gradient test */
    for (int i = 0; i < C; i++) { gamma[i] = 1.0f + randn() * 0.1f; beta[i] = randn() * 0.1f; }
    layernorm_forward(X, gamma, beta, Y_ln, mean_ln, rstd_ln, M, C);

    float dY[24];
    for (int i = 0; i < M * C; i++) dY[i] = randn() * 0.1f;

    float dX[24], dgamma[6] = {0}, dbeta[6] = {0};
    layernorm_backward(dY, X, gamma, mean_ln, rstd_ln, dX, dgamma, dbeta, M, C);

    printf("Analytical gradients (dX):\n");
    print_mat("dX", dX, M, C);

    /* Numerical gradient check */
    printf("\nNumerical gradient check (finite differences):\n");
    float eps = 1e-4f;
    int n_checks = 6;
    float max_err = 0.0f;

    for (int idx = 0; idx < n_checks; idx++) {
        float Xp[24], Xm[24], Yp[24], Ym[24];
        float mp[4], rp[4], mm[4], rm[4];
        memcpy(Xp, X, sizeof(X)); Xp[idx] += eps;
        memcpy(Xm, X, sizeof(X)); Xm[idx] -= eps;

        layernorm_forward(Xp, gamma, beta, Yp, mp, rp, M, C);
        layernorm_forward(Xm, gamma, beta, Ym, mm, rm, M, C);

        /* Compute L = sum(dY * Y) for both */
        float Lp = 0.0f, Lm = 0.0f;
        for (int i = 0; i < M * C; i++) {
            Lp += dY[i] * Yp[i];
            Lm += dY[i] * Ym[i];
        }
        float num_grad = (Lp - Lm) / (2.0f * eps);
        float err = fabsf(dX[idx] - num_grad) / (fabsf(num_grad) + 1e-8f);
        if (err > max_err) max_err = err;
        printf("  dX[%d]: analytical=%.6f  numerical=%.6f  rel_err=%.2e %s\n",
               idx, dX[idx], num_grad, err, err < 1e-3f ? "OK" : "FAIL");
    }
    printf("\n  Max relative error: %.2e %s\n\n",
           max_err, max_err < 1e-3f ? "(PASS)" : "(FAIL)");

    /* Parameter gradients */
    printf("Parameter gradients:\n");
    printf("  dgamma: [");
    for (int i = 0; i < C; i++) printf("%.4f%s", dgamma[i], i < C - 1 ? ", " : "");
    printf("]\n  dbeta:  [");
    for (int i = 0; i < C; i++) printf("%.4f%s", dbeta[i], i < C - 1 ? ", " : "");
    printf("]\n\n");

    printf("=== Summary ===\n");
    printf("  LayerNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta\n");
    printf("  RMSNorm:   y = gamma * x / sqrt(mean(x^2) + eps)  (no beta)\n");
    printf("  Pre-norm:  y = x + sublayer(LN(x))  (modern Transformer default)\n");

    printf("\nDone.\n");
    return 0;
}
