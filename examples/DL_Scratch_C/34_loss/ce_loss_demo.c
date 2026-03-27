/*
 * ce_loss_demo.c -- Softmax + cross-entropy loss with forward/backward
 *
 * Demonstrates: logsumexp, log-softmax, fused CE forward+backward,
 * gradient verification via numerical gradient check.
 *
 * Compile: gcc -std=c11 -Wall -Wextra -O2 -o ce_loss_demo ce_loss_demo.c -lm
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Logsumexp (numerically stable) ---- */

static float logsumexp(const float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += expf(x[i] - mx);
    return mx + logf(sum);
}

/* ---- Log-softmax ---- */

static void log_softmax(const float *x, float *out, int n) {
    float lse = logsumexp(x, n);
    for (int i = 0; i < n; i++) out[i] = x[i] - lse;
}

/* ---- Softmax ---- */

static void softmax(const float *x, float *out, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { out[i] = expf(x[i] - mx); sum += out[i]; }
    for (int i = 0; i < n; i++) out[i] /= sum;
}

/* ---- Cross-Entropy Forward (naive, for reference) ---- */

static float ce_forward_naive(const float *logits, const int *targets,
                              float *losses, int N, int V) {
    float total = 0.0f;
    for (int i = 0; i < N; i++) {
        const float *row = logits + i * V;
        float lse = logsumexp(row, V);
        losses[i] = -(row[targets[i]] - lse);
        total += losses[i];
    }
    return total / (float)N;
}

/* ---- Fused Cross-Entropy Forward + Backward ---- */

static float fused_ce_forward_backward(const float *logits, const int *targets,
                                       float *dlogits, int N, int V) {
    float total_loss = 0.0f;
    float scale = 1.0f / (float)N;

    for (int i = 0; i < N; i++) {
        const float *row = logits + i * V;
        float *drow = dlogits + i * V;

        /* Stable softmax */
        float mx = row[0];
        for (int j = 1; j < V; j++) if (row[j] > mx) mx = row[j];
        float sum = 0.0f;
        for (int j = 0; j < V; j++) {
            drow[j] = expf(row[j] - mx);
            sum += drow[j];
        }
        float inv = 1.0f / sum;
        for (int j = 0; j < V; j++) drow[j] *= inv;

        /* Loss */
        total_loss += -logf(drow[targets[i]] + 1e-10f);

        /* Gradient: (softmax - one_hot) / N */
        for (int j = 0; j < V; j++) drow[j] *= scale;
        drow[targets[i]] -= scale;
    }

    return total_loss / (float)N;
}

/* ---- Numerical gradient check ---- */

static void numerical_grad_check(const float *logits, const int *targets,
                                 const float *analytic_grad, int N, int V) {
    float eps = 1e-4f;
    float *logits_copy = (float *)malloc((size_t)N * V * sizeof(float));
    float *losses = (float *)malloc((size_t)N * sizeof(float));

    float max_diff = 0.0f;
    double sum_diff = 0.0;
    int count = 0;

    /* Check a subset of entries to keep runtime short */
    int check_every = (N * V > 100) ? (N * V / 100) : 1;

    for (int idx = 0; idx < N * V; idx += check_every) {
        memcpy(logits_copy, logits, (size_t)N * V * sizeof(float));

        /* Forward with logits[idx] + eps */
        logits_copy[idx] = logits[idx] + eps;
        float loss_plus = ce_forward_naive(logits_copy, targets, losses, N, V);

        /* Forward with logits[idx] - eps */
        logits_copy[idx] = logits[idx] - eps;
        float loss_minus = ce_forward_naive(logits_copy, targets, losses, N, V);

        float num_grad = (loss_plus - loss_minus) / (2.0f * eps);
        float diff = fabsf(num_grad - analytic_grad[idx]);
        if (diff > max_diff) max_diff = diff;
        sum_diff += (double)diff;
        count++;
    }

    printf("  Checked %d entries (every %d-th)\n", count, check_every);
    printf("  Max |numerical - analytic| = %.2e\n", max_diff);
    printf("  Mean |diff|                = %.2e\n", sum_diff / count);
    printf("  %s\n", max_diff < 1e-3f ? "PASSED" : "FAILED (diff too large)");

    free(logits_copy);
    free(losses);
}

/* ---- Perplexity ---- */

static float compute_perplexity(float mean_loss) {
    return expf(mean_loss);
}

/* ---- main ---- */

int main(void) {
    srand(42);

    printf("=== Cross-Entropy Loss Demo ===\n\n");

    /* --- Part 1: Basic CE computation --- */
    const int V = 10;
    const int N = 4;

    float logits[40];
    int targets[4] = {3, 7, 0, 5};

    /* Synthetic logits */
    for (int i = 0; i < N * V; i++)
        logits[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 4.0f;

    printf("--- Part 1: Softmax + CE Loss ---\n");
    printf("Batch size: %d, Vocab: %d\n", N, V);
    printf("Targets: [%d, %d, %d, %d]\n\n", targets[0], targets[1], targets[2], targets[3]);

    /* Show softmax probabilities for first sample */
    float probs[10];
    softmax(logits, probs, V);
    printf("Sample 0 softmax probabilities:\n");
    for (int j = 0; j < V; j++)
        printf("  p[%d] = %.4f %s\n", j, probs[j], j == targets[0] ? "<-- target" : "");

    /* Log-softmax for first sample */
    float log_probs[10];
    log_softmax(logits, log_probs, V);
    printf("\nSample 0 log-softmax:\n");
    for (int j = 0; j < V; j++)
        printf("  log_p[%d] = %7.4f\n", j, log_probs[j]);

    printf("\nCE loss for sample 0 = -log_p[%d] = %.4f\n",
           targets[0], -log_probs[targets[0]]);

    /* --- Part 2: Batch CE loss --- */
    printf("\n--- Part 2: Batch CE Loss ---\n");
    float losses[4];
    float mean_loss = ce_forward_naive(logits, targets, losses, N, V);
    printf("Per-sample losses: ");
    for (int i = 0; i < N; i++) printf("%.4f ", losses[i]);
    printf("\nMean loss: %.4f\n", mean_loss);
    printf("Perplexity: %.2f\n", compute_perplexity(mean_loss));

    /* --- Part 3: Fused forward + backward --- */
    printf("\n--- Part 3: Fused CE Forward + Backward ---\n");
    float dlogits[40];
    float fused_loss = fused_ce_forward_backward(logits, targets, dlogits, N, V);
    printf("Fused mean loss: %.4f (should match naive: %.4f)\n", fused_loss, mean_loss);

    printf("\nGradients for sample 0:\n");
    for (int j = 0; j < V; j++)
        printf("  dlogits[0][%d] = %+.6f %s\n", j, dlogits[j],
               j == targets[0] ? "<-- target (should be negative)" : "");

    printf("\nGradient interpretation:\n");
    printf("  dlogits = (softmax - one_hot) / N\n");
    printf("  At target position: gradient < 0 (push logit up)\n");
    printf("  At non-target:     gradient > 0 (push logit down)\n");

    /* --- Part 4: Numerical gradient check --- */
    printf("\n--- Part 4: Numerical Gradient Check ---\n");
    numerical_grad_check(logits, targets, dlogits, N, V);

    /* --- Part 5: Effect of loss on gradient magnitude --- */
    printf("\n--- Part 5: Gradient Flow Visualization ---\n");
    printf("Showing how gradient magnitude changes with prediction confidence:\n\n");

    float test_logits[10];
    int test_target = 3;
    float test_dlogits[10];

    float confidences[] = {0.01f, 0.1f, 0.5f, 0.9f, 0.99f};
    for (int c = 0; c < 5; c++) {
        /* Set up logits so target has approximately the desired probability */
        float target_logit = logf(confidences[c] / (1.0f - confidences[c]) * (float)(V - 1));
        for (int j = 0; j < V; j++) test_logits[j] = 0.0f;
        test_logits[test_target] = target_logit;

        int t = test_target;
        float loss = fused_ce_forward_backward(test_logits, &t, test_dlogits, 1, V);
        float actual_prob = expf(-loss);
        printf("  target_prob ~%.2f | loss=%.4f | grad[target]=%+.6f | grad[other]=%+.6f\n",
               actual_prob, loss, test_dlogits[test_target], test_dlogits[0]);
    }

    printf("\nKey insight: gradient magnitude decreases as the model becomes more confident.\n");
    printf("This is correct -- a well-calibrated model needs smaller updates.\n");

    return 0;
}
