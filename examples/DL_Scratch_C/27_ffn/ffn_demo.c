/*
 * ffn_demo.c - Feed-forward network with SwiGLU activation
 *
 * Demonstrates:
 *   - GPT-2 FFN: FC -> GELU -> FC
 *   - Llama FFN: SiLU(gate) * up -> down (SwiGLU)
 *   - GELU and SiLU activation functions
 *   - Gate/up/down projection pipeline
 *   - Parameter count comparison
 *
 * Build:  gcc -std=c11 -Wall -Wextra -O2 -o ffn_demo ffn_demo.c -lm
 * Run:    ./ffn_demo
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define SQRT_2_OVER_PI 0.7978845608f
#define GELU_COEF      0.044715f

static float randn(void) {
    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    float u2 = (float)rand() / ((float)RAND_MAX + 1.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

/* ---- Activation functions ---- */
static float gelu(float x) {
    float inner = SQRT_2_OVER_PI * (x + GELU_COEF * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

static float silu(float x) {
    return x / (1.0f + expf(-x));
}

static float relu(float x) { return x > 0.0f ? x : 0.0f; }

/* ---- Linear: Y[M, out] = X[M, in] * W[out, in]^T + b[out] ---- */
static void linear_forward(const float *X, const float *W, const float *b,
                             float *Y, int M, int in_d, int out_d) {
    for (int m = 0; m < M; m++)
    for (int o = 0; o < out_d; o++) {
        float sum = (b != NULL) ? b[o] : 0.0f;
        for (int i = 0; i < in_d; i++)
            sum += X[m * in_d + i] * W[o * in_d + i];
        Y[m * out_d + o] = sum;
    }
}

/* ---- GPT-2 FFN: FC(d -> 4d) -> GELU -> FC(4d -> d) ---- */
static void gpt2_ffn_forward(const float *input, float *output,
                               const float *fc1_w, const float *fc1_b,
                               const float *fc2_w, const float *fc2_b,
                               float *mid_buf,
                               int M, int d) {
    int d4 = 4 * d;

    /* FC1: [M, d] -> [M, 4d] */
    linear_forward(input, fc1_w, fc1_b, mid_buf, M, d, d4);

    /* GELU in-place */
    for (int i = 0; i < M * d4; i++) mid_buf[i] = gelu(mid_buf[i]);

    /* FC2: [M, 4d] -> [M, d] */
    linear_forward(mid_buf, fc2_w, fc2_b, output, M, d4, d);
}

/* ---- Llama FFN: SwiGLU gated architecture ---- */
static void llama_ffn_forward(const float *input, float *output,
                                const float *gate_w,  /* [d_ffn, d] */
                                const float *up_w,    /* [d_ffn, d] */
                                const float *down_w,  /* [d, d_ffn] */
                                float *gate_buf, float *up_buf,
                                int M, int d, int d_ffn) {

    /* Gate branch: W_gate * x */
    linear_forward(input, gate_w, NULL, gate_buf, M, d, d_ffn);

    /* Up branch: W_up * x */
    linear_forward(input, up_w, NULL, up_buf, M, d, d_ffn);

    /* SwiGLU: SiLU(gate) * up */
    for (int i = 0; i < M * d_ffn; i++)
        gate_buf[i] = silu(gate_buf[i]) * up_buf[i];

    /* Down projection: W_down * swiglu_out */
    linear_forward(gate_buf, down_w, NULL, output, M, d_ffn, d);
}

/* ---- Print vector ---- */
static void print_vec(const char *label, const float *v, int n, int show) {
    if (show > n) show = n;
    printf("  %s: [", label);
    for (int i = 0; i < show; i++) {
        printf("%7.4f", v[i]);
        if (i < show - 1) printf(", ");
    }
    if (n > show) printf(", ... (%d total)", n);
    printf("]\n");
}

int main(void) {
    srand(42);

    int M = 4;      /* number of tokens */
    int d = 8;      /* model dimension */
    int d4 = 4 * d; /* GPT-2 FFN hidden dim */
    int d_ffn = 22; /* Llama d_ffn ~ 8/3 * d, rounded */

    printf("=== Feed-Forward Network Demo ===\n\n");
    printf("Config: M=%d tokens, d=%d, d4=%d (GPT-2), d_ffn=%d (Llama)\n\n", M, d, d4, d_ffn);

    /* ---- Activation function comparison ---- */
    printf("=== Activation Function Comparison ===\n\n");
    printf("  %-8s  %-10s %-10s %-10s\n", "x", "ReLU", "GELU", "SiLU");
    printf("  %-8s  %-10s %-10s %-10s\n", "------", "--------", "--------", "--------");
    float test_x[] = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f};
    for (int i = 0; i < 7; i++) {
        float x = test_x[i];
        printf("  %6.2f    %8.4f   %8.4f   %8.4f\n",
               x, relu(x), gelu(x), silu(x));
    }
    printf("\n  GELU: smoother than ReLU, slight negative region\n");
    printf("  SiLU: x * sigmoid(x), also smooth, used with gating\n\n");

    /* Input */
    float *input = malloc((size_t)M * d * sizeof(float));
    for (int i = 0; i < M * d; i++) input[i] = randn() * 0.5f;

    printf("=== Input ===\n");
    for (int m = 0; m < M; m++) {
        char label[32];
        snprintf(label, sizeof(label), "token %d", m);
        print_vec(label, input + m * d, d, d);
    }
    printf("\n");

    /* ---- GPT-2 FFN ---- */
    printf("=== GPT-2 FFN: FC(d->4d) -> GELU -> FC(4d->d) ===\n\n");

    float *fc1_w = malloc((size_t)d4 * d * sizeof(float));
    float *fc1_b = calloc((size_t)d4, sizeof(float));
    float *fc2_w = malloc((size_t)d * d4 * sizeof(float));
    float *fc2_b = calloc((size_t)d, sizeof(float));
    for (int i = 0; i < d4 * d; i++) fc1_w[i] = randn() * 0.02f;
    for (int i = 0; i < d * d4; i++) fc2_w[i] = randn() * 0.02f;

    float *mid_buf = malloc((size_t)M * d4 * sizeof(float));
    float *gpt2_out = malloc((size_t)M * d * sizeof(float));

    /* Show intermediate values */
    linear_forward(input, fc1_w, fc1_b, mid_buf, M, d, d4);
    printf("  After FC1 (before GELU):\n");
    print_vec("  token 0", mid_buf, d4, 8);

    for (int i = 0; i < M * d4; i++) mid_buf[i] = gelu(mid_buf[i]);
    printf("  After GELU:\n");
    print_vec("  token 0", mid_buf, d4, 8);

    /* Full forward */
    gpt2_ffn_forward(input, gpt2_out, fc1_w, fc1_b, fc2_w, fc2_b, mid_buf, M, d);
    printf("  Final output:\n");
    for (int m = 0; m < M; m++) {
        char label[32];
        snprintf(label, sizeof(label), "  token %d", m);
        print_vec(label, gpt2_out + m * d, d, d);
    }
    printf("\n");

    /* ---- Llama SwiGLU FFN ---- */
    printf("=== Llama FFN: SiLU(gate) * up -> down (SwiGLU) ===\n\n");

    float *gate_w = malloc((size_t)d_ffn * d * sizeof(float));
    float *up_w   = malloc((size_t)d_ffn * d * sizeof(float));
    float *down_w = malloc((size_t)d * d_ffn * sizeof(float));
    for (int i = 0; i < d_ffn * d; i++) gate_w[i] = randn() * 0.02f;
    for (int i = 0; i < d_ffn * d; i++) up_w[i]   = randn() * 0.02f;
    for (int i = 0; i < d * d_ffn; i++) down_w[i]  = randn() * 0.02f;

    float *gate_buf = malloc((size_t)M * d_ffn * sizeof(float));
    float *up_buf   = malloc((size_t)M * d_ffn * sizeof(float));
    float *llama_out = malloc((size_t)M * d * sizeof(float));

    /* Show pipeline stages */
    linear_forward(input, gate_w, NULL, gate_buf, M, d, d_ffn);
    printf("  Gate projection (before SiLU):\n");
    print_vec("  token 0", gate_buf, d_ffn, 8);

    float *gate_activated = malloc((size_t)M * d_ffn * sizeof(float));
    for (int i = 0; i < M * d_ffn; i++) gate_activated[i] = silu(gate_buf[i]);
    printf("  After SiLU(gate):\n");
    print_vec("  token 0", gate_activated, d_ffn, 8);

    linear_forward(input, up_w, NULL, up_buf, M, d, d_ffn);
    printf("  Up projection:\n");
    print_vec("  token 0", up_buf, d_ffn, 8);

    printf("  SiLU(gate) * up (element-wise):\n");
    float *swiglu_mid = malloc((size_t)M * d_ffn * sizeof(float));
    for (int i = 0; i < M * d_ffn; i++)
        swiglu_mid[i] = gate_activated[i] * up_buf[i];
    print_vec("  token 0", swiglu_mid, d_ffn, 8);

    /* Full forward */
    llama_ffn_forward(input, llama_out, gate_w, up_w, down_w,
                       gate_buf, up_buf, M, d, d_ffn);
    printf("  Final output (after down projection):\n");
    for (int m = 0; m < M; m++) {
        char label[32];
        snprintf(label, sizeof(label), "  token %d", m);
        print_vec(label, llama_out + m * d, d, d);
    }
    printf("\n");

    /* ---- Parameter count comparison ---- */
    printf("=== Parameter Count Comparison ===\n\n");
    int gpt2_params = d * d4 + d4 + d4 * d + d;  /* fc1_w,b + fc2_w,b */
    int llama_params = d_ffn * d + d_ffn * d + d * d_ffn;  /* gate_w + up_w + down_w */

    printf("  %-20s  %-15s  %-15s\n", "", "GPT-2 FFN", "Llama SwiGLU");
    printf("  %-20s  %-15s  %-15s\n", "---", "---", "---");
    printf("  %-20s  d x 4d = %-6d  d x d_ffn = %-6d\n",
           "Up projection", d * d4, d_ffn * d);
    printf("  %-20s  %-15s  d x d_ffn = %-6d\n",
           "Gate projection", "N/A", d_ffn * d);
    printf("  %-20s  4d x d = %-6d  d_ffn x d = %-6d\n",
           "Down projection", d4 * d, d * d_ffn);
    printf("  %-20s  %-15d  %-15d\n", "Total params", gpt2_params, llama_params);

    printf("\n  At GPT-2 scale (d=768):\n");
    printf("    GPT-2: 2 x 768 x 3072 + 3072 + 768 = %.1fM\n",
           (2.0 * 768 * 3072 + 3072 + 768) / 1e6);
    printf("    Llama (d_ffn=5461): 3 x 768 x 5461 = %.1fM\n",
           (3.0 * 768 * 5461) / 1e6);
    printf("    SwiGLU uses 3 weight matrices but d_ffn = 2/3 * 4d\n");
    printf("    -> roughly same total parameters, but better accuracy\n");

    printf("\n  FFN compute share: ~73%% of total Transformer FLOPs\n");

    /* Cleanup */
    free(input); free(fc1_w); free(fc1_b); free(fc2_w); free(fc2_b);
    free(mid_buf); free(gpt2_out);
    free(gate_w); free(up_w); free(down_w);
    free(gate_buf); free(up_buf); free(llama_out);
    free(gate_activated); free(swiglu_mid);

    printf("\nDone.\n");
    return 0;
}
