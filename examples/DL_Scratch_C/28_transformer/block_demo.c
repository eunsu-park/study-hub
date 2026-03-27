/*
 * block_demo.c - Complete Transformer decoder block
 *
 * Demonstrates:
 *   - Pre-norm architecture: LN -> MHA -> residual -> LN -> FFN -> residual
 *   - LayerNorm, multi-head causal attention, GELU FFN
 *   - Residual connections preserving gradient flow
 *   - Forward pass on synthetic token sequence
 *   - Intermediate tensor shapes at each stage
 *
 * Build:  gcc -std=c11 -Wall -Wextra -O2 -o block_demo block_demo.c -lm
 * Run:    ./block_demo
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define LN_EPS   1e-5f
#define GELU_C1  0.7978845608f
#define GELU_C2  0.044715f

static float randn(void) {
    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    float u2 = (float)rand() / ((float)RAND_MAX + 1.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

static float gelu(float x) {
    float inner = GELU_C1 * (x + GELU_C2 * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

/* ---- LayerNorm ---- */
static void layernorm(const float *X, const float *gamma, const float *beta,
                        float *Y, int M, int C) {
    for (int m = 0; m < M; m++) {
        const float *x = X + (long)m * C;
        float       *y = Y + (long)m * C;
        float mu = 0.0f;
        for (int i = 0; i < C; i++) mu += x[i];
        mu /= C;
        float var = 0.0f;
        for (int i = 0; i < C; i++) { float d = x[i] - mu; var += d * d; }
        var /= C;
        float rs = 1.0f / sqrtf(var + LN_EPS);
        for (int i = 0; i < C; i++)
            y[i] = gamma[i] * (x[i] - mu) * rs + beta[i];
    }
}

/* ---- Linear: Y = X * W^T + b ---- */
static void linear(const float *X, const float *W, const float *b,
                     float *Y, int M, int in_d, int out_d) {
    for (int m = 0; m < M; m++)
    for (int o = 0; o < out_d; o++) {
        float sum = (b != NULL) ? b[o] : 0.0f;
        for (int i = 0; i < in_d; i++)
            sum += X[m * in_d + i] * W[o * in_d + i];
        Y[m * out_d + o] = sum;
    }
}

/* ---- Softmax ---- */
static void softmax_row(float *row, int n) {
    float mx = row[0];
    for (int i = 1; i < n; i++) if (row[i] > mx) mx = row[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { row[i] = expf(row[i] - mx); sum += row[i]; }
    for (int i = 0; i < n; i++) row[i] /= sum;
}

/* ---- Single-head causal attention ---- */
static void attention(const float *Q, const float *K, const float *V,
                        float *attn_w, float *out, int T, int d_head) {
    float scale = 1.0f / sqrtf((float)d_head);
    for (int t = 0; t < T; t++)
    for (int s = 0; s < T; s++) {
        float dot = 0.0f;
        for (int j = 0; j < d_head; j++)
            dot += Q[t * d_head + j] * K[s * d_head + j];
        attn_w[t * T + s] = (s <= t) ? dot * scale : -1e9f;
    }
    for (int t = 0; t < T; t++) {
        softmax_row(attn_w + t * T, t + 1);
        for (int s = t + 1; s < T; s++) attn_w[t * T + s] = 0.0f;
    }
    for (int t = 0; t < T; t++)
    for (int j = 0; j < d_head; j++) {
        float sum = 0.0f;
        for (int s = 0; s < T; s++)
            sum += attn_w[t * T + s] * V[s * d_head + j];
        out[t * d_head + j] = sum;
    }
}

/* ---- Multi-head attention ---- */
static void mha_forward(const float *X, float *output,
                          const float *W_qkv, const float *b_qkv,
                          const float *W_o, const float *b_o,
                          float *attn_weights,
                          int T, int d, int n_heads) {
    int d_head = d / n_heads;

    /* QKV projection */
    float *qkv = malloc((size_t)T * 3 * d * sizeof(float));
    linear(X, W_qkv, b_qkv, qkv, T, d, 3 * d);

    /* Per-head attention */
    float *concat = malloc((size_t)T * d * sizeof(float));
    for (int h = 0; h < n_heads; h++) {
        float *Qh = malloc((size_t)T * d_head * sizeof(float));
        float *Kh = malloc((size_t)T * d_head * sizeof(float));
        float *Vh = malloc((size_t)T * d_head * sizeof(float));
        float *Oh = malloc((size_t)T * d_head * sizeof(float));

        for (int t = 0; t < T; t++) {
            memcpy(Qh + t * d_head, qkv + t * 3 * d + h * d_head,
                   (size_t)d_head * sizeof(float));
            memcpy(Kh + t * d_head, qkv + t * 3 * d + d + h * d_head,
                   (size_t)d_head * sizeof(float));
            memcpy(Vh + t * d_head, qkv + t * 3 * d + 2 * d + h * d_head,
                   (size_t)d_head * sizeof(float));
        }

        float *aw = attn_weights + (long)h * T * T;
        attention(Qh, Kh, Vh, aw, Oh, T, d_head);

        for (int t = 0; t < T; t++)
            memcpy(concat + t * d + h * d_head, Oh + t * d_head,
                   (size_t)d_head * sizeof(float));

        free(Qh); free(Kh); free(Vh); free(Oh);
    }
    free(qkv);

    /* Output projection */
    linear(concat, W_o, b_o, output, T, d, d);
    free(concat);
}

/* ---- FFN: FC(d->4d) -> GELU -> FC(4d->d) ---- */
static void ffn_forward(const float *X, float *output,
                          const float *fc1_w, const float *fc1_b,
                          const float *fc2_w, const float *fc2_b,
                          int T, int d) {
    int d4 = 4 * d;
    float *mid = malloc((size_t)T * d4 * sizeof(float));
    linear(X, fc1_w, fc1_b, mid, T, d, d4);
    for (int i = 0; i < T * d4; i++) mid[i] = gelu(mid[i]);
    linear(mid, fc2_w, fc2_b, output, T, d4, d);
    free(mid);
}

/* ---- Transformer block weights ---- */
typedef struct {
    float *ln1_w, *ln1_b;
    float *qkv_w, *qkv_b;
    float *proj_w, *proj_b;
    float *ln2_w, *ln2_b;
    float *fc1_w, *fc1_b;
    float *fc2_w, *fc2_b;
} BlockWeights;

static void block_init(BlockWeights *bw, int d) {
    int d4 = 4 * d;
    float s = 0.02f;

    bw->ln1_w = malloc((size_t)d * sizeof(float));
    bw->ln1_b = calloc((size_t)d, sizeof(float));
    bw->ln2_w = malloc((size_t)d * sizeof(float));
    bw->ln2_b = calloc((size_t)d, sizeof(float));
    for (int i = 0; i < d; i++) { bw->ln1_w[i] = 1.0f; bw->ln2_w[i] = 1.0f; }

    bw->qkv_w  = malloc((size_t)3 * d * d * sizeof(float));
    bw->qkv_b  = calloc((size_t)3 * d, sizeof(float));
    bw->proj_w = malloc((size_t)d * d * sizeof(float));
    bw->proj_b = calloc((size_t)d, sizeof(float));
    bw->fc1_w  = malloc((size_t)d4 * d * sizeof(float));
    bw->fc1_b  = calloc((size_t)d4, sizeof(float));
    bw->fc2_w  = malloc((size_t)d * d4 * sizeof(float));
    bw->fc2_b  = calloc((size_t)d, sizeof(float));

    for (int i = 0; i < 3 * d * d; i++) bw->qkv_w[i]  = randn() * s;
    for (int i = 0; i < d * d; i++)     bw->proj_w[i] = randn() * s;
    for (int i = 0; i < d4 * d; i++)    bw->fc1_w[i]  = randn() * s;
    for (int i = 0; i < d * d4; i++)    bw->fc2_w[i]  = randn() * s;
}

static void block_free(BlockWeights *bw) {
    free(bw->ln1_w); free(bw->ln1_b); free(bw->ln2_w); free(bw->ln2_b);
    free(bw->qkv_w); free(bw->qkv_b); free(bw->proj_w); free(bw->proj_b);
    free(bw->fc1_w); free(bw->fc1_b); free(bw->fc2_w); free(bw->fc2_b);
}

/* ---- Full pre-norm Transformer block forward ---- */
static void block_forward(BlockWeights *bw, const float *X, float *Y,
                            float *attn_weights, int T, int d, int n_heads) {

    float *ln1_out  = malloc((size_t)T * d * sizeof(float));
    float *attn_out = malloc((size_t)T * d * sizeof(float));
    float *x1       = malloc((size_t)T * d * sizeof(float));
    float *ln2_out  = malloc((size_t)T * d * sizeof(float));
    float *ffn_out  = malloc((size_t)T * d * sizeof(float));

    printf("  [1] LayerNorm1\n");
    layernorm(X, bw->ln1_w, bw->ln1_b, ln1_out, T, d);

    printf("  [2] Multi-Head Attention (causal, %d heads)\n", n_heads);
    mha_forward(ln1_out, attn_out, bw->qkv_w, bw->qkv_b,
                bw->proj_w, bw->proj_b, attn_weights, T, d, n_heads);

    printf("  [3] Residual Add: X1 = X + Attn(LN1(X))\n");
    for (int i = 0; i < T * d; i++) x1[i] = X[i] + attn_out[i];

    printf("  [4] LayerNorm2\n");
    layernorm(x1, bw->ln2_w, bw->ln2_b, ln2_out, T, d);

    printf("  [5] FFN (GELU, hidden=%d)\n", 4 * d);
    ffn_forward(ln2_out, ffn_out, bw->fc1_w, bw->fc1_b,
                bw->fc2_w, bw->fc2_b, T, d);

    printf("  [6] Residual Add: Y = X1 + FFN(LN2(X1))\n");
    for (int i = 0; i < T * d; i++) Y[i] = x1[i] + ffn_out[i];

    free(ln1_out); free(attn_out); free(x1); free(ln2_out); free(ffn_out);
}

/* ---- Print helpers ---- */
static void print_vec(const char *label, const float *v, int n, int show) {
    if (show > n) show = n;
    printf("  %s: [", label);
    for (int i = 0; i < show; i++) {
        printf("%7.4f", v[i]);
        if (i < show - 1) printf(", ");
    }
    if (n > show) printf(", ...");
    printf("]\n");
}

static void vec_stats(const char *label, const float *v, int n) {
    float mn = 0.0f, sq = 0.0f;
    float lo = v[0], hi = v[0];
    for (int i = 0; i < n; i++) {
        mn += v[i]; sq += v[i] * v[i];
        if (v[i] < lo) lo = v[i];
        if (v[i] > hi) hi = v[i];
    }
    mn /= n; sq = sqrtf(sq / n - mn * mn);
    printf("  %s: mean=%.4f  std=%.4f  range=[%.4f, %.4f]\n", label, mn, sq, lo, hi);
}

int main(void) {
    srand(42);

    int T = 6;
    int d = 16;
    int n_heads = 4;
    int d_head = d / n_heads;

    printf("=== Transformer Decoder Block Demo ===\n\n");
    printf("Architecture: Pre-norm (LN->Attn->+->LN->FFN->+)\n");
    printf("Config: T=%d, d=%d, n_heads=%d, d_head=%d, d_ffn=%d\n\n",
           T, d, n_heads, d_head, 4 * d);

    /* Input */
    float *X = malloc((size_t)T * d * sizeof(float));
    for (int i = 0; i < T * d; i++) X[i] = randn() * 0.5f;

    printf("=== Input ===\n");
    for (int t = 0; t < T; t++) {
        char label[32];
        snprintf(label, sizeof(label), "pos %d", t);
        print_vec(label, X + t * d, d, 8);
    }
    vec_stats("X overall", X, T * d);
    printf("\n");

    /* Initialize block */
    BlockWeights bw;
    block_init(&bw, d);

    /* Forward pass */
    float *Y = malloc((size_t)T * d * sizeof(float));
    float *attn_w = malloc((size_t)n_heads * T * T * sizeof(float));

    printf("=== Block Forward Pass ===\n");
    block_forward(&bw, X, Y, attn_w, T, d, n_heads);
    printf("\n");

    /* Output */
    printf("=== Output ===\n");
    for (int t = 0; t < T; t++) {
        char label[32];
        snprintf(label, sizeof(label), "pos %d", t);
        print_vec(label, Y + t * d, d, 8);
    }
    vec_stats("Y overall", Y, T * d);
    printf("\n");

    /* Residual analysis */
    printf("=== Residual Analysis ===\n");
    float residual_norm = 0.0f, output_norm = 0.0f, input_norm = 0.0f;
    for (int i = 0; i < T * d; i++) {
        float diff = Y[i] - X[i];
        residual_norm += diff * diff;
        output_norm   += Y[i] * Y[i];
        input_norm    += X[i] * X[i];
    }
    printf("  ||X||     = %.4f\n", sqrtf(input_norm));
    printf("  ||Y||     = %.4f\n", sqrtf(output_norm));
    printf("  ||Y - X|| = %.4f  (change from residual)\n", sqrtf(residual_norm));
    printf("  Ratio:      %.4f  (residual adds small perturbation)\n\n",
           sqrtf(residual_norm) / sqrtf(input_norm));

    /* Attention weights visualization */
    printf("=== Attention Weights (Head 0) ===\n");
    for (int t = 0; t < T; t++) {
        printf("  pos %d: [", t);
        for (int s = 0; s < T; s++) {
            printf("%.3f", attn_w[0 * T * T + t * T + s]);
            if (s < T - 1) printf(", ");
        }
        printf("]\n");
    }
    printf("  (upper triangle = 0 due to causal mask)\n\n");

    /* Stacking blocks */
    printf("=== Stacking Two Blocks ===\n\n");
    BlockWeights bw2;
    block_init(&bw2, d);

    float *Y2 = malloc((size_t)T * d * sizeof(float));
    float *aw2 = malloc((size_t)n_heads * T * T * sizeof(float));

    printf("Block 1:\n");
    block_forward(&bw, X, Y, attn_w, T, d, n_heads);
    printf("\nBlock 2:\n");
    block_forward(&bw2, Y, Y2, aw2, T, d, n_heads);
    printf("\n");

    printf("After 2 blocks:\n");
    for (int t = 0; t < T; t++) {
        char label[32];
        snprintf(label, sizeof(label), "pos %d", t);
        print_vec(label, Y2 + t * d, d, 8);
    }
    vec_stats("Y2 overall", Y2, T * d);

    /* Parameter count */
    printf("\n=== Parameter Count Per Block ===\n");
    int params = 2 * d + 3 * d * d + 3 * d + d * d + d + 2 * d + 4 * d * d + 4 * d + d * 4 * d + d;
    printf("  LN1:    %d (gamma + beta)\n", 2 * d);
    printf("  QKV:    %d (W + b)\n", 3 * d * d + 3 * d);
    printf("  Proj:   %d (W + b)\n", d * d + d);
    printf("  LN2:    %d (gamma + beta)\n", 2 * d);
    printf("  FC1:    %d (W + b)\n", 4 * d * d + 4 * d);
    printf("  FC2:    %d (W + b)\n", d * 4 * d + d);
    printf("  Total:  %d\n", params);
    printf("\n  GPT-2 small (d=768, 12 layers): ~%.1fM per block, ~%.1fM total\n",
           (float)(2*768 + 3*768*768 + 3*768 + 768*768 + 768 + 2*768 + 4*768*768 + 4*768 + 768*4*768 + 768) / 1e6,
           12.0f * (2*768 + 3*768*768 + 3*768 + 768*768 + 768 + 2*768 + 4*768*768 + 4*768 + 768*4*768 + 768) / 1e6);

    /* Cleanup */
    block_free(&bw); block_free(&bw2);
    free(X); free(Y); free(Y2); free(attn_w); free(aw2);

    printf("\nDone.\n");
    return 0;
}
