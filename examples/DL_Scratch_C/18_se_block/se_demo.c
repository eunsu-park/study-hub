/*
 * se_demo.c - Squeeze-and-Excitation (SE) block
 *
 * Demonstrates:
 *   - SE block: GlobalAvgPool -> FC -> ReLU -> FC -> Sigmoid -> Scale
 *   - Channel attention weights (dynamic per-input recalibration)
 *   - Integration with a residual block (SE-ResNet style)
 *   - Parameter overhead analysis
 *   - Comparison of input vs output after SE reweighting
 *
 * Build:  gcc -std=c11 -Wall -Wextra -O2 -o se_demo se_demo.c -lm
 * Run:    ./se_demo
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NCHW(ptr, N, C, H, W, n, c, h, w) \
    ((ptr)[(n)*(C)*(H)*(W) + (c)*(H)*(W) + (h)*(W) + (w)])

static float randn(void) {
    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    float u2 = (float)rand() / ((float)RAND_MAX + 1.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

/* ---- Global Average Pooling: [N,C,H,W] -> [N,C] ---- */
static void gap_forward(const float *X, float *out,
                         int N, int C, int H, int W) {
    int spatial = H * W;
    for (int n = 0; n < N; n++)
    for (int c = 0; c < C; c++) {
        float sum = 0.0f;
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++)
            sum += NCHW(X, N, C, H, W, n, c, h, w);
        out[n * C + c] = sum / spatial;
    }
}

/* ---- FC layer: Y[N, out] = X[N, in] x W[in, out] + b[out] ---- */
static void fc_forward(const float *X, const float *W, const float *b,
                        float *Y, int N, int in_d, int out_d) {
    for (int n = 0; n < N; n++)
    for (int o = 0; o < out_d; o++) {
        float sum = b[o];
        for (int i = 0; i < in_d; i++)
            sum += X[n * in_d + i] * W[i * out_d + o];
        Y[n * out_d + o] = sum;
    }
}

/* ---- ReLU ---- */
static void relu(float *X, int size) {
    for (int i = 0; i < size; i++)
        if (X[i] < 0.0f) X[i] = 0.0f;
}

/* ---- Sigmoid ---- */
static void sigmoid(float *X, int size) {
    for (int i = 0; i < size; i++)
        X[i] = 1.0f / (1.0f + expf(-X[i]));
}

/* ---- SE Block ---- */
typedef struct {
    int C;
    int C_r;    /* C / reduction */
    float *fc1_w;   /* [C, C_r] stored as [C x C_r] row-major */
    float *fc1_b;   /* [C_r] */
    float *fc2_w;   /* [C_r, C] */
    float *fc2_b;   /* [C] */
} SEBlock;

static SEBlock *se_create(int C, int reduction) {
    SEBlock *se = (SEBlock *)malloc(sizeof(SEBlock));
    se->C = C;
    se->C_r = C / reduction;
    if (se->C_r < 1) se->C_r = 1;

    se->fc1_w = (float *)malloc((size_t)C * se->C_r * sizeof(float));
    se->fc1_b = (float *)calloc((size_t)se->C_r, sizeof(float));
    se->fc2_w = (float *)malloc((size_t)se->C_r * C * sizeof(float));
    se->fc2_b = (float *)calloc((size_t)C, sizeof(float));

    /* He initialization */
    float std1 = sqrtf(2.0f / C);
    for (int i = 0; i < C * se->C_r; i++) se->fc1_w[i] = randn() * std1;
    float std2 = sqrtf(2.0f / se->C_r);
    for (int i = 0; i < se->C_r * C; i++) se->fc2_w[i] = randn() * std2;

    return se;
}

static void se_free(SEBlock *se) {
    free(se->fc1_w); free(se->fc1_b);
    free(se->fc2_w); free(se->fc2_b);
    free(se);
}

/*
 * SE forward: X[N,C,H,W] -> Y[N,C,H,W] (channel-reweighted)
 * Also outputs attention weights in attn[N,C]
 */
static void se_forward(
    SEBlock *se,
    const float *X, float *Y,
    float *gap,     /* [N, C] */
    float *fc1,     /* [N, C_r] */
    float *attn,    /* [N, C] - attention weights (sigmoid output) */
    int N, int C, int H, int W) {

    /* 1. Squeeze: Global Average Pooling */
    gap_forward(X, gap, N, C, H, W);

    /* 2. Excitation: FC1(C -> C_r) -> ReLU -> FC2(C_r -> C) -> Sigmoid */
    fc_forward(gap, se->fc1_w, se->fc1_b, fc1, N, C, se->C_r);
    relu(fc1, N * se->C_r);
    fc_forward(fc1, se->fc2_w, se->fc2_b, attn, N, se->C_r, C);
    sigmoid(attn, N * C);

    /* 3. Scale: broadcast multiply attention weights */
    memcpy(Y, X, (size_t)N * C * H * W * sizeof(float));
    for (int n = 0; n < N; n++)
    for (int c = 0; c < C; c++) {
        float scale = attn[n * C + c];
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++)
            NCHW(Y, N, C, H, W, n, c, h, w) *= scale;
    }
}

/* ---- Channel energy: L2 norm per channel ---- */
static void channel_energy(const float *X, float *energy,
                            int N, int C, int H, int W) {
    for (int n = 0; n < N; n++)
    for (int c = 0; c < C; c++) {
        float sum = 0.0f;
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++) {
            float v = NCHW(X, N, C, H, W, n, c, h, w);
            sum += v * v;
        }
        energy[n * C + c] = sqrtf(sum / (H * W));
    }
}

int main(void) {
    srand(42);
    printf("=== Squeeze-and-Excitation Block Demo ===\n\n");

    int N = 2, C = 16, H = 8, W = 8;
    int reduction = 4;  /* r = 4, so C_r = 4 */
    int total = N * C * H * W;

    /* Generate synthetic input with varying channel importance */
    float *X = (float *)malloc((size_t)total * sizeof(float));
    for (int n = 0; n < N; n++)
    for (int c = 0; c < C; c++) {
        /* Make some channels more "important" (higher magnitude) */
        float ch_scale = (c % 4 == 0) ? 3.0f : 0.5f;
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++)
            NCHW(X, N, C, H, W, n, c, h, w) = randn() * ch_scale;
    }

    /* Create SE block */
    SEBlock *se = se_create(C, reduction);
    printf("SE Block Configuration:\n");
    printf("  Channels (C):      %d\n", C);
    printf("  Reduction ratio:   %d\n", reduction);
    printf("  Bottleneck (C/r):  %d\n", se->C_r);
    printf("  FC1 params:        %d  (%dx%d)\n", C * se->C_r, C, se->C_r);
    printf("  FC2 params:        %d  (%dx%d)\n", se->C_r * C, se->C_r, C);
    printf("  Total SE params:   %d\n\n", 2 * C * se->C_r + C + se->C_r);

    /* Forward pass */
    float *Y    = (float *)malloc((size_t)total * sizeof(float));
    float *gap  = (float *)malloc((size_t)N * C * sizeof(float));
    float *fc1  = (float *)malloc((size_t)N * se->C_r * sizeof(float));
    float *attn = (float *)malloc((size_t)N * C * sizeof(float));

    se_forward(se, X, Y, gap, fc1, attn, N, C, H, W);

    /* Display results */
    printf("--- Squeeze (Global Average Pooling) ---\n");
    for (int n = 0; n < N; n++) {
        printf("Sample %d GAP: ", n);
        for (int c = 0; c < C; c++)
            printf("%6.3f ", gap[n * C + c]);
        printf("\n");
    }

    printf("\n--- Excitation (Channel Attention Weights) ---\n");
    for (int n = 0; n < N; n++) {
        printf("Sample %d attention: ", n);
        for (int c = 0; c < C; c++)
            printf("%5.3f ", attn[n * C + c]);
        printf("\n");
    }

    printf("\n--- Channel Energy (RMS per channel) ---\n");
    float *energy_in  = (float *)malloc((size_t)N * C * sizeof(float));
    float *energy_out = (float *)malloc((size_t)N * C * sizeof(float));
    channel_energy(X, energy_in, N, C, H, W);
    channel_energy(Y, energy_out, N, C, H, W);

    printf("%-8s %-12s %-12s %-12s %-8s\n",
           "Channel", "Input RMS", "Attn Weight", "Output RMS", "Ratio");
    for (int c = 0; c < C; c++) {
        printf("ch%-6d %10.4f   %10.4f   %10.4f   %6.3f\n",
               c, energy_in[c], attn[c], energy_out[c],
               energy_out[c] / (energy_in[c] + 1e-8f));
    }

    /* ---- SE-ResNet Integration ---- */
    printf("\n--- SE-ResNet Integration ---\n");
    printf("Standard ResNet block:\n");
    printf("  x -> Conv -> BN -> ReLU -> Conv -> BN -> (+x) -> ReLU\n\n");
    printf("SE-ResNet block:\n");
    printf("  x -> Conv -> BN -> ReLU -> Conv -> BN -> [SE] -> (+x) -> ReLU\n");
    printf("  SE is inserted before the shortcut addition.\n\n");

    /* ---- Parameter overhead analysis ---- */
    printf("--- Parameter Overhead Analysis ---\n");
    struct {
        const char *name;
        int C;
        int r;
    } configs[] = {
        {"SE-ResNet-18 (C=64)",  64, 16},
        {"SE-ResNet-50 (C=256)", 256, 16},
        {"SE-ResNet-50 (C=512)", 512, 16},
        {"SE-ResNet-50 (C=2048)", 2048, 16},
        {"SE-MobileNet (C=32)",   32, 4},
    };
    int n_configs = 5;

    printf("%-28s %-8s %-8s %-10s\n", "Config", "C", "C/r", "SE params");
    for (int i = 0; i < n_configs; i++) {
        int c = configs[i].C;
        int cr = c / configs[i].r;
        int params = 2 * c * cr + c + cr;
        printf("%-28s %-8d %-8d %-10d\n", configs[i].name, c, cr, params);
    }

    printf("\nResNet-50 total: ~25M params\n");
    printf("SE-ResNet-50 adds: ~2.5M params (10%% overhead)\n");
    printf("Accuracy gain: +1.5%% top-1 on ImageNet (76.1%% -> 77.6%%)\n");

    /* ---- Demo with different inputs ---- */
    printf("\n--- Different Inputs -> Different Attention ---\n");
    float *X2 = (float *)malloc((size_t)total * sizeof(float));
    float *Y2 = (float *)malloc((size_t)total * sizeof(float));
    float *gap2 = (float *)malloc((size_t)N * C * sizeof(float));
    float *fc1_2 = (float *)malloc((size_t)N * se->C_r * sizeof(float));
    float *attn2 = (float *)malloc((size_t)N * C * sizeof(float));

    /* Create input with reversed channel importance */
    for (int n = 0; n < N; n++)
    for (int c = 0; c < C; c++) {
        float ch_scale = (c % 4 == 3) ? 3.0f : 0.5f;  /* opposite pattern */
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++)
            NCHW(X2, N, C, H, W, n, c, h, w) = randn() * ch_scale;
    }

    se_forward(se, X2, Y2, gap2, fc1_2, attn2, N, C, H, W);

    printf("Input 1 attention (channels 0-7): ");
    for (int c = 0; c < 8; c++) printf("%.3f ", attn[c]);
    printf("\n");
    printf("Input 2 attention (channels 0-7): ");
    for (int c = 0; c < 8; c++) printf("%.3f ", attn2[c]);
    printf("\n");
    printf("SE block produces different channel weights for different inputs.\n");

    /* Cleanup */
    free(X); free(Y); free(gap); free(fc1); free(attn);
    free(energy_in); free(energy_out);
    free(X2); free(Y2); free(gap2); free(fc1_2); free(attn2);
    se_free(se);

    printf("\n=== SE Block Demo Complete ===\n");
    return 0;
}
