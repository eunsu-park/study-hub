/*
 * efficientnet_demo.c - MBConv block with Squeeze-and-Excitation
 *
 * Demonstrates:
 *   - MBConv block: expand (pointwise) -> depthwise conv -> SE -> project
 *   - SiLU (Swish) activation
 *   - Identity residual when input/output shapes match
 *   - Feature map dimensions through each stage
 *   - Compound scaling for EfficientNet-B0..B7
 *
 * Build:  gcc -std=c11 -Wall -Wextra -O2 -o efficientnet_demo efficientnet_demo.c -lm
 * Run:    ./efficientnet_demo
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- Helpers ---- */

static float randn(void) {
    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    float u2 = (float)rand() / ((float)RAND_MAX + 1.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

static float silu(float x) {
    return x / (1.0f + expf(-x));
}

static void silu_inplace(float *x, int n) {
    for (int i = 0; i < n; i++) x[i] = silu(x[i]);
}

static float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/* ---- Pointwise Conv (1x1): [C_in, H, W] -> [C_out, H, W] ---- */
static void pointwise_conv(const float *X, const float *Wt,
                            float *Y, int C_in, int C_out, int H, int Wi) {
    int spatial = H * Wi;
    for (int co = 0; co < C_out; co++)
    for (int hw = 0; hw < spatial; hw++) {
        float sum = 0.0f;
        for (int ci = 0; ci < C_in; ci++)
            sum += Wt[co * C_in + ci] * X[ci * spatial + hw];
        Y[co * spatial + hw] = sum;
    }
}

/* ---- Depthwise Conv (KxK): [C, H, W] -> [C, OH, OW] ---- */
static void depthwise_conv(const float *X, const float *Wt,
                            float *Y, int C, int H, int Wi,
                            int K, int stride, int *OH_out, int *OW_out) {
    int pad = K / 2;
    int OH = (H + 2 * pad - K) / stride + 1;
    int OW = (Wi + 2 * pad - K) / stride + 1;
    *OH_out = OH;
    *OW_out = OW;

    for (int c = 0; c < C; c++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        float sum = 0.0f;
        for (int kh = 0; kh < K; kh++)
        for (int kw = 0; kw < K; kw++) {
            int ih = oh * stride - pad + kh;
            int iw = ow * stride - pad + kw;
            if (ih >= 0 && ih < H && iw >= 0 && iw < Wi)
                sum += X[c * H * Wi + ih * Wi + iw] * Wt[c * K * K + kh * K + kw];
        }
        Y[c * OH * OW + oh * OW + ow] = sum;
    }
}

/* ---- SE block: GAP -> FC_reduce -> ReLU -> FC_expand -> Sigmoid -> Scale ---- */
static void se_forward(const float *X, float *Y,
                        const float *fc1_w, const float *fc2_w,
                        int C, int H, int W, int C_sq) {
    int spatial = H * W;

    /* Global Average Pooling */
    float *gap = calloc((size_t)C, sizeof(float));
    for (int c = 0; c < C; c++) {
        float s = 0.0f;
        for (int hw = 0; hw < spatial; hw++)
            s += X[c * spatial + hw];
        gap[c] = s / spatial;
    }

    /* FC reduce: [C] -> [C_sq] + ReLU */
    float *fc1 = calloc((size_t)C_sq, sizeof(float));
    for (int j = 0; j < C_sq; j++) {
        float s = 0.0f;
        for (int i = 0; i < C; i++) s += gap[i] * fc1_w[j * C + i];
        fc1[j] = fmaxf(0.0f, s);  /* ReLU */
    }

    /* FC expand: [C_sq] -> [C] + Sigmoid */
    float *scale = calloc((size_t)C, sizeof(float));
    for (int j = 0; j < C; j++) {
        float s = 0.0f;
        for (int i = 0; i < C_sq; i++) s += fc1[i] * fc2_w[j * C_sq + i];
        scale[j] = sigmoid(s);
    }

    /* Channel-wise scale */
    for (int c = 0; c < C; c++)
    for (int hw = 0; hw < spatial; hw++)
        Y[c * spatial + hw] = X[c * spatial + hw] * scale[c];

    free(gap); free(fc1); free(scale);
}

/* ---- MBConv Block ---- */
typedef struct {
    int C_in, C_mid, C_out, K, stride;
    float *pw1_w;   /* expand PW: [C_mid, C_in] */
    float *dw_w;    /* depthwise:  [C_mid, K, K] */
    float *se_fc1;  /* SE reduce:  [C_sq, C_mid] */
    float *se_fc2;  /* SE expand:  [C_mid, C_sq] */
    float *pw2_w;   /* project PW: [C_out, C_mid] */
    int use_skip;   /* 1 if stride==1 && C_in==C_out */
} MBConv;

static void mbconv_init(MBConv *blk, int C_in, int C_out, int K,
                          int stride, int expand_ratio) {
    blk->C_in  = C_in;
    blk->C_mid = C_in * expand_ratio;
    blk->C_out = C_out;
    blk->K      = K;
    blk->stride = stride;
    blk->use_skip = (stride == 1 && C_in == C_out) ? 1 : 0;

    int Cm = blk->C_mid;
    int C_sq = (Cm > 4) ? Cm / 4 : 1;

    blk->pw1_w  = malloc((size_t)Cm * C_in * sizeof(float));
    blk->dw_w   = malloc((size_t)Cm * K * K * sizeof(float));
    blk->se_fc1 = malloc((size_t)C_sq * Cm * sizeof(float));
    blk->se_fc2 = malloc((size_t)Cm * C_sq * sizeof(float));
    blk->pw2_w  = malloc((size_t)C_out * Cm * sizeof(float));

    float s = 0.02f;
    for (int i = 0; i < Cm * C_in; i++) blk->pw1_w[i] = randn() * s;
    for (int i = 0; i < Cm * K * K; i++) blk->dw_w[i] = randn() * s;
    for (int i = 0; i < C_sq * Cm; i++) blk->se_fc1[i] = randn() * s;
    for (int i = 0; i < Cm * C_sq; i++) blk->se_fc2[i] = randn() * s;
    for (int i = 0; i < C_out * Cm; i++) blk->pw2_w[i] = randn() * s;
}

static void mbconv_free(MBConv *blk) {
    free(blk->pw1_w); free(blk->dw_w);
    free(blk->se_fc1); free(blk->se_fc2);
    free(blk->pw2_w);
}

static float *mbconv_forward(MBConv *blk, const float *X, int H, int W,
                               int *OH_out, int *OW_out) {
    int Cm = blk->C_mid;
    int C_sq = (Cm > 4) ? Cm / 4 : 1;

    /* 1. Expansion pointwise (skip if expand_ratio == 1) */
    float *expanded;
    if (Cm != blk->C_in) {
        expanded = malloc((size_t)Cm * H * W * sizeof(float));
        pointwise_conv(X, blk->pw1_w, expanded, blk->C_in, Cm, H, W);
        silu_inplace(expanded, Cm * H * W);
        printf("    [Expand]    %d -> %d channels  (%dx%d)\n", blk->C_in, Cm, H, W);
    } else {
        expanded = malloc((size_t)Cm * H * W * sizeof(float));
        memcpy(expanded, X, (size_t)Cm * H * W * sizeof(float));
        printf("    [NoExpand]  %d channels  (%dx%d)\n", Cm, H, W);
    }

    /* 2. Depthwise convolution */
    int OH, OW;
    float *dw_out = malloc((size_t)Cm * H * W * sizeof(float));  /* oversize OK */
    depthwise_conv(expanded, blk->dw_w, dw_out, Cm, H, W, blk->K, blk->stride, &OH, &OW);
    silu_inplace(dw_out, Cm * OH * OW);
    printf("    [DW %dx%d]   %d ch, stride %d  -> %dx%d\n",
           blk->K, blk->K, Cm, blk->stride, OH, OW);
    free(expanded);

    /* 3. Squeeze-and-Excitation */
    float *se_out = malloc((size_t)Cm * OH * OW * sizeof(float));
    se_forward(dw_out, se_out, blk->se_fc1, blk->se_fc2, Cm, OH, OW, C_sq);
    printf("    [SE]        reduce %d -> %d -> %d\n", Cm, C_sq, Cm);
    free(dw_out);

    /* 4. Projection pointwise (no activation) */
    float *proj_out = malloc((size_t)blk->C_out * OH * OW * sizeof(float));
    pointwise_conv(se_out, blk->pw2_w, proj_out, Cm, blk->C_out, OH, OW);
    printf("    [Project]   %d -> %d channels\n", Cm, blk->C_out);
    free(se_out);

    /* 5. Residual skip */
    if (blk->use_skip) {
        int sz = blk->C_out * OH * OW;
        for (int i = 0; i < sz; i++) proj_out[i] += X[i];
        printf("    [Residual]  identity skip connection added\n");
    } else {
        printf("    [NoSkip]    dims changed, no residual\n");
    }

    *OH_out = OH;
    *OW_out = OW;
    return proj_out;
}

/* ---- Compound Scaling ---- */
typedef struct {
    float depth_mult, width_mult;
    int resolution;
} EfficientNetConfig;

static const EfficientNetConfig CONFIGS[] = {
    {1.0f, 1.0f, 224}, {1.1f, 1.0f, 240}, {1.2f, 1.1f, 260},
    {1.4f, 1.2f, 300}, {1.8f, 1.4f, 380}, {2.2f, 1.6f, 456},
    {2.6f, 1.8f, 528}, {3.1f, 2.0f, 600}
};

static int round_channels(float c, float w) {
    int ch = (int)(c * w);
    int d = 8;
    int nc = ((ch + d / 2) / d) * d;
    if (nc < (int)(0.9f * ch)) nc += d;
    return nc < 8 ? 8 : nc;
}

static int round_depth(float d, float dm) {
    return (int)ceilf(d * dm);
}

static void print_scaling_table(void) {
    int base_ch[] = {32, 16, 24, 40, 80, 112, 192, 320};
    int base_d[]  = {1, 2, 2, 3, 3, 4, 1};

    printf("\n=== EfficientNet Compound Scaling Table ===\n");
    printf("%-5s %5s  ", "Model", "Res");
    for (int s = 0; s < 7; s++) printf("S%d      ", s);
    printf("\n");

    for (int b = 0; b < 8; b++) {
        EfficientNetConfig cfg = CONFIGS[b];
        printf("B%-4d %5d  ", b, cfg.resolution);
        for (int s = 0; s < 7; s++) {
            int ci = round_channels((float)base_ch[s], cfg.width_mult);
            int co = round_channels((float)base_ch[s + 1], cfg.width_mult);
            int dp = round_depth((float)base_d[s], cfg.depth_mult);
            printf("%d->%-3dx%d ", ci, co, dp);
        }
        printf("\n");
    }
}

/* ---- Main ---- */
int main(void) {
    srand(42);

    printf("=== MBConv Block Demo ===\n\n");

    /* Stage 0: MBConv1 (no expansion) - simulate small feature map */
    int H = 8, W = 8;
    int C_in = 4, C_out = 4;  /* small dims for demo */

    float *input = malloc((size_t)C_in * H * W * sizeof(float));
    for (int i = 0; i < C_in * H * W; i++) input[i] = randn() * 0.5f;

    printf("--- Stage 0: MBConv1 (expand=1, skip) ---\n");
    printf("  Input: [%d, %d, %d]\n", C_in, H, W);

    MBConv blk0;
    int OH, OW;
    mbconv_init(&blk0, C_in, C_out, 3, 1, 1);
    float *out0 = mbconv_forward(&blk0, input, H, W, &OH, &OW);
    printf("  Output: [%d, %d, %d]\n\n", C_out, OH, OW);

    /* Stage 1: MBConv6 (6x expansion, stride 2) */
    printf("--- Stage 1: MBConv6 (expand=6, stride=2) ---\n");
    printf("  Input: [%d, %d, %d]\n", C_out, OH, OW);

    MBConv blk1;
    int C_out1 = 8;
    int OH1, OW1;
    mbconv_init(&blk1, C_out, C_out1, 3, 2, 6);
    float *out1 = mbconv_forward(&blk1, out0, OH, OW, &OH1, &OW1);
    printf("  Output: [%d, %d, %d]\n\n", C_out1, OH1, OW1);

    /* Stage 2: MBConv6 (5x5 kernel) */
    printf("--- Stage 2: MBConv6 (expand=6, k=5, stride=2) ---\n");
    printf("  Input: [%d, %d, %d]\n", C_out1, OH1, OW1);

    MBConv blk2;
    int C_out2 = 16;
    int OH2, OW2;
    mbconv_init(&blk2, C_out1, C_out2, 5, 2, 6);
    float *out2 = mbconv_forward(&blk2, out1, OH1, OW1, &OH2, &OW2);
    printf("  Output: [%d, %d, %d]\n\n", C_out2, OH2, OW2);

    /* Print feature map statistics */
    printf("=== Feature Map Statistics ===\n");
    float stats[][2] = {{0,0},{0,0},{0,0}};
    float *outputs[] = {out0, out1, out2};
    int sizes[] = {C_out * OH * OW, C_out1 * OH1 * OW1, C_out2 * OH2 * OW2};
    for (int s = 0; s < 3; s++) {
        for (int i = 0; i < sizes[s]; i++) {
            stats[s][0] += outputs[s][i];
            stats[s][1] += outputs[s][i] * outputs[s][i];
        }
        stats[s][0] /= sizes[s];
        stats[s][1] = sqrtf(stats[s][1] / sizes[s] - stats[s][0] * stats[s][0]);
        printf("  Stage %d: mean=%.4f  std=%.4f  (%d elements)\n",
               s, stats[s][0], stats[s][1], sizes[s]);
    }

    /* Compound scaling */
    print_scaling_table();

    /* Cleanup */
    mbconv_free(&blk0); mbconv_free(&blk1); mbconv_free(&blk2);
    free(input); free(out0); free(out1); free(out2);

    printf("\nDone.\n");
    return 0;
}
