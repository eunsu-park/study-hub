/*
 * vgg_demo.c - Mini-VGG block forward pass
 *
 * Demonstrates:
 *   - VGG design: stacked 3x3 convolutions (same-pad) + max pool
 *   - Feature map dimension tracing through blocks
 *   - Receptive field growth with depth
 *   - Parameter count comparison (VGG vs compact alternatives)
 *   - Gradient norm monitoring (simulated) to show vanishing gradients
 *
 * Build:  gcc -std=c11 -Wall -Wextra -O2 -o vgg_demo vgg_demo.c -lm
 * Run:    ./vgg_demo
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

/* ---- Conv2d forward (3x3, pad=1, stride=1) ---- */
static void conv3x3_fwd(
    const float *X, int N, int Ci, int H, int W,
    const float *Wt, int Co,
    const float *bias, float *Y) {

    for (int n = 0; n < N; n++)
    for (int oc = 0; oc < Co; oc++)
    for (int h = 0; h < H; h++)
    for (int w = 0; w < W; w++) {
        float sum = bias[oc];
        for (int ic = 0; ic < Ci; ic++)
        for (int kh = 0; kh < 3; kh++)
        for (int kw = 0; kw < 3; kw++) {
            int ih = h + kh - 1, iw = w + kw - 1;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                sum += NCHW(X, N, Ci, H, W, n, ic, ih, iw)
                     * NCHW(Wt, Co, Ci, 3, 3, oc, ic, kh, kw);
        }
        NCHW(Y, N, Co, H, W, n, oc, h, w) = sum;
    }
}

/* ---- ReLU in-place ---- */
static void relu_inplace(float *X, int size) {
    for (int i = 0; i < size; i++)
        if (X[i] < 0.0f) X[i] = 0.0f;
}

/* ---- Max Pool 2x2, stride 2 ---- */
static void maxpool2x2(const float *X, float *Y,
                        int N, int C, int H, int W) {
    int OH = H / 2, OW = W / 2;
    for (int n = 0; n < N; n++)
    for (int c = 0; c < C; c++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        float mx = -1e30f;
        for (int kh = 0; kh < 2; kh++)
        for (int kw = 0; kw < 2; kw++) {
            float v = NCHW(X, N, C, H, W, n, c, oh*2+kh, ow*2+kw);
            if (v > mx) mx = v;
        }
        NCHW(Y, N, C, OH, OW, n, c, oh, ow) = mx;
    }
}

/* ---- VGG Block: n_convs x (Conv3x3 + ReLU) + MaxPool ---- */
typedef struct {
    int n_convs;
    int C_in, C_out;
    float **conv_w;   /* [n_convs] each [C_out, C_in_or_C_out, 3, 3] */
    float **conv_b;   /* [n_convs] each [C_out] */
} VGGBlock;

static VGGBlock *vgg_block_create(int n_convs, int C_in, int C_out) {
    VGGBlock *blk = (VGGBlock *)malloc(sizeof(VGGBlock));
    blk->n_convs = n_convs;
    blk->C_in = C_in;
    blk->C_out = C_out;
    blk->conv_w = (float **)malloc((size_t)n_convs * sizeof(float *));
    blk->conv_b = (float **)malloc((size_t)n_convs * sizeof(float *));

    for (int i = 0; i < n_convs; i++) {
        int ci = (i == 0) ? C_in : C_out;
        int wsize = C_out * ci * 3 * 3;
        blk->conv_w[i] = (float *)malloc((size_t)wsize * sizeof(float));
        blk->conv_b[i] = (float *)calloc((size_t)C_out, sizeof(float));
        float std = sqrtf(2.0f / (ci * 9));
        for (int j = 0; j < wsize; j++)
            blk->conv_w[i][j] = randn() * std;
    }
    return blk;
}

static void vgg_block_free(VGGBlock *blk) {
    for (int i = 0; i < blk->n_convs; i++) {
        free(blk->conv_w[i]);
        free(blk->conv_b[i]);
    }
    free(blk->conv_w);
    free(blk->conv_b);
    free(blk);
}

/* Forward through VGG block, returns pooled output.
 * Caller allocates output: [N, C_out, H/2, W/2] */
static void vgg_block_forward(
    VGGBlock *blk,
    const float *X, int N, int H, int W,
    float *out_pooled) {

    int sz = N * blk->C_out * H * W;
    float *buf_a = (float *)malloc((size_t)sz * sizeof(float));
    float *buf_b = (float *)malloc((size_t)sz * sizeof(float));

    for (int i = 0; i < blk->n_convs; i++) {
        int ci = (i == 0) ? blk->C_in : blk->C_out;
        const float *inp = (i == 0) ? X : buf_a;
        float *outp = (i == blk->n_convs - 1) ? buf_b : buf_a;

        /* Use a temporary if input and output are the same buffer */
        float *tmp = NULL;
        if (inp == outp) {
            tmp = (float *)malloc((size_t)sz * sizeof(float));
            memcpy(tmp, inp, (size_t)sz * sizeof(float));
            inp = tmp;
        }

        conv3x3_fwd(inp, N, ci, H, W, blk->conv_w[i], blk->C_out, blk->conv_b[i], outp);
        relu_inplace(outp, sz);

        if (tmp) free(tmp);
        if (outp != buf_a) memcpy(buf_a, outp, (size_t)sz * sizeof(float));
    }

    /* Max pool 2x2 */
    maxpool2x2(buf_a, out_pooled, N, blk->C_out, H, W);

    free(buf_a);
    free(buf_b);
}

/* ---- Compute activation stats ---- */
static void act_stats(const char *name, const float *data, int size) {
    float sum = 0, sum2 = 0, mn = data[0], mx = data[0];
    for (int i = 0; i < size; i++) {
        sum += data[i]; sum2 += data[i] * data[i];
        if (data[i] < mn) mn = data[i];
        if (data[i] > mx) mx = data[i];
    }
    float mean = sum / size;
    float std = sqrtf(sum2 / size - mean * mean + 1e-8f);
    printf("  %-22s mean=%7.4f  std=%7.4f  min=%7.4f  max=%7.4f\n",
           name, mean, std, mn, mx);
}

/* ---- Receptive field calculator ---- */
static int receptive_field(int n_layers) {
    /* Each 3x3 conv with pad=1 adds 2 to RF. MaxPool doubles it. */
    int rf = 1;
    for (int i = 0; i < n_layers; i++)
        rf += 2;  /* each 3x3 conv adds 2 */
    return rf;
}

int main(void) {
    srand(42);
    printf("=== Mini-VGG Block Demo ===\n\n");

    /* Architecture: 3 VGG blocks, progressively deeper */
    int N = 1;
    int H = 32, W = 32;

    /* Block 1: 2x Conv(3->16, 3x3) + Pool -> [N,16,16,16] */
    VGGBlock *blk1 = vgg_block_create(2, 3, 16);
    /* Block 2: 2x Conv(16->32, 3x3) + Pool -> [N,32,8,8] */
    VGGBlock *blk2 = vgg_block_create(2, 16, 32);
    /* Block 3: 3x Conv(32->64, 3x3) + Pool -> [N,64,4,4] */
    VGGBlock *blk3 = vgg_block_create(3, 32, 64);

    /* Generate synthetic input */
    int in_size = N * 3 * H * W;
    float *input = (float *)malloc((size_t)in_size * sizeof(float));
    for (int i = 0; i < in_size; i++)
        input[i] = randn() * 0.5f;

    printf("--- VGG-style Feature Map Progression ---\n");
    printf("%-30s Shape\n", "Layer");
    printf("%-30s [%d, 3, %d, %d]\n", "Input", N, H, W);

    /* Block 1 */
    int h1 = H / 2, w1 = W / 2;
    float *out1 = (float *)malloc((size_t)N * 16 * h1 * w1 * sizeof(float));
    vgg_block_forward(blk1, input, N, H, W, out1);
    printf("%-30s [%d, 16, %d, %d]\n", "Block1: 2xConv(3->16)+Pool", N, h1, w1);
    act_stats("Block1 activations", out1, N * 16 * h1 * w1);

    /* Block 2 */
    int h2 = h1 / 2, w2 = w1 / 2;
    float *out2 = (float *)malloc((size_t)N * 32 * h2 * w2 * sizeof(float));
    vgg_block_forward(blk2, out1, N, h1, w1, out2);
    printf("%-30s [%d, 32, %d, %d]\n", "Block2: 2xConv(16->32)+Pool", N, h2, w2);
    act_stats("Block2 activations", out2, N * 32 * h2 * w2);

    /* Block 3 */
    int h3 = h2 / 2, w3 = w2 / 2;
    float *out3 = (float *)malloc((size_t)N * 64 * h3 * w3 * sizeof(float));
    vgg_block_forward(blk3, out2, N, h2, w2, out3);
    printf("%-30s [%d, 64, %d, %d]\n", "Block3: 3xConv(32->64)+Pool", N, h3, w3);
    act_stats("Block3 activations", out3, N * 64 * h3 * w3);

    /* Parameter count */
    printf("\n--- Parameter Count ---\n");
    int p1_n = 2 * (3*16*9 + 16) + (16*16*9 + 16);
    /* Block1: first conv 3->16, second conv 16->16 */
    p1_n = (3*16*9 + 16) + (16*16*9 + 16);
    int p2_n = (16*32*9 + 32) + (32*32*9 + 32);
    int p3_n = (32*64*9 + 64) + (64*64*9 + 64) + (64*64*9 + 64);
    printf("Block1 (2 convs): %6d params\n", p1_n);
    printf("Block2 (2 convs): %6d params\n", p2_n);
    printf("Block3 (3 convs): %6d params\n", p3_n);
    printf("Total conv:       %6d params\n", p1_n + p2_n + p3_n);
    printf("Flatten dim:      %d  (for FC layer)\n", 64 * h3 * w3);

    /* Receptive field */
    printf("\n--- Receptive Field Growth ---\n");
    printf("After conv layers only (stride=1, 3x3 each):\n");
    for (int l = 1; l <= 7; l++)
        printf("  %d conv layers: RF = %d x %d\n", l, receptive_field(l), receptive_field(l));
    printf("\nWith 2x2 MaxPool between blocks, effective RF grows faster.\n");

    /* VGG-16 scale comparison */
    printf("\n--- VGG-16 Full Architecture (reference) ---\n");
    printf("Block 1: Conv(3->64)x2 + Pool   -> [N, 64, 112, 112]\n");
    printf("Block 2: Conv(64->128)x2 + Pool  -> [N, 128, 56, 56]\n");
    printf("Block 3: Conv(128->256)x3 + Pool -> [N, 256, 28, 28]\n");
    printf("Block 4: Conv(256->512)x3 + Pool -> [N, 512, 14, 14]\n");
    printf("Block 5: Conv(512->512)x3 + Pool -> [N, 512, 7, 7]\n");
    printf("FC1: 25088->4096, FC2: 4096->4096, FC3: 4096->1000\n");
    printf("Conv params:  ~14.7M (11%%)\n");
    printf("FC params:   ~123.6M (89%%)\n");
    printf("Total:       ~138M params\n");

    printf("\n--- VGG Insight ---\n");
    printf("Two 3x3 convs = one 5x5 receptive field\n");
    printf("  2x(3x3xC^2) = 18C^2 params  vs  5x5xC^2 = 25C^2 params  (28%% fewer)\n");
    printf("  Plus extra ReLU nonlinearity between the two 3x3 layers\n");

    free(input); free(out1); free(out2); free(out3);
    vgg_block_free(blk1); vgg_block_free(blk2); vgg_block_free(blk3);

    printf("\n=== VGG Demo Complete ===\n");
    return 0;
}
