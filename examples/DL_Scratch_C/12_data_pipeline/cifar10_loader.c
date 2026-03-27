/*
 * cifar10_loader.c - Minimal data pipeline for CIFAR-10-sized images
 *
 * Demonstrates:
 *   - Synthetic CIFAR-10 data generation (32x32x3, 10 classes)
 *   - HWC to CHW layout conversion with float normalization
 *   - Channel-wise normalization (mean/std subtraction)
 *   - Data augmentation: random horizontal flip, random crop with padding
 *   - Fisher-Yates shuffle for index permutation
 *   - Mini-batch loader with epoch iteration
 *
 * No actual file I/O - all data is generated synthetically.
 *
 * Build:  gcc -std=c11 -Wall -Wextra -O2 -o cifar10_loader cifar10_loader.c -lm
 * Run:    ./cifar10_loader
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define IMG_H     32
#define IMG_W     32
#define IMG_C     3
#define IMG_SIZE  (IMG_C * IMG_H * IMG_W)   /* 3072 floats in CHW */
#define NUM_CLASSES 10

/* CIFAR-10 channel statistics */
static const float CIFAR_MEAN[3] = {0.4914f, 0.4822f, 0.4465f};
static const float CIFAR_STD[3]  = {0.2470f, 0.2435f, 0.2616f};

/* ---- Dataset ---- */
typedef struct {
    float   *images;    /* [N, C, H, W] float, raw [0,1] */
    uint8_t *labels;    /* [N] */
    int      N;
} Dataset;

/* Generate synthetic CIFAR-10-like data.
 * Each class gets a distinct color bias so there is a learnable signal. */
static Dataset *generate_synthetic(int N) {
    Dataset *ds = (Dataset *)malloc(sizeof(Dataset));
    ds->N = N;
    ds->images = (float *)malloc((size_t)N * IMG_SIZE * sizeof(float));
    ds->labels = (uint8_t *)malloc((size_t)N);

    for (int i = 0; i < N; i++) {
        uint8_t label = (uint8_t)(rand() % NUM_CLASSES);
        ds->labels[i] = label;

        float *img = ds->images + (long)i * IMG_SIZE;
        /* Base color from class label */
        float r_bias = (float)(label % 3) / 3.0f;
        float g_bias = (float)((label / 3) % 3) / 3.0f;
        float b_bias = (float)(label % 2) / 2.0f;

        /* Fill CHW with noisy patterns */
        for (int h = 0; h < IMG_H; h++)
        for (int w = 0; w < IMG_W; w++) {
            float noise = (float)rand() / RAND_MAX * 0.3f;
            float spatial = sinf((float)(h + w) * 0.2f) * 0.1f;
            img[0 * IMG_H * IMG_W + h * IMG_W + w] =
                fmaxf(0.0f, fminf(1.0f, r_bias + noise + spatial));
            img[1 * IMG_H * IMG_W + h * IMG_W + w] =
                fmaxf(0.0f, fminf(1.0f, g_bias + noise - spatial));
            img[2 * IMG_H * IMG_W + h * IMG_W + w] =
                fmaxf(0.0f, fminf(1.0f, b_bias + noise));
        }
    }
    return ds;
}

static void dataset_free(Dataset *ds) {
    free(ds->images);
    free(ds->labels);
    free(ds);
}

/* ---- Channel-wise normalization ---- */
static void normalize_chw(float *chw, const float *ch_mean, const float *ch_std) {
    for (int c = 0; c < IMG_C; c++) {
        float m = ch_mean[c], s = ch_std[c];
        for (int i = 0; i < IMG_H * IMG_W; i++)
            chw[c * IMG_H * IMG_W + i] = (chw[c * IMG_H * IMG_W + i] - m) / s;
    }
}

/* ---- Compute dataset channel statistics ---- */
static void compute_channel_stats(const float *images, int N,
                                   float *ch_mean, float *ch_std) {
    int M = N * IMG_H * IMG_W;
    for (int c = 0; c < IMG_C; c++) {
        float sum = 0.0f, sum2 = 0.0f;
        for (int n = 0; n < N; n++)
        for (int i = 0; i < IMG_H * IMG_W; i++) {
            float v = images[(long)n * IMG_SIZE + c * IMG_H * IMG_W + i];
            sum  += v;
            sum2 += v * v;
        }
        ch_mean[c] = sum / M;
        ch_std[c]  = sqrtf(sum2 / M - ch_mean[c] * ch_mean[c] + 1e-8f);
    }
}

/* ---- Data augmentation ---- */

/* In-place horizontal flip of a CHW image */
static void flip_horizontal(float *chw) {
    for (int c = 0; c < IMG_C; c++)
    for (int h = 0; h < IMG_H; h++)
    for (int w = 0; w < IMG_W / 2; w++) {
        int idx_a = c * IMG_H * IMG_W + h * IMG_W + w;
        int idx_b = c * IMG_H * IMG_W + h * IMG_W + (IMG_W - 1 - w);
        float tmp = chw[idx_a];
        chw[idx_a] = chw[idx_b];
        chw[idx_b] = tmp;
    }
}

/* Random flip with 50% probability */
static void random_flip(float *chw) {
    if (rand() & 1) flip_horizontal(chw);
}

/* Pad-and-crop: pad by `pad` pixels, then random crop back to original size */
static void pad_and_crop(const float *src, float *dst, int pad_sz) {
    int pH = IMG_H + 2 * pad_sz;
    int pW = IMG_W + 2 * pad_sz;
    float *padded = (float *)calloc((size_t)IMG_C * pH * pW, sizeof(float));

    /* Copy src into center */
    for (int c = 0; c < IMG_C; c++)
    for (int h = 0; h < IMG_H; h++)
    for (int w = 0; w < IMG_W; w++)
        padded[c * pH * pW + (h + pad_sz) * pW + (w + pad_sz)] =
            src[c * IMG_H * IMG_W + h * IMG_W + w];

    /* Random top-left for crop */
    int top  = rand() % (2 * pad_sz + 1);
    int left = rand() % (2 * pad_sz + 1);

    for (int c = 0; c < IMG_C; c++)
    for (int h = 0; h < IMG_H; h++)
    for (int w = 0; w < IMG_W; w++)
        dst[c * IMG_H * IMG_W + h * IMG_W + w] =
            padded[c * pH * pW + (top + h) * pW + (left + w)];

    free(padded);
}

/* ---- DataLoader ---- */
typedef struct {
    Dataset *ds;
    int     *indices;
    int      cursor;
    int      batch_size;
    int      augment;
} DataLoader;

static DataLoader *loader_create(Dataset *ds, int batch_size, int augment) {
    DataLoader *dl = (DataLoader *)malloc(sizeof(DataLoader));
    dl->ds = ds;
    dl->batch_size = batch_size;
    dl->augment = augment;
    dl->cursor = 0;
    dl->indices = (int *)malloc((size_t)ds->N * sizeof(int));
    for (int i = 0; i < ds->N; i++) dl->indices[i] = i;
    return dl;
}

/* Fisher-Yates shuffle */
static void loader_shuffle(DataLoader *dl) {
    for (int i = dl->ds->N - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = dl->indices[i];
        dl->indices[i] = dl->indices[j];
        dl->indices[j] = tmp;
    }
    dl->cursor = 0;
}

/* Get next batch. Returns 1 if batch filled, 0 if epoch done. */
static int loader_next(DataLoader *dl, float *batch_X, uint8_t *batch_y) {
    if (dl->cursor + dl->batch_size > dl->ds->N) return 0;

    for (int b = 0; b < dl->batch_size; b++) {
        int idx = dl->indices[dl->cursor + b];
        float *src = dl->ds->images + (long)idx * IMG_SIZE;
        float *dst = batch_X + (long)b * IMG_SIZE;
        memcpy(dst, src, IMG_SIZE * sizeof(float));

        if (dl->augment) {
            float tmp[IMG_SIZE];
            pad_and_crop(dst, tmp, 4);
            memcpy(dst, tmp, IMG_SIZE * sizeof(float));
            random_flip(dst);
        }

        batch_y[b] = dl->ds->labels[idx];
    }
    dl->cursor += dl->batch_size;
    return 1;
}

static void loader_free(DataLoader *dl) {
    free(dl->indices);
    free(dl);
}

/* ---- Main ---- */
int main(void) {
    srand(42);
    printf("=== CIFAR-10 Data Pipeline Demo ===\n\n");

    /* Generate synthetic dataset */
    int N_train = 200;
    int batch_size = 32;
    printf("Generating %d synthetic CIFAR-10 images (32x32x3)...\n", N_train);
    Dataset *train_ds = generate_synthetic(N_train);

    /* Compute and display channel statistics */
    float comp_mean[3], comp_std[3];
    compute_channel_stats(train_ds->images, N_train, comp_mean, comp_std);
    printf("\nComputed channel statistics (before normalization):\n");
    for (int c = 0; c < 3; c++)
        printf("  Channel %d: mean=%.4f  std=%.4f\n", c, comp_mean[c], comp_std[c]);

    /* Normalize all images */
    printf("\nNormalizing with CIFAR-10 statistics...\n");
    for (int i = 0; i < N_train; i++)
        normalize_chw(train_ds->images + (long)i * IMG_SIZE, CIFAR_MEAN, CIFAR_STD);

    compute_channel_stats(train_ds->images, N_train, comp_mean, comp_std);
    printf("After normalization:\n");
    for (int c = 0; c < 3; c++)
        printf("  Channel %d: mean=%.4f  std=%.4f\n", c, comp_mean[c], comp_std[c]);

    /* Class distribution */
    int class_counts[NUM_CLASSES] = {0};
    for (int i = 0; i < N_train; i++)
        class_counts[train_ds->labels[i]]++;
    printf("\nClass distribution: ");
    for (int c = 0; c < NUM_CLASSES; c++) printf("%d:%d ", c, class_counts[c]);
    printf("\n");

    /* Create data loader and iterate */
    printf("\n--- DataLoader Demo (batch_size=%d, augment=on) ---\n", batch_size);
    DataLoader *dl = loader_create(train_ds, batch_size, 1);

    float   *batch_X = (float *)malloc((size_t)batch_size * IMG_SIZE * sizeof(float));
    uint8_t *batch_y = (uint8_t *)malloc((size_t)batch_size);

    int n_epochs = 2;
    for (int epoch = 0; epoch < n_epochs; epoch++) {
        loader_shuffle(dl);
        int n_batches = 0;
        float batch_mean_sum = 0.0f;

        while (loader_next(dl, batch_X, batch_y)) {
            /* Compute batch-level statistics */
            float bm = 0.0f;
            for (int b = 0; b < batch_size; b++)
                bm += batch_X[(long)b * IMG_SIZE];
            bm /= batch_size;
            batch_mean_sum += bm;
            n_batches++;
        }

        printf("  Epoch %d: %d batches, avg first-pixel=%.4f\n",
               epoch + 1, n_batches, batch_mean_sum / n_batches);
    }

    /* Augmentation demo: show effect on one image */
    printf("\n--- Augmentation Effect Demo ---\n");
    float *orig = (float *)malloc(IMG_SIZE * sizeof(float));
    float *aug  = (float *)malloc(IMG_SIZE * sizeof(float));
    memcpy(orig, train_ds->images, IMG_SIZE * sizeof(float));

    printf("Original pixel[0][0][0..3]: ");
    for (int w = 0; w < 4; w++)
        printf("%.3f ", orig[w]);
    printf("\n");

    /* Apply flip */
    memcpy(aug, orig, IMG_SIZE * sizeof(float));
    flip_horizontal(aug);
    printf("Flipped  pixel[0][0][0..3]: ");
    for (int w = 0; w < 4; w++)
        printf("%.3f ", aug[w]);
    printf("\n");

    /* Apply pad+crop */
    memcpy(aug, orig, IMG_SIZE * sizeof(float));
    float crop_out[IMG_SIZE];
    pad_and_crop(aug, crop_out, 4);
    printf("Cropped  pixel[0][0][0..3]: ");
    for (int w = 0; w < 4; w++)
        printf("%.3f ", crop_out[w]);
    printf("\n");

    printf("\n=== Data Pipeline Demo Complete ===\n");

    free(orig); free(aug);
    free(batch_X); free(batch_y);
    loader_free(dl);
    dataset_free(train_ds);
    return 0;
}
