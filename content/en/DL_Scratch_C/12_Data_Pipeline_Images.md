# 12. Data Pipeline for Images

**Previous**: [Batch Normalization](./11_Batch_Normalization.md) | **Next**: [LeNet and AlexNet](./13_LeNet_and_AlexNet.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Load JPEG and PNG images using the STB header library in C
2. Convert between NHWC (HWC per image) and NCHW memory layouts
3. Implement basic data augmentation: random horizontal flip, random crop, color jitter
4. Normalize images to zero mean, unit variance using channel statistics
5. Build a mini-batch loader that shuffles and batches a CIFAR-10 binary dataset

---

## 1. The STB Image Library

STB is a single-header C library for image I/O — no external dependencies:

```c
// Include the implementation in exactly one .c file:
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"
```

Loading an image:

```c
int width, height, channels;
unsigned char *img = stbi_load("cat.jpg", &width, &height, &channels, 3);
// channels forced to 3 (RGB) by the last argument
// img is in HWC layout: [H, W, 3], uint8 [0,255]
if (!img) { fprintf(stderr, "Failed to load image\n"); exit(1); }

// ... use the image ...
stbi_image_free(img);
```

---

## 2. HWC → CHW Conversion

Neural networks use NCHW (batch, channel, height, width). STB returns HWC per image:

```c
// hwc_to_chw: convert a single uint8 HWC image to float CHW
// Input:  [H, W, C] uint8 [0,255]
// Output: [C, H, W] float [0.0, 1.0]
void hwc_to_chw(
    const unsigned char *hwc,
    float               *chw,
    int H, int W, int C) {

    for (int c = 0; c < C; c++)
    for (int h = 0; h < H; h++)
    for (int w = 0; w < W; w++)
        chw[c * H * W + h * W + w] = hwc[h * W * C + w * C + c] / 255.0f;
}

// chw_to_hwc: reverse (e.g., for saving results)
void chw_to_hwc(
    const float   *chw,
    unsigned char *hwc,
    int H, int W, int C) {

    for (int c = 0; c < C; c++)
    for (int h = 0; h < H; h++)
    for (int w = 0; w < W; w++) {
        float v = chw[c * H * W + h * W + w] * 255.0f;
        hwc[h * W * C + w * C + c] = (unsigned char)fmaxf(0, fminf(255, v));
    }
}
```

---

## 3. Channel-wise Normalization

Standard normalization using ImageNet statistics (or dataset-specific):

```c
// ImageNet channel mean and std (RGB)
static const float IMAGENET_MEAN[3] = {0.485f, 0.456f, 0.406f};
static const float IMAGENET_STD[3]  = {0.229f, 0.224f, 0.225f};

// normalize_chw: subtract mean, divide by std per channel
void normalize_chw(float *chw, int H, int W, int C,
                   const float *mean, const float *std) {
    for (int c = 0; c < C; c++) {
        float m = mean[c], s = std[c];
        for (int i = 0; i < H * W; i++)
            chw[c * H * W + i] = (chw[c * H * W + i] - m) / s;
    }
}

// Compute mean and std from dataset for CIFAR-10 training
void compute_channel_stats(
    const float *batch,  // [N, C, H, W]
    float *mean, float *std,
    int N, int C, int H, int W) {

    int M = N * H * W;
    for (int c = 0; c < C; c++) {
        float sum = 0.0f, sum2 = 0.0f;
        for (int n = 0; n < N; n++)
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++) {
            float v = NCHW(batch, N, C, H, W, n, c, h, w);
            sum  += v;
            sum2 += v * v;
        }
        mean[c] = sum / M;
        std[c]  = sqrtf(sum2 / M - mean[c] * mean[c] + 1e-8f);
    }
}
```

---

## 4. Data Augmentation

### Random Horizontal Flip

```c
// flip_chw: in-place horizontal flip of a [C, H, W] image
void flip_horizontal_chw(float *chw, int C, int H, int W) {
    for (int c = 0; c < C; c++)
    for (int h = 0; h < H; h++)
    for (int w = 0; w < W / 2; w++) {
        float *a = &chw[c * H * W + h * W + w];
        float *b = &chw[c * H * W + h * W + (W - 1 - w)];
        float tmp = *a; *a = *b; *b = tmp;
    }
}

// Apply flip with 50% probability
void random_flip(float *chw, int C, int H, int W) {
    if (rand() & 1)
        flip_horizontal_chw(chw, C, H, W);
}
```

### Random Crop with Padding

Standard augmentation for CIFAR-10 (32×32 → pad 4 → crop back to 32×32):

```c
// pad_and_crop_chw: pad image by `pad` pixels on each side, then random crop
// Input:  [C, H, W] float
// Output: [C, H, W] float (same size)
void pad_and_crop_chw(
    const float *src,
    float       *dst,
    int C, int H, int W, int pad) {

    int pH = H + 2 * pad;
    int pW = W + 2 * pad;
    float *padded = calloc(C * pH * pW, sizeof(float));  // zero-pad

    // Copy src into center of padded
    for (int c = 0; c < C; c++)
    for (int h = 0; h < H; h++)
    for (int w = 0; w < W; w++)
        padded[c * pH * pW + (h + pad) * pW + (w + pad)]
            = src[c * H * W + h * W + w];

    // Random top-left corner for the crop
    int top  = rand() % (2 * pad + 1);  // [0, 2*pad]
    int left = rand() % (2 * pad + 1);

    for (int c = 0; c < C; c++)
    for (int h = 0; h < H; h++)
    for (int w = 0; w < W; w++)
        dst[c * H * W + h * W + w]
            = padded[c * pH * pW + (top + h) * pW + (left + w)];

    free(padded);
}
```

### Color Jitter (Brightness/Contrast)

```c
// jitter_brightness: add uniform noise in [-delta, +delta]
void jitter_brightness(float *chw, int C, int H, int W, float delta) {
    float shift = ((float)rand() / RAND_MAX) * 2 * delta - delta;
    int total = C * H * W;
    for (int i = 0; i < total; i++)
        chw[i] = fmaxf(0.0f, fminf(1.0f, chw[i] + shift));
}
```

---

## 5. CIFAR-10 Binary Format

CIFAR-10 provides binary files: each record is `1 byte label + 3072 bytes (3×32×32) RGB`:

```c
#define CIFAR_IMG_SIZE (3 * 32 * 32)  // 3072 bytes
#define CIFAR_RECORD   (1 + CIFAR_IMG_SIZE)

typedef struct {
    float  *images;   // [N, 3, 32, 32] float, normalized
    uint8_t *labels;  // [N] uint8 [0,9]
    int      N;
} CIFAR10Dataset;

CIFAR10Dataset *cifar10_load(const char *path, int train) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }

    // File size → number of records
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    rewind(f);
    int N = (int)(fsize / CIFAR_RECORD);

    CIFAR10Dataset *ds = malloc(sizeof(CIFAR10Dataset));
    ds->N      = N;
    ds->labels = malloc(N);
    ds->images = malloc((long)N * CIFAR_IMG_SIZE * sizeof(float));

    uint8_t buf[CIFAR_RECORD];
    for (int i = 0; i < N; i++) {
        fread(buf, 1, CIFAR_RECORD, f);
        ds->labels[i] = buf[0];

        float *dst = ds->images + (long)i * CIFAR_IMG_SIZE;
        for (int j = 0; j < CIFAR_IMG_SIZE; j++)
            dst[j] = buf[1 + j] / 255.0f;
    }
    fclose(f);

    // Normalize with CIFAR-10 channel stats
    // Mean: [0.4914, 0.4822, 0.4465]  Std: [0.2470, 0.2435, 0.2616]
    static const float CIFAR_MEAN[3] = {0.4914f, 0.4822f, 0.4465f};
    static const float CIFAR_STD[3]  = {0.2470f, 0.2435f, 0.2616f};
    for (int i = 0; i < N; i++)
        normalize_chw(ds->images + (long)i * CIFAR_IMG_SIZE, 32, 32, 3,
                      CIFAR_MEAN, CIFAR_STD);

    return ds;
}

void cifar10_free(CIFAR10Dataset *ds) {
    free(ds->images); free(ds->labels); free(ds);
}
```

---

## 6. Mini-Batch Loader

Shuffles indices and provides batches with optional augmentation:

```c
typedef struct {
    CIFAR10Dataset *ds;
    int *indices;    // shuffled index permutation
    int  cursor;     // current position in epoch
    int  batch_size;
    int  augment;    // 1 = apply augmentation (training only)
} DataLoader;

DataLoader *dataloader_create(CIFAR10Dataset *ds, int batch_size, int augment) {
    DataLoader *dl = malloc(sizeof(DataLoader));
    dl->ds = ds;
    dl->batch_size = batch_size;
    dl->augment = augment;
    dl->cursor = 0;
    dl->indices = malloc(ds->N * sizeof(int));
    for (int i = 0; i < ds->N; i++) dl->indices[i] = i;
    return dl;
}

void dataloader_shuffle(DataLoader *dl) {
    // Fisher-Yates shuffle
    for (int i = dl->ds->N - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = dl->indices[i];
        dl->indices[i] = dl->indices[j];
        dl->indices[j] = tmp;
    }
    dl->cursor = 0;
}

// Returns 1 if batch filled, 0 if epoch exhausted
int dataloader_next(DataLoader *dl, float *batch_X, uint8_t *batch_y) {
    if (dl->cursor + dl->batch_size > dl->ds->N) return 0;

    for (int b = 0; b < dl->batch_size; b++) {
        int idx = dl->indices[dl->cursor + b];

        float *src = dl->ds->images + (long)idx * CIFAR_IMG_SIZE;
        float *dst = batch_X + (long)b * CIFAR_IMG_SIZE;
        memcpy(dst, src, CIFAR_IMG_SIZE * sizeof(float));

        if (dl->augment) {
            pad_and_crop_chw(dst, dst, 3, 32, 32, 4);
            random_flip(dst, 3, 32, 32);
        }
        batch_y[b] = dl->ds->labels[idx];
    }
    dl->cursor += dl->batch_size;
    return 1;
}
```

---

## 7. Putting It Together

```c
// Example training setup for CIFAR-10
int main(void) {
    srand(42);

    CIFAR10Dataset *train_ds = cifar10_load("cifar-10-batches-bin/data_batch_1.bin", 1);
    DataLoader *dl = dataloader_create(train_ds, 128, /*augment=*/1);

    float   *batch_X = malloc(128L * CIFAR_IMG_SIZE * sizeof(float));
    uint8_t *batch_y = malloc(128);

    for (int epoch = 0; epoch < 100; epoch++) {
        dataloader_shuffle(dl);
        while (dataloader_next(dl, batch_X, batch_y)) {
            // batch_X: [128, 3, 32, 32] ready for CNN forward pass
            // batch_y: [128] class labels 0-9
            // ... forward, loss, backward, update ...
        }
    }
    return 0;
}
```

---

## Key Takeaways

- STB is a zero-dependency C image library — `#define STB_IMAGE_IMPLEMENTATION` in one translation unit
- Images from disk are HWC uint8; CNNs expect NCHW float — convert once at load time
- **Normalization** with per-channel mean/std is essential: accelerates convergence and stabilizes BN
- **Augmentation** (flip + crop) is applied per-batch at training time — never during eval
- CIFAR-10 binary format: `[label:1byte] [pixels:3072bytes]` per record — straightforward to parse
- Shuffle indices (not the data itself) to avoid memory copies

---

**Next**: [13. LeNet and AlexNet](./13_LeNet_and_AlexNet.md) — Build the first two landmark CNNs from scratch in C, connecting all the primitives from the previous lessons into a complete training pipeline.
