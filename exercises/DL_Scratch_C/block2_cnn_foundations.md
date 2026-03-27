# Block 2 — CNN Foundations (L08–L14)

Prerequisites: L08 (im2col), L09 (conv2d forward), L10 (pooling), L11 (BatchNorm), L12 (backprop through conv), L13 (DataLoader), L14 (training loop).

Compile: `gcc -std=c11 -Wall -Wextra -O2 -o ex ex.c -lm`

---

## Exercise 2.1 — `conv2d_dilated`

**Difficulty**: ★★

### Problem

Implement `conv2d_dilated` for a single input channel and single output channel with:
- Kernel size K×K
- Dilation rate `d` (d=1 is standard convolution)
- No padding, stride=1

The output spatial size for input H×W is:
```
H_out = H - d*(K-1)
W_out = W - d*(K-1)
```

Verify that your output shape is correct, then verify correctness against a reference computed with a manually dilated kernel.

### Starter Code

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/*
 * conv2d_dilated: single channel, no padding, stride=1
 *   input : [H][W]
 *   kernel: [K][K]
 *   output: [H_out][W_out]  where H_out = H - d*(K-1), W_out = W - d*(K-1)
 *   d     : dilation factor (>=1)
 */
void conv2d_dilated(const float *input, int H, int W,
                    const float *kernel, int K,
                    float *output, int d) {
    int H_out = H - d * (K - 1);
    int W_out = W - d * (K - 1);
    assert(H_out > 0 && W_out > 0);

    /* TODO: triple-nested loop over output position (i,j) and kernel (ki,kj).
             The input position sampled by kernel element (ki,kj) when computing
             output (i,j) is: input_row = i + ki*d, input_col = j + kj*d       */
}

int main(void) {
    /* 5x5 input, 3x3 kernel, dilation=2 => H_out=1, W_out=1 */
    float input[5*5];
    for (int i = 0; i < 25; i++) input[i] = (float)i;

    float kernel[3*3];
    for (int i = 0; i < 9; i++) kernel[i] = 1.0f;  /* sum filter */

    int H=5, W=5, K=3, d=2;
    int H_out = H - d*(K-1);
    int W_out = W - d*(K-1);
    printf("Output shape: %dx%d (expected 1x1)\n", H_out, W_out);

    float output[1];
    conv2d_dilated(input, H, W, kernel, K, output, d);

    /* With dilation=2 and sum kernel, the sampled positions are
       (0,0),(0,2),(0,4),(2,0),(2,2),(2,4),(4,0),(4,2),(4,4)
       values: 0,2,4,10,12,14,20,22,24 => sum = 108 */
    printf("Output[0] = %.1f (expected 108.0)\n", output[0]);

    /* Also test dilation=1 (standard conv) on same input */
    int H_out1 = H - 1*(K-1);
    int W_out1 = W - 1*(K-1);
    float output1[9];
    conv2d_dilated(input, H, W, kernel, K, output1, 1);
    printf("d=1 output[0] = %.1f (expected 54.0)\n", output1[0]);
    return 0;
}
```

### Test Cases

| Input | Kernel | Dilation | Expected output[0] |
|-------|--------|----------|--------------------|
| 5×5 arange(25), sum-kernel 3×3 | ones | d=2 | 108.0 |
| 5×5 arange(25), sum-kernel 3×3 | ones | d=1 | 54.0 |
| 7×7 ones, sum-kernel 3×3 | ones | d=3 | 9.0 (only 1 output, all ones) |

### Hints

1. The effective receptive field of a dilated kernel is `K_eff = K + (K-1)*(d-1)`.
2. For output position `(i, j)` and kernel tap `(ki, kj)`, the input index is `(i + ki*d, j + kj*d)`.
3. When `d=1` the formula reduces to the standard convolution.

### Solution Approach

The loop structure is identical to standard conv2d except that the input coordinate for kernel tap `(ki, kj)` is scaled by `d`. Four nested loops: over output rows `i`, output cols `j`, kernel rows `ki`, kernel cols `kj`. Compute the output by accumulating `kernel[ki*K+kj] * input[(i+ki*d)*W + (j+kj*d)]`.

---

## Exercise 2.2 — `col2im` and Round-Trip Identity

**Difficulty**: ★★

### Problem

`im2col` reshapes a 2D image into a column matrix so that convolution becomes a matrix multiply. `col2im` is the inverse (transpose) operation used during the backward pass.

Implement `col2im(const float *col, float *img, int H, int W, int K)` for a single channel, no padding, stride=1. Then verify that `col2im(im2col(X)) == X` for a uniform (all-ones) image.

Note: `col2im` **accumulates** (sums overlapping patches), so the round-trip test uses an image where all patches sum to the same value.

### Starter Code

```c
#include <stdio.h>
#include <string.h>
#include <math.h>

/* im2col: input [H][W], kernel K×K, no padding, stride=1
   col shape: [K*K, H_out*W_out] stored row-major */
void im2col(const float *img, int H, int W, int K, float *col) {
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    for (int ki = 0; ki < K; ki++)
        for (int kj = 0; kj < K; kj++)
            for (int i = 0; i < H_out; i++)
                for (int j = 0; j < W_out; j++) {
                    int row = ki * K + kj;
                    int col_idx = i * W_out + j;
                    col[row * (H_out*W_out) + col_idx] =
                        img[(i + ki) * W + (j + kj)];
                }
}

/* col2im: accumulate col back into img (zero img first externally) */
void col2im(const float *col, float *img, int H, int W, int K) {
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    /* TODO: reverse of im2col — for each (ki,kj,i,j) add col value
             back to img[(i+ki)*W + (j+kj)] */
}

int main(void) {
    int H=4, W=4, K=2;
    float img[4*4];
    for (int i = 0; i < H*W; i++) img[i] = 1.0f;  /* uniform image */

    int H_out = H-K+1, W_out = W-K+1;
    float col[K*K * H_out*W_out];
    im2col(img, H, W, K, col);

    /* Now col2im and check all values in recovered img are the same.
       For an all-ones image each pixel is covered by a fixed number
       of patches; the round-trip sums those contributions. */
    float img2[4*4];
    memset(img2, 0, sizeof(img2));
    col2im(col, img2, H, W, K);

    /* Corner pixel (0,0) is covered by exactly 1 patch => img2[0] = 1.0
       Interior pixel (1,1) is covered by K*K=4 patches => img2[1*4+1] = 4.0
       Print and visually verify symmetry */
    printf("Recovered image:\n");
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++)
            printf("%.1f ", img2[i*W+j]);
        printf("\n");
    }
    return 0;
}
```

### Test Cases

For a 4×4 all-ones image with K=2:
- Corner pixels (0,0), (0,3), (3,0), (3,3) should equal **1.0** (covered by 1 patch each).
- Edge pixels should equal **2.0**.
- Interior pixels (1,1), (1,2), (2,1), (2,2) should equal **4.0**.

### Hints

1. `col2im` is the transpose of `im2col` — iterate the same loops and use `+=` instead of `=`.
2. Make sure to zero out `img` before calling `col2im` (the function accumulates).
3. The overlap count for pixel `(r, c)` is `min(r+1, K, H-r) * min(c+1, K, W-c)`.

### Solution Approach

Mirror the `im2col` loop structure exactly, but instead of reading from `img` and writing to `col`, read from `col` and accumulate into `img`. The accumulation is what makes `col2im` the correct adjoint for backpropagation through `im2col`.

---

## Exercise 2.3 — `avgpool_backward`

**Difficulty**: ★★

### Problem

Implement the backward pass for average pooling with non-overlapping windows of size `P×P`:

```
avgpool_forward:  output[i][j] = mean of input patch [i*P:(i+1)*P, j*P:(j+1)*P]
avgpool_backward: distribute d_output[i][j] / (P*P) uniformly to all positions in the patch
```

Verify that the sum of `d_input` equals the sum of `d_output`.

### Starter Code

```c
#include <stdio.h>
#include <string.h>

void avgpool_forward(const float *input, int H, int W,
                     float *output, int P) {
    int H_out = H / P, W_out = W / P;
    for (int i = 0; i < H_out; i++)
        for (int j = 0; j < W_out; j++) {
            float s = 0.0f;
            for (int pi = 0; pi < P; pi++)
                for (int pj = 0; pj < P; pj++)
                    s += input[(i*P+pi)*W + (j*P+pj)];
            output[i*W_out + j] = s / (P*P);
        }
}

void avgpool_backward(const float *d_output, int H_out, int W_out,
                      float *d_input, int P) {
    /* TODO: for each output cell (i,j), spread d_output[i][j]/(P*P)
             to all P*P input positions in the corresponding patch.
             Accumulate with +=. */
}

int main(void) {
    int H=4, W=4, P=2;
    float input[4*4];
    for (int i = 0; i < 16; i++) input[i] = (float)(i+1);

    int H_out=2, W_out=2;
    float output[4], d_output[4], d_input[16];

    avgpool_forward(input, H, W, output, P);
    printf("Forward: %.1f %.1f %.1f %.1f\n",
           output[0], output[1], output[2], output[3]);
    /* Expected: 3.5  5.5  11.5  13.5 */

    /* All-ones gradient from upstream */
    for (int i = 0; i < 4; i++) d_output[i] = 1.0f;
    memset(d_input, 0, sizeof(d_input));
    avgpool_backward(d_output, H_out, W_out, d_input, P);

    float sum_din = 0.0f, sum_dout = 0.0f;
    for (int i = 0; i < 16; i++) sum_din  += d_input[i];
    for (int i = 0; i <  4; i++) sum_dout += d_output[i];
    printf("sum(d_input)=%.2f  sum(d_output)=%.2f (must match)\n",
           sum_din, sum_dout);
    /* All d_input values must be 0.25 (= 1 / (P*P)) */
    printf("d_input[0]=%.2f (expected 0.25)\n", d_input[0]);
    return 0;
}
```

### Test Cases

| Input | P | d_output | Expected each d_input element |
|-------|---|----------|-------------------------------|
| 4×4 any values | 2 | all 1s | 0.25 each |
| 6×6 any values | 3 | all 1s | 1/9 ≈ 0.111 each |

Conservation law: `sum(d_input) == sum(d_output)` must always hold.

### Hints

1. Each output cell `(i, j)` maps to input patch rows `[i*P, (i+1)*P)` and cols `[j*P, (j+1)*P)`.
2. The gradient distributed to each of the P×P input cells is `d_output[i][j] / (P*P)`.
3. Since windows are non-overlapping, accumulation is trivially correct (no pixel is written twice).

### Solution Approach

Two nested loops over output positions `(i, j)`, two inner loops over patch offsets `(pi, pj)`. Compute `d_val = d_output[i*W_out+j] / (float)(P*P)` once per output cell, then add it to the corresponding input gradient positions. The sum conservation test is a good sanity check because average pooling is a linear operation with constant row sums.

---

## Exercise 2.4 — BatchNorm with Momentum EMA

**Difficulty**: ★★

### Problem

Modify a simple 1D BatchNorm (over a batch of vectors) to maintain **running mean** and **running variance** using exponential moving average (EMA) with momentum=0.9:

```
running_mean = momentum * running_mean + (1 - momentum) * batch_mean
running_var  = momentum * running_var  + (1 - momentum) * batch_var
```

During inference (train=false), use `running_mean` and `running_var` instead of batch statistics.

### Starter Code

```c
#include <stdio.h>
#include <math.h>
#include <string.h>

#define EPS 1e-5f

typedef struct {
    float *gamma;      /* scale, shape [C] */
    float *beta;       /* shift, shape [C] */
    float *run_mean;   /* running mean, shape [C] */
    float *run_var;    /* running var,  shape [C] */
    int    C;
    float  momentum;
} BatchNorm;

/* Normalize batch of N vectors, each of dimension C.
   Input/output: x [N][C].
   train=1: compute batch stats, update running stats.
   train=0: use running stats. */
void batchnorm_forward(BatchNorm *bn, float *x, int N, int train) {
    int C = bn->C;
    for (int c = 0; c < C; c++) {
        float mean, var;
        if (train) {
            /* TODO: compute batch mean and variance for feature c */
            mean = 0.0f; var = 0.0f;
            /* ... */

            /* TODO: update running stats with EMA */
        } else {
            /* TODO: use running stats */
            mean = 0.0f; var = 0.0f;
        }

        /* Normalize and apply affine transform */
        float inv_std = 1.0f / sqrtf(var + EPS);
        for (int n = 0; n < N; n++) {
            float x_hat = (x[n*C + c] - mean) * inv_std;
            x[n*C + c] = bn->gamma[c] * x_hat + bn->beta[c];
        }
    }
}

int main(void) {
    int N=4, C=2;
    float x[4*2] = {1,2, 3,4, 5,6, 7,8};
    float gamma[2] = {1,1}, beta[2] = {0,0};
    float run_mean[2] = {0,0}, run_var[2] = {1,1};

    BatchNorm bn = { gamma, beta, run_mean, run_var, C, 0.9f };

    /* First forward pass (training) */
    batchnorm_forward(&bn, x, N, 1);

    /* After training forward, run_mean should have moved toward batch mean */
    printf("run_mean[0] = %.4f (expected ≈ 0.4, i.e. 0.1*mean(1,3,5,7)=0.1*4=0.4)\n",
           run_mean[0]);
    printf("run_mean[1] = %.4f (expected ≈ 0.5, i.e. 0.1*mean(2,4,6,8)=0.1*5=0.5)\n",
           run_mean[1]);

    /* Output should be normalized (mean≈0, std≈1 per feature across batch) */
    float s0=0;
    for (int n = 0; n < N; n++) s0 += x[n*C];
    printf("sum of feature-0 outputs = %.4f (expected ≈ 0.0)\n", s0);
    return 0;
}
```

### Test Cases

After one training forward on batch `[1,2,3,4,5,6,7,8]` (N=4, C=2, momentum=0.9, initial run_mean=0):
- `run_mean[0]` should be `0.9*0 + 0.1*4.0 = 0.4`
- `run_mean[1]` should be `0.9*0 + 0.1*5.0 = 0.5`
- Normalized outputs per feature should sum to ~0 (zero-mean across the batch).

### Hints

1. Batch mean for feature `c`: average `x[n*C + c]` over all `n`.
2. Batch variance (biased): `E[x^2] - E[x]^2`, or equivalently sum of squared deviations / N.
3. The EMA update happens **after** computing batch stats, **before** normalizing.
4. During inference pass, just read `run_mean[c]` and `run_var[c]` directly.

### Solution Approach

For each feature `c`, make one pass over the batch to compute mean, a second pass to compute variance, then update the running statistics, then normalize. The EMA formula is a weighted average that tracks the population statistics across many mini-batches without storing all data.

---

## Exercise 2.5 — Horizontal Flip Augmentation in DataLoader

**Difficulty**: ★

### Problem

Add a horizontal flip augmentation step to a simple image DataLoader. The flip should be applied randomly with probability 0.5 during training.

Implement `hflip_inplace(float *img, int H, int W, int C)` for a CHW-layout image (channels first), then integrate it into the provided `dataloader_next` stub.

### Starter Code

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*
 * Flip image horizontally in-place.
 * Layout: CHW — img[c][h][w] = img[c*H*W + h*W + w]
 */
void hflip_inplace(float *img, int H, int W, int C) {
    /* TODO: for each channel c and row h, swap column w with column (W-1-w)
             for w in [0, W/2) */
}

/* Minimal DataLoader stub */
typedef struct {
    float *images; /* N * C * H * W */
    int    N, C, H, W;
    int    idx;
    int    augment; /* 1=training mode, 0=eval */
} DataLoader;

/* Returns pointer to next image (does NOT copy — apply aug in-place on a scratch buffer) */
float *dataloader_next(DataLoader *dl, float *scratch) {
    int sz = dl->C * dl->H * dl->W;
    float *src = dl->images + dl->idx * sz;
    /* Copy to scratch */
    for (int i = 0; i < sz; i++) scratch[i] = src[i];
    dl->idx = (dl->idx + 1) % dl->N;

    if (dl->augment && (rand() % 2 == 0)) {
        /* TODO: apply hflip_inplace on scratch */
    }
    return scratch;
}

int main(void) {
    srand(42);
    /* Single 1-channel 1×4 "image": [1, 2, 3, 4] */
    float imgs[4] = {1, 2, 3, 4};
    float scratch[4];
    DataLoader dl = { imgs, 1, 1, 4, 1, 0, 1 }; /* augment=1 */

    printf("Original: 1 2 3 4\n");
    /* Run several times to see flip happening */
    for (int t = 0; t < 6; t++) {
        float *out = dataloader_next(&dl, scratch);
        printf("Iter %d: %.0f %.0f %.0f %.0f\n", t, out[0], out[1], out[2], out[3]);
    }
    /* You should see some flipped [4,3,2,1] and some original [1,2,3,4] */
    return 0;
}
```

### Test Cases

- Flipping `[1, 2, 3, 4]` horizontally gives `[4, 3, 2, 1]`.
- For a 3-channel 2×4 image, each channel row should flip independently.
- With augment=0, the output must always match the original.
- Over many iterations with augment=1, roughly half should be flipped.

### Hints

1. Horizontal flip reverses the column dimension (W) within each (channel, row) slice.
2. The swap only needs to go up to `W/2` — swapping beyond the midpoint would undo the flip.
3. Index arithmetic for CHW: element `(c, h, w)` is at offset `c*H*W + h*W + w`.

### Solution Approach

Three nested loops: over channels `c`, rows `h`, and columns `w` up to `W/2`. Swap `img[c*H*W + h*W + w]` with `img[c*H*W + h*W + (W-1-w)]`. This is O(C*H*W/2) — exactly half the image elements are touched per flip.

---

## Exercise 2.6 — Train a 2-Layer CNN on CIFAR-10

**Difficulty**: ★★★

### Problem

Using the primitives you built in L08–L14 (im2col, conv2d, maxpool, batchnorm, relu, fc layer), build and train a 2-layer CNN on CIFAR-10. Report the final test accuracy.

Architecture:
```
Conv(3, 32, K=3, pad=1) -> BN -> ReLU -> MaxPool(2)
Conv(32, 64, K=3, pad=1) -> BN -> ReLU -> MaxPool(2)
Flatten -> FC(64*8*8, 10) -> Softmax
```

Training setup:
- Optimizer: SGD + momentum 0.9
- LR: 0.01, decayed by 0.1 at epoch 50 and 75
- Batch size: 128
- Epochs: 100
- Augmentation: random horizontal flip

Target: test accuracy > 65%.

### Starter Code

```c
/*
 * This is a skeleton. Fill in the missing pieces.
 * You will need: cifar10_load(), your conv2d, batchnorm, maxpool,
 * relu, fc_layer, cross_entropy_loss, and SGD implementations.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Assumed available from earlier lessons */
extern void conv2d_forward(/* ... */);
extern void batchnorm_forward(/* ... */);
extern void maxpool_forward(/* ... */);
extern void fc_forward(/* ... */);
extern void cross_entropy_backward(/* ... */);

#define N_TRAIN 50000
#define N_TEST  10000
#define N_CLASS 10

typedef struct {
    /* Layer weights — allocate and initialize */
    float *conv1_W; /* [32, 3, 3, 3] */
    float *conv2_W; /* [64, 32, 3, 3] */
    float *fc_W;    /* [10, 64*8*8] */
    float *fc_b;    /* [10] */
    /* BN params */
    float *bn1_gamma, *bn1_beta, *bn1_run_mean, *bn1_run_var;
    float *bn2_gamma, *bn2_beta, *bn2_run_mean, *bn2_run_var;
} Model;

void model_init(Model *m) {
    /* TODO: allocate and initialize weights with He initialization */
}

float train_epoch(Model *m, float *images, int *labels, int N, int batch_size, float lr) {
    float total_loss = 0.0f;
    /* TODO: shuffle indices, iterate over mini-batches,
             forward pass, compute loss, backward pass, SGD update */
    return total_loss / N;
}

float evaluate(Model *m, float *images, int *labels, int N) {
    int correct = 0;
    /* TODO: forward pass in eval mode (BN uses running stats, no augmentation),
             argmax of logits, compare to label */
    return (float)correct / N;
}

int main(void) {
    /* TODO: load CIFAR-10 binary format */
    /* See: https://www.cs.toronto.edu/~kriz/cifar.html */

    Model m;
    model_init(&m);

    float lr = 0.01f;
    for (int epoch = 0; epoch < 100; epoch++) {
        if (epoch == 50 || epoch == 75) lr *= 0.1f;
        /* float loss = train_epoch(&m, train_images, train_labels, N_TRAIN, 128, lr); */
        /* float acc  = evaluate(&m, test_images, test_labels, N_TEST); */
        /* printf("Epoch %3d  loss=%.4f  test_acc=%.2f%%\n", epoch, loss, acc*100); */
    }
    return 0;
}
```

### Expected Output

```
Epoch   1  loss=2.1843  test_acc=21.34%
Epoch  10  loss=1.4721  test_acc=48.12%
Epoch  50  loss=0.8832  test_acc=62.50%
Epoch 100  loss=0.6104  test_acc=68.73%
```

*(Exact numbers will vary by random seed. Target: >65% at epoch 100.)*

### Hints

1. He initialization for a layer with fan-in `n`: draw from N(0, sqrt(2/n)).
2. CIFAR-10 binary format: each sample is 1 label byte + 3072 pixel bytes (3×32×32, CHW).
3. Keep BN in training mode during `train_epoch` and eval mode during `evaluate`.
4. If loss diverges, check that you are clipping or not using too large a learning rate.

### Solution Approach

This exercise ties together all the primitives from L08–L14. The key challenges are: (1) correctly connecting layers via intermediate buffers, (2) keeping the backward pass aligned with forward-pass buffers, (3) switching BN between train and eval mode. Start by getting the forward pass to produce reasonable logits before implementing backprop.
