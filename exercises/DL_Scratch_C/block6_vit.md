# Block 6 — Vision Transformer (L31–L33)

Prerequisites: L31 (patch embedding), L32 (ViT architecture and parameter count), L33 (CLIP / contrastive learning).

Compile: `gcc -std=c11 -Wall -Wextra -O2 -o ex ex.c -lm`

---

## Exercise 6.1 — Verify `patch_embed_forward` Against `conv2d_naive`

**Difficulty**: ★★

### Problem

ViT splits an image into non-overlapping P×P patches and projects each patch to a D-dimensional embedding. This is mathematically equivalent to a 2D convolution with kernel size P, stride P, and output channels D.

Implement `patch_embed_forward` that:
1. Extracts each P×P patch (for a single-channel image, for simplicity).
2. Flattens the patch to a vector of length P*P.
3. Multiplies by the projection matrix W `[D, P*P]` to produce a D-dimensional embedding.

Then verify that the output matches a `conv2d_naive` call with kernel size P and stride P.

### Starter Code

```c
#include <stdio.h>
#include <string.h>
#include <math.h>

/*
 * patch_embed_forward
 *   img : [H][W]  single channel
 *   W   : [D][P*P]  projection matrix
 *   out : [N_patches][D]  where N_patches = (H/P)*(W/P)
 */
void patch_embed_forward(const float *img, int H, int W,
                         const float *Wproj, int P, int D,
                         float *out) {
    int n_h = H / P;
    int n_w = W / P;
    /* TODO: for each patch (ph, pw):
         1. flatten patch pixels into tmp[P*P]
         2. multiply Wproj[D][P*P] by tmp to get out[(ph*n_w+pw)*D ... +D] */
}

/*
 * conv2d_naive: kernel size K=P, stride S=P, no padding, single channel in/out
 * Treat Wproj rows as separate filter kernels.
 * out[d][ph][pw] = sum_{ki,kj} Wproj[d][ki*P+kj] * img[(ph*P+ki)*W + (pw*P+kj)]
 * We store out as [N_patches][D] = [(ph*n_w+pw)*D + d]
 */
void conv2d_stride_P(const float *img, int H, int W,
                     const float *Wproj, int P, int D,
                     float *out) {
    int n_h = H / P;
    int n_w = W / P;
    memset(out, 0, n_h * n_w * D * sizeof(float));
    for (int d = 0; d < D; d++)
        for (int ph = 0; ph < n_h; ph++)
            for (int pw = 0; pw < n_w; pw++) {
                float s = 0;
                for (int ki = 0; ki < P; ki++)
                    for (int kj = 0; kj < P; kj++)
                        s += Wproj[d*P*P + ki*P+kj] *
                             img[(ph*P+ki)*W + (pw*P+kj)];
                out[(ph*n_w+pw)*D + d] = s;
            }
}

int main(void) {
    int H=8, W=8, P=4, D=3;
    int n_patches = (H/P) * (W/P);  /* 4 patches */

    float img[8*8];
    for (int i = 0; i < 64; i++) img[i] = (float)i * 0.01f;

    /* Random-ish projection matrix */
    float Wproj[3*16];
    for (int i = 0; i < D*P*P; i++) Wproj[i] = (float)(i%7 - 3) * 0.1f;

    float out_pe[4*3], out_conv[4*3];
    patch_embed_forward(img, H, W, Wproj, P, D, out_pe);
    conv2d_stride_P(img, H, W, Wproj, P, D, out_conv);

    float max_diff = 0;
    for (int i = 0; i < n_patches * D; i++) {
        float d = fabsf(out_pe[i] - out_conv[i]);
        if (d > max_diff) max_diff = d;
    }
    printf("patch_embed vs conv2d max diff: %.2e (expected < 1e-5)\n", max_diff);

    printf("First patch embedding: ");
    for (int d = 0; d < D; d++) printf("%.4f ", out_pe[d]);
    printf("\n");
    return 0;
}
```

### Test Cases

- For any `img`, `Wproj`, P, D: the max element-wise difference between `patch_embed_forward` and `conv2d_stride_P` must be less than 1e-5.
- With an all-ones image and all-ones `Wproj`: each output element equals `P*P` (sum of P*P ones).
- `n_patches = (H/P)*(W/P)` — for H=W=224, P=16: 196 patches.

### Hints

1. The patch at grid position `(ph, pw)` contains pixels `img[(ph*P+ki)*W + pw*P+kj]` for `ki,kj` in `[0,P)`.
2. Flatten the patch into `tmp[P*P]` then compute the matrix-vector product `Wproj @ tmp`.
3. The output layout `[N_patches][D]` puts all D embeddings for patch `p` contiguously.

### Solution Approach

Extract each patch into a temporary buffer, then compute a length-D vector via a dot product of each row of `Wproj` with the flattened patch. The equivalence with strided convolution follows directly from comparing the index formulas — both compute the same weighted sum over the same set of input pixels for each output position.

---

## Exercise 6.2 — Count ViT-Base Parameters

**Difficulty**: ★★

### Problem

Write a C program to compute the total parameter count of ViT-Base and verify it matches the published value of **86.6M parameters**.

ViT-Base/16 configuration:
- Image size: 224×224, patch size P=16
- N_patches = (224/16)² = 196
- Sequence length: N_patches + 1 (CLS token) = 197
- d_model (D) = 768
- n_heads = 12, d_head = 64
- n_layers = 12
- d_ff (MLP hidden) = 3072
- Vocabulary (classification head) = 1000

### Component Formulas

| Component | Parameters |
|-----------|-----------|
| Patch embedding (linear proj) | `P*P*3 * D + D` (with bias, 3 input channels) |
| CLS token | `D` |
| Position embeddings | `(N+1) * D` (N=196) |
| Per-layer: LN1 (weight+bias) | `2*D` |
| Per-layer: Attention QKV proj | `D * 3*D + 3*D` |
| Per-layer: Attention out proj | `D * D + D` |
| Per-layer: LN2 (weight+bias) | `2*D` |
| Per-layer: MLP FC1 | `D * d_ff + d_ff` |
| Per-layer: MLP FC2 | `d_ff * D + D` |
| Final LN | `2*D` |
| Classification head | `D * 1000 + 1000` |

### Starter Code

```c
#include <stdio.h>
#include <stdint.h>

int main(void) {
    int P=16, C_img=3, D=768, N=196, d_ff=3072, n_layers=12, n_class=1000;

    int64_t total = 0;

    /* Patch embedding */
    int64_t patch_embed = (int64_t)P*P*C_img * D + D;
    total += patch_embed;
    printf("Patch embedding:   %10lld\n", (long long)patch_embed);

    /* CLS token */
    int64_t cls = D;
    total += cls;
    printf("CLS token:         %10lld\n", (long long)cls);

    /* Position embeddings */
    int64_t pos_embed = (int64_t)(N+1) * D;
    total += pos_embed;
    printf("Position embed:    %10lld\n", (long long)pos_embed);

    /* Transformer layers */
    int64_t per_layer = 0;
    /* TODO: add LN1, QKV proj, out proj, LN2, MLP FC1, MLP FC2 */
    printf("Per layer:         %10lld\n", (long long)per_layer);
    total += n_layers * per_layer;
    printf("12 layers total:   %10lld\n", (long long)(n_layers * per_layer));

    /* Final LN */
    int64_t final_ln = 2 * D;
    total += final_ln;
    printf("Final LN:          %10lld\n", (long long)final_ln);

    /* Classification head */
    int64_t head = (int64_t)D * n_class + n_class;
    total += head;
    printf("Class head:        %10lld\n", (long long)head);

    printf("\nTotal:             %10lld\n", (long long)total);
    printf("Expected:          %10lld\n", (long long)86567656);
    printf("Difference:        %10lld\n", (long long)(total - 86567656));
    return 0;
}
```

### Test Cases

| Component | Expected count |
|-----------|---------------|
| Patch embedding | 590,592 |
| Per-layer (all components) | 7,087,872 |
| 12 layers | 85,054,464 |
| **Total** | **~86,567,656** (~86.6M) |

### Hints

1. QKV projection: input `D`, output `3*D` (Q, K, V concatenated) → `D * 3*D + 3*D` parameters.
2. Out projection: input `D`, output `D` → `D*D + D`.
3. MLP FC1: `D → d_ff` → `D*d_ff + d_ff`.
4. MLP FC2: `d_ff → D` → `d_ff*D + D`.

### Solution Approach

Fill in the `per_layer` calculation with the six components per transformer block. This exercise builds intuition about where ViT parameters live: the MLP layers dominate (4*D^2 per layer), followed by attention projections (4*D^2 per layer). The patch embedding and classification head are small by comparison.

---

## Exercise 6.3 — InfoNCE Loss for Image-Text Pairs

**Difficulty**: ★★★

### Problem

CLIP is trained with InfoNCE (contrastive) loss. Given a batch of N image embeddings `I[N][D]` and N text embeddings `T[N][D]` (L2-normalized), compute the InfoNCE loss:

```
logits[i][j] = I[i] · T[j] * tau          (tau = temperature, e.g. exp(log_tau))
L_img = -mean_i( log( softmax(logits[i])[i] ) )   (images predict their text)
L_txt = -mean_j( log( softmax(logits[:,j])[j] ) )  (texts predict their image)
L     = (L_img + L_txt) / 2
```

The diagonal is the positive pair; all off-diagonal entries are negatives.

Implement for N=4, D=3 with provided unit embeddings.

### Starter Code

```c
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <float.h>

#define N 4
#define D 3

float dot(const float *a, const float *b, int d) {
    float s = 0;
    for (int i = 0; i < d; i++) s += a[i] * b[i];
    return s;
}

float log_softmax_diag(const float *row, int n, int diag_idx) {
    /* Returns log(softmax(row)[diag_idx]) stably */
    float max_v = -FLT_MAX;
    for (int j = 0; j < n; j++) if (row[j] > max_v) max_v = row[j];
    float s = 0;
    for (int j = 0; j < n; j++) s += expf(row[j] - max_v);
    return (row[diag_idx] - max_v) - logf(s);
}

/*
 * infonce_loss:
 *   I    : [N][D] image embeddings (unit norm)
 *   T_mat: [N][D] text  embeddings (unit norm)
 *   tau  : temperature scalar
 *   Returns scalar loss
 */
float infonce_loss(const float *I, const float *T_mat, float tau) {
    float logits[N*N];  /* [N][N] */

    /* TODO: logits[i][j] = dot(I[i], T_mat[j]) * tau */

    float L_img = 0.0f;
    /* TODO: for each i, L_img -= log_softmax_diag(logits[i], N, i) */

    float L_txt = 0.0f;
    /* TODO: for each j, compute column j of logits, L_txt -= log_softmax_diag(col_j, N, j) */

    return (L_img / N + L_txt / N) / 2.0f;
}

int main(void) {
    /* Perfect case: I[i] == T[i] (unit vectors), diagonal should dominate */
    float I[N*D] = {
        1,0,0,
        0,1,0,
        0,0,1,
        0.707f,0.707f,0,
    };
    float T[N*D] = {
        1,0,0,
        0,1,0,
        0,0,1,
        0.707f,0.707f,0,
    };

    float tau = 10.0f;  /* high temperature sharpens logits */
    float loss = infonce_loss(I, T, tau);
    printf("InfoNCE loss (perfect pairs, tau=10): %.4f\n", loss);
    /* With perfect alignment and tau=10, loss should be close to -log(1/N)
       = log(4) ≈ 1.3863 for random, but lower for aligned pairs */

    /* Shuffled case: T rows permuted, loss should increase */
    float T_shuffled[N*D] = {
        0,1,0,
        1,0,0,
        0.707f,0.707f,0,
        0,0,1,
    };
    float loss_bad = infonce_loss(I, T_shuffled, tau);
    printf("InfoNCE loss (shuffled pairs, tau=10): %.4f\n", loss_bad);
    printf("Shuffled loss > aligned loss: %s\n", loss_bad > loss ? "YES" : "NO");

    /* Lower bound: with perfect alignment, loss = -log(1) = 0 as tau -> inf */
    printf("\nExpected: aligned loss < shuffled loss\n");
    printf("Minimum possible loss (perfect align + tau->inf): 0.0\n");
    return 0;
}
```

### Test Cases

| Scenario | Expected behavior |
|----------|------------------|
| I[i] == T[i] (aligned), tau=10 | Loss < log(N) = 1.386 |
| I and T randomly shuffled | Loss ≈ log(N) (chance level) or higher |
| tau very large (→∞) with aligned pairs | Loss → 0 |
| tau very small (→0) | Loss → log(N) (uniform attention) |

### Hints

1. The logit matrix is `N×N`; the positive pair is on the diagonal.
2. For the image loss, apply softmax along rows and take the diagonal entry.
3. For the text loss, apply softmax along **columns** — you need to extract each column.
4. `log_softmax_diag` computes `log(exp(x_k) / sum(exp(x_j)))` stably.

### Solution Approach

Build the full `N×N` logit matrix first. Image loss: for each row i, compute log-softmax and take index i. Text loss: for each column j, extract the column into a temporary buffer, compute log-softmax, take index j. Average both losses and divide by 2. The InfoNCE loss is a multi-class cross-entropy where the positive class is always the diagonal, making it an N-way classification problem for each row and column.
