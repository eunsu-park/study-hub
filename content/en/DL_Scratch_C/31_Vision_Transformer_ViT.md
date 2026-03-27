# 31. Vision Transformer (ViT)

**Previous**: [Llama Architecture](./30_Llama_Architecture.md) | **Next**: [ViT Training and Fine-Tuning](./32_ViT_Training_and_Fine_Tuning.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement patch embedding that converts a 2D image into a sequence of tokens
2. Prepend the [CLS] classification token
3. Add 2D positional encodings to patch tokens
4. Assemble the full ViT-Base forward pass using the Transformer block from Lesson 28
5. Compare ViT-Base to ResNet-50 on parameter count and compute requirements

---

## 1. ViT Concept: Images as Sequences

ViT (Dosovitskiy et al., 2021) treats an image as a sequence of patches:

```
Image: [3, 224, 224]
Patch size: 16×16 pixels
Number of patches: (224/16) × (224/16) = 14 × 14 = 196 patches

Each patch: [3, 16, 16] = 768 pixels → flattened to 768-dim vector
             → linear projection → d_model-dim token

Sequence: 196 patch tokens + 1 [CLS] token = 197 tokens
  [CLS, patch_0, patch_1, ..., patch_195]

ViT-Base hyperparameters:
  d_model:  768
  n_heads:  12
  n_layers: 12
  d_ffn:    3072
  patch:    16×16
  → 197 tokens × 12 layers of self-attention
```

---

## 2. Patch Embedding

```c
// patch_embed: split image into patches and project to d_model
// image: [N, 3, H, W]  (NCHW)
// proj_w: [d_model, 3, P, P]  — conv weight (equivalent to a K×K conv with stride P)
// proj_b: [d_model]
// output: [N, n_patches, d_model]
void patch_embed_forward(
    const float *image,   // [N, 3, H, W]
    const float *proj_w,  // [d_model, 3*P*P]  (after flattening)
    const float *proj_b,  // [d_model]
    float       *patches, // [N, n_patches, d_model]
    int N, int H, int W, int P, int d_model) {

    int n_h = H / P;  // number of patches vertically
    int n_w = W / P;  // number of patches horizontally
    int n_patches = n_h * n_w;
    int patch_dim = 3 * P * P;  // flattened patch size

    // Extract and flatten each patch into a row
    // Then matmul with proj_w
    float *patch_flat = malloc((long)N * n_patches * patch_dim * sizeof(float));

    for (int n = 0; n < N; n++)
    for (int ph = 0; ph < n_h; ph++)
    for (int pw = 0; pw < n_w; pw++) {
        int patch_idx = ph * n_w + pw;
        float *dst = patch_flat + (long)(n * n_patches + patch_idx) * patch_dim;
        // Copy 3 channels × P × P pixels
        int col = 0;
        for (int c = 0; c < 3; c++)
        for (int i = 0; i < P; i++)
        for (int j = 0; j < P; j++)
            dst[col++] = NCHW(image, N, 3, H, W, n, c, ph*P+i, pw*P+j);
    }

    // Linear projection: [N*n_patches, patch_dim] × [patch_dim, d_model]
    int M = N * n_patches;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, d_model, patch_dim,
                1.0f, patch_flat, patch_dim,
                       proj_w,    patch_dim,
                0.0f, patches, d_model);

    // Add bias
    for (int m = 0; m < M; m++)
    for (int j = 0; j < d_model; j++)
        patches[m * d_model + j] += proj_b[j];

    free(patch_flat);
}
```

---

## 3. CLS Token and Positional Encoding

```c
// vit_embed_forward: patch embed + CLS token + positional encoding
// Output: [N, n_patches+1, d_model]  (CLS at index 0)
void vit_embed_forward(
    const float *image,    // [N, 3, H, W]
    const float *proj_w,   // [d_model, 3*P*P]
    const float *proj_b,   // [d_model]
    const float *cls_tok,  // [d_model]  — learnable [CLS] token
    const float *pos_emb,  // [n_patches+1, d_model]  — learned PE
    float       *output,   // [N, n_patches+1, d_model]
    int N, int H, int W, int P, int d_model) {

    int n_patches = (H / P) * (W / P);
    int T = n_patches + 1;

    // Patch embeddings at positions [1..T-1]
    // First allocate patches without CLS:
    float *patches = malloc((long)N * n_patches * d_model * sizeof(float));
    patch_embed_forward(image, proj_w, proj_b, patches, N, H, W, P, d_model);

    // Assemble: [CLS, patch_0, ..., patch_{n-1}] per sample
    for (int n = 0; n < N; n++) {
        float *out_n = output + (long)n * T * d_model;

        // CLS token at position 0
        memcpy(out_n, cls_tok, d_model * sizeof(float));

        // Patch tokens at positions 1..T-1
        memcpy(out_n + d_model, patches + (long)n * n_patches * d_model,
               (long)n_patches * d_model * sizeof(float));

        // Add positional embedding (broadcast same pos_emb across batch)
        for (int t = 0; t < T; t++)
        for (int j = 0; j < d_model; j++)
            out_n[t * d_model + j] += pos_emb[t * d_model + j];
    }
    free(patches);
}
```

---

## 4. ViT-Base Forward Pass

```c
typedef struct {
    // Patch embedding
    float *proj_w;    // [d_model, 3*P*P]
    float *proj_b;    // [d_model]
    float *cls_tok;   // [d_model]
    float *pos_emb;   // [T, d_model]  T = n_patches+1

    // Transformer encoder blocks
    TransformerBlock *blocks;  // [n_layers]
    BlockBuffers     *bufs;    // [n_layers]

    // Final LayerNorm
    float *ln_w, *ln_b;

    // Classification head (MLP or single linear)
    float *head_w;   // [n_classes, d_model]
    float *head_b;   // [n_classes]

    int n_layers, d_model, n_heads, n_patches, n_classes, P;
} ViT;

// ViT-Base: only the CLS token's final representation → class logits
void vit_forward(
    ViT         *vit,
    const float *image,   // [N, 3, H, W]
    float       *logits,  // [N, n_classes]
    int N, int H, int W) {

    int d = vit->d_model, T = vit->n_patches + 1;
    int M = N * T;

    // 1. Patch embed + CLS + PE
    float *x = malloc(M * d * sizeof(float));
    vit_embed_forward(image, vit->proj_w, vit->proj_b,
                      vit->cls_tok, vit->pos_emb, x,
                      N, H, W, vit->P, d);

    // 2. Transformer encoder blocks (standard pre-norm attention)
    float *y = malloc(M * d * sizeof(float));
    for (int l = 0; l < vit->n_layers; l++) {
        transformer_block_forward(&vit->blocks[l], &vit->bufs[l],
                                  x, y, N, T, d, vit->n_heads, 0);
        float *tmp = x; x = y; y = tmp;
    }
    free(y);

    // 3. Final LayerNorm
    float *ln_out = malloc(M * d * sizeof(float));
    float *mean = malloc(M * sizeof(float)), *rstd = malloc(M * sizeof(float));
    layernorm_forward(x, vit->ln_w, vit->ln_b, ln_out, mean, rstd, M, d);
    free(x); free(mean); free(rstd);

    // 4. Extract CLS token (position 0) for each sample → [N, d]
    float *cls_out = malloc(N * d * sizeof(float));
    for (int n = 0; n < N; n++)
        memcpy(cls_out + n * d, ln_out + (long)n * T * d, d * sizeof(float));
    free(ln_out);

    // 5. Classification head
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                N, vit->n_classes, d,
                1.0f, cls_out, d, vit->head_w, d,
                0.0f, logits, vit->n_classes);
    for (int n = 0; n < N; n++)
    for (int c = 0; c < vit->n_classes; c++)
        logits[n * vit->n_classes + c] += vit->head_b[c];
    free(cls_out);
}
```

---

## 5. ViT-Base Parameter Count

```
ViT-Base (16×16 patches, 224×224 input):
  n_patches = 196,  T = 197,  d = 768,  n_layers = 12

Patch embedding: 3 × 16 × 16 × 768 = 589,824
CLS token:       768
PE table:        197 × 768 = 151,296

Per block (same as GPT-2-style):
  LN1: 2×768 = 1,536
  QKV: 3×768×768 = 1,769,472
  Proj: 768×768 = 589,824
  LN2: 1,536
  FFN: 2×(768×3072) = 4,718,592
  Total: ~7.1M per block

12 blocks: 85.3M
Final LN: 1,536
Head:     768×1000 = 768,000

Total: ~86.6M parameters

Compare ResNet-50: 25.6M params, 4.1B FLOPs
        ViT-Base:  86.6M params, 17.6B FLOPs (for 224×224)
  ViT needs 5B+ ImageNet samples to match ResNet-50 without pretraining tricks
  With DeiT data augmentation or MAE pretraining → matches ResNet at ~1.2M images
```

---

## 6. Patch Embedding vs Conv: Equivalence

```
Patch embedding (no overlap, stride = patch size):
  proj_w [d_model, 3, P, P] = a Conv(3, d_model, kernel=P, stride=P)

The two are mathematically equivalent:
  conv2d(X, proj_w, stride=P) → [N, d_model, H/P, W/P]
  flatten → [N, n_patches, d_model]
  = patch_embed_forward

So patch embedding can be implemented using our existing conv2d_naive function:
  conv2d_naive(image, proj_w, N, 3, H, W, d_model, P, P,
               output_patches, H/P, W/P, P, 0, 1)
  then reshape [N, d_model, H/P, W/P] → [N, (H/P)*(W/P), d_model]
```

---

## Key Takeaways

- **ViT**: split image into P×P patches, flatten each → embed → treat as token sequence for a standard Transformer
- Patch embedding = a large Conv(P×P, stride=P) — can be implemented using conv2d_naive
- **[CLS] token**: prepended learnable vector; its final representation is used for classification
- ViT uses a **learned positional embedding table** (not sinusoidal) — same shape as GPT-2's wpe
- ViT-Base has 86.6M params vs ResNet-50's 25.6M; requires large pretraining datasets (JFT-300M) or strong augmentation (DeiT) to match CNNs

---

**Next**: [32. ViT Training and Fine-Tuning](./32_ViT_Training_and_Fine_Tuning.md) — Training ViT from scratch on ImageNet-scale data: warm-up LR, cosine decay, CutMix augmentation, and fine-tuning a pre-trained ViT.
