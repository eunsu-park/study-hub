# 32. ViT Training and Fine-Tuning

**Previous**: [Vision Transformer (ViT)](./31_Vision_Transformer_ViT.md) | **Next**: [Multimodal CLIP-Style Learning](./33_Multimodal_CLIP_Style.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement a cosine LR schedule with linear warm-up in C
2. Apply CutMix augmentation at the patch level for ViT inputs
3. Add stochastic depth (drop path) to ViT blocks for regularization
4. Design a fine-tuning strategy: replace classification head, freeze lower layers, gradual unfreezing
5. Compare training ViT from scratch versus fine-tuning a pre-trained model on downstream tasks

---

## 1. Why ViT Training Is Different from CNN Training

CNNs have inductive biases baked in — translation equivariance and local connectivity. ViT has neither. Every token attends to every other token from layer 1. This is powerful but means ViT needs far more data or aggressive regularization to generalize.

Key differences in the training recipe:
- **Longer warm-up**: ViT benefits from slow LR ramp-up (5–20 epochs vs. CNN's 1–5 epochs)
- **Cosine decay**: avoids the abrupt loss spike of step-wise LR drops
- **Strong augmentation**: RandAugment, Mixup, CutMix — not just random crop/flip
- **Stochastic depth**: randomly drop entire Transformer blocks during training
- **Label smoothing**: cross-entropy with smooth targets (0.1) helps calibration

---

## 2. Warm-Up + Cosine LR Schedule

The LR schedule used in DeiT and ViT-Base training:

```
                    lr_max
                   /        \
                  /    cosine \
                 /     decay   \_________ lr_min
     warm-up   /
lr_base ______/
              0   warmup_steps         total_steps
```

```c
#include <math.h>
#include <stddef.h>

typedef struct {
    float lr_min;       /* final LR, e.g. 1e-6                         */
    float lr_max;       /* peak LR, e.g. 1e-3                          */
    int   warmup_steps; /* linear ramp duration                         */
    int   total_steps;  /* total training steps                         */
} LRSchedule;

/* Returns the learning rate for the given step (0-indexed). */
float cosine_lr_with_warmup(const LRSchedule *s, int step) {
    if (step < s->warmup_steps) {
        /* linear warm-up: 0 → lr_max */
        return s->lr_max * (float)(step + 1) / (float)s->warmup_steps;
    }
    /* cosine annealing: lr_max → lr_min */
    int decay_steps = s->total_steps - s->warmup_steps;
    int t = step - s->warmup_steps;
    float progress = (float)t / (float)decay_steps;          /* 0 → 1  */
    float cosine   = 0.5f * (1.0f + cosf((float)M_PI * progress));
    return s->lr_min + (s->lr_max - s->lr_min) * cosine;
}

/* Example: 300-epoch ViT-Base on ImageNet
 *   steps_per_epoch = 1281167 / 1024 ≈ 1251
 *   total_steps = 300 * 1251 = 375300
 *   warmup_steps = 10 * 1251 = 12510  (10-epoch warm-up)
 */
```

Apply this in the training loop by calling `cosine_lr_with_warmup` each step and setting the optimizer's `lr` field before `adam_update()`.

---

## 3. CutMix Augmentation (Patch-Level)

CutMix replaces a rectangular region of image A with the corresponding region from image B and mixes the labels proportionally. For ViT, the natural unit is a **patch** rather than a pixel, so we implement patch-level CutMix: whole patches from image B replace whole patches of image A.

### 3.1 Why CutMix Is Effective for ViT

CutMix forces the model to make decisions from partial information (a masked patch sequence). This is closely related to MAE (masked autoencoder) pre-training — both require the model to be robust to missing or replaced patches.

### 3.2 Implementation

```c
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Uniform random float in [0, 1). */
static float randf(void) { return (float)rand() / ((float)RAND_MAX + 1.0f); }

/*
 * cutmix_patches — patch-level CutMix for ViT inputs.
 *
 * tokens_a, tokens_b : [num_patches, d_model]  (row-major)
 * out                 : [num_patches, d_model]  (output)
 * lambda              : mix ratio drawn from Beta(alpha, alpha)
 *
 * We randomly choose a contiguous rectangular block of patches from tokens_b
 * and paste them into tokens_a.  The actual lambda is recomputed as the
 * fraction of patches replaced, so the label mix matches exactly.
 */
float cutmix_patches(const float *tokens_a, const float *tokens_b,
                     float *out,
                     int grid_h, int grid_w, int d_model,
                     float lambda_target)
{
    int num_patches = grid_h * grid_w;

    /* Copy image A into output first */
    memcpy(out, tokens_a, (size_t)num_patches * d_model * sizeof(float));

    /* Compute cut dimensions proportional to sqrt(1 - lambda) */
    float ratio = sqrtf(1.0f - lambda_target);
    int cut_h = (int)(grid_h * ratio);
    int cut_w = (int)(grid_w * ratio);
    if (cut_h < 1) cut_h = 1;
    if (cut_w < 1) cut_w = 1;

    /* Random top-left corner of the cut region */
    int r0 = (int)(randf() * (grid_h - cut_h + 1));
    int c0 = (int)(randf() * (grid_w - cut_w + 1));

    /* Paste patches from image B */
    for (int r = r0; r < r0 + cut_h; r++) {
        for (int c = c0; c < c0 + cut_w; c++) {
            int idx = r * grid_w + c;          /* patch index */
            const float *src = tokens_b + idx * d_model;
            float       *dst = out       + idx * d_model;
            memcpy(dst, src, (size_t)d_model * sizeof(float));
        }
    }

    /* Actual lambda = fraction of patches from image A */
    float actual_lambda = 1.0f - (float)(cut_h * cut_w) / (float)num_patches;
    return actual_lambda;
}

/*
 * CutMix label mixing:
 *   mixed_loss = lambda * CE(pred, label_a) + (1 - lambda) * CE(pred, label_b)
 */
```

Usage in the training loop:

```c
/* Inside batch loop */
float lambda_target = sample_beta(alpha, alpha);   /* alpha = 1.0 typically */
float lambda = cutmix_patches(
    tokens_a, tokens_b, mixed_tokens,
    14, 14, d_model, lambda_target
);
/* Forward pass on mixed_tokens → logits */
/* loss = lambda * CE(logits, label_a) + (1-lambda) * CE(logits, label_b) */
```

---

## 4. Stochastic Depth (Drop Path)

Stochastic depth randomly bypasses Transformer blocks during training. Each block has a survival probability `p_i` that decreases linearly with depth (deeper layers drop more often).

```
p_i = 1 - (i / L) * drop_rate
```

where `L` is the total number of blocks and `drop_rate` is a hyperparameter (typically 0.1–0.2 for ViT-Base).

### 4.1 Forward Pass with Drop Path

```c
#include <stdlib.h>

/*
 * drop_path_forward — apply stochastic depth to a residual addition.
 *
 * During training, with probability (1 - survival_prob) we skip the residual
 * entirely; otherwise we scale it by 1 / survival_prob to keep the expected
 * value unchanged (Bernoulli scaling).
 *
 * x        : [B, T, d_model]  input (bypass branch)
 * residual : [B, T, d_model]  output of the sub-layer
 * out      : [B, T, d_model]  x + scaled_residual
 * B, T, D  : batch, sequence length, hidden dim
 * survival : survival probability for this block
 * training : 1 = training mode, 0 = inference
 * mask     : pre-allocated [B] boolean mask (caller provides)
 *
 * Returns the per-sample drop mask (1 = kept, 0 = dropped).
 */
void drop_path_forward(const float *x, const float *residual, float *out,
                       int B, int T, int D,
                       float survival, int training, int *mask)
{
    int total = B * T * D;

    if (!training || survival >= 1.0f) {
        /* Inference: always add residual (no scaling needed) */
        for (int i = 0; i < total; i++) out[i] = x[i] + residual[i];
        return;
    }

    /* Sample per-sample Bernoulli mask */
    for (int b = 0; b < B; b++) {
        float u  = (float)rand() / ((float)RAND_MAX + 1.0f);
        mask[b]  = (u < survival) ? 1 : 0;
    }

    float scale = 1.0f / survival;   /* expectation correction */

    for (int b = 0; b < B; b++) {
        float s = mask[b] ? scale : 0.0f;
        for (int t = 0; t < T; t++) {
            for (int d = 0; d < D; d++) {
                int i = (b * T + t) * D + d;
                out[i] = x[i] + s * residual[i];
            }
        }
    }
}

/*
 * drop_path_backward — gradient through drop path.
 *
 * dout     : gradient from upstream [B, T, D]
 * mask     : same mask used in forward  [B]
 * dx       : gradient w.r.t. x         [B, T, D]  (accumulated)
 * dresidual: gradient w.r.t. residual  [B, T, D]  (accumulated)
 */
void drop_path_backward(const float *dout, const int *mask,
                        float *dx, float *dresidual,
                        int B, int T, int D, float survival)
{
    float scale = 1.0f / survival;
    for (int b = 0; b < B; b++) {
        float s = mask[b] ? scale : 0.0f;
        for (int t = 0; t < T; t++) {
            for (int d = 0; d < D; d++) {
                int i = (b * T + t) * D + d;
                dx[i]        += dout[i];          /* bypass always passes grad */
                dresidual[i] += s * dout[i];
            }
        }
    }
}

/* Assign survival probabilities linearly across L blocks */
void compute_survival_probs(float *probs, int L, float drop_rate) {
    for (int i = 0; i < L; i++) {
        probs[i] = 1.0f - ((float)i / (float)L) * drop_rate;
    }
}
```

In ViT-Base with 12 blocks and `drop_rate=0.1`:
- Block 0: survival = 1.00 (never dropped)
- Block 6: survival = 0.95
- Block 11: survival = 0.90

---

## 5. Fine-Tuning Strategy

Fine-tuning a pre-trained ViT (e.g., trained on ImageNet-21k) on a smaller downstream dataset requires care to avoid catastrophic forgetting.

### 5.1 Standard Fine-Tuning Protocol

```
Phase 1 — Head replacement (1-2 epochs):
  - Load pre-trained weights
  - Replace classification head (linear: d_model → num_classes_pretrain)
    with a new head (linear: d_model → num_classes_target)
  - Freeze ALL parameters except the new head
  - Train with high LR (1e-3) to initialize the head

Phase 2 — Gradual unfreezing (rest of training):
  - Unfreeze block 11 (last), train for K steps
  - Unfreeze block 10, train for K steps
  - ...
  - Unfreeze patch embedding + positional encoding last
  - Use low LR (1e-5 to 1e-4) with cosine decay
```

### 5.2 Implementation

```c
typedef struct {
    int   num_blocks;
    int   freeze_until_block; /* blocks 0..freeze_until_block are frozen */
    float head_lr;            /* LR for the classification head          */
    float backbone_lr;        /* LR for unfrozen backbone layers         */
} FinetuneConfig;

/*
 * param_requires_grad — returns 1 if the parameter should be updated.
 * We encode layer ownership via a layer_id:
 *   -1  = patch embedding / positional encoding
 *    0  = block 0
 *   ...
 *   11  = block 11
 *   12  = classification head
 */
int param_requires_grad(int layer_id, const FinetuneConfig *cfg) {
    if (layer_id == 12) return 1;                        /* head: always train  */
    if (layer_id < 0)   return (cfg->freeze_until_block < 0);
    return (layer_id > cfg->freeze_until_block);
}

/*
 * get_lr_for_layer — apply layer-wise LR decay (optional but effective).
 * Each layer gets lr * decay^(L - layer_id).
 */
float get_lr_for_layer(int layer_id, float base_lr, float decay, int L) {
    if (layer_id == L) return base_lr;               /* head gets full LR  */
    int depth = L - layer_id;
    float lr  = base_lr;
    for (int i = 0; i < depth; i++) lr *= decay;     /* lr * decay^depth   */
    return lr;
}
```

### 5.3 Head Resolution Mismatch

When fine-tuning on a dataset with a different input resolution (e.g., pre-trained at 224px, fine-tuning at 384px), the number of patches changes and the positional embeddings must be interpolated:

```c
#include <math.h>

/*
 * interpolate_pos_embed — bicubic interpolation of positional embeddings.
 *
 * src      : [N_src + 1, d_model]  source positional embeddings (includes CLS)
 * dst      : [N_dst + 1, d_model]  output (caller allocates)
 * grid_src : source grid size (e.g., 14 for 224px / 16px patches)
 * grid_dst : target grid size (e.g., 24 for 384px / 16px patches)
 * d_model  : embedding dimension
 *
 * CLS token embedding is copied as-is (no interpolation needed).
 */
void interpolate_pos_embed(const float *src, float *dst,
                           int grid_src, int grid_dst, int d_model)
{
    int N_src = grid_src * grid_src;
    int N_dst = grid_dst * grid_dst;

    /* Copy CLS token (first row) verbatim */
    memcpy(dst, src, (size_t)d_model * sizeof(float));

    const float *src_patches = src + d_model;   /* skip CLS */
    float       *dst_patches = dst + d_model;

    /* Bilinear interpolation (sufficient for pos embed) */
    for (int row = 0; row < grid_dst; row++) {
        for (int col = 0; col < grid_dst; col++) {
            /* Map destination grid position to source grid position */
            float sr = (row + 0.5f) * (float)grid_src / (float)grid_dst - 0.5f;
            float sc = (col + 0.5f) * (float)grid_src / (float)grid_dst - 0.5f;

            int r0 = (int)sr; if (r0 < 0) r0 = 0;
            int c0 = (int)sc; if (c0 < 0) c0 = 0;
            int r1 = r0 + 1;  if (r1 >= grid_src) r1 = grid_src - 1;
            int c1 = c0 + 1;  if (c1 >= grid_src) c1 = grid_src - 1;

            float dr = sr - r0;
            float dc = sc - c0;

            float *out_row = dst_patches + (row * grid_dst + col) * d_model;
            for (int d = 0; d < d_model; d++) {
                float v00 = src_patches[(r0 * grid_src + c0) * d_model + d];
                float v01 = src_patches[(r0 * grid_src + c1) * d_model + d];
                float v10 = src_patches[(r1 * grid_src + c0) * d_model + d];
                float v11 = src_patches[(r1 * grid_src + c1) * d_model + d];
                out_row[d] = v00 * (1-dr)*(1-dc) + v01 * (1-dr)*dc
                           + v10 * dr*(1-dc)      + v11 * dr*dc;
            }
        }
    }
    (void)N_src; (void)N_dst;
}
```

---

## 6. Training from Scratch vs. Fine-Tuning

| Aspect | Scratch Training | Fine-Tuning |
|---|---|---|
| Data required | >10M images (ImageNet-21k scale) | As few as a few thousand |
| Training time | 300 epochs (days on 8× A100) | 10-30 epochs |
| LR schedule | Cosine with 10-epoch warm-up | Cosine with 1-epoch warm-up |
| Batch size | 4096 (with gradient accumulation) | 512 |
| Augmentation | RandAugment + CutMix + Mixup | RandAugment or simple crop/flip |
| Stochastic depth | drop_rate = 0.1 | drop_rate = 0.0 or very small |
| Label smoothing | 0.1 | 0.0 or 0.05 |
| Regularization | Weight decay 0.05, Dropout 0.0 | Weight decay 0.01 |
| Expected accuracy (ImageNet) | ~81.8% (ViT-Base/16) | ~85%+ with ImageNet-21k init |

### Key Insight

ViT trained from scratch on ImageNet-1k (1.28M images) achieves only ~77-78% top-1 — inferior to ResNet-50 (~76%) with far fewer parameters. But ViT-Base pre-trained on ImageNet-21k (14M images) then fine-tuned on ImageNet-1k reaches 85%+. The inductive bias of CNNs matters less when you have enough data.

---

## Key Takeaways

- **Cosine LR with warm-up** is the standard schedule for ViT. Warm-up prevents early instability from large gradients on randomly initialized attention weights.
- **CutMix at patch level** aligns naturally with ViT's tokenized representation and serves as an implicit masking regularizer.
- **Stochastic depth** assigns higher survival probabilities to early layers (preserve features learned from data) and lower to late layers (allow more regularization at the top).
- **Fine-tuning protocol** follows a clear hierarchy: train head first, unfreeze top blocks, then progressively unfreeze deeper layers with layer-wise LR decay.
- **Positional embedding interpolation** (bilinear is sufficient) allows transferring pre-trained models to higher resolutions without re-training from scratch.
- **Scale is the key variable**: ViT needs 10× more pre-training data than ResNet to match CNN inductive bias — but scales better beyond that point.
- **Label smoothing + weight decay** together handle the overconfidence that ViT is prone to due to the high capacity of global attention.

---

**Previous**: [Vision Transformer (ViT)](./31_Vision_Transformer_ViT.md) | **Next**: [Multimodal CLIP-Style Learning](./33_Multimodal_CLIP_Style.md)

> Next lesson covers InfoNCE contrastive loss and CLIP-style zero-shot classification.
