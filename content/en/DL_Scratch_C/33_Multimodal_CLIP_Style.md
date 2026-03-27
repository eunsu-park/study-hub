# 33. Multimodal CLIP-Style Contrastive Learning

**Previous**: [ViT Training and Fine-Tuning](./32_ViT_Training_and_Fine_Tuning.md) | **Next**: [Cross-Entropy Loss](./34_Cross_Entropy_Loss.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Derive and implement InfoNCE loss as symmetric cross-entropy on a cosine similarity matrix
2. Describe the CLIP architecture: ViT image encoder + Transformer text encoder
3. Build the N×N similarity matrix and apply temperature scaling
4. Understand the contrastive objective: pull paired (image, text) embeddings together, push non-pairs apart
5. Implement zero-shot image classification using text prompt embeddings

---

## 1. What Is CLIP?

CLIP (Contrastive Language-Image Pretraining, Radford et al. 2021) trains two encoders jointly:
- **Image encoder**: a ViT (or ResNet) that maps image → embedding vector
- **Text encoder**: a Transformer that maps text → embedding vector

The training signal comes from 400 million (image, caption) pairs scraped from the internet. For each batch of N pairs, CLIP maximizes the cosine similarity of the N matched (image, text) pairs while minimizing the N²−N non-matched pairs.

This produces embeddings where `image_embed("a dog") ≈ text_embed("a photo of a dog")`.

### 1.1 Architecture Summary

```
Image: [3, 224, 224]
  → Patch embedding (ViT)
  → 12 Transformer blocks (d_model=512, heads=8)
  → CLS token → Linear projection → image_embed [512]
  → L2-normalize → image_feat [512]

Text: "a photo of a cat" → BPE tokens [T]
  → Token + positional embedding [T, 512]
  → 12 Transformer blocks (d_model=512, heads=8)
  → [EOS] token → Linear projection → text_embed [512]
  → L2-normalize → text_feat [512]
```

Both encoders project to the **same** embedding space of dimension `d_embed`. After L2 normalization, cosine similarity reduces to a dot product.

---

## 2. InfoNCE Loss

### 2.1 Derivation

Given a batch of N (image, text) pairs, we compute an N×N similarity matrix S where:

```
S[i][j] = dot(image_feat[i], text_feat[j]) / temperature
```

Because features are L2-normalized, this is cosine similarity scaled by temperature τ (typically 0.07).

The loss has two symmetric terms:

```
L_image = (1/N) Σ_i  -log[ exp(S[i][i]) / Σ_j exp(S[i][j]) ]
L_text  = (1/N) Σ_j  -log[ exp(S[j][j]) / Σ_i exp(S[i][j]) ]
L_total = (L_image + L_text) / 2
```

`L_image`: for each image, the paired text should be the most similar among all N texts.
`L_text`: for each text, the paired image should be the most similar among all N images.

This is precisely cross-entropy loss applied to row-wise (L_image) and column-wise (L_text) distributions, with the diagonal as the correct class.

### 2.2 Implementation

```c
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/*
 * l2_normalize — normalize each row of [N, D] to unit norm.
 */
void l2_normalize(float *x, int N, int D) {
    for (int i = 0; i < N; i++) {
        float *row = x + i * D;
        float norm2 = 0.0f;
        for (int d = 0; d < D; d++) norm2 += row[d] * row[d];
        float inv = 1.0f / (sqrtf(norm2) + 1e-8f);
        for (int d = 0; d < D; d++) row[d] *= inv;
    }
}

/*
 * compute_similarity_matrix — builds S[N, N] = image_feat @ text_feat^T / tau.
 *
 * image_feat : [N, D] L2-normalized
 * text_feat  : [N, D] L2-normalized
 * S          : [N, N] output (row = image index, col = text index)
 */
void compute_similarity_matrix(const float *image_feat, const float *text_feat,
                               float *S, int N, int D, float temperature)
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float dot = 0.0f;
            for (int d = 0; d < D; d++) {
                dot += image_feat[i * D + d] * text_feat[j * D + d];
            }
            S[i * N + j] = dot / temperature;
        }
    }
}

/*
 * log_softmax_row — compute log-softmax along each row of [N, N] matrix.
 * Numerically stable: subtract row max before exp.
 * Result written in-place.
 */
void log_softmax_rows(float *S, int N) {
    for (int i = 0; i < N; i++) {
        float *row = S + i * N;
        /* find row max */
        float mx = row[0];
        for (int j = 1; j < N; j++) if (row[j] > mx) mx = row[j];
        /* compute logsumexp */
        float sum = 0.0f;
        for (int j = 0; j < N; j++) sum += expf(row[j] - mx);
        float lse = mx + logf(sum);
        /* log softmax */
        for (int j = 0; j < N; j++) row[j] -= lse;
    }
}

/*
 * nll_diagonal — compute mean negative log-likelihood of diagonal elements.
 * For a matrix where the correct class is always index i for row i.
 */
float nll_diagonal(const float *log_probs, int N) {
    float loss = 0.0f;
    for (int i = 0; i < N; i++) {
        loss -= log_probs[i * N + i];
    }
    return loss / (float)N;
}

/*
 * infonce_loss — full CLIP InfoNCE loss.
 *
 * image_feat : [N, D] L2-normalized image embeddings
 * text_feat  : [N, D] L2-normalized text embeddings
 * temperature: scalar τ (learned in CLIP, fixed here)
 * S_buf      : scratch buffer [N, N] (caller provides)
 *
 * Returns scalar loss.
 */
float infonce_loss(const float *image_feat, const float *text_feat,
                   int N, int D, float temperature, float *S_buf)
{
    /* Build similarity matrix */
    compute_similarity_matrix(image_feat, text_feat, S_buf, N, D, temperature);

    /* --- Image side: softmax over rows (each image vs all texts) --- */
    /* Make a copy for column-wise (text) side */
    float *S_T = (float *)malloc((size_t)N * N * sizeof(float));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            S_T[j * N + i] = S_buf[i * N + j];   /* transpose */

    log_softmax_rows(S_buf, N);
    float L_image = nll_diagonal(S_buf, N);

    /* --- Text side: softmax over columns = rows of S^T --- */
    log_softmax_rows(S_T, N);
    float L_text = nll_diagonal(S_T, N);

    free(S_T);
    return (L_image + L_text) * 0.5f;
}
```

### 2.3 Backward Pass (Sketch)

The gradient of InfoNCE with respect to the similarity matrix S is:

```
dL/dS[i][j] = (1/(2N)) * (
    (softmax_row[i][j] - 1_{i==j})    /* image side gradient */
  + (softmax_col[i][j] - 1_{i==j})   /* text side gradient  */
)
```

where `1_{i==j}` is 1 on the diagonal, 0 elsewhere. This flows back through the dot product to produce gradients for both encoders.

---

## 3. Temperature Parameter

CLIP learns the temperature τ as a trainable parameter initialized to 0.07. The raw parameter is `log_tau` (so τ = exp(log_tau)), which ensures τ > 0.

```c
typedef struct {
    float log_tau;     /* log of temperature, initialized to log(0.07) */
    float grad_log_tau;
} TemperatureParam;

/* Forward: apply learned temperature */
void apply_temperature(float *S, int N, float tau) {
    int total = N * N;
    for (int i = 0; i < total; i++) S[i] /= tau;
}

/*
 * Backward: gradient w.r.t. log_tau.
 * Since S = dot / exp(log_tau), dL/dlog_tau = -sum(dL/dS * S)
 * (chain rule: d(dot/tau)/d(log_tau) = -dot/tau = -S)
 */
float grad_temperature(const float *dS, const float *S, int N) {
    float g = 0.0f;
    int total = N * N;
    for (int i = 0; i < total; i++) g += dS[i] * S[i];
    return -g;
}
```

Clamp τ to a reasonable range: `τ ∈ [0.01, 1.0]` to prevent the distribution from collapsing.

---

## 4. Zero-Shot Classification

After training, CLIP can classify images into arbitrary categories **without any labeled fine-tuning data**.

### 4.1 Mechanism

For a dataset with K classes, construct a text prompt for each class:

```
"a photo of a {class_name}"
```

Encode all K prompts through the text encoder → `text_feats [K, D]`.
Encode the query image → `image_feat [D]`.
Compute cosine similarities → pick argmax.

### 4.2 Implementation

```c
/*
 * encode_text_prompts — encode K class name prompts and store in text_feats.
 *
 * In a real system this calls the Transformer text encoder.
 * Here we assume text_encoder() is available as a function pointer.
 */
typedef void (*TextEncoderFn)(const int *tokens, int T, float *out, int D);

void encode_class_prompts(const char **class_names, int K,
                          TextEncoderFn encoder,
                          int *token_buf, int max_T,
                          float *text_feats, int D)
{
    for (int k = 0; k < K; k++) {
        /* In practice: tokenize "a photo of a {class_names[k]}" */
        /* Here we leave tokenization as a stub */
        int T = bpe_encode(class_names[k], token_buf, max_T); /* user-defined */
        encoder(token_buf, T, text_feats + k * D, D);
    }
    l2_normalize(text_feats, K, D);
}

/*
 * zero_shot_classify — returns the predicted class index for one image.
 *
 * image_feat  : [D] L2-normalized image embedding
 * text_feats  : [K, D] L2-normalized text embeddings (one per class)
 */
int zero_shot_classify(const float *image_feat, const float *text_feats,
                       int K, int D)
{
    int best_k = 0;
    float best_sim = -1e30f;
    for (int k = 0; k < K; k++) {
        float sim = 0.0f;
        for (int d = 0; d < D; d++) {
            sim += image_feat[d] * text_feats[k * D + d];
        }
        if (sim > best_sim) { best_sim = sim; best_k = k; }
    }
    return best_k;
}

/*
 * zero_shot_topk — return top-k class indices sorted by similarity.
 */
void zero_shot_topk(const float *image_feat, const float *text_feats,
                    int K, int D, int topk, int *indices, float *sims)
{
    /* Compute all similarities */
    float *all_sims = (float *)malloc((size_t)K * sizeof(float));
    for (int k = 0; k < K; k++) {
        float s = 0.0f;
        for (int d = 0; d < D; d++) s += image_feat[d] * text_feats[k * D + d];
        all_sims[k] = s;
        indices[k]  = k;
    }
    /* Partial selection sort for top-k */
    for (int i = 0; i < topk; i++) {
        for (int j = i + 1; j < K; j++) {
            if (all_sims[indices[j]] > all_sims[indices[i]]) {
                int tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
            }
        }
        sims[i] = all_sims[indices[i]];
    }
    free(all_sims);
}
```

### 4.3 Prompt Engineering

CLIP's zero-shot accuracy depends heavily on prompt phrasing. Effective strategies:

```c
/* Ensemble of prompts — average text embeddings over multiple templates */
const char *templates[] = {
    "a photo of a %s",
    "a picture of a %s",
    "a photo of the %s",
    "an image of a %s",
    NULL
};

/*
 * encode_ensembled_prompts — average over templates for one class.
 * Produces more robust text embeddings than a single prompt.
 */
void encode_ensembled_prompts(const char *class_name,
                              const char **templates,
                              TextEncoderFn encoder,
                              int *token_buf, int max_T,
                              float *out, int D)
{
    memset(out, 0, (size_t)D * sizeof(float));
    int count = 0;
    for (int t = 0; templates[t] != NULL; t++) {
        char prompt[256];
        snprintf(prompt, sizeof(prompt), templates[t], class_name);
        /* tokenize prompt → tokens */
        int T = bpe_encode(prompt, token_buf, max_T);
        float *tmp = (float *)malloc((size_t)D * sizeof(float));
        encoder(token_buf, T, tmp, D);
        /* Accumulate (normalize after) */
        for (int d = 0; d < D; d++) out[d] += tmp[d];
        free(tmp);
        count++;
    }
    /* Normalize the averaged embedding */
    float norm2 = 0.0f;
    for (int d = 0; d < D; d++) norm2 += out[d] * out[d];
    float inv = 1.0f / (sqrtf(norm2) + 1e-8f);
    for (int d = 0; d < D; d++) out[d] *= inv;
    (void)count;
}
```

---

## 5. CLIP at Scale: What Makes It Work

| Factor | Effect |
|---|---|
| 400M pairs | Large dataset eliminates need for manual labels |
| InfoNCE temperature 0.07 | Very peaked distribution → strong learning signal |
| Large batch size (32768) | More negatives per step → harder contrastive task |
| Symmetric loss | Both encoders improve jointly each step |
| Separate projection heads | Final embedding space is separate from encoder representations |

### Batch Size and InfoNCE

The quality of InfoNCE loss improves with larger N (more negatives). With N=32768, each image must be distinguished from 32767 other captions — a much harder task than N=64. This is why CLIP uses massive batch sizes distributed across 256 GPUs.

In our C implementation, we are limited by available RAM. For a batch of N=256 images with D=512 embeddings: `256 × 512 × 4 bytes = 512KB` — manageable.

---

## Key Takeaways

- **InfoNCE** is symmetric cross-entropy on a temperature-scaled cosine similarity matrix. The diagonal holds the correct (positive) pairs.
- **Temperature τ** controls distribution sharpness. Low τ (≈0.07) makes the model very confident about its predictions, providing a strong gradient signal.
- **L2 normalization** before computing similarities reduces cosine sim to a simple dot product, making the computation O(N²·D).
- **Zero-shot classification** works by comparing image embeddings to the embeddings of textual class descriptions — no fine-tuning data needed.
- **Prompt ensembling** (averaging embeddings over multiple templates) consistently improves zero-shot accuracy by 2-5% over a single prompt.
- **Batch size is a first-class hyperparameter** for contrastive learning: more negatives = better gradients = better learned representations.
- The CLIP training objective is simple to implement but requires scale (data + compute) to surpass supervised baselines on downstream tasks.

---

**Previous**: [ViT Training and Fine-Tuning](./32_ViT_Training_and_Fine_Tuning.md) | **Next**: [Cross-Entropy Loss](./34_Cross_Entropy_Loss.md)

> Next lesson covers numerically stable cross-entropy for language model training, including the fused softmax-CE backward pass.
