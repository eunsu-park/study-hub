/*
 * vit_demo.c -- Vision Transformer (ViT) forward pass demo
 *
 * Demonstrates: patch embedding, CLS token, position embeddings,
 * transformer blocks, and classification from CLS output.
 * Uses a tiny synthetic image (8x8, 3 channels) and 4x4 patches.
 *
 * Compile: gcc -std=c11 -Wall -Wextra -O2 -o vit_demo vit_demo.c -lm
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- helpers ---- */

static float randf(void) {
    return (float)rand() / (float)RAND_MAX - 0.5f;
}

static void rand_init(float *a, int n, float scale) {
    for (int i = 0; i < n; i++) a[i] = randf() * scale;
}

/* ---- LayerNorm ---- */

static void layernorm(const float *x, const float *w, const float *b,
                      float *out, int d) {
    float mean = 0.0f;
    for (int i = 0; i < d; i++) mean += x[i];
    mean /= (float)d;
    float var = 0.0f;
    for (int i = 0; i < d; i++) var += (x[i] - mean) * (x[i] - mean);
    var /= (float)d;
    float rstd = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < d; i++)
        out[i] = (x[i] - mean) * rstd * w[i] + b[i];
}

/* ---- GELU ---- */

static float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

/* ---- Patch Embedding ---- */

static void patch_embed(const float *image, const float *proj_w,
                        const float *proj_b, float *patches,
                        int C, int H, int W, int P, int d_model) {
    int n_h = H / P;
    int n_w = W / P;
    int patch_dim = C * P * P;

    for (int ph = 0; ph < n_h; ph++) {
        for (int pw = 0; pw < n_w; pw++) {
            int pidx = ph * n_w + pw;
            /* Flatten patch into a temporary vector */
            float *flat = (float *)malloc((size_t)patch_dim * sizeof(float));
            int col = 0;
            for (int c = 0; c < C; c++)
                for (int i = 0; i < P; i++)
                    for (int j = 0; j < P; j++)
                        flat[col++] = image[c * H * W + (ph * P + i) * W + (pw * P + j)];

            /* Linear projection: flat[patch_dim] @ proj_w[d_model, patch_dim]^T */
            for (int d = 0; d < d_model; d++) {
                float s = proj_b[d];
                for (int k = 0; k < patch_dim; k++)
                    s += flat[k] * proj_w[d * patch_dim + k];
                patches[pidx * d_model + d] = s;
            }
            free(flat);
        }
    }
}

/* ---- Multi-Head Self Attention ---- */

static void mhsa(const float *x, float *out, int T, int d, int n_heads,
                 const float *qkv_w, const float *qkv_b,
                 const float *proj_w, const float *proj_b) {
    int d_head = d / n_heads;
    int qkv_dim = 3 * d;

    /* Compute QKV for all tokens */
    float *qkv = (float *)malloc((size_t)T * qkv_dim * sizeof(float));
    for (int t = 0; t < T; t++)
        for (int j = 0; j < qkv_dim; j++) {
            float s = qkv_b[j];
            for (int k = 0; k < d; k++)
                s += x[t * d + k] * qkv_w[j * d + k];
            qkv[t * qkv_dim + j] = s;
        }

    /* Per-head attention (no causal mask for ViT) */
    float *head_out = (float *)calloc((size_t)T * d, sizeof(float));
    float scale = 1.0f / sqrtf((float)d_head);

    for (int h = 0; h < n_heads; h++) {
        for (int tq = 0; tq < T; tq++) {
            const float *q = qkv + tq * qkv_dim + h * d_head;
            float *scores = (float *)malloc((size_t)T * sizeof(float));

            for (int tk = 0; tk < T; tk++) {
                const float *k = qkv + tk * qkv_dim + d + h * d_head;
                float dot = 0.0f;
                for (int j = 0; j < d_head; j++) dot += q[j] * k[j];
                scores[tk] = dot * scale;
            }

            /* softmax */
            float mx = scores[0];
            for (int t = 1; t < T; t++) if (scores[t] > mx) mx = scores[t];
            float sum = 0.0f;
            for (int t = 0; t < T; t++) { scores[t] = expf(scores[t] - mx); sum += scores[t]; }
            for (int t = 0; t < T; t++) scores[t] /= sum;

            /* weighted sum of V */
            float *o = head_out + tq * d + h * d_head;
            for (int t = 0; t < T; t++) {
                const float *v = qkv + t * qkv_dim + 2 * d + h * d_head;
                for (int j = 0; j < d_head; j++)
                    o[j] += scores[t] * v[j];
            }
            free(scores);
        }
    }

    /* Output projection */
    for (int t = 0; t < T; t++)
        for (int j = 0; j < d; j++) {
            float s = proj_b[j];
            for (int k = 0; k < d; k++)
                s += head_out[t * d + k] * proj_w[j * d + k];
            out[t * d + j] = s;
        }

    free(qkv);
    free(head_out);
}

/* ---- FFN (GELU) ---- */

static void ffn(const float *x, float *out, int d, int d_ffn,
                const float *w1, const float *b1,
                const float *w2, const float *b2) {
    float *hidden = (float *)malloc((size_t)d_ffn * sizeof(float));
    for (int j = 0; j < d_ffn; j++) {
        float s = b1[j];
        for (int k = 0; k < d; k++)
            s += x[k] * w1[j * d + k];
        hidden[j] = gelu(s);
    }
    for (int j = 0; j < d; j++) {
        float s = b2[j];
        for (int k = 0; k < d_ffn; k++)
            s += hidden[k] * w2[j * d_ffn + k];
        out[j] = s;
    }
    free(hidden);
}

/* ---- Transformer Block ---- */

typedef struct {
    float *ln1_w, *ln1_b;
    float *qkv_w, *qkv_b;
    float *proj_w, *proj_b;
    float *ln2_w, *ln2_b;
    float *ffn1_w, *ffn1_b;
    float *ffn2_w, *ffn2_b;
} TransBlock;

static void transformer_block(const float *x, float *y, int T, int d,
                               int n_heads, int d_ffn, TransBlock *blk) {
    float *ln_out  = (float *)malloc((size_t)T * d * sizeof(float));
    float *attn_out = (float *)malloc((size_t)T * d * sizeof(float));
    float *x1      = (float *)malloc((size_t)T * d * sizeof(float));
    float *ln2_out = (float *)malloc((size_t)T * d * sizeof(float));
    float *ffn_out = (float *)malloc((size_t)T * d * sizeof(float));

    /* Pre-norm MHSA */
    for (int t = 0; t < T; t++)
        layernorm(x + t * d, blk->ln1_w, blk->ln1_b, ln_out + t * d, d);
    mhsa(ln_out, attn_out, T, d, n_heads, blk->qkv_w, blk->qkv_b,
         blk->proj_w, blk->proj_b);
    for (int i = 0; i < T * d; i++) x1[i] = x[i] + attn_out[i];

    /* Pre-norm FFN */
    for (int t = 0; t < T; t++) {
        layernorm(x1 + t * d, blk->ln2_w, blk->ln2_b, ln2_out + t * d, d);
        ffn(ln2_out + t * d, ffn_out + t * d, d, d_ffn,
            blk->ffn1_w, blk->ffn1_b, blk->ffn2_w, blk->ffn2_b);
    }
    for (int i = 0; i < T * d; i++) y[i] = x1[i] + ffn_out[i];

    free(ln_out); free(attn_out); free(x1); free(ln2_out); free(ffn_out);
}

/* ---- main ---- */

int main(void) {
    srand(42);

    /* ViT config (tiny) */
    const int C = 3;
    const int H = 8, W = 8;
    const int P = 4;             /* patch size */
    const int n_patches = (H / P) * (W / P);  /* 4 */
    const int T = n_patches + 1; /* +1 for CLS = 5 */
    const int d = 16;            /* model dim */
    const int n_heads = 2;
    const int d_ffn = 32;
    const int n_layers = 2;
    const int n_classes = 3;
    const int patch_dim = C * P * P;  /* 48 */

    printf("=== Vision Transformer (ViT) Demo ===\n");
    printf("Image: %dx%dx%d, Patch: %dx%d, Patches: %d, CLS+Patches: %d\n",
           C, H, W, P, P, n_patches, T);
    printf("d_model=%d, n_heads=%d, d_ffn=%d, n_layers=%d, n_classes=%d\n\n",
           d, n_heads, d_ffn, n_layers, n_classes);

    /* Synthetic image (random pixels) */
    float *image = (float *)malloc((size_t)C * H * W * sizeof(float));
    for (int i = 0; i < C * H * W; i++) image[i] = (float)rand() / (float)RAND_MAX;

    /* Patch embedding weights */
    float *proj_w = (float *)malloc((size_t)d * patch_dim * sizeof(float));
    float *proj_b = (float *)calloc((size_t)d, sizeof(float));
    rand_init(proj_w, d * patch_dim, 0.1f);

    /* CLS token and positional embeddings */
    float *cls_tok = (float *)malloc((size_t)d * sizeof(float));
    float *pos_emb = (float *)malloc((size_t)T * d * sizeof(float));
    rand_init(cls_tok, d, 0.1f);
    rand_init(pos_emb, T * d, 0.1f);

    /* Transformer blocks */
    TransBlock blocks[2];
    for (int l = 0; l < n_layers; l++) {
        blocks[l].ln1_w  = (float *)malloc((size_t)d * sizeof(float));
        blocks[l].ln1_b  = (float *)calloc((size_t)d, sizeof(float));
        blocks[l].qkv_w  = (float *)malloc((size_t)3 * d * d * sizeof(float));
        blocks[l].qkv_b  = (float *)calloc((size_t)3 * d, sizeof(float));
        blocks[l].proj_w = (float *)malloc((size_t)d * d * sizeof(float));
        blocks[l].proj_b = (float *)calloc((size_t)d, sizeof(float));
        blocks[l].ln2_w  = (float *)malloc((size_t)d * sizeof(float));
        blocks[l].ln2_b  = (float *)calloc((size_t)d, sizeof(float));
        blocks[l].ffn1_w = (float *)malloc((size_t)d_ffn * d * sizeof(float));
        blocks[l].ffn1_b = (float *)calloc((size_t)d_ffn, sizeof(float));
        blocks[l].ffn2_w = (float *)malloc((size_t)d * d_ffn * sizeof(float));
        blocks[l].ffn2_b = (float *)calloc((size_t)d, sizeof(float));

        for (int i = 0; i < d; i++) { blocks[l].ln1_w[i] = 1.0f; blocks[l].ln2_w[i] = 1.0f; }
        rand_init(blocks[l].qkv_w, 3 * d * d, 0.1f);
        rand_init(blocks[l].proj_w, d * d, 0.1f);
        rand_init(blocks[l].ffn1_w, d_ffn * d, 0.1f);
        rand_init(blocks[l].ffn2_w, d * d_ffn, 0.1f);
    }

    /* Final LayerNorm */
    float *final_ln_w = (float *)malloc((size_t)d * sizeof(float));
    float *final_ln_b = (float *)calloc((size_t)d, sizeof(float));
    for (int i = 0; i < d; i++) final_ln_w[i] = 1.0f;

    /* Classification head */
    float *head_w = (float *)malloc((size_t)n_classes * d * sizeof(float));
    float *head_b = (float *)calloc((size_t)n_classes, sizeof(float));
    rand_init(head_w, n_classes * d, 0.1f);

    /* ---- Forward Pass ---- */

    /* 1. Patch embedding */
    float *patches = (float *)malloc((size_t)n_patches * d * sizeof(float));
    patch_embed(image, proj_w, proj_b, patches, C, H, W, P, d);
    printf("Step 1: Patch embedding done (%d patches -> [%d, %d])\n",
           n_patches, n_patches, d);

    /* 2. Prepend CLS + add positional embeddings */
    float *x = (float *)malloc((size_t)T * d * sizeof(float));
    memcpy(x, cls_tok, (size_t)d * sizeof(float));  /* CLS at position 0 */
    memcpy(x + d, patches, (size_t)n_patches * d * sizeof(float));
    for (int t = 0; t < T; t++)
        for (int j = 0; j < d; j++)
            x[t * d + j] += pos_emb[t * d + j];
    printf("Step 2: CLS + position embeddings -> [%d, %d]\n", T, d);

    /* 3. Transformer blocks */
    float *y = (float *)malloc((size_t)T * d * sizeof(float));
    for (int l = 0; l < n_layers; l++) {
        transformer_block(x, y, T, d, n_heads, d_ffn, &blocks[l]);
        float *tmp = x; x = y; y = tmp;
        printf("Step 3.%d: Transformer block %d done\n", l + 1, l);
    }
    free(y);

    /* 4. Final LayerNorm */
    float *ln_out = (float *)malloc((size_t)T * d * sizeof(float));
    for (int t = 0; t < T; t++)
        layernorm(x + t * d, final_ln_w, final_ln_b, ln_out + t * d, d);
    printf("Step 4: Final LayerNorm done\n");

    /* 5. Extract CLS output (position 0) and classify */
    float *cls_out = ln_out;  /* first d values */
    float logits[3];
    for (int c = 0; c < n_classes; c++) {
        float s = head_b[c];
        for (int k = 0; k < d; k++)
            s += cls_out[k] * head_w[c * d + k];
        logits[c] = s;
    }
    printf("Step 5: Classification from CLS token\n\n");

    /* Softmax over logits */
    float mx = logits[0];
    for (int c = 1; c < n_classes; c++) if (logits[c] > mx) mx = logits[c];
    float sum = 0.0f;
    float probs[3];
    for (int c = 0; c < n_classes; c++) { probs[c] = expf(logits[c] - mx); sum += probs[c]; }
    for (int c = 0; c < n_classes; c++) probs[c] /= sum;

    printf("Classification logits and probabilities:\n");
    const char *class_names[] = {"cat", "dog", "bird"};
    int predicted = 0;
    for (int c = 0; c < n_classes; c++) {
        printf("  %-6s: logit=%7.4f  prob=%.4f\n", class_names[c], logits[c], probs[c]);
        if (probs[c] > probs[predicted]) predicted = c;
    }
    printf("\nPredicted class: %s (prob=%.4f)\n", class_names[predicted], probs[predicted]);

    printf("\n--- ViT Architecture Summary ---\n");
    printf("  Image [%d,%d,%d] -> %d patches of %dx%d\n", C, H, W, n_patches, P, P);
    printf("  Each patch flattened to %d dims, projected to %d dims\n", patch_dim, d);
    printf("  [CLS] token prepended -> %d tokens total\n", T);
    printf("  Learned positional embedding [%d, %d]\n", T, d);
    printf("  %d transformer blocks (no causal mask)\n", n_layers);
    printf("  Final CLS representation -> classification head\n");

    /* Cleanup */
    free(image); free(proj_w); free(proj_b);
    free(cls_tok); free(pos_emb); free(patches);
    free(x); free(ln_out);
    free(final_ln_w); free(final_ln_b);
    free(head_w); free(head_b);
    for (int l = 0; l < n_layers; l++) {
        free(blocks[l].ln1_w);  free(blocks[l].ln1_b);
        free(blocks[l].qkv_w);  free(blocks[l].qkv_b);
        free(blocks[l].proj_w); free(blocks[l].proj_b);
        free(blocks[l].ln2_w);  free(blocks[l].ln2_b);
        free(blocks[l].ffn1_w); free(blocks[l].ffn1_b);
        free(blocks[l].ffn2_w); free(blocks[l].ffn2_b);
    }

    return 0;
}
