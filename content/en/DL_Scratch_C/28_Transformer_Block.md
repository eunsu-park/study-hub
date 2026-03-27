# 28. Transformer Block

**Previous**: [FFN and Activations](./27_FFN_and_Activations.md) | **Next**: [GPT-2 Forward Pass](./29_GPT2_Forward_Pass.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Assemble a complete pre-norm Transformer block from the components built so far
2. Implement the full block forward pass (LN → Attention → residual → LN → FFN → residual)
3. Implement the block backward pass tracking all intermediate tensors
4. Verify block output numerically against PyTorch on a random input
5. Stack multiple blocks to form a Transformer decoder

---

## 1. Pre-norm Transformer Block Structure

```
Input X [N, T, d_model]
      │
      ├──────────────────────┐  (residual branch 1)
      │                      │
    LN1(X)                   │
      │                      │
  Attention                  │
      │                      │
      +──────────────────────┘ → X1 = X + Attention(LN1(X))
      │
      ├──────────────────────┐  (residual branch 2)
      │                      │
    LN2(X1)                  │
      │                      │
     FFN                     │
      │                      │
      +──────────────────────┘ → X2 = X1 + FFN(LN2(X1))
      │
   Output X2
```

Contrast with post-norm (original Transformer):
```
X → (Attention → X + residual → LN) → (FFN → X + residual → LN)
```
Pre-norm has a cleaner residual highway: gradient flows directly from output to input through the `+` at each stage.

---

## 2. Block Weights Struct

```c
typedef struct {
    // Layer Norm 1 (before attention)
    float *ln1_w, *ln1_b;  // [d]

    // Multi-head attention
    float *qkv_w, *qkv_b;  // [3d, d] and [3d]
    float *proj_w, *proj_b; // [d, d] and [d]

    // Layer Norm 2 (before FFN)
    float *ln2_w, *ln2_b;  // [d]

    // FFN (GPT-2 style)
    float *fc1_w, *fc1_b;  // [4d, d] and [4d]
    float *fc2_w, *fc2_b;  // [d, 4d] and [d]

    int d, n_heads;
} TransformerBlock;

typedef struct {
    float *ln1_out, *ln1_mean, *ln1_rstd;  // LN1 output and saved stats
    float *attn_qkv;                        // [M, 3d]
    float *attn_w;                          // [N, h, T, T]
    float *head_out;                        // [N, h, T, d_head]
    float *attn_out;                        // [M, d]
    float *x1;                              // X + attn_out
    float *ln2_out, *ln2_mean, *ln2_rstd;  // LN2 output and saved stats
    float *ffn_mid;                         // [M, 4d]  after GELU
    float *ffn_out;                         // [M, d]
} BlockBuffers;
```

---

## 3. Block Forward Pass

```c
// transformer_block_forward: GPT-2 style pre-norm block
void transformer_block_forward(
    TransformerBlock *blk,
    BlockBuffers     *buf,
    const float      *X,      // [M, d]  M = N*T
    float            *Y,      // [M, d]  output
    int N, int T, int d, int n_heads, int training) {

    int M = N * T;
    int d4 = 4 * d;

    // ---- Residual branch 1: Attention ----

    // LN1
    layernorm_forward(X, blk->ln1_w, blk->ln1_b,
                      buf->ln1_out, buf->ln1_mean, buf->ln1_rstd,
                      M, d);

    // QKV + Attention + Proj  (all in one call)
    mha_forward(buf->ln1_out, blk->qkv_w, blk->qkv_b,
                blk->proj_w, blk->proj_b,
                buf->attn_out,
                buf->attn_qkv, buf->attn_w, buf->head_out,
                N, T, d, n_heads, /*causal=*/1);

    // X1 = X + attn_out
    for (int i = 0; i < M * d; i++)
        buf->x1[i] = X[i] + buf->attn_out[i];

    // ---- Residual branch 2: FFN ----

    // LN2
    layernorm_forward(buf->x1, blk->ln2_w, blk->ln2_b,
                      buf->ln2_out, buf->ln2_mean, buf->ln2_rstd,
                      M, d);

    // FFN
    gpt2_ffn_forward(buf->ln2_out,
                     blk->fc1_w, blk->fc1_b,
                     blk->fc2_w, blk->fc2_b,
                     buf->ffn_mid, buf->ffn_out,
                     M, d);

    // Y = X1 + ffn_out
    for (int i = 0; i < M * d; i++)
        Y[i] = buf->x1[i] + buf->ffn_out[i];
}
```

---

## 4. Block Backward Pass

```c
// transformer_block_backward: backprop through pre-norm block
// dY: gradient from next layer or loss; produces dX
void transformer_block_backward(
    TransformerBlock *blk,
    BlockBuffers     *buf,
    const float      *X,       // original input [M, d]
    const float      *dY,      // [M, d]
    float            *dX,      // [M, d] — output
    // gradient accumulators for weights (+=):
    float *dln1w, *dln1b,
    float *dqkvw, *dqkvb, *dprojw, *dprojb,
    float *dln2w, *dln2b,
    float *dfc1w, *dfc1b, *dfc2w, *dfc2b,
    int N, int T, int d, int n_heads) {

    int M = N * T;

    // ---- Branch 2 backward (FFN) ----

    // dY passes through residual: dX1 = dY; dffn_out = dY
    float *dX1      = malloc(M * d * sizeof(float));
    float *dffn_out = malloc(M * d * sizeof(float));
    memcpy(dX1,      dY, M * d * sizeof(float));
    memcpy(dffn_out, dY, M * d * sizeof(float));

    // FFN backward: dffn_out → d(ln2_out), update dfc1w, dfc1b, dfc2w, dfc2b
    float *dln2_out = calloc(M * d, sizeof(float));
    gpt2_ffn_backward(dffn_out, buf->ln2_out, buf->ffn_mid,
                      blk->fc1_w, blk->fc2_w,
                      dln2_out, dfc1w, dfc1b, dfc2w, dfc2b,
                      M, d);
    free(dffn_out);

    // LN2 backward: dln2_out → dX1 (accumulated), update dln2w, dln2b
    float *dX1_from_ln2 = calloc(M * d, sizeof(float));
    layernorm_backward(dln2_out, buf->x1, blk->ln2_w,
                       buf->ln2_mean, buf->ln2_rstd,
                       dX1_from_ln2, dln2w, dln2b,
                       M, d);
    free(dln2_out);
    for (int i = 0; i < M * d; i++) dX1[i] += dX1_from_ln2[i];
    free(dX1_from_ln2);

    // ---- Branch 1 backward (Attention) ----

    // dX1 through residual: dX += dX1; dattn_out = dX1
    float *dattn_out = malloc(M * d * sizeof(float));
    memcpy(dattn_out, dX1, M * d * sizeof(float));
    memset(dX, 0, M * d * sizeof(float));
    for (int i = 0; i < M * d; i++) dX[i] += dX1[i];
    free(dX1);

    // MHA backward: dattn_out → d(ln1_out), update attn weights
    float *dln1_out = calloc(M * d, sizeof(float));
    mha_backward(dattn_out, buf->ln1_out, buf->attn_qkv,
                 buf->attn_w, buf->head_out,
                 blk->qkv_w, blk->proj_w,
                 dln1_out, dqkvw, dqkvb, dprojw, dprojb,
                 N, T, d, n_heads);
    free(dattn_out);

    // LN1 backward: dln1_out → dX (accumulated), update dln1w, dln1b
    float *dX_from_ln1 = calloc(M * d, sizeof(float));
    layernorm_backward(dln1_out, X, blk->ln1_w,
                       buf->ln1_mean, buf->ln1_rstd,
                       dX_from_ln1, dln1w, dln1b,
                       M, d);
    free(dln1_out);
    for (int i = 0; i < M * d; i++) dX[i] += dX_from_ln1[i];
    free(dX_from_ln1);
}
```

---

## 5. Block Numerical Verification

```c
static void test_transformer_block(void) {
    // Small model: d=64, n_heads=4, T=8, N=1
    int d=64, n_heads=4, T=8, N=1, M=T;

    TransformerBlock blk;
    BlockBuffers buf;
    // ... allocate and initialize with small random weights ...

    float *X  = malloc(M * d * sizeof(float));
    float *Y  = malloc(M * d * sizeof(float));
    float *dX = malloc(M * d * sizeof(float));
    float *dY = malloc(M * d * sizeof(float));

    // Random input and upstream gradient
    for (int i = 0; i < M * d; i++) X[i] = randn() * 0.02f;
    for (int i = 0; i < M * d; i++) dY[i] = randn() * 0.1f;

    transformer_block_forward(&blk, &buf, X, Y, N, T, d, n_heads, 1);
    transformer_block_backward(&blk, &buf, X, dY, dX,
                               /* ... gradient pointers ... */,
                               N, T, d, n_heads);

    // Finite difference check on first 5 elements of dX
    float eps = 1e-4f;
    for (int i = 0; i < 5; i++) {
        float *Xp = malloc(M * d * sizeof(float));
        float *Xm = malloc(M * d * sizeof(float));
        float *Yp = malloc(M * d * sizeof(float));
        float *Ym = malloc(M * d * sizeof(float));
        memcpy(Xp, X, M * d * sizeof(float)); Xp[i] += eps;
        memcpy(Xm, X, M * d * sizeof(float)); Xm[i] -= eps;
        // ... re-run forward with Xp and Xm, compute numerical dX[i] ...
        float num = /* dot(dY, Yp - Ym) */ 0.0f / (2 * eps);
        float err = fabsf(dX[i] - num) / (fabsf(num) + 1e-8f);
        printf("dX[%d]: ana=%.5f  num=%.5f  err=%.4f%s\n",
               i, dX[i], num, err, err < 1e-3f ? "" : " FAIL");
        free(Xp); free(Xm); free(Yp); free(Ym);
    }
}
```

---

## 6. Stacking Blocks

```c
// A full GPT-2 decoder stack
typedef struct {
    float *wte, *wpe;          // embeddings
    TransformerBlock *blocks;  // [n_layers]
    BlockBuffers     *bufs;    // [n_layers]
    float *ln_f_w, *ln_f_b;   // final LayerNorm
    int n_layers, d, n_heads, T;
} GPT2;

void gpt2_forward(
    GPT2        *model,
    const int   *tokens,  // [N, T]
    float       *logits,  // [N, T, V]
    int N) {

    int M = N * model->T;
    float *x = malloc(M * model->d * sizeof(float));

    // 1. Embedding
    gpt2_embed_forward(tokens, model->wte, model->wpe, x, N, model->T, model->d);

    // 2. Transformer blocks
    float *y = malloc(M * model->d * sizeof(float));
    for (int l = 0; l < model->n_layers; l++) {
        transformer_block_forward(&model->blocks[l], &model->bufs[l],
                                  x, y, N, model->T, model->d, model->n_heads, 1);
        // swap x and y for next layer
        float *tmp = x; x = y; y = tmp;
    }
    free(y);

    // 3. Final LN
    float *ln_out = malloc(M * model->d * sizeof(float));
    float *dummy_mean = malloc(M * sizeof(float));
    float *dummy_rstd = malloc(M * sizeof(float));
    layernorm_forward(x, model->ln_f_w, model->ln_f_b,
                      ln_out, dummy_mean, dummy_rstd, M, model->d);
    free(x); free(dummy_mean); free(dummy_rstd);

    // 4. Unembed (weight-tied with wte)
    unembed_forward(ln_out, model->wte, logits, M, model->d, 50257);
    free(ln_out);
}
```

---

## Key Takeaways

- **Pre-norm block**: `Y = X + FFN(LN2(X + Attn(LN1(X))))` — LN inside the residual branch, + outside
- Backward has two residual splits: gradient always flows through the `+` connections regardless of the block's content
- Save all intermediate tensors during forward (ln1_out, attn_qkv, attn_w, ffn_mid) for backward
- Block backward = reverse the forward: FFN backward → LN2 backward → add residual → Attn backward → LN1 backward → add residual
- Finite difference verification of each block is essential before stacking — errors compound across layers

---

**Next**: [29. GPT-2 Forward Pass](./29_GPT2_Forward_Pass.md) — Load real GPT-2 (124M) weights from disk and verify the full forward pass output matches HuggingFace to 4+ decimal places.
