# 29. GPT-2 Forward Pass

**Previous**: [Transformer Block](./28_Transformer_Block.md) | **Next**: [Llama Architecture](./30_Llama_Architecture.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Load GPT-2 (124M) binary weights from disk into a C struct
2. Execute the full forward pass: embedding → 12 blocks → LN → unembed
3. Verify logit outputs match HuggingFace GPT-2 to within 1e-4 absolute error
4. Implement greedy token generation using the forward pass
5. Profile the forward pass to identify the dominant compute bottleneck

---

## 1. GPT-2 Small (124M) Configuration

```
GPT-2 small hyperparameters:
  n_layers:  12
  n_heads:   12
  d_model:   768
  d_head:    64       (768 / 12)
  d_ffn:     3072     (4 × 768)
  T_max:     1024     (max context length)
  V:         50,257   (vocabulary size)

Parameter count:
  Embedding: 50257 × 768 + 1024 × 768 = 39.4M
  Per layer: 4 × (768×768) [QKV+proj] + 2 × (768×3072) [FFN] + 4×768 [LN]
           ≈ 7.1M per layer
  12 layers: 85.2M
  Final LN:  1.5K
  Total:     ~124M (wte tied with output projection)
```

---

## 2. Loading GPT-2 Weights

The llm.c project provides `gpt2_124M.bin` — download and load:

```bash
# Download pre-converted weight file (llm.c format)
wget https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main/gpt2_124M.bin
```

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define GPT2_MAGIC    20240326

// Load GPT-2 weights (llm.c format)
// Returns pointer to allocated weight memory
float *gpt2_load(const char *path, GPT2Config *cfg) {
    FILE *f = fopen(path, "rb");
    assert(f != NULL);

    int header[256];
    fread(header, sizeof(int), 256, f);
    assert(header[0] == GPT2_MAGIC);
    assert(header[1] == 3);  // version

    cfg->max_seq_len       = header[2];
    cfg->vocab_size        = header[3];
    cfg->padded_vocab_size = header[4];
    cfg->n_layers          = header[5];
    cfg->n_heads           = header[6];
    cfg->channels          = header[8];

    printf("Loaded GPT-2: L=%d, H=%d, d=%d, V=%d, T=%d\n",
           cfg->n_layers, cfg->n_heads, cfg->channels,
           cfg->vocab_size, cfg->max_seq_len);

    // Count parameters
    int L = cfg->n_layers, d = cfg->channels;
    int V = cfg->padded_vocab_size, T = cfg->max_seq_len;
    size_t n_params = (size_t)V * d              // wte
                    + (size_t)T * d              // wpe
                    + L * (2*d                   // ln1 w,b
                         + 3*d*d + 3*d           // qkv w,b
                         + d*d + d               // proj w,b
                         + 2*d                   // ln2 w,b
                         + 4*d*d + 4*d           // fc1 w,b
                         + d*4*d + d)            // fc2 w,b
                    + 2*d;                        // lnf w,b

    float *params = malloc(n_params * sizeof(float));
    fread(params, sizeof(float), n_params, f);
    fclose(f);
    printf("Loaded %.2fM parameters\n", n_params / 1e6);
    return params;
}
```

---

## 3. Setting Up Parameter Pointers

```c
typedef struct {
    float *wte;       // [V, d]
    float *wpe;       // [T, d]
    float **ln1w, **ln1b;   // [L][d]
    float **qkvw, **qkvb;   // [L][3d, d] and [L][3d]
    float **projw, **projb; // [L][d, d] and [L][d]
    float **ln2w, **ln2b;   // [L][d]
    float **fc1w, **fc1b;   // [L][4d, d] and [L][4d]
    float **fc2w, **fc2b;   // [L][d, 4d] and [L][d]
    float *lnfw, *lnfb;     // [d]
    float *_mem;      // backing allocation
    GPT2Config cfg;
} GPT2Params;

void gpt2_setup_pointers(GPT2Params *p) {
    int L = p->cfg.n_layers, d = p->cfg.channels;
    int V = p->cfg.padded_vocab_size, T = p->cfg.max_seq_len;

    p->ln1w  = malloc(L * sizeof(float*));
    p->ln1b  = malloc(L * sizeof(float*));
    p->qkvw  = malloc(L * sizeof(float*));
    p->qkvb  = malloc(L * sizeof(float*));
    p->projw = malloc(L * sizeof(float*));
    p->projb = malloc(L * sizeof(float*));
    p->ln2w  = malloc(L * sizeof(float*));
    p->ln2b  = malloc(L * sizeof(float*));
    p->fc1w  = malloc(L * sizeof(float*));
    p->fc1b  = malloc(L * sizeof(float*));
    p->fc2w  = malloc(L * sizeof(float*));
    p->fc2b  = malloc(L * sizeof(float*));

    float *ptr = p->_mem;
    p->wte = ptr; ptr += (size_t)V * d;
    p->wpe = ptr; ptr += (size_t)T * d;
    for (int l = 0; l < L; l++) {
        p->ln1w[l]  = ptr; ptr += d;
        p->ln1b[l]  = ptr; ptr += d;
        p->qkvw[l]  = ptr; ptr += 3*d*d;
        p->qkvb[l]  = ptr; ptr += 3*d;
        p->projw[l] = ptr; ptr += d*d;
        p->projb[l] = ptr; ptr += d;
        p->ln2w[l]  = ptr; ptr += d;
        p->ln2b[l]  = ptr; ptr += d;
        p->fc1w[l]  = ptr; ptr += 4*d*d;
        p->fc1b[l]  = ptr; ptr += 4*d;
        p->fc2w[l]  = ptr; ptr += d*4*d;
        p->fc2b[l]  = ptr; ptr += d;
    }
    p->lnfw = ptr; ptr += d;
    p->lnfb = ptr;
}
```

---

## 4. Full Forward Pass

```c
// gpt2_forward_single: forward pass for a single sequence (N=1)
// Returns logits [T, V] for the last token
void gpt2_forward(
    GPT2Params  *p,
    const int   *tokens,  // [T]  token IDs
    float       *logits,  // [V]  output logits for last token
    int T) {

    int d = p->cfg.channels, V = p->cfg.vocab_size;
    int n_heads = p->cfg.n_heads, L = p->cfg.n_layers;
    int M = T;

    // 1. Embedding
    float *x = malloc(M * d * sizeof(float));
    gpt2_embed_forward(tokens, p->wte, p->wpe, x, 1, T, d);

    // 2. Transformer blocks
    float *x2 = malloc(M * d * sizeof(float));
    for (int l = 0; l < L; l++) {
        // Allocate block buffers (simplified — real impl reuses)
        BlockBuffers buf = {0};
        buf.ln1_out  = malloc(M * d * sizeof(float));
        buf.ln1_mean = malloc(M * sizeof(float));
        buf.ln1_rstd = malloc(M * sizeof(float));
        buf.attn_qkv = malloc(M * 3 * d * sizeof(float));
        buf.attn_w   = malloc((long)n_heads * T * T * sizeof(float));
        buf.head_out = malloc((long)n_heads * T * (d/n_heads) * sizeof(float));
        buf.attn_out = malloc(M * d * sizeof(float));
        buf.x1       = malloc(M * d * sizeof(float));
        buf.ln2_out  = malloc(M * d * sizeof(float));
        buf.ln2_mean = malloc(M * sizeof(float));
        buf.ln2_rstd = malloc(M * sizeof(float));
        buf.ffn_mid  = malloc(M * 4 * d * sizeof(float));
        buf.ffn_out  = malloc(M * d * sizeof(float));

        TransformerBlock blk = {
            .ln1_w = p->ln1w[l],  .ln1_b = p->ln1b[l],
            .qkv_w = p->qkvw[l],  .qkv_b = p->qkvb[l],
            .proj_w = p->projw[l], .proj_b = p->projb[l],
            .ln2_w = p->ln2w[l],  .ln2_b = p->ln2b[l],
            .fc1_w = p->fc1w[l],  .fc1_b = p->fc1b[l],
            .fc2_w = p->fc2w[l],  .fc2_b = p->fc2b[l],
            .d = d, .n_heads = n_heads
        };

        transformer_block_forward(&blk, &buf, x, x2, 1, T, d, n_heads, 0);

        // Swap
        float *tmp = x; x = x2; x2 = tmp;

        // Free block buffers...
        free(buf.ln1_out); /* ... free all ... */
    }
    free(x2);

    // 3. Final LayerNorm
    float *ln_out = malloc(M * d * sizeof(float));
    float *mean = malloc(M * sizeof(float)), *rstd = malloc(M * sizeof(float));
    layernorm_forward(x, p->lnfw, p->lnfb, ln_out, mean, rstd, M, d);
    free(x); free(mean); free(rstd);

    // 4. Unembed (last token only, weight-tied)
    unembed_forward(ln_out + (long)(T-1) * d, p->wte, logits, 1, d, V);
    free(ln_out);
}
```

---

## 5. Verification Against HuggingFace

```python
# Reference values from Python (run once, save to file)
from transformers import GPT2LMHeadModel
import torch

model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

tokens = torch.tensor([[15496, 11, 995, 0]])  # "Hello, world!"
with torch.no_grad():
    out = model(tokens)
logits = out.logits[0, -1, :5]  # last token, first 5 logits
print("HuggingFace logits (first 5):", logits.tolist())
# Expected: [-35.73, -34.90, -37.81, -38.72, -38.15]
```

```c
// Compare C implementation output
static void verify_gpt2(GPT2Params *p) {
    int tokens[] = {15496, 11, 995, 0};  // "Hello, world!"
    int T = 4;
    float *logits = malloc(p->cfg.vocab_size * sizeof(float));

    gpt2_forward(p, tokens, logits, T);

    printf("C logits (first 5):\n");
    for (int i = 0; i < 5; i++) printf("  [%d] = %.4f\n", i, logits[i]);
    // Target: [-35.73, -34.90, -37.81, -38.72, -38.15]
    // Acceptable: |diff| < 0.01 for FP32 precision

    // Find argmax (predicted next token)
    int pred = 0;
    for (int i = 1; i < p->cfg.vocab_size; i++)
        if (logits[i] > logits[pred]) pred = i;
    printf("Predicted next token: %d\n", pred);
    // For "Hello, world!" → token 50256 (<|endoftext|>) or common continuation

    free(logits);
}
```

---

## 6. Greedy Token Generation

```c
// Generate up to `max_new_tokens` tokens greedily
void gpt2_generate(
    GPT2Params *p,
    const int  *prompt,    // prompt token IDs
    int         prompt_len,
    int        *out,       // output token buffer (prompt + generated)
    int         max_new_tokens) {

    int T_max = p->cfg.max_seq_len;
    int V = p->cfg.vocab_size;
    int *tokens = malloc(T_max * sizeof(int));
    memcpy(tokens, prompt, prompt_len * sizeof(int));
    int T = prompt_len;

    float *logits = malloc(V * sizeof(float));

    printf("Prompt: ");
    for (int i = 0; i < prompt_len; i++) printf("%d ", tokens[i]);
    printf("\nGenerating...\n");

    for (int step = 0; step < max_new_tokens && T < T_max; step++) {
        gpt2_forward(p, tokens, logits, T);

        // Greedy: pick argmax
        int next = 0;
        for (int i = 1; i < V; i++)
            if (logits[i] > logits[next]) next = i;

        tokens[T++] = next;
        printf("token %d: %d\n", T-1, next);

        if (next == 50256) break;  // <|endoftext|>
    }

    memcpy(out, tokens, T * sizeof(int));
    free(tokens); free(logits);
}
```

---

## 7. Profiling the Forward Pass

```c
#include <time.h>

void profile_gpt2(GPT2Params *p, int T) {
    int tokens[T];
    for (int i = 0; i < T; i++) tokens[i] = i % 50256;
    float *logits = malloc(p->cfg.vocab_size * sizeof(float));

    // Warmup
    gpt2_forward(p, tokens, logits, T);

    // Timed run
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int N_RUNS = 5;
    for (int i = 0; i < N_RUNS; i++) gpt2_forward(p, tokens, logits, T);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double ms = ((t1.tv_sec - t0.tv_sec)*1000.0 + (t1.tv_nsec - t0.tv_nsec)/1e6) / N_RUNS;
    printf("GPT-2 forward (T=%d): %.1f ms/iter\n", T, ms);

    // FLOPs estimate
    long L = p->cfg.n_layers, d = p->cfg.channels;
    long flops = (long)L * (long)T * (6*d*d + 2LL*T*d + 8*d*d);
    printf("Estimated FLOPs: %.2f GFLOPs\n", flops / 1e9);
    printf("Effective throughput: %.1f GFLOP/s\n", flops / ms / 1e6);
    // Apple M2 single-thread: ~50-200 GFLOP/s (FP32 BLAS)

    free(logits);
}
```

---

## Key Takeaways

- GPT-2 weights load as a single binary blob: `fread` the entire param array, then set up pointer offsets for each weight tensor
- Full forward pass = embed → L × block → final LN → unembed (weight-tied)
- Verify against HuggingFace with `|diff| < 0.01` on logits — differences larger than that indicate a bug in layer order, weight transposition, or missing bias
- Greedy generation = repeated forward pass + argmax; efficient inference requires KV cache (Lesson 26)
- FFN matmuls dominate runtime — 12 layers × 2 matmuls × (768×3072) = the primary bottleneck

---

**Next**: [30. Llama Architecture](./30_Llama_Architecture.md) — Llama 2/3: RMSNorm, SwiGLU FFN, Grouped Query Attention (GQA), and RoPE — implement and verify the forward pass.
