# 22. Embedding Table

**Previous**: [Tokenization and BPE](./21_Tokenization_BPE.md) | **Next**: [Positional Encodings](./23_Positional_Encodings.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement a token embedding table as a lookup operation
2. Implement the embedding backward pass (scatter-add gradient)
3. Explain weight tying between the input embedding and the output projection
4. Load GPT-2 binary weights from disk in the correct layout
5. Verify your embedding output matches HuggingFace GPT-2 on a sample sequence

---

## 1. The Embedding Table

An embedding table converts discrete token IDs into dense vectors:

```
Table:  [V, d_model]    (V = vocab size, d_model = embedding dimension)
Input:  token_id ∈ [0, V)
Output: table[token_id]   (a d_model-dimensional vector)

For GPT-2:
  V = 50,257  tokens
  d_model = 768  (GPT-2 small), 1024 (medium), 1600 (large), 1280 (XL)

Total parameters: 50,257 × 768 = 38.6M  (for GPT-2 small)
```

### Forward Pass

```c
// embedding_forward: lookup token embeddings
// tokens:  [N, T]  int32 token IDs
// table:   [V, d_model] float32
// output:  [N, T, d_model] float32
void embedding_forward(
    const int   *tokens,   // [N*T] token IDs
    const float *table,    // [V, d_model]
    float       *output,   // [N*T, d_model]
    int N_T, int d_model) {   // N_T = N * T (batch × seq length)

    for (int i = 0; i < N_T; i++) {
        int id = tokens[i];
        memcpy(output + (long)i * d_model,
               table  + (long)id * d_model,
               d_model * sizeof(float));
    }
}
```

### Backward Pass

The embedding backward is a scatter-add: sum gradient contributions for each token ID:

```c
// embedding_backward: compute gradient for the embedding table
// dtable[id] += sum of doutput rows where tokens[i] == id
void embedding_backward(
    const int   *tokens,   // [N*T]
    const float *doutput,  // [N*T, d_model] gradient from above
    float       *dtable,   // [V, d_model] — zero-initialized, accumulated
    int N_T, int d_model) {

    for (int i = 0; i < N_T; i++) {
        int id = tokens[i];
        float       *dst = dtable  + (long)id * d_model;
        const float *src = doutput + (long)i  * d_model;
        for (int j = 0; j < d_model; j++)
            dst[j] += src[j];
    }
}
```

---

## 2. Weight Tying

GPT-2 and most LLMs share the embedding table weights with the output projection:

```
Input path:   token_id → embedding_table[id] → d_model vector
Output path:  d_model vector → matmul(embedding_table^T) → [V] logits

Same matrix used twice — "weight tying":
  Embedding:  E [V, d_model]  (forward: lookup row id)
  Unembedding: E^T [d_model, V]  (forward: matmul)

Benefits:
  - Reduces parameters: saves 50,257 × 768 ≈ 38.6M params
  - Forces input/output representations to be consistent
  - Empirically improves perplexity
```

```c
// unembed_forward: project hidden state to vocabulary logits
// input:   [N*T, d_model]
// table:   [V, d_model]  (SAME as embedding table — weight tying!)
// output:  [N*T, V]
void unembed_forward(
    const float *input,    // [M, d_model]
    const float *table,    // [V, d_model]
    float       *logits,   // [M, V]
    int M, int d_model, int V) {

    // logits = input × table^T
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans, CblasTrans,
                M, V, d_model,
                1.0f, input,  d_model,
                       table, d_model,
                0.0f, logits, V);
}
```

---

## 3. GPT-2 Weight File Format

HuggingFace provides GPT-2 weights as a single `.bin` file (saved via `model.state_dict()`). The llm.c project serializes it as raw float32 arrays with a header:

```c
// GPT-2 weight file layout (llm.c format):
// Header: [magic:int32=20240326, version:int32, config:7×int32]
// Config: [max_seq_len, vocab_size, padded_vocab_size, n_layers, n_heads, n_kv_heads, channels]
// Weights (in order):
//   wte:  [vocab_size, channels]       token embedding
//   wpe:  [max_seq_len, channels]      position embedding
//   For each layer:
//     ln1w [channels], ln1b [channels]
//     qkvw [3*channels, channels], qkvb [3*channels]
//     projw [channels, channels], projb [channels]
//     ln2w [channels], ln2b [channels]
//     fcw  [4*channels, channels], fcb  [4*channels]
//     projw2 [channels, 4*channels], projb2 [channels]
//   lnfw [channels], lnfb [channels]   final LayerNorm

#define GPT2_MAGIC 20240326

typedef struct {
    int max_seq_len;
    int vocab_size;
    int padded_vocab_size;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int channels;
} GPT2Config;

typedef struct {
    GPT2Config config;
    float *wte;    // [vocab_size, channels]
    float *wpe;    // [max_seq_len, channels]
    float **ln1w, **ln1b;    // [n_layers][channels]
    float **qkvw, **qkvb;    // [n_layers][3*channels, channels]
    float **projw, **projb;  // [n_layers][channels, channels]
    float **ln2w, **ln2b;    // [n_layers][channels]
    float **fcw, **fcb;      // [n_layers][4*channels, channels]
    float **projw2, **projb2;// [n_layers][channels, 4*channels]
    float *lnfw, *lnfb;      // [channels]
    float *mem;    // single allocation backing all arrays
    size_t mem_size;
} GPT2Weights;

// Load GPT-2 weights from llm.c format binary file
GPT2Weights *gpt2_load_weights(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }

    // Read header
    int header[256] = {0};
    fread(header, sizeof(int), 256, f);
    if (header[0] != GPT2_MAGIC) {
        fprintf(stderr, "Bad magic number in weight file\n");
        fclose(f); return NULL;
    }

    GPT2Weights *wt = calloc(1, sizeof(GPT2Weights));
    wt->config.max_seq_len       = header[2];
    wt->config.vocab_size        = header[3];
    wt->config.padded_vocab_size = header[4];
    wt->config.n_layers          = header[5];
    wt->config.n_heads           = header[6];
    wt->config.n_kv_heads        = header[7];
    wt->config.channels          = header[8];

    GPT2Config *c = &wt->config;
    int C = c->channels, L = c->n_layers, V = c->padded_vocab_size;
    int T = c->max_seq_len;

    // Calculate total parameter count
    size_t n_params = (size_t)V * C           // wte
                    + (size_t)T * C            // wpe
                    + L * (2*C + 3*C*C + C*C + 2*C + C*4*C + C*4*C)  // layers
                    + 2 * C;                   // final LN
    wt->mem_size = n_params * sizeof(float);
    wt->mem = malloc(wt->mem_size);
    fread(wt->mem, sizeof(float), n_params, f);
    fclose(f);

    // Set up pointers into wt->mem
    float *ptr = wt->mem;
    wt->wte = ptr; ptr += (size_t)V * C;
    wt->wpe = ptr; ptr += (size_t)T * C;

    wt->ln1w = malloc(L * sizeof(float*));
    wt->ln1b = malloc(L * sizeof(float*));
    // ... (assign per-layer pointers similarly)

    return wt;
}
```

---

## 4. Verification Against HuggingFace

```c
static void test_embedding(void) {
    // Load GPT-2 small weights
    GPT2Weights *wt = gpt2_load_weights("gpt2_124M.bin");
    int C = wt->config.channels;  // 768

    // Test sequence: "Hello, world!" → tokens [15496, 11, 995, 0]
    int tokens[] = {15496, 11, 995, 0};
    int T = 4;

    float *emb_out = malloc(T * C * sizeof(float));
    embedding_forward(tokens, wt->wte, emb_out, T, C);

    // Token 15496 embedding — first 5 values should match HuggingFace:
    // Expected (from Python): [-0.0381, -0.0016,  0.0437, -0.0090,  0.0171, ...]
    printf("Token 15496 embedding (first 5 values):\n");
    for (int i = 0; i < 5; i++)
        printf("  [%d] = %.4f\n", i, emb_out[i]);

    free(emb_out);
    // gpt2_free_weights(wt);
}
```

Verification script in Python:

```python
# Verify C embedding output against HuggingFace
from transformers import GPT2Model
import torch

model = GPT2Model.from_pretrained('gpt2')
emb = model.wte.weight.data

tokens = [15496, 11, 995, 0]
for t in tokens:
    print(f"Token {t}: {emb[t, :5].tolist()}")
```

---

## 5. Embedding Initialization

For training from scratch (not fine-tuning GPT-2 weights):

```c
// Initialize embedding table with small random values
void embedding_init(float *table, int V, int d_model) {
    // Normal(0, 0.02) — GPT-2 initialization
    float std = 0.02f;
    for (int i = 0; i < V * d_model; i++)
        table[i] = randn() * std;
}
```

---

## Key Takeaways

- **Embedding forward**: simple row lookup — `output[i] = table[token_id[i]]`
- **Embedding backward**: scatter-add — `dtable[token_id[i]] += doutput[i]`; multiple sequences sharing the same token accumulate gradients
- **Weight tying**: input embedding and output projection use the same matrix `E` — saves 38.6M params in GPT-2 small and improves perplexity
- GPT-2 weights are stored as raw float32 arrays in a binary file with a header — load with a single `fread` call
- Always verify your embedding output against a known reference (HuggingFace) before implementing deeper layers

---

**Next**: [23. Positional Encodings](./23_Positional_Encodings.md) — Sinusoidal, learned, and RoPE positional encodings; implementing RoPE using real arithmetic for complex exponentials.
