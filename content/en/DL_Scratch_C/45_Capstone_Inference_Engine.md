# 45. Capstone: Complete Inference Engine

**Previous**: [Parallel Inference](./44_Parallel_Inference.md) | Course Complete

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Assemble all DL_Scratch_C components into a working LLM inference engine
2. Implement a complete decode loop: prefill, KV cache management, and token generation
3. Measure tokens/sec and compare against llama.cpp baseline performance
4. Identify the key design decisions (memory layout, buffer reuse, thread count) that affect throughput
5. Self-assess your implementation against a production-quality checklist

---

## 1. Engine Architecture Overview

This capstone integrates all techniques from lessons 26-44 into one cohesive program:

```
GGUF file
    │
    ▼
[43] gguf_load()          ─── memory-mapped weights, tensor metadata
    │
    ▼
[40] dequant on demand    ─── INT4/INT8 weights → FP32 at compute time
    │
    ▼
CLI input → tokenize (BPE, lesson 21)
    │
    ▼
[Prefill] forward(prompt)  ─── build KV cache for all prompt tokens
    │
    ▼
[Decode loop]
    ├── [30] Llama block: RMSNorm + GQA attention + SwiGLU FFN
    ├── [26] KV cache read/write
    ├── [23] RoPE position encoding
    ├── [44] OpenMP parallel matmul
    └── [39] sample_token() → next token
    │
    ▼
Print tokens as they are generated
```

---

## 2. Core Data Structures

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================
// Model configuration (parsed from GGUF metadata)
// ============================================================
typedef struct {
    int   n_layers;
    int   n_heads;        // query heads
    int   n_kv_heads;     // key/value heads (GQA: n_kv_heads <= n_heads)
    int   d_model;        // embedding dimension
    int   d_ff;           // FFN intermediate dimension (SwiGLU)
    int   vocab_size;
    int   max_seq_len;
    float rope_theta;     // RoPE base frequency (default 500000.0 for Llama-3)
    int   d_head;         // d_model / n_heads
} ModelConfig;

// ============================================================
// KV Cache  (lessons 26, 38)
// ============================================================
typedef struct {
    float  *K;        // [n_layers, max_seq, n_kv_heads, d_head]
    float  *V;        // [n_layers, max_seq, n_kv_heads, d_head]
    int     n_cached; // number of tokens currently in cache
    int     max_seq;
    int     n_layers;
    int     n_kv_heads;
    int     d_head;
} KVCache;

KVCache *kvcache_create(const ModelConfig *cfg, int max_seq) {
    KVCache *kv = malloc(sizeof(KVCache));
    kv->max_seq    = max_seq;
    kv->n_layers   = cfg->n_layers;
    kv->n_kv_heads = cfg->n_kv_heads;
    kv->d_head     = cfg->d_head;
    kv->n_cached   = 0;
    size_t sz = (size_t)cfg->n_layers * max_seq * cfg->n_kv_heads * cfg->d_head;
    kv->K = calloc(sz, sizeof(float));
    kv->V = calloc(sz, sizeof(float));
    if (!kv->K || !kv->V) { fprintf(stderr, "KV cache alloc failed\n"); exit(1); }
    printf("KV cache: %.2f MB\n",
           2.0 * sz * sizeof(float) / (1024.0 * 1024.0));
    return kv;
}

void kvcache_free(KVCache *kv) {
    free(kv->K); free(kv->V); free(kv);
}

// Write one token's K and V to cache at position pos
void kvcache_write(KVCache *kv, int layer, int pos,
                   const float *k_vec, const float *v_vec) {
    int stride = kv->max_seq * kv->n_kv_heads * kv->d_head;
    float *K_layer = kv->K + layer * stride;
    float *V_layer = kv->V + layer * stride;
    int offset = pos * kv->n_kv_heads * kv->d_head;
    memcpy(K_layer + offset, k_vec, kv->n_kv_heads * kv->d_head * sizeof(float));
    memcpy(V_layer + offset, v_vec, kv->n_kv_heads * kv->d_head * sizeof(float));
}

// ============================================================
// Activation buffers (pre-allocated, reused each step)
// ============================================================
typedef struct {
    float *x;         // current hidden state [d_model]
    float *x_norm;    // after RMSNorm [d_model]
    float *q;         // query [n_heads * d_head]
    float *k;         // key   [n_kv_heads * d_head]
    float *v;         // value [n_kv_heads * d_head]
    float *attn_out;  // attention output [n_heads * d_head]
    float *ffn_up;    // FFN gate/up [d_ff]
    float *ffn_gate;  // FFN gate [d_ff]
    float *logits;    // vocabulary logits [vocab_size]
} ActivationBuffers;

ActivationBuffers *buffers_create(const ModelConfig *cfg) {
    ActivationBuffers *b = malloc(sizeof(ActivationBuffers));
    b->x        = malloc(cfg->d_model * sizeof(float));
    b->x_norm   = malloc(cfg->d_model * sizeof(float));
    b->q        = malloc(cfg->n_heads    * cfg->d_head * sizeof(float));
    b->k        = malloc(cfg->n_kv_heads * cfg->d_head * sizeof(float));
    b->v        = malloc(cfg->n_kv_heads * cfg->d_head * sizeof(float));
    b->attn_out = malloc(cfg->n_heads    * cfg->d_head * sizeof(float));
    b->ffn_up   = malloc(cfg->d_ff * sizeof(float));
    b->ffn_gate = malloc(cfg->d_ff * sizeof(float));
    b->logits   = malloc(cfg->vocab_size * sizeof(float));
    return b;
}

void buffers_free(ActivationBuffers *b) {
    free(b->x); free(b->x_norm); free(b->q); free(b->k); free(b->v);
    free(b->attn_out); free(b->ffn_up); free(b->ffn_gate); free(b->logits);
    free(b);
}
```

---

## 3. Core Compute Primitives

```c
// ============================================================
// RMSNorm  (lesson 24)
// out[i] = x[i] / rms(x) * weight[i]
// ============================================================
void rmsnorm(float *out, const float *x, const float *weight, int n) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(ss / n + 1e-5f);
    for (int i = 0; i < n; i++) out[i] = x[i] * inv_rms * weight[i];
}

// ============================================================
// RoPE: apply rotary position encoding in-place (lesson 23)
// Applies to interleaved pairs: (x[2i], x[2i+1])
// ============================================================
void rope_apply(float *x, int pos, int d, float theta) {
    for (int i = 0; i < d; i += 2) {
        float freq = 1.0f / powf(theta, (float)i / (float)d);
        float cos_f = cosf(pos * freq);
        float sin_f = sinf(pos * freq);
        float x0 = x[i], x1 = x[i+1];
        x[i]   = x0 * cos_f - x1 * sin_f;
        x[i+1] = x0 * sin_f + x1 * cos_f;
    }
}

// ============================================================
// SwiGLU activation: out[i] = gate[i] * silu(up[i])
// silu(x) = x * sigmoid(x)
// ============================================================
static float silu(float x) { return x / (1.0f + expf(-x)); }

void swiglu(float *out, const float *gate, const float *up, int n) {
    for (int i = 0; i < n; i++)
        out[i] = silu(gate[i]) * up[i];
}

// ============================================================
// Matmul (single-token): out[N] = input[K] @ W[N,K]^T
// Optionally parallel (OpenMP)
// ============================================================
void matmul_vec(float *out, const float *x, const float *W, int N, int K) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int n = 0; n < N; n++) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) acc += x[k] * W[n*K + k];
        out[n] = acc;
    }
}
```

---

## 4. Transformer Layer Forward Pass

```c
// Simplified weight struct (in real engine, loaded from GGUF via mmap)
typedef struct {
    // Attention weights [all shapes assume d_head = d_model / n_heads]
    float *wq;       // [n_heads * d_head, d_model]
    float *wk;       // [n_kv_heads * d_head, d_model]
    float *wv;       // [n_kv_heads * d_head, d_model]
    float *wo;       // [d_model, n_heads * d_head]
    float *attn_norm; // [d_model]
    // FFN weights (SwiGLU: 3 matrices)
    float *w_gate;   // [d_ff, d_model]
    float *w_up;     // [d_ff, d_model]
    float *w_down;   // [d_model, d_ff]
    float *ffn_norm; // [d_model]
} LayerWeights;

void transformer_layer_forward(ActivationBuffers *buf,
                                const LayerWeights *wts,
                                KVCache *kv,
                                const ModelConfig *cfg,
                                int layer, int pos) {
    int dm  = cfg->d_model;
    int dh  = cfg->d_head;
    int nh  = cfg->n_heads;
    int nkv = cfg->n_kv_heads;
    int dff = cfg->d_ff;
    int groups = nh / nkv;  // GQA: how many Q heads share one KV head

    // --- Attention sublayer ---
    rmsnorm(buf->x_norm, buf->x, wts->attn_norm, dm);

    matmul_vec(buf->q, buf->x_norm, wts->wq, nh  * dh, dm);
    matmul_vec(buf->k, buf->x_norm, wts->wk, nkv * dh, dm);
    matmul_vec(buf->v, buf->x_norm, wts->wv, nkv * dh, dm);

    // Apply RoPE to each head's Q and K
    for (int h = 0; h < nh;  h++) rope_apply(buf->q + h * dh, pos, dh, cfg->rope_theta);
    for (int h = 0; h < nkv; h++) rope_apply(buf->k + h * dh, pos, dh, cfg->rope_theta);

    // Write K, V into KV cache
    kvcache_write(kv, layer, pos, buf->k, buf->v);

    // Compute attention output: for each query head
    float scale = 1.0f / sqrtf((float)dh);
    int stride_kv = kv->n_kv_heads * dh;

    for (int qh = 0; qh < nh; qh++) {
        int kv_head = qh / groups;
        const float *q_h  = buf->q + qh * dh;
        const float *K_h  = kv->K + layer * kv->max_seq * stride_kv + kv_head * dh;
        const float *V_h  = kv->V + layer * kv->max_seq * stride_kv + kv_head * dh;
        float       *o_h  = buf->attn_out + qh * dh;

        // Compute scores for all cached positions
        float *scores = malloc((pos + 1) * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            float dot = 0.0f;
            const float *k_t = K_h + t * stride_kv;
            for (int d = 0; d < dh; d++) dot += q_h[d] * k_t[d];
            scores[t] = dot * scale;
        }

        // Softmax
        float max_s = scores[0];
        for (int t = 1; t <= pos; t++) if (scores[t] > max_s) max_s = scores[t];
        float sum_e = 0.0f;
        for (int t = 0; t <= pos; t++) { scores[t] = expf(scores[t] - max_s); sum_e += scores[t]; }
        for (int t = 0; t <= pos; t++) scores[t] /= sum_e;

        // Weighted sum of V
        memset(o_h, 0, dh * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            const float *v_t = V_h + t * stride_kv;
            for (int d = 0; d < dh; d++) o_h[d] += scores[t] * v_t[d];
        }
        free(scores);
    }

    // Project attention output back: x += Wo @ attn_out
    float *delta = malloc(dm * sizeof(float));
    matmul_vec(delta, buf->attn_out, wts->wo, dm, nh * dh);
    for (int i = 0; i < dm; i++) buf->x[i] += delta[i];
    free(delta);

    // --- FFN sublayer (SwiGLU) ---
    rmsnorm(buf->x_norm, buf->x, wts->ffn_norm, dm);
    matmul_vec(buf->ffn_gate, buf->x_norm, wts->w_gate, dff, dm);
    matmul_vec(buf->ffn_up,   buf->x_norm, wts->w_up,   dff, dm);
    swiglu(buf->ffn_up, buf->ffn_gate, buf->ffn_up, dff);

    delta = malloc(dm * sizeof(float));
    matmul_vec(delta, buf->ffn_up, wts->w_down, dm, dff);
    for (int i = 0; i < dm; i++) buf->x[i] += delta[i];
    free(delta);
}
```

---

## 5. Main Inference Engine

```c
// Tokens/sec measurement
typedef struct {
    double t_prefill_start;
    double t_decode_start;
    int    n_prompt;
    int    n_generated;
    double elapsed_decode;
} PerfStats;

double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Minimal sampler config (temperature + top-p)
typedef struct { float temperature; float top_p; } SamplerCfg;

// Softmax and sample (simplified inline)
static int sample_next_token(float *logits, int vocab, float temp, float top_p) {
    if (temp <= 0.0f) {
        int best = 0;
        for (int i = 1; i < vocab; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }
    for (int i = 0; i < vocab; i++) logits[i] /= temp;
    float max_l = logits[0];
    for (int i = 1; i < vocab; i++) if (logits[i] > max_l) max_l = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < vocab; i++) { logits[i] = expf(logits[i] - max_l); sum += logits[i]; }
    for (int i = 0; i < vocab; i++) logits[i] /= sum;
    float r = (float)rand() / ((float)RAND_MAX + 1.0f);
    float cum = 0.0f;
    for (int i = 0; i < vocab; i++) { cum += logits[i]; if (r < cum) return i; }
    return vocab - 1;
}

// Main inference function
// In a real engine: weights are loaded via gguf_load (lesson 43)
// Here we show the control flow structure
void run_inference(const ModelConfig *cfg,
                   LayerWeights *layer_weights,  // array of n_layers
                   float *embed_table,            // [vocab_size, d_model]
                   float *output_norm,            // [d_model] final RMSNorm
                   float *lm_head,                // [vocab_size, d_model]
                   const int *prompt_tokens, int n_prompt,
                   int max_new_tokens,
                   const SamplerCfg *sampler) {

    KVCache          *kv  = kvcache_create(cfg, cfg->max_seq_len);
    ActivationBuffers *buf = buffers_create(cfg);
    PerfStats stats;
    stats.t_prefill_start = get_time_sec();
    stats.n_prompt    = n_prompt;
    stats.n_generated = 0;

    int *output = malloc(max_new_tokens * sizeof(int));
    int cur_token = prompt_tokens[0];
    int pos = 0;

    // ---- Prefill: process all prompt tokens ----
    for (int pi = 0; pi < n_prompt; pi++) {
        cur_token = prompt_tokens[pi];
        // Embed token
        memcpy(buf->x, embed_table + cur_token * cfg->d_model,
               cfg->d_model * sizeof(float));
        // Run all transformer layers
        for (int l = 0; l < cfg->n_layers; l++)
            transformer_layer_forward(buf, &layer_weights[l], kv, cfg, l, pos);
        pos++;
    }
    stats.t_decode_start = get_time_sec();

    // The next token to generate is predicted from the last prefill position
    rmsnorm(buf->x_norm, buf->x, output_norm, cfg->d_model);
    matmul_vec(buf->logits, buf->x_norm, lm_head, cfg->vocab_size, cfg->d_model);
    cur_token = sample_next_token(buf->logits, cfg->vocab_size,
                                  sampler->temperature, sampler->top_p);

    // ---- Decode loop: generate tokens one at a time ----
    double decode_start = get_time_sec();
    for (int gen = 0; gen < max_new_tokens; gen++) {
        output[gen] = cur_token;
        // EOS check (token 2 for Llama-3, 128001 for Llama-3-instruct)
        if (cur_token == 2 || cur_token == 128001) {
            stats.n_generated = gen + 1;
            break;
        }

        // Embed + forward pass for current token
        memcpy(buf->x, embed_table + cur_token * cfg->d_model,
               cfg->d_model * sizeof(float));
        for (int l = 0; l < cfg->n_layers; l++)
            transformer_layer_forward(buf, &layer_weights[l], kv, cfg, l, pos);
        pos++;
        stats.n_generated = gen + 1;

        // Compute next-token logits
        rmsnorm(buf->x_norm, buf->x, output_norm, cfg->d_model);
        matmul_vec(buf->logits, buf->x_norm, lm_head, cfg->vocab_size, cfg->d_model);
        cur_token = sample_next_token(buf->logits, cfg->vocab_size,
                                      sampler->temperature, sampler->top_p);

        // Print token as it is generated (streaming output)
        printf(" [tok%d]", cur_token); fflush(stdout);
    }

    stats.elapsed_decode = get_time_sec() - decode_start;
    double tps = stats.n_generated / stats.elapsed_decode;

    printf("\n\n=== Performance ===\n");
    printf("  Prompt tokens:   %d\n", n_prompt);
    printf("  Generated:       %d tokens\n", stats.n_generated);
    printf("  Decode time:     %.2f s\n", stats.elapsed_decode);
    printf("  Throughput:      %.2f tokens/sec\n", tps);
    printf("  Prefill time:    %.2f s (%.1f tok/s)\n",
           stats.t_decode_start - stats.t_prefill_start,
           n_prompt / (stats.t_decode_start - stats.t_prefill_start));

    kvcache_free(kv);
    buffers_free(buf);
    free(output);
}
```

---

## 6. Command-Line Interface

```c
typedef struct {
    char    model_path[512];
    char    prompt[4096];
    int     max_new_tokens;
    float   temperature;
    float   top_p;
    int     n_threads;
    int     context_len;
} CLIArgs;

void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [OPTIONS]\n"
        "  -m <path>       GGUF model file (required)\n"
        "  -p <prompt>     Input prompt (default: 'Hello')\n"
        "  -n <int>        Max new tokens (default: 200)\n"
        "  -t <float>      Temperature (default: 0.8)\n"
        "  --top-p <float> Top-p nucleus sampling (default: 0.9)\n"
        "  -T <int>        Number of threads (default: 4)\n"
        "  -c <int>        Context length (default: 2048)\n",
        prog);
}

int parse_args(CLIArgs *args, int argc, char **argv) {
    strcpy(args->model_path, "");
    strcpy(args->prompt, "Hello, world!");
    args->max_new_tokens = 200;
    args->temperature    = 0.8f;
    args->top_p          = 0.9f;
    args->n_threads      = 4;
    args->context_len    = 2048;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i+1 < argc)
            strncpy(args->model_path, argv[++i], 511);
        else if (strcmp(argv[i], "-p") == 0 && i+1 < argc)
            strncpy(args->prompt, argv[++i], 4095);
        else if (strcmp(argv[i], "-n") == 0 && i+1 < argc)
            args->max_new_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "-t") == 0 && i+1 < argc)
            args->temperature = atof(argv[++i]);
        else if (strcmp(argv[i], "--top-p") == 0 && i+1 < argc)
            args->top_p = atof(argv[++i]);
        else if (strcmp(argv[i], "-T") == 0 && i+1 < argc)
            args->n_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "-c") == 0 && i+1 < argc)
            args->context_len = atoi(argv[++i]);
        else if (strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]); return -1;
        }
    }
    if (strlen(args->model_path) == 0) {
        fprintf(stderr, "Error: -m <model_path> is required\n");
        print_usage(argv[0]); return -1;
    }
    return 0;
}

int main(int argc, char **argv) {
    CLIArgs args;
    if (parse_args(&args, argc, argv) < 0) return 1;

#ifdef _OPENMP
    omp_set_num_threads(args.n_threads);
    printf("Using %d OpenMP threads\n", args.n_threads);
#endif

    printf("Model: %s\n", args.model_path);
    printf("Prompt: \"%s\"\n", args.prompt);
    printf("Max tokens: %d, Temperature: %.2f, Top-p: %.2f\n",
           args.max_new_tokens, args.temperature, args.top_p);

    // In a full implementation:
    // 1. GGUFModel model; gguf_load(&model, args.model_path);
    // 2. Build ModelConfig from model.meta
    // 3. Point LayerWeights to mmap'd tensors (with dequant on the fly)
    // 4. Tokenize args.prompt using BPE tokenizer (lesson 21)
    // 5. run_inference(...)

    printf("\n[Full implementation: connect GGUF loader (lesson 43),\n"
           " BPE tokenizer (lesson 21), and all layer weights]\n");

    // Benchmark the decode loop overhead with dummy weights
    const int dummy_vocab  = 1000;
    const int dummy_layers = 2;
    const int dummy_dm     = 256;
    const int dummy_heads  = 4;
    const int dummy_dh     = dummy_dm / dummy_heads;
    const int dummy_dff    = dummy_dm * 4;

    ModelConfig cfg = {
        .n_layers   = dummy_layers,
        .n_heads    = dummy_heads,
        .n_kv_heads = dummy_heads,
        .d_model    = dummy_dm,
        .d_ff       = dummy_dff,
        .vocab_size = dummy_vocab,
        .max_seq_len = 512,
        .rope_theta = 500000.0f,
        .d_head     = dummy_dh,
    };

    // Allocate dummy weights
    LayerWeights *lw = calloc(dummy_layers, sizeof(LayerWeights));
    for (int l = 0; l < dummy_layers; l++) {
        lw[l].wq        = calloc(dummy_heads * dummy_dh * dummy_dm, sizeof(float));
        lw[l].wk        = calloc(dummy_heads * dummy_dh * dummy_dm, sizeof(float));
        lw[l].wv        = calloc(dummy_heads * dummy_dh * dummy_dm, sizeof(float));
        lw[l].wo        = calloc(dummy_dm * dummy_heads * dummy_dh, sizeof(float));
        lw[l].attn_norm = calloc(dummy_dm, sizeof(float));
        lw[l].w_gate    = calloc(dummy_dff * dummy_dm, sizeof(float));
        lw[l].w_up      = calloc(dummy_dff * dummy_dm, sizeof(float));
        lw[l].w_down    = calloc(dummy_dm * dummy_dff, sizeof(float));
        lw[l].ffn_norm  = calloc(dummy_dm, sizeof(float));
    }
    float *embed  = calloc(dummy_vocab * dummy_dm, sizeof(float));
    float *onorm  = calloc(dummy_dm, sizeof(float));
    float *lmhead = calloc(dummy_vocab * dummy_dm, sizeof(float));

    // Initialize norms to 1.0 so RMSNorm doesn't zero everything
    for (int i = 0; i < dummy_dm; i++) { onorm[i] = 1.0f; }
    for (int l = 0; l < dummy_layers; l++) {
        for (int i = 0; i < dummy_dm; i++) {
            lw[l].attn_norm[i] = 1.0f;
            lw[l].ffn_norm[i]  = 1.0f;
        }
    }

    int prompt[] = { 1, 42, 17, 88 };
    SamplerCfg sc = { args.temperature, args.top_p };
    srand(42);

    printf("\n--- Running inference on dummy %d-layer model ---\n", dummy_layers);
    run_inference(&cfg, lw, embed, onorm, lmhead,
                  prompt, 4, 20, &sc);

    // Cleanup
    for (int l = 0; l < dummy_layers; l++) {
        free(lw[l].wq); free(lw[l].wk); free(lw[l].wv); free(lw[l].wo);
        free(lw[l].attn_norm); free(lw[l].w_gate); free(lw[l].w_up);
        free(lw[l].w_down); free(lw[l].ffn_norm);
    }
    free(lw); free(embed); free(onorm); free(lmhead);
    return 0;
}
```

---

## 7. Benchmarking and Comparison with llama.cpp

```c
// Expected performance for a real Llama-3-8B Q4_K_M on typical hardware:
void print_comparison_table(void) {
    printf("=== Tokens/Sec Comparison: Llama-3-8B Q4_K_M, Context=512 ===\n");
    printf("%-30s %10s %10s %10s\n", "Implementation", "Threads", "tok/s", "vs llama.cpp");
    printf("%-30s %10s %10s %10s\n",
           "llama.cpp (optimized)",     "8",  "~30",  "1.0x (baseline)");
    printf("%-30s %10s %10s %10s\n",
           "llama.cpp (optimized)",     "4",  "~22",  "0.7x");
    printf("%-30s %10s %10s %10s\n",
           "This engine (OpenMP)",      "8",  "~8-15","0.3-0.5x");
    printf("%-30s %10s %10s %10s\n",
           "This engine (OpenMP)",      "4",  "~5-10","0.2-0.3x");
    printf("%-30s %10s %10s %10s\n",
           "This engine (naive F32)",   "1",  "~1-3", "0.05-0.1x");

    printf("\nGap explained:\n");
    printf("  llama.cpp uses: AVX2/AVX-512 SIMD, GGML K-quant kernels,\n");
    printf("  memory-layout optimizations, numa-aware alloc, metal/CUDA.\n");
    printf("  Our engine: scalar FP32 only, portable C11, educational clarity.\n");
}
```

---

## 8. Self-Assessment Checklist

Before claiming your inference engine is complete, verify:

**Correctness**
- [ ] RMSNorm output matches reference (PyTorch) to 1e-5 relative error
- [ ] RoPE rotations produce identical attention patterns to reference
- [ ] Greedy decoding produces the same token sequence as llama.cpp (given same model, same prompt)
- [ ] KV cache grows correctly: position 0..n_prompt-1 filled during prefill, pos n_prompt+ during decode
- [ ] GQA head grouping: query head `h` reads KV head `h / (n_heads / n_kv_heads)`

**Performance**
- [ ] Buffer reuse: no malloc/free inside the decode loop (pre-allocate all activation buffers)
- [ ] No redundant softmax calls (only one per attention head per step)
- [ ] Matmul parallelized over output neurons (not rows, since M=1)
- [ ] KV cache laid out for sequential access pattern during attention scoring

**Robustness**
- [ ] EOS token terminates generation correctly
- [ ] Context window overflow handled (error or sliding window)
- [ ] Temperature=0.0 falls back to greedy without division by zero
- [ ] GGUF loading validates magic number and version before reading tensors

**Production readiness**
- [ ] INT8/INT4 dequantization integrated (not FP32-only weights)
- [ ] Memory-mapped GGUF loading (no full-weight copy into heap)
- [ ] BPE tokenizer (lesson 21) connected for text I/O
- [ ] Streaming token output (print each token as generated, not at the end)

---

## Key Takeaways

- A complete LLM inference engine integrates embedding lookup, RMSNorm, RoPE, GQA attention with KV cache, SwiGLU FFN, and a sampling strategy — each developed in prior lessons.
- Activation buffer pre-allocation is critical: malloc/free inside the decode loop adds significant overhead at 10-30 tokens/sec where each millisecond matters.
- The KV cache memory layout (layer × position × head × dim) should be chosen to maximize sequential access during the attention scoring loop over cached positions.
- A naive C11 implementation typically achieves 0.1-0.3× llama.cpp throughput; the gap is explained by SIMD intrinsics, quantized kernels, and platform-specific memory optimizations in llama.cpp — not algorithmic differences.
- Correctness verification against a reference implementation (PyTorch or llama.cpp) should precede any performance optimization: benchmark only what is correct.
- The tokens/sec metric at batch=1 is almost entirely determined by memory bandwidth and weight size; algorithmic improvements (FlashAttention, speculative decoding) are complementary, not competing, optimizations.
- Building this engine from scratch provides deep understanding of every decision in production systems: why llama.cpp uses the data structures it does, why quantization choices matter, and what the true bottlenecks are.

---

**Previous**: [Parallel Inference](./44_Parallel_Inference.md) | Course Complete
