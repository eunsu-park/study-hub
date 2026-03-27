# Block 8 — Inference & Optimization (L39–L45)

Prerequisites: L39 (sampling strategies), L40 (INT8 quantization), L41 (Flash Attention / online softmax), L42 (GGUF format), L43 (OpenMP parallelism), L44 (speculative decoding), L45 (deployment).

Compile: `gcc -std=c11 -Wall -Wextra -O2 -o ex ex.c -lm`
For OpenMP (Ex 8.5): `gcc -std=c11 -Wall -Wextra -O2 -fopenmp -o ex ex.c -lm`

---

## Exercise 8.1 — Top-p (Nucleus) Sampling

**Difficulty**: ★★

### Problem

Implement `top_p_sample(const float *logits, int V, float p, unsigned int *rng_state)` that:

1. Computes softmax probabilities from logits.
2. Sorts tokens by probability descending.
3. Finds the smallest set of top tokens whose cumulative probability ≥ p (the "nucleus").
4. Renormalizes the nucleus probabilities.
5. Samples from the nucleus using inverse CDF sampling.
6. Returns the sampled token index.

Then verify that no token outside the nucleus is ever sampled over 10,000 trials.

### Starter Code

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>

/* Simple LCG RNG (returns float in [0,1)) */
float lcg_rand(unsigned int *state) {
    *state = *state * 1664525u + 1013904223u;
    return (float)(*state >> 1) / (float)(1u << 31);
}

/* Comparison function for qsort (descending by prob) */
typedef struct { float prob; int idx; } TokenProb;
int cmp_desc(const void *a, const void *b) {
    float da = ((TokenProb*)a)->prob;
    float db = ((TokenProb*)b)->prob;
    return (da < db) - (da > db);
}

int top_p_sample(const float *logits, int V, float p, unsigned int *rng_state) {
    /* Step 1: softmax */
    float probs[V];
    float max_l = -FLT_MAX;
    for (int i=0;i<V;i++) if (logits[i]>max_l) max_l=logits[i];
    float s=0;
    for (int i=0;i<V;i++){probs[i]=expf(logits[i]-max_l); s+=probs[i];}
    for (int i=0;i<V;i++) probs[i]/=s;

    /* Step 2: sort descending */
    TokenProb sorted[V];
    for (int i=0;i<V;i++){sorted[i].prob=probs[i]; sorted[i].idx=i;}
    qsort(sorted, V, sizeof(TokenProb), cmp_desc);

    /* Step 3: find nucleus cutoff */
    float cumsum = 0.0f;
    int nucleus_size = 0;
    /* TODO: increment nucleus_size while cumsum < p */

    /* Step 4: renormalize nucleus */
    float nucleus_sum = 0.0f;
    for (int i=0;i<nucleus_size;i++) nucleus_sum += sorted[i].prob;
    /* TODO: divide each sorted[i].prob by nucleus_sum */

    /* Step 5: sample via inverse CDF */
    float r = lcg_rand(rng_state);
    float cdf = 0.0f;
    for (int i=0;i<nucleus_size;i++){
        cdf += sorted[i].prob;
        if (r <= cdf) return sorted[i].idx;
    }
    return sorted[nucleus_size-1].idx;  /* fallback */
}

int main(void) {
    int V = 10;
    /* Logits: token 3 has highest probability, token 9 lowest */
    float logits[10] = {0.5f, 1.0f, 0.2f, 3.0f, -1.0f, 0.8f, 0.1f, -0.5f, 0.3f, -2.0f};
    float p = 0.9f;

    /* Compute nucleus manually */
    float probs[10];
    float max_l=-FLT_MAX, s=0;
    for(int i=0;i<V;i++) if(logits[i]>max_l) max_l=logits[i];
    for(int i=0;i<V;i++){probs[i]=expf(logits[i]-max_l);s+=probs[i];}
    for(int i=0;i<V;i++) probs[i]/=s;

    printf("Softmax probs: ");
    for(int i=0;i<V;i++) printf("%.3f ",probs[i]);
    printf("\n");

    /* Sample 10000 times and record which tokens appear */
    unsigned int rng = 12345;
    int counts[10] = {0};
    for (int t=0;t<10000;t++) counts[top_p_sample(logits, V, p, &rng)]++;

    printf("Token counts (10000 samples, p=%.1f):\n", p);
    for(int i=0;i<V;i++) printf("  token %d: %d\n", i, counts[i]);

    /* Find nucleus: sort and accumulate */
    /* Tokens with very low prob (far outside nucleus) should have count=0 */
    printf("\nVerification: tokens outside nucleus must have count=0\n");
    /* Expected: tokens 4,7,9 (low probs) should rarely/never be sampled with p=0.9 */
    return 0;
}
```

### Test Cases

For `p=0.9` and the given logits:
- Token 3 (logit=3.0) should have the highest count.
- Tokens with very low probability (logits << 0) should have count=0 over 10,000 trials.
- The nucleus size (number of tokens sampled) should be < V.

For `p=1.0`: all tokens can be sampled (full softmax).
For `p=0.0` or `p=very_small`: only the top-1 token is sampled (greedy).

### Hints

1. Build the nucleus greedily: add tokens in descending probability order until cumulative probability ≥ p.
2. Include the token that pushes the cumulative sum over `p` (that's why it's `>=` not `>`).
3. Inverse CDF sampling: draw uniform `r`, walk the sorted list adding probabilities until the running sum exceeds `r`.
4. Renormalization ensures the nucleus sums to 1 before sampling.

### Solution Approach

Top-p sampling restricts the model to a "nucleus" of high-probability tokens, preventing the model from generating rare tokens even when the temperature is high. The nucleus size is adaptive — for peaked distributions (confident model) it is small; for flat distributions (uncertain model) it is large. This is why top-p is preferred over top-k in most modern LLM deployments.

---

## Exercise 8.2 — Absmax INT8 Quantization

**Difficulty**: ★★

### Problem

Implement absmax quantization of a float32 tensor to INT8:

```
scale = max(|x_i|) / 127
x_quant[i] = round(x_i / scale)       (clamped to [-127, 127])
x_dequant[i] = x_quant[i] * scale     (reconstruction)
```

Verify: `max(|x - x_dequant|) < scale` (the quantization error is bounded by one step).

### Starter Code

```c
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <float.h>

void absmax_quantize(const float *x, int n,
                     int8_t *q, float *scale_out) {
    /* Step 1: find max absolute value */
    float max_abs = 0.0f;
    /* TODO */

    float scale = max_abs / 127.0f;
    *scale_out = scale;

    /* Step 2: quantize each element */
    for (int i = 0; i < n; i++) {
        float qf = x[i] / scale;
        /* TODO: round and clamp to [-127, 127], store in q[i] */
    }
}

void absmax_dequantize(const int8_t *q, int n, float scale, float *out) {
    /* TODO: out[i] = q[i] * scale */
}

int main(void) {
    int n = 8;
    float x[8] = {0.1f, -0.5f, 1.2f, -3.0f, 2.7f, 0.0f, -1.1f, 0.8f};

    int8_t q[8];
    float scale;
    absmax_quantize(x, n, q, &scale);

    printf("Scale: %.6f\n", scale);
    printf("Quantized: ");
    for (int i=0;i<n;i++) printf("%4d ", (int)q[i]);
    printf("\n");

    float x_rec[8];
    absmax_dequantize(q, n, scale, x_rec);

    printf("Original:     ");
    for (int i=0;i<n;i++) printf("%6.3f ", x[i]);
    printf("\nReconstructed:");
    for (int i=0;i<n;i++) printf("%6.3f ", x_rec[i]);
    printf("\nError:        ");

    float max_err = 0.0f;
    for (int i=0;i<n;i++){
        float e = fabsf(x[i] - x_rec[i]);
        printf("%6.4f ", e);
        if (e > max_err) max_err = e;
    }
    printf("\n");
    printf("Max error: %.6f\n", max_err);
    printf("Upper bound (scale=%.6f): %s\n", scale,
           max_err < scale ? "PASS" : "FAIL");

    /* Verify the bound: max|error| < scale */
    printf("Bound check max_err < scale: %s\n",
           max_err <= scale ? "PASS" : "FAIL");
    return 0;
}
```

### Test Cases

For `x = [0.1, -0.5, 1.2, -3.0, 2.7, 0.0, -1.1, 0.8]`:
- `max_abs = 3.0`, `scale = 3.0/127 ≈ 0.02362`
- `q[3] = round(-3.0 / 0.02362) = -127` (clamped)
- `q[4] = round(2.7 / 0.02362) = round(114.3) = 114`
- `max_error < scale` must hold for all elements.

Edge cases:
- All-zeros tensor: handle `max_abs=0` (return scale=1, q=all-zeros).
- Single-element tensor: must still satisfy the bound.

### Hints

1. Use `roundf()` from `<math.h>` for rounding.
2. Clamping: `if (qf > 127) qf = 127; if (qf < -127) qf = -127;` before casting to `int8_t`.
3. The maximum error is one quantization step `scale` — this is the rounding error.
4. Avoid dividing by zero when `max_abs=0`.

### Solution Approach

Absmax is the simplest symmetric quantization scheme. It maps the maximum absolute value to 127, so the entire range fits in INT8. The error bound follows from the fact that rounding error is at most 0.5 quantization steps, and a step is `scale`. This scheme is used in llama.cpp Q8_0 quantization. Block quantization (per-32 or per-64 elements) uses separate scales per block, reducing the quantization range and thus the error.

---

## Exercise 8.3 — Online (Single-Pass) Softmax

**Difficulty**: ★★★

### Problem

Flash Attention uses an **online softmax** algorithm that computes the softmax in a single pass through the input, without pre-scanning for the maximum. Implement it and verify it matches the standard two-pass result.

Online algorithm (one element at a time):
```
Initialize: m = -inf, d = 0, out = zeros

For each i in 0..n-1:
    m_new = max(m, x[i])
    d_new = d * exp(m - m_new) + exp(x[i] - m_new)
    out    = out * exp(m - m_new)    (rescale previous sum)
    out   += exp(x[i] - m_new)      (NOT output but partial; final out = ... / d_new)
    m      = m_new
    d      = d_new

Final: out[i] = exp(x[i] - m) / d   (recompute, or maintain a running output vector)
```

For vectors (not scalars), the running output `out` stores the numerators, and division by `d` at the end gives the softmax.

### Starter Code

```c
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <string.h>

/* Standard two-pass softmax */
void softmax_twopass(const float *x, float *out, int n) {
    float m = -FLT_MAX;
    for (int i=0;i<n;i++) if(x[i]>m) m=x[i];
    float s=0;
    for (int i=0;i<n;i++){out[i]=expf(x[i]-m);s+=out[i];}
    for (int i=0;i<n;i++) out[i]/=s;
}

/* Online single-pass softmax */
void softmax_online(const float *x, float *out, int n) {
    float m = -FLT_MAX;  /* running max */
    float d = 0.0f;      /* running denominator */

    /* First pass: compute m (max) and d (partition function) online */
    for (int i = 0; i < n; i++) {
        float m_new = fmaxf(m, x[i]);
        /* TODO: update d with rescaling: d = d * exp(m - m_new) + exp(x[i] - m_new) */
        m = m_new;
    }

    /* Second pass: compute output using final m and d */
    /* TODO: out[i] = exp(x[i] - m) / d */
    /* Note: this is still two passes over x, but m and d were computed in ONE pass.
       True Flash Attention fuses the two passes using tiles (discussed in L41). */
}

int main(void) {
    int n = 8;
    float x[8] = {1.0f, 3.0f, -1.0f, 2.0f, 0.5f, -2.0f, 4.0f, 1.5f};

    float out_tp[8], out_ol[8];
    softmax_twopass(x, out_tp, n);
    softmax_online(x, out_ol, n);

    float max_diff = 0.0f;
    printf("Two-pass vs Online softmax:\n");
    for (int i=0;i<n;i++){
        float diff = fabsf(out_tp[i]-out_ol[i]);
        if(diff>max_diff) max_diff=diff;
        printf("  [%d] twopass=%.6f  online=%.6f  diff=%.2e\n",
               i, out_tp[i], out_ol[i], diff);
    }
    printf("Max diff: %.2e (expected < 1e-6)\n", max_diff);

    /* Large values test (overflow without online max tracking) */
    float x_large[4] = {1000.0f, 1001.0f, 999.0f, 1002.0f};
    float out_large[4];
    softmax_online(x_large, out_large, 4);
    float s=0;
    for(int i=0;i<4;i++) s+=out_large[i];
    printf("Large-value test: sum=%.6f (expected 1.0), NaN check: %s\n",
           s, (s==s)?"OK":"NaN!");
    return 0;
}
```

### Test Cases

| Test | Expected |
|------|----------|
| Standard input | `|online - twopass| < 1e-6` for all elements |
| `[1000, 1001, 999, 1002]` | Sum = 1.0, no NaN/Inf |
| `[-1000, -999, -998, -1001]` | Sum = 1.0, no NaN/Inf |

### Hints

1. The rescaling factor `exp(m - m_new)` is ≤ 1 when `m_new ≥ m`.
2. When a new maximum is found, old exponents must be rescaled because they were computed relative to the old max.
3. After the online pass, you have the correct `m` and `d`; the second pass just applies them.

### Solution Approach

The online softmax tracks the running maximum `m` and uses it to keep all exponentials in a numerically safe range. When `m` increases, old terms are rescaled by `exp(m_old - m_new)`, which is always < 1, preventing overflow. This is the key insight behind Flash Attention: by processing the input in tiles and maintaining `(m, d)` per tile, softmax can be computed without materializing the full `T×T` attention matrix.

---

## Exercise 8.4 — Parse GGUF Header

**Difficulty**: ★★★

### Problem

GGUF is the binary format used by llama.cpp. Parse a GGUF file header and print all tensor names, their data types, and shapes.

GGUF header structure (simplified):
```
magic        : uint32  (0x46554747, "GGUF")
version      : uint32  (3)
n_tensors    : uint64
n_kv         : uint64
[key-value pairs...]  (skip for now)
[tensor_info entries:]
  name_len   : uint64
  name       : char[name_len]
  n_dims     : uint32
  dims       : uint64[n_dims]
  type       : uint32  (0=F32, 1=F16, 8=Q8_0, ...)
  offset     : uint64
```

### Starter Code

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define GGUF_MAGIC   0x46554747u
#define GGUF_VERSION 3

/* GGUF data types */
const char *gguf_type_name(uint32_t t) {
    switch(t){
        case 0:  return "F32";
        case 1:  return "F16";
        case 7:  return "Q8_0";
        case 8:  return "Q5_0";
        case 15: return "Q4_K";
        case 16: return "Q6_K";
        default: return "???";
    }
}

/* Safe read helpers */
static uint32_t read_u32(FILE *f){ uint32_t v; fread(&v,4,1,f); return v; }
static uint64_t read_u64(FILE *f){ uint64_t v; fread(&v,8,1,f); return v; }

/*
 * Parse GGUF header and print tensor metadata.
 * Returns 0 on success.
 */
int parse_gguf(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror("fopen"); return -1; }

    /* Magic */
    uint32_t magic = read_u32(f);
    if (magic != GGUF_MAGIC) {
        printf("Not a GGUF file (magic=0x%08X)\n", magic);
        fclose(f); return -1;
    }
    uint32_t version = read_u32(f);
    printf("GGUF version: %u\n", version);

    uint64_t n_tensors = read_u64(f);
    uint64_t n_kv      = read_u64(f);
    printf("Tensors: %llu  Key-value pairs: %llu\n",
           (unsigned long long)n_tensors, (unsigned long long)n_kv);

    /* TODO: skip key-value section
       Each KV entry: name (len+bytes), type (uint32), value (variable).
       For now, use fseek to skip to tensor_info — or parse each entry.
       Hint: implementing a full KV parser is complex; for this exercise
       you may open a GGUF file and inspect it with a hex editor to find
       where tensor_info begins, then fseek directly. */

    /* Parse tensor info */
    printf("\n%-40s %-8s %s\n", "Name", "Type", "Shape");
    printf("%-40s %-8s %s\n", "----", "----", "-----");
    for (uint64_t i = 0; i < n_tensors; i++) {
        uint64_t name_len = read_u64(f);
        char name[256] = {0};
        if (name_len < 256) fread(name, 1, name_len, f);
        else { fseek(f, name_len, SEEK_CUR); strcpy(name, "<too_long>"); }

        uint32_t n_dims = read_u32(f);
        uint64_t dims[8] = {0};
        for (uint32_t d=0; d<n_dims && d<8; d++) dims[d] = read_u64(f);

        uint32_t type   = read_u32(f);
        uint64_t offset = read_u64(f);
        (void)offset;

        /* Print shape */
        char shape[64] = "[";
        for (uint32_t d=0; d<n_dims; d++){
            char tmp[32];
            snprintf(tmp, sizeof(tmp), "%llu%s",
                     (unsigned long long)dims[d], d<n_dims-1?", ":"");
            strncat(shape, tmp, sizeof(shape)-strlen(shape)-1);
        }
        strncat(shape, "]", sizeof(shape)-strlen(shape)-1);

        printf("%-40s %-8s %s\n", name, gguf_type_name(type), shape);
    }

    fclose(f);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s model.gguf\n", argv[0]);
        printf("Download a small GGUF model, e.g.:\n");
        printf("  curl -L https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/"
               "resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -o tiny.gguf\n");
        return 1;
    }
    return parse_gguf(argv[1]);
}
```

### Expected Output (for TinyLlama Q4_K_M)

```
GGUF version: 3
Tensors: 201  Key-value pairs: 23

Name                                     Type     Shape
----                                     ----     -----
token_embd.weight                        Q4_K     [2048, 32000]
blk.0.attn_norm.weight                   F32      [2048]
blk.0.ffn_down.weight                   Q4_K     [5632, 2048]
...
```

### Hints

1. GGUF is little-endian — on most modern machines no byte-swapping is needed.
2. The key-value section must be fully parsed (or skipped) before reaching tensor_info; KV entries have variable-length values depending on their type (uint8, uint32, string, array…).
3. Start by printing just the magic, version, and counts — verify those before parsing tensors.
4. Use `xxd model.gguf | head -20` to inspect the raw bytes if offsets seem wrong.

### Solution Approach

GGUF parsing is a sequential binary format walk. The hardest part is the key-value section which contains variable-length entries with multiple types. For this exercise, focus on getting the tensor metadata correct. In production (llama.cpp), the KV metadata encodes the model hyperparameters (n_layers, n_heads, vocab_size, etc.) that are needed to build the computation graph before loading weights.

---

## Exercise 8.5 — OpenMP Attention Head Loop

**Difficulty**: ★★★

### Problem

Parallelize the multi-head attention forward pass across heads using OpenMP, then benchmark tokens/sec for 4 vs 8 threads.

The workload: for each token position and each attention head, compute the attention output independently. Heads are embarrassingly parallel.

### Starter Code

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define T      64    /* sequence length */
#define DMODEL 512
#define NHEAD  8
#define DHEAD  (DMODEL/NHEAD)  /* 64 */

/* Simulated QKV buffers (pre-projected) */
float Q[T * DMODEL];
float K[T * DMODEL];
float V[T * DMODEL];
float OUT[T * DMODEL];

void mha_forward_parallel(int n_threads) {
#ifdef _OPENMP
    omp_set_num_threads(n_threads);
#endif

    float scale = 1.0f / sqrtf((float)DHEAD);

    /* Parallelize over heads */
    #pragma omp parallel for schedule(static) if(n_threads > 1)
    for (int h = 0; h < NHEAD; h++) {
        float attn[T * T];
        int   off_h = h * DHEAD;

        /* Compute attention scores for head h */
        for (int i = 0; i < T; i++) {
            for (int j = 0; j <= i; j++) {  /* causal */
                float s = 0;
                for (int d = 0; d < DHEAD; d++)
                    s += Q[i*DMODEL + off_h + d] * K[j*DMODEL + off_h + d];
                attn[i*T+j] = s * scale;
            }
            for (int j = i+1; j < T; j++) attn[i*T+j] = -1e9f;
        }

        /* Softmax each row */
        for (int i = 0; i < T; i++) {
            float m = -FLT_MAX, s = 0;
            for (int j = 0; j < T; j++) if (attn[i*T+j]>m) m=attn[i*T+j];
            for (int j = 0; j < T; j++){attn[i*T+j]=expf(attn[i*T+j]-m);s+=attn[i*T+j];}
            for (int j = 0; j < T; j++) attn[i*T+j]/=s;
        }

        /* Weighted sum of V */
        for (int i = 0; i < T; i++)
            for (int d = 0; d < DHEAD; d++) {
                float acc = 0;
                for (int j = 0; j < T; j++)
                    acc += attn[i*T+j] * V[j*DMODEL + off_h + d];
                OUT[i*DMODEL + off_h + d] = acc;
            }
    }
}

double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
    /* Initialize with random-ish values */
    for (int i = 0; i < T*DMODEL; i++) {
        Q[i] = (float)(i % 17 - 8) * 0.05f;
        K[i] = (float)(i % 13 - 6) * 0.05f;
        V[i] = (float)(i % 11 - 5) * 0.05f;
    }

    int thread_counts[] = {1, 2, 4, 8};
    int n_repeats = 200;

    printf("%-10s %-15s %-15s\n", "Threads", "Time(ms)/step", "Speedup");
    double base_time = 0;
    for (int ti = 0; ti < 4; ti++) {
        int nt = thread_counts[ti];
        double t0 = now_sec();
        for (int r = 0; r < n_repeats; r++)
            mha_forward_parallel(nt);
        double elapsed = (now_sec() - t0) / n_repeats * 1000.0;

        if (ti == 0) base_time = elapsed;
        printf("%-10d %-15.3f %-15.2fx\n", nt, elapsed, base_time / elapsed);
    }

    /* Verify correctness: compare 1-thread vs 8-thread output */
    float out_single[T*DMODEL], out_multi[T*DMODEL];
    mha_forward_parallel(1); memcpy(out_single, OUT, sizeof(OUT));
    mha_forward_parallel(8); memcpy(out_multi,  OUT, sizeof(OUT));

    float max_diff = 0;
    for (int i = 0; i < T*DMODEL; i++) {
        float d = fabsf(out_single[i] - out_multi[i]);
        if (d > max_diff) max_diff = d;
    }
    printf("\nSingle vs 8-thread max diff: %.2e (expected < 1e-5)\n", max_diff);
    return 0;
}
```

### Expected Output (approximate, hardware-dependent)

```
Threads    Time(ms)/step   Speedup
1          2.341           1.00x
2          1.231           1.90x
4          0.653           3.58x
8          0.389           6.02x

Single vs 8-thread max diff: 0.00e+00 (expected < 1e-5)
```

*(Speedup will vary. On machines with fewer physical cores, 4-thread and 8-thread times may be similar.)*

### Hints

1. Heads are fully independent — no data races when each thread writes to a distinct region of `OUT`.
2. The `attn[T*T]` buffer is **per-thread** (declared inside the parallel region) — this is critical to avoid races.
3. On macOS, OpenMP may require `brew install libomp` and `-Xpreprocessor -fopenmp -lomp`.
4. Measure wall-clock time with `CLOCK_MONOTONIC` (not CPU time) to capture parallel speedup.
5. `schedule(static)` with NHEAD=8 divides heads evenly among threads.

### Solution Approach

Multi-head attention is embarrassingly parallel across heads: each head reads from `Q`, `K`, `V` and writes to a distinct region of `OUT`, with no communication needed between heads. The `#pragma omp parallel for` directive distributes the head loop across threads. The main pitfall is ensuring the `attn` scratch buffer is thread-local (declared inside the loop, which means it lives on each thread's stack). For production inference, this parallelism is typically implemented with BLAS GEMM calls which use multi-threading internally.
