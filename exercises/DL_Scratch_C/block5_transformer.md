# Block 5 — Transformer (L24–L30)

Prerequisites: L24 (LayerNorm/RMSNorm), L25 (attention), L26 (causal masking), L27 (KV cache), L28 (FFN/SwiGLU), L29 (full Transformer block), L30 (GPT-2 weight loading).

Compile: `gcc -std=c11 -Wall -Wextra -O2 -o ex ex.c -lm`

---

## Exercise 5.1 — RMSNorm Forward

**Difficulty**: ★★

### Problem

Implement `rmsnorm_forward(float *out, const float *x, const float *gamma, int d)`:

```
RMS(x)  = sqrt( (1/d) * sum(x_i^2) + eps )
out_i   = (x_i / RMS(x)) * gamma_i
```

Then verify: when `x` has mean=0, RMSNorm and LayerNorm produce the same output (up to numerical precision).

### Starter Code

```c
#include <stdio.h>
#include <math.h>
#include <string.h>

#define EPS 1e-5f

void rmsnorm_forward(float *out, const float *x, const float *gamma, int d) {
    /* Step 1: compute RMS */
    float rms = 0.0f;
    /* TODO */

    /* Step 2: normalize and scale */
    /* TODO */
}

/* Reference LayerNorm (provided) */
void layernorm_forward(float *out, const float *x, const float *gamma, const float *beta, int d) {
    float mean = 0, var = 0;
    for (int i = 0; i < d; i++) mean += x[i];
    mean /= d;
    for (int i = 0; i < d; i++) var += (x[i]-mean)*(x[i]-mean);
    var /= d;
    float inv_std = 1.0f / sqrtf(var + EPS);
    for (int i = 0; i < d; i++)
        out[i] = gamma[i] * (x[i] - mean) * inv_std + beta[i];
}

int main(void) {
    int d = 8;
    /* x with mean=0: [-3,-2,-1,0,1,2,3,0] */
    float x[8]     = {-3,-2,-1,0,1,2,3,0};
    float gamma[8] = {1,1,1,1,1,1,1,1};
    float beta[8]  = {0,0,0,0,0,0,0,0};

    float out_rms[8], out_ln[8];
    rmsnorm_forward(out_rms, x, gamma, d);
    layernorm_forward(out_ln, x, gamma, beta, d);

    printf("RMSNorm vs LayerNorm (mean=0 input):\n");
    float max_diff = 0;
    for (int i = 0; i < d; i++) {
        float diff = fabsf(out_rms[i] - out_ln[i]);
        if (diff > max_diff) max_diff = diff;
        printf("  [%d] rms=%.6f  ln=%.6f  diff=%.2e\n",
               i, out_rms[i], out_ln[i], diff);
    }
    printf("Max diff: %.2e (expected < 1e-4)\n", max_diff);

    /* Non-zero-mean input: should differ */
    float x2[8] = {1,2,3,4,5,6,7,8};
    rmsnorm_forward(out_rms, x2, gamma, d);
    layernorm_forward(out_ln, x2, gamma, beta, d);
    printf("\nNon-zero-mean: should differ\n");
    printf("  rms[0]=%.4f  ln[0]=%.4f\n", out_rms[0], out_ln[0]);
    return 0;
}
```

### Test Cases

| Input x | Expected behavior |
|---------|------------------|
| `[-3,-2,-1,0,1,2,3,0]` (mean=0) | `|rmsnorm - layernorm| < 1e-4` for all elements |
| `[1,2,3,4,5,6,7,8]` (mean≠0) | rmsnorm ≠ layernorm (both valid but different) |
| All-ones `[1,1,...,1]` | `rmsnorm[i] = 1.0 * gamma[i]` |

### Hints

1. RMS = `sqrt(mean(x^2) + eps)` — average the squares, then take the square root.
2. When mean=0, `Var(x) = E[x^2]`, so LayerNorm denominator equals RMSNorm denominator.
3. LayerNorm subtracts the mean first; RMSNorm does not — this is the key difference.

### Solution Approach

One pass over `x` to accumulate sum of squares, divide by `d`, add `eps`, take sqrt. Then a second pass to normalize and multiply by `gamma`. RMSNorm is preferred in modern LLMs (Llama, Mistral) because it removes the mean-centering operation, saving a pass and avoiding the mean subtraction's numerical issues at large scale.

---

## Exercise 5.2 — Causal Mask in `mha_forward`

**Difficulty**: ★★

### Problem

Extend a provided single-head attention implementation to apply a **causal (lower-triangular) mask** that prevents position `i` from attending to positions `j > i`.

After masking, verify that the upper-triangular entries of the attention weight matrix are approximately zero.

### Starter Code

```c
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

void softmax_inplace(float *x, int n) {
    float max_v = -FLT_MAX;
    for (int i = 0; i < n; i++) if (x[i] > max_v) max_v = x[i];
    float s = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max_v); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

/*
 * Single-head attention forward.
 * Q, K, V: [T][d_head]
 * out:     [T][d_head]
 * causal:  if 1, apply causal mask
 */
void attention_forward(const float *Q, const float *K, const float *V,
                       float *out, int T, int d_head, int causal) {
    float scale = 1.0f / sqrtf((float)d_head);
    float *attn = malloc(T * T * sizeof(float));  /* [T][T] */

    /* Compute QK^T * scale */
    for (int i = 0; i < T; i++)
        for (int j = 0; j < T; j++) {
            float s = 0;
            for (int d = 0; d < d_head; d++)
                s += Q[i*d_head+d] * K[j*d_head+d];
            attn[i*T+j] = s * scale;
        }

    /* TODO: apply causal mask — set attn[i][j] = -1e9 for j > i */
    if (causal) {
        /* ... */
    }

    /* Softmax each row */
    for (int i = 0; i < T; i++)
        softmax_inplace(attn + i*T, T);

    /* Print attention weights for inspection */
    printf("Attention weights (T=%d):\n", T);
    for (int i = 0; i < T; i++) {
        for (int j = 0; j < T; j++)
            printf("%.4f ", attn[i*T+j]);
        printf("\n");
    }

    /* Weighted sum of V */
    memset(out, 0, T * d_head * sizeof(float));
    for (int i = 0; i < T; i++)
        for (int j = 0; j < T; j++)
            for (int d = 0; d < d_head; d++)
                out[i*d_head+d] += attn[i*T+j] * V[j*d_head+d];

    free(attn);
}

int main(void) {
    int T=4, d_head=8;
    float Q[4*8], K[4*8], V[4*8], out[4*8];
    for (int i = 0; i < T*d_head; i++) {
        Q[i] = (float)(rand()%10)*0.1f;
        K[i] = (float)(rand()%10)*0.1f;
        V[i] = (float)(rand()%10)*0.1f;
    }

    printf("=== WITHOUT causal mask ===\n");
    attention_forward(Q, K, V, out, T, d_head, 0);

    printf("\n=== WITH causal mask ===\n");
    attention_forward(Q, K, V, out, T, d_head, 1);
    printf("(Upper triangle should be ~0)\n");
    return 0;
}
```

### Test Cases

With causal mask applied (T=4):
- `attn[0][1]`, `attn[0][2]`, `attn[0][3]` ≈ 0 (position 0 cannot see future)
- `attn[1][2]`, `attn[1][3]` ≈ 0
- `attn[2][3]` ≈ 0
- Lower triangle + diagonal can be any positive values summing to 1 per row.

### Hints

1. The mask sets future positions to a very large negative value (e.g., -1e9) before softmax.
2. After softmax, `exp(-1e9) ≈ 0` — the exact value does not matter as long as it is sufficiently negative.
3. `j > i` defines the upper triangle (excluding the diagonal, which is a present-position self-attend).

### Solution Approach

Add a double loop after computing `QK^T`: for `i` in `[0,T)`, for `j` in `[i+1, T)`, set `attn[i*T+j] = -1e9f`. Then proceed to softmax. The mask is applied before softmax so the exponentials of masked positions become negligible.

---

## Exercise 5.3 — KV Cache for Decoder

**Difficulty**: ★★★

### Problem

Implement a KV cache for autoregressive decoding (batch=1). The cache stores previously computed key and value vectors so that at each new token we only compute attention for the new position, not all previous positions.

Implement:
1. `kvcache_init(KVCache *c, int max_seq, int d_head)` — allocate storage.
2. `kvcache_append(KVCache *c, const float *k_new, const float *v_new, int d_head)` — append one new KV pair.
3. `cached_attention_forward(const float *q, KVCache *c, float *out, int d_head)` — attend over all cached KV pairs using query `q` (single vector, not a sequence).

### Starter Code

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>

typedef struct {
    float *K;    /* [max_seq][d_head] */
    float *V;    /* [max_seq][d_head] */
    int    len;  /* current number of cached steps */
    int    max_seq;
} KVCache;

void kvcache_init(KVCache *c, int max_seq, int d_head) {
    c->K = calloc(max_seq * d_head, sizeof(float));
    c->V = calloc(max_seq * d_head, sizeof(float));
    c->len = 0;
    c->max_seq = max_seq;
}

/* Append a single new (k, v) pair to the cache. */
void kvcache_append(KVCache *c, const float *k_new, const float *v_new, int d_head) {
    /* TODO: copy k_new and v_new into c->K[c->len] and c->V[c->len],
             then increment c->len. Assert c->len <= c->max_seq. */
}

/*
 * Attend from a single query vector q[d_head] over all c->len cached KV pairs.
 * out[d_head] = sum_j( softmax(q*K[j]/sqrt(d)) * V[j] )
 */
void cached_attention_forward(const float *q, KVCache *c, float *out, int d_head) {
    int T = c->len;
    float scale = 1.0f / sqrtf((float)d_head);

    float *scores = malloc(T * sizeof(float));
    /* TODO: scores[j] = dot(q, c->K[j]) * scale */

    /* Softmax */
    float max_s = -FLT_MAX;
    for (int j = 0; j < T; j++) if (scores[j] > max_s) max_s = scores[j];
    float s = 0;
    for (int j = 0; j < T; j++) { scores[j] = expf(scores[j]-max_s); s += scores[j]; }
    for (int j = 0; j < T; j++) scores[j] /= s;

    /* TODO: out = sum_j(scores[j] * c->V[j]) */
    memset(out, 0, d_head * sizeof(float));

    free(scores);
}

int main(void) {
    int d_head=4, max_seq=8;
    KVCache cache;
    kvcache_init(&cache, max_seq, d_head);

    /* Simulate 3 decode steps */
    float k0[4]={1,0,0,0}, v0[4]={10,10,10,10};
    float k1[4]={0,1,0,0}, v1[4]={20,20,20,20};
    float k2[4]={0,0,1,0}, v2[4]={30,30,30,30};

    kvcache_append(&cache, k0, v0, d_head);
    kvcache_append(&cache, k1, v1, d_head);
    kvcache_append(&cache, k2, v2, d_head);

    /* Query aligned with k0 => should retrieve v0 */
    float q0[4]={10,0,0,0}, out[4];
    cached_attention_forward(q0, &cache, out, d_head);
    printf("Query~k0: out[0]=%.2f (expected close to 10.0)\n", out[0]);

    /* Query aligned with k2 => should retrieve v2 */
    float q2[4]={0,0,10,0};
    cached_attention_forward(q2, &cache, out, d_head);
    printf("Query~k2: out[0]=%.2f (expected close to 30.0)\n", out[0]);

    free(cache.K); free(cache.V);
    return 0;
}
```

### Test Cases

- After appending k0, k1, k2 with orthogonal key vectors:
  - A query aligned with k0 should return approximately v0.
  - A query aligned with k2 should return approximately v2.
- `cache.len` should equal 3 after three appends.
- Appending beyond `max_seq` should abort or return an error.

### Hints

1. `kvcache_append` is a simple `memcpy` into the next slot followed by `c->len++`.
2. `cached_attention_forward` only loops over `c->len` positions, not `max_seq`.
3. The larger the dot product `q·k_j`, the more the output is pulled toward `V[j]`.

### Solution Approach

The KV cache trades memory for speed: instead of recomputing keys and values for all past tokens at each decode step, we store them once and reuse. Each decode step is O(T*d_head) instead of O(T^2*d_head) amortized. The implementation is straightforward — the interesting part is understanding why this is correct: autoregressive decoding does not change past key/value vectors (with no RoPE recomputation needed in the base case).

---

## Exercise 5.4 — SwiGLU Backward

**Difficulty**: ★★★

### Problem

SwiGLU is the FFN activation used in Llama/PaLM:

```
SwiGLU(x) = SiLU(gate) * up
```

where `gate = x @ W_gate`, `up = x @ W_up`, `SiLU(z) = z * sigmoid(z)`, and the output is then projected: `out = SwiGLU(x) @ W_down`.

Given `d_out` (upstream gradient), derive and implement the backward pass to compute `d_gate`, `d_up`, and `d_down`.

You only need the backward through the element-wise operation `SiLU(gate) * up`, not through the linear projections.

### Starter Code

```c
#include <stdio.h>
#include <math.h>
#include <string.h>

void silu_inplace(float *x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = x[i] / (1.0f + expf(-x[i]));
}

/*
 * SwiGLU forward (element-wise part only).
 * gate[d], up[d] -> out[d] = silu(gate) * up
 */
void swiglu_forward(const float *gate, const float *up, float *out, int d) {
    for (int i = 0; i < d; i++) {
        float sg = gate[i] / (1.0f + expf(-gate[i]));  /* silu(gate[i]) */
        out[i] = sg * up[i];
    }
}

/*
 * SwiGLU backward.
 * Given d_out[d], compute d_gate[d] and d_up[d].
 *
 * Let s = sigmoid(gate), silu = gate * s
 * d_silu/d_gate = s + gate * s * (1-s) = s * (1 + gate*(1-s))
 * d_out / d_gate = d_out * up * d_silu/d_gate
 * d_out / d_up   = d_out * silu(gate)
 */
void swiglu_backward(const float *gate, const float *up,
                     const float *d_out,
                     float *d_gate, float *d_up,
                     int d) {
    /* TODO: for each i:
         sig   = sigmoid(gate[i])
         silu  = gate[i] * sig
         dsilu_dgate = sig * (1 + gate[i] * (1 - sig))
         d_gate[i] = d_out[i] * up[i] * dsilu_dgate
         d_up[i]   = d_out[i] * silu                       */
}

/* Finite difference check */
float fd_gate(int idx, const float *gate, const float *up, int d, float eps) {
    float g[d]; for (int i=0;i<d;i++) g[i]=gate[i];
    float out_p[d], out_m[d];
    g[idx] += eps; swiglu_forward(g, up, out_p, d);
    g[idx] -= 2*eps; swiglu_forward(g, up, out_m, d);
    /* Loss = sum(out), so dL = sum(out_p) - sum(out_m) */
    float lp=0, lm=0;
    for (int i=0;i<d;i++){lp+=out_p[i];lm+=out_m[i];}
    return (lp - lm) / (2*eps);
}

int main(void) {
    int d = 4;
    float gate[4] = {0.5f, -1.0f, 2.0f, 0.0f};
    float up[4]   = {1.0f,  1.0f, 0.5f, 3.0f};
    float d_out[4]= {1.0f,  1.0f, 1.0f, 1.0f}; /* d/d(sum(out)) */

    float d_gate[4], d_up[4];
    swiglu_backward(gate, up, d_out, d_gate, d_up, d);

    printf("Analytical vs numerical gradient for gate:\n");
    float eps = 1e-4f;
    int ok = 1;
    for (int i = 0; i < d; i++) {
        float num = fd_gate(i, gate, up, d, eps);
        float diff = fabsf(d_gate[i] - num);
        printf("  [%d] anal=%.6f  num=%.6f  diff=%.2e\n", i, d_gate[i], num, diff);
        if (diff > 1e-3f) ok = 0;
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
    return 0;
}
```

### Test Cases

For `gate=[0.5, -1.0, 2.0, 0.0]`, `up=[1.0, 1.0, 0.5, 3.0]`, `d_out=all-ones`:
- All `|d_gate[i] - finite_diff[i]|` must be less than 1e-3.
- `d_up[i] = silu(gate[i])` — this part is simpler.

### Hints

1. SiLU derivative: `d/dz [z*σ(z)] = σ(z) + z*σ(z)*(1-σ(z)) = σ(z)*(1 + z*(1-σ(z)))`.
2. Chain rule: `d_gate[i] = d_out[i] * up[i] * d_silu_d_gate[i]`.
3. `d_up[i] = d_out[i] * silu(gate[i])` — silu acts as a coefficient on `up`.
4. Compute and cache `sig = sigmoid(gate[i])` first; reuse for both silu and its derivative.

### Solution Approach

Work out the chain rule on paper first. The forward computation is `out = silu(gate) * up`. The "gate" path requires the SiLU derivative (chain rule through a product of a function and its sigmoid). The "up" path is simpler — the coefficient is just `silu(gate)`. Verify with finite differences.

---

## Exercise 5.5 — Verify GPT-2 Logits

**Difficulty**: ★★★★

### Problem

Load GPT-2 124M weights from the official OpenAI checkpoint and verify that the first 5 logits for the input sequence `[15496, 11, 995, 0]` (tokens for "Hello, world!") match:

```
Expected logits: [-35.73, -34.90, -37.81, -38.72, -38.15] ± 0.01
```

This exercise requires:
1. Downloading GPT-2 weights (via `curl` or Python's `transformers` library).
2. Parsing the checkpoint format (NumPy `.npy` files or `safetensors`).
3. Running a full forward pass through all 12 transformer layers.
4. Comparing the output logits.

### Starter Code

```c
/*
 * gpt2_verify.c
 *
 * This is a high-level scaffold. Implementing a full GPT-2 forward pass
 * is a multi-file project. The key function to implement is:
 *   void gpt2_forward(GPT2 *model, const int *tokens, int T, float *logits)
 *
 * GPT-2 124M architecture:
 *   - vocab_size = 50257
 *   - n_layer    = 12
 *   - n_head     = 12
 *   - d_model    = 768
 *   - d_ff       = 3072
 *   - max_seq    = 1024
 *
 * Weight files (download from HuggingFace or OpenAI):
 *   wte    : [50257, 768]   token embeddings
 *   wpe    : [1024,  768]   position embeddings
 *   blocks : 12 × {
 *              ln_1.{weight,bias}   [768]
 *              attn.{c_attn,c_proj} weights + biases
 *              ln_2.{weight,bias}   [768]
 *              mlp.{c_fc,c_proj}    weights + biases
 *            }
 *   ln_f   : [768]          final layer norm
 *   (lm_head shares weights with wte)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define VOCAB  50257
#define NLAY   12
#define NHEAD  12
#define DMODEL 768

/* TODO: define weight struct and loading function */

int main(void) {
    int tokens[4] = {15496, 11, 995, 0};
    int T = 4;

    /* TODO: load weights */
    /* TODO: gpt2_forward(model, tokens, T, logits) */

    /* Print and compare first 5 logits */
    float expected[5] = {-35.73f, -34.90f, -37.81f, -38.72f, -38.15f};
    printf("Logit comparison (last token position):\n");
    printf("%-6s %-12s %-12s %-10s\n", "Index", "Computed", "Expected", "Error");
    /* TODO: fill in computed logits */
    for (int i = 0; i < 5; i++) {
        float computed = 0.0f; /* placeholder */
        printf("%-6d %-12.4f %-12.4f %-10.4f %s\n",
               i, computed, expected[i], fabsf(computed - expected[i]),
               fabsf(computed - expected[i]) < 0.01f ? "OK" : "FAIL");
    }
    return 0;
}
```

### Steps to Get Weights

```bash
# Option A: via Python
python3 -c "
from transformers import GPT2Model
import numpy as np
m = GPT2Model.from_pretrained('gpt2')
sd = m.state_dict()
for k,v in sd.items():
    np.save(f'{k.replace(\"/\",\"_\")}.npy', v.numpy())
print('Saved', len(sd), 'weight files')
"

# Option B: direct download from OpenAI (checkpoint.zip, ~500MB)
# See: https://github.com/openai/gpt-2
```

### Hints

1. GPT-2 uses GELU activation in the FFN, not SwiGLU.
2. The attention projection `c_attn` is a single `[768, 3*768]` matrix that produces Q, K, V by slicing.
3. Position embedding: add `wpe[t]` to `wte[token[t]]` for each position `t`.
4. After the final layer norm (`ln_f`), the logits are computed as `h @ wte.T` (weight tying).
5. Start by verifying each component individually (embedding lookup, single attention head, etc.) before running the full 12-layer forward.

### Solution Approach

This is an integration exercise. Break it into phases: (1) load weights and verify shapes, (2) implement embedding lookup and verify output matches Python reference, (3) implement a single transformer block and verify against `transformers`, (4) stack all 12 blocks. The expected logits serve as an end-to-end regression test. Passing this exercise means your entire implementation — attention, layer norm, FFN, weight loading — is numerically correct.
