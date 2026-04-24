# Lesson 29 — GPT-2 Single-Token Forward Pass (per-lesson exercise)

Prerequisites: L24 (LayerNorm), L25 (attention), L26 (KV cache), L28 (transformer block).

Compile: `gcc -std=c11 -Wall -Wextra -O2 -o ex ex.c -lm`

A single decoder-only transformer forward pass, for one new token, consists of: embedding lookup → N transformer blocks → final LayerNorm → un-embedding → logits. This exercise wires those pieces together.

---

## Exercise 29.1 — The Skeleton

**Difficulty**: ★★★

### Problem

Implement `gpt2_forward_one_token(const GPT2 *model, int token_id, KVCache *caches, float *logits)` where `caches` is an array of per-layer KV caches already allocated for the full context.

Pseudocode:

```
x = tok_embed[token_id]           // shape [d]
x += pos_embed[current_position]  // shape [d], using the cache's current length

for layer in 0..N-1:
    h = layernorm_1(x, ln1_weight[layer], ln1_bias[layer])
    attn_out = attention_with_cache(h, caches[layer], W_q, W_k, W_v, W_o)
    x = x + attn_out

    h = layernorm_2(x, ln2_weight[layer], ln2_bias[layer])
    ffn_out = ffn(h, W_fc, b_fc, W_proj, b_proj)  // GELU activation
    x = x + ffn_out

x = layernorm_final(x, ln_f_weight, ln_f_bias)
logits = x @ tok_embed.T          // weight tying: un-embedding = transpose of embedding
```

### Starter

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int n_layers;
    int d_model;
    int n_heads;
    int vocab_size;
    int max_context;
    /* Pointers to all weight tensors — omitted for brevity. */
} GPT2;

typedef struct { /* as in lesson 26 exercise */ } KVCache;

void gpt2_forward_one_token(const GPT2 *model, int token_id,
                            KVCache *caches, float *logits) {
    /* TODO: implement the pseudocode above.
       Reuse your functions from lessons 24 (layernorm) and 25 (attention). */
    (void)model; (void)token_id; (void)caches; (void)logits;
}
```

---

## Exercise 29.2 — Sampling Loop

**Difficulty**: ★★

Chain `gpt2_forward_one_token` with your sampler from Lesson 39 to produce 50 tokens from a prompt. Time the first token vs. the 50th. The first call is "prefill" — every layer of the cache is filled for every prompt token. The 50th call is "decode" — only one token's $K$/$V$ are computed and the cache attends to the rest. Expect a ~50× speedup per-token after the first.

---

## Exercise 29.3 — Numeric Parity — Bonus

**Difficulty**: ★★★★

Load the fp32 weights of GPT-2 124M from the Hugging Face checkpoint and verify, on a fixed prompt, that your C implementation's first-token logits match the Python reference to within $10^{-3}$ relative error. The failure mode is usually LayerNorm's `eps` value or attention's softmax ordering — both are historically source of parity bugs.
