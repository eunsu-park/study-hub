# Lesson 38 — Training GPT-2 Small (per-lesson exercise)

Prerequisites: L28 (transformer block), L34 (cross-entropy), L35 (optimizers), L36 (training loop), L37 (backprop through transformer).

Compile: `gcc -std=c11 -Wall -Wextra -O3 -o ex ex.c -lm`

GPT-2 Small is the smallest of the original GPT-2 family: 124M parameters, 12 layers, 12 heads, 768-dim embeddings, 1024-token context. Training it from scratch in pure C is impractical without weeks of compute, but this exercise builds the full training script. Treat it as the project that integrates everything in DL_Scratch_C.

---

## Exercise 38.1 — Architecture Specification

**Difficulty**: ★

### Problem

Build a `GPT2Config` and matching `GPT2Model` struct. The hyperparameters:

```c
typedef struct {
    int vocab_size;        /* 50257 (BPE vocabulary) */
    int n_positions;       /* 1024  (max context length) */
    int n_layer;           /* 12 */
    int n_head;            /* 12 */
    int n_embd;            /* 768 */
    int d_ff;              /* 3072 = 4 * n_embd */
    float layer_norm_eps;  /* 1e-5 */
} GPT2Config;
```

Compute the parameter count:

- Token embedding: $50257 \times 768 = 38.6$M
- Position embedding: $1024 \times 768 = 0.8$M
- Per layer: ~7.1M (LN×2 + attn QKVO + FFN W1 W2 + biases)
- 12 layers total: ~85.2M
- Final LN: small
- LM head: tied to embedding (no extra parameters)
- **Total: ~124M** ✓

A handy sanity-check formula: $L \cdot 12 d^2 + 13 d V$ approximates GPT-2-class params, where $V$ is vocab size.

---

## Exercise 38.2 — Training Script Skeleton

**Difficulty**: ★★★

```c
int main(void) {
    GPT2Config cfg = {.vocab_size=50257, .n_positions=1024,
                       .n_layer=12, .n_head=12, .n_embd=768,
                       .d_ff=3072, .layer_norm_eps=1e-5f};
    GPT2Model *model = gpt2_init(&cfg);

    /* AdamW optimizer (DL_Scratch_C L35) — beta1=0.9, beta2=0.95, eps=1e-8, wd=0.1 */
    Optimizer *opt = adamw_init(model, /*lr*/ 6e-4f);

    /* Tokenized training data — produced by DL_Scratch_C L21 BPE */
    Dataset *data = dataset_load_pretokenized("./tinystories_tokenized.bin");

    /* Training loop with gradient accumulation, gradient clipping */
    int batch_size  = 8;
    int seq_length  = 1024;
    int total_steps = 10000;
    int log_every   = 100;

    for (int step = 0; step < total_steps; step++) {
        /* Sample a batch */
        Batch batch = dataset_sample(data, batch_size, seq_length);

        /* Forward */
        float *logits = gpt2_forward(model, batch.tokens, /*train=*/1);

        /* Cross-entropy loss + backward (fused, see L34) */
        float loss = gpt2_loss_and_backward(model, logits, batch.targets);

        /* Gradient clipping at global norm 1.0 (L36 sidebar) */
        float gnorm = clip_grad_norm(model, /*max_norm=*/1.0f);

        /* Step + zero grads */
        adamw_step(opt, model, /*step=*/step + 1);
        zero_grads(model);

        if (step % log_every == 0) {
            printf("step %5d  loss=%.4f  grad_norm=%.2f\n", step, loss, gnorm);
        }
    }

    gpt2_save_checkpoint(model, "./gpt2_small_step10000.bin");
    return 0;
}
```

The skeleton is short because every component (AdamW, BPE, attention, etc.) is a function from previous exercises. The integration is the lesson — see how the parts compose.

---

## Exercise 38.3 — Compute Budget Estimate

**Difficulty**: ★

Estimate the FLOPs to train one step:

- Forward pass: ~$6 N D^2 L$ FLOPs where $N$ = sequence length, $D$ = model dim, $L$ = layers (the "$6$" comes from QKVO + FFN). For GPT-2 Small at $N = 1024$: $6 \cdot 1024 \cdot 768^2 \cdot 12 \approx 4.6 \times 10^{10}$ FLOPs per token.
- Per training token (forward + backward + optimizer): $\approx 6$ flops per parameter per token = $6 \times 124 \times 10^6 = 7.4 \times 10^8$ FLOPs/token.

For a 10B-token training run: $\sim 7.4 \times 10^{18}$ FLOPs. On a single 80 TFLOPS A100 at 50% utilization: $7.4 \times 10^{18} / (40 \times 10^{12}) \approx 51$ hours. The original GPT-2 training was a multi-day job on multi-GPU; in modern terms, it is a one-A100-night project.

---

## Exercise 38.4 — Validation Loop and Perplexity

**Difficulty**: ★★

Implement a validation loop that runs forward (no backward) on a held-out set and computes perplexity:

$$PPL = \exp\!\left(\frac{1}{N} \sum_n \text{cross\_entropy}(logits_n, target_n)\right)$$

A trained GPT-2 Small reaches ~35 perplexity on WebText-like data. If your training reproduces something between 30 and 40 after the full schedule, you have built a working LLM training pipeline from scratch — celebrate appropriately.

---

## Exercise 38.5 — Generation — Bonus

**Difficulty**: ★★★

Add a `gpt2_generate(model, prompt, n_new_tokens)` that:

1. Tokenizes the prompt with your BPE encoder (DL_Scratch_C L21).
2. Allocates KV caches (L26).
3. Runs the prefill phase (forward pass on all prompt tokens to populate caches).
4. For `n_new_tokens` iterations: forward one new token, sample with top-k or top-p (L39), append to context.
5. Detokenizes the result.

This is the inference loop every chatbot uses internally. Pairing it with your training code closes the loop on the entire DL_Scratch_C course.
