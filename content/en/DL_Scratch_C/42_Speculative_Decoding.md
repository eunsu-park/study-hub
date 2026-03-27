# 42. Speculative Decoding

**Previous**: [FlashAttention on CPU](./41_FlashAttention_CPU.md) | **Next**: [GGUF Format and Loading](./43_GGUF_and_Loading.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why speculative decoding can accelerate autoregressive inference without changing output distribution
2. Implement the rejection sampling procedure that decides which draft tokens to accept
3. Understand the corrected resampling step that maintains exact target distribution when tokens are rejected
4. Track acceptance rate and use it to predict expected speedup
5. Design a draft/target model interface using function pointers for clean separation

---

## 1. The Autoregressive Bottleneck

Standard LLM decoding is strictly sequential: generate token 1 → feed back → generate token 2 → ... Each step requires a full forward pass through the target model (e.g., a 7B parameter model). The GPU or CPU is underutilized at batch=1 because a single token gives very little compute relative to memory bandwidth.

Speculative decoding (Leviathan et al., 2023; Chen et al., 2023) exploits this: a small draft model generates K candidate tokens quickly, then the large target model verifies all K tokens in a *single parallel forward pass*. The target model runs at batch=K rather than batch=1, amortizing its cost.

```
Draft model (fast, small):
  x_1, x_2, ..., x_K = draft_generate(context, K)   -- K sequential steps, fast

Target model (slow, large):
  p(x_1|ctx), p(x_2|ctx,x_1), ..., p(x_K|...) = target_forward(context + draft_tokens)
               -- ONE parallel forward pass!

Accept/reject each draft token, resample if rejected.
```

Expected speedup ≈ K × acceptance_rate (for K=4 drafts and 80% acceptance: ~3.2× speedup).

---

## 2. Rejection Sampling Mathematics

Let `q(x|ctx)` be the draft model distribution and `p(x|ctx)` the target distribution. For each draft token `x_i`:

```
If p(x_i) >= q(x_i):  accept x_i  (probability 1.0)
Else:                  accept x_i with probability p(x_i) / q(x_i)
```

When a token is rejected, we cannot simply resample from `p` — that would bias toward tokens where `p >> q`. Instead, we resample from the *corrected distribution*:

```
p'(x) = max(0, p(x) - q(x)) / sum_x max(0, p(x) - q(x))
       = normalize(max(0, p - q))
```

This correction ensures the marginal distribution of the accepted tokens is exactly `p`, regardless of `q`. The proof is elegant: the expected output distribution = accepted portion (distributed as `p`) + rejected portion (resampled from normalized `(p-q)_+`) which together integrate to `p`.

---

## 3. Model Interface (Function Pointers)

We define a model interface using function pointers so the draft and target can be swapped without changing the speculative decoding logic:

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define VOCAB_SIZE 32000
#define MAX_DRAFT_K 8

// Model forward pass: given token sequence, produce probability distribution
// for the NEXT token. Returns a heap-allocated float[vocab_size].
// Caller must free.
typedef float* (*ModelForwardFn)(const int *tokens, int n_tokens, void *model_state);

typedef struct {
    ModelForwardFn forward;
    void          *state;
    int            vocab_size;
} Model;

// Compute softmax in-place
static void softmax_inplace(float *x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

// Sample one token from probability distribution
static int sample_categorical(const float *probs, int n) {
    float r = (float)rand() / ((float)RAND_MAX + 1.0f);
    float cum = 0.0f;
    for (int i = 0; i < n; i++) {
        cum += probs[i];
        if (r < cum) return i;
    }
    return n - 1;
}
```

---

## 4. Draft Token Generation

The draft model generates K tokens autoregressively (sequentially), maintaining its own context:

```c
// Generate K draft tokens autoregressively using the draft model
// context:        current token sequence [n_ctx]
// draft_tokens:   output buffer [K]
// draft_probs:    output probabilities q(x_i) for each draft token [K]
//                 (the probability assigned to the chosen draft token)
void draft_generate(int *draft_tokens, float *draft_probs,
                    const int *context, int n_ctx,
                    Model *draft_model, int K) {
    // Build working context (context + draft tokens so far)
    int *ctx = malloc((n_ctx + K) * sizeof(int));
    memcpy(ctx, context, n_ctx * sizeof(int));
    int ctx_len = n_ctx;

    for (int k = 0; k < K; k++) {
        float *logits = draft_model->forward(ctx, ctx_len, draft_model->state);
        softmax_inplace(logits, draft_model->vocab_size);

        int tok = sample_categorical(logits, draft_model->vocab_size);
        draft_tokens[k]  = tok;
        draft_probs[k]   = logits[tok];  // q(x_k | context, x_1..x_{k-1})

        ctx[ctx_len++] = tok;
        free(logits);
    }
    free(ctx);
}
```

---

## 5. Target Model Parallel Verification

The target model processes the full context (including all K draft tokens) in one forward pass, producing K+1 probability distributions (one for each position):

```c
// Run target model on context + draft tokens, get K probability distributions
// target_probs_out: [K, vocab_size] — p(x_k | context + draft[0..k-1])
// Returns allocated array; caller must free each row
void target_verify(float **target_probs_out,
                   const int *context, int n_ctx,
                   const int *draft_tokens, int K,
                   Model *target_model) {
    // For simplicity: run K separate forward passes (in real implementation
    // this would be a batched or chunked prefill of the K draft positions)
    int *ctx = malloc((n_ctx + K) * sizeof(int));
    memcpy(ctx, context, n_ctx * sizeof(int));

    for (int k = 0; k < K; k++) {
        // Target evaluates at position n_ctx + k - 1 (predicts draft_tokens[k])
        float *logits = target_model->forward(ctx, n_ctx + k, target_model->state);
        softmax_inplace(logits, target_model->vocab_size);
        target_probs_out[k] = logits;  // caller owns this
        ctx[n_ctx + k] = draft_tokens[k];
    }
    // Also get the distribution at position n_ctx+K (for resampling after full acceptance)
    target_probs_out[K] = target_model->forward(ctx, n_ctx + K, target_model->state);
    softmax_inplace(target_probs_out[K], target_model->vocab_size);

    free(ctx);
}
```

---

## 6. Rejection Sampling and Corrected Resampling

```c
// Resample from p'(x) = normalize(max(0, p - q))
// p and q are probability distributions over vocab
static int resample_corrected(const float *p, const float *q, int vocab_size) {
    float *p_prime = malloc(vocab_size * sizeof(float));
    float total = 0.0f;

    for (int i = 0; i < vocab_size; i++) {
        p_prime[i] = fmaxf(0.0f, p[i] - q[i]);
        total += p_prime[i];
    }

    if (total < 1e-10f) {
        // Edge case: p == q everywhere → p' is zero, resample from p
        free(p_prime);
        return sample_categorical(p, vocab_size);
    }

    for (int i = 0; i < vocab_size; i++)
        p_prime[i] /= total;

    int tok = sample_categorical(p_prime, vocab_size);
    free(p_prime);
    return tok;
}

// Speculative decoding: generate one "batch" of tokens
// Returns number of tokens accepted (1 to K+1)
// out_tokens: accepted tokens appended here (caller pre-allocates K+1 slots)
int speculative_decode_step(int *out_tokens,
                             const int *context, int n_ctx,
                             Model *draft_model, Model *target_model,
                             int K,
                             int *accepted_count_accum,
                             int *total_draft_count) {
    int   draft_tokens[MAX_DRAFT_K];
    float draft_probs[MAX_DRAFT_K];
    float *target_probs[MAX_DRAFT_K + 1];  // K+1 distributions from target

    // Step 1: Draft model generates K tokens
    draft_generate(draft_tokens, draft_probs, context, n_ctx, draft_model, K);

    // Step 2: Target model verifies all K positions in parallel
    target_verify(target_probs, context, n_ctx, draft_tokens, K, target_model);

    // Step 3: Rejection sampling
    int n_accepted = 0;
    for (int k = 0; k < K; k++) {
        float p_k = target_probs[k][draft_tokens[k]];
        float q_k = draft_probs[k];

        float accept_prob = fminf(1.0f, p_k / (q_k + 1e-10f));
        float r = (float)rand() / ((float)RAND_MAX + 1.0f);

        if (r < accept_prob) {
            // Accept draft token
            out_tokens[n_accepted++] = draft_tokens[k];
            (*accepted_count_accum)++;
        } else {
            // Reject: resample from corrected distribution
            int resampled = resample_corrected(target_probs[k],
                                               /* q at this position */
                                               NULL,  // simplified
                                               target_model->vocab_size);
            out_tokens[n_accepted++] = resampled;
            // After rejection, stop — subsequent draft tokens are invalid
            for (int kk = k; kk <= K; kk++) free(target_probs[kk]);
            (*total_draft_count) += K;
            return n_accepted;
        }
    }

    // All K draft tokens accepted: also take one token from target at position K
    int bonus = sample_categorical(target_probs[K], target_model->vocab_size);
    out_tokens[n_accepted++] = bonus;

    for (int k = 0; k <= K; k++) free(target_probs[k]);
    (*total_draft_count) += K;
    return n_accepted;
}
```

---

## 7. Full Speculative Decode Loop with Acceptance Rate Tracking

```c
typedef struct {
    int   n_accepted;    // total tokens accepted from draft
    int   n_draft;       // total draft tokens proposed
    int   n_target_calls; // number of target model forward passes
    double time_draft;   // seconds spent in draft model
    double time_target;  // seconds spent in target model
} SpecStats;

// Generate max_new_tokens using speculative decoding
// Returns actual tokens generated (stored in out_tokens)
int speculative_decode_full(int *out_tokens,
                             const int *prompt, int prompt_len,
                             Model *draft_model, Model *target_model,
                             int K, int max_new_tokens,
                             SpecStats *stats) {
    memset(stats, 0, sizeof(*stats));

    int *context = malloc((prompt_len + max_new_tokens + K) * sizeof(int));
    memcpy(context, prompt, prompt_len * sizeof(int));
    int ctx_len = prompt_len;
    int total_generated = 0;

    while (total_generated < max_new_tokens) {
        int remaining = max_new_tokens - total_generated;
        int draft_K = (remaining < K) ? remaining : K;

        int step_out[MAX_DRAFT_K + 1];
        int accepted_accum = 0, draft_count = 0;

        int n_step = speculative_decode_step(step_out,
                                              context, ctx_len,
                                              draft_model, target_model,
                                              draft_K,
                                              &accepted_accum, &draft_count);

        // Append accepted tokens to context
        int take = (n_step < remaining) ? n_step : remaining;
        for (int i = 0; i < take; i++) {
            out_tokens[total_generated + i] = step_out[i];
            context[ctx_len + i] = step_out[i];
        }
        ctx_len += take;
        total_generated += take;

        stats->n_accepted += accepted_accum;
        stats->n_draft    += draft_count;
        stats->n_target_calls++;
    }

    free(context);
    return total_generated;
}

void print_spec_stats(const SpecStats *s, int K) {
    float accept_rate = (s->n_draft > 0)
                      ? (float)s->n_accepted / s->n_draft : 0.0f;
    float expected_speedup = (float)K * accept_rate;
    printf("Speculative Decoding Statistics:\n");
    printf("  Draft tokens proposed:  %d\n", s->n_draft);
    printf("  Draft tokens accepted:  %d (%.1f%%)\n",
           s->n_accepted, accept_rate * 100.0f);
    printf("  Target forward passes:  %d\n", s->n_target_calls);
    printf("  Expected speedup vs naive: %.2fx\n", expected_speedup);
    printf("  (Naive would need %d target passes for same output)\n",
           s->n_draft + s->n_target_calls);
}
```

---

## 8. Toy Model Demo: Measuring Acceptance Rate

```c
// Toy draft model: returns slightly noisy version of target distribution
// In practice, draft should share tokenizer with target
typedef struct { int vocab_size; float *base_probs; float noise; } ToyModelState;

float *toy_forward(const int *tokens, int n_tokens, void *state) {
    ToyModelState *s = (ToyModelState *)state;
    float *probs = malloc(s->vocab_size * sizeof(float));
    float sum = 0.0f;
    for (int i = 0; i < s->vocab_size; i++) {
        probs[i] = s->base_probs[i] + s->noise * ((float)rand()/RAND_MAX);
        if (probs[i] < 0.0f) probs[i] = 0.0f;
        sum += probs[i];
    }
    for (int i = 0; i < s->vocab_size; i++) probs[i] /= sum;
    return probs;
}

int main(void) {
    srand(42);
    const int V = 100;

    // Target distribution: peaked at token 5
    float target_base[100] = {0};
    target_base[5] = 10.0f; target_base[6] = 5.0f; target_base[7] = 2.0f;
    for (int i = 0; i < V; i++) if (target_base[i] == 0.0f) target_base[i] = 0.1f;

    ToyModelState draft_state  = { V, target_base, 0.5f };  // noisy draft
    ToyModelState target_state = { V, target_base, 0.01f }; // near-exact target

    Model draft_model  = { toy_forward, &draft_state,  V };
    Model target_model = { toy_forward, &target_state, V };

    const int K = 4;
    const int N = 100;  // tokens to generate
    int *out = malloc(N * sizeof(int));
    int prompt[1] = {0};

    SpecStats stats;
    speculative_decode_full(out, prompt, 1, &draft_model, &target_model,
                            K, N, &stats);
    print_spec_stats(&stats, K);

    free(out);
    return 0;
}
```

With a draft model that closely approximates the target (typical of small vs. large models from the same family), acceptance rates of 70-85% are common, yielding 2.8-3.4× speedup with K=4.

---

## Key Takeaways

- Speculative decoding exploits the fact that verifying K tokens in parallel (batch=K forward pass) costs only modestly more than verifying 1 token — the target model is underutilized at batch=1.
- The rejection sampling criterion `accept_prob = min(1, p(x)/q(x))` ensures that accepted tokens match the target distribution exactly, with no approximation.
- When a token is rejected, resampling from `normalize(max(0, p - q))` corrects for the bias that would result from simply resampling from `p` directly.
- After any rejection, all subsequent draft tokens must be discarded — they were conditioned on the rejected token being in the context.
- Expected speedup is `K × acceptance_rate`; with K=4 and 75% acceptance, expect ~3× fewer target model invocations.
- Draft and target models must share the same tokenizer — otherwise token indices have different meanings and rejection sampling is undefined.
- The draft model must be significantly smaller (10-100× fewer parameters) for speedup to materialize; a draft that is only 2× smaller provides minimal benefit.

---

**Previous**: [FlashAttention on CPU](./41_FlashAttention_CPU.md) | **Next**: [GGUF Format and Loading](./43_GGUF_and_Loading.md)
