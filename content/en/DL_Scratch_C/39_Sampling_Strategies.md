# 39. Sampling Strategies

**Previous**: [KV Cache Optimization](./38_KV_Cache_Optimization.md) | **Next**: [Quantization: INT8 and INT4](./40_Quantization_Int8_Int4.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement greedy decoding (argmax) and explain when it fails to produce diverse output
2. Apply temperature scaling to logits before softmax to control output randomness
3. Implement top-k filtering to restrict the candidate token pool
4. Implement top-p (nucleus) sampling to adaptively cut off the tail of the distribution
5. Apply min-p and repetition penalty to further control output quality

---

## 1. The Token Sampling Problem

At each decode step, the model produces a logit vector of size `vocab_size` (e.g., 32,000 for Llama-3). Sampling strategy determines which token is selected next. The choice has a major impact on output quality:

- **Greedy**: always picks the highest-probability token. Deterministic, but can get stuck in repetitive loops.
- **Random sampling**: samples from the full distribution. Maximally diverse, but often incoherent.
- **Filtered sampling**: the practical sweet spot — restrict candidates, then sample.

All strategies operate on logits (pre-softmax scores). The general pipeline is:

```
logits[vocab] → apply filters → softmax → sample
```

We work in logit space because softmax is numerically sensitive and we want to avoid computing it multiple times.

---

## 2. Greedy Decoding (Argmax)

The simplest strategy: always pick the token with the highest logit.

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

// Returns index of maximum value in array
int argmax(const float *logits, int vocab_size) {
    int best = 0;
    float best_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return best;
}

// Greedy decode: deterministic, zero temperature
int greedy_sample(const float *logits, int vocab_size) {
    return argmax(logits, vocab_size);
}
```

**When greedy fails**: consider a model predicting the next word in "The cat sat on the ___". Greedy picks "mat" every time. The distribution might have meaningful probability mass on "floor", "chair", "roof" — greedy ignores all of that and produces repetitive, low-entropy text.

---

## 3. Temperature Sampling

Temperature `T` scales the logits before softmax:

```
logits_scaled[i] = logits[i] / T
probs = softmax(logits_scaled)
token = sample(probs)
```

- `T → 0`: approaches greedy (distribution sharpens to a spike)
- `T = 1.0`: standard softmax (no scaling)
- `T > 1.0`: flattens the distribution (more random)
- `T → ∞`: uniform distribution over all tokens

```c
// Numerically stable softmax (subtract max before exp)
void softmax(float *probs, const float *logits, int n) {
    float max_val = logits[0];
    for (int i = 1; i < n; i++)
        if (logits[i] > max_val) max_val = logits[i];

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        probs[i] = expf(logits[i] - max_val);
        sum += probs[i];
    }
    for (int i = 0; i < n; i++)
        probs[i] /= sum;
}

// Sample a token index from a probability distribution
int sample_from_probs(const float *probs, int n) {
    float r = (float)rand() / ((float)RAND_MAX + 1.0f);
    float cumsum = 0.0f;
    for (int i = 0; i < n; i++) {
        cumsum += probs[i];
        if (r < cumsum) return i;
    }
    return n - 1;  // fallback for floating-point edge case
}

// Temperature sampling: scale logits, then softmax, then sample
int temperature_sample(const float *logits, int vocab_size, float temperature) {
    if (temperature <= 0.0f) return greedy_sample(logits, vocab_size);

    float *scaled = malloc(vocab_size * sizeof(float));
    float *probs  = malloc(vocab_size * sizeof(float));

    for (int i = 0; i < vocab_size; i++)
        scaled[i] = logits[i] / temperature;

    softmax(probs, scaled, vocab_size);
    int token = sample_from_probs(probs, vocab_size);

    free(scaled);
    free(probs);
    return token;
}
```

Typical values: `T = 0.8` for focused creative text, `T = 1.2` for brainstorming.

---

## 4. Top-K Filtering

Top-k restricts sampling to only the `k` highest-logit tokens. All other logits are set to `-infinity` (so they get ~0 probability after softmax).

```c
// Comparison function for qsort (descending by value)
typedef struct { float val; int idx; } IndexedFloat;

int cmp_desc(const void *a, const void *b) {
    float va = ((IndexedFloat *)a)->val;
    float vb = ((IndexedFloat *)b)->val;
    return (va < vb) - (va > vb);
}

// Set all logits except top-k to -infinity
// Modifies logits_out in place; call before softmax
void top_k_filter(float *logits_out, const float *logits, int vocab_size, int k) {
    if (k <= 0 || k >= vocab_size) {
        memcpy(logits_out, logits, vocab_size * sizeof(float));
        return;
    }

    IndexedFloat *ranked = malloc(vocab_size * sizeof(IndexedFloat));
    for (int i = 0; i < vocab_size; i++) {
        ranked[i].val = logits[i];
        ranked[i].idx = i;
    }
    qsort(ranked, vocab_size, sizeof(IndexedFloat), cmp_desc);

    // Set all to -inf, then restore top-k
    for (int i = 0; i < vocab_size; i++)
        logits_out[i] = -FLT_MAX;
    for (int i = 0; i < k; i++)
        logits_out[ranked[i].idx] = ranked[i].val;

    free(ranked);
}

int top_k_sample(const float *logits, int vocab_size, int k, float temperature) {
    float *filtered = malloc(vocab_size * sizeof(float));
    float *probs    = malloc(vocab_size * sizeof(float));

    top_k_filter(filtered, logits, vocab_size, k);

    // Apply temperature after top-k filter
    if (temperature > 0.0f)
        for (int i = 0; i < vocab_size; i++)
            if (filtered[i] > -FLT_MAX) filtered[i] /= temperature;

    softmax(probs, filtered, vocab_size);
    int token = sample_from_probs(probs, vocab_size);

    free(filtered);
    free(probs);
    return token;
}
```

Typical values: `k = 50` or `k = 40`. Setting `k = 1` is equivalent to greedy.

---

## 5. Top-P (Nucleus) Sampling

Top-p, introduced by Holtzman et al. (2020), is adaptive: keep the smallest set of tokens whose cumulative probability exceeds `p`.

Why adaptive? With top-k, a fixed number of tokens are always considered regardless of how concentrated the distribution is. When the model is very confident (one token has 99% probability), top-k=50 still samples from 50 tokens. Top-p automatically collapses to a few tokens in this case.

```c
// Top-p (nucleus) filter: zero out tokens outside the nucleus
// Returns number of tokens kept
int top_p_filter(float *logits_out, const float *logits, int vocab_size, float p) {
    if (p >= 1.0f) {
        memcpy(logits_out, logits, vocab_size * sizeof(float));
        return vocab_size;
    }

    // Sort by logit value descending
    IndexedFloat *ranked = malloc(vocab_size * sizeof(IndexedFloat));
    for (int i = 0; i < vocab_size; i++) {
        ranked[i].val = logits[i];
        ranked[i].idx = i;
    }
    qsort(ranked, vocab_size, sizeof(IndexedFloat), cmp_desc);

    // Compute cumulative softmax probabilities
    float max_val = ranked[0].val;
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        ranked[i].val = expf(ranked[i].val - max_val);  // softmax numerator
        sum += ranked[i].val;
    }

    // Find cutoff index where cumulative prob > p
    float cumsum = 0.0f;
    int cutoff = vocab_size;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += ranked[i].val / sum;
        if (cumsum >= p) { cutoff = i + 1; break; }
    }

    // Build output: keep only nucleus tokens
    for (int i = 0; i < vocab_size; i++)
        logits_out[i] = -FLT_MAX;
    for (int i = 0; i < cutoff; i++)
        logits_out[ranked[i].idx] = logits[ranked[i].idx];  // restore original logit

    free(ranked);
    return cutoff;
}

int top_p_sample(const float *logits, int vocab_size, float p, float temperature) {
    float *filtered = malloc(vocab_size * sizeof(float));
    float *probs    = malloc(vocab_size * sizeof(float));

    top_p_filter(filtered, logits, vocab_size, p);

    if (temperature > 0.0f)
        for (int i = 0; i < vocab_size; i++)
            if (filtered[i] > -FLT_MAX) filtered[i] /= temperature;

    softmax(probs, filtered, vocab_size);
    int token = sample_from_probs(probs, vocab_size);

    free(filtered);
    free(probs);
    return token;
}
```

Typical value: `p = 0.9` or `p = 0.95`. Often combined with temperature: apply temperature first, then top-p.

---

## 6. Min-P Sampling

Min-p (introduced 2023) is simpler than top-p: filter out any token whose probability is less than `min_p × max_prob`. This scales the threshold relative to the model's confidence.

```c
// Min-p filter: remove tokens with prob < min_p * max_prob
void min_p_filter(float *logits_out, const float *logits, int vocab_size, float min_p) {
    // Compute softmax probabilities first
    float *probs = malloc(vocab_size * sizeof(float));
    softmax(probs, logits, vocab_size);

    float max_prob = 0.0f;
    for (int i = 0; i < vocab_size; i++)
        if (probs[i] > max_prob) max_prob = probs[i];

    float threshold = min_p * max_prob;
    for (int i = 0; i < vocab_size; i++)
        logits_out[i] = (probs[i] >= threshold) ? logits[i] : -FLT_MAX;

    free(probs);
}
```

Min-p is faster than top-p because it avoids sorting, and empirically produces similar or better output quality.

---

## 7. Repetition Penalty

Repetition penalty discourages the model from repeating recently generated tokens by dividing their logits by a penalty factor `> 1.0`:

```c
// Divide logits of previously seen tokens by penalty (> 1.0)
// tokens_seen: array of previously generated token IDs
// n_seen: number of tokens in history to penalize
void repetition_penalty_apply(float *logits, int vocab_size,
                               const int *tokens_seen, int n_seen,
                               float penalty) {
    if (penalty <= 1.0f) return;
    for (int i = 0; i < n_seen; i++) {
        int tok = tokens_seen[i];
        if (tok < 0 || tok >= vocab_size) continue;
        // Positive logits get reduced; negative logits get pushed more negative
        if (logits[tok] > 0.0f)
            logits[tok] /= penalty;
        else
            logits[tok] *= penalty;
    }
}
```

Why asymmetric handling of positive vs. negative logits? A positive logit contributes to high probability; dividing reduces it. A negative logit contributes to low probability; dividing would accidentally increase it. Multiplying pushes it further negative, consistently reducing probability.

Typical value: `penalty = 1.1` to `1.3`. Too high causes the model to avoid all common words.

---

## 8. The Sample Token Dispatch Function

Combining all strategies into a single configurable function:

```c
typedef enum {
    SAMPLE_GREEDY    = 0,
    SAMPLE_TEMP      = 1,
    SAMPLE_TOP_K     = 2,
    SAMPLE_TOP_P     = 3,
    SAMPLE_MIN_P     = 4,
} SampleStrategy;

typedef struct {
    SampleStrategy strategy;
    float temperature;   // for TEMP, TOP_K, TOP_P, MIN_P
    int   top_k;         // for TOP_K
    float top_p;         // for TOP_P
    float min_p;         // for MIN_P (e.g. 0.05)
    float rep_penalty;   // repetition penalty (1.0 = disabled)
} SamplerConfig;

// Modifies logits buffer (copy before calling if you need originals)
int sample_token(float *logits, int vocab_size,
                 const SamplerConfig *cfg,
                 const int *tokens_seen, int n_seen) {
    // 1. Apply repetition penalty in-place
    if (cfg->rep_penalty > 1.0f)
        repetition_penalty_apply(logits, vocab_size, tokens_seen, n_seen, cfg->rep_penalty);

    float *work = malloc(vocab_size * sizeof(float));
    float *probs = malloc(vocab_size * sizeof(float));
    int token;

    switch (cfg->strategy) {
        case SAMPLE_GREEDY:
            token = greedy_sample(logits, vocab_size);
            break;

        case SAMPLE_TEMP:
            token = temperature_sample(logits, vocab_size, cfg->temperature);
            break;

        case SAMPLE_TOP_K:
            top_k_filter(work, logits, vocab_size, cfg->top_k);
            if (cfg->temperature > 0.0f)
                for (int i = 0; i < vocab_size; i++)
                    if (work[i] > -FLT_MAX) work[i] /= cfg->temperature;
            softmax(probs, work, vocab_size);
            token = sample_from_probs(probs, vocab_size);
            break;

        case SAMPLE_TOP_P:
            top_p_filter(work, logits, vocab_size, cfg->top_p);
            if (cfg->temperature > 0.0f)
                for (int i = 0; i < vocab_size; i++)
                    if (work[i] > -FLT_MAX) work[i] /= cfg->temperature;
            softmax(probs, work, vocab_size);
            token = sample_from_probs(probs, vocab_size);
            break;

        case SAMPLE_MIN_P:
            min_p_filter(work, logits, vocab_size, cfg->min_p);
            if (cfg->temperature > 0.0f)
                for (int i = 0; i < vocab_size; i++)
                    if (work[i] > -FLT_MAX) work[i] /= cfg->temperature;
            softmax(probs, work, vocab_size);
            token = sample_from_probs(probs, vocab_size);
            break;

        default:
            token = greedy_sample(logits, vocab_size);
    }

    free(work);
    free(probs);
    return token;
}
```

---

## 9. Comparing Strategies on a Test Distribution

```c
// Simple test: 8-token vocabulary with known logits
// Shows how each strategy affects which tokens get sampled
void test_sampling_strategies(void) {
    srand(42);

    const int V = 8;
    // Logits: token 0 dominates, but tokens 1-3 have meaningful mass
    float logits_orig[8] = { 4.0f, 2.5f, 2.0f, 1.5f, 0.5f, -1.0f, -2.0f, -3.0f };
    const char *names[8] = {"cat","mat","sat","hat","bat","rat","vat","pat"};

    float probs[8];
    softmax(probs, logits_orig, V);

    printf("=== Base distribution ===\n");
    for (int i = 0; i < V; i++)
        printf("  %s: %.3f\n", names[i], probs[i]);

    // Count how often each token is sampled (1000 trials)
    int counts[4][8] = {0};
    SamplerConfig cfgs[4] = {
        { SAMPLE_GREEDY, 1.0f, 0,  0.0f, 0.0f, 1.0f },
        { SAMPLE_TEMP,   1.2f, 0,  0.0f, 0.0f, 1.0f },
        { SAMPLE_TOP_K,  1.0f, 3,  0.0f, 0.0f, 1.0f },
        { SAMPLE_TOP_P,  1.0f, 0,  0.9f, 0.0f, 1.0f },
    };
    const char *cfg_names[4] = {"Greedy", "Temp=1.2", "Top-K=3", "Top-P=0.9"};

    for (int s = 0; s < 4; s++) {
        for (int trial = 0; trial < 1000; trial++) {
            float logits[8];
            memcpy(logits, logits_orig, sizeof(logits));
            int tok = sample_token(logits, V, &cfgs[s], NULL, 0);
            counts[s][tok]++;
        }
    }

    for (int s = 0; s < 4; s++) {
        printf("\n=== %s ===\n", cfg_names[s]);
        for (int i = 0; i < V; i++)
            printf("  %s: %d/1000\n", names[i], counts[s][i]);
    }
}

int main(void) {
    test_sampling_strategies();
    return 0;
}
```

Expected output pattern:
- **Greedy**: "cat" 1000/1000 — completely deterministic
- **Temp=1.2**: spreads mass further into tail tokens
- **Top-K=3**: only "cat", "mat", "sat" appear
- **Top-P=0.9**: adaptive — likely just "cat" and "mat" since they already sum to ~90%

---

## Key Takeaways

- Greedy decoding is deterministic and fast but prone to repetitive, low-diversity outputs; it is the correct choice only for tasks requiring exact reproducibility.
- Temperature scales logits before softmax: values below 1 sharpen the distribution (more focused), values above 1 flatten it (more creative).
- Top-k filtering keeps a fixed number of candidates regardless of how concentrated or diffuse the distribution is — a potential mismatch when the model is very confident.
- Top-p (nucleus) sampling adapts the candidate pool size to the distribution shape, naturally collapsing to fewer tokens when the model is confident.
- Min-p is a simpler, O(n) alternative to top-p that scales the threshold relative to the top token's probability.
- Repetition penalty should handle positive and negative logits asymmetrically: divide positive logits, multiply negative logits to consistently reduce probability.
- In practice, combining temperature + top-p (e.g., T=0.8, p=0.9) is the default in most production inference engines.

---

**Previous**: [KV Cache Optimization](./38_KV_Cache_Optimization.md) | **Next**: [Quantization: INT8 and INT4](./40_Quantization_Int8_Int4.md)
