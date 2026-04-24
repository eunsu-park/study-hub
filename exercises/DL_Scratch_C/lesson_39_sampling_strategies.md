# Lesson 39 — Sampling Strategies (per-lesson exercise)

Prerequisites: L34 (cross-entropy / softmax), C stdlib random.

Compile: `gcc -std=c11 -Wall -Wextra -O2 -o ex ex.c -lm`

After a model produces a vector of logits $z \in \mathbb{R}^V$, a sampler turns it into a single token id. Two common strategies:

- **Top-k**: keep only the $k$ largest logits; re-softmax; sample.
- **Top-p (nucleus)**: sort logits descending, take the smallest prefix whose cumulative probability exceeds $p$; re-softmax; sample.

Greedy decoding is the degenerate case $k=1$ (or $p \to 0$).

---

## Exercise 39.1 — Argmax (Greedy)

**Difficulty**: ★

### Problem

Implement `int argmax(const float *logits, int V)` — simplest sampler, returns the index of the largest logit.

### Starter

```c
#include <stdio.h>

int argmax(const float *logits, int V) {
    /* TODO: linear scan, O(V) */
    (void)logits; (void)V;
    return 0;
}

int main(void) {
    float z[] = {0.1f, 2.3f, -1.0f, 2.29f, 1.5f};
    printf("argmax = %d (expected 1)\n", argmax(z, 5));
    return 0;
}
```

---

## Exercise 39.2 — Temperature + Softmax

**Difficulty**: ★★

### Problem

Implement `void softmax_t(const float *logits, float *probs, int V, float temperature)` that divides logits by `temperature`, applies the numerically-stable softmax, and writes the result into `probs`.

Key numerical trick: subtract `max(logits)` before exponentiating. Otherwise `expf(large)` overflows to infinity for models with ~30k vocab and a high-confidence logit.

### Starter

```c
#include <math.h>
#include <float.h>

void softmax_t(const float *logits, float *probs, int V, float temperature) {
    /* 1. find max_z for numerical stability */
    /* 2. compute exp((z[i] - max_z) / temperature) into probs */
    /* 3. normalize by the sum */
    /* TODO */
    (void)logits; (void)probs; (void)V; (void)temperature;
}
```

### Verification

With `logits = {1.0, 2.0, 3.0}` and `temperature = 1.0`, probabilities should be approximately `{0.0900, 0.2447, 0.6652}`. At `temperature = 0.1`, the largest element dominates (`~1.0`). At `temperature = 100.0`, the distribution approaches uniform.

---

## Exercise 39.3 — Top-k Sampling

**Difficulty**: ★★★

### Problem

Implement `int sample_topk(const float *logits, int V, int k, float temperature)` returning the sampled token index. Steps:

1. Copy logits to a scratch buffer.
2. Set all but the top-`k` logits to `-INFINITY`.
3. Apply `softmax_t` (reuses your 39.2 code).
4. Draw from the resulting distribution using `((float)rand() / RAND_MAX)` to get `u ∈ [0, 1)`, walk the cumulative probability array, and return the index where `u` is first exceeded.

### Starter

```c
#include <stdlib.h>
#include <string.h>

int sample_topk(const float *logits, int V, int k, float temperature) {
    /* Hint: sort (logit, index) pairs descending by logit to find top-k.
       qsort with a custom comparator works fine at this scale. */
    /* TODO */
    (void)logits; (void)V; (void)k; (void)temperature;
    return 0;
}
```

### Verification

Seed `srand(42)`. For `logits = {3.0, 2.5, 2.0, 1.0, 0.5, 0.1}`, `k=3`, `temperature=1.0`, over 10000 samples the empirical frequencies of the top-3 tokens (indices 0, 1, 2) should be close to their re-softmax probabilities `(approx 0.50, 0.31, 0.19)` within a couple of percent. The remaining three tokens should NEVER be sampled (frequency 0 exactly).

---

## Exercise 39.4 — Top-p (Nucleus) — Bonus

**Difficulty**: ★★★★

Implement `int sample_topp(const float *logits, int V, float p, float temperature)`. Same pipeline as top-k but the cutoff is determined by cumulative probability instead of count.

Subtle point: top-p is NOT equivalent to top-k for any fixed `k`, because the nucleus size varies with the distribution — a peaky distribution may have a nucleus of 3, a flat one may have a nucleus of 50. That adaptivity is exactly what top-p is designed for.
