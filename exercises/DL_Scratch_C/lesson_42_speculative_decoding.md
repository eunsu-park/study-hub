# Lesson 42 — Speculative Decoding (per-lesson exercise)

Prerequisites: L29 (GPT-2 forward), L39 (sampling strategies).

Compile: `gcc -std=c11 -Wall -Wextra -O2 -o ex ex.c -lm`

Speculative decoding accelerates autoregressive inference by having a SMALL "draft" model propose several tokens at once, which a LARGE "target" model verifies in one parallel forward pass. If the target agrees with the draft up to token $k$, all $k$ tokens are accepted for free — with zero change to the final distribution.

The win comes from the fact that target-model forward passes are latency-bound: generating one token takes almost the same wall-clock as generating eight in parallel. If the draft is right 3 tokens out of every 4 attempted, you get ~3× speedup with identical output quality.

---

## Exercise 42.1 — Rejection Sampling Correctness

**Difficulty**: ★★★

### Problem

The correctness of speculative decoding rests on an acceptance rule that preserves the target distribution $p(x)$:

```
for each proposed token x_i with draft probability q_i:
    compute p_i = target probability of x_i at position i
    if p_i >= q_i: accept unconditionally
    else:
        accept with probability p_i / q_i
        on rejection, resample from the adjusted distribution:
            r(x) = max(0, p(x) - q(x)) / sum_x max(0, p(x) - q(x))
```

Implement `bool speculative_accept(float p_i, float q_i, float u)` where `u ∈ [0, 1)` is a uniform draw. Then implement the resampling step for a small vocab (say $V = 10$).

### Starter

```c
#include <stdbool.h>
#include <math.h>

/* Acceptance rule for a single token */
bool speculative_accept(float p_i, float q_i, float u) {
    /* TODO: return true if p_i >= q_i || u < p_i / q_i */
    (void)p_i; (void)q_i; (void)u;
    return false;
}

/* Resample from the adjusted distribution r(x) when a token is rejected */
int speculative_resample(const float *p, const float *q, int V, float u) {
    /* 1. r[x] = fmaxf(0, p[x] - q[x]) for each x
       2. Z = sum(r)
       3. walk cumulative r / Z and return the first index where u falls */
    /* TODO */
    (void)p; (void)q; (void)V; (void)u;
    return 0;
}
```

---

## Exercise 42.2 — End-to-End Sampling Loop

**Difficulty**: ★★★

Chain draft + target models:

```
while not EOS:
    draft k tokens autoregressively from the draft model
    target forward pass in parallel, producing p for each of the k positions
    walk the k proposals, applying the acceptance rule
    if any rejection happens at position j:
        discard proposals j+1..k
        resample at position j using speculative_resample
        continue outer loop at position j+1
    else (all k accepted):
        emit all k tokens; the target model's final-position logits feed the next round
```

Wrap this so you can plug in any pair of forward functions. Test with a tiny toy where draft == target (acceptance rate must be 100%).

---

## Exercise 42.3 — Measured Speedup

**Difficulty**: ★★

Count the number of target-model forward passes used to generate 200 tokens at various draft lengths $k \in \{1, 2, 4, 8\}$. Plot the average tokens-per-target-call curve. Typical result: gains flatten out around $k = 4$ because acceptance probability drops geometrically with draft length.

The important takeaway is that speculative decoding NEVER changes the output distribution — it is lossless. Any speedup it provides is pure win.
