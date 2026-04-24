# Lesson 41 — FlashAttention on CPU (per-lesson exercise)

Prerequisites: L25 (attention), L26 (KV cache), familiarity with block-matrix multiplication.

Compile: `gcc -std=c11 -Wall -Wextra -O2 -o ex ex.c -lm`

Standard attention materializes the $[N, N]$ score matrix. FlashAttention computes the same result without ever storing that matrix — it processes queries in blocks and streams keys/values through, fusing softmax with the matmul. On GPU this is a massive memory-bandwidth win; on CPU the idea still reduces the working set enough to stay in L2 cache, which is often what matters.

This exercise implements a single-threaded CPU version so you can verify correctness and understand the online-softmax math before doing the CUDA version (covered in the CUDA course).

---

## Exercise 41.1 — Online Softmax

**Difficulty**: ★★★

### Problem

A naive softmax over an array requires two passes: one for the max, one for the normalizer. FlashAttention computes softmax of an unknown-length sequence in a SINGLE streaming pass using the recurrence:

```
state: m (running max), l (running normalizer), o (running output accumulator)
init:  m = -inf, l = 0, o = 0

for each new block of (scores, values):
    m_block = max(scores in block)
    m_new = max(m, m_block)
    l = exp(m - m_new) * l + sum(exp(scores - m_new))
    o = exp(m - m_new) * o + sum(exp(scores - m_new) * values)
    m = m_new

final:  output = o / l
```

Implement this for one query against an arbitrary-length key/value sequence:

```c
void attention_online_softmax(const float *q,     // [d]
                              const float *K,     // [N, d]
                              const float *V,     // [N, d]
                              int N, int d,
                              int block_size,
                              float *o);          // [d]
```

---

## Exercise 41.2 — Correctness Check vs. Naive

**Difficulty**: ★★

Write a naive `attention_baseline(q, K, V, N, d, o)` that materializes the full score array, then compare the two implementations. For the same $q, K, V$, the outputs must agree to within `1e-5` per element for any `block_size` from 1 to `N`. A mismatch larger than `1e-5` almost always means the `exp(m - m_new)` rescaling is applied in the wrong order.

---

## Exercise 41.3 — Block-Size Sweep

**Difficulty**: ★★

Time the `attention_online_softmax` function at `block_size` ∈ {1, 8, 32, 128, 512} for `N = 4096, d = 64`. On a typical CPU you should see a sweet spot around `block_size = 32–64` where the inner-loop working set fits in L1 cache. Record the timings and explain the curve.

---

## Exercise 41.4 — Backward Pass — Bonus

**Difficulty**: ★★★★

The backward pass of FlashAttention is the real trick — it avoids materializing attention scores OR their gradients, at the cost of a second forward pass during backward. Reproduce the forward from your exercise 41.1 while re-computing softmax weights on the fly during the backward. Verify against autograd on a toy problem ($N = 8, d = 4$).
