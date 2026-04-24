# Lesson 16 — Parallel Sort (Radix) (per-lesson exercise)

Prerequisites: L15 (parallel scan), L08 (memory coalescing).

Compile: `nvcc -O3 -arch=sm_80 ex.cu -o ex`

GPU sorting algorithms differ from CPU ones. Comparison sorts (mergesort, quicksort) parallelize awkwardly because they branch on data. Radix sort does not branch — it processes one bit at a time uniformly across all keys — and it is the de-facto choice on GPU.

---

## Exercise 16.1 — Single-Pass Radix Step

**Difficulty**: ★★★

### Problem

A radix-sort pass for one bit of `uint32_t` keys works as follows:

1. For each key, extract bit $b$.
2. Compute the prefix sum (scan) of "bit is 0" flags. The sum gives each "0-bit" key its new index.
3. Compute the count of "0-bit" keys in the array (the last scan value + the last "0-bit" flag).
4. The new index of a "1-bit" key is `scan_of_1_flags[i] + count_of_0_keys`.
5. Scatter each key to its new index.

Implement one pass:

```cuda
__global__ void radix_pass(const uint32_t *in, uint32_t *out,
                           const int *prefix_zero, int total_zero, int N, int bit) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    uint32_t key = in[i];
    int b = (key >> bit) & 1;
    int new_idx = (b == 0) ? prefix_zero[i] : (i - prefix_zero[i] + total_zero);
    out[new_idx] = key;
}
```

You will need a separate scan kernel (or use Thrust's `exclusive_scan`) to produce `prefix_zero`.

---

## Exercise 16.2 — Full 32-Bit Sort

**Difficulty**: ★★★

Loop the single-bit pass 32 times (one per bit, low-to-high). After each pass, the array is sorted by the considered bits. After all 32 passes, the array is fully sorted.

Time it on $N = 16$ million keys. Compare against `thrust::sort` (CUDA L28). Your hand-rolled version will likely be 1.5–3× slower — that gap is what cuRAND's optimized radix sort buys you.

---

## Exercise 16.3 — Multi-Bit Radix — Bonus

**Difficulty**: ★★★★

A 2-bit radix processes 4 buckets at a time (00, 01, 10, 11). It cuts the number of passes in half (32 → 16) at the cost of a more complex per-pass kernel. Real-world libraries like CUB use 4–8 bits per pass.

Implement a 2-bit radix and time vs. your single-bit version. Speedup is typically 1.4–1.8× — close to half the passes, less than half because each pass does more work.

---

## Exercise 16.4 — Sorting Pairs — Bonus

**Difficulty**: ★★

Sort `(key, value)` pairs where `key` is `uint32_t` and `value` is, say, `int` indices into another array. The pattern: scatter both `key` and `value` together based on the index permutation each radix pass computes.

Used everywhere: sorting points by Morton code in BVH construction, sorting tokens by their attention scores in beam search, sorting events by timestamp in physics simulations.
