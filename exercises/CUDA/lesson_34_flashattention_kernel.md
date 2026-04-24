# Lesson 34 — FlashAttention Kernel (per-lesson exercise)

Prerequisites: L05 (shared memory), L14 (reduction), L33 (softmax), familiarity with attention math.

Compile: `nvcc -O3 -arch=sm_80 ex.cu -o ex`

FlashAttention computes attention $\text{softmax}(QK^T / \sqrt{d}) V$ without ever materializing the $[N, N]$ score matrix. On GPU this is an enormous memory-bandwidth win because for moderate-to-long sequences the score matrix is far larger than the model fits.

This exercise walks through a CUDA implementation of the **forward** pass. (The backward is canonically harder and is a CUDA-domain doctoral-thesis exercise; a CPU version was covered in DL_Scratch_C lesson 41.)

---

## Exercise 34.1 — Block-Wise Online Softmax (Single Query)

**Difficulty**: ★★★

### Problem

Each thread block computes the attention output for a single query vector. The kernel iterates over keys/values in tiles, maintaining running statistics of the softmax (max and sum) and the partial output.

```
for each tile (K_tile, V_tile) of size [Br, d]:
    1. compute scores = Q · K_tile^T / sqrt(d)              // [Br]
    2. m_new = max(m_old, max(scores))
    3. l = exp(m_old - m_new) * l + sum(exp(scores - m_new))
    4. o = exp(m_old - m_new) * o + sum(exp(scores - m_new) * V_tile)
    5. m = m_new

final: o = o / l
```

### Starter

```cuda
#include <cstdio>
#include <cuda_runtime.h>

template <int Br, int d>
__global__ void flash_attn_one_query(const float *Q,    // [d]
                                     const float *K,    // [N, d]
                                     const float *V,    // [N, d]
                                     int N,
                                     float scale,       // 1/sqrt(d)
                                     float *O) {        // [d]
    extern __shared__ float smem[];
    float *K_tile = smem;                  // [Br, d]
    float *V_tile = smem + Br * d;         // [Br, d]
    float *scores = smem + 2 * Br * d;     // [Br]

    /* Running statistics — single-thread for simplicity (you should reduce later) */
    if (threadIdx.x == 0) {
        for (int j = 0; j < d; j++) O[j] = 0;
    }
    __syncthreads();

    /* Iterate tiles ... TODO */
    /* For each tile of Br rows from K, V:
         load into shared memory cooperatively
         compute scores = Q · K_tile^T * scale
         online-softmax update of m, l, O
    */
}
```

The key correctness check: for any `Br ≥ 1`, the result must equal the standard attention output to within `1e-4`.

---

## Exercise 34.2 — Multi-Query Per Block

**Difficulty**: ★★★★

A real FlashAttention block handles `Bq` queries at once (not one). Each thread within a block owns one query; the block tiles over keys/values. This amortizes the K/V loads across queries and is what makes FlashAttention fast.

Generalize 34.1 to `flash_attn_multi_query<Bq, Br, d>`. Each thread maintains its own `(m, l, O)` state. After the K/V loop, every thread divides its `O` by its `l` and writes the output.

This is a **rite-of-passage** kernel: get it right and you understand modern attention kernels. Most published FlashAttention v2/v3 kernels are this template specialized for specific `(Bq, Br, d)` triples and tuned for one architecture.

---

## Exercise 34.3 — Comparing Memory Traffic

**Difficulty**: ★★

For a sequence length $N$ and head dimension $d$, count the bytes read from DRAM by:

- **Naive attention**: materializes `[N, N]` scores → $O(N^2)$ writes + reads of scores + $O(N \cdot d)$ for K, V.
- **FlashAttention**: streams K, V through SRAM; never writes scores → $O(N \cdot d)$ for K, V + $O(d)$ for the output per query.

For $N = 8192, d = 64$, the naive version reads/writes $\sim 256$ MiB of scores. FlashAttention reads $\sim 4$ MiB total. That 64× ratio is the source of the speedup — DRAM bandwidth, not compute.

---

## Exercise 34.4 — Causal Mask — Bonus

**Difficulty**: ★★★

Add a causal mask (each query only sees keys at or before its position). The trick: the K-iteration loop can EARLY-EXIT when the tile starts past the query position. This makes the work proportional to the number of valid (q, k) pairs ($N^2/2$) instead of $N^2$.
