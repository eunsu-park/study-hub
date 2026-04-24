# Lesson 6 — Warp Execution and Divergence (per-lesson exercise)

Prerequisites: L03 (thread indexing).

Compile: `nvcc -O3 -arch=sm_80 ex.cu -o ex`

A warp is 32 threads executing in lockstep. When threads in a warp take different branches of a conditional, the warp executes BOTH branches with masking — only the threads that "wanted" each branch produce useful work, while the others stall. This is **warp divergence** and it can halve, quarter, or worse your throughput.

This exercise makes the cost concrete and walks through the standard mitigations.

---

## Exercise 6.1 — Measuring Divergence Overhead

**Difficulty**: ★★

### Problem

Implement two kernels that compute the same thing — `output[i] = (i % 2 == 0) ? f(i) : g(i)` — but with different mappings.

- `kernel_divergent`: `i = blockIdx.x * blockDim.x + threadIdx.x`. Adjacent threads alternate branches → every warp diverges.
- `kernel_uniform`: split work so that warp $w$ runs entirely $f$ or entirely $g$ — i.e., the branch decision is uniform across each warp.

### Starter

```cuda
#include <cstdio>
#include <cuda_runtime.h>

__device__ float f(int i) { float x = (float)i; for (int k=0;k<8;k++) x = x*1.001f + 0.5f; return x; }
__device__ float g(int i) { float x = (float)i; for (int k=0;k<8;k++) x = x*0.999f - 0.5f; return x; }

__global__ void kernel_divergent(float *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = (i % 2 == 0) ? f(i) : g(i);
}

__global__ void kernel_uniform(float *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    /* Uniform-by-warp: branch on (i / 32) % 2 — every thread in a warp picks the same path. */
    int warp_id = i / 32;
    out[i] = (warp_id % 2 == 0) ? f(i) : g(i);
}

/* Host: time both kernels; expect kernel_uniform to be ~1.8x faster */
```

Time both with `cudaEventElapsedTime`. The uniform version should be roughly 1.5–2× faster on most GPUs because divergent warps cannot overlap the two branch executions.

---

## Exercise 6.2 — Avoiding Divergence with Predication

**Difficulty**: ★★

For very short branches, the compiler will emit predicated instructions instead of taking branches. Predicated instructions still issue, but their result is conditionally written. Rewrite:

```cuda
if (x[i] > 0) y[i] = sqrt(x[i]); else y[i] = 0;
```

as the equivalent branch-free form:

```cuda
y[i] = (x[i] > 0) ? sqrtf(x[i]) : 0.0f;
```

For these short alternates, both compile to the same predicated instructions on modern GPUs and execute with no measurable divergence cost. The "ternary" form makes the predication intent explicit.

---

## Exercise 6.3 — Sort-Based Divergence Removal — Bonus

**Difficulty**: ★★★★

When the per-thread branch is data-dependent and not aligned to warps, you can sort the input so that threads with the same branch end up contiguous. Combined with a permutation array that records the original order, you can reorder threads at runtime.

Implement: take an array of `(int key, float value)` pairs; sort by `key`; then run a divergence-free kernel that operates on the sorted array. Compare against the unsorted divergent version.

This pattern shows up in graph processing (sort by neighbor count) and ray tracing (sort by ray direction). The sort itself is expensive, so it only pays off when the kernel runs many times on the same partition.
