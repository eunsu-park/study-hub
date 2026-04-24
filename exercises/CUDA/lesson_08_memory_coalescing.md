# Lesson 8 — Memory Coalescing (per-lesson exercise)

Prerequisites: L03 (thread indexing), L04 (memory model).

Compile: `nvcc -O3 -arch=sm_80 ex.cu -o ex`

When 32 threads of a warp access global memory, the hardware coalesces their requests into as few 128-byte transactions as possible. The layout of your data — and the pattern your threads use to index it — determines whether you hit 1 transaction per warp (best) or 32 (worst).

This exercise makes the performance difference tangible.

---

## Exercise 8.1 — Coalesced vs. Strided Read

**Difficulty**: ★★

### Problem

Implement two kernels that both sum `N` floats but access memory differently:

- `reduce_coalesced`: thread $t$ reads `input[blockIdx.x * blockDim.x + t]`. Adjacent threads → adjacent addresses → 1 transaction per warp.
- `reduce_strided`: thread $t$ reads `input[t * blockDim.x + blockIdx.x]`. Adjacent threads → addresses spaced `blockDim.x` apart → up to 32 transactions per warp.

Compute the partial sum in each case and measure execution time for `N = 16 * 1024 * 1024` floats.

### Starter

```cuda
#include <cstdio>
#include <cuda_runtime.h>

__global__ void reduce_coalesced(const float *input, float *partial, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) atomicAdd(partial, input[idx]);
}

__global__ void reduce_strided(const float *input, float *partial, int N) {
    int idx = threadIdx.x * gridDim.x + blockIdx.x;
    if (idx < N) atomicAdd(partial, input[idx]);
}

/* Measure both with cudaEvent timing (omitted in starter; add the harness).
   Typical result on A100: the strided version is 10–30× slower. */
```

---

## Exercise 8.2 — Struct of Arrays vs. Array of Structs

**Difficulty**: ★★★

### Problem

Consider particle data stored as:

```c
// Array of Structs (AoS)
struct Particle { float x, y, z, vx, vy, vz; };
struct Particle particles[N];
```

vs.

```c
// Struct of Arrays (SoA)
struct Particles {
    float *x, *y, *z;
    float *vx, *vy, *vz;
};
```

Write a kernel that updates positions (`x += vx * dt`) using each layout. Measure the performance.

Expected: SoA is 2–5× faster because the `x` reads and `vx` reads are each coalesced across threads, whereas in AoS every thread reads 6 non-contiguous floats from a 24-byte struct.

---

## Exercise 8.3 — Padding for Alignment — Bonus

**Difficulty**: ★★★

Allocate an array with `cudaMallocPitch` (which adds padding so each row starts on a 128-byte boundary) vs. plain `cudaMalloc`. Compare the performance of a 2D stencil kernel on both. The aligned version should show no degradation even when the logical row size is not a multiple of 32 floats.
