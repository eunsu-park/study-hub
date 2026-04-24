# Lesson 17 — Stencil Computations (per-lesson exercise)

Prerequisites: L05 (shared memory), L08 (coalescing).

Compile: `nvcc -O3 -arch=sm_80 ex.cu -o ex`

Stencils — where each output element is a function of its neighborhood in the input — show up in PDE solvers, image filters, and convolutions. The naive implementation is bandwidth-bound because each input is read multiple times. Tiling with shared memory plus halo cuts those redundant reads.

---

## Exercise 17.1 — 1D Heat Equation Update

**Difficulty**: ★★

The discretized 1D heat equation update is:

$$u_i^{n+1} = u_i^n + \frac{\alpha \Delta t}{\Delta x^2}\left(u_{i-1}^n - 2u_i^n + u_{i+1}^n\right)$$

Implement `__global__ void heat_step(const float *u, float *u_next, int N, float c)` where `c = α·Δt/Δx²`.

```cuda
__global__ void heat_step(const float *u, float *u_next, int N, float c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 1 && i < N - 1) {
        u_next[i] = u[i] + c * (u[i-1] - 2.0f * u[i] + u[i+1]);
    }
}
```

Boundary conditions: leave `u_next[0]` and `u_next[N-1]` to be set by the host (or use Dirichlet zeros). Iterate this kernel 1000 times alternating buffers and verify that an initial step function diffuses into a smooth profile.

---

## Exercise 17.2 — 2D Stencil with Shared-Memory Tiling

**Difficulty**: ★★★

The 5-point 2D stencil for the heat equation is:

$$u_{i,j}^{n+1} = u_{i,j}^n + c\left(u_{i-1,j} + u_{i+1,j} + u_{i,j-1} + u_{i,j+1} - 4 u_{i,j}\right)$$

Naive: each output reads 5 inputs from global memory, with 4 of them shared with neighboring outputs.

Tiled: each block loads a `(TILE_Y + 2) × (TILE_X + 2)` patch (interior + halo) into shared memory. Each thread then computes one output by reading from shared memory.

```cuda
#define TILE 16

__global__ void heat_2d_tiled(const float *u, float *u_next, int W, int H, float c) {
    __shared__ float tile[TILE + 2][TILE + 2];

    int gx = blockIdx.x * TILE + threadIdx.x;
    int gy = blockIdx.y * TILE + threadIdx.y;
    int lx = threadIdx.x + 1;
    int ly = threadIdx.y + 1;

    // Load interior
    if (gx < W && gy < H) tile[ly][lx] = u[gy * W + gx];

    // Load halo (one element per thread for the four edges; corners may need extra logic)
    if (threadIdx.x == 0     && gx > 0     ) tile[ly][0]        = u[gy * W + gx - 1];
    if (threadIdx.x == TILE-1 && gx < W - 1) tile[ly][TILE + 1] = u[gy * W + gx + 1];
    if (threadIdx.y == 0     && gy > 0     ) tile[0][lx]        = u[(gy - 1) * W + gx];
    if (threadIdx.y == TILE-1 && gy < H - 1) tile[TILE + 1][lx] = u[(gy + 1) * W + gx];

    __syncthreads();

    if (gx > 0 && gx < W - 1 && gy > 0 && gy < H - 1) {
        float center = tile[ly][lx];
        u_next[gy * W + gx] = center + c * (tile[ly][lx-1] + tile[ly][lx+1] +
                                            tile[ly-1][lx] + tile[ly+1][lx] - 4.0f * center);
    }
}
```

Time vs. naive — expect 2–4× speedup at `TILE = 16`. The exact ratio depends on cache behavior of the naive version.

---

## Exercise 17.3 — Wider Stencils — Bonus

**Difficulty**: ★★★

A 5×5 stencil (25-point) is common in PDE solvers and image filters. Generalize 17.2 to take a `RADIUS` parameter and load a `(TILE + 2 RADIUS) × (TILE + 2 RADIUS)` patch. Consider register-tiling the inner reductions to avoid issuing 25 shared-memory loads per output.
