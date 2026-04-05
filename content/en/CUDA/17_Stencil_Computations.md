# 17. Stencil Computations

**Previous**: [Parallel Sort](./16_Parallel_Sort.md) | **Next**: [Histogram and Binning](./18_Histogram_and_Binning.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Write 1D, 2D, and 3D stencil kernels that read neighbor values
2. Use shared memory tiling with halo cells to eliminate redundant global memory loads
3. Implement periodic and Dirichlet boundary conditions in a stencil kernel
4. Build a time-stepping loop for explicit finite difference simulation
5. Identify the arithmetic intensity of a stencil and predict its roofline performance

---

## 1. What Is a Stencil?

A **stencil computation** updates each point in a grid using a fixed pattern of neighboring values. Each thread computes one output grid point by reading from a fixed neighborhood of the input grid.

```
1D 3-point stencil (r=1):    out[i] = f(in[i-1], in[i], in[i+1])
2D 5-point stencil (4-neighbor):
    out[i][j] = f(in[i-1][j], in[i][j-1], in[i][j], in[i][j+1], in[i+1][j])
2D 9-point stencil (8-neighbor):
    includes diagonals in addition to above
```

Stencils are the core computation in:
- Finite difference PDE solvers (heat, wave, diffusion equations)
- Finite element methods
- Image convolution (Gaussian blur, Sobel, Laplacian)
- Lattice Boltzmann simulations

---

## 2. Naive 2D Stencil Kernel

The simplest implementation — one thread per output point, reads directly from global memory:

```c
// 2D 5-point Laplacian stencil: out[i][j] = in[i-1][j] + in[i+1][j]
//                                           + in[i][j-1] + in[i][j+1]
//                                           - 4*in[i][j]
__global__ void laplacian_2d_naive(
    const float *in, float *out, int Nx, int Ny)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // row
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // col

    if (i > 0 && i < Ny - 1 && j > 0 && j < Nx - 1) {
        out[i * Nx + j] =
            in[(i - 1) * Nx + j] + in[(i + 1) * Nx + j] +
            in[i * Nx + (j - 1)] + in[i * Nx + (j + 1)] -
            4.0f * in[i * Nx + j];
    }
}
```

**Problem**: each element is loaded 5 times (once per stencil application). For a 32×32 thread block with 5-point stencil, border threads load data that interior neighbors also need — significant redundancy.

---

## 3. Shared Memory Tiling with Halo Cells

Load a tile of data into shared memory, including a border of **halo cells** (radius 1 for a 5-point stencil) to satisfy all reads within the block:

```c
#define TILE_W 32
#define TILE_H 32
#define RADIUS  1   // stencil radius

__global__ void laplacian_2d_tiled(
    const float *in, float *out, int Nx, int Ny)
{
    // Shared memory tile includes halo: (TILE_H+2*R) x (TILE_W+2*R)
    __shared__ float s[(TILE_H + 2 * RADIUS)][(TILE_W + 2 * RADIUS)];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // Global indices (interior threads)
    int i = blockIdx.y * TILE_H + ty;
    int j = blockIdx.x * TILE_W + tx;

    // Shared memory index includes halo offset
    int si = ty + RADIUS;
    int sj = tx + RADIUS;

    // Load interior of tile
    s[si][sj] = (i < Ny && j < Nx) ? in[i * Nx + j] : 0.0f;

    // Load halo rows (top, bottom)
    if (ty < RADIUS) {
        int above = (i - RADIUS >= 0) ? (i - RADIUS) : 0;
        int below = (i + TILE_H < Ny) ? (i + TILE_H) : Ny - 1;
        s[ty][sj]               = in[above * Nx + j];          // top halo
        s[ty + TILE_H + RADIUS][sj] = in[below * Nx + j];      // bottom halo
    }
    // Load halo columns (left, right)
    if (tx < RADIUS) {
        int left  = (j - RADIUS >= 0) ? (j - RADIUS) : 0;
        int right = (j + TILE_W < Nx) ? (j + TILE_W) : Nx - 1;
        s[si][tx]               = in[i * Nx + left];           // left halo
        s[si][tx + TILE_W + RADIUS] = in[i * Nx + right];      // right halo
    }
    __syncthreads();

    // Apply stencil (skip boundary threads)
    if (i > 0 && i < Ny - 1 && j > 0 && j < Nx - 1) {
        out[i * Nx + j] =
            s[si - 1][sj] + s[si + 1][sj] +
            s[si][sj - 1] + s[si][sj + 1] -
            4.0f * s[si][sj];
    }
}
```

**Memory access reduction**: without tiling, each element is read ~5 times from global memory. With tiling, interior elements are loaded once into shared memory and read 5 times from there. Global loads drop from 5 per point to ~1.06 per point (halo overhead).

---

## 4. Arithmetic Intensity of Stencil Kernels

```
5-point 2D stencil, float (4 bytes):
  Arithmetic:  5 reads + 1 write → 4 adds + 1 multiply = 5 FLOP
  Memory:      5 reads + 1 write = 6 × 4 = 24 bytes (naive, no cache)
  AI:          5 / 24 ≈ 0.21 FLOP/byte

With shared memory tiling (effective):
  Each element loaded once from global, used 5 times →
  Memory: ~1.06 × 4 = 4.24 bytes/output
  AI:     5 / 4.24 ≈ 1.18 FLOP/byte

RTX 3090: memory bandwidth = 936 GB/s, FP32 peak = 35.6 TFLOPS
Ridge point: 35600 / 936 ≈ 38 FLOP/byte → stencil is MEMORY BOUND
Maximum throughput (tiled): 936 GB/s × 1.18 FLOP/byte ≈ 1.1 TFLOPS
```

Stencils are almost always memory-bandwidth bound. Tiling helps but doesn't change the fundamental AI limit.

---

## 5. Time-Stepping Loop for Heat Equation

The 2D heat equation: ∂u/∂t = α (∂²u/∂x² + ∂²u/∂y²)

Explicit finite difference discretization (forward Euler in time, central differences in space):

```
u[i,j,t+1] = u[i,j,t] + α·Δt/Δx² · (u[i-1,j,t] + u[i+1,j,t] +
                                       u[i,j-1,t] + u[i,j+1,t] - 4·u[i,j,t])
```

Let `r = α·Δt/Δx²`. Stability condition (CFL): `r ≤ 0.25` (2D).

```c
// Heat equation stencil kernel
__global__ void heat_step(const float *u_old, float *u_new,
                          float r, int Nx, int Ny) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i > 0 && i < Ny - 1 && j > 0 && j < Nx - 1) {
        float center = u_old[i * Nx + j];
        float laplacian =
            u_old[(i - 1) * Nx + j] + u_old[(i + 1) * Nx + j] +
            u_old[i * Nx + (j - 1)] + u_old[i * Nx + (j + 1)] -
            4.0f * center;
        u_new[i * Nx + j] = center + r * laplacian;
    } else if (i < Ny && j < Nx) {
        // Dirichlet BC: boundary stays fixed
        u_new[i * Nx + j] = u_old[i * Nx + j];
    }
}

// Host: time-stepping loop with ping-pong buffers
void run_heat_simulation(int Nx, int Ny, int steps,
                         float alpha, float dt, float dx) {
    float r = alpha * dt / (dx * dx);
    if (r > 0.25f) {
        fprintf(stderr, "CFL violation: r=%.4f > 0.25, simulation unstable\n", r);
        return;
    }

    size_t bytes = Nx * Ny * sizeof(float);
    float *d_u0, *d_u1;
    cudaMalloc(&d_u0, bytes);
    cudaMalloc(&d_u1, bytes);

    // Initialize (e.g., Gaussian heat source in center)
    init_gaussian<<<dim3((Nx+15)/16,(Ny+15)/16), dim3(16,16)>>>(d_u0, Nx, Ny);

    dim3 block(16, 16);
    dim3 grid((Nx + 15) / 16, (Ny + 15) / 16);

    for (int t = 0; t < steps; t++) {
        heat_step<<<grid, block>>>(d_u0, d_u1, r, Nx, Ny);
        float *tmp = d_u0; d_u0 = d_u1; d_u1 = tmp;  // ping-pong swap
    }

    // d_u0 holds the final state
    cudaFree(d_u0); cudaFree(d_u1);
}
```

---

## 6. Periodic Boundary Conditions

For periodic (wrap-around) boundaries, replace clamped indices with modular indices:

```c
__global__ void laplacian_periodic(const float *in, float *out, int Nx, int Ny) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= Ny || j >= Nx) return;

    // Periodic wrap: (i-1+Ny)%Ny avoids negative modulo
    int im = (i - 1 + Ny) % Ny;
    int ip = (i + 1) % Ny;
    int jm = (j - 1 + Nx) % Nx;
    int jp = (j + 1) % Nx;

    out[i * Nx + j] =
        in[im * Nx + j] + in[ip * Nx + j] +
        in[i * Nx + jm] + in[i * Nx + jp] -
        4.0f * in[i * Nx + j];
}
```

**Performance note**: the modulo operation (`%`) is expensive on GPU (division). For powers of 2, replace `% N` with `& (N-1)`. For arbitrary N, use a conditional add/subtract:

```c
// Faster periodic index without division
__device__ int periodic(int idx, int n) {
    if (idx < 0)  return idx + n;
    if (idx >= n) return idx - n;
    return idx;
}
```

---

## 7. 3D Stencil

Extending to 3D uses a 3D thread block and a 7-point stencil (6 face neighbors):

```c
// 3D 7-point stencil (Laplacian in 3D)
__global__ void laplacian_3d(const float *in, float *out,
                              int Nx, int Ny, int Nz) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix < 1 || ix >= Nx-1 || iy < 1 || iy >= Ny-1 || iz < 1 || iz >= Nz-1)
        return;

    int stride_y = Nx;
    int stride_z = Nx * Ny;
    int idx      = iz * stride_z + iy * stride_y + ix;

    out[idx] =
        in[idx - 1]         + in[idx + 1]         +   // x neighbors
        in[idx - stride_y]  + in[idx + stride_y]  +   // y neighbors
        in[idx - stride_z]  + in[idx + stride_z]  -   // z neighbors
        6.0f * in[idx];

    // Launch: dim3 block(8,8,8), grid((Nx+7)/8, (Ny+7)/8, (Nz+7)/8)
    // Max grid dim in z: 65535 — sufficient for Nz < 524280
}
```

**Shared memory for 3D**: 3D halo loading is complex but critical for high FLOP counts. A (8+2)×(8+2)×(8+2) tile uses 10³ × 4 = 4000 bytes — comfortably within 48 KB shared memory.

---

## Key Takeaways

- Stencil kernels update each grid point from a fixed neighborhood of the **previous** time step (or adjacent neighbors in space)
- **Naive** stencils reload each element 5+ times from global memory; **shared memory tiling** with halo cells reduces this to ~1 load per element
- Stencils are almost always **memory-bandwidth bound** — arithmetic intensity is low (0.2–1.2 FLOP/byte)
- **Ping-pong buffers** (`u_old`, `u_new`) avoid read-write races in time-stepping loops
- **Dirichlet BC**: fix boundary values; **Neumann BC**: fix boundary gradient; **Periodic BC**: wrap-around indices
- The CFL stability condition constrains the maximum time step: `Δt ≤ Δx²/(4α)` in 2D

---

**Next**: [18. Histogram and Binning](./18_Histogram_and_Binning.md) — Implement parallel histogram computation using global atomics, shared memory privatization, and CUB DeviceHistogram.
