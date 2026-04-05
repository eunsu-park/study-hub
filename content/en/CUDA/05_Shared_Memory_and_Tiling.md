# 05. Shared Memory and Tiling

**Previous**: [CUDA Memory Model](./04_CUDA_Memory_Model.md) | **Next**: [Warp Execution and Divergence](./06_Warp_Execution_and_Divergence.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why global memory reuse requires shared memory staging
2. Implement tiled matrix multiplication with correct `__syncthreads()` placement
3. Identify and eliminate shared memory bank conflicts using padding
4. Measure shared memory utilization with Nsight Compute
5. Profile the performance gap: naive → tiled matmul

---

## 1. The Reuse Problem

Consider matrix multiply C = A × B (N×N matrices):

```
Element C[i][j] = sum_k A[i][k] * B[k][j]
```

In a naive kernel, thread (i, j) loads all N elements of row i from A and column j from B — **N loads from global memory per output element**.

```
For N=1024: each thread reads 1024 + 1024 = 2048 floats from global memory
Total reads: N³ = 1 billion reads for a 1024×1024 matmul
Global memory bandwidth needed: 4 GB × ... far exceeds what's achievable
```

The key observation: adjacent threads in the same block share row A[i][*]. If 16 threads compute 16 elements of the same row of C, they each need the same row of A — **that row could be loaded once into shared memory and reused 16 times**.

---

## 2. The Tiling Strategy

Divide A, B, and C into `TILE×TILE` submatrices. Each block computes one tile of C:

```
┌───────────────────────────────────────────────┐
│  Matrix A (N×N)    Matrix B (N×N)             │
│                                               │
│  ┌──┬──┬──┬──┐    ┌──┬──┬──┬──┐             │
│  │A₀│A₁│A₂│A₃│    │B₀│  │  │  │             │
│  ├──┼──┼──┼──┤    ├──┼──┼──┼──┤             │
│  │  │  │  │  │    │B₁│  │  │  │             │
│  └──┴──┴──┴──┘    ├──┼──┼──┼──┤             │
│                    │B₂│  │  │  │             │
│  C[0][0] += A₀×B₀ + A₁×B₁ + A₂×B₂ + ...   │
└───────────────────────────────────────────────┘

Each TILE×TILE block:
  1. Load one tile of A into shared memory  (TILE² reads from global)
  2. Load one tile of B into shared memory  (TILE² reads from global)
  3. Compute partial dot products            (TILE² mults + adds, all in registers)
  4. Repeat for all tiles along the K dimension
```

Global memory reads per block: `2 × TILE² × (N/TILE)` = `2 × N × TILE`

Versus naive: `N²` reads (each thread reads N elements, TILE² threads)

**Arithmetic intensity improvement**: `N / (2 × TILE)` × better

---

## 3. Tiled Matmul Implementation

```c
// matmul_tiled.cu
#include <cuda_runtime.h>
#define TILE 16

__global__ void matmul_tiled(const float *A, const float *B, float *C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;  // accumulator in register

    // Iterate over tiles along the K dimension
    for (int t = 0; t < N / TILE; t++) {

        // Phase 1: Load tile of A and B into shared memory
        As[threadIdx.y][threadIdx.x] = A[row * N + (t * TILE + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];

        // Phase 2: Synchronize — all threads must finish loading before compute
        __syncthreads();

        // Phase 3: Compute partial dot product for this tile
        for (int k = 0; k < TILE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // Phase 4: Synchronize — must finish computing before loading next tile
        __syncthreads();
    }

    // Write final result (only if within bounds)
    if (row < N && col < N)
        C[row * N + col] = sum;
}
```

### Critical: Two `__syncthreads()` barriers

```
Without barrier 1 (after loads):
  Thread 0 might start computing before Thread 15 has loaded its element.
  → Wrong results.

Without barrier 2 (before next tile load):
  Thread 0 might overwrite As[0][0] with the next tile
  while Thread 15 is still reading As[0][0] from the current tile.
  → Wrong results (data race).
```

`__syncthreads()` ensures all threads in the block reach the barrier before any thread proceeds. It has no effect across blocks.

---

## 4. Handling Non-Square / Non-Tile-Multiple Sizes

The tiled kernel above assumes N is a multiple of TILE. The robust version:

```c
__global__ void matmul_tiled_safe(const float *A, const float *B, float *C,
                                   int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        // Guard out-of-bounds loads with 0
        As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ?
                                        A[row * K + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ?
                                        B[bRow * N + col]  : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}
```

---

## 5. Shared Memory Bank Conflicts

Shared memory is organized into **32 banks** (Ampere), each 4 bytes wide. Banks serve one request per cycle. A **bank conflict** occurs when multiple threads in a warp access different addresses in the **same bank** — the accesses serialize.

```
Bank layout (32 banks, 4-byte words):
  Address 0  → Bank 0     Address 32 → Bank 0
  Address 4  → Bank 1     Address 36 → Bank 1
  Address 8  → Bank 2     ...
  ...
  Address 124 → Bank 31   Address 156 → Bank 31

Bank number for address a: (a / 4) % 32
```

### Example: 2-way bank conflict in transpose

Reading `tile[threadIdx.x][threadIdx.y]` in a warp:
- Thread 0 reads `tile[0][0]` → bank 0
- Thread 1 reads `tile[1][0]` → bank 0 (conflict! offset = 16 elements = 64 bytes, bank = 0)
- ...
- Thread 31 reads `tile[31][0]` → bank 0

All 32 threads hit bank 0 → 32-way bank conflict → 32× slower.

### Fix: Padding

```c
__shared__ float tile[TILE][TILE + 1];  // Add 1 column of padding
```

With padding:
- `tile[0][0]` → bank 0, `tile[1][0]` → bank (16+1)%32 = 17, `tile[2][0]` → bank (34)%32 = 2

Each column access now hits a different bank → no conflicts.

```c
// Bank conflict check macro (debug only)
// Run with: ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
```

---

## 6. Performance Comparison

```
Matrix multiply 1024×1024 (FP32) on RTX 3090:

  Naive kernel:         ~3.2 ms  ~  670 GFLOPS  (9.6% of peak FP32)
  Tiled TILE=16:        ~0.54 ms ~ 3,970 GFLOPS  (56% of peak FP32)
  Tiled TILE=32:        ~0.41 ms ~ 5,230 GFLOPS  (75% of peak FP32)
  cuBLAS:               ~0.37 ms ~ 5,800 GFLOPS  (83% of peak FP32)
```

The tiled kernel achieves ~4× speedup with a straightforward shared memory optimization.

---

## 7. Profiling with Nsight Compute

```bash
# Profile shared memory metrics
ncu --metrics \
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
    sm__warps_active.avg.pct_of_peak_sustained_active,\
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second \
    ./matmul_tiled

# Key metrics to check:
# - bank_conflicts = 0 (or very low) ✓
# - warps_active > 50% (good occupancy) ✓
# - global load bytes/s close to peak BW ✓
```

---

## 8. Register Tiling (Preview of L32)

Tiling can go further: each thread computes a **tile of output elements** using registers:

```c
#define RX 4  // each thread computes RX×RY output elements
#define RY 4

__global__ void matmul_register_tiled(...) {
    float C_reg[RY][RX] = {};  // register accumulator — RY×RX output values

    // ... load tiles into shared memory, then:
    for (int k = 0; k < TILE; k++) {
        float a_reg[RY];   // register file — one column of A tile
        float b_reg[RX];   // register file — one row of B tile
        // load a_reg, b_reg from shared memory (fast)
        for (int ry = 0; ry < RY; ry++)
            for (int rx = 0; rx < RX; rx++)
                C_reg[ry][rx] += a_reg[ry] * b_reg[rx];
    }
    // write C_reg back to global memory
}
```

This increases arithmetic intensity further by reusing data in registers (0-cycle). L32 develops this into a complete high-performance GEMM.

---

## Key Takeaways

- **Shared memory** acts as a software-managed cache — load data once, use it many times within a block
- Tiled matmul reduces global memory reads from O(N³) to O(N²√N) — a √N improvement
- **Two `__syncthreads()` per tile iteration** are mandatory: after loading (before compute) and before the next load
- **Bank conflicts** serialize warp accesses — pad shared arrays by 1 element to eliminate column conflicts
- Profile with `ncu` to confirm: zero bank conflicts, high occupancy, near-peak memory throughput

---

**Next**: [06. Warp Execution and Divergence](./06_Warp_Execution_and_Divergence.md) — Understand how branch divergence serializes warps, implement warp-level shuffle reductions, and use `__shfl_sync` for inter-thread communication.
