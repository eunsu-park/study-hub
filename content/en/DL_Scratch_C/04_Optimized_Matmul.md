# 04. Optimized Matrix Multiplication

**Previous**: [Tensor Ops and BLAS](./03_Tensor_Ops_BLAS.md) | **Next**: [Autograd Engine](./05_Autograd_Engine.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why naive matmul is cache-inefficient and quantify the miss penalty
2. Implement a loop-tiled SGEMM that keeps data in L1/L2 cache
3. Add register-level micro-kernel blocking for compute density
4. Use AVX2 intrinsics to process 8 floats per instruction
5. Profile with `perf stat` and measure achieved GFLOP/s

---

## 1. Why Naive Matmul is Slow

Recall the naive triple-loop:

```c
for (i) for (j) for (k)
    C[i][j] += A[i][k] * B[k][j];
```

For `N=1024`, `B[k][j]` access pattern:
- `k` advances → stride of `N=1024` elements = **4096 bytes**
- Each access to `B[k][j]` is in a different cache line → **cache miss every iteration**
- L1 cache can hold `32 KB / 4 = 8192` floats, but we trash it with column-strided access

**Result**: ~99% of time spent waiting for memory, not computing.

---

## 2. Loop Tiling (Blocking)

**Tiling** processes the matrix in small blocks (tiles) that fit in L1/L2 cache.

```c
// Tiled matmul with tile size TILE
#define TILE 64   // 64 floats = 256 bytes = 4 cache lines — fits in L1

void matmul_tiled(float *C, const float *A, const float *B,
                  size_t M, size_t K, size_t N) {
    memset(C, 0, M * N * sizeof(float));

    for (size_t i0 = 0; i0 < M; i0 += TILE)
    for (size_t j0 = 0; j0 < N; j0 += TILE)
    for (size_t k0 = 0; k0 < K; k0 += TILE) {
        // Compute tile boundary with clamping
        size_t imax = i0 + TILE < M ? i0 + TILE : M;
        size_t jmax = j0 + TILE < N ? j0 + TILE : N;
        size_t kmax = k0 + TILE < K ? k0 + TILE : K;

        // Process the tile — stays in cache
        for (size_t i = i0; i < imax; i++)
        for (size_t k = k0; k < kmax; k++) {
            float a_ik = A[i * K + k];      // Load once, reuse across j
            for (size_t j = j0; j < jmax; j++)
                C[i * N + j] += a_ik * B[k * N + j];
        }
    }
}
```

**Why this works**:
- Tile `A[i0..imax, k0..kmax]` fits in L1: `TILE*TILE*4 = 16KB` (for TILE=64)
- Tile `B[k0..kmax, j0..jmax]` fits in L1: same
- Inner loop accesses `B` contiguously → cache-friendly

**Performance improvement**: Typically **4–10× faster** than naive for large N.

---

## 3. Register Blocking (Micro-Kernel)

After tiling, we can further improve by computing a small output block using **registers only**.

A `4×4` micro-kernel accumulates `4×4 = 16` C values entirely in CPU registers:

```c
// 4×4 register micro-kernel for inner tile
// Computes C[i:i+4, j:j+4] += A[i:i+4, k] * B[k, j:j+4]
static inline void micro_kernel_4x4(
    float *C, const float *A, const float *B,
    size_t K, size_t N, size_t i, size_t j)
{
    // Load 4×4 output accumulator into registers
    float c00=0, c01=0, c02=0, c03=0;
    float c10=0, c11=0, c12=0, c13=0;
    float c20=0, c21=0, c22=0, c23=0;
    float c30=0, c31=0, c32=0, c33=0;

    for (size_t k = 0; k < K; k++) {
        float a0 = A[(i+0)*K + k],  a1 = A[(i+1)*K + k];
        float a2 = A[(i+2)*K + k],  a3 = A[(i+3)*K + k];
        float b0 = B[k*N + (j+0)],  b1 = B[k*N + (j+1)];
        float b2 = B[k*N + (j+2)],  b3 = B[k*N + (j+3)];

        c00 += a0*b0; c01 += a0*b1; c02 += a0*b2; c03 += a0*b3;
        c10 += a1*b0; c11 += a1*b1; c12 += a1*b2; c13 += a1*b3;
        c20 += a2*b0; c21 += a2*b1; c22 += a2*b2; c23 += a2*b3;
        c30 += a3*b0; c31 += a3*b1; c32 += a3*b2; c33 += a3*b3;
    }

    // Write back to C
    C[(i+0)*N+j+0]+=c00; C[(i+0)*N+j+1]+=c01;
    C[(i+0)*N+j+2]+=c02; C[(i+0)*N+j+3]+=c03;
    C[(i+1)*N+j+0]+=c10; C[(i+1)*N+j+1]+=c11;
    C[(i+1)*N+j+2]+=c12; C[(i+1)*N+j+3]+=c13;
    C[(i+2)*N+j+0]+=c20; C[(i+2)*N+j+1]+=c21;
    C[(i+2)*N+j+2]+=c22; C[(i+2)*N+j+3]+=c23;
    C[(i+3)*N+j+0]+=c30; C[(i+3)*N+j+1]+=c31;
    C[(i+3)*N+j+2]+=c32; C[(i+3)*N+j+3]+=c33;
}
```

The compiler will keep `c00..c33` in registers, eliminating all intermediate memory writes.

---

## 4. AVX2 Intrinsics

**AVX2** (Advanced Vector Extensions 2) processes **8 floats in a single instruction** using 256-bit registers (`__m256`).

```
Scalar FMA:  1 multiply-add → 2 FLOP  per cycle
AVX2   FMA:  8 multiply-adds → 16 FLOP per cycle   (8× improvement)
```

### Key Intrinsics

```c
#include <immintrin.h>

__m256 a = _mm256_loadu_ps(ptr);       // Load 8 unaligned floats
__m256 b = _mm256_set1_ps(scalar);    // Broadcast scalar to 8 floats
__m256 c = _mm256_fmadd_ps(a, b, c);  // c = a*b + c (FMA)
_mm256_storeu_ps(ptr, c);             // Store 8 floats
```

### AVX2 Matmul Inner Loop

```c
#include <immintrin.h>

// Compute one row of C using AVX2: C[i, 0..N] += A[i,k] * B[k, 0..N]
void matmul_row_avx2(float *C_row, const float *B_row,
                     float a_scalar, size_t N) {
    __m256 a_vec = _mm256_set1_ps(a_scalar);  // Broadcast A[i,k]

    size_t j = 0;
    for (; j + 8 <= N; j += 8) {
        __m256 b_vec = _mm256_loadu_ps(B_row + j);
        __m256 c_vec = _mm256_loadu_ps(C_row + j);
        c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);  // c += a * b
        _mm256_storeu_ps(C_row + j, c_vec);
    }
    // Handle remainder (N not divisible by 8)
    for (; j < N; j++)
        C_row[j] += a_scalar * B_row[j];
}

void matmul_avx2(float *C, const float *A, const float *B,
                 size_t M, size_t K, size_t N) {
    memset(C, 0, M * N * sizeof(float));
    for (size_t i = 0; i < M; i++)
        for (size_t k = 0; k < K; k++)
            matmul_row_avx2(C + i*N, B + k*N, A[i*K + k], N);
}
```

**Compile with**: `gcc -std=c11 -O3 -march=native -mavx2 -mfma`

> **Check AVX2 support**: `grep avx2 /proc/cpuinfo` (Linux) or `sysctl -a | grep avx2` (macOS)

---

## 5. Combining Tiling + AVX2

The best single-threaded matmul combines both:

```c
#define TILE_M 64
#define TILE_N 64
#define TILE_K 256

void matmul_tiled_avx2(float *C, const float *A, const float *B,
                        size_t M, size_t K, size_t N) {
    memset(C, 0, M * N * sizeof(float));

    for (size_t i0 = 0; i0 < M; i0 += TILE_M)
    for (size_t j0 = 0; j0 < N; j0 += TILE_N)
    for (size_t k0 = 0; k0 < K; k0 += TILE_K) {
        size_t imax = i0 + TILE_M < M ? i0 + TILE_M : M;
        size_t jmax = j0 + TILE_N < N ? j0 + TILE_N : N;
        size_t kmax = k0 + TILE_K < K ? k0 + TILE_K : K;
        size_t klen = kmax - k0;

        for (size_t i = i0; i < imax; i++) {
            const float *A_row = A + i * K + k0;
            float       *C_row = C + i * N + j0;
            size_t       jlen  = jmax - j0;

            for (size_t k = 0; k < klen; k++) {
                float a_val = A_row[k];
                const float *B_row = B + (k0 + k) * N + j0;
                matmul_row_avx2(C_row, B_row, a_val, jlen);
            }
        }
    }
}
```

---

## 6. Performance Comparison

Build and benchmark all three variants:

```makefile
# Makefile
CC     = gcc
CFLAGS = -std=c11 -O3 -march=native -mavx2 -mfma -Wall
LIBS   = -lopenblas -lm

benchmark: benchmark_matmul.c ops.c
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)
```

**Expected results (Apple M2, AVX equivalent via ARM NEON)**:

| N | Naive | Tiled | Tiled+AVX2 | OpenBLAS |
|---|-------|-------|------------|----------|
| 256 | 20 ms | 3 ms | 0.8 ms | 0.2 ms |
| 512 | 160 ms | 18 ms | 5 ms | 1.3 ms |
| 1024 | ~1400 ms | 130 ms | 35 ms | 11 ms |

**Roofline analysis for N=1024**:
- Peak FLOPs: `2 * 1024^3 ≈ 2.1 GFLOP`
- Naive: `1400 ms → 1.5 GFLOP/s` (2% of peak)
- Tiled+AVX2: `35 ms → 60 GFLOP/s` (80% of peak single-core)
- OpenBLAS: `11 ms → 190 GFLOP/s` (multi-threaded)

---

## 7. Profiling with `perf`

On Linux, profile cache behavior:

```bash
# Install perf (Ubuntu)
sudo apt-get install linux-tools-common linux-tools-$(uname -r)

# Profile cache misses for naive vs tiled
perf stat -e cache-misses,cache-references,instructions,cycles ./benchmark 1024 naive
perf stat -e cache-misses,cache-references,instructions,cycles ./benchmark 1024 tiled
```

**Expected output (naive, N=1024)**:
```
  12,589,123,456      cache-misses              # 98.2% of all cache refs
  12,822,450,321      cache-references
   8,590,000,000      instructions
```

**Expected output (tiled, N=1024)**:
```
     45,123,456      cache-misses              # 3.5% of all cache refs
   1,289,000,000      cache-references
   8,620,000,000      instructions
```

The tiled version has **280× fewer cache misses** — that is where the speedup comes from.

---

## Key Takeaways

- Naive matmul's `B[k][j]` column access causes L1 cache thrashing — the dominant bottleneck
- **Loop tiling** keeps active data in L1/L2 cache; tile size should satisfy `3 * TILE^2 * 4 ≤ L1_size`
- **Register blocking** (4×4 micro-kernel) maximizes compute per memory load
- **AVX2 FMA** computes 8 multiply-adds per cycle — 8× scalar throughput
- Combining tiling + AVX2 reaches ~80% of peak single-core FLOP/s

---

**Next**: [05. Autograd Engine](./05_Autograd_Engine.md) — Build a computational graph and automatic differentiation engine in C using function pointers and topological sort.
