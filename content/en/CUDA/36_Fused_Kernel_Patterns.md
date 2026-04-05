# 36. Fused Kernel Patterns

**Previous**: [Quantized Kernels INT8](./35_Quantized_Kernels_INT8.md) | **Next**: [Multi-GPU and NCCL](./37_Multi_GPU_and_NCCL.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain how memory-bound kernels waste GPU bandwidth and why fusion helps
2. Implement a fused bias+ReLU kernel that avoids a separate memory pass
3. Implement a fused residual+LayerNorm kernel in a single pass
4. Write a fused bias+GELU kernel using the tanh approximation
5. Quantify the bandwidth savings of fused vs unfused kernels using the roofline model

---

## 1. Why Kernel Fusion?

Every separate kernel launch requires reading input from and writing output to HBM (GPU main memory). For short elementwise ops, the arithmetic is trivial — the bottleneck is memory:

```
Example: GEMM output → bias add → ReLU (unfused)

Without fusion:                          HBM traffic per element
  1. GEMM writes C[M×N] to HBM           → 1 write
  2. bias_add reads C, writes C+b        → 1 read + 1 write
  3. relu reads C+b, writes max(0,C+b)   → 1 read + 1 write
  Total: 3 reads + 3 writes = 6 × M × N × 4 bytes

With fusion (one epilogue kernel):
  1. bias+ReLU reads C once, writes once → 1 read + 1 write
  Total: 1 read + 1 write = 2 × M × N × 4 bytes  (3× less traffic)

For M=N=4096, FP32: 3× = 192 MB saved per GEMM block
At 900 GB/s HBM bandwidth: saves 0.21 ms per GEMM
```

---

## 2. Fused Bias + ReLU

```c
// Output of GEMM stored in C[M×N]; bias b[N] (one per output column)
// Apply in-place: C[row][col] = max(0, C[row][col] + b[col])

__global__ void bias_relu_inplace(float *C, const float *bias, int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;

    float val = C[row * N + col] + bias[col];
    C[row * N + col] = fmaxf(0.f, val);   // ReLU
}

// Vectorized version: process 4 columns at once
__global__ void bias_relu_vec4(float *C, const float *bias, int M, int N) {
    int col4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int row  =  blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col4 + 3 >= N) return;

    // Load 4 bias values and 4 C values with single 128-bit loads
    float4 b4 = *reinterpret_cast<const float4*>(bias + col4);
    float4 c4 = *reinterpret_cast<const float4*>(C + row * N + col4);

    c4.x = fmaxf(0.f, c4.x + b4.x);
    c4.y = fmaxf(0.f, c4.y + b4.y);
    c4.z = fmaxf(0.f, c4.z + b4.z);
    c4.w = fmaxf(0.f, c4.w + b4.w);

    *reinterpret_cast<float4*>(C + row * N + col4) = c4;
}
```

---

## 3. Fused Bias + GELU

GELU (Gaussian Error Linear Unit) is used in BERT, GPT-2, and many modern transformers. The tanh approximation avoids the expensive `erff()`:

```
Exact GELU:        x * 0.5 * (1 + erf(x / sqrt(2)))
Approximate GELU:  x * 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
                   = x * 0.5 * (1 + tanh(0.7978845 * (x + 0.044715 * x³)))

Error: < 0.001 for all x; used by PyTorch's F.gelu(approximate='tanh')
```

```c
__device__ __forceinline__ float gelu_approx(float x) {
    const float k0 = 0.7978845608f;   // sqrt(2/pi)
    const float k1 = 0.044715f;
    float inner = k0 * (x + k1 * x * x * x);
    return 0.5f * x * (1.f + tanhf(inner));
}

__global__ void bias_gelu_inplace(float *C, const float *bias, int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;

    float val = C[row * N + col] + bias[col];
    C[row * N + col] = gelu_approx(val);
}

// GELU backward (for training):
// d_GELU/dx = 0.5*(1+tanh(inner)) + 0.5*x*(1-tanh²(inner))*d(inner)/dx
__device__ __forceinline__ float gelu_approx_grad(float x) {
    const float k0 = 0.7978845608f, k1 = 0.044715f;
    float inner = k0 * (x + k1 * x * x * x);
    float tanh_v = tanhf(inner);
    float dtanh  = 1.f - tanh_v * tanh_v;
    float dinner = k0 * (1.f + 3.f * k1 * x * x);
    return 0.5f * (1.f + tanh_v) + 0.5f * x * dtanh * dinner;
}
```

---

## 4. Fused Residual + LayerNorm

In transformer blocks, LayerNorm is typically applied after a residual connection:
`y = LayerNorm(x + residual)`

Fusing avoids writing the intermediate `x + residual` to HBM:

```c
__global__ void fused_residual_layernorm(
    const float *x,        // [batch × H] — main stream
    const float *res,      // [batch × H] — residual to add
    const float *gamma,    // [H] weight
    const float *beta,     // [H] bias
    float *out,            // [batch × H] output
    float *mean_out,       // [batch] saved mean (for backward)
    float *rstd_out,       // [batch] saved 1/std
    int H, float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const float *xi  = x   + row * H;
    const float *ri  = res + row * H;
    float       *yi  = out + row * H;

    // Pass 1: compute sum and sum_sq of (x + residual)
    float sum = 0.f, sum_sq = 0.f;
    for (int i = tid; i < H; i += blockDim.x) {
        float v = xi[i] + ri[i];   // residual add (no intermediate write)
        sum    += v;
        sum_sq += v * v;
    }

    // Warp-level reduction
    for (int off = 16; off > 0; off >>= 1) {
        sum    += __shfl_down_sync(0xffffffff, sum,    off);
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, off);
    }

    __shared__ float s_sum[32], s_sq[32];
    int warp = tid / 32, lane = tid % 32;
    if (lane == 0) { s_sum[warp] = sum; s_sq[warp] = sum_sq; }
    __syncthreads();

    if (tid == 0) {
        float ts = 0.f, tsq = 0.f;
        int nw = blockDim.x / 32;
        for (int w = 0; w < nw; w++) { ts += s_sum[w]; tsq += s_sq[w]; }
        float mn  = ts / H;
        float var = tsq / H - mn * mn;
        float rs  = rsqrtf(var + eps);
        s_sum[0] = mn;
        s_sq[0]  = rs;
        if (mean_out) mean_out[row] = mn;
        if (rstd_out) rstd_out[row] = rs;
    }
    __syncthreads();

    float mn = s_sum[0], rstd = s_sq[0];

    // Pass 2: apply LayerNorm transform
    for (int i = tid; i < H; i += blockDim.x) {
        float v = xi[i] + ri[i];   // recompute (cheaper than storing to shared)
        yi[i] = (v - mn) * rstd * gamma[i] + beta[i];
    }
}
```

---

## 5. Fused Dropout + Add

Dropout zeroes random activations and scales remaining by 1/(1-p). Fusing with the residual add saves two memory passes:

```c
// Fused dropout + residual add
// out = dropout(x, p) + residual
__global__ void fused_dropout_add(
    const float *x, const float *residual,
    float *out, uint8_t *mask_out,   // save mask for backward
    int N, float p, curandState *states)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    curandState local = states[i];
    float u    = curand_uniform(&local);
    states[i]  = local;

    float keep  = (u >= p) ? 1.f : 0.f;   // 1 = keep, 0 = drop
    float scale = 1.f / (1.f - p);        // inverted dropout scaling

    float val = x[i] * keep * scale + residual[i];
    out[i]     = val;
    mask_out[i] = (uint8_t)keep;
}
```

---

## 6. CUTLASS Epilogue Fusion (Concept)

CUTLASS (CUDA Templates for Linear Algebra Subroutines) implements GEMM with a customizable epilogue:

```cpp
// CUTLASS epilogue concept: apply per-element transform to GEMM output
// without writing the full matrix to HBM between GEMM and epilogue

// Define epilogue operation (bias + ReLU)
using BiasReluEpilogue = cutlass::epilogue::threadblock::LinearCombinationRelu<
    cutlass::half_t,        // element type
    4,                      // elements per vector store (float4 equivalent)
    float,                  // accumulator type
    float,                  // scale factor type
    cutlass::epilogue::threadblock::ScaleType::NoBetaScaling
>;

// The epilogue runs in the same kernel as the GEMM, reading C from registers
// (never touching HBM between GEMM output and bias+ReLU)
// This is the technique cuBLAS uses internally for fused ops

// CUTLASS epilogue visitor pattern (CUTLASS 3.x):
// Define a computation graph of per-element ops that fuse into GEMM epilogue
//   compute: (alpha * A*B + beta * C) → bias_add → gelu → output
```

---

## 7. Performance Comparison

```
Configuration: M=4096, N=4096, FP32 on A100

Operation sequence: GEMM → bias_add → ReLU
Approach              Time    HBM reads    HBM writes
------------------------------------------------------
Separate kernels      3.1ms   192 MB       192 MB     (3 extra passes over MN)
Fused epilogue        2.2ms    64 MB        64 MB     (1 pass, in-register)
Savings:              1.9ms   132 MB       132 MB     (2.8× less memory traffic)

Note: GEMM itself is 1.8ms; the separate elementwise ops add 72% overhead
      With fusion: only 22% overhead → total 2.2ms vs 3.1ms

Bandwidth budget:
  Unfused: 3 × M × N × 4 bytes = 192 MB at 900 GB/s = 0.21 ms wasted
  Fused:   1 × M × N × 4 bytes = 64 MB                → saves 0.14 ms

For a 24-layer BERT:
  24 × 4 GEMM blocks × 0.9ms savings = 86ms total speedup per forward pass
```

---

## 8. General Fusion Guidelines

```
Good candidates for fusion:
  - Elementwise ops after GEMM/Conv (bias, activation, dropout, LayerNorm)
  - Reduction followed by broadcast (mean subtraction, softmax)
  - Read-modify-write chains where intermediate results fit in registers

Poor candidates for fusion:
  - Two GEMMs (compute-bound; fusion doesn't help bandwidth)
  - Ops with very different parallelism (e.g., 1D + 2D kernels)
  - Ops requiring global synchronization between them

Decision rule (roofline):
  If bandwidth-limited: fuse (eliminate memory passes)
  If compute-limited:   don't bother (memory savings won't help)

Implementation order:
  1. Profile with ncu → identify bandwidth-bound ops
  2. Fuse those ops → re-profile
  3. Verify correctness (especially for backward passes)
```

---

## Key Takeaways

- Every separate kernel pass reads and writes the full tensor from HBM; fusing two operations into one kernel halves the HBM traffic for bandwidth-bound workloads
- **Fused bias+ReLU**: single in-place kernel; use float4 loads for 4× memory throughput
- **Approximate GELU**: `0.5 * x * (1 + tanh(0.7978 * (x + 0.044715 * x³)))` is widely used and has <0.1% error vs exact GELU
- **Fused residual+LayerNorm**: compute `x + res` twice (once for stats, once for output) rather than writing to HBM — register recomputation is faster than HBM round-trip
- **CUTLASS epilogue**: GEMM output remains in registers/shared memory across the epilogue, enabling zero-HBM-cost bias/activation fusion
- Apply fusion only to **bandwidth-bound** operations; fusing two compute-bound ops provides no benefit and increases code complexity

---

**Next**: [37. Multi-GPU and NCCL](./37_Multi_GPU_and_NCCL.md) — Scale beyond a single GPU using CUDA peer-to-peer transfers and NCCL collective operations to implement data, tensor, and pipeline parallelism.
