# 33. Softmax and LayerNorm Kernels

**Previous**: [GEMM from Scratch](./32_GEMM_from_Scratch.md) | **Next**: [FlashAttention Kernel](./34_FlashAttention_Kernel.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why naive softmax is numerically unstable and implement the max-subtraction fix
2. Write a single-pass "online softmax" kernel that computes max, sum, and output in one pass using warp shuffles
3. Implement a fused LayerNorm kernel that computes mean and variance in one warp-shuffle pass
4. Write an RMSNorm kernel (simpler: no mean subtraction) used in Llama/Mistral models
5. Understand why fusing these operations reduces memory round-trips and improves throughput

---

## 1. Why Normalization Kernels Are Critical

In transformer models, softmax and LayerNorm are called thousands of times per forward pass:
```
BERT-large (24 layers):
  - 24 × 2 = 48 attention softmax ops  (batch × heads × seq_len)
  - 48 LayerNorm ops
  - These are memory-bandwidth bound: fast implementation → direct end-to-end speedup

Memory issue with naive 3-pass softmax:
  Pass 1: read row, find max               → 1× memory read
  Pass 2: read row, compute exp(x-max)     → 1× memory read
  Pass 3: read row, divide by sum(exp)     → 1× memory read + 1× write
  Total: 3 reads + 1 write per element

Online (1-pass) softmax:
  Single pass over row: 1 read + 1 write  → 3× memory bandwidth reduction
```

---

## 2. Naive Softmax (3-Pass, Numerically Stable)

```c
// softmax(x_i) = exp(x_i - max_x) / Σ exp(x_j - max_x)
// Each row of a [batch × seq_len] matrix is normalized independently
__global__ void softmax_naive(const float *in, float *out, int N) {
    // One block per row
    int row = blockIdx.x;
    const float *x = in + row * N;
    float       *y = out + row * N;

    // Pass 1: find max
    float maxval = -1e30f;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        maxval = fmaxf(maxval, x[i]);
    // Block-level max reduction (shared memory)
    maxval = block_reduce_max(maxval);  // see Lesson 14

    // Pass 2: compute exp(x - max) and sum
    float sum = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float e = expf(x[i] - maxval);
        y[i] = e;  // temporary store
        sum += e;
    }
    sum = block_reduce_sum(sum);

    // Pass 3: normalize
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        y[i] /= sum;
}
```

---

## 3. Online Softmax (Single Pass)

Online softmax maintains a running (max, sum) pair as it scans the row. The key insight: when a new max is found, rescale the existing partial sum:

```
Online algorithm:
  Initialize: m = -inf, d = 0
  For each x_i:
    m_new = max(m, x_i)
    d_new = d * exp(m - m_new) + exp(x_i - m_new)
    m = m_new, d = d_new

  Final: softmax(x_i) = exp(x_i - m) / d
```

```c
// Online softmax: one warp (32 threads) handles one row up to 32×unroll elements
// For rows longer than warp_size, use a thread-block with shared memory
__global__ void online_softmax(const float *in, float *out, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float *x = in  + row * cols;
    float       *y = out + row * cols;

    float m = -1e30f, d = 0.f;

    // --- Single pass: compute (max, normalizer) ---
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float xi = x[i];
        float m_new = fmaxf(m, xi);
        d = d * expf(m - m_new) + expf(xi - m_new);
        m = m_new;
    }

    // --- Warp-level reduction of (m, d) ---
    // Each thread has a local (m, d); reduce to find global max and rescaled sum

    // Step 1: reduce max across warp
    for (int offset = 16; offset > 0; offset >>= 1) {
        float m2 = __shfl_down_sync(0xffffffff, m, offset);
        float d2 = __shfl_down_sync(0xffffffff, d, offset);
        if (m2 > m) {
            d = d * expf(m - m2) + d2;
            m = m2;
        } else {
            d = d + d2 * expf(m2 - m);
        }
    }
    // Broadcast result from lane 0
    m = __shfl_sync(0xffffffff, m, 0);
    d = __shfl_sync(0xffffffff, d, 0);

    // --- Write output ---
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        y[i] = expf(x[i] - m) / d;
}
```

---

## 4. Online Softmax with Shared Memory (for Long Rows)

For rows longer than 32 elements, coordinate across the full thread block using shared memory:

```c
__global__ void online_softmax_block(
    const float *in, float *out, int rows, int cols)
{
    extern __shared__ float smem[];  // 2 * blockDim.x floats (m and d per thread)
    float *sm = smem;
    float *sd = smem + blockDim.x;

    int row = blockIdx.x;
    const float *x = in  + row * cols;
    float       *y = out + row * cols;
    int tid = threadIdx.x;

    float m = -1e30f, d = 0.f;

    // Thread-local online accumulation
    for (int i = tid; i < cols; i += blockDim.x) {
        float xi = x[i];
        float m_new = fmaxf(m, xi);
        d = d * expf(m - m_new) + expf(xi - m_new);
        m = m_new;
    }

    // Store to shared memory
    sm[tid] = m;
    sd[tid] = d;
    __syncthreads();

    // Tree reduction: parallel merge of (m, d) pairs
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float m2 = sm[tid + stride];
            float d2 = sd[tid + stride];
            float m_new = fmaxf(sm[tid], m2);
            sd[tid] = sd[tid] * expf(sm[tid] - m_new)
                    + d2      * expf(m2      - m_new);
            sm[tid] = m_new;
        }
        __syncthreads();
    }

    // Broadcast from thread 0
    m = sm[0];
    d = sd[0];
    __syncthreads();

    // Write output in one final pass
    for (int i = tid; i < cols; i += blockDim.x)
        y[i] = expf(x[i] - m) / d;
}
```

---

## 5. LayerNorm Kernel

LayerNorm normalizes each feature vector (row) to zero mean and unit variance, then scales and shifts:

```
y = (x - mean(x)) / sqrt(var(x) + ε) * γ + β

mean(x) = (1/H) Σ x_i
var(x)  = (1/H) Σ (x_i - mean)²
```

```c
// Fused LayerNorm: compute mean and variance in a single pass (Welford's algorithm)
// γ (weight) and β (bias) are learned parameters of shape [H]
__global__ void layernorm_forward(
    const float *x,      // [batch × H]
    const float *gamma,  // [H]
    const float *beta,   // [H]
    float *out,          // [batch × H]
    float *mean_out,     // [batch] (saved for backward)
    float *var_out,      // [batch]
    int H, float eps)
{
    int row = blockIdx.x;
    const float *xi = x   + row * H;
    float       *yi = out + row * H;

    // Practical approach: two-value reduction (sum and sum_sq)
    // Note: Welford's online algorithm is elegant for sequential updates but
    // does not parallelize directly — merging partial Welford states requires
    // a non-trivial combine step. The sum/sum_sq approach below is simpler
    // and equally numerically stable for float32 at typical hidden-dim sizes.
    float sum = 0.f, sum_sq = 0.f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        float v = xi[i];
        sum    += v;
        sum_sq += v * v;
    }

    // Warp-level reduce sum and sum_sq simultaneously
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum    += __shfl_down_sync(0xffffffff, sum,    offset);
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    __shared__ float s_sum[32], s_sum_sq[32];
    int warp_id = threadIdx.x / 32;
    int lane    = threadIdx.x % 32;

    if (lane == 0) { s_sum[warp_id] = sum; s_sum_sq[warp_id] = sum_sq; }
    __syncthreads();

    // Final reduce across warps (thread 0 only)
    if (threadIdx.x == 0) {
        float total_sum = 0.f, total_sq = 0.f;
        int n_warps = blockDim.x / 32;
        for (int w = 0; w < n_warps; w++) {
            total_sum += s_sum[w];
            total_sq  += s_sum_sq[w];
        }
        float mn  = total_sum / H;
        float var = total_sq / H - mn * mn;
        s_sum[0]    = mn;
        s_sum_sq[0] = var;
        if (mean_out) mean_out[row] = mn;
        if (var_out)  var_out[row]  = var;
    }
    __syncthreads();

    float mn   = s_sum[0];
    float var  = s_sum_sq[0];
    float rstd = rsqrtf(var + eps);

    // Normalize and apply affine transform
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        yi[i] = (xi[i] - mn) * rstd * gamma[i] + beta[i];
}
```

---

## 6. RMSNorm (No Mean Subtraction)

RMSNorm (used in LLaMA, Mistral) is simpler: normalize by root-mean-square, no mean subtraction:

```
RMSNorm(x)_i = x_i / RMS(x) * γ_i
RMS(x)       = sqrt((1/H) Σ x_i²)
```

```c
__global__ void rmsnorm_forward(
    const float *x,     // [batch × H]
    const float *gamma, // [H]
    float *out,         // [batch × H]
    int H, float eps)
{
    int row = blockIdx.x;
    const float *xi = x   + row * H;
    float       *yi = out + row * H;

    // Compute sum of squares
    float sum_sq = 0.f;
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        sum_sq += xi[i] * xi[i];

    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);

    __shared__ float s_sq[32];
    int warp_id = threadIdx.x / 32;
    int lane    = threadIdx.x % 32;
    if (lane == 0) s_sq[warp_id] = sum_sq;
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = 0.f;
        for (int w = 0; w < blockDim.x/32; w++) total += s_sq[w];
        s_sq[0] = rsqrtf(total / H + eps);  // 1/RMS
    }
    __syncthreads();

    float rrms = s_sq[0];
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        yi[i] = xi[i] * rrms * gamma[i];
}
```

---

## 7. Performance Analysis

```
Configuration: batch=128, H=768 (BERT-base), float32

Kernel             Time    Bandwidth util    Memory passes
----------------------------------------------------------
softmax 3-pass      0.8ms     65%              3 reads + 1 write
online softmax      0.3ms     85%              1 read + 1 write  (2.6× faster)
layernorm 2-pass    0.6ms     60%              2 reads + 1 write
layernorm 1-pass    0.35ms    80%              1 read + 1 write  (1.7× faster)
rmsnorm             0.25ms    90%              1 read + 1 write  (simpler)

Note: for H ≤ 32: all fits in a single warp → no shared memory needed
      for H ≤ 1024: shared-memory reduction sufficient
      for H > 1024: need multi-block reduction with atomics or two-pass
```

---

## Key Takeaways

- **Naive softmax** reads each row 3 times; **online softmax** merges all passes into one by maintaining a running (max, rescaled-sum) pair
- The online merge rule: when encountering a new max `m_new`, rescale the existing sum: `d_new = d * exp(m - m_new) + exp(x_i - m_new)`
- **Warp shuffle reduction** for (max, sum) pairs requires a custom merge step (not a simple add): compare maxes, then rescale the losing side's sum
- **LayerNorm**: compute sum and sum_sq in one pass → mean = sum/H, var = sum_sq/H - mean² → apply affine transform; save mean and rstd for backward pass
- **RMSNorm** skips mean subtraction entirely, computing only the root-mean-square for normalization; ~30% faster than LayerNorm for the same hidden dimension
- These kernels are all **memory-bandwidth bound**: the key optimization is minimizing the number of passes over the input row

---

**Next**: [34. FlashAttention Kernel](./34_FlashAttention_Kernel.md) — Implement the FlashAttention tiling algorithm that computes exact attention in O(N²/B) HBM operations instead of O(N²), enabling long-context transformers without out-of-memory errors.
