# 35. Quantized Kernels — INT8

**Previous**: [FlashAttention Kernel](./34_FlashAttention_Kernel.md) | **Next**: [Fused Kernel Patterns](./36_Fused_Kernel_Patterns.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement absmax quantization for converting FP32 tensors to INT8
2. Distinguish per-tensor (activation) and per-channel (weight) quantization
3. Use the `dp4a` instruction for INT8 dot products with INT32 accumulation
4. Sketch an INT8 GEMM kernel with dequantization fused into the output epilogue
5. Implement INT4 weight-only dequantization for memory-bandwidth-limited inference

---

## 1. Why Quantization?

```
INT8 vs FP32 comparison:
  Storage:      4× less memory (1 byte vs 4 bytes)
  Bandwidth:    4× more values per memory transaction
  Compute:      2× throughput on CUDA cores (INT8 vs FP32)
                4-8× throughput with dp4a / Tensor Cores (INT8 TC)

A100 peak throughput:
  FP32 CUDA cores:   19.5 TFLOPS
  INT8 Tensor Cores: 624 TOPS  (32× vs FP32!)

Use cases:
  Inference quantization: weights (INT4/INT8) + activations (INT8/FP16)
  Training quantization (QAT): simulate quantization noise during training

Accuracy cost:
  FP32 → INT8 weights:      <0.5 perplexity increase (large models)
  FP32 → INT4 weights:      0.5-2 perplexity increase
  FP32 → INT8 activations:  requires careful calibration
```

---

## 2. Absmax Quantization

The simplest uniform quantization: map [-max_val, +max_val] to [-127, 127]:

```
scale       = max(|x|) / 127
x_quantized = round(x / scale)        clamp to [-127, 127]
x_dequant   = x_quantized * scale

Error analysis:
  Maximum quantization error ≈ scale / 2 = max(|x|) / 254
  Relative error ≈ 1/254 ≈ 0.4%  (for typical weight distributions)
```

```c
// Quantize a float array to INT8 (absmax per-tensor)
__global__ void quantize_absmax(
    const float *x, int8_t *x_q, float *scale_out, int N)
{
    extern __shared__ float smem[];
    int i   = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float val = (i < N) ? fabsf(x[i]) : 0.f;
    smem[tid] = val;
    __syncthreads();

    // Max reduction within block
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }

    // Thread 0 publishes block max via atomicMax (using int representation)
    if (tid == 0) {
        // Atomic float max: use integer atomicMax on IEEE 754 bits
        unsigned int old_bits = __float_as_uint(smem[0]);
        unsigned int *addr    = (unsigned int*)scale_out;
        atomicMax(addr, old_bits);
    }
    __syncthreads();

    // Wait for global max to be available; in practice use a 2-pass approach
    // Final quantize (using scale = global_max / 127)
    if (i < N) {
        float scale = (*scale_out) / 127.f;
        float q = __float2int_rn(x[i] / scale);   // round-to-nearest
        x_q[i] = (int8_t)fminf(127.f, fmaxf(-127.f, q));
    }
}

// Dequantize: int8 → float
__global__ void dequantize(
    const int8_t *x_q, float *x_out, float scale, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x_out[i] = x_q[i] * scale;
}
```

---

## 3. Per-Channel Quantization for Weights

Per-channel quantization assigns a separate scale to each output channel (row) of the weight matrix. This captures per-channel value distribution differences:

```c
// Quantize weight matrix W [out_features × in_features]
// Each row gets its own scale (per output channel)
__global__ void quantize_per_channel(
    const float *W,        // [OC × IC]
    int8_t *W_q,           // [OC × IC]
    float  *scales,        // [OC]
    int OC, int IC)
{
    int row = blockIdx.x;   // one block per output channel
    int tid = threadIdx.x;

    // Find max abs in this row
    float maxval = 0.f;
    for (int i = tid; i < IC; i += blockDim.x)
        maxval = fmaxf(maxval, fabsf(W[row * IC + i]));

    // Block-level max reduction
    extern __shared__ float smax[];
    smax[tid] = maxval;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) smax[tid] = fmaxf(smax[tid], smax[tid + s]);
        __syncthreads();
    }

    float scale = smax[0] / 127.f;
    if (tid == 0) scales[row] = scale;
    __syncthreads();

    // Quantize
    for (int i = tid; i < IC; i += blockDim.x) {
        float q = W[row * IC + i] / scale;
        W_q[row * IC + i] = (int8_t)fminf(127.f, fmaxf(-127.f, __float2int_rn(q)));
    }
}
```

---

## 4. dp4a: INT8 Dot Product in 4 Operations

`dp4a` computes a 4-element INT8 dot product with INT32 accumulation in a single instruction:

```c
// dp4a: a = Σ_{k=0}^{3} a_k * b_k  (each a_k, b_k is int8; result int32)
// Available since Pascal (sm_61)

// Pack 4 int8 values into one int32
__device__ int pack_int8x4(int8_t a, int8_t b, int8_t c, int8_t d) {
    return ((int)a & 0xFF) | (((int)b & 0xFF) << 8)
         | (((int)c & 0xFF) << 16) | (((int)d & 0xFF) << 24);
}

// Manual dp4a (compiler usually generates this automatically with int8 loads)
__device__ int dp4a(int a_packed, int b_packed, int c_acc) {
    return __dp4a(a_packed, b_packed, c_acc);
    // Intrinsic: int __dp4a(int a, int b, int c)
    //   Returns c + (a[7:0]*b[7:0]) + (a[15:8]*b[15:8])
    //              + (a[23:16]*b[23:16]) + (a[31:24]*b[31:24])
}

// INT8 dot product for K elements (K must be multiple of 4)
__device__ int int8_dot(const int8_t *a, const int8_t *b, int K) {
    const int *a4 = (const int*)a;   // treat as array of packed int8×4
    const int *b4 = (const int*)b;
    int acc = 0;
    for (int k = 0; k < K/4; k++)
        acc = __dp4a(a4[k], b4[k], acc);
    return acc;  // INT32 accumulator
}
```

---

## 5. INT8 GEMM Kernel with Dequantization

```c
// INT8 GEMM: C_int32 = A_int8 · B_int8^T, then dequantize
// A: [M × K] int8 (activations, per-tensor scale scale_a)
// B: [N × K] int8 (weights, per-channel scales scales_b[N])
// C: [M × N] float (output after dequantization)

#define TILE 32

__global__ void int8_gemm(
    const int8_t *A, const int8_t *B,
    float *C,
    int M, int N, int K,
    float scale_a, const float *scales_b)
{
    __shared__ int8_t sA[TILE][TILE];
    __shared__ int8_t sB[TILE][TILE];  // B^T stored as [N × K] → tile [TILE × TILE]

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    int acc = 0;  // INT32 accumulator

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        // Load INT8 tiles (4-byte aligned loads for dp4a)
        int a_col = t * TILE + tx;
        int b_col = t * TILE + ty;  // B accessed as [col][k]

        sA[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0;
        sB[ty][tx] = (col < N && b_col < K) ? B[col * K + b_col] : 0;

        __syncthreads();

        // Dot product using dp4a (pack 4 elements at a time)
        // For dp4a alignment, ensure TILE is multiple of 4
        for (int k = 0; k < TILE; k += 4) {
            int a_packed = *reinterpret_cast<const int*>(&sA[ty][k]);
            int b_packed = *reinterpret_cast<const int*>(&sB[tx][k]);
            acc = __dp4a(a_packed, b_packed, acc);
        }

        __syncthreads();
    }

    // Dequantize and write output
    if (row < M && col < N) {
        float scale = scale_a * scales_b[col];
        C[row * N + col] = (float)acc * scale;
    }
}
```

---

## 6. INT4 Weight-Only Dequantization

INT4 (4-bit) weights pack two values per byte. Dequantize on the fly during GEMM to save memory bandwidth:

```c
// Pack two INT4 values into one uint8:
//   high nibble = first  value (bits 7:4)
//   low  nibble = second value (bits 3:0)
// Range: signed INT4 → [-8, 7]

__device__ void unpack_int4x2(uint8_t packed, int8_t &hi, int8_t &lo) {
    // Sign-extend 4-bit to 8-bit
    hi = (int8_t)((int8_t)(packed >> 4) << 4 >> 4);   // arithmetic right shift
    lo = (int8_t)((int8_t)(packed << 4)       >> 4);
}

// INT4 weight dequantization kernel
// W_int4: packed [OC × (IC/2)] uint8 (2 weights per byte)
// scales: [OC × (IC/group_size)] (group quantization)
__global__ void dequant_int4_to_fp16(
    const uint8_t *W_int4,  // [OC × IC/2] packed
    __half *W_fp16,         // [OC × IC]   output
    const __half *scales,   // [OC × ngroups] per group scale
    int OC, int IC, int group_size)
{
    int oc = blockIdx.y * blockDim.y + threadIdx.y;
    int ic = (blockIdx.x * blockDim.x + threadIdx.x) * 2;  // two INT4 per iteration
    if (oc >= OC || ic >= IC) return;

    // Load packed byte (2 INT4 values)
    uint8_t packed = W_int4[oc * (IC/2) + ic/2];
    int8_t hi, lo;
    unpack_int4x2(packed, hi, lo);

    // Scale: group quantization (one scale per group_size weights)
    int group = ic / group_size;
    float scale = __half2float(scales[oc * (IC / group_size) + group]);

    W_fp16[oc * IC + ic]   = __float2half(hi * scale);
    W_fp16[oc * IC + ic+1] = __float2half(lo * scale);
}
```

---

## 7. Perplexity Impact and Calibration

```
Quantization accuracy for LLaMA-7B (WikiText-2 perplexity):

Precision        Perplexity  Memory (7B params)
------------------------------------------------
FP32             5.68        28 GB
FP16             5.68        14 GB  (no accuracy loss)
INT8 (W8A8)      5.72        7 GB   (+0.04 ppl)
INT8 (W8A16)     5.70        7 GB   (+0.02 ppl)
INT4 (W4A16)     5.85        3.5 GB (+0.17 ppl)  ← 4× memory reduction!
INT4 NF4 (QLoRA) 5.80        3.5 GB (+0.12 ppl)  (NF4 = normalized 4-bit)

Calibration set:
  Collect ~512 representative input samples
  Run forward pass to measure activation ranges
  Compute per-channel scales based on actual distribution
  Bad calibration → accuracy drops 10× more than good calibration
```

---

## Key Takeaways

- **Absmax quantization**: `scale = max(|x|) / 127`, `x_q = round(x / scale)` clamped to [-127, 127]; dequantize with `x = x_q * scale`
- **Per-tensor vs per-channel**: activations use per-tensor scale (unknown at compile time); weights use per-channel scale (one scale per output row) for better accuracy
- **dp4a** (`__dp4a`): single instruction computing a 4-element INT8 dot product with INT32 accumulation — the fundamental building block of INT8 GEMM on GPU
- **INT8 GEMM epilogue**: the INT32 accumulator must be dequantized before writing: `C_float[i] = acc_int32 * scale_A * scale_B[col]`
- **INT4 weight-only**: pack 2 weights per byte; dequantize to FP16 on the fly before the matrix multiply; 4× memory bandwidth savings vs FP16 weights
- **Calibration** is critical: run representative data through the model to measure true activation ranges; poor calibration causes 5-10× larger accuracy loss than theoretical minimum

---

**Next**: [36. Fused Kernel Patterns](./36_Fused_Kernel_Patterns.md) — Learn why kernel fusion reduces memory round-trips and implement fused bias+ReLU, fused residual+LayerNorm, and other common deep learning kernel fusion patterns.
