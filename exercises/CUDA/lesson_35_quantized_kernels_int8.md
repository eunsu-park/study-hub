# Lesson 35 — Quantized Kernels (Int8) (per-lesson exercise)

Prerequisites: L30 (Tensor Cores), L32 (GEMM), DL_Scratch_C L40 (quantization theory).

Compile: `nvcc -O3 -arch=sm_80 ex.cu -lcublas -o ex`

Int8 GEMM on tensor cores delivers 2-4× the FP16 throughput (and 4-8× the FP32 throughput) at the cost of small accuracy loss. Modern LLM inference engines (vLLM, TensorRT-LLM, TGI) lean heavily on int8 and int4 kernels.

---

## Exercise 35.1 — cuBLAS Int8 GEMM via GemmEx

**Difficulty**: ★★

### Problem

Multiply two int8 matrices using cuBLAS's tensor-core path:

```cuda
#include <cublas_v2.h>

cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
             M, N, K,
             &alpha,
             dA, CUDA_R_8I, K,           /* int8 input */
             dB, CUDA_R_8I, N,           /* int8 input */
             &beta,
             dC, CUDA_R_32I, N,          /* int32 accumulator and output */
             CUBLAS_COMPUTE_32I,         /* int32 compute */
             CUBLAS_GEMM_DEFAULT_TENSOR_OP);
```

The accumulator is INT32 because INT8 × INT8 can overflow INT8 in just a few multiplications.

Time at $M = N = K = 4096$ and compare to fp32 GEMM (CUDA L29.1) and fp16 GEMM (CUDA L29.2). On A100 expect:

| Precision | TFLOPS | Speedup vs fp32 |
|-----------|--------|-----------------|
| fp32 | ~17 | 1× |
| fp16 (tensor cores) | ~80 | ~5× |
| int8 (tensor cores) | ~160 | ~9× |

---

## Exercise 35.2 — Per-Channel Quantize and Dequantize

**Difficulty**: ★★

### Problem

For matrix $W$ shape `[O, I]`, compute one scale per row (per output channel):

```
scale[o] = max_i |W[o, i]| / 127
W_int8[o, i] = round(W[o, i] / scale[o])
```

Implement two kernels:

```cuda
__global__ void quantize_per_channel(const float *W, int8_t *W_q, float *scale,
                                     int O, int I);

__global__ void dequantize_per_channel(const int8_t *W_q, const float *scale,
                                       float *W_back, int O, int I);
```

Verify round-trip error on a 1024×4096 random matrix: per-element error should be ≤ scale[o]/2 for that row. Per-channel quantization (vs single-scale) typically halves the average error.

---

## Exercise 35.3 — Mixed Precision Matmul Pipeline

**Difficulty**: ★★★★

In a real LLM inference loop, each Linear layer does:

1. Activations are fp16 (input).
2. Weights are int8 stored on-disk.
3. Compute path: fp16 act × int8 weight → fp32 accumulator.
4. Output is rescaled per-output-channel and cast back to fp16 for the next layer.

Implement this pipeline for a single Linear (`y = x @ W^T + b`) with `M = 512, K = 4096, N = 11008` (Llama 7B FFN dimensions). Build it as a fused kernel that:

- Loads activations as fp16 from global memory
- Loads weights as int8
- Computes the matmul using INT8 tensor cores (with fp16→int8 quantization on the fly for the activations using a per-tensor scale)
- Accumulates in INT32, then converts back to fp16 with the per-channel weight scale

Compare against the fp16 reference (cuBLAS GemmEx with both fp16 inputs). Quality: |output_int8 - output_fp16|/|output_fp16| should be < 0.01 on average. Speed: 2-4× faster than fp16.

This is the kernel that lets a 70B-parameter Llama run on a single 80GB H100 instead of needing two.

---

## Exercise 35.4 — Activation Outliers — Bonus

**Difficulty**: ★★★

Real activations from large transformer models have **outliers** — a few channels with magnitudes 10× higher than the median. Naive per-tensor quantization wastes bits on the outliers, degrading accuracy.

Implement **outlier separation**: split each activation matrix into a "main" int8 part and an "outlier" fp16 part (top 0.1% by magnitude). Multiply each separately, then sum. This is the technique behind LLM.int8() and BitsAndBytes.

Verify on a synthetic input where 0.1% of channels have magnitude 50 and the rest have magnitude 1. Pure int8 quantization will lose ~5% accuracy on this distribution; mixed-precision recovers it within 0.1%.
