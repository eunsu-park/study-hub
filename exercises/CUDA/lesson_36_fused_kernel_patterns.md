# Lesson 36 — Fused Kernel Patterns (per-lesson exercise)

Prerequisites: L04 (memory model), L32 (GEMM), deep-learning kernel basics.

Compile: `nvcc -O3 -arch=sm_80 ex.cu -o ex`

Many deep-learning operations are **bandwidth-bound** on GPU — memory I/O is the limit, not compute. Fusing two or more bandwidth-bound ops into a single kernel halves (or better) the global-memory traffic.

The paradigmatic examples:

- **Bias + Activation**: `out = relu(mat + bias)` fused with the preceding GEMM output write
- **LayerNorm + Residual**: `out = layernorm(x + residual)`
- **Softmax + Dropout**: saves two full passes over the `[N, V]` logit tensor during training

---

## Exercise 36.1 — Fused Bias + GELU

**Difficulty**: ★★

### Problem

Write `__global__ void fused_bias_gelu(float *y, const float *bias, int N, int D)` that adds `bias[d]` to each element `y[n, d]` and applies GELU in place — one pass over the matrix.

GELU (approximation used by GPT-2 and Llama):
$$\text{GELU}(x) = 0.5 \cdot x \cdot \left(1 + \tanh\!\left(\sqrt{2/\pi}\,(x + 0.044715 x^3)\right)\right)$$

### Starter

```cuda
#include <cmath>
__global__ void fused_bias_gelu(float *y, const float *bias, int N, int D) {
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N && d < D) {
        float x = y[n * D + d] + bias[d];
        float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
        y[n * D + d] = 0.5f * x * (1.0f + tanhf(inner));
    }
}
```

Benchmark vs. two separate kernels (add_bias, then gelu_in_place). The fused version should be essentially 2× faster on any bandwidth-bound size, because the same data is read/written once instead of twice.

---

## Exercise 36.2 — Fused Residual + LayerNorm

**Difficulty**: ★★★

The transformer block runs `x = LayerNorm(x + residual)` after every attention and FFN sub-layer. In an unfused pipeline, the `x + residual` write and the three LayerNorm passes (mean, variance, normalize) are four trips to DRAM.

Fuse them: one kernel launches with one block per row, does the add, the mean+variance (Welford), and the normalize — all while the row sits in shared memory.

Measure the speedup vs. the unfused version at sizes `[batch × seqlen, d_model]` ∈ {`[1024, 768]`, `[4096, 4096]`}. At the larger size expect a 3–4× speedup; at the smaller size the fixed kernel-launch cost erodes the win.

---

## Exercise 36.3 — Fused Softmax + Dropout — Bonus

**Difficulty**: ★★★★

Combine a block-per-row softmax (as in CUDA L33) with per-element inverted-dropout scaling (`x = x * mask / (1 - p)` where `mask` is 0 or 1). Use `curand_uniform` for the mask generation inline. This is the pattern used in every modern transformer training kernel; understanding it is a prerequisite to reading FlashAttention's dropout path.

---

## When to Fuse — a Rule of Thumb

Fusing pays off when:

- Both ops are bandwidth-bound (compute density < roofline knee).
- The operand that would cross kernel boundaries is large compared to shared-memory or register budget.
- The combined kernel still fits the occupancy constraint.

Fusing does NOT pay off when:

- One of the ops is already compute-bound (GEMM vs. a cheap activation — the activation time is in the noise).
- The resulting kernel becomes too register-heavy and occupancy drops materially.
- Numerical stability requires intermediate high-precision storage (rare but real).

Profile-before-and-after is the only reliable test.
