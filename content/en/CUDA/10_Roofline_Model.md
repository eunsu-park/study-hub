# 10. Roofline Model

**Previous**: [Occupancy and Launch Configuration](./09_Occupancy_and_Launch_Config.md) | **Next**: [Profiling with NCU and NSYS](./11_Profiling_with_NCU_NSYS.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Define arithmetic intensity (AI) and calculate it for any kernel
2. Build the roofline chart for a specific GPU
3. Determine whether a kernel is memory-bound or compute-bound
4. Set the achievable performance ceiling for any kernel before profiling
5. Apply roofline analysis to guide optimization decisions

---

## 1. The Core Question

**Why can't we just run the kernel and see?**

Without a performance model, you don't know whether your kernel is slow because:
- (a) You're limited by memory bandwidth → optimize memory access
- (b) You're limited by compute → vectorize, unroll, use Tensor Cores
- (c) You're achieving near-peak performance → stop optimizing

The **roofline model** answers this before you start optimizing.

---

## 2. Arithmetic Intensity

**Arithmetic Intensity (AI)** = FLOP / byte of memory traffic

```
AI = (floating-point operations) / (bytes read from + written to global memory)
```

This is a property of the **algorithm**, not the hardware.

### Examples

**Vector add** (`c[i] = a[i] + b[i]`):
```
FLOP = 1 add per element = N FLOPs
Bytes = 3N × 4 (read a, read b, write c)
AI = N / (12N) = 1/12 ≈ 0.083 FLOPs/byte
```

**Dot product** (`sum += a[i] * b[i]`):
```
FLOP = 2N (multiply + accumulate)
Bytes = 2N × 4 (read a, b; no write of the reduction result counted per element)
AI = 2N / 8N = 0.25 FLOPs/byte
```

**Matrix multiply** (`C = A × B`, N×N):
```
FLOP = 2N³ (N³ multiplications + N³ additions)
Bytes = (N² + N² + N²) × 4 = 12N² bytes (read A, B; write C)
AI = 2N³ / 12N² = N/6 FLOPs/byte

For N=1024: AI = 170 FLOPs/byte  (strongly compute-bound)
For N=64:   AI = 10.7 FLOPs/byte (borderline)
For N=16:   AI = 2.7  FLOPs/byte (memory-bound at small sizes)
```

**Stencil** (7-point 3D, `u[i] = a*u[i-1] + b*u[i] + a*u[i+1] + ...`):
```
FLOP = 13N (6 adds, 6 multiplies, 1 scale)
Bytes ≈ 7N × 4 (7 reads, ~1 write if neighbors are reused)
AI = 13N / 28N ≈ 0.46 FLOPs/byte  (memory-bound)
```

---

## 3. The Roofline Chart

The roofline chart has two axes:
- x-axis: Arithmetic Intensity (FLOPs/byte, log scale)
- y-axis: Performance (GFLOPS, log scale)

Two "roofs":
1. **Memory bandwidth roof**: `performance = AI × bandwidth` (diagonal line)
2. **Compute roof**: `performance = peak_FLOPS` (horizontal line)

```
GFLOPS
  │
  │ Compute roof (19,500 GFLOPS for A100 FP32)
  │─────────────────────────────────────────────────────────
  │                              / ← ridge point (9.75 FLOP/byte)
  │                             /
  │                            /   ← roofline (2.0 TB/s slope)
  │                           /
  │                          /      ●  GEMM (170 FLOP/byte) → near compute roof
  │                         /
  │                        /     ● Stencil (0.46 FLOP/byte) → memory-bound
  │                       /
  │         ● Vec add (0.083)
  │                      /
  └────────────────────────────────────────────────────────── AI (FLOP/byte)
   0.01   0.1    1    10   100  1000
```

**The ridge point** is where the memory bandwidth line meets the compute roof:
```
ridge = peak_FLOPS / bandwidth = 19,500 GFLOPS / 2,000 GB/s = 9.75 FLOP/byte (A100 FP32)
```

Any kernel with AI < ridge is memory-bound. AI > ridge is compute-bound.

---

## 4. GPU Ridge Points by Architecture

```python
# Ridge point = peak_FLOPS / bandwidth (units: FLOP/byte)
gpus = {
    "A100 (FP32)":   19_500 / 2_000,   # = 9.75  FLOP/byte
    "A100 (FP16)":  312_000 / 2_000,   # = 156   FLOP/byte (Tensor Core)
    "H100 (FP32)":   67_000 / 3_350,   # = 20.0  FLOP/byte
    "H100 (FP8)":  1_979_000 / 3_350,  # = 590   FLOP/byte (Tensor Core)
    "RTX 4090 (FP32)": 82_600 / 1_008, # = 81.9  FLOP/byte
    "RTX 3090 (FP32)": 35_600 /   936, # = 38.0  FLOP/byte
}
```

**Key insight**: H100's Tensor Core ridge point is 590 FLOP/byte — only FlashAttention-style kernels with massive data reuse can be compute-bound on Tensor Cores.

---

## 5. Building the Roofline for a Kernel

Step 1: Measure or calculate FLOP count.
Step 2: Measure actual bytes read/written (use `ncu` metrics or calculate analytically).
Step 3: Compute AI = FLOP / bytes.
Step 4: Compare AI to ridge point → determine bottleneck.
Step 5: Compute achievable peak performance = `min(AI × BW, peak_FLOPS)`.

### Example: Vector Scale (`y[i] = alpha * x[i]`)

```python
N = 1e6  # 1M elements

# Analytical:
FLOP  = 1 * N         # 1 multiply per element
bytes = 2 * N * 4     # read x, write y (8 bytes per element)
AI    = FLOP / bytes  # = 1 / 8 = 0.125 FLOP/byte

# A100 FP32 ridge = 9.75 FLOP/byte → memory-bound
achievable_GFLOPS = min(AI * 2000, 19500)  # = 0.125 * 2000 = 250 GFLOPS
achievable_time   = FLOP / (achievable_GFLOPS * 1e9)  # = 4 μs

# Measured: 4.2 μs → 238 GFLOPS → 95% of roofline → excellent!
```

---

## 6. Roofline Analysis: BLAS Level 1–3

| Operation | Example | Typical AI | Bottleneck |
|-----------|---------|------------|-----------|
| **BLAS-1** | `axpy`, `dot` | 0.08–0.5 | Memory bandwidth |
| **BLAS-2** | `gemv` (N=1024) | 0.5–2 | Memory bandwidth |
| **BLAS-3** | `gemm` (N=1024) | 170 | Compute |
| **Conv** | ResNet layer (C=256, K=3) | 5–50 | Depends on size |
| **Stencil** | 7-point 3D | 0.3–1.0 | Memory bandwidth |
| **Softmax** | sequence=2048 | 0.5–2 | Memory bandwidth |
| **FlashAttention** | T=2048, d=64 | 10–40 | Compute (with tiling) |

---

## 7. Roofline Script

```python
import numpy as np
import matplotlib.pyplot as plt

def roofline_plot(peak_gflops, peak_bw_gbs, kernels):
    """
    peak_gflops: compute ceiling in GFLOPS
    peak_bw_gbs: memory bandwidth in GB/s
    kernels: list of (name, ai, achieved_gflops)
    """
    ridge = peak_gflops / peak_bw_gbs

    ai_range = np.logspace(-2, 4, 500)
    bw_line  = peak_bw_gbs * ai_range
    roof     = np.minimum(bw_line, peak_gflops)

    plt.figure(figsize=(10, 6))
    plt.loglog(ai_range, roof, 'b-', linewidth=2, label='Roofline')
    plt.axvline(ridge, color='gray', linestyle='--', alpha=0.5)
    plt.text(ridge * 1.1, peak_gflops * 0.5, f'Ridge: {ridge:.1f}', color='gray')

    for name, ai, perf in kernels:
        efficiency = perf / min(ai * peak_bw_gbs, peak_gflops)
        plt.scatter(ai, perf, s=100, zorder=5)
        plt.annotate(f'{name}\n({efficiency:.0%})', (ai, perf),
                     textcoords='offset points', xytext=(5, 5))

    plt.xlabel('Arithmetic Intensity (FLOP/byte)')
    plt.ylabel('Performance (GFLOPS)')
    plt.title('Roofline Model — NVIDIA A100')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.savefig('roofline.png', dpi=150, bbox_inches='tight')

# A100 example
roofline_plot(
    peak_gflops = 19_500,
    peak_bw_gbs = 2_000,
    kernels = [
        ("vec_add",   0.083, 150),      # memory-bound, near roof
        ("gemm1024",  170,   17_000),   # compute-bound, near roof
        ("stencil",   0.46,  800),      # memory-bound, gap indicates room
    ]
)
```

---

## 8. Using Roofline to Guide Optimization

If your kernel is **below the roofline**:
- Memory-bound kernel below memory roof → improve coalescing, reduce strided access, use shared memory, reduce redundant loads
- Compute-bound kernel below compute roof → reduce arithmetic redundancy, vectorize with `float4`, use Tensor Cores, unroll loops

If your kernel is **on the roofline**:
- Memory-bound and on the memory roof → you've hit the bandwidth limit; no pure software optimization will help
- Compute-bound and on the compute roof → you're at theoretical peak; algorithm change (lower AI through tiling) or better hardware needed

---

## Key Takeaways

- **Arithmetic Intensity (AI)** = FLOPs / bytes — a property of the algorithm, independent of hardware
- **Ridge point** = peak_FLOPS / bandwidth — AI above ridge is compute-bound, below is memory-bound
- **Achievable performance** = `min(AI × bandwidth, peak_FLOPS)` — the hard ceiling for any kernel
- Most BLAS-1/2, stencil, and softmax operations are memory-bound on modern GPUs
- GEMM (large N) and FlashAttention (with tiling) are compute-bound
- The roofline tells you where to spend optimization effort before you write a single line of optimized code

---

**Next**: [11. Profiling with NCU and NSYS](./11_Profiling_with_NCU_NSYS.md) — Translate roofline theory into practice: collect the metrics that confirm bottlenecks and identify the root cause of performance gaps.
