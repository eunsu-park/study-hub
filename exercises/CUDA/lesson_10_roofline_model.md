# Lesson 10 — Roofline Model (per-lesson exercise)

Prerequisites: L09 (occupancy), L11 (profiling).

Compile: `nvcc -O3 -arch=sm_80 ex.cu -o ex`

The roofline model plots **achieved performance** vs. **arithmetic intensity** (FLOPs per byte of DRAM traffic). The two roofs are:

- **Memory roof**: $\text{performance} = \text{intensity} \times \text{bandwidth}$ (a sloped line from origin)
- **Compute roof**: $\text{performance} = \text{peak FLOPS}$ (a horizontal line)

Your kernel sits below both roofs; the closer it sits to one, the more dominated by that resource it is.

---

## Exercise 10.1 — Compute the Roofs for Your GPU

**Difficulty**: ★

### Problem

Query the device properties and compute:

```cuda
#include <cstdio>
#include <cuda_runtime.h>

int main(void) {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);

    /* Memory: clock × bus_width / 8 × 2 (DDR) */
    double bw_GBs = p.memoryClockRate * 1e3 * (p.memoryBusWidth / 8) * 2.0 / 1e9;

    /* Peak FP32: SMs × cores_per_SM × 2 (FMA) × clock */
    int cores_per_sm;
    if (p.major >= 8) cores_per_sm = 64;       /* A100, A40, etc. */
    else if (p.major == 7) cores_per_sm = 64;  /* V100 */
    else cores_per_sm = 128;
    double peak_TFLOPS = p.multiProcessorCount * cores_per_sm * 2.0 *
                         p.clockRate * 1e3 / 1e12;

    /* Knee = peak_TFLOPS / bandwidth = arithmetic intensity at the corner */
    double knee = peak_TFLOPS * 1e3 / bw_GBs;   /* in FLOPs/byte */

    printf("Device: %s\n", p.name);
    printf("  DRAM bandwidth: %.1f GB/s\n", bw_GBs);
    printf("  Peak FP32: %.1f TFLOPS\n", peak_TFLOPS);
    printf("  Roofline knee: %.2f FLOPs/byte\n", knee);
    return 0;
}
```

For an A100: bandwidth ~1555 GB/s, peak FP32 ~19.5 TFLOPS, knee ~12.5 FLOPs/byte. A kernel below 12.5 FLOPs/byte is memory-bound; above is compute-bound.

---

## Exercise 10.2 — Place a Kernel on the Roofline

**Difficulty**: ★★

For your tiled GEMM from CUDA L32:

- FLOPs = $2 \cdot M \cdot N \cdot K$
- DRAM bytes ≈ $4 \cdot (MN + MK + NK)$ bytes (assumes the matrices fit in DRAM and pass through the cache hierarchy once)
- Arithmetic intensity = FLOPs / bytes ≈ $\frac{2 MNK}{4(MN + MK + NK)} = \frac{MNK}{2(MN+MK+NK)}$

For $M = N = K = 4096$: intensity ≈ $\frac{4096^3}{2 \cdot 3 \cdot 4096^2} = \frac{4096}{6} \approx 683$ FLOPs/byte. Way above the A100 knee — GEMM is **compute-bound** at this size, which is why tensor cores matter.

Time the kernel and compute its achieved TFLOPS. Compare to the compute roof. A well-tuned GEMM hits 80-95% of peak; a hand-rolled one might hit 30-50%.

---

## Exercise 10.3 — Three Kernels, Three Bottlenecks

**Difficulty**: ★★

Time and place these three kernels on your roofline plot:

| Kernel | Intensity (FLOPs/byte) | Bound by |
|--------|------------------------|----------|
| `vec_add` (CUDA L02) | 0.25 (1 FMA / 12 bytes) | Memory |
| `softmax` over `[B, V]` | ~5 (one row read + write) | Memory |
| `gemm` `[4096³]` | 683 | Compute |

For each, plot a point (intensity, achieved TFLOPS) on a log-log graph with the two roofs. The shape of the data tells you what each kernel needs:

- vec_add at memory roof → optimize coalescing, do nothing else; you cannot beat physics.
- softmax also at memory roof → fuse with the preceding GEMM (CUDA L36) to avoid the round-trip.
- GEMM well below compute roof → switch to tensor cores (CUDA L30).

---

## Exercise 10.4 — Roofline-Driven Optimization Plan — Bonus

**Difficulty**: ★

For a kernel from your existing project:

1. Measure intensity (cProfile-style: count FLOPs in source; measure DRAM bytes via `ncu`'s "DRAM Bytes" metric).
2. Plot on the roofline.
3. Decide: am I memory-bound, compute-bound, or both?
4. Pick ONE optimization that moves the dot toward the relevant roof — and ONLY that.

The discipline of "one bottleneck, one optimization, re-measure" is what professional kernel engineers practice. Random hyperparameter sweeps without a roofline picture are usually wasted effort.
