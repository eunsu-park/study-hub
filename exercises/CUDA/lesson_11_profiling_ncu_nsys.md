# Lesson 11 — Profiling with Nsight Compute and Nsight Systems (per-lesson exercise)

Prerequisites: NVIDIA Nsight Systems and Nsight Compute installed. The kernel from CUDA L05 (tiled transpose) or any non-trivial kernel.

Compile: `nvcc -O3 -arch=sm_80 -lineinfo ex.cu -o ex` (the `-lineinfo` flag is essential — it lets the profilers attribute time to source lines).

Two complementary tools:

- **Nsight Systems (nsys)** — system-wide timeline. Shows when each kernel, memcpy, and CPU API call happens. Good for spotting "GPU is idle 80% of the time waiting on the CPU."
- **Nsight Compute (ncu)** — per-kernel deep dive. Shows occupancy, memory bandwidth, instruction mix, warp stall reasons. Good for "this kernel is using 30% of memory bandwidth — why?"

---

## Exercise 11.1 — Timeline with nsys

**Difficulty**: ★

### Problem

Profile your binary's complete execution:

```bash
nsys profile --trace=cuda,nvtx --output=run.qdrep ./ex
nsys-ui run.qdrep      # opens the GUI
```

In the timeline you should see:
- A "CUDA HW" track showing each kernel as a colored bar
- A "CUDA API" track showing host-side `cudaMemcpy`, `cudaLaunchKernel`, etc.
- Idle gaps between kernels (white space)

Check the gap-to-active ratio. For a tight inference loop, gaps should be < 10% of timeline. Larger gaps mean the host is doing CPU work between kernels — moving that work off the critical path is the next optimization.

---

## Exercise 11.2 — NVTX Markers for Annotation

**Difficulty**: ★★

Add `nvtxRangePushA("phase_name")` / `nvtxRangePop()` calls in your host code:

```cuda
#include <nvtx3/nvToolsExt.h>

nvtxRangePushA("preprocess");
preprocess_on_host(...);
nvtxRangePop();

nvtxRangePushA("inference");
launch_inference_kernels(...);
cudaDeviceSynchronize();
nvtxRangePop();
```

In `nsys-ui`, these phases appear as labeled bands on the NVTX track — making it trivial to see "preprocess takes 30 ms, inference takes 50 ms." Critical for understanding where time goes in real workloads.

---

## Exercise 11.3 — Per-Kernel Profile with ncu

**Difficulty**: ★★

Profile a single kernel of interest:

```bash
ncu --set full --kernel-name regex:transpose --launch-skip 0 --launch-count 1 \
    --export run.ncu-rep ./ex
ncu-ui run.ncu-rep
```

Key metrics to read:

- **SM Active**: how busy the SMs are. Below 50% suggests low occupancy or memory bottleneck.
- **Memory Throughput**: % of peak DRAM bandwidth in use. Above 80% means you are bandwidth-bound — optimize memory access, not compute.
- **Warp State Stall Reasons**: where warps are stuck. "Long Scoreboard" = waiting on global memory. "Short Scoreboard" = waiting on shared memory or atomic. Each suggests a different fix.

For your tiled transpose from CUDA L05:

| Without `+1` padding | With `+1` padding |
|----------------------|-------------------|
| Bank conflicts: ~1.0 | Bank conflicts: 0 |
| Active SMs: ~80% | Active SMs: ~95% |

Reading these numbers and connecting them to a code change you can make is the core skill of GPU performance engineering.

---

## Exercise 11.4 — Roofline Analysis — Bonus

**Difficulty**: ★★★

Nsight Compute's roofline view plots your kernel's arithmetic intensity (FLOPs per byte) against achieved performance. The two roofs are:

- Memory roof: FLOPs/sec achievable at this arithmetic intensity given peak bandwidth.
- Compute roof: peak FLOPs/sec of the GPU.

Where your kernel sits relative to those roofs tells you the optimization to pursue:

- Below memory roof → improve coalescing, use shared memory, reduce traffic.
- Below compute roof but at memory roof → either re-architect for higher arithmetic intensity, or accept this as the bandwidth-bound limit.
- At or near compute roof → done; further work has diminishing returns.

Run the roofline view on `gemm_naive` vs `gemm_tiled` from CUDA L32. The tiled version should sit much higher on the chart — visualizing why register tiling matters.
