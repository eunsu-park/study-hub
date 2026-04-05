# Block 2 — Performance

**Lessons covered**: L08 (Memory Coalescing), L09 (Occupancy), L10 (Roofline Model),
L11 (Streams & Concurrency), L12 (CUDA Graphs), L13 (Profiling with Nsight)

---

## Exercise 2.1 — Stride Access Penalty

**Concept introduced in**: L08 (Memory Coalescing)

### Problem Statement

Measure how memory access stride affects effective bandwidth. Write two kernel variants:

- `stride1_read`: reads with stride 1 (fully coalesced).
- `strideS_read`: reads with stride `S` (partially or fully uncoalesced).

Test `S` in `{1, 2, 4, 8, 16, 32}` and print the bandwidth at each stride. Profile with
`ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum` to confirm cache behavior.

### Requirements

- Array size: 64M floats (256 MB).
- Use CUDA events for timing (10 iterations each, average).
- Only count the actual bytes touched: `n / S` unique cache lines for stride S.
- Print a table: `Stride | GB/s | % of stride-1`.

### Starter Code

```cuda
// ex2_1_stride.cu
// Compile: nvcc -O2 -arch=sm_80 -o ex2_1 ex2_1_stride.cu
// Profile: ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum ./ex2_1

#include <cuda_runtime.h>
#include <cstdio>

#define NELEMS (1 << 26)   // 64M floats
#define BLOCK_SIZE 256
#define NRUNS 10

// Read kernel: each thread reads d_in[gid * stride] and writes the sum to d_out.
// The write to d_out is necessary so the compiler doesn't optimize the read away.
__global__ void stride_read(const float* __restrict__ d_in,
                            float* __restrict__       d_out,
                            int n, int stride) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = gid * stride;
    // TODO: if (idx < n) d_out[gid] = d_in[idx];
}

float measure_bw(const float* d_in, float* d_out, int n, int stride) {
    int active_threads = n / stride;  // threads with valid addresses
    int nblocks = (active_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);

    // Warm-up
    stride_read<<<nblocks, BLOCK_SIZE>>>(d_in, d_out, n, stride);

    float total_ms = 0.0f;
    for (int r = 0; r < NRUNS; ++r) {
        cudaEventRecord(s);
        stride_read<<<nblocks, BLOCK_SIZE>>>(d_in, d_out, n, stride);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms, s, e);
        total_ms += ms;
    }

    cudaEventDestroy(s); cudaEventDestroy(e);
    float avg_ms = total_ms / NRUNS;
    // Bytes actually requested (each access is 4 bytes, active_threads accesses)
    double bytes_requested = (double)active_threads * sizeof(float);
    return bytes_requested / (avg_ms * 1e-3) / 1e9;
}

int main() {
    const int n = NELEMS;
    float *d_in, *d_out;
    // TODO: cudaMalloc d_in (size n), d_out (size n/1 worst case)
    // TODO: cudaMemset d_in to 0

    int strides[] = {1, 2, 4, 8, 16, 32};
    float bw_s1 = 0.0f;

    printf("%-8s %-12s %-12s\n", "Stride", "GB/s", "% of S=1");
    for (int s : strides) {
        float bw = measure_bw(d_in, d_out, n, s);
        if (s == 1) bw_s1 = bw;
        printf("%-8d %-12.1f %-12.1f\n", s, bw, 100.0f * bw / bw_s1);
    }

    // TODO: cudaFree d_in, d_out
    return 0;
}
```

### Expected Output (A100 example)

```
Stride   GB/s         % of S=1
1        620.4        100.0
2        310.1        50.0
4        155.0        25.0
8        77.5         12.5
16       38.9         6.3
32       19.5         3.1
```

### Hints

- At stride 32, each warp of 32 threads accesses 32 × 32 × 4 = 4 KB spread across 128 different cache lines — maximum waste.
- Run `ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./ex2_1` to see occupancy alongside the metric.
- The `__restrict__` keyword tells the compiler the pointers don't alias, enabling better load/store scheduling.

### Performance Target

Confirm that stride-32 bandwidth is approximately 1/32 of stride-1 bandwidth, matching the theoretical cache-line waste ratio.

---

## Exercise 2.2 — Occupancy Tuning

**Concept introduced in**: L09 (Occupancy)

### Problem Statement

Use the CUDA occupancy API to automatically find the block size that maximizes occupancy for
your block reduction kernel from Exercise 1.5. Compare the performance of the auto-tuned
block size against block sizes of 64, 128, 256, and 512.

### Requirements

- Use `cudaOccupancyMaxPotentialBlockSize` to query the optimal block size.
- Print the recommended block size and the theoretical occupancy.
- Benchmark each block size with `cudaEventElapsedTime`, 10 iterations.
- Print a table: `BlockSize | OccupancyPct | Time (ms) | Bandwidth (GB/s)`.

### Starter Code

```cuda
// ex2_2_occupancy.cu
// Compile: nvcc -O2 -arch=sm_80 -o ex2_2 ex2_2_occupancy.cu

#include <cuda_runtime.h>
#include <cstdio>

#define N (1 << 24)   // 16M elements
#define NRUNS 10

// Reuse your reduction kernel here (simplified — single-pass using atomics for brevity).
// TODO: paste or implement a block_reduce kernel that writes partial sums.
__global__ void block_reduce(const float* d_in, float* d_partial, int n) {
    // TODO: shared memory reduction (basic version is fine — the point is occupancy tuning)
}

float benchmark(int block_size, const float* d_in, float* d_partial, int n) {
    int nblocks = (n + block_size - 1) / block_size;
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);

    // Warm-up
    cudaMemset(d_partial, 0, nblocks * sizeof(float));
    block_reduce<<<nblocks, block_size>>>(d_in, d_partial, n);

    float total_ms = 0.0f;
    for (int r = 0; r < NRUNS; ++r) {
        cudaMemset(d_partial, 0, nblocks * sizeof(float));
        cudaEventRecord(s);
        block_reduce<<<nblocks, block_size>>>(d_in, d_partial, n);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms, s, e);
        total_ms += ms;
    }
    cudaEventDestroy(s); cudaEventDestroy(e);
    return total_ms / NRUNS;
}

int main() {
    const int n = N;
    const size_t bytes = n * sizeof(float);

    float *d_in, *d_partial;
    // TODO: cudaMalloc and initialize d_in

    // Query optimal block size
    int min_grid_size, opt_block_size;
    // TODO: cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &opt_block_size, block_reduce, 0, 0)
    printf("Recommended block size: %d  (min grid: %d)\n", opt_block_size, min_grid_size);

    // Print occupancy for the recommended block size
    int max_active_blocks;
    // TODO: cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, block_reduce, opt_block_size, 0)
    // TODO: query cudaDeviceProp to get multiProcessorCount and maxThreadsPerMultiProcessor
    // TODO: compute and print occupancy %

    int block_sizes[] = {64, 128, 256, 512, opt_block_size};
    int nb_sizes = 5;

    printf("%-12s %-15s %-12s %-15s\n",
           "BlockSize", "OccupancyPct", "Time(ms)", "Bandwidth(GB/s)");
    for (int i = 0; i < nb_sizes; ++i) {
        int bs = block_sizes[i];
        // TODO: cudaMalloc or resize d_partial for current nblocks
        float ms = benchmark(bs, d_in, d_partial, n);
        double bw = bytes / (ms * 1e-3) / 1e9;

        // TODO: compute occupancy pct for this block size
        printf("%-12d %-15s %-12.3f %-15.1f\n", bs, "?%", ms, bw);
    }

    // TODO: cudaFree
    return 0;
}
```

### Expected Output (A100 example)

```
Recommended block size: 256  (min grid: 108)
BlockSize    OccupancyPct    Time(ms)     Bandwidth(GB/s)
64           25.0            4.21         15.2
128          50.0            2.14         29.9
256          100.0           1.08         59.3
512          100.0           1.10         58.1
256          100.0           1.08         59.3
```

### Hints

- `cudaOccupancyMaxPotentialBlockSize` takes the kernel function pointer, dynamic shared memory size per block, and an optional block size limit.
- `cudaOccupancyMaxActiveBlocksPerMultiprocessor` returns blocks per SM; multiply by `maxThreadsPerBlock / block_size` for thread occupancy, divide by `maxThreadsPerMultiProcessor / block_size` for pct.
- Low occupancy does not always mean slow — memory-bound kernels are less sensitive.

### Performance Target

Confirm that the API-recommended block size is at or near the empirical best. Occupancy should reach 100% for block sizes ≥ 256 on modern GPUs.

---

## Exercise 2.3 — Roofline Chart

**Concept introduced in**: L10 (Roofline Model)

### Problem Statement

Build a roofline chart for your GPU by measuring two kernels:

1. **Memory-bound kernel**: streaming copy (from Ex 1.3).
2. **Compute-bound kernel**: SGEMV (matrix–vector multiply with many FLOPs per byte).

For each kernel, measure:
- Arithmetic intensity (AI) = FLOPs / bytes transferred.
- Achieved GFLOP/s.

Plot (or print) both points against the theoretical roofline:
- Memory bandwidth roofline: `perf = AI * BW_peak`.
- Compute roofline: `perf = FLOP_peak`.

### Requirements

- Measure `BW_peak` using the copy kernel from Ex 1.3.
- For SGEMV: matrix M×K, compute AI = `2*M*K / (bytes read + bytes written)`.
- Print both points and state whether each kernel is bandwidth-bound or compute-bound.

### Starter Code

```cuda
// ex2_3_roofline.cu
// Compile: nvcc -O2 -arch=sm_80 -o ex2_3 ex2_3_roofline.cu

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define COPY_N  (1 << 26)    // 64M floats for bandwidth test
#define SGEMV_M 4096
#define SGEMV_K 4096
#define BLOCK_SIZE 256

// Streaming copy kernel (same as Ex 1.3)
__global__ void mem_copy(const float* __restrict__ src, float* __restrict__ dst, int n) {
    // TODO
}

// SGEMV kernel: y = A * x where A is M x K row-major
// Each thread computes one row of y.
__global__ void sgemv(const float* __restrict__ A,
                      const float* __restrict__ x,
                      float* __restrict__       y,
                      int M, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
            sum += A[row * K + k] * x[k];
        y[row] = sum;
    }
}

float time_ms(auto fn) {
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    fn();  // warm-up
    cudaEventRecord(s);
    for (int i = 0; i < 10; ++i) fn();
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms / 10.0f;
}

int main() {
    // --- Memory bandwidth measurement ---
    const int n_copy = COPY_N;
    float *d_src, *d_dst_copy;
    // TODO: cudaMalloc d_src, d_dst_copy

    float ms_copy = time_ms([&]() {
        int nblocks = (n_copy + BLOCK_SIZE - 1) / BLOCK_SIZE;
        mem_copy<<<nblocks, BLOCK_SIZE>>>(d_src, d_dst_copy, n_copy);
    });
    double bw_peak = (2.0 * n_copy * sizeof(float)) / (ms_copy * 1e-3) / 1e9;
    printf("[Copy]  AI = 0.0 FLOP/byte  BW = %.1f GB/s\n", bw_peak);

    // --- SGEMV measurement ---
    const int M = SGEMV_M, K = SGEMV_K;
    float *d_A, *d_x, *d_y;
    // TODO: cudaMalloc d_A (M*K), d_x (K), d_y (M)

    float ms_sgemv = time_ms([&]() {
        int nblocks = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
        sgemv<<<nblocks, BLOCK_SIZE>>>(d_A, d_x, d_y, M, K);
    });

    double flops      = 2.0 * M * K;       // multiply + add per element
    double bytes_read = sizeof(float) * ((double)M * K + K + M);  // A + x + y
    double ai_sgemv   = flops / bytes_read;
    double gflops     = flops / (ms_sgemv * 1e-3) / 1e9;
    double roofline   = ai_sgemv * bw_peak;  // bandwidth-bound prediction

    printf("[SGEMV] AI = %.2f FLOP/byte  Achieved = %.1f GFLOP/s  Roofline = %.1f GFLOP/s\n",
           ai_sgemv, gflops, roofline);

    // Determine bottleneck
    // TODO: query cudaDeviceProp.clockRate and cudaDeviceProp.multiProcessorCount to estimate FLOP_peak
    //       For now, print whether achieved is close to roofline or far below.
    if (gflops > 0.8 * roofline)
        printf("Kernel is BANDWIDTH-BOUND (close to BW roofline)\n");
    else
        printf("Kernel is COMPUTE-BOUND or LATENCY-BOUND (well below BW roofline)\n");

    // TODO: cudaFree
    return 0;
}
```

### Expected Output (A100 example)

```
[Copy]  AI = 0.0 FLOP/byte  BW = 618.3 GB/s
[SGEMV] AI = 0.50 FLOP/byte  Achieved = 309.2 GFLOP/s  Roofline = 309.2 GFLOP/s
Kernel is BANDWIDTH-BOUND (close to BW roofline)
```

### Hints

- SGEMV is inherently bandwidth-bound (AI ≈ 0.5) unless M >> K.
- To create a compute-bound kernel for contrast, try an SGEMM with large square tiles.
- The roofline intersection (ridge point) = `FLOP_peak / BW_peak`. Kernels to the left are bandwidth-bound.

### Performance Target

SGEMV should land within 10% of the bandwidth roofline. If it's 50% below, your kernel is latency-bound — investigate with `ncu`.

---

## Exercise 2.4 — Async Overlap with Streams

**Concept introduced in**: L11 (Streams & Concurrency)

### Problem Statement

Overlap a compute kernel with a `cudaMemcpyAsync` using two CUDA streams. Split a large
array into two equal halves. While the GPU processes the first half, transfer the second
half from host to device. Measure wall-clock speedup versus the sequential (no overlap)
version.

### Requirements

- Total array: 128M floats (512 MB). Split into two 64M chunks.
- Kernel: `scale_vector` (multiply by a scalar).
- Use `cudaMemcpyAsync` with pinned host memory (`cudaMallocHost`).
- Measure both the sequential and pipelined versions with CUDA events.

### Starter Code

```cuda
// ex2_4_streams.cu
// Compile: nvcc -O2 -arch=sm_80 -o ex2_4 ex2_4_streams.cu

#include <cuda_runtime.h>
#include <cstdio>

#define TOTAL_N (1 << 27)   // 128M floats = 512 MB
#define BLOCK_SIZE 256
#define ALPHA 1.5f

__global__ void scale_vector(const float* in, float* out, float alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = alpha * in[i];
}

int main() {
    const int n      = TOTAL_N;
    const int half_n = n / 2;
    const size_t half_bytes = half_n * sizeof(float);

    // Pinned host memory — required for async transfers
    float *h_in, *h_out;
    cudaMallocHost(&h_in,  n * sizeof(float));
    cudaMallocHost(&h_out, n * sizeof(float));
    for (int i = 0; i < n; ++i) h_in[i] = static_cast<float>(i);

    float *d_in0, *d_out0, *d_in1, *d_out1;
    // TODO: cudaMalloc d_in0, d_out0, d_in1, d_out1 (each of size half_n)

    cudaStream_t stream0, stream1;
    // TODO: cudaStreamCreate(&stream0), cudaStreamCreate(&stream1)

    // --- Sequential baseline ---
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);

    // Half 0
    cudaMemcpy(d_in0, h_in, half_bytes, cudaMemcpyHostToDevice);
    int nblocks = (half_n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    scale_vector<<<nblocks, BLOCK_SIZE>>>(d_in0, d_out0, ALPHA, half_n);
    cudaMemcpy(h_out, d_out0, half_bytes, cudaMemcpyDeviceToHost);

    // Half 1
    cudaMemcpy(d_in1, h_in + half_n, half_bytes, cudaMemcpyHostToDevice);
    scale_vector<<<nblocks, BLOCK_SIZE>>>(d_in1, d_out1, ALPHA, half_n);
    cudaMemcpy(h_out + half_n, d_out1, half_bytes, cudaMemcpyDeviceToHost);

    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms_seq; cudaEventElapsedTime(&ms_seq, s, e);
    printf("Sequential: %.2f ms\n", ms_seq);

    // --- Pipelined (overlapped) version ---
    cudaEventRecord(s);

    // TODO: Issue half0 H2D transfer on stream0
    // TODO: Issue half0 kernel on stream0
    // TODO: Issue half1 H2D transfer on stream1 (overlaps with half0 kernel)
    // TODO: Issue half1 kernel on stream1
    // TODO: Issue half0 D2H transfer on stream0
    // TODO: Issue half1 D2H transfer on stream1
    // TODO: cudaStreamSynchronize both streams

    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms_pipe; cudaEventElapsedTime(&ms_pipe, s, e);
    printf("Pipelined:  %.2f ms  (%.2fx speedup)\n", ms_pipe, ms_seq / ms_pipe);

    // Verify
    bool ok = true;
    for (int i = 0; i < n; ++i)
        if (h_out[i] != ALPHA * h_in[i]) { ok = false; break; }
    printf("Correctness: %s\n", ok ? "PASS" : "FAIL");

    // TODO: cudaFree x4, cudaStreamDestroy x2, cudaFreeHost x2
    cudaEventDestroy(s); cudaEventDestroy(e);
    return 0;
}
```

### Expected Output

```
Sequential: 184.3 ms
Pipelined:  97.1 ms  (1.90x speedup)
Correctness: PASS
```

### Hints

- Pinned (page-locked) memory is required for `cudaMemcpyAsync` to actually overlap with kernel execution.
- The GPU can only overlap H2D transfers with kernel execution if they are on different streams **and** the GPU has a DMA copy engine (verify with `cudaDeviceProp.asyncEngineCount`).
- Check the Nsight Systems timeline to visualize the overlap.

### Performance Target

Overlap should yield at least 1.5× speedup. Near-ideal 2× means the transfer and compute times are perfectly matched. A value below 1.2× suggests the GPU lacks a separate copy engine or the host memory is not pinned.

---

## Exercise 2.5 — CUDA Graph Launch

**Concept introduced in**: L12 (CUDA Graphs)

### Problem Statement

Capture a sequence of kernels (scale → reduce → scale again) into a CUDA graph and measure
the launch overhead versus regular (non-graph) launches. CUDA Graphs amortize the per-kernel
launch overhead (~5–20 µs on CPU) into a single instantiation cost.

### Requirements

- Kernel sequence: `scale_vector` → `block_reduce` (partial sums only) → `scale_vector` on partial sums.
- Measure 1000 sequential launches without graph.
- Measure 1000 graph instantiation replays.
- Print overhead per launch (in µs) for both approaches.

### Starter Code

```cuda
// ex2_5_graph.cu
// Compile: nvcc -O2 -arch=sm_80 -o ex2_5 ex2_5_graph.cu

#include <cuda_runtime.h>
#include <cstdio>

#define N          (1 << 16)   // 64K elements — small so compute is fast, overhead dominates
#define BLOCK_SIZE 256
#define NREPLAYS   1000

__global__ void scale_vector(float* d, float alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] *= alpha;
}

__global__ void block_reduce(const float* in, float* partial, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x, gid = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (gid < n) ? in[gid] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = sdata[0];
}

int main() {
    const int n = N;
    const int nblocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float *d_data, *d_partial;
    cudaMalloc(&d_data,    n       * sizeof(float));
    cudaMalloc(&d_partial, nblocks * sizeof(float));
    cudaMemset(d_data, 1, n * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // --- Baseline: regular launches ---
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);

    cudaEventRecord(s, stream);
    for (int r = 0; r < NREPLAYS; ++r) {
        scale_vector<<<nblocks, BLOCK_SIZE, 0, stream>>>(d_data, 1.0001f, n);
        block_reduce<<<nblocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float), stream>>>(d_data, d_partial, n);
        scale_vector<<<nblocks, BLOCK_SIZE, 0, stream>>>(d_partial, 0.9999f, nblocks);
    }
    cudaEventRecord(e, stream); cudaEventSynchronize(e);
    float ms_regular; cudaEventElapsedTime(&ms_regular, s, e);
    printf("Regular launches: %.2f ms total  %.1f us/replay\n",
           ms_regular, ms_regular * 1000.0f / NREPLAYS);

    // --- CUDA Graph capture ---
    cudaGraph_t     graph;
    cudaGraphExec_t graphExec;

    // TODO: cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal)
    // TODO: issue the same 3 kernels on stream (they are captured, not run)
    // TODO: cudaStreamEndCapture(stream, &graph)
    // TODO: cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0)

    // --- Graph replay ---
    cudaEventRecord(s, stream);
    for (int r = 0; r < NREPLAYS; ++r) {
        // TODO: cudaGraphLaunch(graphExec, stream)
    }
    cudaEventRecord(e, stream); cudaEventSynchronize(e);
    float ms_graph; cudaEventElapsedTime(&ms_graph, s, e);
    printf("Graph launches:   %.2f ms total  %.1f us/replay\n",
           ms_graph, ms_graph * 1000.0f / NREPLAYS);
    printf("Overhead reduction: %.1fx\n", ms_regular / ms_graph);

    // TODO: cudaGraphExecDestroy, cudaGraphDestroy, cudaFree x2, cudaStreamDestroy
    cudaEventDestroy(s); cudaEventDestroy(e);
    return 0;
}
```

### Expected Output (A100 example)

```
Regular launches: 25.3 ms total  25.3 us/replay
Graph launches:   3.1 ms total   3.1 us/replay
Overhead reduction: 8.2x
```

### Hints

| Step | API |
|------|-----|
| Begin capture | `cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal)` |
| End capture | `cudaStreamEndCapture(stream, &graph)` |
| Instantiate | `cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0)` |
| Launch | `cudaGraphLaunch(exec, stream)` |
| Cleanup | `cudaGraphExecDestroy(exec)` then `cudaGraphDestroy(graph)` |

- Graphs are most beneficial when kernel execution time is short (< 50 µs) and launch count is large.
- You can inspect the graph structure with `cudaGraphDebugDotPrint`.

### Performance Target

Graph replay overhead should be < 5 µs per replay. Regular launch overhead is typically 10–20 µs per kernel on the CPU side.
