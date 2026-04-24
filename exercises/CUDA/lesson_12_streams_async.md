# Lesson 12 — CUDA Streams and Async Operations (per-lesson exercise)

Prerequisites: L04 (memory model), L09 (occupancy).

Compile: `nvcc -O3 -arch=sm_80 ex.cu -o ex`

A CUDA stream is a queue of operations that execute in order on the device. Operations in **different** streams can overlap. The single biggest performance win on CPU↔GPU bound workloads is overlapping a kernel with the next batch's `cudaMemcpyAsync`.

---

## Exercise 12.1 — Sequential vs. Streamed Pipeline

**Difficulty**: ★★

### Problem

Simulate a typical inference pipeline: copy batch from host → run kernel → copy result back. Measure two implementations:

- **Sequential**: do all three steps for batch $i$ before starting batch $i+1$.
- **Pipelined**: split batches into chunks; while chunk $i$ is computing, copy chunk $i+1$ in and chunk $i-1$ out, all on different streams.

### Starter

```cuda
#include <cstdio>
#include <cuda_runtime.h>

__global__ void busy_kernel(float *data, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = data[i];
        for (int k = 0; k < iters; k++) x = x * 1.0001f + 0.5f;
        data[i] = x;
    }
}

int main(void) {
    const int N        = 1 << 20;            /* 1M floats per chunk */
    const int CHUNKS   = 8;
    const int N_TOTAL  = N * CHUNKS;
    const int ITERS    = 1000;

    float *h_in  = nullptr;
    float *h_out = nullptr;
    cudaMallocHost(&h_in,  N_TOTAL * sizeof(float));   /* PINNED memory needed for async copy */
    cudaMallocHost(&h_out, N_TOTAL * sizeof(float));
    for (int i = 0; i < N_TOTAL; i++) h_in[i] = 1.0f;

    float *d_buf;
    cudaMalloc(&d_buf, N * sizeof(float) * 2);          /* double-buffered */

    /* SEQUENTIAL */
    cudaEvent_t s0, s1;
    cudaEventCreate(&s0); cudaEventCreate(&s1);
    cudaEventRecord(s0);
    for (int c = 0; c < CHUNKS; c++) {
        cudaMemcpy(d_buf, h_in + c * N, N * sizeof(float), cudaMemcpyHostToDevice);
        busy_kernel<<<(N + 255) / 256, 256>>>(d_buf, N, ITERS);
        cudaMemcpy(h_out + c * N, d_buf, N * sizeof(float), cudaMemcpyDeviceToHost);
    }
    cudaEventRecord(s1); cudaEventSynchronize(s1);
    float seq_ms = 0; cudaEventElapsedTime(&seq_ms, s0, s1);
    printf("sequential: %.2f ms\n", seq_ms);

    /* PIPELINED with two streams */
    cudaStream_t streams[2];
    cudaStreamCreate(&streams[0]); cudaStreamCreate(&streams[1]);
    cudaEventRecord(s0);
    for (int c = 0; c < CHUNKS; c++) {
        int s = c % 2;
        float *d_chunk = d_buf + s * N;
        cudaMemcpyAsync(d_chunk, h_in + c * N, N * sizeof(float),
                        cudaMemcpyHostToDevice, streams[s]);
        busy_kernel<<<(N + 255) / 256, 256, 0, streams[s]>>>(d_chunk, N, ITERS);
        cudaMemcpyAsync(h_out + c * N, d_chunk, N * sizeof(float),
                        cudaMemcpyDeviceToHost, streams[s]);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(s1); cudaEventSynchronize(s1);
    float pipe_ms = 0; cudaEventElapsedTime(&pipe_ms, s0, s1);
    printf("pipelined : %.2f ms (%.2fx)\n", pipe_ms, seq_ms / pipe_ms);

    cudaFreeHost(h_in); cudaFreeHost(h_out);
    cudaFree(d_buf);
    cudaStreamDestroy(streams[0]); cudaStreamDestroy(streams[1]);
    return 0;
}
```

Expected: 1.5–2× speedup. The exact ratio depends on your kernel:copy time ratio — the closer they are, the better the pipelining wins.

---

## Exercise 12.2 — Why Pinned Memory Matters

**Difficulty**: ★★

Replace `cudaMallocHost` with plain `malloc` for `h_in` and `h_out`. The async copy then **silently** falls back to synchronous behavior because the kernel cannot DMA from pageable memory. Re-time and observe that the speedup vanishes.

This is the single most common reason "I added streams and it did not help" — the host buffers were not pinned.

---

## Exercise 12.3 — Stream Priorities — Bonus

**Difficulty**: ★★★

Some workloads have a latency-critical kernel mixed with bulk batch work. Create two streams with different priorities:

```cuda
int low, high;
cudaDeviceGetStreamPriorityRange(&low, &high);

cudaStream_t s_bulk, s_realtime;
cudaStreamCreateWithPriority(&s_bulk,    cudaStreamNonBlocking, low);
cudaStreamCreateWithPriority(&s_realtime, cudaStreamNonBlocking, high);
```

Submit a steady stream of 100 large kernels to `s_bulk`, then submit one tiny kernel to `s_realtime` mid-stream. Measure when the small kernel completes — it should preempt the bulk work and run almost immediately, even though the bulk submission was earlier.

Use this for inference servers that need to interleave a small "control" path with high-throughput batch processing.
