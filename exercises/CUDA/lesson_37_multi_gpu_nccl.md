# Lesson 37 — Multi-GPU with NCCL (per-lesson exercise)

Prerequisites: L12 (streams), L14 (reduction), basic MPI or multi-process familiarity.

Compile: `nvcc -O3 ex.cu -lnccl -o ex` (requires NCCL installed).

NCCL implements collective communication primitives (AllReduce, AllGather, ReduceScatter, Broadcast) over NVLink or InfiniBand with topology-aware routing. For data-parallel training, the bandwidth ceiling of these collectives sets how fast you can scale.

This exercise does NOT require a multi-GPU machine for Exercise 37.1 — NCCL supports a single-process multi-device setup. Exercise 37.2 and 37.3 assume ≥2 GPUs; skip if you only have one.

---

## Exercise 37.1 — Single-Process NCCL AllReduce

**Difficulty**: ★★

### Problem

On a single node with `N` GPUs, implement a sum AllReduce across `N` float vectors of length `M`. Each GPU holds one vector; after the AllReduce, every GPU holds the element-wise sum.

### Starter

```cuda
#include <cstdio>
#include <cuda_runtime.h>
#include <nccl.h>

#define NCCL_CHECK(call) do {                                               \
    ncclResult_t r = (call);                                                \
    if (r != ncclSuccess) {                                                 \
        fprintf(stderr, "NCCL error: %s\n", ncclGetErrorString(r));         \
        std::exit(1);                                                       \
    }                                                                       \
} while (0)

int main(void) {
    int n_dev = 0;
    cudaGetDeviceCount(&n_dev);
    if (n_dev < 1) { printf("no CUDA devices\n"); return 1; }
    if (n_dev > 4) n_dev = 4;   // cap for the exercise

    const int M = 1024;

    // 1. Allocate on each device
    float **h_bufs = new float*[n_dev];
    float **d_bufs = new float*[n_dev];
    for (int i = 0; i < n_dev; i++) {
        cudaSetDevice(i);
        cudaMalloc(&d_bufs[i], M * sizeof(float));
        h_bufs[i] = new float[M];
        for (int k = 0; k < M; k++) h_bufs[i][k] = (float)(i + 1);   // device i → i+1
        cudaMemcpy(d_bufs[i], h_bufs[i], M * sizeof(float), cudaMemcpyHostToDevice);
    }

    // 2. Bootstrap NCCL communicators
    ncclComm_t *comms = new ncclComm_t[n_dev];
    int *devs = new int[n_dev];
    for (int i = 0; i < n_dev; i++) devs[i] = i;
    NCCL_CHECK(ncclCommInitAll(comms, n_dev, devs));

    // 3. AllReduce (sum)
    NCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < n_dev; i++) {
        cudaSetDevice(i);
        NCCL_CHECK(ncclAllReduce(d_bufs[i], d_bufs[i], M, ncclFloat, ncclSum, comms[i], /*stream*/ 0));
    }
    NCCL_CHECK(ncclGroupEnd());

    // 4. Sync all devices
    for (int i = 0; i < n_dev; i++) { cudaSetDevice(i); cudaDeviceSynchronize(); }

    // 5. Verify — each entry should be 1 + 2 + ... + n_dev
    cudaSetDevice(0);
    cudaMemcpy(h_bufs[0], d_bufs[0], M * sizeof(float), cudaMemcpyDeviceToHost);
    float expected = 0;
    for (int i = 1; i <= n_dev; i++) expected += i;
    printf("AllReduce result on dev 0: h[0] = %.1f (expected %.1f)\n", h_bufs[0][0], expected);

    // 6. Cleanup
    for (int i = 0; i < n_dev; i++) {
        cudaSetDevice(i); cudaFree(d_bufs[i]); delete[] h_bufs[i];
        ncclCommDestroy(comms[i]);
    }
    delete[] d_bufs; delete[] h_bufs; delete[] comms; delete[] devs;
    return 0;
}
```

---

## Exercise 37.2 — Bandwidth Benchmark

**Difficulty**: ★★★

Extend 37.1 to measure the AllReduce bandwidth. Time a loop of 100 iterations for each of these buffer sizes: 1 MiB, 16 MiB, 256 MiB. Bandwidth is `message_size / mean_time_per_iter` and is typically reported as algorithmic bandwidth (what the user sees) vs. bus bandwidth (what hardware measures).

For AllReduce, each rank sends and receives `(N-1)/N * message_size` of data. The algorithmic bandwidth should approach NVLink theoretical max (e.g., 600 GB/s on an A100 SXM) for large messages; small messages are latency-bound and bandwidth collapses.

---

## Exercise 37.3 — ReduceScatter + AllGather vs. AllReduce — Bonus

**Difficulty**: ★★★★

AllReduce can be implemented as `ReduceScatter + AllGather`. Benchmark both implementations on your hardware. The fused AllReduce is usually the same speed as the two-step version because NCCL already implements AllReduce as a ring algorithm that is internally ReduceScatter + AllGather; show this empirically and explain what you observe.

---

## Exercise 37.4 — Gradient Averaging Pattern

**Difficulty**: ★★★

In a data-parallel training step, each rank computes gradients for its mini-batch and then AllReduces them to get the mean. Write a minimal host-side wrapper `nccl_average_inplace(float *d_grad, size_t n, ncclComm_t comm, cudaStream_t s)` that performs AllReduce with `ncclSum` followed by an in-place scale-by-1/world_size. Keep the operation asynchronous on the provided stream.
