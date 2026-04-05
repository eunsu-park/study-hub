/*
 * allreduce_demo.cu — Lesson 37: Multi-GPU and NCCL
 *
 * Demonstrates multi-GPU collective communication using NCCL:
 *   - Initialise NCCL communicators for all available GPUs
 *   - ncclAllReduce (ring-allreduce, sum) across GPUs
 *   - Peer memory access (cudaDeviceEnablePeerAccess) check
 *   - Bandwidth measurement
 *
 * Requirements:
 *   - NCCL library installed (-lnccl)
 *   - At least 2 CUDA GPUs (will skip gracefully with 1 GPU)
 *
 * Build:  nvcc -O2 -arch=sm_80 allreduce_demo.cu -o allreduce_demo -lnccl
 * Run:    ./allreduce_demo
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <nccl.h>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e)); exit(1); } } while(0)
#define NCCL_CHECK(x) do { ncclResult_t r=(x); if(r!=ncclSuccess){ \
    fprintf(stderr,"NCCL %s\n",ncclGetErrorString(r)); exit(1); } } while(0)

static const int COUNT = 1 << 22;   // 16 M floats per GPU
static const int ITERS = 10;

int main(void) {
    int n_gpus;
    CUDA_CHECK(cudaGetDeviceCount(&n_gpus));
    printf("Detected %d GPU(s)\n", n_gpus);

    if (n_gpus < 2) {
        printf("Need ≥2 GPUs for NCCL allreduce demo. Single-GPU NOP.\n");
        // Still demonstrates single-GPU NCCL init and self-allreduce
        n_gpus = 1;
    }

    // Allocate per-GPU resources
    float     **d_data  = (float    **)malloc(n_gpus * sizeof(float *));
    float     **h_data  = (float    **)malloc(n_gpus * sizeof(float *));
    cudaStream_t *streams = (cudaStream_t*)malloc(n_gpus * sizeof(cudaStream_t));

    for (int g = 0; g < n_gpus; g++) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaMalloc(&d_data[g], COUNT * sizeof(float)));
        h_data[g] = (float *)malloc(COUNT * sizeof(float));
        for (int i = 0; i < COUNT; i++) h_data[g][i] = (float)(g + 1);
        CUDA_CHECK(cudaMemcpy(d_data[g], h_data[g], COUNT * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaStreamCreate(&streams[g]));
    }

    // Enable peer access between all GPU pairs
    for (int i = 0; i < n_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        for (int j = 0; j < n_gpus; j++) {
            if (i == j) continue;
            int can_access;
            cudaDeviceCanAccessPeer(&can_access, i, j);
            if (can_access) cudaDeviceEnablePeerAccess(j, 0);
        }
    }

    // Initialise NCCL communicators
    ncclComm_t *comms = (ncclComm_t*)malloc(n_gpus * sizeof(ncclComm_t));
    int *dev_ids = (int *)malloc(n_gpus * sizeof(int));
    for (int g = 0; g < n_gpus; g++) dev_ids[g] = g;
    NCCL_CHECK(ncclCommInitAll(comms, n_gpus, dev_ids));

    // ── Allreduce benchmark ────────────────────────────────────────────────────
    // Reset data
    for (int g = 0; g < n_gpus; g++) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaMemcpy(d_data[g], h_data[g], COUNT * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    // Warmup
    NCCL_CHECK(ncclGroupStart());
    for (int g = 0; g < n_gpus; g++) {
        CUDA_CHECK(cudaSetDevice(g));
        NCCL_CHECK(ncclAllReduce(d_data[g], d_data[g], COUNT,
                                  ncclFloat, ncclSum, comms[g], streams[g]));
    }
    NCCL_CHECK(ncclGroupEnd());
    for (int g = 0; g < n_gpus; g++) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaStreamSynchronize(streams[g]));
    }

    // Timed runs
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaSetDevice(0));
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0, streams[0]);

    for (int it = 0; it < ITERS; it++) {
        NCCL_CHECK(ncclGroupStart());
        for (int g = 0; g < n_gpus; g++) {
            CUDA_CHECK(cudaSetDevice(g));
            NCCL_CHECK(ncclAllReduce(d_data[g], d_data[g], COUNT,
                                      ncclFloat, ncclSum, comms[g], streams[g]));
        }
        NCCL_CHECK(ncclGroupEnd());
    }

    for (int g = 0; g < n_gpus; g++) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaStreamSynchronize(streams[g]));
    }
    CUDA_CHECK(cudaSetDevice(0));
    cudaEventRecord(t1, streams[0]);
    cudaEventSynchronize(t1);

    float ms; cudaEventElapsedTime(&ms, t0, t1);
    ms /= ITERS;
    double bytes = (double)COUNT * sizeof(float);
    double algbw = bytes / (ms * 1e-3) / 1e9;   // algorithm bandwidth
    double busbw = algbw * 2.0 * (n_gpus - 1) / n_gpus;  // bus bandwidth

    // Verify: each GPU should have sum = 1+2+...+n_gpus
    CUDA_CHECK(cudaSetDevice(0));
    float *h_result = (float *)malloc(COUNT * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_result, d_data[0], COUNT * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float expected = (float)(n_gpus * (n_gpus + 1) / 2);
    bool ok = (fabsf(h_result[0] - expected) < 0.5f);
    free(h_result);

    printf("AllReduce (%d GPUs, %.0f MB)\n",
           n_gpus, bytes / 1e6);
    printf("  %.2f ms  AlgBW=%.1f GB/s  BusBW=%.1f GB/s  %s\n",
           ms, algbw, busbw, ok ? "PASS" : "FAIL");

    // Cleanup
    for (int g = 0; g < n_gpus; g++) {
        ncclCommDestroy(comms[g]);
        cudaSetDevice(g);
        cudaFree(d_data[g]);
        free(h_data[g]);
        cudaStreamDestroy(streams[g]);
    }
    free(comms); free(dev_ids);
    free(d_data); free(h_data); free(streams);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return 0;
}
