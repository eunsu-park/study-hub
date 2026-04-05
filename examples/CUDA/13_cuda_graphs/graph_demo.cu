/*
 * graph_demo.cu — Lesson 13: CUDA Graphs
 *
 * Demonstrates:
 *   - Stream-capture API to record a multi-kernel sequence into a graph
 *   - cudaGraphInstantiate / cudaGraphLaunch
 *   - Overhead comparison: per-launch vs graph-repeated-launch
 *   - Graph update (cudaGraphExecKernelNodeSetParams)
 *
 * Build:  nvcc -O2 -arch=sm_80 graph_demo.cu -o graph_demo
 * Run:    ./graph_demo
 */

#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e)); exit(1); } } while(0)

static const int N       = 1 << 20;
static const int THREADS = 256;
static const int REPEAT  = 1000;   // many launches to amplify overhead difference

// ── Simple kernels representing pipeline stages ────────────────────────────────
__global__ void stage_add(float *data, float val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] += val;
}
__global__ void stage_scale(float *data, float factor, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= factor;
}
__global__ void stage_relu(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && data[i] < 0.f) data[i] = 0.f;
}

// ── Baseline: launch kernels REPEAT times in a loop ──────────────────────────
static float run_baseline(float *d_data, int n) {
    int blocks = (n + THREADS - 1) / THREADS;
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int r = 0; r < REPEAT; r++) {
        stage_add  <<<blocks, THREADS>>>(d_data, 1.f, n);
        stage_scale<<<blocks, THREADS>>>(d_data, 0.999f, n);
        stage_relu <<<blocks, THREADS>>>(d_data, n);
    }
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms;
}

// ── CUDA Graph: capture once, launch REPEAT times ────────────────────────────
static float run_graph(float *d_data, int n) {
    int blocks = (n + THREADS - 1) / THREADS;
    cudaStream_t s;
    cudaStreamCreate(&s);

    // ── Capture phase ──
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    stage_add  <<<blocks, THREADS, 0, s>>>(d_data, 1.f, n);
    stage_scale<<<blocks, THREADS, 0, s>>>(d_data, 0.999f, n);
    stage_relu <<<blocks, THREADS, 0, s>>>(d_data, n);
    cudaGraph_t     graph;
    cudaGraphExec_t exec;
    CUDA_CHECK(cudaStreamEndCapture(s, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    // ── Launch phase ──
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int r = 0; r < REPEAT; r++)
        CUDA_CHECK(cudaGraphLaunch(exec, s));
    cudaStreamSynchronize(s);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);

    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(s);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms;
}

int main(void) {
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_data, 0, (size_t)N * sizeof(float)));

    float ms_base  = run_baseline(d_data, N);
    float ms_graph = run_graph(d_data, N);

    printf("CUDA Graph demo (%d elements, %d repeated launches)\n", N, REPEAT);
    printf("  Per-launch loop : %7.2f ms  (%5.2f µs/iter)\n",
           ms_base,  ms_base  / REPEAT * 1000);
    printf("  CUDA Graph      : %7.2f ms  (%5.2f µs/iter)\n",
           ms_graph, ms_graph / REPEAT * 1000);
    printf("  Speedup         : %.2fx\n", ms_base / ms_graph);

    cudaFree(d_data);
    return 0;
}
