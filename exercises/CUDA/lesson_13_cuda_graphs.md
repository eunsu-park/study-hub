# Lesson 13 — CUDA Graphs (per-lesson exercise)

Prerequisites: L12 (streams).

Compile: `nvcc -O3 -arch=sm_80 ex.cu -o ex`

CUDA graphs let you record a sequence of operations (kernels, copies, library calls) once and replay them with one driver call. For workloads that launch many short kernels, this saves the per-launch CPU overhead — typically 5-50 µs per launch — which can dominate when kernels themselves are < 100 µs.

---

## Exercise 13.1 — Stream-Capture Graph

**Difficulty**: ★★★

### Problem

Capture a sequence of kernel launches into a graph using stream capture, then replay the graph in a tight loop:

```cuda
#include <cstdio>
#include <cuda_runtime.h>

__global__ void noop(float *x) { *x += 1.0f; }

int main(void) {
    float *d;
    cudaMalloc(&d, sizeof(float));
    cudaMemset(d, 0, sizeof(float));

    cudaStream_t s;
    cudaStreamCreate(&s);

    /* Capture phase */
    cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
    for (int i = 0; i < 100; i++)
        noop<<<1, 1, 0, s>>>(d);
    cudaGraph_t graph;
    cudaStreamEndCapture(s, &graph);

    /* Instantiate (compile) */
    cudaGraphExec_t inst;
    cudaGraphInstantiate(&inst, graph, nullptr, nullptr, 0);

    /* Replay */
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int N_REPLAY = 10000;

    cudaEventRecord(e0);
    for (int r = 0; r < N_REPLAY; r++) cudaGraphLaunch(inst, s);
    cudaStreamSynchronize(s);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float graph_ms = 0; cudaEventElapsedTime(&graph_ms, e0, e1);

    /* Compare to direct launches */
    cudaEventRecord(e0);
    for (int r = 0; r < N_REPLAY; r++)
        for (int i = 0; i < 100; i++) noop<<<1, 1, 0, s>>>(d);
    cudaStreamSynchronize(s);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float direct_ms = 0; cudaEventElapsedTime(&direct_ms, e0, e1);

    printf("graph:  %.2f ms\n", graph_ms);
    printf("direct: %.2f ms (%.2fx slower)\n", direct_ms, direct_ms / graph_ms);

    cudaGraphExecDestroy(inst); cudaGraphDestroy(graph);
    cudaStreamDestroy(s); cudaFree(d);
    return 0;
}
```

For 100 trivial kernels per iteration × 10000 iterations, the graph version typically beats direct launches by 5-20× — because each direct launch costs ~5 µs of CPU work, while a graph replay is one driver call.

---

## Exercise 13.2 — When Graphs Help (and When They Do Not)

**Difficulty**: ★★

The break-even point between direct and graph launches depends on per-kernel duration:

| Kernel duration | Direct overhead share | Graph win |
|-----------------|----------------------|-----------|
| 1 µs | 80%+ | 4-5× |
| 10 µs | 30% | 1.5-2× |
| 100 µs | 5% | 1.05× |
| 1 ms | <1% | barely measurable |

Time a varying-size vector add (`N` from $10^4$ to $10^7$) inside both a direct loop and a graph. The graph win shrinks as $N$ grows because each launch's compute work outweighs the per-launch overhead.

Take-home: graphs are pure win for tight inner loops of small kernels (e.g., per-step physics integrators, per-token decode); they are unnecessary for large bulk kernels.

---

## Exercise 13.3 — Conditional and Update — Bonus

**Difficulty**: ★★★

A captured graph is immutable, but `cudaGraphExecUpdate` lets you change a kernel's parameters without re-capturing. Use this to vary the launch grid size of a kernel inside the graph between iterations — useful when the workload is data-dependent but the GRAPH STRUCTURE stays the same.

For decisions that change graph structure (different kernels run in different cases), re-capture is the only option. The graph framework is best for fixed-shape pipelines.
