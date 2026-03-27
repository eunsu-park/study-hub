# 44. Parallel Inference

**Previous**: [GGUF Format and Loading](./43_GGUF_and_Loading.md) | **Next**: [Capstone: Inference Engine](./45_Capstone_Inference_Engine.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Parallelize matrix multiplication across CPU cores using OpenMP and POSIX threads
2. Explain why single-token LLM inference is memory-bandwidth-bound rather than compute-bound
3. Apply the roofline model to predict achievable throughput for CPU inference
4. Implement attention head parallelism to distribute multi-head attention across threads
5. Measure memory bandwidth and use it to predict maximum token generation speed

---

## 1. Why Parallelism Matters — and When It Doesn't

Single-token LLM inference (batch=1) is dominated by weight matrix reads. For a linear layer with weight matrix `[out, in]`:

```
FLOPs:    2 * out * in          (one multiply-add per weight)
Bytes:    out * in * bytes_per_weight  (read each weight once)

Arithmetic intensity = FLOPs / Bytes
  For INT8:  2 / 1 = 2.0 FLOP/byte
  For F16:   2 / 2 = 1.0 FLOP/byte
  For F32:   2 / 4 = 0.5 FLOP/byte
```

A typical x86 CPU:
- Peak FP32 GFLOP/s: 40-200 (depending on AVX-512 and clock)
- Peak memory bandwidth: 50-100 GB/s (DDR5)
- Ridge point: 1-4 FLOP/byte

Since single-token inference has intensity < 2 FLOP/byte, it sits left of the ridge point on the roofline — **memory-bandwidth bound**. Adding more compute cores helps only linearly with bandwidth, not with FLOP/s.

---

## 2. Roofline Analysis for CPU Inference

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// Roofline model: achievable GFLOP/s = min(peak_flops, bw * intensity)
void roofline_analysis(void) {
    const double peak_flops_gflops = 80.0;   // typical modern desktop CPU
    const double bandwidth_GBs     = 60.0;   // DDR5 dual-channel

    printf("=== Roofline Analysis: Single-Token LLM Inference ===\n\n");
    printf("%-20s %8s %10s %12s %12s\n",
           "Operation", "FLOP/B", "Peak (GFLOP)", "BW-limit", "Achieved");

    struct { const char *name; double ai; double gflops; } ops[] = {
        // ai = arithmetic intensity (FLOP/byte), gflops = actual compute need
        { "F32 matmul (1×N×M)",  0.5,  2.0  },
        { "F16 matmul (1×N×M)",  1.0,  2.0  },
        { "INT8 matmul (1×N×M)", 2.0,  2.0  },
        { "INT4 matmul (1×N×M)", 4.0,  2.0  },
        { "Softmax (T=2048)",    0.1,  0.05 },
        { "RMSNorm",             0.1,  0.01 },
    };

    for (int i = 0; i < 6; i++) {
        double ai      = ops[i].ai;
        double gflops  = ops[i].gflops;
        double bw_lim  = bandwidth_GBs * ai;       // bandwidth ceiling at this AI
        double achieved = fmin(peak_flops_gflops, bw_lim);
        const char *bound = (bw_lim < peak_flops_gflops) ? "BW-bound" : "Compute";
        printf("%-20s %8.1f %10.2f %10.2f %10.2f (%s)\n",
               ops[i].name, ai, peak_flops_gflops, bw_lim, achieved, bound);
    }

    printf("\nConclusion: all ops at batch=1 are memory-bandwidth bound.\n");
    printf("Max token rate ≈ bandwidth / bytes_per_token\n");

    // For a 7B INT8 model: ~7GB weights, ~14 GFLOP per token
    double model_bytes  = 7e9;
    double flops_token  = 14e9;
    double ai_model     = flops_token / model_bytes;
    double tok_per_sec  = bandwidth_GBs * 1e9 / model_bytes;
    printf("7B INT8 model: AI=%.2f, theoretical max = %.1f tokens/sec\n",
           ai_model, tok_per_sec);
}
```

---

## 3. Baseline Matmul (Single-Threaded)

```c
// Dense matmul: out[M, N] = input[M, K] @ weight[N, K]^T  (weight is transposed)
// weight stored as [N, K] row-major (each row is one output neuron's weights)
void matmul_st(float *out, const float *input, const float *weight,
               int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++)
                acc += input[m*K + k] * weight[n*K + k];
            out[m*N + n] = acc;
        }
    }
}
```

---

## 4. OpenMP Parallelism

OpenMP distributes the outer loop over output rows across threads. This is safe because each output row is independent.

```c
// OpenMP parallel matmul: parallelize over output rows (m)
void matmul_omp(float *out, const float *input, const float *weight,
                int M, int N, int K) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++)
                acc += input[m*K + k] * weight[n*K + k];
            out[m*N + n] = acc;
        }
    }
}

// For single-token inference, M=1 so parallelizing over m gives no benefit.
// Instead, parallelize over output neurons (n):
void matmul_omp_single_token(float *out, const float *input, const float *weight,
                              int N, int K) {
    // M=1 case: input[K], weight[N, K], out[N]
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int n = 0; n < N; n++) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++)
            acc += input[k] * weight[n*K + k];
        out[n] = acc;
    }
}
```

Compile with: `gcc -O3 -march=native -fopenmp matmul.c -o matmul -lm`

For single-token inference, thread overhead is significant for small N/K. Use `OMP_NUM_THREADS` to tune. For typical LLM dimensions (N=4096, K=4096), 4-8 threads gives near-linear speedup.

---

## 5. POSIX Threads Alternative

For fine-grained control without OpenMP overhead:

```c
typedef struct {
    const float *input;
    const float *weight;
    float       *out;
    int          N, K;
    int          row_start, row_end;  // range of output neurons this thread handles
} MatmulArgs;

void *matmul_thread(void *arg) {
    MatmulArgs *a = (MatmulArgs *)arg;
    for (int n = a->row_start; n < a->row_end; n++) {
        float acc = 0.0f;
        for (int k = 0; k < a->K; k++)
            acc += a->input[k] * a->weight[n * a->K + k];
        a->out[n] = acc;
    }
    return NULL;
}

// Parallel matmul using POSIX threads (single-token: M=1)
void matmul_pthread(float *out, const float *input, const float *weight,
                    int N, int K, int n_threads) {
    pthread_t   *threads = malloc(n_threads * sizeof(pthread_t));
    MatmulArgs  *args    = malloc(n_threads * sizeof(MatmulArgs));

    int rows_per_thread = (N + n_threads - 1) / n_threads;
    for (int t = 0; t < n_threads; t++) {
        args[t].input     = input;
        args[t].weight    = weight;
        args[t].out       = out;
        args[t].N         = N;
        args[t].K         = K;
        args[t].row_start = t * rows_per_thread;
        args[t].row_end   = args[t].row_start + rows_per_thread;
        if (args[t].row_end > N) args[t].row_end = N;
        pthread_create(&threads[t], NULL, matmul_thread, &args[t]);
    }
    for (int t = 0; t < n_threads; t++)
        pthread_join(threads[t], NULL);

    free(threads); free(args);
}
```

---

## 6. Memory Bandwidth Measurement

```c
// Measure sustained memory bandwidth: read a large array sequentially
double measure_bandwidth_GBs(size_t array_bytes) {
    float *buf = malloc(array_bytes);
    for (size_t i = 0; i < array_bytes / sizeof(float); i++)
        buf[i] = (float)i;

    // Warm up the cache
    volatile float sink = 0.0f;
    for (size_t i = 0; i < array_bytes / sizeof(float); i += 16)
        sink += buf[i];

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    // Read the array 4 times (avoid any caching effects)
    for (int rep = 0; rep < 4; rep++) {
        for (size_t i = 0; i < array_bytes / sizeof(float); i += 8)
            sink += buf[i];
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    double bytes_read = 4.0 * array_bytes;
    double bw = bytes_read / elapsed / 1e9;

    free(buf);
    (void)sink;
    return bw;
}

void benchmark_bandwidth(void) {
    size_t sizes[] = { 256*1024, 4*1024*1024, 64*1024*1024, 512*1024*1024 };
    const char *labels[] = { "256KB (L2)", "4MB (L3)", "64MB (DRAM)", "512MB (DRAM)" };
    printf("=== Memory Bandwidth ===\n");
    for (int i = 0; i < 4; i++) {
        double bw = measure_bandwidth_GBs(sizes[i]);
        printf("  %-20s %.1f GB/s\n", labels[i], bw);
    }
}
```

---

## 7. Tensor Parallelism: Splitting Attention Heads Across Threads

For multi-head attention, each head is completely independent in the forward pass. This is an ideal parallelism target:

```c
// Attention head parallel computation
// Each thread handles a subset of the h attention heads
typedef struct {
    const float *Q;         // [T, h, d_k] row-major
    const float *K;         // [T, h, d_k]
    const float *V;         // [T, h, d_k]
    float       *out;       // [T, h, d_k]
    int          T, h, d_k;
    int          head_start, head_end;
} AttnHeadArgs;

// Single-head attention (standalone, no thread concerns)
static void single_head_attn(float *out_h, const float *Q_h, const float *K_h,
                              const float *V_h, int T, int d_k) {
    float scale = 1.0f / sqrtf((float)d_k);
    float *scores = malloc(T * sizeof(float));
    float *attn   = malloc(T * sizeof(float));

    // For the last query position (inference: one new token)
    // Q_h: [d_k], K_h: [T, d_k], V_h: [T, d_k]
    for (int j = 0; j < T; j++) {
        float dot = 0.0f;
        for (int k = 0; k < d_k; k++)
            dot += Q_h[k] * K_h[j*d_k + k];
        scores[j] = dot * scale;
    }

    // Softmax over T keys
    float max_s = scores[0];
    for (int j = 1; j < T; j++) if (scores[j] > max_s) max_s = scores[j];
    float sum = 0.0f;
    for (int j = 0; j < T; j++) { attn[j] = expf(scores[j] - max_s); sum += attn[j]; }
    for (int j = 0; j < T; j++) attn[j] /= sum;

    // Output: weighted sum of V
    for (int k = 0; k < d_k; k++) {
        float acc = 0.0f;
        for (int j = 0; j < T; j++) acc += attn[j] * V_h[j*d_k + k];
        out_h[k] = acc;
    }
    free(scores); free(attn);
}

void *attn_head_thread(void *arg) {
    AttnHeadArgs *a = (AttnHeadArgs *)arg;
    int d_k = a->d_k;
    int T   = a->T;
    for (int hi = a->head_start; hi < a->head_end; hi++) {
        const float *Q_h   = a->Q   + hi * d_k;  // query for this head (last token)
        const float *K_h   = a->K   + hi * d_k;  // K for this head [T, d_k]
        const float *V_h   = a->V   + hi * d_k;  // V for this head [T, d_k]
        float       *out_h = a->out + hi * d_k;
        single_head_attn(out_h, Q_h, K_h, V_h, T, d_k);
    }
    return NULL;
}

void parallel_multihead_attn(float *out,
                              const float *Q, const float *K, const float *V,
                              int T, int n_heads, int d_k,
                              int n_threads) {
    pthread_t    *threads = malloc(n_threads * sizeof(pthread_t));
    AttnHeadArgs *args    = malloc(n_threads * sizeof(AttnHeadArgs));

    int heads_per_thread = (n_heads + n_threads - 1) / n_threads;
    for (int t = 0; t < n_threads; t++) {
        args[t].Q          = Q;
        args[t].K          = K;
        args[t].V          = V;
        args[t].out        = out;
        args[t].T          = T;
        args[t].h          = n_heads;
        args[t].d_k        = d_k;
        args[t].head_start = t * heads_per_thread;
        args[t].head_end   = args[t].head_start + heads_per_thread;
        if (args[t].head_end > n_heads) args[t].head_end = n_heads;
        pthread_create(&threads[t], NULL, attn_head_thread, &args[t]);
    }
    for (int t = 0; t < n_threads; t++) pthread_join(threads[t], NULL);
    free(threads); free(args);
}
```

---

## 8. Performance Scaling Benchmark

```c
double now_sec_mt(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void benchmark_matmul_scaling(void) {
    // Typical LLM FFN dimensions for a 7B model
    const int N = 4096, K = 4096;  // weight [N, K]
    float *input  = malloc(K * sizeof(float));
    float *weight = malloc((size_t)N * K * sizeof(float));
    float *out    = malloc(N * sizeof(float));

    srand(42);
    for (int i = 0; i < K; i++) input[i] = (float)rand()/RAND_MAX;
    for (size_t i = 0; i < (size_t)N*K; i++) weight[i] = (float)rand()/RAND_MAX - 0.5f;

    printf("=== Matmul Scaling: N=%d, K=%d (single-token M=1) ===\n", N, K);
    printf("%-12s %10s %10s %8s\n", "Threads", "Time(ms)", "GFLOP/s", "Speedup");

    double t_single = 0.0;
    int thread_counts[] = {1, 2, 4, 8, 16};
    for (int ti = 0; ti < 5; ti++) {
        int nt = thread_counts[ti];
        const int REPS = 100;
        double t0 = now_sec_mt();
        for (int r = 0; r < REPS; r++)
            matmul_pthread(out, input, weight, N, K, nt);
        double elapsed = (now_sec_mt() - t0) / REPS;
        double gflops = 2.0 * N * K / elapsed / 1e9;
        if (ti == 0) t_single = elapsed;
        printf("%-12d %10.2f %10.2f %8.2fx\n",
               nt, elapsed * 1000.0, gflops, t_single / elapsed);
    }

    // Theoretical bandwidth-limited prediction
    double bw = measure_bandwidth_GBs(64 * 1024 * 1024);
    double bytes_per_token = (double)N * K * 4;  // F32
    double tokens_per_sec  = bw * 1e9 / bytes_per_token;
    printf("\nMeasured DRAM bandwidth: %.1f GB/s\n", bw);
    printf("Bandwidth-limited max: %.1f tokens/sec (F32, single layer)\n", tokens_per_sec);

    free(input); free(weight); free(out);
}

int main(void) {
    roofline_analysis();
    printf("\n");
    benchmark_bandwidth();
    printf("\n");
    benchmark_matmul_scaling();
    return 0;
}
```

Expected results on a modern 8-core desktop with 60 GB/s bandwidth:
- Single-threaded: ~2-5 GFLOP/s (bandwidth-limited for large K)
- 4 threads: ~3-4× speedup (near-linear — bandwidth scales with core count on multi-channel DDR)
- 8 threads: ~4-6× speedup (diminishing returns as single memory controller saturates)

---

## Key Takeaways

- Single-token LLM inference is memory-bandwidth-bound: arithmetic intensity (FLOP/byte) is below the roofline ridge point for all quantization formats, so the bottleneck is how fast weights can be read from DRAM.
- The maximum tokens/sec is approximately `DRAM_bandwidth / bytes_per_token` — for a 7B INT8 model on 60 GB/s DDR5, this is roughly 60e9 / 7e9 ≈ 8.5 tokens/sec single-threaded.
- Parallelizing the outer loop over output neurons (`n`) is the correct approach for batch=1 matmul; parallelizing over batch rows (`m`) gives no benefit when M=1.
- OpenMP `#pragma omp parallel for schedule(static)` is sufficient for matmul parallelism; POSIX threads give finer control but require more boilerplate.
- Attention head parallelism is embarrassingly parallel — each head's computation is fully independent — making it an ideal target for thread-level parallelism.
- Multi-channel DRAM bandwidth scales with the number of active memory channels; more threads accessing different weight rows simultaneously can improve bandwidth utilization.
- Thread creation overhead is significant relative to small matmul times; for very small N×K, use fewer threads or batch multiple operations per thread launch.

---

**Previous**: [GGUF Format and Loading](./43_GGUF_and_Loading.md) | **Next**: [Capstone: Inference Engine](./45_Capstone_Inference_Engine.md)
