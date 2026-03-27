/*
 * parallel_infer.c -- Model parallelism concepts demo
 *
 * Demonstrates splitting a large matrix multiply across simulated "devices"
 * (array partitions), merging results, and showing speedup simulation.
 * Does NOT use OpenMP -- simulates parallelism conceptually with timing.
 *
 * Compile: gcc -std=c11 -Wall -Wextra -O2 -o parallel_infer parallel_infer.c -lm
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- Timer ---- */

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---- Standard (non-parallel) matmul ---- */

static void matmul_full(float *out, const float *input, const float *weight,
                        int M, int N, int K) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++)
                acc += input[m * K + k] * weight[n * K + k];
            out[m * N + n] = acc;
        }
}

/* ---- Tensor Parallel: split output dimension across devices ---- */

/*
 * Column parallelism: split the weight matrix along the output dimension.
 * Each "device" computes a subset of output neurons.
 *
 *   Device 0: weight[0:N/P, K]       -> out_partial[M, N/P]
 *   Device 1: weight[N/P:2*N/P, K]   -> out_partial[M, N/P]
 *   ...
 *   Merge: concatenate along N dimension -> out[M, N]
 */

typedef struct {
    const float *input;   /* shared input [M, K] */
    const float *weight;  /* this device's weight shard [shard_N, K] */
    float *out_shard;     /* output shard [M, shard_N] */
    int M, K, shard_N;
    int device_id;
    double compute_time;  /* measured time for this shard */
} DeviceWork;

static void device_compute(DeviceWork *work) {
    double t0 = now_sec();
    matmul_full(work->out_shard, work->input, work->weight,
                work->M, work->shard_N, work->K);
    work->compute_time = now_sec() - t0;
}

static void tensor_parallel_matmul(float *out, const float *input,
                                    const float *weight,
                                    int M, int N, int K, int n_devices) {
    int shard_N = N / n_devices;
    DeviceWork *devices = (DeviceWork *)malloc((size_t)n_devices * sizeof(DeviceWork));

    /* Allocate output shards */
    float **shards = (float **)malloc((size_t)n_devices * sizeof(float *));
    for (int d = 0; d < n_devices; d++) {
        shards[d] = (float *)malloc((size_t)M * shard_N * sizeof(float));
        devices[d].input     = input;
        devices[d].weight    = weight + d * shard_N * K;  /* shard of weight */
        devices[d].out_shard = shards[d];
        devices[d].M         = M;
        devices[d].K         = K;
        devices[d].shard_N   = shard_N;
        devices[d].device_id = d;
    }

    /* Simulate parallel execution (sequential here, but timing each) */
    for (int d = 0; d < n_devices; d++)
        device_compute(&devices[d]);

    /* Merge: concatenate output shards along N dimension */
    for (int m = 0; m < M; m++)
        for (int d = 0; d < n_devices; d++)
            memcpy(out + m * N + d * shard_N,
                   shards[d] + m * shard_N,
                   (size_t)shard_N * sizeof(float));

    /* Report per-device timing */
    double max_time = 0.0;
    for (int d = 0; d < n_devices; d++) {
        if (devices[d].compute_time > max_time)
            max_time = devices[d].compute_time;
    }

    for (int d = 0; d < n_devices; d++)
        free(shards[d]);
    free(shards);
    free(devices);
}

/* ---- Row parallelism: split along K, then reduce ---- */

/*
 * Row parallelism: split the weight matrix along the input (K) dimension.
 * Each "device" computes a partial sum, then an all-reduce (sum) merges.
 *
 *   Device 0: input[:, 0:K/P] @ weight[:, 0:K/P]^T  -> partial[M, N]
 *   Device 1: input[:, K/P:2K/P] @ weight[:, K/P:2K/P]^T -> partial[M, N]
 *   Reduce: out = sum(partials) along device dimension
 */

static void row_parallel_matmul(float *out, const float *input,
                                 const float *weight,
                                 int M, int N, int K, int n_devices) {
    int shard_K = K / n_devices;
    float *partial = (float *)calloc((size_t)M * N, sizeof(float));

    for (int d = 0; d < n_devices; d++) {
        /* Each device computes partial dot products */
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) {
                float acc = 0.0f;
                int k_start = d * shard_K;
                for (int k = 0; k < shard_K; k++)
                    acc += input[m * K + k_start + k] * weight[n * K + k_start + k];
                partial[m * N + n] += acc;
            }
    }

    memcpy(out, partial, (size_t)M * N * sizeof(float));
    free(partial);
}

/* ---- Attention head parallelism ---- */

static void single_head_attn(float *out_h, const float *Q, const float *K,
                              const float *V, int T, int d_k) {
    float scale = 1.0f / sqrtf((float)d_k);
    for (int tq = 0; tq < T; tq++) {
        float *scores = (float *)malloc((size_t)T * sizeof(float));
        for (int tk = 0; tk < T; tk++) {
            float dot = 0.0f;
            for (int j = 0; j < d_k; j++)
                dot += Q[tq * d_k + j] * K[tk * d_k + j];
            scores[tk] = dot * scale;
        }
        float mx = scores[0];
        for (int t = 1; t < T; t++) if (scores[t] > mx) mx = scores[t];
        float sum = 0.0f;
        for (int t = 0; t < T; t++) { scores[t] = expf(scores[t] - mx); sum += scores[t]; }
        for (int t = 0; t < T; t++) scores[t] /= sum;

        for (int j = 0; j < d_k; j++) {
            float acc = 0.0f;
            for (int t = 0; t < T; t++) acc += scores[t] * V[t * d_k + j];
            out_h[tq * d_k + j] = acc;
        }
        free(scores);
    }
}

static void parallel_multihead_attn(float *out, const float *Q, const float *K,
                                     const float *V, int T, int n_heads,
                                     int d_k, int n_devices) {
    int heads_per_device = n_heads / n_devices;

    printf("\n  Head distribution across %d devices:\n", n_devices);
    for (int d = 0; d < n_devices; d++) {
        int h_start = d * heads_per_device;
        int h_end = h_start + heads_per_device;
        if (d == n_devices - 1) h_end = n_heads;

        double t0 = now_sec();
        for (int h = h_start; h < h_end; h++) {
            single_head_attn(out + h * T * d_k,
                             Q + h * T * d_k,
                             K + h * T * d_k,
                             V + h * T * d_k,
                             T, d_k);
        }
        double dt = now_sec() - t0;
        printf("    Device %d: heads [%d-%d), time=%.3f ms\n",
               d, h_start, h_end, dt * 1000.0);
    }
}

/* ---- Roofline analysis ---- */

static void roofline_analysis(void) {
    printf("--- Roofline Analysis ---\n\n");
    printf("  Single-token inference is MEMORY-BANDWIDTH BOUND:\n\n");
    printf("  %-22s %8s %12s %12s\n", "Format", "FLOP/B", "7B Model", "Max tok/s*");
    printf("  %-22s %8s %12s %12s\n", "------", "------", "--------", "---------");
    printf("  %-22s %8.1f %12s %12s\n", "FP32", 0.5, "28.0 GB", "~2");
    printf("  %-22s %8.1f %12s %12s\n", "FP16", 1.0, "14.0 GB", "~4");
    printf("  %-22s %8.1f %12s %12s\n", "INT8", 2.0, "7.0 GB", "~8.5");
    printf("  %-22s %8.1f %12s %12s\n", "INT4", 4.0, "3.5 GB", "~17");
    printf("\n  * Assuming 60 GB/s DDR5 bandwidth\n");
    printf("  * Max tok/s = bandwidth / model_bytes\n\n");
}

/* ---- main ---- */

int main(void) {
    srand(42);

    printf("=== Parallel Inference Demo ===\n\n");

    roofline_analysis();

    /* --- Part 1: Column (tensor) parallelism --- */
    printf("--- Part 1: Column Parallelism (Split Output Dim) ---\n\n");

    const int M = 1;    /* batch=1, single-token inference */
    const int N = 256;  /* output neurons */
    const int K = 256;  /* input dim */

    float *input  = (float *)malloc((size_t)M * K * sizeof(float));
    float *weight = (float *)malloc((size_t)N * K * sizeof(float));
    float *out_full   = (float *)malloc((size_t)M * N * sizeof(float));
    float *out_par    = (float *)malloc((size_t)M * N * sizeof(float));

    for (int i = 0; i < M * K; i++) input[i] = (float)rand() / (float)RAND_MAX - 0.5f;
    for (int i = 0; i < N * K; i++) weight[i] = (float)rand() / (float)RAND_MAX - 0.5f;

    /* Full matmul */
    double t0 = now_sec();
    int reps = 200;
    for (int r = 0; r < reps; r++) matmul_full(out_full, input, weight, M, N, K);
    double t_full = (now_sec() - t0) / reps;

    printf("  Full matmul [%d,%d] x [%d,%d]^T: %.3f ms\n\n", M, K, N, K, t_full * 1000.0);

    int device_counts[] = {1, 2, 4, 8};
    printf("  %-10s %10s %10s %10s\n", "Devices", "Time(ms)", "GFLOP/s", "Speedup");
    printf("  %-10s %10s %10s %10s\n", "-------", "--------", "-------", "-------");

    for (int di = 0; di < 4; di++) {
        int nd = device_counts[di];
        if (N % nd != 0) continue;

        t0 = now_sec();
        for (int r = 0; r < reps; r++)
            tensor_parallel_matmul(out_par, input, weight, M, N, K, nd);
        double t_par = (now_sec() - t0) / reps;
        double gflops = 2.0 * M * N * K / t_par / 1e9;

        /* Verify correctness */
        float max_err = 0.0f;
        for (int i = 0; i < M * N; i++) {
            float err = fabsf(out_full[i] - out_par[i]);
            if (err > max_err) max_err = err;
        }

        /* Simulated speedup: in true parallelism, time = t_full / nd + comm_overhead */
        double sim_time = t_full / nd + 0.00001 * nd;  /* simulate small comm cost */
        double sim_speedup = t_full / sim_time;

        printf("  %-10d %10.3f %10.2f %8.2fx (simulated: %.2fx)\n",
               nd, t_par * 1000.0, gflops, t_full / t_par, sim_speedup);

        if (max_err > 1e-5f)
            printf("    WARNING: max error = %.2e\n", max_err);
    }

    /* --- Part 2: Row parallelism --- */
    printf("\n--- Part 2: Row Parallelism (Split Input Dim) ---\n\n");

    float *out_row = (float *)malloc((size_t)M * N * sizeof(float));
    row_parallel_matmul(out_row, input, weight, M, N, K, 4);

    float max_err = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float err = fabsf(out_full[i] - out_row[i]);
        if (err > max_err) max_err = err;
    }
    printf("  4-way row parallel vs full: max error = %.2e %s\n",
           max_err, max_err < 1e-5f ? "OK" : "ERROR");

    printf("\n  Column parallel: split output dim N -> concatenate results\n");
    printf("  Row parallel:    split input dim K  -> all-reduce (sum) results\n");

    /* --- Part 3: Attention head parallelism --- */
    printf("\n--- Part 3: Attention Head Parallelism ---\n\n");

    const int T = 32;
    const int n_heads = 8;
    const int d_k = 16;

    float *QQ = (float *)malloc((size_t)n_heads * T * d_k * sizeof(float));
    float *KK = (float *)malloc((size_t)n_heads * T * d_k * sizeof(float));
    float *VV = (float *)malloc((size_t)n_heads * T * d_k * sizeof(float));
    float *attn_out = (float *)malloc((size_t)n_heads * T * d_k * sizeof(float));

    for (int i = 0; i < n_heads * T * d_k; i++) {
        QQ[i] = (float)rand() / (float)RAND_MAX - 0.5f;
        KK[i] = (float)rand() / (float)RAND_MAX - 0.5f;
        VV[i] = (float)rand() / (float)RAND_MAX - 0.5f;
    }

    printf("  %d heads, T=%d, d_k=%d\n", n_heads, T, d_k);

    int head_devices[] = {1, 2, 4, 8};
    for (int di = 0; di < 4; di++) {
        int nd = head_devices[di];
        if (n_heads % nd != 0) continue;
        printf("\n  With %d device(s):", nd);
        parallel_multihead_attn(attn_out, QQ, KK, VV, T, n_heads, d_k, nd);
    }

    /* --- Part 4: Speedup simulation summary --- */
    printf("\n\n--- Speedup Simulation Summary ---\n\n");
    printf("  Ideal parallelism: time = T_serial / n_devices\n");
    printf("  Real parallelism:  time = T_serial / n + comm_overhead\n\n");

    printf("  %-10s %12s %12s %12s\n",
           "Devices", "Ideal", "w/ 1%% comm", "w/ 5%% comm");
    for (int nd = 1; nd <= 8; nd *= 2) {
        float ideal = (float)nd;
        float comm1 = 1.0f / (1.0f / (float)nd + 0.01f * (float)(nd - 1));
        float comm5 = 1.0f / (1.0f / (float)nd + 0.05f * (float)(nd - 1));
        printf("  %-10d %10.2fx %10.2fx %10.2fx\n", nd, ideal, comm1, comm5);
    }

    printf("\n  Key insight: communication overhead limits scaling.\n");
    printf("  For batch=1 inference (memory-bound), adding devices\n");
    printf("  helps by increasing aggregate memory bandwidth.\n");

    free(input); free(weight);
    free(out_full); free(out_par); free(out_row);
    free(QQ); free(KK); free(VV); free(attn_out);

    return 0;
}
