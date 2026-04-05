/*
 * thrust_demo.cu — Lesson 28: Thrust and CUB
 *
 * Demonstrates Thrust (header-only) for common GPU primitives:
 *   - thrust::device_vector (automatic device memory management)
 *   - thrust::sort
 *   - thrust::reduce (sum)
 *   - thrust::transform (element-wise operation)
 *   - thrust::exclusive_scan
 *   - Custom functor with thrust::transform
 *
 * Build:  nvcc -O2 -arch=sm_80 thrust_demo.cu -o thrust_demo
 * Run:    ./thrust_demo
 */

#include <cstdio>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <cuda_runtime.h>

static const int N = 1 << 20;   // 1 M elements

// ── Custom functor: clamp to [0,1] ────────────────────────────────────────────
struct Clamp01 {
    __host__ __device__
    float operator()(float x) const {
        return fminf(1.f, fmaxf(0.f, x));
    }
};

// ── Timing helper ─────────────────────────────────────────────────────────────
static float elapsed_ms(cudaEvent_t t0, cudaEvent_t t1) {
    float ms; cudaEventElapsedTime(&ms, t0, t1); return ms;
}

int main(void) {
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    // ── 1. Sort ───────────────────────────────────────────────────────────────
    thrust::device_vector<float> d_data(N);
    // Fill with reversed sequence for worst-case sort
    thrust::sequence(d_data.begin(), d_data.end(), (float)N, -1.f);

    cudaEventRecord(t0);
    thrust::sort(d_data.begin(), d_data.end());
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    printf("thrust::sort        %6.2f ms\n", elapsed_ms(t0, t1));
    printf("  Sorted? %s  d_data[0]=%.0f d_data[N-1]=%.0f\n",
           (d_data[0] <= d_data[N-1]) ? "yes" : "no",
           (float)d_data[0], (float)d_data[N-1]);

    // ── 2. Reduce ─────────────────────────────────────────────────────────────
    thrust::fill(d_data.begin(), d_data.end(), 1.f);
    cudaEventRecord(t0);
    float sum = thrust::reduce(d_data.begin(), d_data.end(), 0.f, thrust::plus<float>());
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    printf("thrust::reduce      %6.2f ms  sum=%.0f (%s)\n",
           elapsed_ms(t0, t1), sum, (int)sum == N ? "OK" : "FAIL");

    // ── 3. Transform with custom functor ──────────────────────────────────────
    // Fill with values in [-1, 2] range, then clamp to [0,1]
    thrust::host_vector<float> h_tmp(N);
    for (int i = 0; i < N; i++) h_tmp[i] = -1.f + 3.f * (float)i / N;
    d_data = h_tmp;

    thrust::device_vector<float> d_out(N);
    cudaEventRecord(t0);
    thrust::transform(d_data.begin(), d_data.end(), d_out.begin(), Clamp01());
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    printf("thrust::transform   %6.2f ms  (clamp to [0,1])\n", elapsed_ms(t0, t1));
    printf("  d_out[0]=%.2f d_out[N/2]=%.2f d_out[N-1]=%.2f\n",
           (float)d_out[0], (float)d_out[N/2], (float)d_out[N-1]);

    // ── 4. Exclusive scan ─────────────────────────────────────────────────────
    thrust::fill(d_data.begin(), d_data.end(), 1.f);
    cudaEventRecord(t0);
    thrust::exclusive_scan(d_data.begin(), d_data.end(), d_out.begin());
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    printf("thrust::exc_scan    %6.2f ms  out[0]=%.0f out[4]=%.0f\n",
           elapsed_ms(t0, t1), (float)d_out[0], (float)d_out[4]);

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return 0;
}
