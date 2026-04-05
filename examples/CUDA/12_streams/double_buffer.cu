/*
 * double_buffer.cu — Lesson 12: Streams and Async
 *
 * Demonstrates:
 *   - Asynchronous H2D / D2H transfers with cudaMemcpyAsync
 *   - CUDA streams for overlapping compute and data transfer
 *   - Double-buffering pattern (N streams, each with its own chunk)
 *   - cudaEvent for cross-stream synchronization and timing
 *
 * Build:  nvcc -O2 -arch=sm_80 double_buffer.cu -o double_buffer
 * Run:    ./double_buffer
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e)); exit(1); } } while(0)

static const int N_TOTAL  = 1 << 23;  // 8 M floats total
static const int N_CHUNKS = 4;        // double-buffer uses 2, but 4 shows pipelining
static const int CHUNK    = N_TOTAL / N_CHUNKS;
static const int THREADS  = 256;

// ── Simple element-wise kernel (scale by 2) ───────────────────────────────────
__global__ void scale(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= 2.f;
}

// ── Synchronous baseline (no overlap) ────────────────────────────────────────
static float sync_baseline(float *h_in, float *h_out, float *d_buf, size_t bytes_total) {
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    CUDA_CHECK(cudaMemcpy(d_buf, h_in, bytes_total, cudaMemcpyHostToDevice));
    scale<<<(N_TOTAL + THREADS - 1) / THREADS, THREADS>>>(d_buf, N_TOTAL);
    CUDA_CHECK(cudaMemcpy(h_out, d_buf, bytes_total, cudaMemcpyDeviceToHost));

    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms;
}

// ── Double-buffered (overlapped) ──────────────────────────────────────────────
// Two streams alternate; while chunk k is being processed on GPU,
// chunk k+1 is being transferred H→D on the other stream.
static float async_pipeline(float *h_in, float *h_out,
                             float *d_ping, float *d_pong) {
    cudaStream_t streams[2];
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    float *d_bufs[2] = {d_ping, d_pong};
    size_t chunk_bytes = CHUNK * sizeof(float);
    int blocks = (CHUNK + THREADS - 1) / THREADS;

    for (int i = 0; i < N_CHUNKS; i++) {
        int s     = i & 1;                    // alternate between stream 0/1
        float *hi = h_in  + i * CHUNK;
        float *ho = h_out + i * CHUNK;
        float *db = d_bufs[s];

        // Wait for previous use of this stream's buffer to finish D2H
        cudaStreamSynchronize(streams[s]);

        CUDA_CHECK(cudaMemcpyAsync(db, hi, chunk_bytes, cudaMemcpyHostToDevice, streams[s]));
        scale<<<blocks, THREADS, 0, streams[s]>>>(db, CHUNK);
        CUDA_CHECK(cudaMemcpyAsync(ho, db, chunk_bytes, cudaMemcpyDeviceToHost, streams[s]));
    }

    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);

    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);
    return ms;
}

int main(void) {
    size_t bytes = (size_t)N_TOTAL * sizeof(float);

    // Pinned host memory is required for async transfers
    float *h_in, *h_out;
    CUDA_CHECK(cudaMallocHost(&h_in,  bytes));
    CUDA_CHECK(cudaMallocHost(&h_out, bytes));
    for (int i = 0; i < N_TOTAL; i++) h_in[i] = (float)i;

    float *d_ping, *d_pong, *d_full;
    CUDA_CHECK(cudaMalloc(&d_full, bytes));
    CUDA_CHECK(cudaMalloc(&d_ping, CHUNK * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pong, CHUNK * sizeof(float)));

    float ms_sync  = sync_baseline(h_in, h_out, d_full, bytes);
    float ms_async = async_pipeline(h_in, h_out, d_ping, d_pong);

    printf("Stream double-buffer benchmark (%d M floats, %d chunks)\n",
           N_TOTAL >> 20, N_CHUNKS);
    printf("  Synchronous (serial) : %.3f ms\n", ms_sync);
    printf("  Double-buffered async: %.3f ms   speedup=%.2fx\n",
           ms_async, ms_sync / ms_async);

    cudaFreeHost(h_in); cudaFreeHost(h_out);
    cudaFree(d_ping); cudaFree(d_pong); cudaFree(d_full);
    return 0;
}
