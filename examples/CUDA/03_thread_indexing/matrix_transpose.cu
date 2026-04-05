/*
 * matrix_transpose.cu — Lesson 03: Thread Indexing and Grids
 *
 * Demonstrates:
 *   - 2-D grid/block indexing
 *   - Row-major ↔ column-major index mapping
 *   - Naive vs shared-memory transpose (bank-conflict-free via padding)
 *   - cudaEvent timing
 *
 * Build:  nvcc -O2 -arch=sm_80 matrix_transpose.cu -o matrix_transpose
 * Run:    ./matrix_transpose
 */

#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
    exit(1); } } while(0)

static const int N       = 1024;   // square matrix size
static const int TILE    = 32;     // tile width
static const int PAD     = 1;      // +1 column padding avoids bank conflicts

// ── Naive transpose ───────────────────────────────────────────────────────────
// Each thread reads A[row][col] and writes to B[col][row].
// Read access to A is coalesced; write to B is not → high latency.
__global__ void transpose_naive(const float *A, float *B, int n) {
    int col = blockIdx.x * TILE + threadIdx.x;
    int row = blockIdx.y * TILE + threadIdx.y;
    if (row < n && col < n)
        B[col * n + row] = A[row * n + col];
}

// ── Tiled (shared-memory) transpose ──────────────────────────────────────────
// Load tile into shared mem (coalesced read), then write transposed (coalesced write).
// PAD avoids shared-memory bank conflicts on the column dimension.
__global__ void transpose_tiled(const float *A, float *B, int n) {
    __shared__ float tile[TILE][TILE + PAD];

    int col_in  = blockIdx.x * TILE + threadIdx.x;
    int row_in  = blockIdx.y * TILE + threadIdx.y;

    if (row_in < n && col_in < n)
        tile[threadIdx.y][threadIdx.x] = A[row_in * n + col_in];

    __syncthreads();

    // Swapped block indices for the output
    int col_out = blockIdx.y * TILE + threadIdx.x;
    int row_out = blockIdx.x * TILE + threadIdx.y;

    if (row_out < n && col_out < n)
        B[row_out * n + col_out] = tile[threadIdx.x][threadIdx.y];
}

// ── Helper: time a kernel with cudaEvents ─────────────────────────────────────
static float time_kernel(void (*launch)(const float*, float*, int, dim3, dim3),
                          const float *d_A, float *d_B, int n,
                          dim3 grid, dim3 block) {
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    launch(d_A, d_B, n, grid, block);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms;
}

static void launch_naive(const float *A, float *B, int n, dim3 g, dim3 b) {
    transpose_naive<<<g, b>>>(A, B, n);
}
static void launch_tiled(const float *A, float *B, int n, dim3 g, dim3 b) {
    transpose_tiled<<<g, b>>>(A, B, n);
}

int main(void) {
    const size_t bytes = (size_t)N * N * sizeof(float);

    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    for (int i = 0; i < N * N; i++) h_A[i] = (float)i;

    float *d_A, *d_B;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    float ms_naive = time_kernel(launch_naive, d_A, d_B, N, grid, block);
    float ms_tiled = time_kernel(launch_tiled, d_A, d_B, N, grid, block);

    // Verify correctness of tiled version
    CUDA_CHECK(cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int r = 0; r < N && ok; r++)
        for (int c = 0; c < N && ok; c++)
            if (h_B[c * N + r] != h_A[r * N + c]) ok = false;

    printf("Matrix transpose (%dx%d)\n", N, N);
    printf("  Naive  : %.3f ms\n", ms_naive);
    printf("  Tiled  : %.3f ms\n", ms_tiled);
    printf("  Correct: %s\n", ok ? "yes" : "NO");

    cudaFree(d_A); cudaFree(d_B);
    free(h_A); free(h_B);
    return ok ? 0 : 1;
}
