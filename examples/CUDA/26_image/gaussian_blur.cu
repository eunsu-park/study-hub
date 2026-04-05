/*
 * gaussian_blur.cu — Lesson 26: Image Processing on GPU
 *
 * Demonstrates:
 *   - 2-D separable Gaussian blur (horizontal then vertical pass)
 *   - Constant memory for the convolution kernel coefficients
 *   - Shared-memory tile with halo loading
 *   - Clamped boundary conditions
 *
 * Build:  nvcc -O2 -arch=sm_80 gaussian_blur.cu -o gaussian_blur
 * Run:    ./gaussian_blur
 */

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e)); exit(1); } } while(0)

static const int   WIDTH  = 1920;
static const int   HEIGHT = 1080;
static const int   RADIUS = 8;           // kernel half-size
static const int   KSIZE  = 2*RADIUS+1;  // 17
static const int   TILE   = 32;

// Precomputed Gaussian kernel stored in constant memory
__constant__ float c_kernel[KSIZE];

// ── Horizontal pass ───────────────────────────────────────────────────────────
// Shared memory tile width = TILE + 2*RADIUS (includes left/right halo)
__global__ void blur_h(const float *in, float *out, int w, int h) {
    extern __shared__ float s[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gx = blockIdx.x * TILE + tx - RADIUS;   // global x (may be negative)
    int gy = blockIdx.y * TILE + ty;
    int row_width = TILE + 2 * RADIUS;

    // Load halo + interior
    float val = (gx >= 0 && gx < w && gy < h) ? in[gy * w + gx] : 0.f;
    s[ty * row_width + tx] = val;
    __syncthreads();

    // Only interior threads write output
    if (tx >= RADIUS && tx < TILE + RADIUS && gx < w && gy < h) {
        float sum = 0.f;
        for (int k = -RADIUS; k <= RADIUS; k++)
            sum += c_kernel[k + RADIUS] * s[ty * row_width + tx + k];
        out[gy * w + (gx)] = sum;
    }
}

// ── Vertical pass ─────────────────────────────────────────────────────────────
__global__ void blur_v(const float *in, float *out, int w, int h) {
    extern __shared__ float s[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gx = blockIdx.x * TILE + tx;
    int gy = blockIdx.y * TILE + ty - RADIUS;
    int col_height = TILE + 2 * RADIUS;

    float val = (gx < w && gy >= 0 && gy < h) ? in[gy * w + gx] : 0.f;
    s[ty * TILE + tx] = val;
    __syncthreads();

    if (ty >= RADIUS && ty < TILE + RADIUS && gx < w && gy < h) {
        float sum = 0.f;
        for (int k = -RADIUS; k <= RADIUS; k++)
            sum += c_kernel[k + RADIUS] * s[(ty + k) * TILE + tx];
        out[gy * w + gx] = sum;
    }
    (void)col_height;
}

int main(void) {
    // Build Gaussian kernel (σ = RADIUS/2)
    float sigma = RADIUS / 2.f;
    float h_ker[KSIZE], sum = 0.f;
    for (int i = 0; i < KSIZE; i++) {
        float x = i - RADIUS;
        h_ker[i] = expf(-x*x / (2.f * sigma * sigma));
        sum += h_ker[i];
    }
    for (int i = 0; i < KSIZE; i++) h_ker[i] /= sum;
    CUDA_CHECK(cudaMemcpyToSymbol(c_kernel, h_ker, KSIZE * sizeof(float)));

    // Synthetic input image (gradient)
    const size_t pixels = (size_t)WIDTH * HEIGHT;
    float *h_in = (float *)malloc(pixels * sizeof(float));
    for (int i = 0; i < (int)pixels; i++)
        h_in[i] = (float)(i % 256) / 255.f;

    float *d_in, *d_tmp, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  pixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tmp, pixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, pixels * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, pixels * sizeof(float), cudaMemcpyHostToDevice));

    // Block: (TILE+2*RADIUS) x TILE — covers tile + halo in x
    dim3 block_h(TILE + 2 * RADIUS, TILE);
    dim3 grid_h((WIDTH  + TILE - 1) / TILE, (HEIGHT + TILE - 1) / TILE);
    dim3 block_v(TILE, TILE + 2 * RADIUS);
    dim3 grid_v((WIDTH  + TILE - 1) / TILE, (HEIGHT + TILE - 1) / TILE);

    size_t smem_h = (size_t)TILE * (TILE + 2 * RADIUS) * sizeof(float);
    size_t smem_v = (size_t)(TILE + 2 * RADIUS) * TILE * sizeof(float);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    blur_h<<<grid_h, block_h, smem_h>>>(d_in,  d_tmp, WIDTH, HEIGHT);
    blur_v<<<grid_v, block_v, smem_v>>>(d_tmp, d_out, WIDTH, HEIGHT);

    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);

    printf("Gaussian blur (%dx%d, radius=%d):\n", WIDTH, HEIGHT, RADIUS);
    printf("  Time: %.3f ms  BW=%.1f GB/s\n", ms,
           3.f * pixels * sizeof(float) / (ms * 1e-3) / 1e9);

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_in); cudaFree(d_tmp); cudaFree(d_out);
    free(h_in);
    return 0;
}
