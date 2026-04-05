/*
 * bias_act_fused.cu — Lesson 36: Fused Kernel Patterns
 *
 * Demonstrates kernel fusion to eliminate intermediate memory round-trips:
 *
 *   Unfused pipeline:
 *     out = activation(matmul(x, W) + bias)
 *     → 3 separate kernels, 3 HBM reads + 3 HBM writes
 *
 *   Fused kernel:
 *     out[i] = activation(in[i] + bias[i % cols])
 *     → 1 kernel, 1 HBM read + 1 HBM write
 *
 * Activation variants: ReLU, GELU (approx), SiLU (Swish)
 * Also demonstrates elementwise bias + residual + layer norm fusion.
 *
 * Build:  nvcc -O2 -arch=sm_80 bias_act_fused.cu -o bias_act_fused
 * Run:    ./bias_act_fused
 */

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e)); exit(1); } } while(0)

static const int ROWS    = 4096;
static const int COLS    = 2048;
static const int THREADS = 256;
static const int ITERS   = 100;

// ── Activation functions ──────────────────────────────────────────────────────
__device__ __forceinline__ float relu(float x)   { return fmaxf(0.f, x); }
__device__ __forceinline__ float gelu_approx(float x) {
    // tanh approximation of GELU
    const float c = 0.7978845608f;   // sqrt(2/pi)
    float y = c * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.f + tanhf(y));
}
__device__ __forceinline__ float silu(float x) { return x / (1.f + expf(-x)); }

// ── Unfused: separate add_bias and activation kernels ────────────────────────
__global__ void add_bias(float *data, const float *bias, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * cols) data[i] += bias[i % cols];
}
__global__ void apply_gelu(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = gelu_approx(data[i]);
}

// ── Fused: bias + GELU in one pass ────────────────────────────────────────────
__global__ void bias_gelu_fused(const float *in, float *out,
                                  const float *bias, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * cols)
        out[i] = gelu_approx(in[i] + bias[i % cols]);
}

// ── Fused: bias + SiLU + residual add ────────────────────────────────────────
__global__ void bias_silu_residual(const float *in, const float *residual,
                                    float *out, const float *bias,
                                    int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * cols)
        out[i] = silu(in[i] + bias[i % cols]) + residual[i];
}

// ── Vector4 fused: processes 4 floats per thread for better BW utilisation ───
__global__ void bias_relu_vec4(const float4 *in, float4 *out,
                                const float *bias, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total4 = rows * cols / 4;
    if (i >= total4) return;
    float4 v   = in[i];
    int col_base = (i * 4) % cols;
    v.x = relu(v.x + bias[col_base]);
    v.y = relu(v.y + bias[(col_base + 1) % cols]);
    v.z = relu(v.z + bias[(col_base + 2) % cols]);
    v.w = relu(v.w + bias[(col_base + 3) % cols]);
    out[i] = v;
}

int main(void) {
    int total = ROWS * COLS;
    size_t bytes = (size_t)total * sizeof(float);

    float *d_in, *d_out, *d_bias, *d_tmp, *d_residual;
    CUDA_CHECK(cudaMalloc(&d_in,       bytes));
    CUDA_CHECK(cudaMalloc(&d_out,      bytes));
    CUDA_CHECK(cudaMalloc(&d_tmp,      bytes));
    CUDA_CHECK(cudaMalloc(&d_bias,     COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_residual, bytes));
    CUDA_CHECK(cudaMemset(d_in,  0, bytes));
    CUDA_CHECK(cudaMemset(d_residual, 0, bytes));

    int blocks = (total + THREADS - 1) / THREADS;

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    auto bench = [&](const char *name, auto fn) {
        fn();   // warmup
        cudaEventRecord(t0);
        for (int i = 0; i < ITERS; i++) fn();
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1);
        double bw = 2.0 * bytes / ((ms / ITERS) * 1e-3) / 1e9;
        printf("  %-30s %6.3f ms  BW=%5.1f GB/s\n", name, ms / ITERS, bw);
    };

    printf("Fused kernel benchmark (%dx%d matrix)\n", ROWS, COLS);

    bench("unfused bias+GELU (2 kernels)", [&](){
        CUDA_CHECK(cudaMemcpy(d_tmp, d_in, bytes, cudaMemcpyDeviceToDevice));
        add_bias  <<<blocks, THREADS>>>(d_tmp, d_bias, ROWS, COLS);
        apply_gelu<<<blocks, THREADS>>>(d_tmp, total);
    });

    bench("fused bias+GELU", [&](){
        bias_gelu_fused<<<blocks, THREADS>>>(d_in, d_out, d_bias, ROWS, COLS);
    });

    bench("fused bias+SiLU+residual", [&](){
        bias_silu_residual<<<blocks, THREADS>>>(d_in, d_residual, d_out, d_bias, ROWS, COLS);
    });

    int blocks4 = (total/4 + THREADS - 1) / THREADS;
    bench("fused bias+ReLU (float4)", [&](){
        bias_relu_vec4<<<blocks4, THREADS>>>(
            (const float4*)d_in, (float4*)d_out, d_bias, ROWS, COLS);
    });

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_bias);
    cudaFree(d_tmp); cudaFree(d_residual);
    return 0;
}
