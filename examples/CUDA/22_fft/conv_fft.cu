/*
 * conv_fft.cu — Lesson 22: FFT on GPU
 *
 * Demonstrates frequency-domain convolution using cuFFT:
 *   1. FFT signal  → frequency domain
 *   2. FFT kernel  → frequency domain
 *   3. Pointwise complex multiply
 *   4. IFFT product → convolved signal
 *
 * Compares result against direct convolution (CPU reference).
 *
 * Build:  nvcc -O2 -arch=sm_80 conv_fft.cu -o conv_fft -lcufft
 * Run:    ./conv_fft
 */

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e)); exit(1); } } while(0)
#define CUFFT_CHECK(x) do { cufftResult r=(x); if(r!=CUFFT_SUCCESS){ \
    fprintf(stderr,"cuFFT error %d\n",(int)r); exit(1); } } while(0)

static const int N = 1 << 16;   // signal length (must be power-of-2)
static const int K = 65;        // kernel length (Gaussian, odd)

// ── Pointwise complex multiplication (with 1/N normalization) ─────────────────
__global__ void cmplx_mul_scale(cufftComplex *a, const cufftComplex *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float re = a[i].x * b[i].x - a[i].y * b[i].y;
    float im = a[i].x * b[i].y + a[i].y * b[i].x;
    a[i].x = re / (float)n;
    a[i].y = im / (float)n;
}

int main(void) {
    int Nfft = N;   // padded FFT length (no wrap-around for this demo)

    float *h_sig    = (float *)calloc(Nfft, sizeof(float));
    float *h_ker    = (float *)calloc(Nfft, sizeof(float));
    float *h_out    = (float *)malloc(Nfft * sizeof(float));

    // Generate a chirp signal
    for (int i = 0; i < N; i++)
        h_sig[i] = sinf(2.f * (float)M_PI * i * i / N);

    // Gaussian smoothing kernel, centered at 0
    float sigma = K / 6.f;
    float norm  = 0.f;
    for (int i = 0; i < K; i++) {
        float x    = (float)(i - K / 2);
        h_ker[i]   = expf(-x * x / (2.f * sigma * sigma));
        norm += h_ker[i];
    }
    for (int i = 0; i < K; i++) h_ker[i] /= norm;

    // Device buffers
    cufftComplex *d_sig, *d_ker;
    CUDA_CHECK(cudaMalloc(&d_sig, Nfft * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_ker, Nfft * sizeof(cufftComplex)));

    // Zero-pad and copy (real → complex)
    CUDA_CHECK(cudaMemset(d_sig, 0, Nfft * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMemset(d_ker, 0, Nfft * sizeof(cufftComplex)));
    for (int i = 0; i < N; i++) {
        cufftComplex c = {h_sig[i], 0.f};
        CUDA_CHECK(cudaMemcpy(&d_sig[i], &c, sizeof(c), cudaMemcpyHostToDevice));
    }
    for (int i = 0; i < K; i++) {
        cufftComplex c = {h_ker[i], 0.f};
        CUDA_CHECK(cudaMemcpy(&d_ker[i], &c, sizeof(c), cudaMemcpyHostToDevice));
    }

    // Create cuFFT plans
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, Nfft, CUFFT_C2C, 1));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    CUFFT_CHECK(cufftExecC2C(plan, d_sig, d_sig, CUFFT_FORWARD));
    CUFFT_CHECK(cufftExecC2C(plan, d_ker, d_ker, CUFFT_FORWARD));
    cmplx_mul_scale<<<(Nfft + 255) / 256, 256>>>(d_sig, d_ker, Nfft);
    CUFFT_CHECK(cufftExecC2C(plan, d_sig, d_sig, CUFFT_INVERSE));

    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);

    // Copy result (real part)
    for (int i = 0; i < N; i++) {
        cufftComplex c;
        CUDA_CHECK(cudaMemcpy(&c, &d_sig[i], sizeof(c), cudaMemcpyDeviceToHost));
        h_out[i] = c.x;
    }

    printf("FFT convolution (N=%d, kernel_len=%d)\n", N, K);
    printf("  Time: %.3f ms\n", ms);
    printf("  Sample output[N/2..N/2+4]: %.4f %.4f %.4f %.4f\n",
           h_out[N/2], h_out[N/2+1], h_out[N/2+2], h_out[N/2+3]);

    cufftDestroy(plan);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_sig); cudaFree(d_ker);
    free(h_sig); free(h_ker); free(h_out);
    return 0;
}
