# 22. FFT on GPU

**Previous**: [Monte Carlo Methods](./21_Monte_Carlo_Methods.md) | **Next**: [PDE Solvers Heat Equation](./23_PDE_Solvers_Heat_Equation.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Use the cuFFT API to compute 1D, 2D, and 3D FFTs on the GPU
2. Execute batched FFTs for many independent transforms in a single call
3. Implement convolution via FFT (O(N log N)) and compare to direct convolution (O(N·K))
4. Apply spectral filtering to remove frequency components from a signal
5. Understand cuFFT's normalization convention and correctly apply the 1/N scale factor

---

## 1. The Discrete Fourier Transform

The DFT transforms N complex numbers from the time/space domain to the frequency domain:

```
X[k] = Σ_{n=0}^{N-1} x[n] · e^(-2πi·k·n/N)    for k = 0, 1, ..., N-1

Inverse DFT:
x[n] = (1/N) · Σ_{k=0}^{N-1} X[k] · e^(2πi·k·n/N)

Key property: DFT decomposes a signal into N frequency components (sinusoids).
FFT computes DFT in O(N log N) instead of O(N²).
```

cuFFT implements the Cooley-Tukey radix-2 FFT and its generalizations, optimized for GPU execution.

---

## 2. cuFFT Basics: 1D FFT

```c
#include <cufft.h>

void fft_1d_example(int N) {
    // Allocate complex arrays on device
    cufftComplex *d_signal;
    cudaMalloc(&d_signal, N * sizeof(cufftComplex));

    // Create plan: 1D FFT of N complex-to-complex points
    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_C2C, /*batch=*/1);

    // Forward FFT: CUFFT_FORWARD = -1 (exponent sign convention)
    cufftExecC2C(plan, d_signal, d_signal, CUFFT_FORWARD);
    // d_signal now contains X[k] (NOT normalized)

    // Inverse FFT: CUFFT_INVERSE = +1
    cufftExecC2C(plan, d_signal, d_signal, CUFFT_INVERSE);
    // d_signal now contains N * x[n]  ← cuFFT does NOT normalize!

    // MUST manually divide by N after inverse:
    scale_kernel<<<(N + 255) / 256, 256>>>(d_signal, 1.0f / N, N);

    cufftDestroy(plan);
    cudaFree(d_signal);
}

// Normalization kernel
__global__ void scale_complex(cufftComplex *data, float scale, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        data[i].x *= scale;
        data[i].y *= scale;
    }
}
```

**Normalization convention**: cuFFT applies no normalization — forward and inverse FFT are both un-normalized. After forward + inverse, each element is multiplied by N. Always divide by N after `CUFFT_INVERSE`.

---

## 3. Real-to-Complex FFT (R2C)

For real-valued input (e.g., audio, sensor data), cuFFT exploits conjugate symmetry to store only N/2+1 complex output coefficients:

```c
void fft_real_1d(int N) {
    float        *d_real;     // N real input values
    cufftComplex *d_freq;     // N/2 + 1 complex output values (Hermitian)

    cudaMalloc(&d_real, N * sizeof(float));
    cudaMalloc(&d_freq, (N / 2 + 1) * sizeof(cufftComplex));

    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_R2C, 1);
    cufftExecR2C(plan, d_real, d_freq);
    // d_freq[0] = DC component (real)
    // d_freq[k] = complex amplitude at frequency k / (N * dt)
    // d_freq[N/2] = Nyquist frequency (real)

    // Inverse: C2R
    cufftHandle iplan;
    cufftPlan1d(&iplan, N, CUFFT_C2R, 1);
    cufftExecC2R(iplan, d_freq, d_real);
    scale_real_kernel<<<(N+255)/256, 256>>>(d_real, 1.0f/N, N);

    cufftDestroy(plan); cufftDestroy(iplan);
    cudaFree(d_real); cudaFree(d_freq);
}
```

R2C saves ~50% memory and runs ~50% faster than C2C for real inputs.

---

## 4. 2D and 3D FFT

```c
// 2D FFT (e.g., image processing, 2D convolution)
void fft_2d(int Nx, int Ny) {
    cufftComplex *d_image;
    cudaMalloc(&d_image, Nx * Ny * sizeof(cufftComplex));

    cufftHandle plan;
    cufftPlan2d(&plan, Ny, Nx, CUFFT_C2C);  // note: rows first (Ny, Nx)
    cufftExecC2C(plan, d_image, d_image, CUFFT_FORWARD);
    // Each element d_image[ky*Nx + kx] = X[ky, kx]

    cufftDestroy(plan);
    cudaFree(d_image);
}

// 3D FFT (e.g., volumetric data, 3D convolution)
void fft_3d(int Nx, int Ny, int Nz) {
    cufftComplex *d_vol;
    cudaMalloc(&d_vol, Nx * Ny * Nz * sizeof(cufftComplex));

    cufftHandle plan;
    cufftPlan3d(&plan, Nz, Ny, Nx, CUFFT_C2C);
    cufftExecC2C(plan, d_vol, d_vol, CUFFT_FORWARD);

    cufftDestroy(plan);
    cudaFree(d_vol);
}
```

---

## 5. Batched FFT

Compute many independent FFTs of the same size in one cuFFT call — crucial for neural networks, audio processing, and multi-channel signal analysis:

```c
// Batched 1D FFT: compute B independent FFTs of size N simultaneously
void batched_fft_1d(int N, int B) {
    cufftComplex *d_signals;
    // Layout: batch b is at d_signals[b * N ... b*N + N - 1]
    cudaMalloc(&d_signals, B * N * sizeof(cufftComplex));

    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_C2C, B);  // third arg = batch count
    cufftExecC2C(plan, d_signals, d_signals, CUFFT_FORWARD);

    cufftDestroy(plan);
    cudaFree(d_signals);
}

// Advanced: PlanMany for non-contiguous batches or strided layouts
void batched_fft_strided(int N, int B, int stride) {
    cufftHandle plan;
    int rank  = 1;           // 1D transforms
    int n[]   = {N};
    int inembed[] = {0};     // NULL-equivalent (auto)
    int onembed[] = {0};
    int idist = stride;      // distance between batch starts in input
    int odist = stride;      // distance between batch starts in output
    int istride = 1;
    int ostride = 1;

    cufftPlanMany(&plan, rank, n,
                  inembed, istride, idist,
                  onembed, ostride, odist,
                  CUFFT_C2C, B);
    // ... exec and destroy
}
```

---

## 6. Convolution via FFT

**Direct convolution**: O(N·K) for signal of length N and kernel of length K.
**FFT convolution**: O(N log N) via the convolution theorem: conv(x, h) = IFFT(FFT(x) · FFT(h))

```c
// 1D convolution via FFT
// x: signal of length N, h: kernel of length K, output length = N + K - 1
void fft_convolution(
    const float *d_x, int N,
    const float *d_h, int K,
    float *d_out)
{
    int M = N + K - 1;                // linear convolution length
    int padded = next_power_of_two(M); // cuFFT fastest for power-of-2

    cufftComplex *d_X, *d_H;
    cudaMalloc(&d_X, padded * sizeof(cufftComplex));
    cudaMalloc(&d_H, padded * sizeof(cufftComplex));

    // Zero-pad and copy inputs to complex arrays
    pad_real_to_complex<<<(padded+255)/256, 256>>>(d_X, d_x, N, padded);
    pad_real_to_complex<<<(padded+255)/256, 256>>>(d_H, d_h, K, padded);

    // Forward FFT both signals
    cufftHandle plan;
    cufftPlan1d(&plan, padded, CUFFT_C2C, 1);
    cufftExecC2C(plan, d_X, d_X, CUFFT_FORWARD);
    cufftExecC2C(plan, d_H, d_H, CUFFT_FORWARD);

    // Pointwise multiply in frequency domain
    complex_multiply<<<(padded+255)/256, 256>>>(d_X, d_H, padded);

    // Inverse FFT
    cufftExecC2C(plan, d_X, d_X, CUFFT_INVERSE);

    // Extract real part and normalize
    extract_real_scaled<<<(M+255)/256, 256>>>(d_out, d_X, 1.0f/padded, M);

    cufftDestroy(plan);
    cudaFree(d_X); cudaFree(d_H);
}

// Frequency-domain multiplication kernel
__global__ void complex_multiply(cufftComplex *A, const cufftComplex *B, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float ar = A[i].x, ai = A[i].y;
    float br = B[i].x, bi = B[i].y;
    A[i].x = ar * br - ai * bi;   // real part of A[i] * B[i]
    A[i].y = ar * bi + ai * br;   // imag part
}
```

**Break-even**: FFT convolution beats direct convolution when K > log₂(N) ≈ 20 for N=10⁶.

---

## 7. Spectral Filtering

Apply a low-pass filter by zeroing high-frequency coefficients:

```c
// Low-pass filter: keep only frequencies below cutoff_frac * N/2
__global__ void lowpass_filter(cufftComplex *freq, int N, float cutoff_frac) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;

    // Map k to signed frequency (-N/2 to N/2)
    int signed_k = (k <= N / 2) ? k : k - N;
    float normalized_freq = fabsf((float)signed_k) / (N / 2.0f);

    if (normalized_freq > cutoff_frac) {
        freq[k].x = 0.0f;
        freq[k].y = 0.0f;
    }
}

// Power spectrum: |X[k]|² for each frequency bin
__global__ void power_spectrum(const cufftComplex *freq, float *power, int N) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < N) {
        power[k] = freq[k].x * freq[k].x + freq[k].y * freq[k].y;
    }
}
```

---

## 8. Parseval's Theorem

Energy is conserved between time and frequency domains (with proper normalization):

```
Σ |x[n]|² = (1/N) · Σ |X[k]|²

Verification code (sanity check for correctness):
```

```c
// Verify Parseval's theorem after forward FFT
bool verify_parseval(const float *h_signal, const cufftComplex *h_freq, int N) {
    double energy_time = 0.0, energy_freq = 0.0;
    for (int n = 0; n < N; n++) energy_time += h_signal[n] * h_signal[n];
    for (int k = 0; k < N; k++) energy_freq +=
        h_freq[k].x * h_freq[k].x + h_freq[k].y * h_freq[k].y;
    energy_freq /= N;  // normalization factor

    double ratio = energy_time / energy_freq;
    printf("Parseval ratio (should be 1.0): %.6f\n", ratio);
    return fabs(ratio - 1.0) < 1e-4;
}
```

---

## Key Takeaways

- cuFFT computes un-normalized FFTs: after forward + inverse, each element is multiplied by N — always divide by N after `CUFFT_INVERSE`
- **R2C/C2R** plans save 50% memory and runtime for real-valued signals by exploiting Hermitian symmetry
- **Batched FFT** (`cufftPlan1d` with batch > 1 or `cufftPlanMany`) is the correct API for many independent transforms
- **FFT convolution** is O(N log N) vs direct O(N·K) — break-even is typically K ≈ 20 for large N
- Spectral filtering is trivially parallel: zero unwanted frequency bins after forward FFT, then apply inverse
- **Parseval's theorem** provides a cheap sanity check: time-domain energy equals frequency-domain energy/N

---

**Next**: [23. PDE Solvers Heat Equation](./23_PDE_Solvers_Heat_Equation.md) — Solve the 2D heat equation with explicit finite differences, analyze stability conditions, and implement multi-step time integration on the GPU.
