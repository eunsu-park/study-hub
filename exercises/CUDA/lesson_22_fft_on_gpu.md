# Lesson 22 — FFT on GPU (cuFFT) (per-lesson exercise)

Prerequisites: L29 (cuBLAS / library calls).

Compile: `nvcc -O3 -arch=sm_80 ex.cu -lcufft -o ex`

The Fast Fourier Transform is one of the foundational algorithms in numerical computing — used in audio, image processing, PDE solvers, and convolutional neural networks (small-kernel convolutions are sometimes faster as FFT × FFT × inverse FFT).

NVIDIA's cuFFT library covers 1D/2D/3D, real/complex, single/double precision. Hand-rolling an FFT on GPU is a worthwhile teaching exercise but is essentially never the right production choice.

---

## Exercise 22.1 — 1D Complex FFT with cuFFT

**Difficulty**: ★★

### Problem

Take a sine wave, compute its FFT, and verify the spectrum has a single peak at the expected frequency.

```cuda
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>

int main(void) {
    const int N = 1024;
    const int K = 50;            /* 50 cycles in N samples */

    cufftComplex *h = new cufftComplex[N];
    for (int i = 0; i < N; i++) {
        h[i].x = sinf(2.0f * (float)M_PI * K * i / N);
        h[i].y = 0.0f;
    }

    cufftComplex *d;
    cudaMalloc(&d, N * sizeof(cufftComplex));
    cudaMemcpy(d, h, N * sizeof(cufftComplex), cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_C2C, /*batch*/ 1);
    cufftExecC2C(plan, d, d, CUFFT_FORWARD);

    cudaMemcpy(h, d, N * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    /* Find the largest magnitude bin */
    float max_mag = 0; int max_bin = 0;
    for (int i = 0; i < N; i++) {
        float mag = sqrtf(h[i].x * h[i].x + h[i].y * h[i].y);
        if (mag > max_mag) { max_mag = mag; max_bin = i; }
    }
    printf("peak at bin %d (expected %d or %d)\n", max_bin, K, N - K);
    /* Real input → spectrum is conjugate-symmetric, peak shows at K AND N-K */

    cufftDestroy(plan);
    cudaFree(d); delete[] h;
    return 0;
}
```

The peak should appear at bin 50 (and the symmetric bin 974). If your peak is elsewhere, the most common cause is forgetting to normalize: cuFFT does NOT divide by N; an inverse FFT of the forward result returns the input × N.

---

## Exercise 22.2 — Batched 1D FFTs

**Difficulty**: ★★

In ML and signal-processing pipelines, you usually FFT many sequences at once. cuFFT batches them efficiently:

```cuda
const int batch = 1024;
cufftPlan1d(&plan, N, CUFFT_C2C, batch);
/* d points to a [batch * N] array, contiguous in N then batch */
cufftExecC2C(plan, d, d, CUFFT_FORWARD);
```

Time a single 16384-point FFT vs. 1024 batched 1024-point FFTs vs. 1024 individual 1024-point FFTs (loop). The batched version should be 5-50× faster than the loop because cuFFT amortizes setup and launches one optimized kernel.

---

## Exercise 22.3 — 2D FFT for Image Filtering

**Difficulty**: ★★★

Convolution in the spatial domain equals multiplication in the frequency domain. For a 1024×1024 image and a 1024×1024 PSF:

```
spatial conv (direct):    O(N^2 K^2)  = 1024^2 * 1024^2 = ~10^12 ops (impractical)
FFT-based conv:           O(N^2 log N) = 1024^2 * 10   = ~10^7 ops
```

Implement using `cufftPlan2d`. The pipeline:

1. FFT the image and the PSF (zero-padded to the same size).
2. Multiply elementwise.
3. Inverse FFT.

For small kernels (3×3, 5×5) the spatial conv wins; the crossover is around 11×11 to 15×15 depending on hardware.

---

## Exercise 22.4 — Real-to-Complex (R2C) — Bonus

**Difficulty**: ★★

For real input, the spectrum is conjugate-symmetric and you only need to store half. cuFFT's R2C plans output `N/2 + 1` complex values:

```cuda
cufftHandle plan;
cufftPlan1d(&plan, N, CUFFT_R2C, batch);
cufftExecR2C(plan, d_in_real, d_out_complex);
```

This halves both the FFT time and the output memory — which matters in audio (typically real input) and many physics codes.
