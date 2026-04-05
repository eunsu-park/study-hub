# 26. Image Processing on GPU

**Previous**: [Molecular Dynamics](./25_Molecular_Dynamics.md) | **Next**: [Random Number and Stochastic](./27_Random_Number_and_Stochastic.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement separable Gaussian blur using two independent 1D convolution passes
2. Implement Sobel edge detection (Gx, Gy magnitude and angle)
3. Implement bilateral filtering for edge-preserving smoothing
4. Apply global histogram equalization and describe CLAHE for local equalization
5. Use CUDA texture memory to accelerate image kernels with hardware interpolation and boundary handling

---

## 1. Why GPU for Image Processing?

Images are inherently 2D arrays where each pixel's output depends only on a fixed neighborhood — the same structure as stencil computations. GPUs map pixels to threads, and the regularity of image layouts enables excellent memory coalescing.

```
Typical image kernel characteristics:
  Input:  H × W × C pixels (e.g., 4K = 3840×2160×3)
  Work:   Each output pixel reads K×K input pixels
  Parallelism: H × W independent outputs  → perfect for GPU

For 4K 3×3 kernel (direct):
  3840 × 2160 × 9 reads = 74.6M memory accesses per channel
  GPU: ~1 ms @ 900 GB/s;  CPU: ~50 ms (10 threads) → 50× GPU speedup
```

---

## 2. Separable Gaussian Blur

A 2D Gaussian filter G(x,y) = G(x) × G(y) is separable — apply 1D convolution horizontally, then vertically. This reduces O(K²) to O(2K) operations per pixel:

```c
// Horizontal 1D Gaussian pass
__global__ void gaussian_h(
    const uchar *in, uchar *out, int W, int H,
    const float *kernel, int K)  // K = radius (kernel has 2K+1 taps)
{
    __shared__ uchar sdata[16][16 + 2 * 8];  // 8 = max K

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // Load tile + left/right halo into shared memory
    sdata[ty][tx + K] = (x < W && y < H) ? in[y * W + x] : 0;
    if (tx < K) {
        sdata[ty][tx] = (x - K >= 0 && y < H) ? in[y * W + (x - K)] : 0;
        sdata[ty][tx + blockDim.x + K] =
            (x + blockDim.x < W && y < H) ? in[y * W + (x + blockDim.x)] : 0;
    }
    __syncthreads();

    if (x >= W || y >= H) return;

    float sum = 0.f;
    for (int k = -K; k <= K; k++)
        sum += kernel[k + K] * sdata[ty][tx + K + k];

    out[y * W + x] = (uchar)min(255.f, max(0.f, sum + 0.5f));
}

// Generate Gaussian kernel on host
void make_gaussian_kernel(float *kernel, int K, float sigma) {
    float sum = 0.f;
    for (int k = -K; k <= K; k++) {
        kernel[k + K] = expf(-0.5f * k * k / (sigma * sigma));
        sum += kernel[k + K];
    }
    for (int k = 0; k <= 2 * K; k++) kernel[k] /= sum;
}

// Full separable Gaussian blur
void gaussian_blur(const uchar *d_in, uchar *d_out, int W, int H,
                   float sigma, int K) {
    float h_kern[17];  // max K=8
    make_gaussian_kernel(h_kern, K, sigma);
    float *d_kern; cudaMalloc(&d_kern, (2*K+1) * sizeof(float));
    cudaMemcpy(d_kern, h_kern, (2*K+1)*sizeof(float), cudaMemcpyHostToDevice);

    uchar *d_tmp; cudaMalloc(&d_tmp, W * H);

    dim3 block(16, 16), grid((W+15)/16, (H+15)/16);
    gaussian_h<<<grid, block>>>(d_in,  d_tmp, W, H, d_kern, K);  // horizontal
    gaussian_v<<<grid, block>>>(d_tmp, d_out, W, H, d_kern, K);  // vertical

    cudaFree(d_tmp); cudaFree(d_kern);
}
```

---

## 3. Sobel Edge Detection

The Sobel operator estimates the image gradient magnitude:

```
Gx kernel:    Gy kernel:
[-1  0 +1]   [+1 +2 +1]
[-2  0 +2]   [ 0  0  0]
[-1  0 +1]   [-1 -2 -1]

Gradient magnitude: M = sqrt(Gx² + Gy²)
Gradient angle:     θ = atan2(Gy, Gx)
```

```c
__global__ void sobel_edge(
    const uchar *in, uchar *out_mag, float *out_angle,
    int W, int H)
{
    __shared__ uchar s[18][18];  // 16×16 + 1-cell halo

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x + 1, ty = threadIdx.y + 1;

    // Load interior
    s[ty][tx] = (x < W && y < H) ? in[y * W + x] : 0;
    // Load halos (top, bottom, left, right corners omitted for brevity)
    if (threadIdx.y == 0)
        s[0][tx] = (y > 0 && x < W) ? in[(y-1) * W + x] : 0;
    if (threadIdx.y == 15)
        s[17][tx] = (y+1 < H && x < W) ? in[(y+1) * W + x] : 0;
    if (threadIdx.x == 0)
        s[ty][0] = (x > 0 && y < H) ? in[y * W + (x-1)] : 0;
    if (threadIdx.x == 15)
        s[ty][17] = (x+1 < W && y < H) ? in[y * W + (x+1)] : 0;
    __syncthreads();

    if (x >= W || y >= H) return;

    float gx = -s[ty-1][tx-1] + s[ty-1][tx+1]
               -2*s[ty][tx-1] + 2*s[ty][tx+1]
               -s[ty+1][tx-1] + s[ty+1][tx+1];
    float gy =  s[ty-1][tx-1] + 2*s[ty-1][tx] + s[ty-1][tx+1]
               -s[ty+1][tx-1] - 2*s[ty+1][tx] - s[ty+1][tx+1];

    float mag = sqrtf(gx*gx + gy*gy);
    out_mag[y * W + x] = (uchar)min(255.f, mag);
    if (out_angle) out_angle[y * W + x] = atan2f(gy, gx);
}
```

---

## 4. Bilateral Filter (Edge-Preserving Smoothing)

Bilateral filtering is a non-linear, edge-preserving smoothing filter. Unlike Gaussian blur, it weighs neighbors by both spatial distance AND intensity difference, preserving edges while smoothing flat regions:

```
BF(I)[p] = (1/W_p) Σ_q G_s(|p-q|) · G_r(|I[p] - I[q]|) · I[q]

G_s: spatial Gaussian (σ_s)
G_r: range Gaussian  (σ_r = intensity sensitivity)
W_p: normalization factor

When σ_r → ∞: bilateral → Gaussian
When σ_r → 0: bilateral → identity (no smoothing at edges)
```

```c
__global__ void bilateral_filter(
    const float *in, float *out, int W, int H,
    float sigma_s, float sigma_r, int K)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    float center = in[y * W + x];
    float inv_2ss = 1.0f / (2.f * sigma_s * sigma_s);
    float inv_2sr = 1.0f / (2.f * sigma_r * sigma_r);

    float sum_w = 0.f, sum_wI = 0.f;

    for (int dy = -K; dy <= K; dy++) {
        int yy = min(max(y + dy, 0), H - 1);
        for (int dx = -K; dx <= K; dx++) {
            int xx = min(max(x + dx, 0), W - 1);
            float neighbor = in[yy * W + xx];

            float spatial = (dx*dx + dy*dy) * inv_2ss;
            float range   = (center - neighbor) * (center - neighbor) * inv_2sr;
            float weight  = expf(-(spatial + range));

            sum_w  += weight;
            sum_wI += weight * neighbor;
        }
    }
    out[y * W + x] = sum_wI / sum_w;
}
```

**Performance note**: bilateral filter is not separable — must use a 2D neighborhood. Use shared memory tile + halo for large K (K > 5). For real-time performance, use the bilateral grid approximation (separate into spatial-range histogram, then interpolate).

---

## 5. Histogram Equalization

Global histogram equalization redistributes pixel intensities to make the histogram approximately uniform — enhancing contrast:

```c
// Step 1: compute histogram (256 bins for grayscale)
// Step 2: compute CDF via exclusive scan
// Step 3: apply CDF as LUT to remap pixels

__global__ void apply_equalization(
    const uchar *in, uchar *out,
    const float *cdf,   // cumulative distribution function [256]
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    // CDF is normalized to [0, 255]; round to nearest integer
    out[i] = (uchar)(255.f * cdf[in[i]] + 0.5f);
}

// Host: full pipeline
void histogram_equalize(const uchar *d_in, uchar *d_out, int W, int H) {
    int N = W * H;

    // Step 1: histogram (use shared-mem privatized from Lesson 18)
    int *d_hist; cudaMalloc(&d_hist, 256 * sizeof(int));
    cudaMemset(d_hist, 0, 256 * sizeof(int));
    histogram_smem<<<min((N+255)/256, 1024), 256, 256*sizeof(int)>>>(
        d_in, d_hist, N, 256);

    // Step 2: CDF via inclusive scan
    float *d_cdf; cudaMalloc(&d_cdf, 256 * sizeof(float));
    // First convert hist to float and normalize by N
    normalize_hist<<<1, 256>>>(d_hist, d_cdf, N);
    // Then inclusive prefix sum on d_cdf
    cub_inclusive_scan_float(d_cdf, 256);

    // Step 3: apply CDF remapping
    apply_equalization<<<(N+255)/256, 256>>>(d_in, d_out, d_cdf, N);

    cudaFree(d_hist); cudaFree(d_cdf);
}
```

**CLAHE (Contrast Limited Adaptive Histogram Equalization)**: divides the image into tiles (e.g., 8×8), equalizes each tile's histogram independently, clips the histogram at a contrast limit to prevent noise amplification, then bilinearly interpolates the local transforms. CLAHE is the standard for medical imaging contrast enhancement.

---

## 6. Texture Memory for Image Kernels

CUDA texture memory provides hardware-accelerated 2D caching, bilinear interpolation, and automatic boundary handling — ideal for image operations:

```c
// Declare a 2D texture
cudaTextureObject_t create_image_texture(const uchar *d_img, int W, int H) {
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr    = (void*)d_img;
    resDesc.res.pitch2D.width     = W;
    resDesc.res.pitch2D.height    = H;
    resDesc.res.pitch2D.pitchInBytes = W * sizeof(uchar);
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<uchar>();

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;  // clamp to border
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode     = cudaFilterModeLinear;  // bilinear interpolation
    texDesc.readMode       = cudaReadModeNormalizedFloat;  // [0,1] output

    cudaTextureObject_t tex;
    cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);
    return tex;
}

// Kernel using texture: hardware handles clamping and interpolation
__global__ void blur_with_texture(
    cudaTextureObject_t tex, float *out, int W, int H,
    const float *kernel, int K)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    // Normalized texture coordinates: (x+0.5)/W maps to center of pixel x
    float u0 = (x + 0.5f) / W;
    float v0 = (y + 0.5f) / H;
    float du = 1.0f / W, dv = 1.0f / H;

    float sum = 0.f;
    for (int dy = -K; dy <= K; dy++)
        for (int dx = -K; dx <= K; dx++)
            sum += kernel[(dy+K)*(2*K+1)+(dx+K)] *
                   tex2D<float>(tex, u0 + dx*du, v0 + dy*dv);

    out[y * W + x] = sum;
}
```

Texture memory benefits for images:
- 2D spatial cache (L1 hit for 2D neighborhood access pattern)
- Hardware bilinear interpolation (free, 1 cycle)
- Automatic boundary clamping, mirroring, or wrapping
- No bank conflicts (separate from shared memory)

---

## Key Takeaways

- **Separable Gaussian blur** decomposes a 2D K×K convolution into two 1D passes, reducing O(K²N) to O(2KN) — always prefer separable implementations
- **Sobel edge detection** applies two 3×3 convolutions (Gx, Gy) and combines magnitude; use shared memory tiles for the neighborhood reads
- **Bilateral filter** preserves edges by weighting neighbors by intensity similarity; it is not separable, so is expensive for large K
- **Histogram equalization** = compute 256-bin histogram → inclusive scan → apply CDF LUT; entire pipeline runs on GPU without CPU round trips
- **Texture memory** provides hardware 2D caching, free bilinear interpolation, and automatic boundary modes — reduces register pressure vs manual clamping
- CLAHE (adaptive equalization) tiles the image and clips histogram contrast per tile — the gold standard for medical image contrast enhancement

---

**Next**: [27. Random Number and Stochastic](./27_Random_Number_and_Stochastic.md) — Explore quasi-random Sobol sequences, parallel MCMC with Metropolis-Hastings, and GPU-accelerated Bayesian inference.
