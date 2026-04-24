# Lesson 26 — Image Processing on the GPU (per-lesson exercise)

Prerequisites: L05 (shared memory), L17 (stencil computations), L08 (memory coalescing).

Compile: `nvcc -O3 -arch=sm_80 ex.cu -o ex`

Image processing maps cleanly to the GPU: each output pixel is independent, and most kernels are stencils on the input. Three classic operations of increasing complexity.

---

## Exercise 26.1 — Box Blur (3×3 Average)

**Difficulty**: ★★

### Problem

Each output pixel is the average of itself and its 8 neighbors. The naive kernel reads 9 inputs per output:

```cuda
__global__ void box_blur(const unsigned char *in, unsigned char *out,
                         int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= W - 1 || y < 1 || y >= H - 1) return;

    int sum = 0;
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++)
            sum += in[(y + dy) * W + (x + dx)];
    out[y * W + x] = sum / 9;
}
```

For a 4096×4096 grayscale image this is bandwidth-bound. Use the tiled-stencil pattern from CUDA L17.2 to bring per-pixel reads from global memory down to ~1 (vs 9 in the naive version). Expect 4–7× speedup.

---

## Exercise 26.2 — Sobel Edge Detection

**Difficulty**: ★★

The Sobel operator computes horizontal and vertical gradients with two 3×3 kernels:

$$G_x = \begin{bmatrix}-1 & 0 & 1\\-2 & 0 & 2\\-1 & 0 & 1\end{bmatrix}, \quad G_y = \begin{bmatrix}-1 & -2 & -1\\0 & 0 & 0\\1 & 2 & 1\end{bmatrix}$$

Output: $|G_x| + |G_y|$ (or $\sqrt{G_x^2 + G_y^2}$ if you prefer).

Implement using the same tiled-stencil pattern as 26.1 — both Sobel kernels share the same input patch in shared memory, so loading the patch once and computing both Sobel responses in registers is roughly free.

---

## Exercise 26.3 — Separable Gaussian Blur

**Difficulty**: ★★★

A 2D Gaussian is **separable**: the 5×5 kernel can be computed as a horizontal 1×5 pass followed by a vertical 5×1 pass. Cost drops from 25 multiply-accumulates per pixel to 10.

```cuda
__constant__ float gauss5[5] = {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f};

__global__ void gauss_horizontal(const float *in, float *tmp, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 2 || x >= W - 2 || y < 0 || y >= H) return;
    float s = 0;
    for (int k = 0; k < 5; k++) s += gauss5[k] * in[y * W + (x + k - 2)];
    tmp[y * W + x] = s;
}

/* Then a similar gauss_vertical(tmp, out, ...) */
```

Compare to the equivalent non-separable 5×5 filter. Speedup is usually 2–3× — the smaller per-output work also reduces shared-memory pressure, allowing higher occupancy.

---

## Exercise 26.4 — Bilateral Filter — Bonus

**Difficulty**: ★★★★

Bilateral filtering preserves edges by weighting neighbors not just by spatial distance but by intensity similarity. Each output pixel is:

$$\text{out}(p) = \frac{\sum_{q \in \Omega} w(p, q) \cdot \text{in}(q)}{\sum_{q \in \Omega} w(p, q)}, \quad w(p, q) = e^{-\|p - q\|^2 / (2\sigma_s^2)} \cdot e^{-(I_p - I_q)^2 / (2\sigma_r^2)}$$

The intensity-dependent weight prevents averaging across edges. Implement the brute-force version with an 11×11 kernel; tile the input patch into shared memory the same way as 26.2; compute the dual-Gaussian weight inline. Expect 5-10× slower than box blur because each pixel's weights are different.

This is the canonical "edge-preserving denoise" used by every consumer camera's portrait mode.
