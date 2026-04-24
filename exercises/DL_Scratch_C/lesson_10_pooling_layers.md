# Lesson 10 — Pooling Layers (per-lesson exercise)

Prerequisites: L02 (memory layout), L08 (convolution).

Compile: `gcc -std=c11 -Wall -Wextra -O2 -o ex ex.c -lm`

Pooling reduces spatial resolution by aggregating each `k × k` window. Two common variants:

- **Max pooling**: take the maximum activation in each window. Sparse gradient (only the argmax channel gets the gradient).
- **Average pooling**: take the mean. Dense gradient (every input contributes equally).

Pooling is parameter-free, computationally cheap, and the standard "downsample" operation in CNNs since 1989.

---

## Exercise 10.1 — Max Pooling Forward

**Difficulty**: ★★

### Problem

Implement `max_pool_2d(input, output, indices, N, C, H, W, k_size, stride)`. Storing the argmax index per output element makes the backward pass trivial.

```c
void max_pool_2d(const float *in, float *out, int *indices,
                 int N, int C, int H, int W, int k, int stride) {
    int H_out = (H - k) / stride + 1;
    int W_out = (W - k) / stride + 1;

    for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
            for (int h = 0; h < H_out; h++)
                for (int w = 0; w < W_out; w++) {
                    float maxv = -INFINITY;
                    int   maxi = 0;
                    for (int dy = 0; dy < k; dy++)
                        for (int dx = 0; dx < k; dx++) {
                            int yy = h * stride + dy;
                            int xx = w * stride + dx;
                            float v = in[((n * C + c) * H + yy) * W + xx];
                            if (v > maxv) { maxv = v; maxi = yy * W + xx; }
                        }
                    int idx = ((n * C + c) * H_out + h) * W_out + w;
                    out[idx]     = maxv;
                    indices[idx] = maxi;
                }
}
```

For `k=2, stride=2` on a `28 × 28` input, the output is `14 × 14` — exactly half resolution per axis, one-quarter total elements.

---

## Exercise 10.2 — Max Pool Backward

**Difficulty**: ★★

### Problem

The gradient flows back ONLY to the position that "won" the max. Use the `indices` array stored during forward:

```c
void max_pool_2d_backward(const float *dY, const int *indices,
                          float *dX, int N, int C, int H, int W,
                          int H_out, int W_out) {
    /* zero dX first */
    for (int i = 0; i < N * C * H * W; i++) dX[i] = 0;

    for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
            for (int h = 0; h < H_out; h++)
                for (int w = 0; w < W_out; w++) {
                    int idx_out = ((n * C + c) * H_out + h) * W_out + w;
                    int idx_in_spatial = indices[idx_out];          /* yy * W + xx */
                    int idx_in = (n * C + c) * H * W + idx_in_spatial;
                    dX[idx_in] += dY[idx_out];
                }
}
```

The `+=` is critical because overlapping pool windows (stride < k) can produce the same input position as the argmax for multiple outputs — gradients accumulate.

---

## Exercise 10.3 — Average Pooling

**Difficulty**: ★

Average pool forward: average over the `k × k` window. Backward: each input in the window receives `dY / (k * k)`.

Average pool gradients are smoother (every input contributes), which is sometimes preferred in deeper networks. Modern architectures often use **global average pooling** (average over the whole spatial extent) at the network's end — replacing the giant FC layer that AlexNet used and saving millions of parameters.

---

## Exercise 10.4 — Adaptive Pooling — Bonus

**Difficulty**: ★★

PyTorch's `nn.AdaptiveAvgPool2d(output_size)` lets you specify the OUTPUT size; the pool window is computed automatically. Implement:

```c
void adaptive_avg_pool_2d(const float *in, float *out,
                          int N, int C, int H, int W,
                          int H_out, int W_out);
```

For each output position `(h, w)`, the input window is `[h * H / H_out, (h + 1) * H / H_out)` along each axis. This handles non-divisible cases gracefully — useful when the input size varies (e.g., variable-resolution images).
