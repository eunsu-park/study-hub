# Lesson 9 — Convolution Backward Pass (per-lesson exercise)

Prerequisites: L05 (autograd), L08 (convolution forward).

Compile: `gcc -std=c11 -Wall -Wextra -O3 -o ex ex.c -lm`

The backward pass of convolution computes gradients with respect to the input, the weights, and (if any) the bias. Each is itself a convolution-like operation:

- $\partial L / \partial W$ is the convolution of the input with the gradient flowing back.
- $\partial L / \partial X$ is the convolution of the gradient with the rotated weight (a "transposed convolution").
- $\partial L / \partial b$ is the sum of the gradient over spatial dimensions.

---

## Exercise 9.1 — Bias Gradient (warm-up)

**Difficulty**: ★

### Problem

For an output gradient `dY` shape `[N, C_out, H_out, W_out]`, compute `db[c]` = sum over `n, h, w` of `dY[n, c, h, w]`. One sum per output channel.

```c
void conv2d_grad_bias(const float *dY, float *db,
                      int N, int C_out, int H_out, int W_out) {
    for (int c = 0; c < C_out; c++) {
        float sum = 0;
        for (int n = 0; n < N; n++)
            for (int h = 0; h < H_out; h++)
                for (int w = 0; w < W_out; w++)
                    sum += dY[((n * C_out + c) * H_out + h) * W_out + w];
        db[c] = sum;
    }
}
```

Verify: a bias of `1.0` added to a constant `dY = 1.0` should produce `db[c] = N * H_out * W_out`.

---

## Exercise 9.2 — Weight Gradient

**Difficulty**: ★★★

### Problem

The weight gradient is itself a convolution of the input with the upstream gradient. For the standard layout `X: [N, C_in, H, W]`, `dY: [N, C_out, H_out, W_out]`, the gradient `dW: [C_out, C_in, kH, kW]`:

$$\frac{\partial L}{\partial W_{co, ci, kh, kw}} = \sum_{n, h, w} dY_{n, co, h, w} \cdot X_{n, ci, h+kh, w+kw}$$

Implement this as 7 nested loops. The key insight: this is the SAME loop structure as forward conv, with $W$ and $dY$ swapped — convolution's backward is itself a convolution.

```c
void conv2d_grad_weight(const float *X, const float *dY, float *dW,
                        int N, int C_in, int H, int W,
                        int C_out, int kH, int kW) {
    int H_out = H - kH + 1;
    int W_out = W - kW + 1;
    /* zero dW first */
    /* TODO: 7-nested loop matching the formula above */
    (void)X; (void)dY; (void)dW;
    (void)N; (void)C_in; (void)H; (void)W;
    (void)C_out; (void)kH; (void)kW;
}
```

Verify with finite differences: pick a small (`N=1, C_in=1, C_out=1, kH=kW=2, H=W=4`) case, compute `dW` analytically, then perturb each weight by `1e-3` and confirm `(loss(W+h) - loss(W-h)) / (2h)` matches.

---

## Exercise 9.3 — Input Gradient

**Difficulty**: ★★★

### Problem

The input gradient is a "transposed convolution" of the weights with `dY`:

$$\frac{\partial L}{\partial X_{n, ci, y, x}} = \sum_{co, kh, kw} dY_{n, co, y - kh, x - kw} \cdot W_{co, ci, kh, kw}$$

(when the indexed `dY` element is in bounds; outside the valid range the contribution is zero.)

This is sometimes called a "deconvolution" — but it is not the inverse of convolution; it is just the gradient.

Implement as another 7-nested loop. Verify with finite differences on a small case.

---

## Exercise 9.4 — im2col Tricks Reused — Bonus

**Difficulty**: ★★★

Both gradients can be computed via GEMM by reusing the `im2col` matrix from the forward pass. Specifically:

- `dW = dY_cols @ X_im2col^T`
- `dX_im2col = W^T @ dY_cols`, then "col2im" gathers back to image shape.

Implement both. The total backward cost is then 2 GEMMs of the same size as the forward — making convolution training roughly 3× the cost of inference, matching what frameworks report.
