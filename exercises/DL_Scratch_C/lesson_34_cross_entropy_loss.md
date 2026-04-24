# Lesson 34 — Cross-Entropy Loss (per-lesson exercise)

Prerequisites: L05 (autograd), L27 (FFN/activations), basic probability.

Compile: `gcc -std=c11 -Wall -Wextra -O2 -o ex ex.c -lm`

For a model that outputs logits $z \in \mathbb{R}^V$ over $V$ classes and a true class index $y$, the categorical cross-entropy loss is:

$$L = -\log \frac{e^{z_y}}{\sum_v e^{z_v}}$$

Two essential implementation details:

1. **Numerical stability**: subtract `max(z)` before exponentiating. Without it, `expf(50) = inf` for typical vocab sizes.
2. **Fused softmax + cross-entropy**: the gradient is just `softmax(z) - one_hot(y)`. Computing softmax and cross-entropy separately is wasteful; fusing them is faster AND more stable.

---

## Exercise 34.1 — Forward Pass

**Difficulty**: ★

### Problem

Implement `float cross_entropy_loss(const float *z, int y, int V)` for a single example.

```c
#include <math.h>

float cross_entropy_loss(const float *z, int y, int V) {
    /* 1. m = max over z (for stability)
       2. log_sum_exp = m + logf(sum_v exp(z[v] - m))
       3. return log_sum_exp - z[y]
    */
    /* TODO */
    (void)z; (void)y; (void)V;
    return 0.0f;
}

int main(void) {
    /* Sanity: a perfectly confident correct prediction should have loss ≈ 0 */
    float z1[] = {100, 0, 0, 0, 0};
    printf("loss(confident correct) = %.4f (expect ≈ 0)\n", cross_entropy_loss(z1, 0, 5));

    /* Uniform logits → loss = log(V) */
    float z2[] = {1, 1, 1, 1, 1};
    printf("loss(uniform)            = %.4f (expect ≈ log(5) = %.4f)\n",
           cross_entropy_loss(z2, 0, 5), logf(5.0f));

    return 0;
}
```

---

## Exercise 34.2 — Fused Forward + Gradient

**Difficulty**: ★★

### Problem

For a batch of $B$ examples, implement:

```c
void cross_entropy_fused(const float *z,    /* [B, V] logits        */
                         const int *y,       /* [B]    true labels   */
                         float *grad_z,      /* [B, V] gradient out  */
                         float *loss_out,    /* [B]    per-example   */
                         int B, int V);
```

The gradient at row $b$ is `softmax(z[b]) - one_hot(y[b])`, divided by $B$ if you want the batch-averaged loss derivative.

This single function replaces a separate `softmax` + `cross_entropy` + manual backward — saving one pass over the `[B, V]` matrix and avoiding the intermediate buffer.

---

## Exercise 34.3 — Label Smoothing

**Difficulty**: ★★

Label smoothing replaces the one-hot target with a soft distribution:

$$q_v = \begin{cases}1 - \alpha + \alpha/V & v = y \\ \alpha/V & v \neq y\end{cases}$$

with $\alpha = 0.1$ typically. Implement `cross_entropy_smoothed` using the same fused pattern. Verify that with $\alpha = 0$ it reduces exactly to the unsmoothed version.

Why label smoothing helps: it discourages the model from producing logits that diverge to infinity, which improves calibration and slightly improves test accuracy on language modeling and image classification.

---

## Exercise 34.4 — Top-K Accuracy — Bonus

**Difficulty**: ★

Implement `int topk_correct(const float *z, int y, int V, int k)` that returns 1 if the true label `y` is among the top-`k` predicted classes by `z`. Useful for ImageNet-style "top-5 accuracy" reporting. Use `qsort` (or a partial-sort) on a copy of the logits with their original indices.
