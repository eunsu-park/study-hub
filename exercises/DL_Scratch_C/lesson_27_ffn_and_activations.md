# Lesson 27 — FFN and Activations (per-lesson exercise)

Prerequisites: L03 (BLAS basics), L05 (autograd).

Compile: `gcc -std=c11 -Wall -Wextra -O2 -o ex ex.c -lm`

The transformer FFN is two linear layers around a non-linearity:

$$\text{FFN}(x) = W_2 \cdot \phi(W_1 x + b_1) + b_2$$

Where $\phi$ has been GELU (GPT-2/3), ReLU (older), and SwiGLU (modern open LLMs). The choice matters for both quality and compute.

---

## Exercise 27.1 — Activation Functions and Their Gradients

**Difficulty**: ★

Implement four scalar activations and their gradients:

```c
float relu(float x);              /* max(0, x)              */
float relu_grad(float x);         /* x > 0 ? 1 : 0          */
float gelu(float x);              /* GPT-2 approximation    */
float gelu_grad(float x);         /* derivative             */
float silu(float x);              /* x * sigmoid(x)         */
float silu_grad(float x);
float sigmoid(float x);           /* 1 / (1 + exp(-x))      */
```

Verify each gradient with finite differences on three test points: $x \in \{-2, 0, 3\}$.

### Starter

```c
#include <math.h>
#include <stdio.h>

float relu(float x) { return x > 0 ? x : 0; }
float relu_grad(float x) { return x > 0 ? 1.0f : 0.0f; }

float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

float silu(float x) { return x * sigmoid(x); }
float silu_grad(float x) {
    float s = sigmoid(x);
    return s + x * s * (1.0f - s);
}

float gelu(float x) {
    /* GPT-2 approximation: 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))) */
    float c = 0.7978845608f;     /* sqrt(2/pi) */
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}

float gelu_grad(float x) {
    /* TODO: derive analytically and implement */
    (void)x;
    return 0.0f;
}

int main(void) {
    /* Compare analytic vs finite-difference for each activation */
    float test[] = {-2.0f, 0.0f, 3.0f};
    for (int i = 0; i < 3; i++) {
        float x = test[i];
        float h = 1e-3f;
        float fd = (silu(x + h) - silu(x - h)) / (2 * h);
        printf("silu'(%.1f): analytic=%+.4f, finite-diff=%+.4f\n",
               x, silu_grad(x), fd);
    }
    return 0;
}
```

---

## Exercise 27.2 — A Full FFN Forward Pass

**Difficulty**: ★★

### Problem

Implement `ffn_forward(const float *x, const float *W1, const float *b1, const float *W2, const float *b2, float *y, int d_in, int d_hidden, int d_out)`:

1. `h1 = W1 @ x + b1`           shape `[d_hidden]`
2. `h2 = gelu(h1)`              elementwise
3. `y  = W2 @ h2 + b2`          shape `[d_out]`

Reuse a small `matvec` helper (matrix-vector multiply) — this is the only routine that touches the weight matrices.

---

## Exercise 27.3 — SwiGLU Variant

**Difficulty**: ★★★

SwiGLU replaces the GELU FFN with:

$$\text{FFN}(x) = W_2 \cdot \big(\text{SiLU}(W_a x) \odot (W_b x)\big)$$

Three weight matrices instead of two — but the hidden dimension is usually scaled to $\frac{2}{3} \cdot 4d$ to match the parameter count of the GELU variant.

Implement and compare parameter counts and FLOP counts for `d_model = 4096`, GELU FFN with `d_hidden = 16384`, vs SwiGLU with `d_hidden = 10923`.

---

## Exercise 27.4 — Activation Choice Sanity Check — Bonus

**Difficulty**: ★★

Train a tiny MLP (2 hidden layers, 64 units) on a synthetic regression task with each of {ReLU, GELU, SiLU} as the activation. Compare final training loss after 1000 steps with the same optimizer and seed. Differences should be small (<5% relative) — the activation matters less than people think for a well-conditioned problem. The bigger gaps appear at scale, where SiLU/GELU's smooth gradient near zero helps optimization.
