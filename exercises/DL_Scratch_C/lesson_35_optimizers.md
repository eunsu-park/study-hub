# Lesson 35 — Optimizers: SGD-Momentum and Adam (per-lesson exercise)

Prerequisites: L34 (cross-entropy & autograd), basic C pointer arithmetic, one-dimensional calculus.

Compile: `gcc -std=c11 -Wall -Wextra -O2 -o ex ex.c -lm`

Optimizers consume gradients and update parameters in place. The three standard choices:

- **SGD**: $\theta \leftarrow \theta - \eta g$
- **SGD-Momentum**: $v \leftarrow \mu v + g;\quad \theta \leftarrow \theta - \eta v$
- **Adam**: a bias-corrected moving average of the gradient and its square.

---

## Exercise 35.1 — SGD with Momentum

**Difficulty**: ★

### Problem

Implement `sgd_momentum_step(float *theta, const float *grad, float *v, int N, float lr, float mu)`. The arrays are flat; `N` is the total parameter count. `v` is the velocity buffer, maintained across calls.

### Starter Code

```c
#include <stdio.h>
#include <string.h>

void sgd_momentum_step(float *theta, const float *grad, float *v,
                       int N, float lr, float mu) {
    /* v[i] = mu * v[i] + grad[i];   theta[i] -= lr * v[i]; */
    /* TODO */
    (void)theta; (void)grad; (void)v; (void)N; (void)lr; (void)mu;
}

int main(void) {
    float theta[3] = {1.0f, 2.0f, 3.0f};
    float grad[3]  = {0.1f, 0.1f, 0.1f};
    float v[3]     = {0, 0, 0};

    /* Run 3 steps with constant gradient and observe the velocity building up. */
    for (int step = 0; step < 3; step++) {
        sgd_momentum_step(theta, grad, v, 3, 0.1f, 0.9f);
        printf("step %d: theta = [%.4f %.4f %.4f]  v = [%.4f %.4f %.4f]\n",
               step, theta[0], theta[1], theta[2], v[0], v[1], v[2]);
    }
    return 0;
}
```

### Expected (hand-verified)

```
step 0: theta = [0.9900 1.9900 2.9900]  v = [0.1000 0.1000 0.1000]
step 1: theta = [0.9710 1.9710 2.9710]  v = [0.1900 0.1900 0.1900]
step 2: theta = [0.9439 1.9439 2.9439]  v = [0.2710 0.2710 0.2710]
```

Notice the step size grows each iteration — this is momentum "accumulating" a consistent direction, which helps the optimizer traverse long shallow valleys in the loss surface.

---

## Exercise 35.2 — Adam

**Difficulty**: ★★★

### Problem

Implement `adam_step(float *theta, const float *grad, float *m, float *v, int N, int t, float lr, float beta1, float beta2, float eps)`. The buffers `m` and `v` are the first and second moment estimates; `t` is the step counter (1-indexed). Bias-corrected form:

```
m[i] = beta1 * m[i] + (1 - beta1) * grad[i];
v[i] = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i];
m_hat = m[i] / (1 - beta1^t);
v_hat = v[i] / (1 - beta2^t);
theta[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
```

### Starter Code

```c
#include <math.h>

void adam_step(float *theta, const float *grad, float *m, float *v,
               int N, int t, float lr, float beta1, float beta2, float eps) {
    /* TODO: implement the formula above for each i in [0, N) */
    (void)theta; (void)grad; (void)m; (void)v;
    (void)N; (void)t; (void)lr; (void)beta1; (void)beta2; (void)eps;
}
```

### Verification

Pick a simple 1-D problem: minimize $f(\theta) = \theta^2$. The gradient is $2\theta$. Starting at $\theta = 1.0$ with `lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8`, run 100 Adam steps and print $\theta$ at steps 0, 10, 50, 100. You should see $\theta$ converge toward 0 — not as fast as pure SGD for this convex problem, because Adam's per-parameter adaptive learning rate is overkill here, but it MUST converge.

---

## Exercise 35.3 — Numerical Check Against Finite Differences

**Difficulty**: ★★

Before trusting your optimizer, sanity-check it on a problem whose exact minimum you already know. Minimize $f(\theta_1, \theta_2) = 3\theta_1^2 + \theta_2^2$ (exact min at origin). The analytical gradient is $(6\theta_1, 2\theta_2)$. Run 200 Adam steps from initial $\theta = (5, -3)$ and confirm the final $\theta$ is within $0.01$ of the origin. If it is not, inspect whether the sign of your gradient descent step is correct — that is the single most common bug.
