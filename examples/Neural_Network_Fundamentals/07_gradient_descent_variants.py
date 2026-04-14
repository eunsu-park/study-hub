"""
07. Gradient Descent Variants
===============================
Implements SGD, SGD+Momentum, RMSProp, and Adam optimizers,
and compares their convergence on a 2D optimization problem.

Key Concepts:
  - Vanilla SGD, Momentum, RMSProp, Adam
  - Bias correction in Adam
  - Learning rate scheduling
"""

import numpy as np


# ============================================================
# 1. Test Function: Rosenbrock-like
# ============================================================
print("=" * 60)
print("Gradient Descent Variants Comparison")
print("=" * 60)


def loss_fn(params):
    """2D quadratic with elongated contours: f(x,y) = 10*(y-x^2)^2 + (1-x)^2"""
    x, y = params
    return 10 * (y - x**2)**2 + (1 - x)**2

def grad_fn(params):
    """Gradient of the loss function."""
    x, y = params
    dx = -40 * x * (y - x**2) - 2 * (1 - x)
    dy = 20 * (y - x**2)
    return np.array([dx, dy])


# ============================================================
# 2. Optimizer Implementations
# ============================================================

def sgd(grad_fn, x0, lr=0.001, n_steps=500):
    x = x0.copy()
    history = [loss_fn(x)]
    for _ in range(n_steps):
        g = grad_fn(x)
        x -= lr * g
        history.append(loss_fn(x))
    return x, history

def sgd_momentum(grad_fn, x0, lr=0.001, beta=0.9, n_steps=500):
    x = x0.copy()
    v = np.zeros_like(x)
    history = [loss_fn(x)]
    for _ in range(n_steps):
        g = grad_fn(x)
        v = beta * v + (1 - beta) * g
        x -= lr * v
        history.append(loss_fn(x))
    return x, history

def rmsprop(grad_fn, x0, lr=0.001, beta=0.999, eps=1e-8, n_steps=500):
    x = x0.copy()
    s = np.zeros_like(x)
    history = [loss_fn(x)]
    for _ in range(n_steps):
        g = grad_fn(x)
        s = beta * s + (1 - beta) * g**2
        x -= lr * g / (np.sqrt(s) + eps)
        history.append(loss_fn(x))
    return x, history

def adam(grad_fn, x0, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, n_steps=500):
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    history = [loss_fn(x)]
    for t in range(1, n_steps + 1):
        g = grad_fn(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x -= lr * m_hat / (np.sqrt(v_hat) + eps)
        history.append(loss_fn(x))
    return x, history


# ============================================================
# 3. Run Comparison
# ============================================================
x0 = np.array([-1.0, 1.0])
n_steps = 1000

print(f"\nStarting point: {x0}")
print(f"Optimal: [1.0, 1.0], loss = 0.0")
print(f"Steps: {n_steps}\n")

results = {}
for name, opt_fn, kwargs in [
    ("SGD",          sgd,          {"lr": 0.001}),
    ("SGD+Momentum", sgd_momentum, {"lr": 0.001, "beta": 0.9}),
    ("RMSProp",      rmsprop,      {"lr": 0.01}),
    ("Adam",         adam,         {"lr": 0.01}),
]:
    x_final, hist = opt_fn(grad_fn, x0, n_steps=n_steps, **kwargs)
    results[name] = hist
    print(f"  {name:15s}: final loss = {hist[-1]:.6f}, "
          f"x = [{x_final[0]:.4f}, {x_final[1]:.4f}]")


# ============================================================
# 4. Learning Rate Scheduling
# ============================================================
print("\n" + "=" * 60)
print("Learning Rate Schedules")
print("=" * 60)


def step_decay(epoch, lr_init=0.1, gamma=0.1, step_size=30):
    return lr_init * gamma ** (epoch // step_size)

def cosine_annealing(epoch, lr_max=0.1, lr_min=1e-6, T=100):
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * epoch / T))

def warmup_cosine(epoch, lr_max=0.001, lr_min=1e-6, warmup=10, total=100):
    if epoch < warmup:
        return lr_max * epoch / warmup
    progress = (epoch - warmup) / (total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * progress))


print("\nStep Decay (init=0.1, gamma=0.1, step=30):")
for e in [0, 15, 30, 60, 90]:
    print(f"  Epoch {e:3d}: lr = {step_decay(e):.6f}")

print("\nCosine Annealing (max=0.1, T=100):")
for e in [0, 25, 50, 75, 100]:
    print(f"  Epoch {e:3d}: lr = {cosine_annealing(e):.6f}")

print("\nWarmup + Cosine (warmup=10, total=100):")
for e in [0, 5, 10, 50, 100]:
    print(f"  Epoch {e:3d}: lr = {warmup_cosine(e):.6f}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
