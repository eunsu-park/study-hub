"""
Optimization Theory

Demonstrates optimization algorithms and convergence:
- Gradient descent, momentum, Adam on the Rosenbrock function
- Convergence rate vs condition number
- Learning rate schedules (cosine, warmup)

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


def rosenbrock(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

def grad_rosenbrock(x):
    return np.array([
        -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
        200*(x[1] - x[0]**2)
    ])


def compare_optimizers():
    """Compare SGD, Momentum, and Adam on Rosenbrock."""
    print("=" * 60)
    print("OPTIMIZER COMPARISON ON ROSENBROCK")
    print("=" * 60)

    x0 = np.array([-1.0, 1.0])
    n_steps = 2000

    # SGD
    x = x0.copy()
    for _ in range(n_steps):
        x -= 0.0005 * grad_rosenbrock(x)
    print(f"SGD:      x = {x.round(4)}, f = {rosenbrock(x):.6f}")

    # Momentum
    x, v = x0.copy(), np.zeros(2)
    for _ in range(n_steps):
        v = 0.9 * v + grad_rosenbrock(x)
        x -= 0.0005 * v
    print(f"Momentum: x = {x.round(4)}, f = {rosenbrock(x):.6f}")

    # Adam
    x = x0.copy()
    m, v = np.zeros(2), np.zeros(2)
    for t in range(1, n_steps + 1):
        g = grad_rosenbrock(x)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g**2
        mh = m / (1 - 0.9**t)
        vh = v / (1 - 0.999**t)
        x -= 0.01 * mh / (np.sqrt(vh) + 1e-8)
    print(f"Adam:     x = {x.round(4)}, f = {rosenbrock(x):.6f}")
    print(f"Optimum:  x = [1, 1], f = 0")


def lr_schedules():
    """Demonstrate learning rate schedules."""
    print("\n" + "=" * 60)
    print("LEARNING RATE SCHEDULES")
    print("=" * 60)

    T = 1000
    t = np.arange(T)
    eta0 = 0.01

    schedules = {
        'Constant': np.full(T, eta0),
        'Step (x0.1 @300,600)': eta0 * 0.1 ** (t // 300),
        'Cosine': eta0/2 * (1 + np.cos(np.pi * t / T)),
        'Warmup+Cosine': np.where(t < 100, eta0*t/100,
                                   eta0/2*(1+np.cos(np.pi*(t-100)/(T-100)))),
    }

    for name, lr in schedules.items():
        print(f"  {name:25s}: start={lr[0]:.5f}, mid={lr[T//2]:.5f}, end={lr[-1]:.5f}")


if __name__ == "__main__":
    compare_optimizers()
    lr_schedules()
