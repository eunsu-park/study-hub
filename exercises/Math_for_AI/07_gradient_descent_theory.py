"""
Exercises for Lesson 07: Gradient Descent Theory
Topic: Math_for_AI

Solutions to practice problems from the lesson.
"""

import numpy as np
import time


# === Exercise 1: Convergence Rate Analysis ===
# Problem: For f(x) = 0.5 x^T A x, compare convergence rates when
# condition number kappa = 10 vs kappa = 100.

def exercise_1():
    """Compare GD convergence rates for different condition numbers."""
    np.random.seed(42)

    def run_gd_quadratic(kappa, dim=10, n_steps=500):
        """Run GD on quadratic form with given condition number."""
        # Create matrix with specified condition number
        eigenvalues = np.linspace(1, kappa, dim)
        # Random orthogonal matrix
        Q, _ = np.linalg.qr(np.random.randn(dim, dim))
        A = Q @ np.diag(eigenvalues) @ Q.T

        # Gradient descent on f(x) = 0.5 x^T A x
        x = np.random.randn(dim)
        f_values = [0.5 * x @ A @ x]

        lr = 2.0 / (eigenvalues[0] + eigenvalues[-1])  # optimal for quadratic

        for _ in range(n_steps):
            grad = A @ x
            x = x - lr * grad
            f_values.append(0.5 * x @ A @ x)

        return np.array(f_values)

    n_steps = 500

    for kappa in [10, 100]:
        losses = run_gd_quadratic(kappa, n_steps=n_steps)

        # Theoretical rate: (1 - 1/kappa)^t
        theoretical_rate = (1 - 2.0 / (kappa + 1)) ** 2  # per-step ratio
        actual_rate = losses[-1] / losses[0] if losses[0] > 0 else 0

        print(f"Condition number kappa = {kappa}:")
        print(f"  Optimal learning rate: {2.0 / (1 + kappa):.6f}")
        print(f"  Loss at step 0: {losses[0]:.6f}")
        print(f"  Loss at step 50: {losses[50]:.6f}")
        print(f"  Loss at step 200: {losses[200]:.6f}")
        print(f"  Loss at step {n_steps}: {losses[-1]:.2e}")
        print(f"  Theoretical convergence rate per step: {theoretical_rate:.6f}")

        # Estimate actual rate from data
        if losses[1] > 1e-15 and losses[0] > 1e-15:
            empirical_rate = (losses[100] / losses[0]) ** (1.0 / 100)
            print(f"  Empirical convergence rate per step: {empirical_rate:.6f}")
        print()


# === Exercise 2: SGD vs Batch GD ===
# Problem: Compare batch GD and mini-batch SGD on a simple task.

def exercise_2():
    """Compare SGD with different batch sizes on linear regression."""
    np.random.seed(42)

    # Generate data
    n = 1000
    d = 10
    X = np.random.randn(n, d)
    true_w = np.random.randn(d)
    y = X @ true_w + np.random.randn(n) * 0.5

    def mse_loss(w, X, y):
        return 0.5 * np.mean((y - X @ w) ** 2)

    def mse_gradient(w, X_batch, y_batch):
        return -X_batch.T @ (y_batch - X_batch @ w) / len(y_batch)

    def train(batch_size, n_epochs=50, lr=0.01):
        w = np.zeros(d)
        losses = []
        start_time = time.time()

        for epoch in range(n_epochs):
            # Shuffle data
            perm = np.random.permutation(n)
            X_shuffled = X[perm]
            y_shuffled = y[perm]

            for i in range(0, n, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                grad = mse_gradient(w, X_batch, y_batch)
                w -= lr * grad

            losses.append(mse_loss(w, X, y))

        elapsed = time.time() - start_time
        return w, losses, elapsed

    # Optimal solution for reference
    w_optimal = np.linalg.lstsq(X, y, rcond=None)[0]
    optimal_loss = mse_loss(w_optimal, X, y)

    print(f"Optimal loss (analytical): {optimal_loss:.6f}")
    print()

    batch_sizes = [1, 32, 128, 512, n]
    batch_names = ['Full SGD (bs=1)', 'Mini-batch (bs=32)',
                   'Mini-batch (bs=128)', 'Mini-batch (bs=512)',
                   f'Batch GD (bs={n})']

    for bs, name in zip(batch_sizes, batch_names):
        w, losses, elapsed = train(bs, n_epochs=50, lr=0.01)
        dist_to_opt = np.linalg.norm(w - w_optimal)
        print(f"{name}:")
        print(f"  Final loss: {losses[-1]:.6f} (gap: {losses[-1]-optimal_loss:.6f})")
        print(f"  Distance to optimal w: {dist_to_opt:.6f}")
        print(f"  Time: {elapsed:.4f}s")
        print()


# === Exercise 3: Adam Optimizer Implementation ===
# Problem: Implement Adam from scratch and test on Rosenbrock function.

def exercise_3():
    """Adam optimizer implementation and comparison on Rosenbrock."""
    def rosenbrock(x):
        return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

    def grad_rosenbrock(x):
        dx = -2*(1-x[0]) - 400*x[0]*(x[1]-x[0]**2)
        dy = 200*(x[1]-x[0]**2)
        return np.array([dx, dy])

    def adam(x0, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, n_steps=5000):
        x = np.array(x0, dtype=float)
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        losses = [rosenbrock(x)]

        for t in range(1, n_steps + 1):
            g = grad_rosenbrock(x)
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g**2

            # With bias correction
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            x -= lr * m_hat / (np.sqrt(v_hat) + eps)
            losses.append(rosenbrock(x))

        return x, losses

    def adam_no_correction(x0, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, n_steps=5000):
        x = np.array(x0, dtype=float)
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        losses = [rosenbrock(x)]

        for t in range(1, n_steps + 1):
            g = grad_rosenbrock(x)
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g**2

            # Without bias correction
            x -= lr * m / (np.sqrt(v) + eps)
            losses.append(rosenbrock(x))

        return x, losses

    x0 = [-1.0, 1.0]
    n_steps = 10000

    x_adam, losses_adam = adam(x0, lr=0.01, n_steps=n_steps)
    x_no_corr, losses_no_corr = adam_no_correction(x0, lr=0.01, n_steps=n_steps)

    print("Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2")
    print(f"Start: {x0}, Optimal: (1, 1)")
    print()

    print("Adam with bias correction:")
    for step in [0, 100, 1000, 5000, n_steps]:
        print(f"  Step {step:5d}: loss = {losses_adam[step]:.6f}")
    print(f"  Final: {np.round(x_adam, 4)}")

    print()
    print("Adam without bias correction:")
    for step in [0, 100, 1000, 5000, n_steps]:
        print(f"  Step {step:5d}: loss = {losses_no_corr[step]:.6f}")
    print(f"  Final: {np.round(x_no_corr, 4)}")

    print()
    print("Early iterations comparison (bias correction matters most here):")
    for step in [1, 5, 10, 20, 50]:
        print(f"  Step {step:3d}: corrected={losses_adam[step]:.4f}, "
              f"uncorrected={losses_no_corr[step]:.4f}")


# === Exercise 4: Momentum vs Nesterov ===
# Problem: Compare momentum SGD and Nesterov on ill-conditioned quadratic.

def exercise_4():
    """Compare momentum and Nesterov accelerated gradient."""
    np.random.seed(42)
    dim = 10
    kappa = 100

    # Create ill-conditioned quadratic: f(x) = 0.5 x^T A x
    eigenvalues = np.linspace(1, kappa, dim)
    Q, _ = np.linalg.qr(np.random.randn(dim, dim))
    A = Q @ np.diag(eigenvalues) @ Q.T

    def f(x):
        return 0.5 * x @ A @ x

    def grad_f(x):
        return A @ x

    x0 = np.random.randn(dim)
    lr = 2.0 / (1 + kappa)
    beta = 0.9
    n_steps = 500

    # Standard momentum
    def momentum_gd(x0, lr, beta, n_steps):
        x = x0.copy()
        v = np.zeros_like(x)
        losses = [f(x)]
        for _ in range(n_steps):
            g = grad_f(x)
            v = beta * v - lr * g
            x = x + v
            losses.append(f(x))
        return losses

    # Nesterov accelerated gradient
    def nesterov_gd(x0, lr, beta, n_steps):
        x = x0.copy()
        v = np.zeros_like(x)
        losses = [f(x)]
        for _ in range(n_steps):
            # Look ahead
            x_lookahead = x + beta * v
            g = grad_f(x_lookahead)
            v = beta * v - lr * g
            x = x + v
            losses.append(f(x))
        return losses

    losses_mom = momentum_gd(x0, lr, beta, n_steps)
    losses_nes = nesterov_gd(x0, lr, beta, n_steps)

    print(f"Quadratic function with condition number {kappa}")
    print(f"Learning rate: {lr:.6f}, Momentum: {beta}")
    print()

    print("Step | Momentum Loss | Nesterov Loss | Nesterov Better?")
    print("-" * 60)
    for step in [0, 10, 50, 100, 200, 500]:
        mom_l = losses_mom[step]
        nes_l = losses_nes[step]
        better = "Yes" if nes_l < mom_l else "No"
        print(f"{step:4d} | {mom_l:13.6f} | {nes_l:13.6f} | {better}")

    # Steps to reach certain accuracy
    threshold = 1e-6
    mom_steps = next((i for i, l in enumerate(losses_mom) if l < threshold), n_steps)
    nes_steps = next((i for i, l in enumerate(losses_nes) if l < threshold), n_steps)
    print(f"\nSteps to reach loss < {threshold}:")
    print(f"  Momentum: {mom_steps}")
    print(f"  Nesterov: {nes_steps}")


# === Exercise 5: Learning Rate Schedules ===
# Problem: Compare fixed LR, step decay, cosine annealing on a test problem.

def exercise_5():
    """Compare learning rate schedules on quadratic optimization."""
    np.random.seed(42)

    # Use a simple quadratic for clean comparison
    dim = 20
    A = np.diag(np.linspace(1, 50, dim))

    def f(x):
        return 0.5 * x @ A @ x

    def grad_f(x):
        return A @ x

    x0 = np.random.randn(dim) * 3
    n_steps = 1000
    base_lr = 0.02

    def train_with_schedule(schedule_fn):
        x = x0.copy()
        losses = [f(x)]
        for t in range(n_steps):
            lr = schedule_fn(t)
            g = grad_f(x)
            x = x - lr * g
            losses.append(f(x))
        return losses

    # 1. Fixed LR
    losses_fixed = train_with_schedule(lambda t: base_lr)

    # 2. Step decay: halve every 200 steps
    def step_decay(t):
        return base_lr * (0.5 ** (t // 200))
    losses_step = train_with_schedule(step_decay)

    # 3. Cosine annealing
    def cosine_anneal(t):
        return base_lr * 0.5 * (1 + np.cos(np.pi * t / n_steps))
    losses_cosine = train_with_schedule(cosine_anneal)

    # 4. Linear warmup + cosine decay
    warmup_steps = 100
    def warmup_cosine(t):
        if t < warmup_steps:
            return base_lr * t / warmup_steps
        else:
            progress = (t - warmup_steps) / (n_steps - warmup_steps)
            return base_lr * 0.5 * (1 + np.cos(np.pi * progress))
    losses_warmup = train_with_schedule(warmup_cosine)

    print("Learning rate schedule comparison on quadratic function")
    print(f"Dimension: {dim}, Condition number: {50/1:.0f}")
    print()

    schedules = {
        'Fixed': losses_fixed,
        'Step Decay': losses_step,
        'Cosine Annealing': losses_cosine,
        'Warmup + Cosine': losses_warmup,
    }

    print(f"{'Schedule':<20} {'Loss@100':>12} {'Loss@500':>12} {'Loss@1000':>12}")
    print("-" * 58)
    for name, losses in schedules.items():
        print(f"{name:<20} {losses[100]:>12.6f} {losses[500]:>12.6f} {losses[-1]:>12.6f}")


if __name__ == "__main__":
    print("=== Exercise 1: Convergence Rate Analysis ===")
    exercise_1()
    print("\n=== Exercise 2: SGD vs Batch GD ===")
    exercise_2()
    print("\n=== Exercise 3: Adam Optimizer ===")
    exercise_3()
    print("\n=== Exercise 4: Momentum vs Nesterov ===")
    exercise_4()
    print("\n=== Exercise 5: Learning Rate Schedules ===")
    exercise_5()
    print("\nAll exercises completed!")
