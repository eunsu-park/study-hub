"""
Exercises for Lesson 05: Multivariate Calculus
Topic: Math_for_AI

Solutions to practice problems from the lesson.
"""

import numpy as np


# === Exercise 1: Critical Point Classification ===
# Problem: For f(x,y) = x^3 - 3xy^2:
# 1. Compute gradient and find critical points
# 2. Compute Hessian and classify each critical point
# 3. (Visualization skipped for runnable-only solution)

def exercise_1():
    """Find and classify critical points of f(x,y) = x^3 - 3xy^2."""
    # f(x,y) = x^3 - 3xy^2
    # grad f = [3x^2 - 3y^2, -6xy]
    # Setting grad = 0:
    #   3x^2 - 3y^2 = 0  =>  x^2 = y^2  =>  y = +/- x
    #   -6xy = 0           =>  x=0 or y=0

    # Case 1: x = 0 => y^2 = 0 => y = 0 => critical point (0,0)
    # Case 2: y = 0 => x^2 = 0 => x = 0 => same point
    # So the only critical point is (0, 0)

    print("f(x,y) = x^3 - 3xy^2")
    print()
    print("Gradient: nabla f = [3x^2 - 3y^2, -6xy]")
    print("Setting nabla f = 0:")
    print("  3x^2 - 3y^2 = 0  =>  x^2 = y^2")
    print("  -6xy = 0  =>  x = 0 or y = 0")
    print("  Combined: x = 0, y = 0")
    print("  Critical point: (0, 0)")

    # Hessian
    # H = [[6x, -6y], [-6y, -6x]]
    # At (0,0): H = [[0, 0], [0, 0]]
    print()
    print("Hessian:")
    print("  H = [[6x, -6y], [-6y, -6x]]")
    print("  H(0,0) = [[0, 0], [0, 0]]")

    H_origin = np.array([[0, 0], [0, 0]])
    eigenvalues = np.linalg.eigvalsh(H_origin)
    print(f"  Eigenvalues at (0,0): {eigenvalues}")
    print("  Hessian is zero => second derivative test is inconclusive")

    # Further analysis: check function values near (0,0)
    print()
    print("Further analysis by checking nearby values:")
    test_points = [(0.1, 0), (-0.1, 0), (0, 0.1), (0.1, 0.1)]
    for x, y in test_points:
        val = x**3 - 3*x*y**2
        print(f"  f({x}, {y}) = {val:.6f}")

    print()
    print("Along x-axis: f(x,0) = x^3 (changes sign => not extremum)")
    print("Along y=x line: f(x,x) = x^3 - 3x^3 = -2x^3 (changes sign)")
    print("Conclusion: (0,0) is a degenerate critical point (monkey saddle)")


# === Exercise 2: Constrained Optimization ===
# Problem: min f(x,y) = x^2 + 2y^2  s.t.  x^2 + y^2 = 1
# Use Lagrange multipliers.

def exercise_2():
    """Constrained optimization with Lagrange multipliers."""
    print("Problem: min x^2 + 2y^2  s.t.  x^2 + y^2 = 1")
    print()
    print("Lagrangian: L = x^2 + 2y^2 - lambda(x^2 + y^2 - 1)")
    print()
    print("KKT conditions:")
    print("  dL/dx = 2x - 2*lambda*x = 0  =>  x(1 - lambda) = 0")
    print("  dL/dy = 4y - 2*lambda*y = 0  =>  y(2 - lambda) = 0")
    print("  x^2 + y^2 = 1")
    print()

    # Case 1: x = 0 => y^2 = 1 => y = +/- 1, lambda = 2
    # f(0, +/-1) = 2
    print("Case 1: x = 0")
    print("  y^2 = 1, lambda = 2")
    print("  Points: (0, 1) and (0, -1)")
    print("  f = 0 + 2(1) = 2")

    # Case 2: y = 0 => x^2 = 1 => x = +/- 1, lambda = 1
    # f(+/-1, 0) = 1
    print()
    print("Case 2: y = 0")
    print("  x^2 = 1, lambda = 1")
    print("  Points: (1, 0) and (-1, 0)")
    print("  f = 1 + 0 = 1")

    print()
    print("Minimum: f = 1 at (+/-1, 0)")
    print("Maximum: f = 2 at (0, +/-1)")

    # Numerical verification using scipy
    from scipy.optimize import minimize

    def objective(xy):
        return xy[0]**2 + 2*xy[1]**2

    def constraint(xy):
        return xy[0]**2 + xy[1]**2 - 1

    result = minimize(objective, x0=[0.5, 0.5],
                      constraints={'type': 'eq', 'fun': constraint})
    print()
    print(f"Numerical solution: x={result.x[0]:.6f}, y={result.x[1]:.6f}")
    print(f"Minimum value: {result.fun:.6f}")
    print(f"Constraint satisfied: x^2+y^2 = {result.x[0]**2 + result.x[1]**2:.6f}")


# === Exercise 3: Momentum Gradient Descent ===
# Problem: Implement GD with momentum and compare with vanilla GD
# on ill-conditioned function.

def exercise_3():
    """Compare vanilla GD and momentum GD on ill-conditioned function."""
    # f(x,y) = x^2 + 100*y^2  (condition number = 100)
    def f(x, y):
        return x**2 + 100*y**2

    def grad_f(x, y):
        return np.array([2*x, 200*y])

    # Vanilla gradient descent
    def gd_vanilla(x0, lr, n_steps):
        x = x0.copy()
        path = [x.copy()]
        for _ in range(n_steps):
            g = grad_f(x[0], x[1])
            x = x - lr * g
            path.append(x.copy())
        return np.array(path)

    # Momentum gradient descent
    def gd_momentum(x0, lr, beta, n_steps):
        x = x0.copy()
        v = np.zeros_like(x)
        path = [x.copy()]
        for _ in range(n_steps):
            g = grad_f(x[0], x[1])
            v = beta * v - lr * g
            x = x + v
            path.append(x.copy())
        return np.array(path)

    x0 = np.array([3.0, 0.3])
    n_steps = 200

    # Vanilla GD (small lr needed due to condition number)
    lr_vanilla = 0.005
    path_vanilla = gd_vanilla(x0, lr_vanilla, n_steps)

    # Momentum GD
    lr_momentum = 0.005
    beta = 0.9
    path_momentum = gd_momentum(x0, lr_momentum, beta, n_steps)

    # Compare convergence
    losses_vanilla = [f(p[0], p[1]) for p in path_vanilla]
    losses_momentum = [f(p[0], p[1]) for p in path_momentum]

    print(f"Function: f(x,y) = x^2 + 100y^2 (condition number = 100)")
    print(f"Start: {x0}, Optimum: [0, 0]")
    print(f"Learning rate: {lr_vanilla}, Momentum beta: {beta}")
    print()

    for step in [0, 10, 50, 100, 200]:
        print(f"Step {step:3d}: Vanilla loss = {losses_vanilla[step]:.6f}, "
              f"Momentum loss = {losses_momentum[step]:.6f}")

    print()
    print(f"Final position (Vanilla):  {path_vanilla[-1]}")
    print(f"Final position (Momentum): {path_momentum[-1]}")
    print(f"Final loss (Vanilla):  {losses_vanilla[-1]:.8f}")
    print(f"Final loss (Momentum): {losses_momentum[-1]:.8f}")

    speedup = losses_vanilla[-1] / max(losses_momentum[-1], 1e-20)
    print(f"\nMomentum advantage: vanilla loss / momentum loss = {speedup:.1f}x")


# === Exercise 4: Normal Equation Derivation ===
# Problem: Derive w* = (X^T X)^{-1} X^T y by setting gradient of MSE to zero.

def exercise_4():
    """Derive and verify normal equation for linear regression."""
    print("Derivation of the normal equation:")
    print()
    print("Loss: L(w) = (1/2n) ||y - Xw||^2 = (1/2n)(y - Xw)^T(y - Xw)")
    print()
    print("Expanding:")
    print("  L = (1/2n)(y^T y - 2w^T X^T y + w^T X^T X w)")
    print()
    print("Gradient:")
    print("  nabla_w L = (1/n)(-X^T y + X^T X w)")
    print()
    print("Setting nabla_w L = 0:")
    print("  X^T X w = X^T y")
    print("  w* = (X^T X)^{-1} X^T y  (if X^T X is invertible)")
    print()
    print("Hessian:")
    print("  H = (1/n) X^T X")
    print("  X^T X is positive semidefinite (x^T X^T X x = ||Xx||^2 >= 0)")
    print("  If X has full column rank, X^T X is positive definite => unique minimum")

    # Numerical verification
    np.random.seed(42)
    n, d = 50, 3
    X = np.random.randn(n, d)
    true_w = np.array([2, -1, 0.5])
    y = X @ true_w + np.random.randn(n) * 0.1

    # Normal equation
    w_normal = np.linalg.solve(X.T @ X, X.T @ y)

    # Check Hessian positive definiteness
    H = X.T @ X / n
    eigenvalues = np.linalg.eigvalsh(H)

    print()
    print("=== Numerical Verification ===")
    print(f"True weights: {true_w}")
    print(f"Normal equation solution: {np.round(w_normal, 4)}")
    print(f"Hessian eigenvalues: {np.round(eigenvalues, 4)}")
    print(f"All positive (PD): {np.all(eigenvalues > 0)}")

    # Verify gradient is zero at optimum
    grad_at_opt = (-X.T @ y + X.T @ X @ w_normal) / n
    print(f"Gradient at optimum: {np.round(grad_at_opt, 10)}")


# === Exercise 5: Adam Optimizer Implementation ===
# Problem: Implement Adam and compare with GD and Momentum on anisotropic loss.

def exercise_5():
    """Adam optimizer implementation and comparison."""
    def f(x, y):
        return x**2 + 50*y**2

    def grad_f(x, y):
        return np.array([2*x, 100*y])

    # Vanilla GD
    def gd(x0, lr, n_steps):
        x = x0.copy()
        losses = [f(x[0], x[1])]
        for _ in range(n_steps):
            g = grad_f(x[0], x[1])
            x -= lr * g
            losses.append(f(x[0], x[1]))
        return x, losses

    # Momentum
    def momentum(x0, lr, beta, n_steps):
        x = x0.copy()
        v = np.zeros(2)
        losses = [f(x[0], x[1])]
        for _ in range(n_steps):
            g = grad_f(x[0], x[1])
            v = beta * v - lr * g
            x += v
            losses.append(f(x[0], x[1]))
        return x, losses

    # Adam optimizer
    def adam(x0, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, n_steps=200):
        x = x0.copy()
        m = np.zeros(2)  # first moment
        v = np.zeros(2)  # second moment
        losses = [f(x[0], x[1])]

        for t in range(1, n_steps + 1):
            g = grad_f(x[0], x[1])

            # Update biased first moment estimate
            m = beta1 * m + (1 - beta1) * g
            # Update biased second moment estimate
            v = beta2 * v + (1 - beta2) * g**2

            # Bias correction
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)

            # Update parameters
            x -= lr * m_hat / (np.sqrt(v_hat) + eps)
            losses.append(f(x[0], x[1]))

        return x, losses

    x0 = np.array([3.0, 0.3])
    n_steps = 200

    x_gd, losses_gd = gd(x0, lr=0.005, n_steps=n_steps)
    x_mom, losses_mom = momentum(x0, lr=0.005, beta=0.9, n_steps=n_steps)
    x_adam, losses_adam = adam(x0, lr=0.1, n_steps=n_steps)

    print(f"Function: f(x,y) = x^2 + 50y^2")
    print(f"Start: {x0}, Optimum: [0, 0]")
    print()

    print("Convergence comparison:")
    for step in [0, 10, 50, 100, 200]:
        print(f"  Step {step:3d}: "
              f"GD={losses_gd[step]:.6f}, "
              f"Momentum={losses_mom[step]:.6f}, "
              f"Adam={losses_adam[step]:.6f}")

    print()
    print(f"Final position (GD):       {np.round(x_gd, 6)}")
    print(f"Final position (Momentum): {np.round(x_mom, 6)}")
    print(f"Final position (Adam):     {np.round(x_adam, 6)}")
    print(f"Final loss (GD):       {losses_gd[-1]:.10f}")
    print(f"Final loss (Momentum): {losses_mom[-1]:.10f}")
    print(f"Final loss (Adam):     {losses_adam[-1]:.10f}")


if __name__ == "__main__":
    print("=== Exercise 1: Critical Point Classification ===")
    exercise_1()
    print("\n=== Exercise 2: Constrained Optimization ===")
    exercise_2()
    print("\n=== Exercise 3: Momentum Gradient Descent ===")
    exercise_3()
    print("\n=== Exercise 4: Normal Equation Derivation ===")
    exercise_4()
    print("\n=== Exercise 5: Adam Optimizer ===")
    exercise_5()
    print("\nAll exercises completed!")
