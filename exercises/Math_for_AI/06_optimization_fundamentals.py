"""
Exercises for Lesson 06: Optimization Fundamentals
Topic: Math_for_AI

Solutions to practice problems from the lesson.
"""

import numpy as np
from scipy.optimize import minimize


# === Exercise 1: Convexity Determination ===
# Problem: Determine convexity of:
# (a) f(x,y) = e^x + e^y
# (b) f(x,y) = x^2 - xy + y^2
# (c) f(x,y) = log(e^x + e^y)

def exercise_1():
    """Verify convexity using Hessian positive semidefiniteness."""
    print("A function is convex if its Hessian is PSD everywhere.\n")

    # (a) f(x,y) = e^x + e^y
    print("(a) f(x,y) = e^x + e^y")
    print("  H = [[e^x, 0], [0, e^y]]")
    print("  Eigenvalues: e^x > 0 and e^y > 0 for all x,y")
    print("  => H is positive definite everywhere => f is strictly convex")

    # Numerical verification
    for x, y in [(0, 0), (1, -1), (-2, 3)]:
        H = np.array([[np.exp(x), 0], [0, np.exp(y)]])
        eigs = np.linalg.eigvalsh(H)
        print(f"  At ({x},{y}): eigenvalues = {np.round(eigs, 4)}, PD = {np.all(eigs > 0)}")

    # (b) f(x,y) = x^2 - xy + y^2
    print("\n(b) f(x,y) = x^2 - xy + y^2")
    print("  H = [[2, -1], [-1, 2]]  (constant Hessian)")
    H_b = np.array([[2, -1], [-1, 2]])
    eigs_b = np.linalg.eigvalsh(H_b)
    print(f"  Eigenvalues: {eigs_b}")
    print(f"  All positive: {np.all(eigs_b > 0)} => strictly convex")

    # (c) f(x,y) = log(e^x + e^y)  (log-sum-exp)
    print("\n(c) f(x,y) = log(e^x + e^y)  (log-sum-exp)")
    print("  This is the well-known LogSumExp function, known to be convex.")

    # Hessian: H_ij = p_i * (delta_ij - p_j), where p_i = e^{x_i} / sum(e^{x_j})
    for x, y in [(0, 0), (1, -1), (-2, 3)]:
        s = np.exp(x) + np.exp(y)
        px = np.exp(x) / s
        py = np.exp(y) / s
        H = np.array([
            [px * (1 - px), -px * py],
            [-px * py, py * (1 - py)]
        ])
        eigs = np.linalg.eigvalsh(H)
        print(f"  At ({x},{y}): eigenvalues = {np.round(eigs, 6)}, PSD = {np.all(eigs >= -1e-10)}")

    print("  => H is PSD everywhere => f is convex")


# === Exercise 2: Lagrange Multipliers ===
# Problem: min x^2 + 2y^2  s.t. x + y = 1

def exercise_2():
    """Solve constrained optimization with Lagrange multipliers."""
    print("Problem: min x^2 + 2y^2  s.t. x + y = 1")
    print()
    print("Lagrangian: L = x^2 + 2y^2 - lambda(x + y - 1)")
    print()
    print("KKT conditions:")
    print("  dL/dx = 2x - lambda = 0  =>  x = lambda/2")
    print("  dL/dy = 4y - lambda = 0  =>  y = lambda/4")
    print("  x + y = 1")
    print()
    print("Substituting:")
    print("  lambda/2 + lambda/4 = 1")
    print("  3*lambda/4 = 1")
    print("  lambda = 4/3")
    print()

    lam = 4.0 / 3.0
    x_opt = lam / 2
    y_opt = lam / 4
    f_opt = x_opt**2 + 2 * y_opt**2

    print(f"Analytical solution:")
    print(f"  lambda = {lam:.6f}")
    print(f"  x* = {x_opt:.6f}")
    print(f"  y* = {y_opt:.6f}")
    print(f"  f(x*, y*) = {f_opt:.6f}")
    print(f"  Constraint: x + y = {x_opt + y_opt:.6f}")

    # Numerical verification
    result = minimize(
        lambda xy: xy[0]**2 + 2*xy[1]**2,
        x0=[0.5, 0.5],
        constraints={'type': 'eq', 'fun': lambda xy: xy[0] + xy[1] - 1}
    )
    print(f"\nNumerical solution:")
    print(f"  x* = {result.x[0]:.6f}")
    print(f"  y* = {result.x[1]:.6f}")
    print(f"  f* = {result.fun:.6f}")


# === Exercise 3: KKT Conditions Application ===
# Problem: min (x-3)^2 + (y-2)^2  s.t. x+2y<=4, x>=0, y>=0

def exercise_3():
    """Apply KKT conditions to inequality-constrained problem."""
    print("Problem: min (x-3)^2 + (y-2)^2  s.t. x+2y<=4, x>=0, y>=0")
    print()

    # Unconstrained optimum: (3, 2)
    # Check if feasible: 3 + 2*2 = 7 > 4 => NOT feasible
    print("Unconstrained optimum (3,2): 3+4=7 > 4, not feasible")
    print()

    print("KKT conditions:")
    print("  Stationarity:")
    print("    2(x-3) + mu1 - mu2 = 0")
    print("    2(y-2) + 2*mu1 - mu3 = 0")
    print("  Complementary slackness:")
    print("    mu1*(x+2y-4) = 0")
    print("    mu2*(-x) = 0")
    print("    mu3*(-y) = 0")
    print("  mu1, mu2, mu3 >= 0")
    print()

    # Try: constraint x+2y=4 active, x>0, y>0 (mu2=mu3=0)
    # 2(x-3) + mu1 = 0 => mu1 = 6 - 2x
    # 2(y-2) + 2*mu1 = 0 => y - 2 + mu1 = 0 => mu1 = 2 - y
    # So: 6 - 2x = 2 - y => y = 2x - 4
    # Constraint: x + 2(2x-4) = 4 => 5x = 12 => x = 12/5

    x_kkt = 12.0 / 5.0
    y_kkt = 2 * x_kkt - 4
    mu1 = 6 - 2 * x_kkt
    f_kkt = (x_kkt - 3)**2 + (y_kkt - 2)**2

    print("Trying x+2y=4 active, x>0, y>0:")
    print(f"  x* = {x_kkt:.4f}")
    print(f"  y* = {y_kkt:.4f}")
    print(f"  mu1 = {mu1:.4f}")
    print(f"  x+2y = {x_kkt + 2*y_kkt:.4f}")
    print(f"  f* = {f_kkt:.4f}")

    # Check KKT validity
    print(f"\n  KKT validity: mu1={mu1:.4f} >= 0: {mu1 >= -1e-10}")
    print(f"  x={x_kkt:.4f} >= 0: {x_kkt >= 0}")
    print(f"  y={y_kkt:.4f} >= 0: {y_kkt >= 0}")

    # Numerical verification
    from scipy.optimize import minimize
    result = minimize(
        lambda xy: (xy[0]-3)**2 + (xy[1]-2)**2,
        x0=[1, 1],
        constraints=[
            {'type': 'ineq', 'fun': lambda xy: 4 - xy[0] - 2*xy[1]},
        ],
        bounds=[(0, None), (0, None)]
    )
    print(f"\nNumerical solution:")
    print(f"  x* = {result.x[0]:.4f}, y* = {result.x[1]:.4f}")
    print(f"  f* = {result.fun:.4f}")
    print(f"  Active constraint (x+2y=4): x+2y = {result.x[0] + 2*result.x[1]:.4f}")


# === Exercise 4: SVM Dual Problem ===
# Problem: Generate linearly separable 2D dataset and solve SVM dual.

def exercise_4():
    """Simple SVM dual problem on linearly separable 2D data."""
    np.random.seed(42)

    # Generate linearly separable data
    n = 20
    X_pos = np.random.randn(n, 2) + np.array([2, 2])
    X_neg = np.random.randn(n, 2) + np.array([-2, -2])
    X = np.vstack([X_pos, X_neg])
    y = np.array([1]*n + [-1]*n, dtype=float)

    # SVM dual: max sum(alpha) - 0.5 * sum_ij(alpha_i * alpha_j * y_i * y_j * x_i.x_j)
    # s.t. alpha_i >= 0, sum(alpha_i * y_i) = 0

    # Using scipy minimize (negate for maximization)
    N = len(y)
    K = X @ X.T  # kernel matrix (linear)

    def dual_objective(alpha):
        return -np.sum(alpha) + 0.5 * np.sum(alpha[:, None] * alpha[None, :] * y[:, None] * y[None, :] * K)

    def dual_gradient(alpha):
        return -np.ones(N) + (alpha * y) @ (y[:, None] * K)

    from scipy.optimize import minimize
    constraints = {'type': 'eq', 'fun': lambda a: np.dot(a, y)}
    bounds = [(0, 10)] * N  # soft margin with C=10

    result = minimize(dual_objective, x0=np.zeros(N), jac=dual_gradient,
                      bounds=bounds, constraints=constraints, method='SLSQP')
    alpha = result.x

    # Identify support vectors
    sv_threshold = 1e-4
    support_mask = alpha > sv_threshold
    n_sv = np.sum(support_mask)

    print(f"SVM Dual Problem Solution:")
    print(f"  Number of samples: {N}")
    print(f"  Number of support vectors: {n_sv}")

    # Compute w and b
    w = np.sum(alpha[:, None] * y[:, None] * X, axis=0)

    # Compute b from support vectors
    sv_indices = np.where(support_mask)[0]
    b_values = y[sv_indices] - X[sv_indices] @ w
    b = np.mean(b_values)

    print(f"  w = {np.round(w, 4)}")
    print(f"  b = {b:.4f}")

    # Verify classification
    predictions = np.sign(X @ w + b)
    accuracy = np.mean(predictions == y)
    print(f"  Training accuracy: {accuracy:.2%}")

    # Show support vector details
    print(f"\n  Support vectors (alpha > {sv_threshold}):")
    for idx in sv_indices:
        print(f"    Point {idx}: alpha={alpha[idx]:.4f}, "
              f"y={y[idx]:+.0f}, x={np.round(X[idx], 2)}")


# === Exercise 5: Non-convex Optimization (Rosenbrock) ===
# Problem: Run GD from multiple starts on Rosenbrock function.

def exercise_5():
    """Gradient descent on Rosenbrock function from multiple starts."""
    # Rosenbrock: f(x,y) = (1-x)^2 + 100(y-x^2)^2
    def rosenbrock(xy):
        x, y = xy
        return (1 - x)**2 + 100 * (y - x**2)**2

    def grad_rosenbrock(xy):
        x, y = xy
        dx = -2*(1-x) + 200*(y-x**2)*(-2*x)
        dy = 200*(y-x**2)
        return np.array([dx, dy])

    # GD with momentum
    def gd_momentum(x0, lr, beta, n_steps):
        x = np.array(x0, dtype=float)
        v = np.zeros(2)
        path = [x.copy()]
        for _ in range(n_steps):
            g = grad_rosenbrock(x)
            v = beta * v - lr * g
            x = x + v
            path.append(x.copy())
        return np.array(path)

    # Vanilla GD
    def gd_vanilla(x0, lr, n_steps):
        x = np.array(x0, dtype=float)
        path = [x.copy()]
        for _ in range(n_steps):
            g = grad_rosenbrock(x)
            x = x - lr * g
            path.append(x.copy())
        return np.array(path)

    print("Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2")
    print("Global minimum: (1, 1) with f = 0")
    print()

    starts = [
        np.array([-1.0, 1.0]),
        np.array([0.0, 0.0]),
        np.array([2.0, 2.0]),
        np.array([-2.0, -1.0]),
    ]
    n_steps = 5000
    lr = 0.001

    print("=== Vanilla GD ===")
    for x0 in starts:
        path = gd_vanilla(x0, lr=lr, n_steps=n_steps)
        final = path[-1]
        final_loss = rosenbrock(final)
        print(f"  Start {x0} -> End {np.round(final, 4)}, "
              f"loss = {final_loss:.6f}")

    print()
    print("=== Momentum GD (beta=0.9) ===")
    for x0 in starts:
        path = gd_momentum(x0, lr=lr, beta=0.9, n_steps=n_steps)
        final = path[-1]
        final_loss = rosenbrock(final)
        print(f"  Start {x0} -> End {np.round(final, 4)}, "
              f"loss = {final_loss:.6f}")

    # Scipy for comparison
    print()
    print("=== Scipy L-BFGS-B (for reference) ===")
    for x0 in starts:
        result = minimize(rosenbrock, x0, jac=grad_rosenbrock, method='L-BFGS-B')
        print(f"  Start {x0} -> End {np.round(result.x, 4)}, "
              f"loss = {result.fun:.6f}, iters = {result.nit}")


if __name__ == "__main__":
    print("=== Exercise 1: Convexity Determination ===")
    exercise_1()
    print("\n=== Exercise 2: Lagrange Multipliers ===")
    exercise_2()
    print("\n=== Exercise 3: KKT Conditions ===")
    exercise_3()
    print("\n=== Exercise 4: SVM Dual Problem ===")
    exercise_4()
    print("\n=== Exercise 5: Non-convex Optimization ===")
    exercise_5()
    print("\nAll exercises completed!")
