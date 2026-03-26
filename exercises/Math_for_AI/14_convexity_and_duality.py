"""
Exercises for Lesson 14: Convexity and Duality
Topic: Math_for_AI

Solutions to practice problems from the lesson.
"""

import numpy as np
from scipy.optimize import minimize


# === Exercise 1: Proving Convexity ===
# Problem: Verify convexity of (a) e^x, (b) -log(x), (c) x^2 + xy + y^2

def exercise_1():
    """Verify convexity using second-order conditions (Hessian)."""
    print("A function is convex if its Hessian is positive semidefinite.\n")

    # (a) f(x) = e^x
    print("(a) f(x) = e^x")
    print("    f''(x) = e^x > 0 for all x")
    print("    => f is strictly convex")

    # Numerical: verify Jensen's inequality f(E[X]) <= E[f(X)]
    np.random.seed(42)
    x_samples = np.random.randn(1000)
    f_mean = np.exp(np.mean(x_samples))
    mean_f = np.mean(np.exp(x_samples))
    print(f"    Jensen's inequality: f(E[X])={f_mean:.4f} <= E[f(X)]={mean_f:.4f}: {f_mean <= mean_f}")

    # (b) f(x) = -log(x), x > 0
    print("\n(b) f(x) = -log(x), x > 0")
    print("    f'(x) = -1/x")
    print("    f''(x) = 1/x^2 > 0 for all x > 0")
    print("    => f is strictly convex on (0, inf)")

    x_pos = np.abs(np.random.randn(1000)) + 0.1
    f_mean = -np.log(np.mean(x_pos))
    mean_f = np.mean(-np.log(x_pos))
    print(f"    Jensen's inequality: f(E[X])={f_mean:.4f} <= E[f(X)]={mean_f:.4f}: {f_mean <= mean_f}")

    # (c) f(x, y) = x^2 + xy + y^2
    print("\n(c) f(x, y) = x^2 + xy + y^2")
    H = np.array([[2, 1], [1, 2]])
    eigenvalues = np.linalg.eigvalsh(H)
    print(f"    Hessian: [[2, 1], [1, 2]]")
    print(f"    Eigenvalues: {eigenvalues}")
    print(f"    All positive: {np.all(eigenvalues > 0)}")
    print(f"    => f is strictly convex")

    # Jensen's for 2D
    xy = np.random.randn(1000, 2)
    f_vals = xy[:, 0]**2 + xy[:, 0]*xy[:, 1] + xy[:, 1]**2
    mean_xy = np.mean(xy, axis=0)
    f_at_mean = mean_xy[0]**2 + mean_xy[0]*mean_xy[1] + mean_xy[1]**2
    print(f"    Jensen's: f(E[X])={f_at_mean:.4f} <= E[f(X)]={np.mean(f_vals):.4f}: {f_at_mean <= np.mean(f_vals)}")


# === Exercise 2: Lagrangian Duality ===
# Problem: min x^2 + y^2 s.t. x+y >= 1, x,y >= 0

def exercise_2():
    """Lagrangian duality with KKT conditions."""
    print("Problem: min x^2 + y^2  s.t. x+y >= 1, x >= 0, y >= 0")
    print()
    print("Rewrite constraints: g1(x,y) = 1 - x - y <= 0, -x <= 0, -y <= 0")
    print()
    print("Lagrangian: L = x^2 + y^2 + lambda1*(1-x-y) + lambda2*(-x) + lambda3*(-y)")
    print()
    print("KKT conditions:")
    print("  Stationarity: 2x - lambda1 - lambda2 = 0")
    print("                2y - lambda1 - lambda3 = 0")
    print("  Complementary slackness:")
    print("    lambda1*(1-x-y) = 0")
    print("    lambda2*(-x) = 0")
    print("    lambda3*(-y) = 0")
    print("  lambda1, lambda2, lambda3 >= 0")
    print()

    # By symmetry, try x = y, with constraint x+y = 1 active
    # lambda2 = lambda3 = 0 (x, y > 0)
    # 2x - lambda1 = 0 => lambda1 = 2x
    # x + y = 1, x = y => x = y = 1/2
    # lambda1 = 1

    x_opt = 0.5
    y_opt = 0.5
    lambda1 = 1.0
    f_opt = x_opt**2 + y_opt**2

    print(f"Analytical solution: x* = {x_opt}, y* = {y_opt}")
    print(f"Multiplier: lambda1 = {lambda1}")
    print(f"Optimal value: f* = {f_opt}")
    print()

    # Dual function
    print("Dual function: g(lambda1) = min_{x,y>=0} [x^2 + y^2 + lambda1*(1-x-y)]")
    print("  Taking derivatives: x* = lambda1/2, y* = lambda1/2 (for lambda1 >= 0)")
    print("  g(lambda1) = lambda1^2/4 + lambda1*(1 - lambda1) = lambda1 - 3*lambda1^2/4")
    print()

    # Dual problem: max g(lambda1) s.t. lambda1 >= 0
    # g'(lambda1) = 1 - 3*lambda1/2 = 0 => lambda1 = 2/3... wait, let me recompute
    # Actually: g(l) = (l/2)^2 + (l/2)^2 + l*(1 - l/2 - l/2) = l^2/4 + l^2/4 + l - l^2 = l^2/2 + l - l^2 = l - l^2/2
    # g'(l) = 1 - l = 0 => l = 1
    # g(1) = 1 - 1/2 = 1/2

    print("Corrected dual: g(lambda) = lambda - lambda^2/2")
    print("g'(lambda) = 1 - lambda = 0 => lambda* = 1")
    print(f"Dual optimal: g(1) = {1 - 0.5:.4f}")
    print(f"Primal optimal: f* = {f_opt:.4f}")
    print(f"Strong duality holds: {np.isclose(f_opt, 1 - 0.5)}")

    # Numerical verification
    result = minimize(lambda xy: xy[0]**2 + xy[1]**2,
                      x0=[1, 1],
                      constraints={'type': 'ineq', 'fun': lambda xy: xy[0] + xy[1] - 1},
                      bounds=[(0, None), (0, None)])
    print(f"\nNumerical: x*={result.x[0]:.4f}, y*={result.x[1]:.4f}, f*={result.fun:.4f}")


# === Exercise 3: SVM Implementation ===
# Problem: Implement linear SVM solving both primal and dual.

def exercise_3():
    """Linear SVM with primal and dual problem solution."""
    np.random.seed(42)

    # Generate linearly separable data
    n = 30
    X_pos = np.random.randn(n, 2) + np.array([2, 2])
    X_neg = np.random.randn(n, 2) + np.array([-2, -2])
    X = np.vstack([X_pos, X_neg])
    y = np.array([1]*n + [-1]*n, dtype=float)
    N = len(y)

    # Primal problem: min 0.5||w||^2 s.t. y_i(w.x_i + b) >= 1
    # Using scipy with slack variables for numerical stability
    def svm_primal(X, y, C=100):
        """Solve SVM primal with soft margin."""
        d = X.shape[1]
        N_pts = len(y)

        # Variables: [w1, w2, b, xi_1, ..., xi_N]
        def objective(params):
            w = params[:d]
            xi = params[d+1:]
            return 0.5 * np.dot(w, w) + C * np.sum(xi)

        constraints = []
        for i in range(N_pts):
            def margin_constraint(params, i=i):
                w = params[:d]
                b_val = params[d]
                xi = params[d+1:]
                return y[i] * (np.dot(w, X[i]) + b_val) - 1 + xi[i]
            constraints.append({'type': 'ineq', 'fun': margin_constraint})

        bounds = [(None, None)] * d + [(None, None)] + [(0, None)] * N_pts
        x0 = np.zeros(d + 1 + N_pts)
        result = minimize(objective, x0, bounds=bounds, constraints=constraints,
                          method='SLSQP')
        w = result.x[:d]
        b_val = result.x[d]
        return w, b_val, result.fun

    w_primal, b_primal, obj_primal = svm_primal(X, y)

    # Dual problem
    K = X @ X.T
    def dual_objective(alpha):
        return -(np.sum(alpha) - 0.5 * np.sum(alpha[:, None] * alpha[None, :] * y[:, None] * y[None, :] * K))

    bounds_dual = [(0, 100)] * N
    constraints_dual = {'type': 'eq', 'fun': lambda a: np.dot(a, y)}
    result_dual = minimize(dual_objective, x0=np.zeros(N), bounds=bounds_dual,
                           constraints=constraints_dual, method='SLSQP')
    alpha = result_dual.x

    w_dual = np.sum(alpha[:, None] * y[:, None] * X, axis=0)
    sv_mask = alpha > 1e-4
    sv_indices = np.where(sv_mask)[0]
    b_dual = np.mean(y[sv_mask] - X[sv_mask] @ w_dual)

    print("(a) Primal and Dual solutions:")
    print(f"    Primal: w={np.round(w_primal, 4)}, b={b_primal:.4f}")
    print(f"    Dual:   w={np.round(w_dual, 4)}, b={b_dual:.4f}")

    # (b) Strong duality check
    primal_obj = 0.5 * np.dot(w_primal, w_primal)
    dual_obj = -result_dual.fun
    print(f"\n(b) Strong duality:")
    print(f"    Primal objective: {primal_obj:.4f}")
    print(f"    Dual objective:   {dual_obj:.4f}")
    print(f"    Gap: {abs(primal_obj - dual_obj):.6f}")

    # (c) Support vectors
    print(f"\n(c) Support vectors: {len(sv_indices)} out of {N} points")
    predictions = np.sign(X @ w_dual + b_dual)
    accuracy = np.mean(predictions == y)
    print(f"    Training accuracy: {accuracy:.2%}")


# === Exercise 4: Fenchel Conjugate ===
# Problem: Derive Fenchel conjugate for (a) 0.5x^2 + x, (b) |x|, (c) max(0,x)

def exercise_4():
    """Fenchel conjugate computation and verification."""
    print("Fenchel conjugate: f*(y) = sup_x {xy - f(x)}")
    print()

    # (a) f(x) = 0.5*x^2 + x
    print("(a) f(x) = 0.5*x^2 + x")
    print("    f*(y) = sup_x {xy - 0.5*x^2 - x}")
    print("    d/dx [xy - 0.5*x^2 - x] = y - x - 1 = 0 => x* = y - 1")
    print("    f*(y) = (y-1)*y - 0.5*(y-1)^2 - (y-1)")
    print("          = y^2 - y - 0.5*(y^2 - 2y + 1) - y + 1")
    print("          = 0.5*y^2 - y + 0.5 - y + 1... let me compute carefully:")
    # f*(y) = (y-1)y - 0.5(y-1)^2 - (y-1)
    #       = y(y-1) - (y-1)[0.5(y-1) + 1]
    #       = (y-1)[y - 0.5y + 0.5 - 1]
    #       = (y-1)(0.5y - 0.5)
    #       = 0.5(y-1)^2
    print("    f*(y) = 0.5*(y-1)^2")

    # Verify numerically
    y_test = 3.0
    f_star_analytical = 0.5 * (y_test - 1)**2
    # Numerical: maximize xy - f(x) over x
    x_grid = np.linspace(-10, 10, 10000)
    vals = y_test * x_grid - 0.5 * x_grid**2 - x_grid
    f_star_numerical = np.max(vals)
    print(f"    Verification at y={y_test}: analytical={f_star_analytical:.4f}, numerical={f_star_numerical:.4f}")
    print()

    # (b) f(x) = |x|  (indicator constraint: f*(y) = 0 if |y| <= 1, else inf)
    print("(b) f(x) = |x|")
    print("    f*(y) = sup_x {xy - |x|}")
    print("    If |y| <= 1: xy - |x| <= |y||x| - |x| = (|y|-1)|x| <= 0, achieved at x=0")
    print("    If |y| > 1: xy - |x| is unbounded (take x -> sign(y)*inf)")
    print("    => f*(y) = 0 if |y| <= 1, +inf otherwise")
    print("    This is the indicator function of [-1, 1]")

    for y_val in [0.5, 1.0, 1.5]:
        x_grid = np.linspace(-100, 100, 100000)
        vals = y_val * x_grid - np.abs(x_grid)
        f_star = np.max(vals)
        print(f"    f*({y_val}) = {f_star:.4f} ({'bounded' if f_star < 10 else 'unbounded'})")
    print()

    # (c) f(x) = max(0, x) = ReLU(x)
    print("(c) f(x) = max(0, x) = ReLU(x)")
    print("    f*(y) = sup_x {xy - max(0,x)}")
    print("    For x >= 0: xy - x = x(y-1)")
    print("      If y < 1: maximized at x=0, giving 0")
    print("      If y = 1: sup = 0")
    print("      If y > 1: unbounded")
    print("    For x < 0: xy, maximized at x=0 if y >= 0")
    print("    For y < 0: xy for x<0 is unbounded")
    print("    => f*(y) = 0 if 0 <= y <= 1, +inf otherwise")
    print("    This is the indicator function of [0, 1]")

    for y_val in [-0.5, 0.0, 0.5, 1.0, 1.5]:
        x_grid = np.linspace(-100, 100, 100000)
        vals = y_val * x_grid - np.maximum(0, x_grid)
        f_star = np.max(vals)
        status = 'finite' if f_star < 10 else 'unbounded'
        print(f"    f*({y_val}) = {f_star:.2f} ({status})")


# === Exercise 5: ADMM Application ===
# Problem: Solve distributed linear regression using ADMM.

def exercise_5():
    """ADMM for distributed linear regression."""
    np.random.seed(42)

    # Generate data distributed across N nodes
    N_nodes = 4
    n_per_node = 50
    d = 5
    true_w = np.array([2, -1, 0.5, 3, -0.5])

    A_list = []
    b_list = []
    for i in range(N_nodes):
        A_i = np.random.randn(n_per_node, d)
        b_i = A_i @ true_w + np.random.randn(n_per_node) * 0.5
        A_list.append(A_i)
        b_list.append(b_i)

    # Centralized solution for reference
    A_full = np.vstack(A_list)
    b_full = np.concatenate(b_list)
    w_centralized = np.linalg.lstsq(A_full, b_full, rcond=None)[0]

    # ADMM: min sum_i 0.5||A_i x_i - b_i||^2  s.t. x_i = z (consensus)
    # x_i update: argmin 0.5||A_i x_i - b_i||^2 + (rho/2)||x_i - z + u_i||^2
    # z update: z = mean(x_i + u_i)
    # u_i update: u_i = u_i + x_i - z

    rho = 1.0
    n_iterations = 100

    # Initialize
    x_list = [np.zeros(d) for _ in range(N_nodes)]
    z = np.zeros(d)
    u_list = [np.zeros(d) for _ in range(N_nodes)]

    print(f"ADMM for Distributed Linear Regression")
    print(f"  {N_nodes} nodes, {n_per_node} samples each, {d} features")
    print(f"  rho = {rho}")
    print()

    for iteration in range(n_iterations):
        # x-update (per node)
        for i in range(N_nodes):
            # Solve: (A_i^T A_i + rho*I) x_i = A_i^T b_i + rho*(z - u_i)
            lhs = A_list[i].T @ A_list[i] + rho * np.eye(d)
            rhs = A_list[i].T @ b_list[i] + rho * (z - u_list[i])
            x_list[i] = np.linalg.solve(lhs, rhs)

        # z-update (consensus)
        z = np.mean([x_list[i] + u_list[i] for i in range(N_nodes)], axis=0)

        # u-update (dual variable)
        for i in range(N_nodes):
            u_list[i] = u_list[i] + x_list[i] - z

        # Compute residuals
        primal_residual = np.sqrt(sum(np.sum((x_list[i] - z)**2) for i in range(N_nodes)))
        dist_to_optimal = np.linalg.norm(z - w_centralized)

        if iteration < 5 or (iteration + 1) % 20 == 0:
            print(f"  Iter {iteration+1:3d}: ||x-z|| = {primal_residual:.6f}, "
                  f"||z - w*|| = {dist_to_optimal:.6f}")

    print()
    print(f"ADMM solution:       {np.round(z, 4)}")
    print(f"Centralized solution: {np.round(w_centralized, 4)}")
    print(f"True weights:         {true_w}")
    print(f"Distance ADMM->centralized: {np.linalg.norm(z - w_centralized):.6f}")


if __name__ == "__main__":
    print("=== Exercise 1: Proving Convexity ===")
    exercise_1()
    print("\n=== Exercise 2: Lagrangian Duality ===")
    exercise_2()
    print("\n=== Exercise 3: SVM Implementation ===")
    exercise_3()
    print("\n=== Exercise 4: Fenchel Conjugate ===")
    exercise_4()
    print("\n=== Exercise 5: ADMM Application ===")
    exercise_5()
    print("\nAll exercises completed!")
