"""
Convexity, Lagrangian Duality, and Proximal Operators

Demonstrates:
- Convex function verification (Jensen's inequality, second-order condition)
- Lagrangian formulation and KKT conditions
- Strong vs weak duality
- Proximal operators for L1 (soft-thresholding) and L2 regularization

Dependencies: numpy, scipy
"""

import numpy as np
from scipy.optimize import minimize, linprog


# ---------------------------------------------------------------------------
# 1. Convexity Verification
# ---------------------------------------------------------------------------

def convexity_verification():
    """Verify convexity using Jensen's inequality and Hessian PSD check"""
    print("=" * 60)
    print("CONVEXITY VERIFICATION")
    print("=" * 60)

    print("\nA function f is convex iff:")
    print("  (a) Jensen: f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)  for λ∈[0,1]")
    print("  (b) Second-order: Hessian H is positive semidefinite everywhere")

    # ----- Example 1: f(x) = x^2 (convex) -----
    print("\n--- f(x) = x²  (convex) ---")

    def f_sq(x):
        return x**2

    np.random.seed(42)
    n_checks = 1000
    x_vals = np.random.uniform(-5, 5, n_checks)
    y_vals = np.random.uniform(-5, 5, n_checks)
    lambdas = np.random.uniform(0, 1, n_checks)

    midpoints = lambdas * x_vals + (1 - lambdas) * y_vals
    lhs = f_sq(midpoints)
    rhs = lambdas * f_sq(x_vals) + (1 - lambdas) * f_sq(y_vals)
    violations = np.sum(lhs > rhs + 1e-10)
    print(f"Jensen check ({n_checks} random triples): {violations} violations → convex: {violations == 0}")

    # Second-order: f''(x) = 2 > 0 everywhere
    print("Second-order: f''(x) = 2 > 0 for all x → PSD ✓")

    # ----- Example 2: f(x) = -x^2 (concave/non-convex) -----
    print("\n--- f(x) = -x²  (concave) ---")

    def f_neg_sq(x):
        return -x**2

    lhs2 = f_neg_sq(midpoints)
    rhs2 = lambdas * f_neg_sq(x_vals) + (1 - lambdas) * f_neg_sq(y_vals)
    violations2 = np.sum(lhs2 > rhs2 + 1e-10)
    print(f"Jensen check: {violations2} violations → convex: {violations2 == 0}")
    print("Second-order: f''(x) = -2 < 0 → NOT PSD → concave (not convex)")

    # ----- Example 3: Multivariate — check Hessian PSD -----
    print("\n--- f(x,y) = x² + xy + y²  (convex) ---")
    print("Hessian H = [[2, 1], [1, 2]]")
    H = np.array([[2.0, 1.0], [1.0, 2.0]])
    eigenvalues = np.linalg.eigvalsh(H)
    print(f"Eigenvalues of H: {eigenvalues}")
    print(f"All eigenvalues ≥ 0: {np.all(eigenvalues >= 0)} → PSD → convex")

    print("\n--- f(x,y) = x² - 2xy + y²  (convex but not strictly) ---")
    print("Hessian H = [[2, -2], [-2, 2]]")
    H2 = np.array([[2.0, -2.0], [-2.0, 2.0]])
    eigenvalues2 = np.linalg.eigvalsh(H2)
    print(f"Eigenvalues: {eigenvalues2}")
    print(f"All eigenvalues ≥ 0: {np.all(eigenvalues2 >= 0)} → PSD (note: λ_min = 0 → not strict)")

    # ----- Convex combination preserves convexity -----
    print("\n--- Operations Preserving Convexity ---")
    print("If f, g are convex and α, β ≥ 0: αf + βg is convex")
    print("Composition: f(Ax + b) is convex if f is convex")
    print("Pointwise max: max(f, g) is convex if both f, g are convex")
    print("Affine: f(x) = aᵀx + b is both convex and concave")


# ---------------------------------------------------------------------------
# 2. Lagrangian Duality
# ---------------------------------------------------------------------------

def lagrangian_duality():
    """
    Demonstrate Lagrangian duality with a quadratic program.

    Primal:  min  ½‖x‖²
             s.t. aᵀx = b

    Dual:    max  -½ λ²‖a‖² + λb
             (over λ ∈ R)
    """
    print("\n" + "=" * 60)
    print("LAGRANGIAN DUALITY")
    print("=" * 60)

    print("\nPrimal problem:")
    print("  min_x  ½‖x‖²")
    print("  s.t.   aᵀx = b    (equality constraint)")
    print()
    print("Lagrangian: L(x, λ) = ½‖x‖² + λ(aᵀx - b)")
    print()
    print("Dual function: g(λ) = min_x L(x, λ) = -½λ²‖a‖² - λb")
    print("  (obtained by setting ∂L/∂x = x + λa = 0  →  x* = -λa)")
    print()
    print("Dual problem: max_λ g(λ) = -½λ²‖a‖² - λb")

    a = np.array([1.0, 2.0, 3.0])
    b = 6.0

    # Primal solution: project 0 onto hyperplane aᵀx = b
    # x* = b * a / ‖a‖²
    x_primal = b * a / np.dot(a, a)
    primal_obj = 0.5 * np.dot(x_primal, x_primal)

    print(f"\nWith a = {a},  b = {b}:")
    print(f"  Primal solution: x* = {x_primal}")
    print(f"  Primal objective: {primal_obj:.6f}")

    # Dual solution: λ* = -b / ‖a‖²
    lambda_star = -b / np.dot(a, a)
    dual_obj = -0.5 * lambda_star**2 * np.dot(a, a) - lambda_star * b

    print(f"  Dual solution: λ* = {lambda_star:.6f}")
    print(f"  Dual objective: {dual_obj:.6f}")

    gap = primal_obj - dual_obj
    print(f"  Duality gap (primal - dual): {gap:.2e}")
    print(f"  Strong duality holds (gap ≈ 0): {abs(gap) < 1e-10}")

    # KKT conditions
    print("\n--- KKT Conditions (Necessary & Sufficient for Convex Programs) ---")
    print("1. Stationarity:  ∇f(x*) + λ* ∇g(x*) = 0")
    grad_L = x_primal + lambda_star * a
    print(f"   x* + λ* a = {grad_L}  (should be ≈ 0)")
    print(f"   Satisfied: {np.allclose(grad_L, 0, atol=1e-10)}")

    print("2. Primal feasibility: aᵀx* - b = 0")
    feasibility = np.dot(a, x_primal) - b
    print(f"   aᵀx* - b = {feasibility:.2e}  (should be ≈ 0)")
    print(f"   Satisfied: {abs(feasibility) < 1e-10}")

    print("\n--- Weak Duality (always holds) ---")
    print("For any primal-feasible x and dual-feasible λ:")
    print("  g(λ) ≤ p* ≤ f(x)   (dual is lower bound on primal)")

    # Demonstrate weak duality
    lambdas = np.linspace(-3, 0, 50)
    dual_vals = -0.5 * lambdas**2 * np.dot(a, a) - lambdas * b
    print(f"  max g(λ) = {dual_vals.max():.6f} ≤ p* = {primal_obj:.6f}")
    print(f"  Weak duality confirmed: {dual_vals.max() <= primal_obj + 1e-10}")


def inequality_constraints_kkt():
    """KKT conditions for inequality constraints"""
    print("\n" + "=" * 60)
    print("KKT CONDITIONS: INEQUALITY CONSTRAINTS")
    print("=" * 60)

    print("\nProblem:")
    print("  min  (x - 3)² + (y - 2)²")
    print("  s.t. x + y ≤ 4       (constraint 1)")
    print("       x ≥ 0, y ≥ 0    (non-negativity)")

    print("\nKKT conditions for inequality g_i(x) ≤ 0:")
    print("1. Stationarity:  ∇f + Σ μᵢ ∇gᵢ = 0")
    print("2. Primal feas.:  g_i(x*) ≤ 0")
    print("3. Dual feas.:    μᵢ ≥ 0")
    print("4. Complementary slackness: μᵢ g_i(x*) = 0")

    # Solve numerically
    def objective(xy):
        x, y = xy
        return (x - 3)**2 + (y - 2)**2

    def jac_objective(xy):
        x, y = xy
        return np.array([2*(x - 3), 2*(y - 2)])

    constraints = [{'type': 'ineq', 'fun': lambda xy: 4 - xy[0] - xy[1]}]
    bounds = [(0, None), (0, None)]

    result = minimize(objective, x0=[1.0, 1.0], jac=jac_objective,
                      method='SLSQP', bounds=bounds, constraints=constraints)

    x_star, y_star = result.x
    print(f"\nNumerical solution: x* = ({x_star:.4f}, {y_star:.4f})")
    print(f"Objective value: {result.fun:.4f}")

    # KKT verification
    g1 = x_star + y_star - 4
    print(f"\nConstraint g₁ = x + y - 4 = {g1:.6f} (≤ 0: {g1 <= 1e-8})")
    print(f"x ≥ 0: {x_star >= -1e-8},  y ≥ 0: {y_star >= -1e-8}")

    # Check if constraint is active
    active = abs(g1) < 1e-6
    print(f"Constraint g₁ active (binding): {active}")
    if active:
        print("  → μ₁ can be non-zero (constraint matters at solution)")
    else:
        print("  → μ₁ = 0 (inactive constraint by complementary slackness)")


# ---------------------------------------------------------------------------
# 3. Proximal Operators
# ---------------------------------------------------------------------------

def proximal_operators():
    """
    Proximal operators: prox_f(v) = argmin_x [f(x) + ½‖x - v‖²]

    Useful in proximal gradient methods for non-smooth objectives.
    """
    print("\n" + "=" * 60)
    print("PROXIMAL OPERATORS")
    print("=" * 60)

    print("\nprox_{λf}(v) = argmin_x [ f(x) + (1/2λ)‖x - v‖² ]")

    # --- Prox of L1 norm: soft-thresholding ---
    print("\n--- Prox of λ‖·‖₁: Soft-Thresholding ---")
    print("prox_{λ‖·‖₁}(v) = sign(v) · max(|v| - λ, 0)")

    def soft_threshold(v, lam):
        return np.sign(v) * np.maximum(np.abs(v) - lam, 0)

    v = np.array([-3.0, -0.5, 0.0, 0.5, 1.0, 2.5])
    for lam in [0.5, 1.0, 2.0]:
        prox = soft_threshold(v, lam)
        print(f"  λ={lam}: prox({v}) = {prox}")

    print("\nKey property: values with |v| < λ are set to 0 (sparsity!)")

    # Verify by solving the subproblem numerically for one case
    v0, lam0 = 2.5, 1.0
    result = minimize(
        lambda x: lam0 * np.abs(x[0]) + 0.5 * (x[0] - v0)**2,
        x0=[0.0], method='Nelder-Mead',
        options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 10000}
    )
    print(f"\nVerification (numerical argmin): prox_{{{lam0}·‖·‖₁}}({v0})")
    print(f"  Analytical: {soft_threshold(v0, lam0):.6f}")
    print(f"  Numerical:  {result.x[0]:.6f}")

    # --- Prox of L2 squared: shrinkage ---
    print("\n--- Prox of (λ/2)‖·‖₂²: L2 Shrinkage ---")
    print("prox_{(λ/2)‖·‖₂²}(v) = v / (1 + λ)")

    def prox_l2sq(v, lam):
        return v / (1 + lam)

    v_vec = np.array([3.0, -1.0, 2.0])
    for lam in [0.5, 1.0, 2.0]:
        prox = prox_l2sq(v_vec, lam)
        print(f"  λ={lam}: prox({v_vec}) = {prox}")

    print("\nKey property: L2 prox shrinks all coordinates uniformly (no sparsity)")

    # --- Prox of indicator: projection ---
    print("\n--- Prox of Indicator Function: Projection ---")
    print("For a convex set C, prox_{δ_C}(v) = argmin_{x ∈ C} ‖x - v‖² = Proj_C(v)")

    # Project onto L2 ball: ‖x‖₂ ≤ r
    def proj_l2_ball(v, r=1.0):
        norm = np.linalg.norm(v)
        if norm <= r:
            return v.copy()
        return v * r / norm

    test_points = [np.array([0.5, 0.3]), np.array([2.0, 1.0]), np.array([0.0, 3.0])]
    print("\nProjection onto unit L2 ball (r=1):")
    for p in test_points:
        proj = proj_l2_ball(p, r=1.0)
        print(f"  v={p} → proj={np.round(proj, 4)},  ‖proj‖={np.linalg.norm(proj):.4f}")

    # --- Proximal Gradient Descent sketch ---
    print("\n--- Proximal Gradient Descent (Lasso) ---")
    print("Objective: min_x ½‖Ax - b‖² + λ‖x‖₁")
    print("Update:    x ← prox_{λα·‖·‖₁}(x - α·Aᵀ(Ax - b))")

    np.random.seed(7)
    n, p = 50, 20
    A = np.random.randn(n, p)
    x_true = np.zeros(p)
    x_true[:5] = np.array([2.0, -1.5, 3.0, -2.0, 1.0])  # sparse true signal
    b = A @ x_true + 0.1 * np.random.randn(n)

    lam = 0.5
    alpha = 1.0 / np.linalg.eigvalsh(A.T @ A).max()  # step size = 1/L

    x = np.zeros(p)
    losses = []

    for i in range(200):
        grad = A.T @ (A @ x - b)
        x = soft_threshold(x - alpha * grad, lam * alpha)
        loss = 0.5 * np.sum((A @ x - b)**2) + lam * np.sum(np.abs(x))
        losses.append(loss)

    print(f"\nAfter 200 iterations:")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Non-zero coefficients: {np.sum(np.abs(x) > 1e-6)} / {p}")
    print(f"  True support: 5 / {p}")
    print(f"  Recovered coefficients (top 5): {np.round(x[:5], 4)}")
    print(f"  True coefficients (top 5):      {x_true[:5]}")


if __name__ == "__main__":
    convexity_verification()
    lagrangian_duality()
    inequality_constraints_kkt()
    proximal_operators()

    print("\n" + "=" * 60)
    print("All demonstrations completed!")
    print("=" * 60)
