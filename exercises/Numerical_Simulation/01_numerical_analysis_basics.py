"""
Exercises for Lesson 01: Numerical Analysis Basics

Topics: Machine epsilon, numerical differentiation, numerical integration,
        condition number, optimal step size.
"""

import numpy as np
from scipy.integrate import quad

# ---------------------------------------------------------------------------
# Exercise 1: Machine Epsilon and Precision Loss
# ---------------------------------------------------------------------------
# Explain what machine epsilon represents for float64 and float32.
# Demonstrate catastrophic cancellation: compute (a + b) - a for a = 1e15
# and b = 1.0, explain why the result is incorrect, and show how to
# rewrite the expression to avoid the issue.
# ---------------------------------------------------------------------------

def exercise_1():
    """Machine epsilon and catastrophic cancellation demonstration."""
    # Machine epsilon values
    print(f"float64 machine epsilon: {np.finfo(np.float64).eps:.2e}")
    print(f"float32 machine epsilon: {np.finfo(np.float32).eps:.2e}")

    # Catastrophic cancellation
    a = 1e15
    b = 1.0

    # Incorrect: a and b differ by 15 orders of magnitude
    result_bad = (a + b) - a
    print(f"\n(a + b) - a = {result_bad}")  # precision of b is lost

    # Correct: subtract before adding
    result_good = b + (a - a)
    print(f"b + (a - a) = {result_good}")  # mathematically equivalent, numerically stable

    # When a >> b, use stable_form = b directly
    # Key insight: reorder operations to avoid adding small to large


# ---------------------------------------------------------------------------
# Exercise 2: Forward vs. Central Difference Error Scaling
# ---------------------------------------------------------------------------
# For f(x) = cos(x) at x = 0.7, compute numerical derivatives using the
# forward difference (O(h)) and central difference (O(h^2)) formulas at
# step sizes h = 1e-1, 1e-3, 1e-5, 1e-7. Compute errors relative to
# f'(x) = -sin(x) and verify the expected convergence orders.
# ---------------------------------------------------------------------------

def exercise_2():
    """Forward vs. central difference error scaling."""
    x = 0.7
    true_val = -np.sin(x)
    h_values = [1e-1, 1e-3, 1e-5, 1e-7]

    print(f"True derivative: {true_val:.10f}\n")
    print(f"{'h':>8}  {'Forward error':>14}  {'Central error':>14}")
    print("-" * 42)

    prev_fwd, prev_cen = None, None
    for h in h_values:
        fwd = (np.cos(x + h) - np.cos(x)) / h
        cen = (np.cos(x + h) - np.cos(x - h)) / (2 * h)
        e_fwd = abs(fwd - true_val)
        e_cen = abs(cen - true_val)
        print(f"{h:8.0e}  {e_fwd:14.2e}  {e_cen:14.2e}", end="")
        if prev_fwd is not None:
            print(f"  [fwd ratio: {prev_fwd/e_fwd:.0f}, cen ratio: {prev_cen/e_cen:.0f}]", end="")
        print()
        prev_fwd, prev_cen = e_fwd, e_cen


# ---------------------------------------------------------------------------
# Exercise 3: Composite Simpson's Rule Convergence
# ---------------------------------------------------------------------------
# Implement composite Simpson's rule and verify its O(h^4) convergence
# by computing integral_0^1 x^3 e^x dx for n = 4, 8, 16, 32, 64.
# Exact value: e - 2. Compute estimated convergence order from successive
# error ratios.
# ---------------------------------------------------------------------------

def exercise_3():
    """Composite Simpson's rule convergence verification."""

    def simpson(f, a, b, n):
        """Composite Simpson's 1/3 rule (n must be even)."""
        if n % 2 != 0:
            n += 1
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = f(x)
        return h / 3 * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]) + y[-1])

    f = lambda x: x**3 * np.exp(x)
    exact, _ = quad(f, 0, 1)

    print(f"Exact value: {exact:.10f}")
    print(f"\n{'n':>6}  {'Approx':>14}  {'Error':>12}  {'Order':>8}")
    print("-" * 46)

    prev_error = None
    for n in [4, 8, 16, 32, 64]:
        approx = simpson(f, 0, 1, n)
        error = abs(approx - exact)
        if prev_error is not None:
            order = np.log2(prev_error / error)
            print(f"{n:>6}  {approx:14.10f}  {error:12.2e}  {order:8.2f}")
        else:
            print(f"{n:>6}  {approx:14.10f}  {error:12.2e}  {'---':>8}")
        prev_error = error


# ---------------------------------------------------------------------------
# Exercise 4: Condition Number and Linear System Sensitivity
# ---------------------------------------------------------------------------
# Create a nearly-singular matrix with condition number > 1e5 and solve
# Ax = b with exact and perturbed b. Show that relative error in x can be
# much larger than relative error in b.
# ---------------------------------------------------------------------------

def exercise_4():
    """Condition number and linear system sensitivity."""
    # Hilbert matrix (notoriously ill-conditioned)
    n = 5
    A = np.array([[1 / (i + j + 1) for j in range(n)] for i in range(n)])
    cond = np.linalg.cond(A)
    print(f"Condition number of 5x5 Hilbert matrix: {cond:.2e}")

    # Exact right-hand side: x_true = [1, 1, 1, 1, 1]
    x_true = np.ones(n)
    b_exact = A @ x_true

    # Perturbed right-hand side
    rng = np.random.default_rng(0)
    delta_b = rng.standard_normal(n)
    delta_b *= 1e-6 / np.linalg.norm(delta_b)

    b_perturbed = b_exact + delta_b

    # Solve both systems
    x_exact = np.linalg.solve(A, b_exact)
    x_perturbed = np.linalg.solve(A, b_perturbed)

    rel_b_error = np.linalg.norm(delta_b) / np.linalg.norm(b_exact)
    rel_x_error = np.linalg.norm(x_perturbed - x_exact) / np.linalg.norm(x_exact)

    print(f"\nRelative error in b:  {rel_b_error:.2e}")
    print(f"Relative error in x:  {rel_x_error:.2e}")
    print(f"Amplification factor: {rel_x_error / rel_b_error:.1f}")
    print(f"Condition number:     {cond:.2e}")
    print(f"\nTheory: ||dx||/||x|| <= kappa(A) * ||db||/||b||")
    print(f"Bound:  {cond * rel_b_error:.2e}  (actual: {rel_x_error:.2e})")


# ---------------------------------------------------------------------------
# Exercise 5: Optimal Step Size for Central Difference
# ---------------------------------------------------------------------------
# For f(x) = sin(x) at x = 1.0, use central differences with step sizes
# from h = 1e-1 to h = 1e-15 and identify the optimal step size where
# truncation error and rounding error balance. Derive h_opt ~ eps^(1/3).
# ---------------------------------------------------------------------------

def exercise_5():
    """Optimal step size for central difference."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    f = np.sin
    x = 1.0
    f_true = np.cos(x)

    h_values = np.logspace(-1, -15, 150)
    errors = [abs((f(x + h) - f(x - h)) / (2 * h) - f_true) for h in h_values]

    eps = np.finfo(float).eps
    h_opt = eps ** (1 / 3)
    print(f"Machine epsilon:  {eps:.2e}")
    print(f"Theoretical h_opt ~ eps^(1/3) = {h_opt:.2e}")

    plt.figure(figsize=(9, 5))
    plt.loglog(h_values, errors, 'b-', linewidth=1.5, label='Actual error')
    plt.axvline(h_opt, color='r', linestyle='--', label=f'Theoretical h_opt ~ {h_opt:.0e}')
    plt.xlabel('Step size h')
    plt.ylabel('Absolute error')
    plt.title('Central Difference Error vs Step Size')
    plt.legend()
    plt.grid(True, which='both', alpha=0.4)
    plt.savefig('ex01_optimal_step_size.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved to ex01_optimal_step_size.png")

    # Observed minimum
    min_idx = np.argmin(errors)
    print(f"Observed optimal h ~ {h_values[min_idx]:.2e}  (min error: {errors[min_idx]:.2e})")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 1: Machine Epsilon and Precision Loss")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("Exercise 2: Forward vs. Central Difference Error Scaling")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("Exercise 3: Composite Simpson's Rule Convergence")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("Exercise 4: Condition Number and Linear System Sensitivity")
    print("=" * 60)
    exercise_4()

    print("\n" + "=" * 60)
    print("Exercise 5: Optimal Step Size for Central Difference")
    print("=" * 60)
    exercise_5()
