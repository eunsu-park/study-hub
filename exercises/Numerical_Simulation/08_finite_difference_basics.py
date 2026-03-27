"""
Exercises for Lesson 08: Finite Difference Basics

Topics: Difference accuracy verification, numerical second derivative,
        CFL condition calculation, sparse matrix Poisson equation.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

# ---------------------------------------------------------------------------
# Exercise 1: Verify Difference Accuracy
# ---------------------------------------------------------------------------
# For f(x) = e^x at x = 1, calculate forward/backward/central differences
# and compare errors at h = 0.1, 0.01, 0.001.
# ---------------------------------------------------------------------------

def exercise_1():
    """Verify difference formula accuracy for f(x) = e^x at x = 1."""
    f = np.exp
    x = 1.0
    f_exact = np.exp(x)  # f'(x) = e^x

    h_values = [0.1, 0.01, 0.001]

    print(f"Exact f'(1) = e = {f_exact:.10f}")
    print()
    print(f"{'h':>8}  {'Forward':>14}  {'Backward':>14}  {'Central':>14}")
    print(f"{'':>8}  {'Error':>14}  {'Error':>14}  {'Error':>14}")
    print("-" * 58)

    prev_fwd = prev_bwd = prev_cen = None
    for h in h_values:
        fwd = (f(x + h) - f(x)) / h
        bwd = (f(x) - f(x - h)) / h
        cen = (f(x + h) - f(x - h)) / (2 * h)

        e_fwd = abs(fwd - f_exact)
        e_bwd = abs(bwd - f_exact)
        e_cen = abs(cen - f_exact)

        line = f"{h:>8.3f}  {e_fwd:>14.2e}  {e_bwd:>14.2e}  {e_cen:>14.2e}"
        if prev_fwd is not None:
            r_fwd = prev_fwd / e_fwd
            r_bwd = prev_bwd / e_bwd
            r_cen = prev_cen / e_cen
            line += f"  ratios: {r_fwd:.0f} / {r_bwd:.0f} / {r_cen:.0f}"
        print(line)
        prev_fwd, prev_bwd, prev_cen = e_fwd, e_bwd, e_cen

    print()
    print("Forward/Backward: error ratio ~ 10 when h shrinks 10x => O(h)")
    print("Central:          error ratio ~ 100 when h shrinks 10x => O(h^2)")


# ---------------------------------------------------------------------------
# Exercise 2: Numerical Second Derivative
# ---------------------------------------------------------------------------
# For f(x) = x^4, calculate f''(x) using central differences and compare
# with the exact value 12*x^2.
# ---------------------------------------------------------------------------

def exercise_2():
    """Numerical second derivative of f(x) = x^4."""
    f = lambda x: x**4
    f_2nd_exact = lambda x: 12 * x**2

    x_test = 2.0
    h_values = [0.1, 0.05, 0.01, 0.005, 0.001]

    print(f"f(x) = x^4,  f''(x) = 12x^2")
    print(f"Exact f''({x_test}) = {f_2nd_exact(x_test):.6f}")
    print()
    print(f"{'h':>8}  {'Numerical':>14}  {'Error':>12}  {'Order':>8}")
    print("-" * 46)

    prev_err = None
    for h in h_values:
        # Central difference second derivative: (f(x+h) - 2f(x) + f(x-h)) / h^2
        d2f = (f(x_test + h) - 2 * f(x_test) + f(x_test - h)) / h**2
        err = abs(d2f - f_2nd_exact(x_test))
        if prev_err is not None and err > 0:
            order = np.log(prev_err / err) / np.log(h_values[h_values.index(h) - 1] / h)
            print(f"{h:>8.4f}  {d2f:>14.6f}  {err:>12.2e}  {order:>8.2f}")
        else:
            print(f"{h:>8.4f}  {d2f:>14.6f}  {err:>12.2e}  {'---':>8}")
        prev_err = err

    print()
    print("Note: For f(x)=x^4, the 4th derivative f''''=24 is constant,")
    print("so the central 2nd-derivative formula has truncation error")
    print("(h^2/12)*f''''(x) = 2h^2, which is O(h^2).")

    # Show for a grid of x values
    x_grid = np.linspace(0, 3, 100)
    h = 0.01
    d2f_numerical = (f(x_grid + h) - 2 * f(x_grid) + f(x_grid - h)) / h**2
    d2f_exact = f_2nd_exact(x_grid)
    max_err = np.max(np.abs(d2f_numerical - d2f_exact))
    print(f"\nGrid test (h={h}): max error over [0,3] = {max_err:.2e}")


# ---------------------------------------------------------------------------
# Exercise 3: Calculate CFL Condition
# ---------------------------------------------------------------------------
# For thermal diffusivity alpha = 0.05 and grid spacing dx = 0.01,
# find the maximum time step dt for FTCS method stability.
# ---------------------------------------------------------------------------

def exercise_3():
    """Calculate CFL condition for FTCS heat equation."""
    alpha = 0.05
    dx = 0.01

    # FTCS stability condition: r = alpha * dt / dx^2 <= 0.5
    dt_max = 0.5 * dx**2 / alpha
    print(f"Heat equation FTCS stability condition:")
    print(f"  r = alpha * dt / dx^2 <= 0.5")
    print()
    print(f"Parameters:")
    print(f"  alpha = {alpha}")
    print(f"  dx    = {dx}")
    print()
    print(f"Maximum dt = 0.5 * dx^2 / alpha")
    print(f"           = 0.5 * {dx}^2 / {alpha}")
    print(f"           = 0.5 * {dx**2:.4e} / {alpha}")
    print(f"           = {dt_max:.4e}")
    print()

    # Verify with a quick simulation
    print("Verification: running FTCS with dt_stable and dt_unstable...")
    nx = 101
    L = 1.0
    x = np.linspace(0, L, nx)
    dx_actual = L / (nx - 1)

    u0 = np.sin(np.pi * x)

    def ftcs_run(u0, alpha, dx, dt, n_steps):
        r = alpha * dt / dx**2
        u = u0.copy()
        for _ in range(n_steps):
            u_new = u.copy()
            u_new[1:-1] = u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2])
            u_new[0] = 0
            u_new[-1] = 0
            u = u_new
        return u, r

    # Stable
    dt_s = 0.4 * dx_actual**2 / alpha
    u_s, r_s = ftcs_run(u0, alpha, dx_actual, dt_s, 100)
    print(f"  Stable:   dt={dt_s:.6e}, r={r_s:.4f}, max|u|={np.max(np.abs(u_s)):.6f}")

    # Unstable
    dt_u = 1.2 * dx_actual**2 / alpha
    u_u, r_u = ftcs_run(u0, alpha, dx_actual, dt_u, 100)
    print(f"  Unstable: dt={dt_u:.6e}, r={r_u:.4f}, max|u|={np.max(np.abs(u_u)):.2e}")


# ---------------------------------------------------------------------------
# Exercise 4: Sparse Matrix Poisson Equation
# ---------------------------------------------------------------------------
# Modify the 1D Poisson example to solve for f(x) = 1 (constant source).
# Analytical solution: u(x) = x(1-x)/2.
# ---------------------------------------------------------------------------

def exercise_4():
    """Solve 1D Poisson equation with constant source f(x) = 1."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # -d2u/dx2 = 1 on [0, 1], u(0) = u(1) = 0
    # Analytical solution: u(x) = x(1-x)/2
    nx = 101
    L = 1.0
    dx = L / (nx - 1)
    x = np.linspace(0, L, nx)

    # Interior points
    x_inner = x[1:-1]
    f = np.ones(nx - 2)  # f(x) = 1

    # Laplacian matrix (-d2/dx2)
    n = nx - 2
    main_diag = 2.0 * np.ones(n)
    off_diag = -1.0 * np.ones(n - 1)
    A = sparse.diags([off_diag, main_diag, off_diag], [-1, 0, 1], format='csr')
    A = A / dx**2

    # Solve
    u_inner = spsolve(A, f)

    # Full solution
    u = np.zeros(nx)
    u[1:-1] = u_inner

    # Analytical solution
    u_exact = x * (1 - x) / 2

    error = np.max(np.abs(u - u_exact))
    print(f"1D Poisson Equation: -u'' = 1, u(0)=u(1)=0")
    print(f"Analytical solution: u(x) = x(1-x)/2")
    print(f"Grid points: {nx}")
    print(f"Maximum error: {error:.2e}")

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(x, u_exact, 'b-', linewidth=2, label='Analytical')
    plt.plot(x, u, 'ro', markersize=3, markevery=5, label='Numerical')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Solution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(x, np.abs(u - u_exact), 'g-', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('|Error|')
    plt.title(f'Error (max: {error:.2e})')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex08_poisson_constant.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved to ex08_poisson_constant.png")

    # Convergence test
    print("\nConvergence test:")
    print(f"{'nx':>6}  {'dx':>10}  {'Max error':>12}  {'Order':>8}")
    print("-" * 42)
    prev_err = None
    prev_dx = None
    for nx_test in [11, 21, 41, 81, 161]:
        dx_test = L / (nx_test - 1)
        x_test = np.linspace(0, L, nx_test)
        n_test = nx_test - 2
        main = 2.0 * np.ones(n_test)
        off = -1.0 * np.ones(n_test - 1)
        A_test = sparse.diags([off, main, off], [-1, 0, 1], format='csr') / dx_test**2
        u_test = np.zeros(nx_test)
        u_test[1:-1] = spsolve(A_test, np.ones(n_test))
        u_ex = x_test * (1 - x_test) / 2
        err = np.max(np.abs(u_test - u_ex))
        if prev_err is not None:
            order = np.log(prev_err / err) / np.log(prev_dx / dx_test)
            print(f"{nx_test:>6}  {dx_test:>10.4e}  {err:>12.2e}  {order:>8.2f}")
        else:
            print(f"{nx_test:>6}  {dx_test:>10.4e}  {err:>12.2e}  {'---':>8}")
        prev_err = err
        prev_dx = dx_test

    print("\nOrder ~ 2.0 confirms the 2nd-order central difference scheme.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 1: Verify Difference Accuracy")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("Exercise 2: Numerical Second Derivative")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("Exercise 3: Calculate CFL Condition")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("Exercise 4: Sparse Matrix Poisson Equation")
    print("=" * 60)
    exercise_4()
