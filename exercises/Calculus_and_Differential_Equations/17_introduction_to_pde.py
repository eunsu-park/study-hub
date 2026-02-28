"""
Exercise Solutions: Lesson 17 - Introduction to Partial Differential Equations
Calculus and Differential Equations

Topics covered:
- PDE classification (elliptic, parabolic, hyperbolic)
- Heat equation with source term derivation
- Boundary condition identification (Dirichlet, Neumann, Robin)
- Backward heat equation ill-posedness
- CFL violation numerical experiment and implicit method
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# Problem 1: PDE Classification
# ============================================================
def exercise_1():
    """
    Classify:
    (a) u_xx + 4u_xy + 4u_yy = 0
    (b) u_xx - 4u_xy + 4u_yy = 0
    (c) y*u_xx + u_yy = 0 (depends on sign of y)
    """
    print("=" * 60)
    print("Problem 1: PDE Classification")
    print("=" * 60)

    # For Au_xx + 2Bu_xy + Cu_yy = 0 (or Au_xx + Bu_xy + Cu_yy = 0)
    # Discriminant D = B^2 - AC (or B^2 - 4AC depending on convention)
    # Here the exercise uses: Au_xx + Bu_xy + Cu_yy
    # D = B^2 - 4AC

    # (a) A=1, B=4, C=4
    A, B, C = 1, 4, 4
    D = B**2 - 4*A*C
    print(f"\n(a) u_xx + 4u_xy + 4u_yy = 0")
    print(f"    A=1, B=4, C=4")
    print(f"    D = B^2 - 4AC = 16 - 16 = {D}")
    print(f"    D = 0 => PARABOLIC")

    # (b) A=1, B=-4, C=4
    A, B, C = 1, -4, 4
    D = B**2 - 4*A*C
    print(f"\n(b) u_xx - 4u_xy + 4u_yy = 0")
    print(f"    A=1, B=-4, C=4")
    print(f"    D = 16 - 16 = {D}")
    print(f"    D = 0 => PARABOLIC")

    # (c) A=y, B=0, C=1
    print(f"\n(c) y*u_xx + u_yy = 0")
    print(f"    A=y, B=0, C=1")
    print(f"    D = 0 - 4y")
    print(f"    If y > 0: D = -4y < 0 => ELLIPTIC (like Laplace equation)")
    print(f"    If y = 0: D = 0     => PARABOLIC (degenerate)")
    print(f"    If y < 0: D = -4y > 0 => HYPERBOLIC (like wave equation)")
    print(f"\n    This is the Tricomi equation -- its type changes with y!")
    print(f"    It arises in transonic aerodynamics where the flow transitions")
    print(f"    from subsonic (elliptic) to supersonic (hyperbolic).")


# ============================================================
# Problem 2: Heat Equation with Source
# ============================================================
def exercise_2():
    """
    Derive 1D heat equation with internal source q(x,t).
    Show u_t = alpha*u_xx + q/(rho*c).
    """
    print("\n" + "=" * 60)
    print("Problem 2: Heat Equation with Source Term")
    print("=" * 60)

    print(f"\n  Derivation of 1D heat equation with source:")
    print(f"\n  Consider a thin rod segment [x, x+dx]:")
    print(f"  Energy balance: rate of energy change = heat flux in - heat flux out + source")
    print(f"\n  rho*c*A*dx * du/dt = -kA*u_x(x+dx,t) + kA*u_x(x,t) + q(x,t)*A*dx")
    print(f"                     = kA*[u_x(x,t) - u_x(x+dx,t)] + q*A*dx")
    print(f"                     = -kA*u_xx*dx + q*A*dx  (Taylor expansion)")
    print(f"\n  Dividing by rho*c*A*dx:")
    print(f"  u_t = (k/(rho*c))*u_xx + q/(rho*c)")
    print(f"      = alpha*u_xx + q/(rho*c)")
    print(f"\n  where alpha = k/(rho*c) is the thermal diffusivity.")

    print(f"\n  For q(x,t) = q_0*sin(pi*x/L):")
    print(f"  This represents a spatially varying heat source that is")
    print(f"  maximum at the center of the rod (x=L/2) and zero at the endpoints.")
    print(f"  Physical scenarios:")
    print(f"  - Electromagnetic induction heating with a specific coil geometry")
    print(f"  - A laser beam with a sinusoidal intensity profile")
    print(f"  - Joule heating from a non-uniform current distribution")


# ============================================================
# Problem 3: Boundary Condition Identification
# ============================================================
def exercise_3():
    """
    Rod: left end at 0C, right end losing heat via Newton cooling.
    Identify BC types.
    """
    print("\n" + "=" * 60)
    print("Problem 3: Boundary Condition Identification")
    print("=" * 60)

    print(f"\n  Rod of length L = 1, heat equation u_t = alpha*u_xx")
    print(f"\n  Left end (x = 0): u(0, t) = 0")
    print(f"    Type: DIRICHLET (specifies the value of u)")
    print(f"    Physical: left end maintained at 0 degrees C (ice bath)")

    print(f"\n  Right end (x = L = 1):")
    print(f"    -kappa*u_x(L, t) = h*(u(L, t) - 20)")
    print(f"    Rearranging: -kappa*u_x(L, t) - h*u(L, t) = -20h")
    print(f"    Or: kappa*u_x(L, t) + h*u(L, t) = 20h")
    print(f"    Type: ROBIN (mixed: involves both u and u_x)")
    print(f"    Physical: right end loses heat to surroundings at 20 C")
    print(f"    via Newton's law of cooling with coefficient h")

    print(f"\n  Summary of BC types:")
    print(f"    Dirichlet: specifies u (value)")
    print(f"    Neumann: specifies u_x (flux)")
    print(f"    Robin: specifies a*u + b*u_x (linear combination)")
    print(f"    The Robin condition arises naturally in convective cooling")


# ============================================================
# Problem 4: Backward Heat Equation Ill-Posedness
# ============================================================
def exercise_4():
    """
    Explain why u_t = -alpha*u_xx is ill-posed.
    Show that high-frequency perturbations grow exponentially.
    """
    print("\n" + "=" * 60)
    print("Problem 4: Backward Heat Equation Ill-Posedness")
    print("=" * 60)

    print(f"\n  Backward heat equation: u_t = -alpha*u_xx")
    print(f"\n  Consider Fourier mode: u(x,t) = A*e^(sigma*t)*sin(k*x)")
    print(f"  u_t = sigma*A*e^(sigma*t)*sin(kx)")
    print(f"  u_xx = -k^2*A*e^(sigma*t)*sin(kx)")
    print(f"\n  Substituting: sigma = -alpha*(-k^2) = alpha*k^2")
    print(f"\n  Since alpha > 0 and k^2 > 0: sigma = alpha*k^2 > 0")
    print(f"  Every Fourier mode GROWS exponentially!")
    print(f"  Moreover, the growth rate increases as k^2:")
    print(f"  high-frequency perturbations grow MUCH faster.")

    print(f"\n  This means:")
    print(f"  - Any tiny error in initial data gets amplified exponentially")
    print(f"  - Higher frequencies are amplified more strongly")
    print(f"  - The solution does not depend continuously on data")
    print(f"  - The problem is ILL-POSED (violates Hadamard's third condition)")

    # Demonstration: growth factors for different k
    alpha = 0.01
    print(f"\n  Growth factors e^(alpha*k^2*t) at t=1 (alpha={alpha}):")
    print(f"  {'k':>5s}  {'sigma = alpha*k^2':>18s}  {'Growth at t=1':>18s}")
    print(f"  {'-----':>5s}  {'------------------':>18s}  {'------------------':>18s}")
    for k in [1, 5, 10, 50, 100]:
        sigma = alpha * k**2
        growth = np.exp(sigma * 1.0)
        if growth < 1e100:
            print(f"  {k:>5d}  {sigma:>18.2f}  {growth:>18.4e}")
        else:
            print(f"  {k:>5d}  {sigma:>18.2f}  {'OVERFLOW':>18s}")

    print(f"\n  For comparison, forward heat equation (u_t = +alpha*u_xx):")
    print(f"  sigma = -alpha*k^2 < 0: ALL modes DECAY, problem is WELL-POSED")


# ============================================================
# Problem 5: CFL Violation and Implicit Method
# ============================================================
def exercise_5():
    """
    Forward Euler heat equation with r=0.6 (CFL violated).
    Then implement backward Euler (implicit, unconditionally stable).
    """
    print("\n" + "=" * 60)
    print("Problem 5: CFL Violation and Implicit Method")
    print("=" * 60)

    # Parameters
    alpha = 1.0
    L = 1.0
    Nx = 50
    dx = L / Nx
    dt_unstable = 0.6 * dx**2 / alpha  # r = 0.6 > 0.5 (unstable for explicit)
    r = alpha * dt_unstable / dx**2
    Nt = 200

    x = np.linspace(0, L, Nx + 1)
    u_init = np.sin(np.pi * x)

    print(f"\n  Heat equation: u_t = {alpha}*u_xx on [0, {L}]")
    print(f"  u(0,t) = u(1,t) = 0, u(x,0) = sin(pi*x)")
    print(f"  dx = {dx:.4f}, dt = {dt_unstable:.6f}")
    print(f"  r = alpha*dt/dx^2 = {r:.4f}")
    print(f"  CFL condition: r <= 0.5 => VIOLATED (r = {r:.4f})")

    # Forward Euler (explicit) -- should blow up
    u_explicit = u_init.copy()
    blew_up = False
    for n in range(Nt):
        u_new = u_explicit.copy()
        for i in range(1, Nx):
            u_new[i] = u_explicit[i] + r * (u_explicit[i+1] - 2*u_explicit[i] + u_explicit[i-1])
        u_new[0] = 0
        u_new[Nx] = 0
        u_explicit = u_new
        if np.max(np.abs(u_explicit)) > 1e10:
            print(f"\n  Forward Euler: BLEW UP at step {n+1}!")
            print(f"    max|u| = {np.max(np.abs(u_explicit)):.2e}")
            blew_up = True
            break

    if not blew_up:
        print(f"  Forward Euler: max|u| at final time = {np.max(np.abs(u_explicit)):.2e}")

    # Backward Euler (implicit) -- unconditionally stable
    # u^{n+1}_i - r*(u^{n+1}_{i+1} - 2*u^{n+1}_i + u^{n+1}_{i-1}) = u^n_i
    # This is a tridiagonal system: (I + r*A)*u^{n+1} = u^n
    # where A is the tridiagonal matrix [-1, 2, -1]/dx^2

    # Build tridiagonal matrix
    N_int = Nx - 1  # interior points
    diag_main = np.ones(N_int) * (1 + 2*r)
    diag_off = np.ones(N_int - 1) * (-r)

    # Thomas algorithm (tridiagonal solver)
    def solve_tridiag(a, b, c, d):
        """Solve tridiagonal system. a=lower, b=main, c=upper, d=rhs."""
        n = len(d)
        c_star = np.zeros(n)
        d_star = np.zeros(n)
        c_star[0] = c[0] / b[0]
        d_star[0] = d[0] / b[0]
        for i in range(1, n):
            m = b[i] - a[i-1] * c_star[i-1]
            c_star[i] = c[i] / m if i < n-1 else 0
            d_star[i] = (d[i] - a[i-1] * d_star[i-1]) / m
        x = np.zeros(n)
        x[-1] = d_star[-1]
        for i in range(n-2, -1, -1):
            x[i] = d_star[i] - c_star[i] * x[i+1]
        return x

    u_implicit = u_init.copy()
    for n in range(Nt):
        # RHS is the interior values
        rhs = u_implicit[1:Nx].copy()
        u_int = solve_tridiag(diag_off, diag_main, diag_off, rhs)
        u_implicit[1:Nx] = u_int

    # Exact solution: u(x,t) = sin(pi*x)*exp(-pi^2*alpha*t)
    t_final = Nt * dt_unstable
    u_exact = np.sin(np.pi * x) * np.exp(-np.pi**2 * alpha * t_final)

    max_err_implicit = np.max(np.abs(u_implicit - u_exact))

    print(f"\n  Backward Euler (implicit):")
    print(f"    Completed {Nt} steps with r = {r:.4f}")
    print(f"    max|u| at t = {t_final:.4f}: {np.max(u_implicit):.6f}")
    print(f"    max error vs exact: {max_err_implicit:.6e}")
    print(f"    STABLE! (implicit method is unconditionally stable)")

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Forward Euler at early time (before blowup)
    u_early = u_init.copy()
    for n in range(5):
        u_new = u_early.copy()
        for i in range(1, Nx):
            u_new[i] = u_early[i] + r * (u_early[i+1] - 2*u_early[i] + u_early[i-1])
        u_new[0] = 0
        u_new[Nx] = 0
        u_early = u_new

    axes[0].plot(x, u_init, 'b--', linewidth=1.5, label='Initial')
    axes[0].plot(x, u_early, 'r-', linewidth=2, label=f'Forward Euler (5 steps)')
    axes[0].set_title('Forward Euler: Oscillations Growing', fontsize=12)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('u')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, u_implicit, 'b-', linewidth=2, label='Backward Euler')
    axes[1].plot(x, u_exact, 'r--', linewidth=1.5, label='Exact')
    axes[1].set_title(f'Backward Euler at t={t_final:.3f}', fontsize=12)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('u')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(x, u_implicit - u_exact, 'b-', linewidth=2)
    axes[2].set_title('Backward Euler Error', fontsize=12)
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('Error')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f'CFL Violation (r={r:.2f}): Explicit Fails, Implicit Succeeds', fontsize=14)
    plt.tight_layout()
    plt.savefig('ex17_cfl_violation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  [Plot saved: ex17_cfl_violation.png]")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
    print("\n" + "=" * 60)
    print("All exercises for Lesson 17 completed.")
    print("=" * 60)
