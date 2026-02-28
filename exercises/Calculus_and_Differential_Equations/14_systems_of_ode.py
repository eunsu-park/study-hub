"""
Exercise Solutions: Lesson 14 - Systems of Ordinary Differential Equations
Calculus and Differential Equations

Topics covered:
- Third-order to first-order conversion
- 2x2 system eigenvalue solution and phase portrait
- Competing species model (nonlinear)
- Damped pendulum simulation
- Matrix exponential verification
"""

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.linalg import expm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# Problem 1: Third-Order to First-Order Conversion
# ============================================================
def exercise_1():
    """
    Convert y''' - 6y'' + 11y' - 6y = 0 to a first-order system.
    Find eigenvalues and general solution.
    """
    print("=" * 60)
    print("Problem 1: Third-Order to First-Order Conversion")
    print("=" * 60)

    # Let x1 = y, x2 = y', x3 = y''
    # x1' = x2
    # x2' = x3
    # x3' = 6y - 11y' + 6y'' = 6*x1 - 11*x2 + 6*x3

    print(f"\n  y''' - 6y'' + 11y' - 6y = 0")
    print(f"\n  Let x1 = y, x2 = y', x3 = y''")
    print(f"  System:")
    print(f"    x1' = x2")
    print(f"    x2' = x3")
    print(f"    x3' = 6*x1 - 11*x2 + 6*x3")

    A = sp.Matrix([
        [0, 1, 0],
        [0, 0, 1],
        [6, -11, 6]
    ])

    print(f"\n  Matrix A =")
    sp.pprint(A)

    eigenvals = A.eigenvals()
    print(f"\n  Eigenvalues: {eigenvals}")
    print(f"  r = 1, 2, 3 (all distinct)")

    # Characteristic polynomial: r^3 - 6r^2 + 11r - 6 = (r-1)(r-2)(r-3) = 0
    print(f"\n  Characteristic equation: r^3 - 6r^2 + 11r - 6 = 0")
    print(f"  = (r-1)(r-2)(r-3) = 0")
    print(f"\n  General solution:")
    print(f"  y(t) = C1*e^t + C2*e^(2t) + C3*e^(3t)")

    # SymPy verification
    t = sp.Symbol('t')
    y = sp.Function('y')
    ode = y(t).diff(t, 3) - 6*y(t).diff(t, 2) + 11*y(t).diff(t) - 6*y(t)
    sol = sp.dsolve(ode, y(t))
    print(f"\n  SymPy: {sol}")


# ============================================================
# Problem 2: 2x2 System
# ============================================================
def exercise_2():
    """
    X' = [[3, -2], [4, -1]] X, X(0) = (1, 1)^T.
    Classify equilibrium, sketch phase portrait.
    """
    print("\n" + "=" * 60)
    print("Problem 2: 2x2 System")
    print("=" * 60)

    A = sp.Matrix([[3, -2], [4, -1]])
    eigendata = A.eigenvects()

    print(f"\n  A = [[3, -2], [4, -1]]")
    print(f"  Eigenvalues and eigenvectors:")
    for eigenval, mult, vecs in eigendata:
        print(f"    lambda = {eigenval} (multiplicity {mult}), v = {vecs[0].T}")

    # Eigenvalues: trace=2, det=3+8=5 ... wait
    # det = 3*(-1) - (-2)*4 = -3 + 8 = 5
    # trace = 3 + (-1) = 2
    # lambda^2 - 2*lambda + 5 = 0 => lambda = 1 +/- 2i
    print(f"\n  Trace = 2, Det = 5")
    print(f"  Char eq: lambda^2 - 2*lambda + 5 = 0")
    print(f"  lambda = 1 +/- 2i (complex with positive real part)")
    print(f"  Classification: UNSTABLE SPIRAL (spiral source)")

    # General solution: X(t) = e^t * (C1*[cos(2t)*v_r - sin(2t)*v_i] + C2*[sin(2t)*v_r + cos(2t)*v_i])
    # For lambda = 1 + 2i, eigenvector: (A - (1+2i)I)v = 0
    # [[2-2i, -2], [4, -2-2i]] v = 0
    # v = (1, (2-2i)/2) = (1, 1-i) = (1, 1) + i*(0, -1)
    # v_r = (1, 1), v_i = (0, -1)

    print(f"\n  For lambda = 1+2i, eigenvector v = (1, 1-i)")
    print(f"  v_r = (1, 1), v_i = (0, -1)")
    print(f"\n  General solution:")
    print(f"  X(t) = e^t * [C1*(cos(2t)*(1,1) - sin(2t)*(0,-1))")
    print(f"              + C2*(sin(2t)*(1,1) + cos(2t)*(0,-1))]")

    # Apply IC: X(0) = (1, 1)
    # C1*(1,1) + C2*(0,-1) = (1, 1)
    # C1 = 1, C1 - C2 = 1 => C2 = 0
    print(f"\n  Apply X(0) = (1, 1):")
    print(f"  C1 = 1, C2 = 0")
    print(f"  X(t) = e^t * (cos(2t), cos(2t) + sin(2t))")

    # Numerical phase portrait
    A_num = np.array([[3, -2], [4, -1]], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot several trajectories
    for x0, y0 in [(1, 1), (0.5, 0), (0, 0.5), (-0.5, 0), (0, -0.5), (0.5, 0.5)]:
        sol = solve_ivp(lambda t, s: A_num @ s, [0, 3], [x0, y0],
                        t_eval=np.linspace(0, 3, 500), method='RK45')
        ax.plot(sol.y[0], sol.y[1], 'b-', linewidth=0.8, alpha=0.7)
        ax.plot(x0, y0, 'go', markersize=5)

    # Highlight the specific IC trajectory
    sol_ic = solve_ivp(lambda t, s: A_num @ s, [0, 2.5], [1, 1],
                       t_eval=np.linspace(0, 2.5, 500), method='RK45')
    ax.plot(sol_ic.y[0], sol_ic.y[1], 'r-', linewidth=2, label='X(0) = (1, 1)')
    ax.plot(1, 1, 'ro', markersize=10)

    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title('Phase Portrait: Unstable Spiral', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    plt.tight_layout()
    plt.savefig('ex14_phase_portrait.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  [Plot saved: ex14_phase_portrait.png]")


# ============================================================
# Problem 3: Competing Species
# ============================================================
def exercise_3():
    """
    dx/dt = x(3 - x - 2y), dy/dt = y(2 - y - x).
    (a) Equilibria, (b) Linearize and classify, (c) Phase portrait.
    """
    print("\n" + "=" * 60)
    print("Problem 3: Competing Species")
    print("=" * 60)

    x, y = sp.symbols('x y')

    f = x * (3 - x - 2*y)
    g = y * (2 - y - x)

    # (a) Equilibrium points: f = 0, g = 0
    equil = sp.solve([f, g], [x, y])
    print(f"\n(a) Equilibrium points:")
    for pt in equil:
        print(f"    ({pt[0]}, {pt[1]})")

    # (b) Jacobian
    J = sp.Matrix([[sp.diff(f, x), sp.diff(f, y)],
                    [sp.diff(g, x), sp.diff(g, y)]])
    print(f"\n(b) Jacobian:")
    sp.pprint(J)

    print(f"\n  Classification at each equilibrium:")
    for pt in equil:
        J_at = J.subs([(x, pt[0]), (y, pt[1])])
        eigenvals = J_at.eigenvals()
        trace = J_at.trace()
        det = J_at.det()

        if all(sp.re(ev) < 0 for ev in eigenvals):
            stability = "STABLE"
        elif all(sp.re(ev) > 0 for ev in eigenvals):
            stability = "UNSTABLE"
        elif any(sp.re(ev) > 0 for ev in eigenvals) and any(sp.re(ev) < 0 for ev in eigenvals):
            stability = "SADDLE"
        else:
            stability = "MARGINAL"

        print(f"    ({pt[0]}, {pt[1]}): eigenvalues = {list(eigenvals.keys())}, {stability}")

    # (c) Phase portrait
    def system(t, state):
        x_v, y_v = state
        return [x_v*(3 - x_v - 2*y_v), y_v*(2 - y_v - x_v)]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Trajectories from various ICs
    ics = [(0.5, 0.5), (2.5, 0.5), (0.5, 1.5), (2, 1.5), (0.1, 0.1),
           (3, 0.1), (0.1, 2), (1.5, 0.5), (0.5, 1), (2.5, 1.5)]
    for x0, y0 in ics:
        sol = solve_ivp(system, [0, 20], [x0, y0],
                        t_eval=np.linspace(0, 20, 2000), method='RK45')
        ax.plot(sol.y[0], sol.y[1], 'b-', linewidth=0.8, alpha=0.6)
        ax.plot(x0, y0, 'go', markersize=4)
        # Arrow at midpoint
        mid = len(sol.t) // 4
        if mid > 0:
            dx = sol.y[0, mid+1] - sol.y[0, mid]
            dy = sol.y[1, mid+1] - sol.y[1, mid]
            ax.annotate('', xy=(sol.y[0, mid+1], sol.y[1, mid+1]),
                        xytext=(sol.y[0, mid], sol.y[1, mid]),
                        arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

    # Mark equilibria
    for pt in equil:
        xp, yp = float(pt[0]), float(pt[1])
        ax.plot(xp, yp, 'rs', markersize=12)
        ax.annotate(f'({xp}, {yp})', xy=(xp, yp), xytext=(xp+0.15, yp+0.15), fontsize=10, color='red')

    # Nullclines
    x_vals = np.linspace(0, 3.5, 200)
    ax.plot(x_vals, (3 - x_vals)/2, 'r--', linewidth=1.5, alpha=0.5, label='x-nullcline: 3-x-2y=0')
    ax.plot(x_vals, 2 - x_vals, 'g--', linewidth=1.5, alpha=0.5, label='y-nullcline: 2-y-x=0')

    ax.set_xlabel('x (species 1)', fontsize=12)
    ax.set_ylabel('y (species 2)', fontsize=12)
    ax.set_title('Competing Species Phase Portrait', fontsize=14)
    ax.set_xlim(0, 3.5)
    ax.set_ylim(0, 2.5)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex14_competing_species.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n(c) [Plot saved: ex14_competing_species.png]")
    print(f"    Interpretation: Species 1 dominates (x -> 3, y -> 0)")
    print(f"    The equilibrium (3, 0) is stable; (0, 2) is also stable.")
    print(f"    The coexistence point (1, 1) is a saddle -- unstable.")
    print(f"    Which species survives depends on initial conditions.")


# ============================================================
# Problem 4: Damped Pendulum
# ============================================================
def exercise_4():
    """
    omega_0 = 3, beta = 0.5, theta(0) = pi - 0.1, theta'(0) = 0.
    Simulate and plot.
    """
    print("\n" + "=" * 60)
    print("Problem 4: Damped Pendulum")
    print("=" * 60)

    omega_0 = 3.0
    beta = 0.5
    theta_0 = np.pi - 0.1
    dtheta_0 = 0.0

    # theta'' + 2*beta*theta' + omega_0^2*sin(theta) = 0
    def pendulum(t, state):
        theta, dtheta = state
        return [dtheta, -2*beta*dtheta - omega_0**2 * np.sin(theta)]

    # (a) The pendulum starts near the unstable equilibrium (theta=pi).
    # It will fall toward the stable equilibrium theta=0 (or 2*pi).
    print(f"\n  theta'' + 2*{beta}*theta' + {omega_0}^2*sin(theta) = 0")
    print(f"  theta(0) = pi - 0.1 = {theta_0:.4f} rad")
    print(f"  theta'(0) = 0")
    print(f"\n(a) Starting near theta=pi (unstable inverted position)")
    print(f"    With damping, the pendulum will settle at theta=0 (or 2*pi mod 2*pi)")

    # (b) Simulate
    t_span = [0, 30]
    t_eval = np.linspace(0, 30, 3000)
    sol = solve_ivp(pendulum, t_span, [theta_0, dtheta_0], t_eval=t_eval, method='RK45')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # theta(t)
    ax1.plot(sol.t, sol.y[0], 'b-', linewidth=2)
    ax1.axhline(y=0, color='g', linestyle='--', alpha=0.5, label=r'$\theta = 0$ (stable)')
    ax1.axhline(y=np.pi, color='r', linestyle='--', alpha=0.5, label=r'$\theta = \pi$ (unstable)')
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel(r'$\theta$ (rad)', fontsize=12)
    ax1.set_title(r'Damped Pendulum: $\theta(t)$', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Phase portrait
    ax2.plot(sol.y[0], sol.y[1], 'b-', linewidth=1.5)
    ax2.plot(theta_0, dtheta_0, 'ro', markersize=10, label='Start')
    ax2.plot(sol.y[0, -1], sol.y[1, -1], 'gs', markersize=10, label='End')
    ax2.set_xlabel(r'$\theta$', fontsize=12)
    ax2.set_ylabel(r'$\dot{\theta}$', fontsize=12)
    ax2.set_title('Phase Portrait', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex14_pendulum.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n(b) [Plot saved: ex14_pendulum.png]")
    print(f"    Final theta ~ {sol.y[0, -1]:.4f} rad")

    # (c) No damping
    def pendulum_nodamp(t, state):
        theta, dtheta = state
        return [dtheta, -omega_0**2 * np.sin(theta)]

    sol_nd = solve_ivp(pendulum_nodamp, t_span, [theta_0, dtheta_0],
                       t_eval=t_eval, method='RK45')
    print(f"\n(c) Without damping (beta=0):")
    print(f"    The pendulum oscillates indefinitely without settling.")
    print(f"    Energy is conserved: the trajectory is a closed orbit in phase space.")


# ============================================================
# Problem 5: Matrix Exponential
# ============================================================
def exercise_5():
    """
    A = [[0, 1], [-4, 0]]. Compute e^{At} at various t.
    Verify closed orbits, find the period.
    """
    print("\n" + "=" * 60)
    print("Problem 5: Matrix Exponential")
    print("=" * 60)

    A = np.array([[0, 1], [-4, 0]], dtype=float)

    # Eigenvalues: lambda^2 + 4 = 0 => lambda = +/- 2i
    # Pure imaginary => center (closed orbits)
    # Period T = 2*pi/omega = 2*pi/2 = pi
    print(f"\n  A = [[0, 1], [-4, 0]]")
    print(f"  Eigenvalues: lambda^2 + 4 = 0 => lambda = +/- 2i")
    print(f"  Pure imaginary eigenvalues => CENTER (closed orbits)")
    print(f"  Angular frequency omega = 2, period T = 2*pi/2 = pi = {np.pi:.6f}")

    # Compute e^{At} at various times
    t_values = [0, np.pi/4, np.pi/2, np.pi]
    print(f"\n  e^{{At}} at various times:")
    for t_val in t_values:
        eAt = expm(A * t_val)
        print(f"\n  t = {t_val:.4f} ({t_val/np.pi:.2f}*pi):")
        print(f"    e^{{At}} = [[{eAt[0,0]:>8.4f}, {eAt[0,1]:>8.4f}],")
        print(f"              [{eAt[1,0]:>8.4f}, {eAt[1,1]:>8.4f}]]")

    # At t=0: should be identity
    # At t=pi: should return to initial state
    eA_0 = expm(A * 0)
    eA_pi = expm(A * np.pi)
    print(f"\n  Verification:")
    print(f"    e^{{A*0}} = I? {np.allclose(eA_0, np.eye(2))}")
    print(f"    e^{{A*pi}} = -I? {np.allclose(eA_pi, -np.eye(2), atol=1e-10)}")
    print(f"    (At t=pi, the state reverses sign)")
    eA_2pi = expm(A * 2*np.pi)
    print(f"    e^{{A*2pi}} = I? {np.allclose(eA_2pi, np.eye(2), atol=1e-10)}")
    print(f"    Full period is T = 2*pi/2 = pi... Actually let's check:")

    # The period: e^{AT} = I requires T such that the eigenvalues e^{lambda*T} = 1
    # lambda*T = 2i*T = 2*pi*n*i => T = n*pi. Smallest: T = pi
    print(f"\n  Eigenvalue check: e^{{2i*T}} = 1 requires 2T = 2*pi*n")
    print(f"  Smallest T = pi")

    # Plot orbits
    fig, ax = plt.subplots(figsize=(8, 8))
    t_fine = np.linspace(0, 2*np.pi, 500)

    for x0 in [0.5, 1.0, 1.5, 2.0]:
        state0 = np.array([x0, 0])
        traj = np.array([expm(A*t) @ state0 for t in t_fine])
        ax.plot(traj[:, 0], traj[:, 1], '-', linewidth=1.5)

    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title('Center: Closed Orbits (Period = $\\pi$)', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex14_matrix_exp.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  [Plot saved: ex14_matrix_exp.png]")


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
    print("All exercises for Lesson 14 completed.")
    print("=" * 60)
