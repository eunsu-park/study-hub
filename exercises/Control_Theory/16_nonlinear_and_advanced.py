"""
Exercises for Lesson 16: Nonlinear Control and Advanced Topics
Topic: Control_Theory
Solutions to practice problems from the lesson.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def exercise_1():
    """
    Exercise 1: Lyapunov Analysis of Van der Pol Oscillator
    x'' - mu*(1 - x^2)*x' + x = 0
    """
    mu = 1.0

    print(f"Van der Pol oscillator: x'' - mu*(1-x^2)*x' + x = 0, mu = {mu}")

    # Part 1: Equilibrium point
    print(f"\nPart 1: Equilibrium point")
    print(f"  Let x1 = x, x2 = x' = dx/dt")
    print(f"  dx1/dt = x2")
    print(f"  dx2/dt = mu*(1 - x1^2)*x2 - x1")
    print(f"  At equilibrium: x2 = 0 and mu*(1-x1^2)*0 - x1 = 0 => x1 = 0")
    print(f"  Equilibrium: (x1, x2) = (0, 0)")

    # Part 2: Linearization
    print(f"\nPart 2: Linearization around (0, 0)")
    print(f"  f1(x1, x2) = x2")
    print(f"  f2(x1, x2) = mu*(1-x1^2)*x2 - x1")
    print(f"  Jacobian:")
    print(f"  df1/dx1 = 0,            df1/dx2 = 1")
    print(f"  df2/dx1 = -2*mu*x1*x2 - 1,  df2/dx2 = mu*(1-x1^2)")
    print(f"  At (0,0):")
    print(f"  A = [[0, 1], [-1, mu]] = [[0, 1], [-1, {mu}]]")

    A_lin = np.array([[0, 1], [-1, mu]])
    eigenvalues = np.linalg.eigvals(A_lin)
    print(f"\n  Eigenvalues of linearized system: {np.round(eigenvalues, 4)}")
    print(f"  Real parts: {np.round(eigenvalues.real, 4)}")

    if all(e.real > 0 for e in eigenvalues):
        print(f"  Both eigenvalues have positive real parts => origin is UNSTABLE")
    elif all(e.real < 0 for e in eigenvalues):
        print(f"  Both eigenvalues have negative real parts => origin is stable")
    else:
        print(f"  Mixed signs => saddle point (unstable)")

    print(f"  For mu > 0: trace(A) = mu > 0 => at least one eigenvalue has Re > 0")
    print(f"  The equilibrium is an UNSTABLE SPIRAL (for 0 < mu < 2)")

    # Part 3: Try V = (x^2 + xdot^2) / 2
    print(f"\nPart 3: Lyapunov function V = (x1^2 + x2^2) / 2")
    print(f"  dV/dt = x1*dx1/dt + x2*dx2/dt")
    print(f"        = x1*x2 + x2*[mu*(1-x1^2)*x2 - x1]")
    print(f"        = x1*x2 + mu*(1-x1^2)*x2^2 - x1*x2")
    print(f"        = mu*(1-x1^2)*x2^2")
    print()
    print(f"  dV/dt = mu*(1-x1^2)*x2^2")
    print(f"  For |x1| < 1: dV/dt > 0 (V increasing => trajectories move AWAY from origin)")
    print(f"  For |x1| > 1: dV/dt < 0 (V decreasing => trajectories move TOWARD origin)")
    print(f"  For |x1| = 1: dV/dt = 0 (no conclusion)")
    print()
    print(f"  Conclusion: We CANNOT conclude stability with this V.")
    print(f"  In fact, dV/dt > 0 near the origin confirms the origin is unstable.")
    print(f"  The mixed sign of dV/dt suggests energy is gained near origin")
    print(f"  and lost far from origin => existence of a limit cycle.")

    # Part 4: Limit cycle
    print(f"\nPart 4: Van der Pol behavior for mu > 0")
    print(f"  The Van der Pol oscillator exhibits a STABLE LIMIT CYCLE.")
    print(f"  - Near the origin: energy increases (unstable equilibrium)")
    print(f"  - Far from origin: energy decreases (dissipation)")
    print(f"  - A unique periodic orbit exists where energy gain = energy loss")
    print(f"  - All trajectories (except the origin) converge to this limit cycle")
    print(f"  - This cannot happen in linear systems!")

    # Simulate
    def vdp(t, y):
        return [y[1], mu * (1 - y[0]**2) * y[1] - y[0]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Phase portrait
    for x0 in [(0.1, 0), (0.5, 0), (2, 0), (4, 0), (0, 3)]:
        sol = solve_ivp(vdp, [0, 30], x0, max_step=0.01, dense_output=True)
        ax1.plot(sol.y[0], sol.y[1], linewidth=1)

    ax1.set_xlabel('x')
    ax1.set_ylabel('dx/dt')
    ax1.set_title(f'Van der Pol Phase Portrait (mu={mu})')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Time domain
    sol = solve_ivp(vdp, [0, 30], [0.1, 0], max_step=0.01)
    ax2.plot(sol.t, sol.y[0], 'b-', linewidth=1)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('x(t)')
    ax2.set_title('Van der Pol: Buildup to Limit Cycle')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/exercises/Control_Theory/ex16_vanderpol.png',
                dpi=100)
    plt.close()
    print("  Phase portrait and time response saved to 'ex16_vanderpol.png'")


def exercise_2():
    """
    Exercise 2: Describing Function
    Relay nonlinearity (+/-1) with G(s) = 10/[s(s+1)(s+2)]
    """
    M = 1.0  # relay amplitude

    print(f"Relay nonlinearity: output = +/- {M}")
    print(f"Linear part: G(s) = 10/[s(s+1)(s+2)]")

    # Part 1: Describing function for relay
    print(f"\nPart 1: Describing function of ideal relay")
    print(f"  For ideal relay with amplitude M, the describing function is:")
    print(f"  N(a) = 4M/(pi*a)")
    print(f"  where a is the input sinusoid amplitude")
    print(f"  N(a) is real (no phase shift for symmetric memoryless nonlinearity)")

    # Part 2: -1/N(a)
    print(f"\nPart 2: -1/N(a)")
    print(f"  -1/N(a) = -pi*a/(4M) = -{np.pi/(4*M):.4f} * a")
    print(f"  This is a point on the NEGATIVE REAL AXIS")
    print(f"  As a varies from 0 to infinity:")
    print(f"    a -> 0: -1/N -> 0")
    print(f"    a -> inf: -1/N -> -infinity")
    print(f"  So -1/N(a) traces the negative real axis from 0 to -infinity")

    # Part 3: Find intersection with G(jw)
    print(f"\nPart 3: Intersection => limit cycle prediction")
    print(f"  Need: G(jw) = -1/N(a)")
    print(f"  Since -1/N(a) is on the negative real axis,")
    print(f"  we need Im[G(jw)] = 0 (real-axis crossing of Nyquist plot)")

    # G(jw) = 10/[jw(jw+1)(jw+2)]
    # Denominator: jw(jw+1)(jw+2) = jw[-w^2 + 3jw + 2]
    #            = jw(2-w^2) - 3w^2 = -3w^2 + jw(2-w^2)
    # G(jw) = 10 / [-3w^2 + jw(2-w^2)]
    #       = 10[-3w^2 - jw(2-w^2)] / [9w^4 + w^2(2-w^2)^2]

    # Im[G(jw)] = -10w(2-w^2) / [9w^4 + w^2(2-w^2)^2] = 0
    # => 2 - w^2 = 0 => w = sqrt(2)

    w_lc = np.sqrt(2)
    print(f"  Im[G(jw)] = 0 when 2 - w^2 = 0 => w = sqrt(2) = {w_lc:.4f} rad/s")

    # Real part at w = sqrt(2):
    # Re[G] = 10(-3*2) / [9*4 + 2*0] = -60/36 = -5/3
    re_G = -10 * 3 * 2 / (9 * 4)
    print(f"  Re[G(j*sqrt(2))] = 10*(-6) / 36 = {re_G:.4f}")

    # Limit cycle: -1/N(a) = re_G
    # -pi*a/(4M) = re_G
    # a = -4M*re_G / pi
    a_lc = -4 * M * re_G / np.pi
    print(f"\n  Limit cycle prediction:")
    print(f"  -1/N(a) = {re_G:.4f}")
    print(f"  -pi*a/(4*{M}) = {re_G:.4f}")
    print(f"  a = {a_lc:.4f}")
    print(f"  Predicted limit cycle amplitude: a = {a_lc:.4f}")
    print(f"  Predicted limit cycle frequency: w = {w_lc:.4f} rad/s")
    print(f"  Predicted period: T = 2*pi/w = {2*np.pi/w_lc:.4f} s")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # G(jw) Nyquist plot
    w = np.logspace(-2, 2, 10000)
    G_vals = 10 / (1j * w * (1j * w + 1) * (1j * w + 2))

    ax.plot(G_vals.real, G_vals.imag, 'b-', linewidth=2, label='G(jw)')
    ax.plot(G_vals.real, -G_vals.imag, 'b--', linewidth=1, alpha=0.5)

    # -1/N(a) locus
    a_vals = np.linspace(0.01, 10, 100)
    neg_inv_N = -np.pi * a_vals / (4 * M)
    ax.plot(neg_inv_N, np.zeros_like(neg_inv_N), 'r-', linewidth=2, label='-1/N(a)')

    # Mark intersection
    ax.plot(re_G, 0, 'ko', markersize=10, label=f'Limit cycle (a={a_lc:.2f}, w={w_lc:.2f})')

    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_title('Describing Function Analysis: Relay + G(s)=10/[s(s+1)(s+2)]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-5, 1])
    ax.set_ylim([-5, 5])

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/exercises/Control_Theory/'
                'ex16_describing_function.png', dpi=100)
    plt.close()
    print("  Describing function plot saved to 'ex16_describing_function.png'")


def exercise_3():
    """
    Exercise 3: MPC Concept
    Discrete double integrator with constraints |u| <= 1
    """
    T = 0.1

    print(f"Discrete-time double integrator, T = {T} s")

    A = np.array([[1, T], [0, 1]])
    B = np.array([[T**2/2], [T]])
    Q = np.eye(2)
    R = np.array([[0.1]])
    N_horizon = 10
    u_max = 1.0

    print(f"  A = \n{A}")
    print(f"  B = {B.flatten()}")
    print(f"  Q = I, R = {R[0,0]}, N = {N_horizon}")
    print(f"  Constraint: |u| <= {u_max}")

    # Part 1: MPC optimization problem formulation
    print(f"\nPart 1: MPC optimization problem")
    print(f"  At each step k, solve:")
    print(f"  min sum_{{i=0}}^{{N-1}} [x[k+i]^T Q x[k+i] + u[k+i]^T R u[k+i]] + x[k+N]^T Pf x[k+N]")
    print(f"  subject to:")
    print(f"    x[k+i+1] = A*x[k+i] + B*u[k+i]  for i = 0,...,N-1")
    print(f"    -1 <= u[k+i] <= 1                  for i = 0,...,N-1")
    print(f"    x[k] = x_current  (initial condition)")
    print()
    print(f"  This is a Quadratic Program (QP):")
    print(f"  - Decision variables: u[k], u[k+1], ..., u[k+N-1] ({N_horizon} variables)")
    print(f"  - Quadratic cost (convex)")
    print(f"  - Linear equality constraints (dynamics)")
    print(f"  - Linear inequality constraints (input bounds)")
    print(f"  => Convex QP, can be solved efficiently")

    # Part 2: Unconstrained vs constrained
    print(f"\nPart 2: Unconstrained (LQR) vs constrained")
    print(f"  Without constraints:")
    print(f"  - This is the finite-horizon LQR problem")
    print(f"  - Optimal gain is time-varying: u[k+i] = -K_i * x[k+i]")
    print(f"  - Solved offline via backward Riccati recursion")
    print(f"  - Control inputs can be arbitrarily large")
    print()
    print(f"  With |u| <= {u_max}:")
    print(f"  - The LQR solution may violate constraints (e.g., large initial error)")
    print(f"  - MPC clips control inputs to feasible range")
    print(f"  - The solution is no longer a simple linear gain")
    print(f"  - The optimizer finds the BEST feasible trajectory")
    print(f"  - Constraint handling is the primary advantage of MPC over LQR")

    # Part 3: Receding horizon benefits
    print(f"\nPart 3: Why receding horizon helps with disturbances")
    print(f"  Without receding horizon (open-loop optimal):")
    print(f"  - Compute u[0], ..., u[N-1] at t=0")
    print(f"  - Apply the entire sequence")
    print(f"  - If disturbances occur, the plan is suboptimal or infeasible")
    print()
    print(f"  With receding horizon:")
    print(f"  - At each step k, re-solve with current state x[k]")
    print(f"  - Only apply u[k], discard u[k+1],...,u[k+N-1]")
    print(f"  - At step k+1, solve again from x[k+1] (which includes disturbance effects)")
    print(f"  - This creates implicit feedback: the re-optimization reacts to disturbances")
    print(f"  - Effectively combines feedforward (model prediction) with feedback (re-solving)")
    print()
    print(f"  Key insight: MPC achieves feedback through repeated optimization,")
    print(f"  not through an explicit feedback law like LQR.")

    # Simple simulation demonstrating constrained vs unconstrained
    print(f"\n  Simulation: initial state x0 = [5, 0] (position=5, velocity=0)")

    x0 = np.array([5.0, 0.0])
    N_sim = 100

    # Unconstrained LQR (infinite horizon approximation)
    from scipy import linalg
    P_dare = linalg.solve_discrete_are(A, B, Q, R)
    K_lqr = np.linalg.inv(R + B.T @ P_dare @ B) @ B.T @ P_dare @ A

    x_unc = np.zeros((N_sim + 1, 2))
    u_unc = np.zeros(N_sim)
    x_unc[0] = x0

    x_con = np.zeros((N_sim + 1, 2))
    u_con = np.zeros(N_sim)
    x_con[0] = x0

    for k in range(N_sim):
        # Unconstrained
        u_unc[k] = -(K_lqr @ x_unc[k])[0]
        x_unc[k+1] = A @ x_unc[k] + B.flatten() * u_unc[k]

        # Constrained (simple clipping as MPC approximation)
        u_desired = -(K_lqr @ x_con[k])[0]
        u_con[k] = np.clip(u_desired, -u_max, u_max)
        x_con[k+1] = A @ x_con[k] + B.flatten() * u_con[k]

    t_sim = np.arange(N_sim + 1) * T

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(t_sim, x_unc[:, 0], 'b-', linewidth=2, label='Unconstrained (LQR)')
    ax1.plot(t_sim, x_con[:, 0], 'r--', linewidth=2, label='Constrained (|u|<=1)')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_ylabel('Position x1')
    ax1.set_title('MPC vs LQR: Double Integrator Regulation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(np.arange(N_sim) * T, u_unc, 'b-', linewidth=2, label='Unconstrained')
    ax2.plot(np.arange(N_sim) * T, u_con, 'r--', linewidth=2, label='Constrained')
    ax2.axhline(y=u_max, color='k', linestyle=':', alpha=0.5)
    ax2.axhline(y=-u_max, color='k', linestyle=':', alpha=0.5)
    ax2.set_ylabel('Control input u')
    ax2.set_xlabel('Time (s)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/exercises/Control_Theory/ex16_mpc.png',
                dpi=100)
    plt.close()
    print("  MPC vs LQR comparison saved to 'ex16_mpc.png'")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Lyapunov Analysis ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Describing Function ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: MPC Concept ===")
    print("=" * 60)
    exercise_3()

    print("\n\nAll exercises completed!")
