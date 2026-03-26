"""
Exercises for Lesson 17: Calculus of Variations
Topic: Mathematical_Methods
Solutions to practice problems from the lesson.
"""

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp, quad
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def exercise_1():
    """
    Problem 1: Find Euler-Lagrange equations for three functionals.
    (a) J[y] = int_0^1 (y'^2 + 2yy') dx, y(0)=0, y(1)=1
    (b) J[y] = int_0^pi (y'^2 - y^2) dx, y(0)=0, y(pi)=0
    (c) J[y] = int_1^2 sqrt(1 + y'^2)/x dx
    """
    print("=" * 60)
    print("Problem 1: Euler-Lagrange Equations")
    print("=" * 60)

    x = sp.Symbol('x')
    y = sp.Function('y')(x)
    yp = y.diff(x)

    # (a) F = y'^2 + 2yy'
    print("\n(a) F = y'^2 + 2*y*y'")
    F_a = yp**2 + 2 * y * yp
    Fy = sp.diff(F_a, y)
    Fyp = sp.diff(F_a, yp)
    EL_a = Fy - sp.diff(Fyp, x)
    print(f"  dF/dy = {sp.simplify(Fy)}")
    print(f"  dF/dy' = {sp.simplify(Fyp)}")
    print(f"  E-L: dF/dy - d/dx(dF/dy') = 0")
    print(f"  => 2*y' - d/dx(2*y' + 2*y) = 0")
    print(f"  => 2*y' - 2*y'' - 2*y' = 0")
    print(f"  => y'' = 0  =>  y = Ax + B")
    print(f"  With y(0)=0, y(1)=1: y = x")

    # (b) F = y'^2 - y^2
    print("\n(b) F = y'^2 - y^2")
    print(f"  dF/dy = -2*y")
    print(f"  dF/dy' = 2*y'")
    print(f"  E-L: -2*y - 2*y'' = 0  =>  y'' + y = 0")
    print(f"  General solution: y = A*sin(x) + B*cos(x)")
    print(f"  y(0) = 0 => B = 0")
    print(f"  y(pi) = 0 => A*sin(pi) = 0  (satisfied for any A)")
    print(f"  => y_n = sin(nx) are extremals (eigenvalue problem)")

    # (c) F = sqrt(1 + y'^2)/x
    print("\n(c) F = sqrt(1 + y'^2)/x")
    print(f"  dF/dy = 0  (F independent of y)")
    print(f"  => dF/dy' = const => y'/(x*sqrt(1+y'^2)) = C")
    print(f"  This gives a catenary-like curve")

    # Verify (a) numerically
    x_num = np.linspace(0, 1, 100)
    y_exact = x_num  # y = x
    J_exact = np.trapz((1.0)**2 + 2 * x_num * 1.0, x_num)  # y'=1

    # Compare with a trial function y = x + epsilon*sin(pi*x)
    eps = 0.1
    y_trial = x_num + eps * np.sin(np.pi * x_num)
    yp_trial = 1.0 + eps * np.pi * np.cos(np.pi * x_num)
    J_trial = np.trapz(yp_trial**2 + 2 * y_trial * yp_trial, x_num)

    print(f"\n  Verification (a): J[y=x] = {J_exact:.6f}")
    print(f"  J[y=x+0.1*sin(pi*x)] = {J_trial:.6f}")
    print(f"  y=x is {'extremal' if abs(J_exact) <= abs(J_trial) + 0.01 else 'not extremal'}")


def exercise_2():
    """
    Problem 2: Use Beltrami identity for J[y] = int_0^1 (y'^2 + y^2) dx,
    y(0)=0, y(1)=1.
    """
    print("\n" + "=" * 60)
    print("Problem 2: Beltrami Identity Application")
    print("=" * 60)

    print("\nJ[y] = int_0^1 (y'^2 + y^2) dx, y(0)=0, y(1)=1")
    print("\nF = y'^2 + y^2")
    print("Since F depends explicitly on y (not just x and y'),")
    print("we use the standard E-L equation:")
    print("  dF/dy - d/dx(dF/dy') = 0")
    print("  2y - 2y'' = 0  =>  y'' - y = 0")
    print("\nGeneral solution: y = A*sinh(x) + B*cosh(x)")
    print("  y(0) = 0 => B = 0")
    print("  y(1) = 1 => A = 1/sinh(1)")

    # Exact solution
    x = np.linspace(0, 1, 500)
    A = 1.0 / np.sinh(1)
    y_exact = A * np.sinh(x)

    print(f"\ny(x) = sinh(x)/sinh(1)")
    print(f"  A = 1/sinh(1) = {A:.6f}")
    print(f"  y(0.5) = {A * np.sinh(0.5):.6f}")

    # Compute functional value
    yp_exact = A * np.cosh(x)
    J_exact = np.trapz(yp_exact**2 + y_exact**2, x)
    print(f"  J[y*] = {J_exact:.6f}")

    # Compare with linear trial y = x
    y_trial = x
    yp_trial = np.ones_like(x)
    J_trial = np.trapz(yp_trial**2 + y_trial**2, x)
    print(f"  J[y=x] = {J_trial:.6f}")
    print(f"  J[y*] < J[y=x]: {J_exact < J_trial}")


def exercise_3():
    """
    Problem 3: Isoperimetric problem - maximize area under curve with
    fixed arc length L connecting (0,0) and (a,0).
    """
    print("\n" + "=" * 60)
    print("Problem 3: Isoperimetric Problem")
    print("=" * 60)

    print("\nFind y(x) connecting (0,0) to (a,0) with arc length L")
    print("that maximizes A = int_0^a y dx")
    print("\nConstrained functional with Lagrange multiplier lambda:")
    print("  J[y] = int_0^a [y + lambda*sqrt(1+y'^2)] dx")
    print("\nE-L: 1 - lambda*d/dx[y'/sqrt(1+y'^2)] = 0")
    print("=> lambda * y''/((1+y'^2)^(3/2)) = 1")
    print("=> (1+y'^2)^(3/2) / y'' = lambda")
    print("\nSolution: circular arc!  (x-a/2)^2 + (y-y0)^2 = R^2")

    # Numerical example: a = 2, L = pi (semicircle)
    a = 2.0
    L_arc = np.pi  # arc length
    R = L_arc / np.pi  # For semicircle: L = pi*R, so R = 1

    # Parametric semicircle centered at (a/2, 0) with radius R
    theta = np.linspace(0, np.pi, 500)
    x_circle = a / 2 + R * np.cos(np.pi - theta)  # from 0 to a
    y_circle = R * np.sin(theta)

    area_circle = np.trapz(y_circle, x_circle)
    arc_len = np.sum(np.sqrt(np.diff(x_circle)**2 + np.diff(y_circle)**2))

    print(f"\nNumerical example: a = {a}, L = pi")
    print(f"  Semicircle radius: R = {R:.4f}")
    print(f"  Area = {area_circle:.6f}")
    print(f"  Arc length = {arc_len:.6f} (target: {L_arc:.6f})")

    # Compare with triangle (non-optimal)
    h_tri = np.sqrt((L_arc / 2)**2 - (a / 2)**2) if L_arc / 2 > a / 2 else 0
    area_tri = 0.5 * a * h_tri
    print(f"\n  Triangular shape: area = {area_tri:.6f}")
    print(f"  Circular arc gives {area_circle / area_tri:.2f}x more area")


def exercise_4():
    """
    Problem 4: Geodesics on a sphere are great circles.
    """
    print("\n" + "=" * 60)
    print("Problem 4: Geodesics on a Sphere")
    print("=" * 60)

    print("\nMetric on sphere: ds^2 = R^2(dtheta^2 + sin^2(theta)*dphi^2)")
    print("\nArc length: S = R*int sqrt(theta'^2 + sin^2(theta)) dphi")
    print("where theta' = dtheta/dphi")
    print("\nF = sqrt(theta'^2 + sin^2(theta))")
    print("\nSince F is independent of phi, use Beltrami identity:")
    print("  F - theta'*(dF/dtheta') = C")
    print("  sqrt(theta'^2 + sin^2(theta)) - theta'^2/sqrt(theta'^2 + sin^2(theta)) = C")
    print("  sin^2(theta)/sqrt(theta'^2 + sin^2(theta)) = C")
    print("\nThis simplifies to: sin(theta)*dphi/ds = const")
    print("=> The path maintains constant angle with meridians")
    print("=> This is the equation of a great circle!")

    # Numerical verification: geodesic on unit sphere
    R = 1.0
    # Great circle from (theta=pi/4, phi=0) to (theta=pi/4, phi=pi)
    phi = np.linspace(0, np.pi, 200)

    # For a great circle through these points on the equator:
    # theta(phi) = pi/2 is the equator (trivial great circle)
    # A tilted great circle: cot(theta) = A*cos(phi - phi0)
    A_gc = 1.0  # tilt parameter
    theta_gc = np.arctan(1.0 / (A_gc * np.cos(phi)))
    # Only keep valid values
    mask = (theta_gc > 0) & (theta_gc < np.pi)

    # Arc length of equator (great circle)
    theta_eq = np.pi / 2 * np.ones_like(phi)
    ds_eq = R * np.sqrt(0 + np.sin(theta_eq)**2)
    S_eq = np.trapz(ds_eq, phi)

    print(f"\nNumerical: Equatorial great circle (phi: 0 to pi)")
    print(f"  Arc length = {S_eq:.6f} (expected: pi = {np.pi:.6f})")

    # Compare with non-geodesic path (constant latitude)
    theta_lat = np.pi / 3  # 60 degrees
    ds_lat = R * np.sqrt(0 + np.sin(theta_lat)**2)
    S_lat = np.trapz(ds_lat * np.ones_like(phi), phi)
    print(f"\n  Constant latitude theta=pi/3: arc length = {S_lat:.6f}")
    print(f"  Great circle through same endpoints would be shorter")


def exercise_5():
    """
    Problem 5: Brachistochrone - cycloid from (0,0) to (1,1).
    Compare travel time with straight line.
    """
    print("\n" + "=" * 60)
    print("Problem 5: Brachistochrone Problem")
    print("=" * 60)

    print("\nMinimize time T = int_0^1 sqrt((1+y'^2)/(2gy)) dx")
    print("Solution: cycloid x = R(t - sin(t)), y = R(1 - cos(t))")

    # Find R and t_f such that cycloid passes through (1, 1)
    # x = R(t - sin(t)) = 1, y = R(1 - cos(t)) = 1
    from scipy.optimize import fsolve

    def cycloid_eqs(params):
        R, t_f = params
        return [R * (t_f - np.sin(t_f)) - 1.0,
                R * (1 - np.cos(t_f)) - 1.0]

    R, t_f = fsolve(cycloid_eqs, [0.6, 2.5])
    print(f"\nCycloid parameters: R = {R:.6f}, t_f = {t_f:.6f}")

    # Parametric cycloid
    t_param = np.linspace(0, t_f, 1000)
    x_cyc = R * (t_param - np.sin(t_param))
    y_cyc = R * (1 - np.cos(t_param))

    # Travel time on cycloid
    g = 9.81
    ds_cyc = np.sqrt(np.diff(x_cyc)**2 + np.diff(y_cyc)**2)
    v_cyc = np.sqrt(2 * g * (y_cyc[:-1] + y_cyc[1:]) / 2)  # velocity
    dt_cyc = ds_cyc / v_cyc
    T_cyc = np.sum(dt_cyc)
    print(f"\nCycloid travel time: T = {T_cyc:.6f} s")

    # Straight line from (0,0) to (1,1): y = x
    x_line = np.linspace(1e-8, 1, 1000)
    y_line = x_line
    ds_line = np.sqrt(1 + 1) * np.diff(x_line)
    v_line = np.sqrt(2 * g * (y_line[:-1] + y_line[1:]) / 2)
    dt_line = ds_line / v_line
    T_line = np.sum(dt_line)
    print(f"Straight line time: T = {T_line:.6f} s")
    print(f"Ratio (line/cycloid): {T_line / T_cyc:.4f}")

    # Plot comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_cyc, y_cyc, 'b-', linewidth=2, label=f'Cycloid (T={T_cyc:.4f}s)')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label=f'Straight (T={T_line:.4f}s)')
    ax.set_xlabel('x')
    ax.set_ylabel('y (downward)')
    ax.set_title('Brachistochrone: Cycloid vs Straight Line')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig('ex17_brachistochrone.png', dpi=150)
    plt.close()
    print("Plot saved to ex17_brachistochrone.png")


def exercise_6():
    """
    Problem 6: Double pendulum Lagrangian and equations of motion.
    """
    print("\n" + "=" * 60)
    print("Problem 6: Double Pendulum")
    print("=" * 60)

    t = sp.Symbol('t')
    m1, m2, l1, l2, g = sp.symbols('m1 m2 l1 l2 g', positive=True)
    theta1 = sp.Function('theta1')(t)
    theta2 = sp.Function('theta2')(t)
    dtheta1 = theta1.diff(t)
    dtheta2 = theta2.diff(t)

    # Positions
    x1 = l1 * sp.sin(theta1)
    y1 = -l1 * sp.cos(theta1)
    x2 = x1 + l2 * sp.sin(theta2)
    y2 = y1 - l2 * sp.cos(theta2)

    # Velocities squared
    vx1 = x1.diff(t)
    vy1 = y1.diff(t)
    vx2 = x2.diff(t)
    vy2 = y2.diff(t)

    # Kinetic and potential energy
    T = sp.Rational(1, 2) * m1 * (vx1**2 + vy1**2) + sp.Rational(1, 2) * m2 * (vx2**2 + vy2**2)
    V = m1 * g * y1 + m2 * g * y2
    L = sp.expand(sp.trigsimp(T - V))

    print("Lagrangian: L = T - V")
    print(f"\nT = (1/2)*m1*l1^2*theta1_dot^2")
    print(f"  + (1/2)*m2*[l1^2*theta1_dot^2 + l2^2*theta2_dot^2")
    print(f"              + 2*l1*l2*theta1_dot*theta2_dot*cos(theta1-theta2)]")
    print(f"\nV = -(m1+m2)*g*l1*cos(theta1) - m2*g*l2*cos(theta2)")

    # Euler-Lagrange equations
    EL1 = sp.diff(sp.diff(L, dtheta1), t) - sp.diff(L, theta1)
    EL2 = sp.diff(sp.diff(L, dtheta2), t) - sp.diff(L, theta2)

    print("\nEuler-Lagrange equations:")
    print("  d/dt(dL/dtheta1_dot) - dL/dtheta1 = 0")
    print("  d/dt(dL/dtheta2_dot) - dL/dtheta2 = 0")

    # Numerical simulation for small oscillations
    print("\nNumerical simulation (m1=m2=1, l1=l2=1, g=9.81):")
    m1_val, m2_val, l1_val, l2_val, g_val = 1.0, 1.0, 1.0, 1.0, 9.81

    def double_pendulum(t_val, state):
        th1, w1, th2, w2 = state
        delta = th1 - th2
        den1 = (m1_val + m2_val) * l1_val - m2_val * l1_val * np.cos(delta)**2
        den2 = (l2_val / l1_val) * den1

        dw1 = (-m2_val * l1_val * w1**2 * np.sin(delta) * np.cos(delta)
               - m2_val * l2_val * w2**2 * np.sin(delta)
               - (m1_val + m2_val) * g_val * np.sin(th1)
               + m2_val * g_val * np.sin(th2) * np.cos(delta)) / den1

        dw2 = (m2_val * l2_val * w2**2 * np.sin(delta) * np.cos(delta)
               + (m1_val + m2_val) * l1_val * w1**2 * np.sin(delta)
               + (m1_val + m2_val) * g_val * np.sin(th1) * np.cos(delta)
               - (m1_val + m2_val) * g_val * np.sin(th2)) / den2

        return [w1, dw1, w2, dw2]

    sol = solve_ivp(double_pendulum, [0, 10], [np.pi / 6, 0, np.pi / 4, 0],
                    t_eval=np.linspace(0, 10, 2000), rtol=1e-10)
    print(f"  Initial: theta1={np.pi/6:.4f}, theta2={np.pi/4:.4f}")
    print(f"  Final:   theta1={sol.y[0, -1]:.4f}, theta2={sol.y[2, -1]:.4f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sol.t, np.degrees(sol.y[0]), label='theta_1')
    ax.plot(sol.t, np.degrees(sol.y[2]), label='theta_2')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title('Double Pendulum Motion')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex17_double_pendulum.png', dpi=150)
    plt.close()
    print("Plot saved to ex17_double_pendulum.png")


def exercise_7():
    """
    Problem 7: 1D harmonic oscillator.
    (a) Lagrange equation, (b) Hamiltonian, (c) Hamilton's equations.
    """
    print("\n" + "=" * 60)
    print("Problem 7: Harmonic Oscillator - Lagrangian & Hamiltonian")
    print("=" * 60)

    t = sp.Symbol('t')
    m, k = sp.symbols('m k', positive=True)
    x = sp.Function('x')(t)
    xd = x.diff(t)

    L = sp.Rational(1, 2) * m * xd**2 - sp.Rational(1, 2) * k * x**2

    # (a) Lagrange equation
    print("\n(a) L = (1/2)*m*x_dot^2 - (1/2)*k*x^2")
    p = sp.diff(L, xd)
    EL = sp.diff(p, t) - sp.diff(L, x)
    EL_simplified = sp.simplify(EL)
    print(f"  dL/dx_dot = m*x_dot = p")
    print(f"  d/dt(dL/dx_dot) - dL/dx = m*x_ddot + k*x = 0")
    print(f"  => x_ddot + (k/m)*x = 0  (SHM with omega = sqrt(k/m))")

    # (b) Hamiltonian
    print("\n(b) Legendre transform: H = p*x_dot - L")
    print(f"  p = m*x_dot  =>  x_dot = p/m")
    print(f"  H = p*(p/m) - (1/2)*m*(p/m)^2 + (1/2)*k*x^2")
    print(f"  H = p^2/(2m) + (1/2)*k*x^2  (total energy)")

    # (c) Hamilton's canonical equations
    print("\n(c) Hamilton's canonical equations:")
    print(f"  dx/dt = dH/dp = p/m")
    print(f"  dp/dt = -dH/dx = -k*x")
    print(f"\n  Combining: d^2x/dt^2 = (1/m)*dp/dt = -k*x/m")
    print(f"  => x'' + (k/m)*x = 0  (same as Lagrange equation)")

    # Numerical verification
    m_val, k_val = 1.0, 4.0
    omega = np.sqrt(k_val / m_val)
    t_num = np.linspace(0, 4 * np.pi / omega, 500)
    x0, v0 = 1.0, 0.0
    x_exact = x0 * np.cos(omega * t_num)
    p_exact = -m_val * x0 * omega * np.sin(omega * t_num)

    # Verify energy conservation
    H_vals = p_exact**2 / (2 * m_val) + 0.5 * k_val * x_exact**2
    print(f"\n  Numerical (m={m_val}, k={k_val}, omega={omega}):")
    print(f"  H(t=0) = {H_vals[0]:.6f}, H(t=T) = {H_vals[-1]:.6f}")
    print(f"  Energy conservation: max|dH| = {np.max(np.abs(H_vals - H_vals[0])):.2e}")


def exercise_8():
    """
    Problem 8: Elastic beam bending.
    J[y] = int_0^L [(EI/2)(y'')^2 - q*y] dx
    BCs: y(0) = y'(0) = 0, y''(L) = y'''(L) = 0 (cantilever).
    """
    print("\n" + "=" * 60)
    print("Problem 8: Elastic Beam (Cantilever)")
    print("=" * 60)

    print("\nJ[y] = int_0^L [(EI/2)(y'')^2 - q*y] dx")
    print("BCs: y(0) = y'(0) = 0, y''(L) = y'''(L) = 0")
    print("\nEuler-Lagrange for higher-order functional:")
    print("  dF/dy - d/dx(dF/dy') + d^2/dx^2(dF/dy'') = 0")
    print("  -q + EI*y'''' = 0")
    print("  => y'''' = q/(EI)")

    x_sym = sp.Symbol('x')
    EI, q, L_sym = sp.symbols('EI q L', positive=True)

    # General solution: y = (q/(24*EI)) * x^4 + C1*x^3/6 + C2*x^2/2 + C3*x + C4
    # Integrate y'''' = q/EI four times
    y_gen = q / (24 * EI) * x_sym**4 + sp.Rational(1, 6) * sp.Symbol('C1') * x_sym**3 + \
            sp.Rational(1, 2) * sp.Symbol('C2') * x_sym**2 + sp.Symbol('C3') * x_sym + sp.Symbol('C4')

    print(f"\nGeneral solution: y = (q/(24EI))*x^4 + C1*x^3/6 + C2*x^2/2 + C3*x + C4")
    print(f"\nApplying BCs:")
    print(f"  y(0) = 0  => C4 = 0")
    print(f"  y'(0) = 0 => C3 = 0")
    print(f"  y''(L) = 0 => q*L^2/(2*EI) + C1*L + C2 = 0")
    print(f"  y'''(L) = 0 => q*L/EI + C1 = 0  => C1 = -qL/EI")
    print(f"  => C2 = -q*L^2/(2*EI) + q*L^2/EI = q*L^2/(2*EI)")
    print(f"\ny(x) = (q/(24EI))*(x^4 - 4Lx^3 + 6L^2*x^2)")

    # Numerical example
    EI_val, q_val, L_val = 1.0, 1.0, 1.0
    x_num = np.linspace(0, L_val, 200)
    y_beam = (q_val / (24 * EI_val)) * (x_num**4 - 4 * L_val * x_num**3 + 6 * L_val**2 * x_num**2)

    y_tip = (q_val / (24 * EI_val)) * (L_val**4 - 4 * L_val**4 + 6 * L_val**4)
    print(f"\nNumerical (EI=1, q=1, L=1):")
    print(f"  Max deflection at tip: y(L) = {y_tip:.6f}")
    print(f"  = q*L^4/(8*EI) = {q_val * L_val**4 / (8 * EI_val):.6f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_num, y_beam, 'b-', linewidth=2)
    ax.fill_between(x_num, 0, y_beam, alpha=0.2)
    ax.set_xlabel('x / L')
    ax.set_ylabel('y(x) (deflection)')
    ax.set_title('Cantilever Beam Deflection Under Uniform Load')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex17_cantilever_beam.png', dpi=150)
    plt.close()
    print("Plot saved to ex17_cantilever_beam.png")


def exercise_9():
    """
    Problem 9: Show that the E-L equation of
    J[phi] = int [(eps0/2)|grad(phi)|^2 + rho*phi] d^3r
    is the Poisson equation.
    """
    print("\n" + "=" * 60)
    print("Problem 9: Variational Principle for Poisson Equation")
    print("=" * 60)

    print("\nJ[phi] = int [(eps0/2)|grad(phi)|^2 + rho*phi] d^3r")
    print("\nF = (eps0/2)(phi_x^2 + phi_y^2 + phi_z^2) + rho*phi")
    print("\nMulti-dimensional E-L equation:")
    print("  dF/dphi - d/dx(dF/dphi_x) - d/dy(dF/dphi_y) - d/dz(dF/dphi_z) = 0")
    print("\n  dF/dphi = rho")
    print("  dF/dphi_x = eps0*phi_x  =>  d/dx(dF/dphi_x) = eps0*phi_xx")
    print("  Similarly for y, z")
    print("\n  rho - eps0*(phi_xx + phi_yy + phi_zz) = 0")
    print("  => eps0*nabla^2(phi) = rho")
    print("  => nabla^2(phi) = rho/eps0")
    print("\nThis is the Poisson equation!  QED")

    # 1D numerical verification: phi'' = -rho/eps0 with Dirichlet BCs
    print("\n1D verification: phi''(x) = -rho(x)/eps0")
    eps0 = 1.0
    N = 200
    L = 1.0
    x = np.linspace(0, L, N)
    dx = x[1] - x[0]

    # Source: rho(x) = sin(pi*x)
    rho = np.sin(np.pi * x)

    # Exact solution: phi = sin(pi*x)/(pi^2*eps0)
    phi_exact = np.sin(np.pi * x) / (np.pi**2 * eps0)

    # Solve by minimizing the functional
    # Discrete version: J = sum [(eps0/2)*(phi[i+1]-phi[i])^2/dx + rho[i]*phi[i]*dx]
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve

    # Finite difference: -phi''(x) = rho/eps0 => tridiagonal system
    main_diag = 2 * np.ones(N - 2)
    off_diag = -1 * np.ones(N - 3)
    A = diags([off_diag, main_diag, off_diag], [-1, 0, 1]).toarray()
    b = (dx**2 / eps0) * rho[1:-1]

    phi_interior = np.linalg.solve(A, b)
    phi_num = np.zeros(N)
    phi_num[1:-1] = phi_interior

    print(f"  Max|phi_num - phi_exact| = {np.max(np.abs(phi_num - phi_exact)):.2e}")


def exercise_10():
    """
    Problem 10: Rayleigh-Ritz method for y'' + y = 1, y(0)=y(pi)=0.
    Trial: y = c1*sin(x) + c2*sin(2x) + c3*sin(3x).
    """
    print("\n" + "=" * 60)
    print("Problem 10: Rayleigh-Ritz Method")
    print("=" * 60)

    print("\ny'' + y = 1, y(0) = y(pi) = 0")
    print("Trial function: y = c1*sin(x) + c2*sin(2x) + c3*sin(3x)")

    # Exact solution: y_p = 1, y_h = A*sin(x) + B*cos(x)
    # y(0) = 0: B + 1 = 0 => B = -1
    # y(pi) = 0: -1 = 0  -- contradiction!
    # Actually: y = A*sin(x) + B*cos(x) + 1
    # y(0) = B + 1 = 0 => B = -1
    # y(pi) = -B + 1 = 0 + 1 = 2 ... hmm
    # Let me redo: y'' + y = 1
    # y_p = 1, y'' + y = 0 + 1 = 1. Good.
    # y_general = A sin(x) + B cos(x) + 1
    # y(0) = B + 1 = 0 => B = -1
    # y(pi) = -B + 1 = 1 + 1 = 2 != 0

    # The problem y'' + y = 1 with y(0)=y(pi)=0 has no solution because
    # lambda=1 is an eigenvalue! Let's proceed with Rayleigh-Ritz as stated.
    # The variational form: minimize J[y] = int_0^pi [y'^2 - y^2 + 2y] dx
    # (This is related to the weak form of y'' + y = 1)

    x = np.linspace(0, np.pi, 1000)
    dx_val = x[1] - x[0]

    # Basis functions
    phi = [np.sin(n * x) for n in range(1, 4)]
    dphi = [n * np.cos(n * x) for n in range(1, 4)]

    # Stiffness-like matrix: A_ij = int (phi_i'*phi_j' - phi_i*phi_j) dx
    # Load vector: b_i = int phi_i dx
    N_basis = 3
    A_mat = np.zeros((N_basis, N_basis))
    b_vec = np.zeros(N_basis)

    for i in range(N_basis):
        b_vec[i] = np.trapz(phi[i], x)
        for j in range(N_basis):
            A_mat[i, j] = np.trapz(dphi[i] * dphi[j] - phi[i] * phi[j], x)

    print(f"\nStiffness matrix A (int phi_i'*phi_j' - phi_i*phi_j):")
    for i in range(N_basis):
        print(f"  [{A_mat[i, 0]:8.4f} {A_mat[i, 1]:8.4f} {A_mat[i, 2]:8.4f}]")
    print(f"\nLoad vector b (int phi_i):")
    print(f"  [{b_vec[0]:.4f}, {b_vec[1]:.4f}, {b_vec[2]:.4f}]")

    # Note: A_11 = int (cos^2(x) - sin^2(x)) dx = 0 on [0,pi]
    # This means the system is singular for sin(x) component
    # because lambda=1 is an eigenvalue. The n=1 mode is resonant.
    print("\nNote: A[0,0] ~ 0 because lambda=1 is an eigenvalue of y''+lambda*y=0.")
    print("The sin(x) mode is resonant. Rayleigh-Ritz with sin(2x), sin(3x):")

    # Use only n=2 and n=3 (non-resonant modes)
    A_sub = A_mat[1:, 1:]
    b_sub = b_vec[1:]

    # For n=2: A_22 = int(4cos^2(2x) - sin^2(2x))dx = pi/2*(4-1) = 3*pi/2
    # For n=3: A_33 = int(9cos^2(3x) - sin^2(3x))dx = pi/2*(9-1) = 4*pi
    c_sub = np.linalg.solve(A_sub, b_sub)

    print(f"\n  c2 = {c_sub[0]:.6f}")
    print(f"  c3 = {c_sub[1]:.6f}")

    # Approximate solution (without sin(x) term)
    y_rr = c_sub[0] * np.sin(2 * x) + c_sub[1] * np.sin(3 * x)

    # Check residual: y'' + y - 1
    ypp_rr = -4 * c_sub[0] * np.sin(2 * x) - 9 * c_sub[1] * np.sin(3 * x)
    residual = ypp_rr + y_rr - 1
    print(f"  RMS residual: {np.sqrt(np.mean(residual**2)):.6f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y_rr, 'b-', linewidth=2, label='Rayleigh-Ritz (n=2,3)')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y(x)')
    ax.set_title("Rayleigh-Ritz: y'' + y = 1, y(0)=y(pi)=0")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex17_rayleigh_ritz.png', dpi=150)
    plt.close()
    print("Plot saved to ex17_rayleigh_ritz.png")


if __name__ == "__main__":
    print("=== Exercise 1 ===")
    exercise_1()
    print("\n=== Exercise 2 ===")
    exercise_2()
    print("\n=== Exercise 3 ===")
    exercise_3()
    print("\n=== Exercise 4 ===")
    exercise_4()
    print("\n=== Exercise 5 ===")
    exercise_5()
    print("\n=== Exercise 6 ===")
    exercise_6()
    print("\n=== Exercise 7 ===")
    exercise_7()
    print("\n=== Exercise 8 ===")
    exercise_8()
    print("\n=== Exercise 9 ===")
    exercise_9()
    print("\n=== Exercise 10 ===")
    exercise_10()
    print("\nAll exercises completed!")
