"""
Exercises for Lesson 13: Partial Differential Equations
Topic: Mathematical_Methods
Solutions to practice problems from the lesson.
"""

import numpy as np
import sympy as sp
from scipy.special import jn_zeros, jv, legendre
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def exercise_1():
    """
    Problem 1: Classify PDEs.
    (a) u_xx + 4u_xy + u_yy = 0
    (b) u_xx + 2u_xy + u_yy = 0
    (c) 3u_xx - 6u_yy = 0
    """
    print("=" * 60)
    print("Problem 1: PDE Classification")
    print("=" * 60)

    print("\nGeneral form: A u_xx + 2B u_xy + C u_yy + ... = 0")
    print("Discriminant: Delta = B^2 - AC")

    cases = [
        ("(a) u_xx + 4u_xy + u_yy = 0", 1, 2, 1),
        ("(b) u_xx + 2u_xy + u_yy = 0", 1, 1, 1),
        ("(c) 3u_xx - 6u_yy = 0", 3, 0, -6),
    ]

    for desc, A, B, C in cases:
        delta = B**2 - A * C
        if delta < 0:
            pde_type = "Elliptic"
        elif delta == 0:
            pde_type = "Parabolic"
        else:
            pde_type = "Hyperbolic"
        print(f"\n{desc}")
        print(f"  A={A}, B={B}, C={C}")
        print(f"  Delta = {B}^2 - ({A})({C}) = {delta}")
        print(f"  Type: {pde_type}")


def exercise_2():
    """
    Problem 2: Heat equation L=pi, alpha=1, f(x) = sin(x) + 3*sin(2x).
    (a) u(x,t), (b) plot at t=0.5, (c) decay rate ratio n=2 vs n=1.
    """
    print("\n" + "=" * 60)
    print("Problem 2: Heat Equation")
    print("=" * 60)

    L = np.pi
    alpha = 1.0
    x = np.linspace(0, L, 500)

    print("\n(a) f(x) = sin(x) + 3*sin(2x)")
    print("  Already in Fourier sine series: b_1=1, b_2=3")
    print("  u(x,t) = sin(x)*exp(-t) + 3*sin(2x)*exp(-4t)")

    def u(x_val, t):
        return np.sin(x_val) * np.exp(-t) + 3 * np.sin(2 * x_val) * np.exp(-4 * t)

    # (b) Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for t_val in [0, 0.1, 0.5, 1.0, 2.0]:
        ax.plot(x, u(x, t_val), linewidth=2, label=f't={t_val}')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')
    ax.set_title('Heat Equation: f(x) = sin(x) + 3sin(2x)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex13_heat_equation.png', dpi=150)
    plt.close()
    print("\n(b) Plot saved to ex13_heat_equation.png")
    print(f"  u(pi/2, 0.5) = {u(np.pi / 2, 0.5):.6f}")

    # (c) Decay rates
    tau1 = 1.0 / (alpha**2 * (1 * np.pi / L)**2)
    tau2 = 1.0 / (alpha**2 * (2 * np.pi / L)**2)
    print(f"\n(c) Time constants:")
    print(f"  tau_1 = 1/(alpha^2*(pi/L)^2) = {tau1:.4f}")
    print(f"  tau_2 = 1/(alpha^2*(2pi/L)^2) = {tau2:.4f}")
    print(f"  n=2 decays {tau1 / tau2:.1f}x faster than n=1")


def exercise_3():
    """
    Problem 3: Wave equation L=1, c=2, u(x,0)=0, u_t(x,0)=sin(pi*x).
    """
    print("\n" + "=" * 60)
    print("Problem 3: Wave Equation")
    print("=" * 60)

    L, c = 1.0, 2.0
    x = np.linspace(0, L, 500)

    print("\nu_tt = c^2 u_xx, u(x,0)=0, u_t(x,0)=sin(pi*x)")
    print(f"\n(a) u(x,0)=0 => A_n = 0 for all n")
    print(f"  u_t(x,0) = sin(pi*x) = sum B_n * omega_n * sin(n*pi*x)")
    print(f"  omega_n = n*pi*c/L = {np.pi * c / L:.4f}*n")
    print(f"  B_1 * omega_1 = 1  =>  B_1 = 1/omega_1 = {1 / (np.pi * c / L):.6f}")
    print(f"  B_n = 0 for n >= 2")
    print(f"\n  u(x,t) = (1/omega_1) sin(pi*x) sin(omega_1*t)")

    omega1 = np.pi * c / L

    def u(x_val, t):
        return (1.0 / omega1) * np.sin(np.pi * x_val / L) * np.sin(omega1 * t)

    print(f"\n(b) Fundamental frequency:")
    print(f"  f_1 = c/(2L) = {c / (2 * L):.2f} Hz")
    print(f"  omega_1 = {omega1:.4f} rad/s")

    fig, ax = plt.subplots(figsize=(10, 6))
    period = 2 * L / c
    for t_val in [0, period / 8, period / 4, 3 * period / 8, period / 2]:
        ax.plot(x, u(x, t_val), linewidth=2, label=f't={t_val:.3f}')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')
    ax.set_title('Wave Equation Solution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex13_wave_equation.png', dpi=150)
    plt.close()
    print("Plot saved to ex13_wave_equation.png")


def exercise_4():
    """
    Problem 4: Laplace's equation in unit disk, u(1,theta) = cos^2(theta).
    """
    print("\n" + "=" * 60)
    print("Problem 4: Laplace's Equation in Unit Disk")
    print("=" * 60)

    a = 1.0

    print("\n(a) Fourier expansion of cos^2(theta):")
    print("  cos^2(theta) = (1 + cos(2*theta))/2")
    print("  a_0/2 = 1/2, a_2 = 1/2, all others zero")

    print("\n(b) Solution in disk:")
    print("  u(r,theta) = a_0/2 + (r/a)^2 * (1/2)*cos(2*theta)")
    print("  u(r,theta) = 1/2 + (r^2/2)*cos(2*theta)")

    print("\n(c) Temperature at center:")
    print("  u(0,theta) = 1/2")
    print("  This is the mean value of the boundary temperature,")
    print("  which follows from the mean value property of harmonic functions.")

    # Verify: mean of cos^2(theta) over [0, 2pi]
    theta = np.linspace(0, 2 * np.pi, 10000)
    mean_val = np.mean(np.cos(theta)**2)
    print(f"  Mean of cos^2(theta) = {mean_val:.6f}")

    # Visualization
    r = np.linspace(0, a, 100)
    th = np.linspace(0, 2 * np.pi, 200)
    R, Th = np.meshgrid(r, th)
    U = 0.5 + 0.5 * R**2 * np.cos(2 * Th)

    fig, ax = plt.subplots(figsize=(8, 7))
    X = R * np.cos(Th)
    Y = R * np.sin(Th)
    cs = ax.contourf(X, Y, U, levels=30, cmap='RdBu_r')
    plt.colorbar(cs, label='u(r,theta)')
    circle = plt.Circle((0, 0), a, fill=False, color='k', linewidth=2)
    ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.set_title('Laplace equation: u(1,theta) = cos^2(theta)')
    plt.tight_layout()
    plt.savefig('ex13_laplace_disk.png', dpi=150)
    plt.close()
    print("Plot saved to ex13_laplace_disk.png")


def exercise_5():
    """
    Problem 5: 2D square well energy levels and degeneracies.
    """
    print("\n" + "=" * 60)
    print("Problem 5: 2D Infinite Square Well")
    print("=" * 60)

    a = 1.0
    E = lambda nx, ny: (np.pi**2 / 2) * (nx**2 + ny**2)
    E1 = E(1, 1)

    print("\nE(nx,ny) = (pi^2/2)*(nx^2 + ny^2)")
    print(f"\n(a) First 5 energy levels (in units of E_11 = {E1:.4f}):")

    # Find all energy levels up to some cutoff
    levels = {}
    for nx in range(1, 6):
        for ny in range(1, 6):
            e = nx**2 + ny**2
            key = e
            if key not in levels:
                levels[key] = []
            levels[key].append((nx, ny))

    sorted_levels = sorted(levels.items())[:5]
    for i, (e_key, states) in enumerate(sorted_levels):
        e_val = (np.pi**2 / 2) * e_key
        degeneracy = len(states)
        state_str = ", ".join(f"({nx},{ny})" for nx, ny in states)
        print(f"  Level {i + 1}: E = {e_val / E1:.1f}*E_11, "
              f"degeneracy = {degeneracy}, states: {state_str}")

    # (b) Compare |psi|^2 for (1,2) and (2,1)
    print(f"\n(b) States (1,2) and (2,1) are degenerate: E = {E(1, 2) / E1:.1f}*E_11")
    print("  |psi_12|^2 and |psi_21|^2 have different spatial distributions")
    print("  psi_12 = (2/a) sin(pi*x) sin(2pi*y)")
    print("  psi_21 = (2/a) sin(2pi*x) sin(pi*y)")

    x = np.linspace(0, a, 100)
    y = np.linspace(0, a, 100)
    X, Y = np.meshgrid(x, y)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (nx, ny) in zip(axes, [(1, 2), (2, 1)]):
        psi = (2 / a) * np.sin(nx * np.pi * X / a) * np.sin(ny * np.pi * Y / a)
        cs = ax.contourf(X, Y, psi**2, levels=30, cmap='inferno')
        plt.colorbar(cs, ax=ax)
        ax.set_title(f'|psi_{{{nx}{ny}}}|^2')
        ax.set_aspect('equal')

    plt.suptitle('Degenerate States (1,2) and (2,1)', fontsize=14)
    plt.tight_layout()
    plt.savefig('ex13_2d_well.png', dpi=150)
    plt.close()
    print("Plot saved to ex13_2d_well.png")


def exercise_6():
    """
    Problem 6: Helmholtz equation in circular region, axisymmetric modes.
    """
    print("\n" + "=" * 60)
    print("Problem 6: Helmholtz Equation - Circular Domain")
    print("=" * 60)

    a = 1.0
    print(f"\nnabla^2 u + k^2 u = 0 in disk r < {a}, u(a,theta) = 0")
    print("Axisymmetric (m=0): R(r) = J_0(kr)")
    print("BC: J_0(ka) = 0  =>  ka = j_{0,n}")

    zeros = jn_zeros(0, 5)
    print(f"\nEigenvalues k_{{0,n}} = j_{{0,n}}/a:")
    for n in range(5):
        print(f"  k_{{0,{n + 1}}} = {zeros[n] / a:.6f}")

    print(f"\nEigenfunctions: u_n(r) = J_0(k_{{0,n}} r)")

    fig, ax = plt.subplots(figsize=(8, 5))
    r = np.linspace(0, a, 200)
    for n in range(3):
        ax.plot(r, jv(0, zeros[n] * r / a), linewidth=2,
                label=f'J_0(j_{{0,{n + 1}}} r), k={zeros[n]:.3f}')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xlabel('r')
    ax.set_ylabel('J_0(kr)')
    ax.set_title('Axisymmetric Helmholtz Eigenfunctions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex13_helmholtz.png', dpi=150)
    plt.close()
    print("Plot saved to ex13_helmholtz.png")


def exercise_7():
    """
    Problem 7: Nonhomogeneous PDE u_t = u_xx + 2sin(3pi*x),
    u(0,t)=u(1,t)=0, u(x,0)=sin(pi*x).
    """
    print("\n" + "=" * 60)
    print("Problem 7: Nonhomogeneous Heat Equation")
    print("=" * 60)

    L = 1.0
    x = np.linspace(0, L, 500)

    print("\nu_t = u_xx + 2*sin(3*pi*x), u(0,t)=u(1,t)=0, u(x,0)=sin(pi*x)")
    print("\nEigenfunction expansion: u(x,t) = sum T_n(t) sin(n*pi*x)")
    print("Source: F(x,t) = 2*sin(3*pi*x)  =>  F_3 = 2, F_n=0 for n!=3")
    print("IC: u(x,0) = sin(pi*x)  =>  T_1(0)=1, T_n(0)=0 for n!=1")
    print("\nODE for each mode:")
    print("  T_n' + (n*pi)^2 T_n = F_n")
    print("\nFor n=1: T_1' + pi^2 T_1 = 0, T_1(0)=1")
    print("  T_1(t) = exp(-pi^2 * t)")
    print("\nFor n=3: T_3' + 9*pi^2 T_3 = 2, T_3(0)=0")
    print("  T_3(t) = (2/(9*pi^2)) * [1 - exp(-9*pi^2 * t)]")
    print("\nu(x,t) = exp(-pi^2*t)*sin(pi*x)")
    print("       + (2/(9*pi^2))*[1-exp(-9*pi^2*t)]*sin(3*pi*x)")

    def u(x_val, t):
        T1 = np.exp(-np.pi**2 * t)
        T3 = (2 / (9 * np.pi**2)) * (1 - np.exp(-9 * np.pi**2 * t))
        return T1 * np.sin(np.pi * x_val) + T3 * np.sin(3 * np.pi * x_val)

    fig, ax = plt.subplots(figsize=(10, 6))
    for t_val in [0, 0.01, 0.05, 0.1, 0.5, 2.0]:
        ax.plot(x, u(x, t_val), linewidth=2, label=f't={t_val}')

    # Steady state: u_ss = (2/(9*pi^2))*sin(3*pi*x)
    u_ss = (2 / (9 * np.pi**2)) * np.sin(3 * np.pi * x)
    ax.plot(x, u_ss, 'k--', linewidth=2, label='steady state')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')
    ax.set_title('Nonhomogeneous Heat Equation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex13_nonhomogeneous.png', dpi=150)
    plt.close()
    print("\nPlot saved to ex13_nonhomogeneous.png")


def exercise_8():
    """
    Problem 8: Uniqueness proof for wave equation using energy method.
    """
    print("\n" + "=" * 60)
    print("Problem 8: Uniqueness of Wave Equation (Energy Method)")
    print("=" * 60)

    print("\nGoal: Prove uniqueness for u_tt = c^2 u_xx, u(0,t)=u(L,t)=0")
    print("\nProof:")
    print("  Suppose two solutions u_1, u_2. Let w = u_1 - u_2.")
    print("  Then w_tt = c^2 w_xx, w(0,t)=w(L,t)=0,")
    print("  w(x,0) = 0, w_t(x,0) = 0")
    print("\n  Define energy: E(t) = int_0^L [w_t^2 + c^2 w_x^2] dx")
    print("\n  dE/dt = 2 int_0^L [w_t w_tt + c^2 w_x w_xt] dx")
    print("        = 2 int_0^L [w_t (c^2 w_xx) + c^2 w_x w_xt] dx")
    print("        = 2c^2 int_0^L [w_t w_xx + w_x w_xt] dx")
    print("        = 2c^2 int_0^L d/dx[w_x w_t] dx")
    print("        = 2c^2 [w_x w_t]_0^L")
    print("        = 0  (since w(0,t)=w(L,t)=0 => w_t(0,t)=w_t(L,t)=0)")
    print("\n  Therefore E(t) = E(0) = 0 (from zero IC)")
    print("  Since w_t^2 >= 0 and c^2 w_x^2 >= 0, both must be zero.")
    print("  Hence w_t = 0 and w_x = 0, so w = constant = 0.")
    print("  Therefore u_1 = u_2 (uniqueness). QED")


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
    print("\nAll exercises completed!")
