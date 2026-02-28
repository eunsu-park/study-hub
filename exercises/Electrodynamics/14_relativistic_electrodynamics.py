"""
Exercises for Lesson 14: Relativistic Electrodynamics
Topic: Electrodynamics
Solutions to practice problems from the lesson.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Constants
c = 2.998e8
mu_0 = 4.0 * np.pi * 1e-7
epsilon_0 = 8.854e-12


def exercise_1():
    """
    Exercise 1: Relativistic Current Loop
    Circular loop at z=0. Charge q at (0,0,d) on axis.
    Compute force in lab frame, then boost and verify.
    """
    R = 0.1       # loop radius (m)
    I_loop = 1.0  # current (A)
    q = 1e-9      # charge (nC)
    d = 0.2       # distance on axis (m)

    # (a) Lab frame: charge at rest on axis, force = qE
    # But E from a current loop on axis = 0 (no net charge)
    # B on axis: B_z = mu_0*I*R^2 / (2*(R^2 + d^2)^(3/2))
    B_z = mu_0 * I_loop * R**2 / (2 * (R**2 + d**2)**1.5)

    # Force on charge at rest in B field: F = q*v x B = 0 (v = 0)
    print(f"  Current loop: R = {R*100:.0f} cm, I = {I_loop} A")
    print(f"  Charge q = {q*1e9:.1f} nC at (0,0,{d}) m")
    print()
    print(f"  (a) Lab frame: charge at rest")
    print(f"    B on axis: B_z = {B_z*1e6:.4f} uT")
    print(f"    Force: F = q*v x B = 0 (charge is stationary)")

    # (b) Boost to frame where charge moves with v = v0 x_hat
    v0 = 1e5  # boost velocity (m/s)
    beta = v0 / c
    gamma = 1.0 / np.sqrt(1 - beta**2)

    print(f"\n  (b) Boosted frame: v = {v0:.0e} m/s, beta = {beta:.6f}, gamma = {gamma:.8f}")

    # Field transformation (Lorentz boost in x-direction):
    # E'_x = E_x (= 0)
    # E'_y = gamma*(E_y + v*B_z) = gamma*v*B_z (since E_y = 0 in lab)
    # E'_z = gamma*(E_z - v*B_y) = 0 (B_y = 0 on axis, E_z = 0)
    # B'_x = B_x (= 0)
    # B'_y = gamma*(B_y - (v/c^2)*E_z) = 0
    # B'_z = gamma*(B_z - (v/c^2)*E_y) = gamma*B_z

    E_prime_y = gamma * v0 * B_z
    B_prime_z = gamma * B_z

    print(f"    E'_y = gamma*v*B_z = {E_prime_y:.4e} V/m")
    print(f"    B'_z = gamma*B_z = {B_prime_z*1e6:.6f} uT")

    # (c) Force in boosted frame: F' = q(E' + v' x B')
    # Charge moves with v' = -v0 x_hat in the boosted frame
    # F'_x = q*E'_x = 0
    # F'_y = q*(E'_y + (-v0)*B'_z) = q*(gamma*v0*B_z - v0*gamma*B_z) = 0
    # Actually: F'_y = q*(E'_y - v0*B'_z) ... let's be careful
    # v' x B' = (-v0 x_hat) x (B'_z z_hat) = -v0*B'_z*(x_hat x z_hat) = v0*B'_z*y_hat
    # Wait: x_hat x z_hat = -y_hat, so v' x B' = v0*B'_z*y_hat

    # Actually: (-v0 x_hat) x (B'_z z_hat) = -v0*B'_z * (x_hat x z_hat) = -v0*B'_z*(-y_hat)
    # = v0*B'_z * y_hat
    F_prime_y = q * (E_prime_y + v0 * B_prime_z)
    # Hmm, that's not zero. Let me reconsider.
    # Actually the charge in the boosted frame has v'_x = -v0
    # F' = q*(E' + v' x B')
    # (v' x B')_y = v'_x * B'_z - v'_z * B'_x = (-v0)*B'_z
    # So F'_y = q*(E'_y + (-v0)*B'_z) = q*(gamma*v0*B_z - v0*gamma*B_z) = 0

    F_prime_y_correct = q * (E_prime_y + (-v0) * B_prime_z)

    print(f"\n  (c) Force in boosted frame:")
    print(f"    v' x B' component y: (-v0)*B'_z = {(-v0)*B_prime_z:.4e}")
    print(f"    F'_y = q*(E'_y + (-v0)*B'_z) = {F_prime_y_correct:.4e} N")
    print(f"    Force is zero in both frames (consistent!)")
    print("    The electric and magnetic forces exactly cancel in the boosted frame.")


def exercise_2():
    """
    Exercise 2: Field Tensor Invariants
    EM wave: E = E0 y_hat cos(kz-wt), B = (E0/c) x_hat cos(kz-wt).
    Compute Lorentz invariants.
    """
    E0 = 100.0  # V/m

    # Invariant 1: E dot B
    # E is in y-direction, B is in x-direction: E dot B = 0
    E_dot_B = 0

    # Invariant 2: E^2 - c^2*B^2
    # E^2 = E0^2*cos^2, c^2*B^2 = c^2*(E0/c)^2*cos^2 = E0^2*cos^2
    E2_minus_c2B2 = E0**2 - c**2 * (E0 / c)**2  # = 0

    print(f"  EM wave: E = {E0} V/m * cos(kz-wt) y_hat")
    print(f"           B = {E0/c:.4e} T * cos(kz-wt) x_hat")
    print()
    print(f"  (a) Invariant 1: E dot B = {E_dot_B}")
    print(f"      Invariant 2: E^2 - c^2*B^2 = {E2_minus_c2B2}")
    print()
    print(f"  (b) Both invariants are zero. This means:")
    print(f"      - There is NO frame where only E or only B exists")
    print(f"      - In every frame, E and B are perpendicular (E.B = 0)")
    print(f"      - In every frame, |E| = c|B| (E^2 = c^2*B^2)")
    print(f"      - This is a characteristic of electromagnetic radiation")
    print()

    # (c) Field tensor
    # F^{mu nu} = [[0, -Ex/c, -Ey/c, -Ez/c],
    #               [Ex/c, 0, -Bz, By],
    #               [Ey/c, Bz, 0, -Bx],
    #               [Ez/c, -By, Bx, 0]]
    # For our wave at t=0, z=0: Ex=0, Ey=E0, Ez=0, Bx=E0/c, By=0, Bz=0
    F = np.array([
        [0, 0, -E0 / c, 0],
        [0, 0, 0, E0 / c],
        [E0 / c, 0, 0, -E0 / c],
        [0, -E0 / c, E0 / c, 0]
    ])

    # F_mu_nu F^{mu nu} = -2(E^2/c^2 - B^2) with metric signature (-, +, +, +)
    # Using covariant tensor with lowered indices via metric:
    eta = np.diag([-1, 1, 1, 1])
    F_lower = eta @ F @ eta
    invariant = np.sum(F_lower * F)

    print(f"  (c) Field tensor F^{{mu nu}} (at t=0, z=0):")
    for row in F:
        print(f"    [{', '.join(f'{v:10.4e}' for v in row)}]")
    print(f"  F_{{mu nu}} F^{{mu nu}} = {invariant:.4e}")
    print(f"  Expected: -2*(E^2 - c^2*B^2)/c^2 = 0")


def exercise_3():
    """
    Exercise 3: Covariant Continuity
    Show charge conservation follows from antisymmetry of F^{mu nu}.
    """
    print("  Covariant Maxwell's equation: d_mu F^{mu nu} = mu_0 J^{nu}")
    print()
    print("  Take divergence: d_nu d_mu F^{mu nu} = mu_0 d_nu J^{nu}")
    print()
    print("  Left side: d_nu d_mu F^{mu nu}")
    print("  Since partial derivatives commute: d_nu d_mu = d_mu d_nu")
    print("  And F^{mu nu} = -F^{nu mu} (antisymmetric)")
    print("  d_nu d_mu F^{mu nu} = d_mu d_nu F^{mu nu} = -d_mu d_nu F^{nu mu}")
    print("  Relabel: mu <-> nu: = -d_nu d_mu F^{mu nu}")
    print("  So: d_nu d_mu F^{mu nu} = -d_nu d_mu F^{mu nu}")
    print("  Therefore: 2 * d_nu d_mu F^{mu nu} = 0")
    print("  Hence: d_nu d_mu F^{mu nu} = 0")
    print()
    print("  This means: mu_0 * d_nu J^{nu} = 0")
    print("  => d_nu J^{nu} = 0")
    print()
    print("  In 3D notation: d(rho)/dt + div(J) = 0")
    print("  This is the continuity equation (charge conservation)!")
    print()
    print("  Charge conservation is not an independent assumption -- it follows")
    print("  automatically from the antisymmetry of the field tensor.")


def exercise_4():
    """
    Exercise 4: Stress-Energy Tensor for a Plane Wave
    Compute all 16 components of T^{mu nu}.
    """
    E0 = 100.0  # V/m (arbitrary)

    # Plane wave in z-direction: E = E0 y_hat, B = (E0/c) x_hat (at peak)
    E = np.array([0, 0, E0, 0])  # (t, x, y, z) -> E_y = E0
    B = np.array([0, E0 / c, 0, 0])  # B_x = E0/c

    # Energy density: u = (1/2)(eps_0*E^2 + B^2/mu_0) = eps_0*E0^2
    u = epsilon_0 * E0**2

    # Poynting vector: S = ExB/mu_0 = E0*(E0/c)/mu_0 * z_hat = eps_0*c*E0^2 * z_hat
    S_z = epsilon_0 * c * E0**2

    # Stress-energy tensor components:
    # T^{00} = u (energy density)
    # T^{0i} = S_i/c (momentum density / energy flux)
    # T^{ij} = Maxwell stress tensor

    T = np.zeros((4, 4))
    T[0, 0] = u
    T[0, 3] = S_z / c
    T[3, 0] = S_z / c
    T[3, 3] = u  # radiation pressure = energy density for EM wave

    # Off-diagonal stress components
    # T^{11} = eps_0*(E_x^2 - E^2/2) + (B_x^2 - B^2/2)/mu_0
    # = eps_0*(-E0^2/2) + ((E0/c)^2 - (E0/c)^2/2)/mu_0
    # = -eps_0*E0^2/2 + eps_0*E0^2/2 = 0
    T[1, 1] = epsilon_0 * (0 - E0**2 / 2) + ((E0 / c)**2 - (E0 / c)**2 / 2) / mu_0
    T[2, 2] = epsilon_0 * (E0**2 - E0**2 / 2) + (0 - (E0 / c)**2 / 2) / mu_0

    print(f"  Plane wave in z-direction: E0 = {E0} V/m")
    print(f"  Energy density: u = {u:.4e} J/m^3")
    print(f"  Poynting flux: S_z = {S_z:.4e} W/m^2")
    print()
    print("  Stress-energy tensor T^{mu nu} (in J/m^3):")
    labels = ['t', 'x', 'y', 'z']
    print("        " + "".join(f"{l:>12s}" for l in labels))
    for i in range(4):
        print(f"  {labels[i]:>4s}: " + "".join(f"{T[i,j]:12.4e}" for j in range(4)))
    print()
    print(f"  T^00 = {T[0,0]:.4e} (energy density)")
    print(f"  T^03/c = {T[0,3]:.4e} (momentum density * c)")
    print(f"  T^33 = {T[3,3]:.4e} (radiation pressure)")
    print(f"  T^00 = T^33: {abs(T[0,0] - T[3,3]) < 1e-20}")
    print("  This equality T^00 = T^33 is the hallmark of massless radiation.")


if __name__ == "__main__":
    print("=== Exercise 1: Relativistic Current Loop ===")
    exercise_1()
    print("\n=== Exercise 2: Field Tensor Invariants ===")
    exercise_2()
    print("\n=== Exercise 3: Covariant Continuity ===")
    exercise_3()
    print("\n=== Exercise 4: Stress-Energy Tensor ===")
    exercise_4()
    print("\nAll exercises completed!")
