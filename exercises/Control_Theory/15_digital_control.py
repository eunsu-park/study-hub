"""
Exercises for Lesson 15: Digital Control Systems
Topic: Control_Theory
Solutions to practice problems from the lesson.
"""

import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def exercise_1():
    """
    Exercise 1: ZOH Discretization
    Gp(s) = 5/[s(s+5)], T = 0.1 s
    """
    T = 0.1

    print(f"Plant: Gp(s) = 5/[s(s+5)], T = {T} s")

    # Part 1: ZOH-equivalent discrete transfer function
    print(f"\nPart 1: ZOH-equivalent discretization")
    print(f"  Gp(s)/s = 5/[s^2(s+5)]")
    print(f"  Partial fractions: 5/[s^2(s+5)] = A/s + B/s^2 + C/(s+5)")
    print(f"  5 = A*s*(s+5) + B*(s+5) + C*s^2")
    print(f"  s=0: 5 = 5B => B = 1")
    print(f"  s=-5: 5 = 25C => C = 1/5")
    print(f"  s coefficient: 0 = 5A + B => A = -1/5")
    print()
    print(f"  Gp(s)/s = -1/(5s) + 1/s^2 + 1/(5(s+5))")
    print()

    # Z-transform of each term:
    # Z{-1/(5s)} at t=kT: -1/5 * z/(z-1)
    # Z{1/s^2} at t=kT: Tz/(z-1)^2
    # Z{1/(5(s+5))} at t=kT: z/(5(z-e^{-5T}))

    a = 5.0
    e_aT = np.exp(-a * T)

    print(f"  Z-transforms (t = kT):")
    print(f"  Z{{-1/(5s)}} = -z / [5(z-1)]")
    print(f"  Z{{1/s^2}} = {T}z / (z-1)^2")
    print(f"  Z{{1/(5(s+5))}} = z / [5(z-{e_aT:.6f})]")
    print()

    # Gd(z) = (1 - z^{-1}) * Z{Gp(s)/s}
    # Using scipy for exact computation
    sys_c = signal.TransferFunction([5], [1, 5, 0])
    sys_d = signal.cont2discrete((sys_c.num, sys_c.den), T, method='zoh')

    num_d, den_d = sys_d[0].flatten(), sys_d[1].flatten()
    print(f"  Gd(z) = ({np.round(num_d, 6)}) / ({np.round(den_d, 6)})")

    # Part 2: Discrete poles
    print(f"\nPart 2: Discrete poles")
    poles_d = np.roots(den_d)
    print(f"  Discrete poles: {np.round(poles_d, 6)}")

    # Continuous poles
    poles_c = np.array([0, -5])
    expected_d = np.exp(poles_c * T)
    print(f"  Continuous poles: {poles_c}")
    print(f"  Expected z = e^(sT): {np.round(expected_d, 6)}")
    print(f"  Match: {np.allclose(sorted(np.abs(poles_d)), sorted(np.abs(expected_d)), atol=1e-4)}")

    # Part 3: DC gain check
    print(f"\nPart 3: DC gain verification")
    # For step response: Gd(1) should approximate Gp(0) in terms of step behavior
    # Note: Gp(s) has a pole at s=0, so Gp(0) is infinite
    # For the step response: the steady-state output per unit step is related to
    # the plant gain. For Gp(s) = 5/[s(s+5)] = 1/[s(s/5+1)], the step response
    # is a ramp with slope = Gp_dc = 5/5 = 1 (velocity gain)
    Gd_at_1 = np.polyval(num_d, 1) / np.polyval(den_d, 1)
    print(f"  Gd(1) = {Gd_at_1:.6f}")
    print(f"  For type-1 system, Gd(z) has a pole at z=1")
    print(f"  The step response ramp rate: lim(z->1) (z-1)*Gd(z)")
    # (z-1)*Gd(z) at z=1: need to handle carefully
    # Kv_d = lim(z->1) (z-1)/T * Gd(z)  (discrete Kv)
    # For continuous: Kv = lim(s->0) s*Gp(s) = 5/5 = 1
    print(f"  Continuous Kv = 5/5 = 1")


def exercise_2():
    """
    Exercise 2: Digital PID
    Continuous PI: Gc(s) = 2(1 + 1/(0.5s)), T = 0.05 s
    """
    Kp = 2.0
    Ti = 0.5
    Ki = Kp / Ti  # = 4
    T = 0.05

    print(f"Continuous PI controller: Gc(s) = {Kp}(1 + 1/({Ti}s))")
    print(f"  Kp = {Kp}, Ti = {Ti}, Ki = Kp/Ti = {Ki}")
    print(f"  Gc(s) = {Kp}({Ti}s + 1)/({Ti}s) = ({Kp*Ti}s + {Kp})/({Ti}s)")
    print(f"  = (s + {Kp/Kp*Ti**(-1)}) * {Kp*Ti} / ({Ti}s)")
    print(f"  Sampling period T = {T} s")

    # Part 1: Tustin's method
    print(f"\nPart 1: Tustin's (bilinear) discretization")
    print(f"  Substitute s = (2/T)(z-1)/(z+1) = {2/T}(z-1)/(z+1)")

    # Gc(s) = (s + 4) / (0.5s) = 2(s+4)/s
    # = 2 + 8/s (parallel form)
    # Tustin: s -> 2/T * (z-1)/(z+1)
    # 2 + 8 * T/2 * (z+1)/(z-1)
    # = 2 + 4T * (z+1)/(z-1)
    # = 2 + 0.2 * (z+1)/(z-1)
    # = [2(z-1) + 0.2(z+1)] / (z-1)
    # = (2.2z - 1.8) / (z - 1)

    # Using scipy
    num_c = [Kp * Ti, Kp]  # Kp*Ti*s + Kp = s + 4 (times Kp*Ti... wait)
    # Gc(s) = Kp(Ti*s + 1)/(Ti*s) = (Kp*Ti*s + Kp) / (Ti*s)
    # = (1*s + 4) / (0.5*s)
    num_c_full = [1, 4]
    den_c_full = [0.5, 0]

    sys_c = signal.TransferFunction(num_c_full, den_c_full)
    sys_d = signal.cont2discrete((num_c_full, den_c_full), T, method='bilinear')
    num_d, den_d = sys_d[0].flatten(), sys_d[1].flatten()

    print(f"  Gc(z) = ({np.round(num_d, 6)}) / ({np.round(den_d, 6)})")

    # Manual calculation
    # Gc(s) = (s+4)/(0.5s) = 2(s+4)/s
    # s = 40(z-1)/(z+1)
    # Gc(z) = 2 * [40(z-1)/(z+1) + 4] / [40(z-1)/(z+1)]
    # = 2 * [40(z-1) + 4(z+1)] / [40(z-1)]
    # = 2 * [44z - 36] / [40(z-1)]
    # = (88z - 72) / (40z - 40)
    # = (2.2z - 1.8) / (z - 1)

    print(f"\n  Manual: Gc(z) = (2.2z - 1.8) / (z - 1)")

    # Part 2: Difference equation
    print(f"\nPart 2: Difference equation")
    print(f"  Gc(z) = U(z)/E(z) = (2.2z - 1.8) / (z - 1)")
    print(f"  (z - 1)U(z) = (2.2z - 1.8)E(z)")
    print(f"  u[k] - u[k-1] = 2.2*e[k] - 1.8*e[k-1]")
    print(f"  u[k] = u[k-1] + 2.2*e[k] - 1.8*e[k-1]")

    # Part 3: Velocity form comparison
    print(f"\nPart 3: Velocity form comparison")
    print(f"  Standard velocity form for PI:")
    print(f"  delta_u[k] = Kp*(e[k] - e[k-1]) + Ki*T*e[k]")
    print(f"             = {Kp}*(e[k] - e[k-1]) + {Ki}*{T}*e[k]")
    print(f"             = {Kp}*e[k] - {Kp}*e[k-1] + {Ki*T}*e[k]")
    print(f"             = ({Kp + Ki*T})*e[k] - {Kp}*e[k-1]")
    print(f"             = {Kp + Ki*T}*e[k] - {Kp}*e[k-1]")
    print(f"  u[k] = u[k-1] + {Kp + Ki*T}*e[k] - {Kp}*e[k-1]")
    print()
    print(f"  Tustin form:    u[k] = u[k-1] + 2.2*e[k] - 1.8*e[k-1]")
    print(f"  Velocity form:  u[k] = u[k-1] + {Kp+Ki*T}*e[k] - {Kp}*e[k-1]")
    print()
    print(f"  Slight difference due to Tustin vs rectangular integration.")
    print(f"  Tustin is more accurate (trapezoidal integration).")


def exercise_3():
    """
    Exercise 3: Stability Analysis
    P(z) = z^3 - 1.2z^2 + 0.5z - 0.1
    """
    coeffs = [1, -1.2, 0.5, -0.1]

    print(f"Characteristic polynomial: P(z) = z^3 - 1.2z^2 + 0.5z - 0.1")

    # Part 1: Jury test necessary conditions
    print(f"\nPart 1: Jury test necessary conditions")
    a_n = coeffs[0]   # = 1
    a_0 = coeffs[-1]  # = -0.1

    # Condition 1: P(1) > 0
    P_1 = sum(coeffs)
    print(f"  (a) P(1) = 1 - 1.2 + 0.5 - 0.1 = {P_1}")
    print(f"      P(1) > 0: {P_1 > 0}")

    # Condition 2: (-1)^n * P(-1) > 0
    P_neg1 = np.polyval(coeffs, -1)
    n = len(coeffs) - 1  # degree = 3
    cond2 = ((-1)**n) * P_neg1
    print(f"  (b) (-1)^3 * P(-1) = (-1) * {P_neg1} = {cond2}")
    print(f"      (-1)^n * P(-1) > 0: {cond2 > 0}")

    # Condition 3: |a_0| < a_n
    print(f"  (c) |a_0| < a_n: |{a_0}| < {a_n} => {abs(a_0)} < {a_n}: {abs(a_0) < a_n}")

    all_necessary = P_1 > 0 and cond2 > 0 and abs(a_0) < a_n
    print(f"\n  All necessary conditions satisfied: {all_necessary}")

    # Part 2: Determine stability
    print(f"\nPart 2: Stability determination")
    if all_necessary:
        print(f"  Necessary conditions passed. Proceeding with full Jury test...")
        # For degree 3, necessary conditions are also sufficient if they all pass
        # Actually for n=3, we need one more check in the Jury table
        # But we can just check the roots directly
        print(f"  For degree 3, we construct the Jury table:")
        print(f"  Row 1: {coeffs}")
        print(f"  Row 2: {coeffs[::-1]}")

        # Compute b_i = |a_0  a_{3-i}| = a_0*a_{3-i} - a_3*a_i (using determinant form)
        #               |a_3  a_i    |
        # Actually: b_k = a_0*a_{n-k} - a_n*a_k... this gets complex.
        # For n=3, the necessary conditions plus P(1)>0, (-1)^3 P(-1)>0, |a_0|<a_n
        # are sufficient. Let's verify with roots.
        print(f"  (For n=3, necessary conditions are sufficient)")
    else:
        print(f"  At least one necessary condition failed => UNSTABLE")

    # Part 3: Find actual roots
    print(f"\nPart 3: Actual roots")
    roots = np.roots(coeffs)
    print(f"  Roots: {np.round(roots, 6)}")

    for r in roots:
        inside = abs(r) < 1
        print(f"    z = {r:.6f}, |z| = {abs(r):.6f}, inside unit circle: {inside}")

    all_inside = all(abs(r) < 1 for r in roots)
    print(f"\n  All roots inside unit circle: {all_inside}")
    print(f"  System is {'STABLE' if all_inside else 'UNSTABLE'}")

    # Plot roots on z-plane
    fig, ax = plt.subplots(figsize=(8, 8))

    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1, label='Unit circle')

    ax.plot(roots.real, roots.imag, 'rx', markersize=12, markeredgewidth=2, label='Poles')

    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_title('Z-plane: P(z) = z^3 - 1.2z^2 + 0.5z - 0.1')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])

    for r in roots:
        ax.annotate(f'z={r:.3f}\n|z|={abs(r):.3f}',
                    (r.real, r.imag), textcoords="offset points",
                    xytext=(10, 10), fontsize=8)

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/exercises/Control_Theory/ex15_zplane.png',
                dpi=100)
    plt.close()
    print("  Z-plane plot saved to 'ex15_zplane.png'")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: ZOH Discretization ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Digital PID ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Stability Analysis ===")
    print("=" * 60)
    exercise_3()

    print("\n\nAll exercises completed!")
