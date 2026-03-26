"""
Laplace Transform Methods for ODEs

Demonstrates:
  - Computing Laplace transforms with SymPy
  - Inverse Laplace transforms
  - Solving initial value problems using Laplace transforms
  - Step response and transfer functions for LTI systems

Dependencies: numpy, matplotlib, sympy
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


# ---------------------------------------------------------------------------
# 1. Computing Laplace Transforms
# ---------------------------------------------------------------------------
def laplace_transform_examples():
    """Compute Laplace transforms of common functions.

    The Laplace transform converts a time-domain function f(t) into a
    complex-frequency domain function F(s):

      F(s) = integral_0^inf f(t) * exp(-s*t) dt

    This is powerful because differentiation becomes multiplication by s,
    turning ODEs into algebraic equations.
    """
    t = sp.Symbol("t", positive=True)
    s = sp.Symbol("s")

    functions = [
        ("1 (unit step)", sp.Heaviside(t)),
        ("t", t),
        ("t^2", t ** 2),
        ("exp(-a*t)", sp.exp(-sp.Symbol("a", positive=True) * t)),
        ("sin(omega*t)", sp.sin(sp.Symbol("omega", positive=True) * t)),
        ("cos(omega*t)", sp.cos(sp.Symbol("omega", positive=True) * t)),
        ("t*exp(-t)", t * sp.exp(-t)),
        ("delta(t-a)", sp.DiracDelta(t - sp.Symbol("a", positive=True))),
    ]

    print("Laplace Transform Table")
    print("=" * 65)
    for name, f_t in functions:
        F_s = sp.laplace_transform(f_t, t, s, noconds=True)
        print(f"  L{{{name}}} = {F_s}")
    print()


# ---------------------------------------------------------------------------
# 2. Inverse Laplace Transform
# ---------------------------------------------------------------------------
def inverse_laplace_examples():
    """Compute inverse Laplace transforms (partial fraction decomposition).

    The inverse transform recovers the time-domain function.  For rational
    F(s) = P(s)/Q(s), partial fractions reduce the problem to a table lookup.
    """
    s = sp.Symbol("s")
    t = sp.Symbol("t", positive=True)

    transforms = [
        ("1/s", 1 / s),
        ("1/(s+2)", 1 / (s + 2)),
        ("1/(s^2+4)", 1 / (s ** 2 + 4)),
        ("s/(s^2+1)", s / (s ** 2 + 1)),
        ("1/((s+1)(s+3))", 1 / ((s + 1) * (s + 3))),
        ("(2s+3)/(s^2+2s+5)", (2 * s + 3) / (s ** 2 + 2 * s + 5)),
    ]

    print("Inverse Laplace Transform Examples")
    print("=" * 65)
    for name, F_s in transforms:
        # Show partial fraction decomposition for rational functions
        pf = sp.apart(F_s, s)
        f_t = sp.inverse_laplace_transform(F_s, s, t)
        print(f"  L^-1{{{name}}} = {sp.simplify(f_t)}")
        if pf != F_s:
            print(f"    Partial fractions: {pf}")
    print()


# ---------------------------------------------------------------------------
# 3. Solving IVPs with Laplace Transforms
# ---------------------------------------------------------------------------
def solve_ivp_laplace():
    """Solve an IVP using the Laplace transform method.

    Problem: y'' + 3y' + 2y = exp(-t),  y(0) = 1, y'(0) = 0

    Method:
      1. Take Laplace transform of both sides
      2. Use L{y'} = s*Y - y(0), L{y''} = s^2*Y - s*y(0) - y'(0)
      3. Solve algebraically for Y(s)
      4. Inverse Laplace transform to get y(t)
    """
    t = sp.Symbol("t", positive=True)
    s = sp.Symbol("s")
    Y = sp.Symbol("Y")

    # IVP: y'' + 3y' + 2y = exp(-t), y(0)=1, y'(0)=0
    y0, y0_prime = 1, 0

    # Laplace of each term:
    # L{y''} = s^2 Y - s*y(0) - y'(0)  = s^2 Y - s
    # L{y'}  = s Y - y(0)               = s Y - 1
    # L{exp(-t)} = 1/(s+1)

    lhs = (s ** 2 * Y - s * y0 - y0_prime) + 3 * (s * Y - y0) + 2 * Y
    rhs = 1 / (s + 1)
    eq = sp.Eq(lhs, rhs)

    # Solve for Y
    Y_sol = sp.solve(eq, Y)[0]
    Y_simplified = sp.apart(Y_sol, s)

    print("Solving IVP: y'' + 3y' + 2y = exp(-t), y(0)=1, y'(0)=0")
    print(f"  Y(s) = {Y_sol}")
    print(f"  Partial fractions: {Y_simplified}")

    # Inverse Laplace
    y_t = sp.inverse_laplace_transform(Y_sol, s, t)
    y_t_simplified = sp.simplify(y_t)
    print(f"  y(t) = {y_t_simplified}")

    # Verify using SymPy's ODE solver
    y = sp.Function("y")
    ode = sp.Eq(y(t).diff(t, 2) + 3 * y(t).diff(t) + 2 * y(t), sp.exp(-t))
    sol_verify = sp.dsolve(ode, y(t), ics={y(0): y0, y(t).diff(t).subs(t, 0): y0_prime})
    y_verify = sp.simplify(sol_verify.rhs)
    print(f"  Verification (dsolve): {y_verify}")

    return y_t_simplified


# ---------------------------------------------------------------------------
# 4. Step Response of LTI Systems
# ---------------------------------------------------------------------------
def step_response():
    """Compute and plot the step response of a second-order LTI system.

    Transfer function: H(s) = omega_n^2 / (s^2 + 2*zeta*omega_n*s + omega_n^2)

    The step response Y(s) = H(s) * (1/s) shows how the system responds
    to a unit step input.  We vary the damping ratio zeta.
    """
    t_sym = sp.Symbol("t", positive=True)
    s_sym = sp.Symbol("s")
    omega_n = 2.0  # natural frequency

    fig, ax = plt.subplots(figsize=(9, 5))

    zeta_values = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(zeta_values)))

    for zeta, color in zip(zeta_values, colors):
        # H(s) = omega_n^2 / (s^2 + 2*zeta*omega_n*s + omega_n^2)
        H = omega_n ** 2 / (s_sym ** 2 + 2 * zeta * omega_n * s_sym + omega_n ** 2)
        # Step response: Y(s) = H(s) / s
        Y = H / s_sym
        y_t = sp.inverse_laplace_transform(Y, s_sym, t_sym)
        y_func = sp.lambdify(t_sym, y_t, "numpy")

        t_arr = np.linspace(0.01, 10, 500)
        try:
            y_arr = np.real(np.array([complex(y_func(ti)) for ti in t_arr]))
        except (TypeError, ValueError):
            # Fallback: use numerical evaluation for each point
            y_arr = np.array([float(y_t.subs(t_sym, ti).evalf()) for ti in t_arr])

        ax.plot(t_arr, y_arr, "-", color=color, lw=1.5,
                label=f"zeta = {zeta}")

    ax.axhline(1.0, color="gray", ls="--", alpha=0.5, label="Steady state")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("y(t)")
    ax.set_title(f"Step Response: H(s) = omega_n^2 / (s^2 + 2*zeta*omega_n*s + omega_n^2), omega_n={omega_n}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 2.0)
    plt.tight_layout()
    plt.savefig("12_step_response.png", dpi=100)
    plt.close()
    print("[Saved] 12_step_response.png")


# ---------------------------------------------------------------------------
# 5. Transfer Function Analysis
# ---------------------------------------------------------------------------
def transfer_function_analysis():
    """Analyze poles and zeros of a transfer function.

    Poles determine stability: a system is stable iff all poles have
    negative real parts (lie in the left half of the s-plane).
    """
    s = sp.Symbol("s")

    # H(s) = (s + 1) / (s^2 + 3s + 2) = (s+1) / ((s+1)(s+2)) = 1/(s+2)
    H = (s + 1) / (s ** 2 + 3 * s + 2)
    num = sp.Poly(sp.numer(H), s)
    den = sp.Poly(sp.denom(H), s)

    zeros = sp.solve(sp.numer(H), s)
    poles = sp.solve(sp.denom(H), s)

    print("Transfer Function Analysis")
    print(f"  H(s) = {H}")
    print(f"  Simplified: {sp.cancel(H)}")
    print(f"  Zeros: {zeros}")
    print(f"  Poles: {poles}")

    # Check stability
    all_stable = all(sp.re(p) < 0 for p in poles)
    print(f"  Stable: {all_stable} (all poles in left half-plane)")

    # Pole-zero plot
    fig, ax = plt.subplots(figsize=(6, 6))
    for z in zeros:
        ax.plot(float(sp.re(z)), float(sp.im(z)), "bo", ms=10,
                label="Zero" if z == zeros[0] else "")
    for p in poles:
        ax.plot(float(sp.re(p)), float(sp.im(p)), "rx", ms=12, mew=2,
                label="Pole" if p == poles[0] else "")

    ax.axvline(0, color="gray", lw=0.5)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.set_title("Pole-Zero Map")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig("12_pole_zero.png", dpi=100)
    plt.close()
    print("[Saved] 12_pole_zero.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Laplace Transform Methods for ODEs")
    print("=" * 60)

    # --- Demo 1: Laplace transform table ---
    print("\nDemo 1: Laplace transform computation")
    laplace_transform_examples()

    # --- Demo 2: Inverse Laplace ---
    print("Demo 2: Inverse Laplace transforms")
    inverse_laplace_examples()

    # --- Demo 3: Solve IVP ---
    print("Demo 3: Solving IVP with Laplace transform")
    y_sol = solve_ivp_laplace()

    # --- Demo 4: Step response ---
    print("\nDemo 4: Step response of second-order system")
    step_response()

    # --- Demo 5: Transfer function analysis ---
    print("\nDemo 5: Transfer function pole-zero analysis")
    transfer_function_analysis()
