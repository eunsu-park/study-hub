"""
Multiple Integrals and Change of Variables

Demonstrates:
  - Double integral in Cartesian and polar coordinates
  - Triple integral with SciPy
  - Jacobian computation for coordinate transforms
  - Change of variables (Cartesian <-> polar)
  - Monte Carlo integration comparison

Dependencies: numpy, scipy, matplotlib, sympy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import sympy as sp

np.random.seed(42)  # reproducibility for Monte Carlo


# ---------------------------------------------------------------------------
# 1. Double Integral (Cartesian)
# ---------------------------------------------------------------------------
def double_integral_cartesian():
    """Compute integral of x^2 + y^2 over the unit square [0,1]x[0,1].

    Analytical answer: int_0^1 int_0^1 (x^2 + y^2) dy dx
                     = int_0^1 [x^2*y + y^3/3]_0^1 dx
                     = int_0^1 (x^2 + 1/3) dx = 1/3 + 1/3 = 2/3.
    """
    f = lambda y, x: x ** 2 + y ** 2
    result, error = integrate.dblquad(f, 0, 1, 0, 1)

    print("Double Integral (Cartesian)")
    print(f"  integral int_0^1 int_0^1 (x^2 + y^2) dy dx")
    print(f"  Numerical : {result:.10f}")
    print(f"  Exact     : {2 / 3:.10f}")
    print(f"  Error est : {error:.2e}")
    return result


# ---------------------------------------------------------------------------
# 2. Double Integral (Polar)
# ---------------------------------------------------------------------------
def double_integral_polar():
    """Compute integral of exp(-(x^2+y^2)) over the disk x^2+y^2 <= R^2.

    In polar: int_0^{2pi} int_0^R exp(-r^2) * r dr dtheta
            = 2 pi * [-1/2 exp(-r^2)]_0^R = pi * (1 - exp(-R^2)).

    The key insight is that the Jacobian of the polar transform (r) turns
    the area element dx dy into r dr dtheta, simplifying the integral.
    """
    R = 2.0
    # In polar, the integrand is exp(-r^2) * r (the extra r is the Jacobian)
    f_polar = lambda r, theta: np.exp(-r ** 2) * r
    result, error = integrate.dblquad(f_polar, 0, 2 * np.pi, 0, R)
    exact = np.pi * (1 - np.exp(-R ** 2))

    print("\nDouble Integral (Polar Coordinates)")
    print(f"  integral exp(-(x^2+y^2)) over disk of radius {R}")
    print(f"  Numerical : {result:.10f}")
    print(f"  Exact     : {exact:.10f}")
    return result


# ---------------------------------------------------------------------------
# 3. Triple Integral
# ---------------------------------------------------------------------------
def triple_integral_demo():
    """Compute the volume of the unit sphere using a triple integral.

    V = int int int_{x^2+y^2+z^2 <= 1} 1 dV

    We use scipy.integrate.tplquad with the appropriate limits.
    For the unit sphere: z in [-1,1], y in [-sqrt(1-z^2), sqrt(1-z^2)],
    x in [-sqrt(1-z^2-y^2), sqrt(1-z^2-y^2)].
    """
    # Limits for z, y(z), x(y,z)
    z_lo, z_hi = -1, 1
    y_lo = lambda z: -np.sqrt(max(1 - z ** 2, 0))
    y_hi = lambda z: np.sqrt(max(1 - z ** 2, 0))
    x_lo = lambda y, z: -np.sqrt(max(1 - y ** 2 - z ** 2, 0))
    x_hi = lambda y, z: np.sqrt(max(1 - y ** 2 - z ** 2, 0))

    result, error = integrate.tplquad(
        lambda x, y, z: 1.0, z_lo, z_hi, y_lo, y_hi, x_lo, x_hi
    )
    exact = 4 / 3 * np.pi

    print("\nTriple Integral (Volume of Unit Sphere)")
    print(f"  Numerical : {result:.10f}")
    print(f"  Exact     : {exact:.10f}")
    print(f"  Error est : {error:.2e}")
    return result


# ---------------------------------------------------------------------------
# 4. Jacobian Computation
# ---------------------------------------------------------------------------
def jacobian_demo():
    """Compute the Jacobian for polar and spherical coordinate transforms.

    The Jacobian determinant tells us how area/volume elements scale under
    a change of variables.  For polar:  |J| = r.
    For spherical: |J| = r^2 sin(phi).
    """
    r, theta, phi = sp.symbols("r theta phi", positive=True)

    # Polar coordinates: (r, theta) -> (x, y)
    x_polar = r * sp.cos(theta)
    y_polar = r * sp.sin(theta)
    J_polar = sp.Matrix([
        [sp.diff(x_polar, r), sp.diff(x_polar, theta)],
        [sp.diff(y_polar, r), sp.diff(y_polar, theta)],
    ])
    det_polar = sp.simplify(J_polar.det())

    print("\nJacobian Computation")
    print(f"  Polar: (r,theta) -> (r*cos(theta), r*sin(theta))")
    print(f"  J = {J_polar.tolist()}")
    print(f"  |J| = {det_polar}")

    # Spherical coordinates: (r, theta, phi) -> (x, y, z)
    x_sph = r * sp.sin(phi) * sp.cos(theta)
    y_sph = r * sp.sin(phi) * sp.sin(theta)
    z_sph = r * sp.cos(phi)
    J_sph = sp.Matrix([
        [sp.diff(x_sph, r), sp.diff(x_sph, theta), sp.diff(x_sph, phi)],
        [sp.diff(y_sph, r), sp.diff(y_sph, theta), sp.diff(y_sph, phi)],
        [sp.diff(z_sph, r), sp.diff(z_sph, theta), sp.diff(z_sph, phi)],
    ])
    det_sph = sp.simplify(sp.trigsimp(J_sph.det()))

    print(f"\n  Spherical: (r,theta,phi) -> (r sin(phi) cos(theta), ...)")
    print(f"  |J| = {det_sph}")

    return det_polar, det_sph


# ---------------------------------------------------------------------------
# 5. Monte Carlo Integration
# ---------------------------------------------------------------------------
def monte_carlo_2d(f_np, x_range, y_range, n_samples=100000):
    """Estimate a 2D integral by random sampling (Monte Carlo method).

    The idea: the integral equals (area of bounding box) * (mean of f).
    Convergence is O(1/sqrt(N)) regardless of dimension — this makes
    Monte Carlo competitive in high dimensions where deterministic
    quadrature suffers from the curse of dimensionality.
    """
    x_lo, x_hi = x_range
    y_lo, y_hi = y_range
    area = (x_hi - x_lo) * (y_hi - y_lo)

    x = np.random.uniform(x_lo, x_hi, n_samples)
    y = np.random.uniform(y_lo, y_hi, n_samples)
    values = f_np(x, y)

    estimate = area * np.mean(values)
    std_error = area * np.std(values) / np.sqrt(n_samples)
    return estimate, std_error


def plot_mc_convergence(f_np, x_range, y_range, exact_val):
    """Show Monte Carlo convergence vs sample count."""
    sample_counts = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000]
    estimates = []
    errors = []

    for n in sample_counts:
        est, se = monte_carlo_2d(f_np, x_range, y_range, n)
        estimates.append(est)
        errors.append(abs(est - exact_val))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(sample_counts, errors, "ro-", ms=6, label="MC error")

    # Reference O(1/sqrt(N)) line
    n_arr = np.array(sample_counts, dtype=float)
    ax.loglog(n_arr, 0.5 / np.sqrt(n_arr), "k--", alpha=0.4,
              label="O(1/sqrt(N))")

    ax.set_xlabel("Number of samples")
    ax.set_ylabel("Absolute error")
    ax.set_title("Monte Carlo Integration Convergence")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("07_mc_convergence.png", dpi=100)
    plt.close()
    print("[Saved] 07_mc_convergence.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Multiple Integrals and Change of Variables")
    print("=" * 60)

    # --- Demo 1: Double integral (Cartesian) ---
    double_integral_cartesian()

    # --- Demo 2: Double integral (polar) ---
    double_integral_polar()

    # --- Demo 3: Triple integral ---
    triple_integral_demo()

    # --- Demo 4: Jacobian ---
    jacobian_demo()

    # --- Demo 5: Monte Carlo vs deterministic ---
    print("\nMonte Carlo Integration Comparison")
    f_np = lambda x, y: x ** 2 + y ** 2
    exact = 2 / 3
    mc_est, mc_se = monte_carlo_2d(f_np, (0, 1), (0, 1))
    det_est = integrate.dblquad(lambda y, x: x ** 2 + y ** 2, 0, 1, 0, 1)[0]
    print(f"  Exact       : {exact:.10f}")
    print(f"  Deterministic: {det_est:.10f}")
    print(f"  Monte Carlo : {mc_est:.10f} +/- {mc_se:.2e}")

    plot_mc_convergence(f_np, (0, 1), (0, 1), exact)
