"""
Exercise Solutions: Lesson 10 - Multiple Integrals
Calculus and Differential Equations

Topics covered:
- Double integral in both orders
- Switching order of integration
- Polar coordinate integration
- Volume under paraboloid
- Mass and center of mass of hemisphere (spherical coordinates)
"""

import numpy as np
import sympy as sp
from scipy import integrate as sci_int
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# Problem 1: Double Integral in Both Orders
# ============================================================
def exercise_1():
    """
    Evaluate double_integral_R (x^2 + y) dA where R = [0,1] x [0,2].
    Verify by computing in both orders.
    """
    print("=" * 60)
    print("Problem 1: Double Integral in Both Orders")
    print("=" * 60)

    x, y = sp.symbols('x y')

    integrand = x**2 + y

    # Order 1: dy dx
    inner_1 = sp.integrate(integrand, (y, 0, 2))
    result_1 = sp.integrate(inner_1, (x, 0, 1))
    print(f"\n  integral_0^1 integral_0^2 (x^2 + y) dy dx")
    print(f"  Inner (dy): integral_0^2 (x^2 + y) dy = {inner_1}")
    print(f"  Outer (dx): integral_0^1 ({inner_1}) dx = {result_1}")

    # Order 2: dx dy
    inner_2 = sp.integrate(integrand, (x, 0, 1))
    result_2 = sp.integrate(inner_2, (y, 0, 2))
    print(f"\n  integral_0^2 integral_0^1 (x^2 + y) dx dy")
    print(f"  Inner (dx): integral_0^1 (x^2 + y) dx = {inner_2}")
    print(f"  Outer (dy): integral_0^2 ({inner_2}) dy = {result_2}")

    print(f"\n  Both orders give: {result_1} = {float(result_1):.10f}")
    print(f"  Match: {result_1 == result_2}")

    # Numerical verification
    result_num, _ = sci_int.dblquad(
        lambda y_val, x_val: x_val**2 + y_val,
        0, 1, 0, 2
    )
    print(f"  Numerical: {result_num:.10f}")


# ============================================================
# Problem 2: Switching Order of Integration
# ============================================================
def exercise_2():
    """
    Evaluate integral_0^1 integral_{sqrt(y)}^1 sin(x^2) dx dy
    by switching the order of integration.
    """
    print("\n" + "=" * 60)
    print("Problem 2: Switching Order of Integration")
    print("=" * 60)

    x, y = sp.symbols('x y')

    print(f"\n  Original: integral_0^1 integral_{{sqrt(y)}}^1 sin(x^2) dx dy")
    print(f"\n  Sketch the region:")
    print(f"    x ranges from sqrt(y) to 1")
    print(f"    y ranges from 0 to 1")
    print(f"    Equivalently: y ranges from 0 to x^2, x from 0 to 1")
    print(f"\n  The region is: 0 <= y <= x^2, 0 <= x <= 1")
    print(f"  (Below the parabola y = x^2, in the unit square)")

    print(f"\n  Switched order: integral_0^1 integral_0^{{x^2}} sin(x^2) dy dx")

    # Inner integral (dy)
    inner = sp.integrate(sp.sin(x**2), (y, 0, x**2))
    print(f"  Inner (dy): integral_0^{{x^2}} sin(x^2) dy = {inner}")

    # Outer integral (dx)
    result = sp.integrate(inner, (x, 0, 1))
    print(f"  Outer (dx): integral_0^1 x^2*sin(x^2) dx")
    print(f"  Let u = x^2, du = 2x dx... but we have x^2*sin(x^2) dx")
    print(f"  Actually: let u = x^2, then x^2*sin(x^2) = u*sin(u)*du/(2*sqrt(u))")
    print(f"  Better: direct SymPy evaluation = {result}")
    print(f"  = {float(result):.10f}")

    # Numerical verification
    result_num, _ = sci_int.dblquad(
        lambda y_val, x_val: np.sin(x_val**2),
        0, 1,
        lambda x_val: 0,
        lambda x_val: x_val**2
    )
    print(f"\n  Numerical verification: {result_num:.10f}")

    # Note: the original order integral_0^1 integral_{sqrt(y)}^1 sin(x^2) dx dy
    # cannot be evaluated in closed form because integral sin(x^2) dx = Fresnel integral
    result_num_orig, _ = sci_int.dblquad(
        lambda x_val, y_val: np.sin(x_val**2),
        0, 1,
        lambda y_val: np.sqrt(y_val),
        lambda y_val: 1.0
    )
    print(f"  Original order (numerical): {result_num_orig:.10f}")
    print(f"  Both orders agree: {abs(result_num - result_num_orig) < 1e-8}")

    # Visualization
    fig, ax = plt.subplots(figsize=(7, 7))
    x_vals = np.linspace(0, 1, 100)
    ax.fill_between(x_vals, 0, x_vals**2, alpha=0.3, color='blue', label=r'Region: $0 \le y \le x^2$')
    ax.plot(x_vals, x_vals**2, 'b-', linewidth=2, label=r'$y = x^2$')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Integration Region', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex10_switch_order.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [Plot saved: ex10_switch_order.png]")


# ============================================================
# Problem 3: Polar Coordinate Integration
# ============================================================
def exercise_3():
    """
    Evaluate double_integral_D (x^2+y^2)^(3/2) dA
    where D is the annulus 1 <= x^2+y^2 <= 4.
    """
    print("\n" + "=" * 60)
    print("Problem 3: Polar Coordinate Integration")
    print("=" * 60)

    r, theta = sp.symbols('r theta', positive=True)

    # In polar: x^2+y^2 = r^2, so (x^2+y^2)^(3/2) = r^3
    # dA = r dr d(theta)
    # Annulus: 1 <= r <= 2, 0 <= theta <= 2*pi
    integrand = r**3 * r  # r^3 * r (from Jacobian)

    inner = sp.integrate(integrand, (r, 1, 2))
    result = sp.integrate(inner, (theta, 0, 2*sp.pi))

    print(f"\n  integral_D (x^2+y^2)^(3/2) dA, D: annulus 1 <= r <= 2")
    print(f"\n  In polar coordinates:")
    print(f"  (x^2+y^2)^(3/2) = r^3, dA = r*dr*d(theta)")
    print(f"  = integral_0^{{2pi}} integral_1^2 r^3 * r dr d(theta)")
    print(f"  = integral_0^{{2pi}} integral_1^2 r^4 dr d(theta)")
    print(f"\n  Inner: integral_1^2 r^4 dr = [r^5/5]_1^2 = {inner}")
    print(f"  Outer: integral_0^{{2pi}} {inner} d(theta) = {result}")
    print(f"  = {float(result):.10f}")

    # Numerical verification
    result_num, _ = sci_int.dblquad(
        lambda r_val, th_val: r_val**4,
        0, 2*np.pi,
        1, 2
    )
    print(f"\n  Numerical: {result_num:.10f}")


# ============================================================
# Problem 4: Volume Under Paraboloid
# ============================================================
def exercise_4():
    """
    Volume bounded above by z = 4 - x^2 - y^2, below by z = 0.
    (a) Cartesian
    (b) Polar (easier)
    """
    print("\n" + "=" * 60)
    print("Problem 4: Volume Under Paraboloid")
    print("=" * 60)

    x, y, r, theta = sp.symbols('x y r theta')

    # z = 4 - x^2 - y^2 >= 0 when x^2 + y^2 <= 4 (disk of radius 2)

    # (a) Cartesian
    print(f"\n  z = 4 - x^2 - y^2, z >= 0 when x^2+y^2 <= 4")
    print(f"\n(a) Cartesian:")
    print(f"  V = integral_{{-2}}^2 integral_{{-sqrt(4-x^2)}}^{{sqrt(4-x^2)}} (4-x^2-y^2) dy dx")

    inner_cart = sp.integrate(4 - x**2 - y**2, (y, -sp.sqrt(4 - x**2), sp.sqrt(4 - x**2)))
    inner_simplified = sp.simplify(inner_cart)
    vol_cart = sp.integrate(inner_simplified, (x, -2, 2))

    print(f"  Inner: {inner_simplified}")
    print(f"  Outer: {vol_cart} = {float(vol_cart):.10f}")

    # (b) Polar (much easier)
    print(f"\n(b) Polar coordinates:")
    print(f"  V = integral_0^{{2pi}} integral_0^2 (4 - r^2) * r dr d(theta)")

    inner_polar = sp.integrate((4 - r**2) * r, (r, 0, 2))
    vol_polar = sp.integrate(inner_polar, (theta, 0, 2*sp.pi))

    print(f"  Inner: integral_0^2 (4r - r^3) dr = [2r^2 - r^4/4]_0^2 = {inner_polar}")
    print(f"  Outer: 2*pi * {inner_polar} = {vol_polar}")
    print(f"  = {float(vol_polar):.10f}")
    print(f"\n  Both methods agree: {sp.simplify(vol_cart - vol_polar) == 0}")
    print(f"  (Polar is significantly easier -- no square roots in limits!)")


# ============================================================
# Problem 5: Mass and Center of Mass of Hemisphere
# ============================================================
def exercise_5():
    """
    Solid hemisphere x^2+y^2+z^2 <= R^2, z >= 0, density rho = z.
    Use spherical coordinates. Verify with scipy.
    """
    print("\n" + "=" * 60)
    print("Problem 5: Mass and Center of Mass of Hemisphere")
    print("=" * 60)

    rho_s, theta, phi, R = sp.symbols('rho theta phi R', positive=True)

    # Spherical coordinates: x = rho*sin(phi)*cos(theta), y = rho*sin(phi)*sin(theta), z = rho*cos(phi)
    # dV = rho^2 * sin(phi) * d(rho) d(phi) d(theta)
    # Hemisphere: 0 <= rho <= R, 0 <= phi <= pi/2, 0 <= theta <= 2*pi
    # Density: rho(x,y,z) = z = rho_s * cos(phi)

    # Mass = integral rho_density * dV
    density = rho_s * sp.cos(phi)
    dV = rho_s**2 * sp.sin(phi)

    mass_integrand = density * dV
    mass_inner = sp.integrate(mass_integrand, (rho_s, 0, R))
    mass_middle = sp.integrate(mass_inner, (phi, 0, sp.pi/2))
    mass = sp.integrate(mass_middle, (theta, 0, 2*sp.pi))

    print(f"\n  Hemisphere: x^2+y^2+z^2 <= R^2, z >= 0")
    print(f"  Density: rho(x,y,z) = z = rho*cos(phi) in spherical")
    print(f"  dV = rho^2 * sin(phi) * d(rho) d(phi) d(theta)")
    print(f"\n  Mass = integral rho_density * dV")
    print(f"  = integral_0^{{2pi}} integral_0^{{pi/2}} integral_0^R (rho*cos(phi)) * rho^2*sin(phi) d(rho) d(phi) d(theta)")
    print(f"  = integral_0^{{2pi}} integral_0^{{pi/2}} integral_0^R rho^3*cos(phi)*sin(phi) d(rho) d(phi) d(theta)")
    print(f"\n  Inner (rho): {mass_inner}")
    print(f"  Middle (phi): {mass_middle}")
    print(f"  Outer (theta): {mass}")
    print(f"  Mass = {sp.simplify(mass)}")

    # Center of mass: by symmetry, x_bar = y_bar = 0
    # z_bar = (1/M) * integral z * rho_density * dV
    # z = rho_s * cos(phi), so z * density = rho_s^2 * cos^2(phi)
    z_integrand = rho_s**2 * sp.cos(phi)**2 * dV  # = rho_s^4 * cos^2(phi) * sin(phi)
    z_inner = sp.integrate(z_integrand, (rho_s, 0, R))
    z_middle = sp.integrate(z_inner, (phi, 0, sp.pi/2))
    z_moment = sp.integrate(z_middle, (theta, 0, 2*sp.pi))

    z_bar = sp.simplify(z_moment / mass)

    print(f"\n  By symmetry: x_bar = y_bar = 0")
    print(f"  z_bar = (1/M) * integral z * rho * dV")
    print(f"  z moment = {sp.simplify(z_moment)}")
    print(f"  z_bar = {z_bar}")

    # Numerical verification with R = 1
    R_val = 1.0
    mass_num = float(mass.subs(R, 1))
    z_bar_num = float(z_bar.subs(R, 1))

    print(f"\n  For R = 1:")
    print(f"    Mass = {mass_num:.10f}")
    print(f"    z_bar = {z_bar_num:.10f}")

    # Verify with scipy tplquad
    from scipy.integrate import tplquad

    # Triple integral in spherical coordinates
    mass_scipy, _ = tplquad(
        lambda rho_v, phi_v, theta_v: (rho_v * np.cos(phi_v)) * rho_v**2 * np.sin(phi_v),
        0, 2*np.pi,       # theta
        0, np.pi/2,       # phi
        0, R_val           # rho
    )

    z_moment_scipy, _ = tplquad(
        lambda rho_v, phi_v, theta_v: (rho_v**2 * np.cos(phi_v)**2) * rho_v**2 * np.sin(phi_v),
        0, 2*np.pi,
        0, np.pi/2,
        0, R_val
    )

    z_bar_scipy = z_moment_scipy / mass_scipy

    print(f"\n  scipy.tplquad verification (R=1):")
    print(f"    Mass = {mass_scipy:.10f}")
    print(f"    z_bar = {z_bar_scipy:.10f}")
    print(f"    Mass agreement: {abs(mass_num - mass_scipy):.2e}")
    print(f"    z_bar agreement: {abs(z_bar_num - z_bar_scipy):.2e}")


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
    print("All exercises for Lesson 10 completed.")
    print("=" * 60)
