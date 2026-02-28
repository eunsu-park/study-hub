"""
Exercise Solutions: Lesson 06 - Applications of Integration
Calculus and Differential Equations

Topics covered:
- Area between curves (sin x, cos x)
- Volume by disk/washer method
- Volume by shell method
- Arc length computation
- Physical application (work to pump water from a conical tank)
"""

import numpy as np
import sympy as sp
from scipy import integrate as sci_int
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# Problem 1: Area Between Curves
# ============================================================
def exercise_1():
    """
    Find the area enclosed by y = sin(x) and y = cos(x)
    between x = 0 and x = pi/2.
    """
    print("=" * 60)
    print("Problem 1: Area Between Curves")
    print("=" * 60)

    x = sp.Symbol('x')

    # Find intersection: sin(x) = cos(x) => tan(x) = 1 => x = pi/4
    intersection = sp.pi / 4
    print(f"\n  Curves: y = sin(x) and y = cos(x) on [0, pi/2]")
    print(f"  Intersection: sin(x) = cos(x) => x = pi/4")

    # On [0, pi/4]: cos(x) >= sin(x)
    # On [pi/4, pi/2]: sin(x) >= cos(x)
    area1 = sp.integrate(sp.cos(x) - sp.sin(x), (x, 0, sp.pi/4))
    area2 = sp.integrate(sp.sin(x) - sp.cos(x), (x, sp.pi/4, sp.pi/2))
    total_area = area1 + area2

    print(f"\n  On [0, pi/4]: cos(x) >= sin(x)")
    print(f"    Area_1 = integral_0^{{pi/4}} (cos(x) - sin(x)) dx = {area1}")
    print(f"           = {sp.simplify(area1)} = {float(area1):.10f}")
    print(f"\n  On [pi/4, pi/2]: sin(x) >= cos(x)")
    print(f"    Area_2 = integral_{{pi/4}}^{{pi/2}} (sin(x) - cos(x)) dx = {area2}")
    print(f"           = {sp.simplify(area2)} = {float(area2):.10f}")
    print(f"\n  Total area = {sp.simplify(total_area)} = {float(total_area):.10f}")
    print(f"  = 2*(sqrt(2) - 1) = 2*sqrt(2) - 2")

    # Visualization
    x_vals = np.linspace(0, np.pi/2, 500)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_vals, np.sin(x_vals), 'b-', linewidth=2, label=r'$y = \sin x$')
    ax.plot(x_vals, np.cos(x_vals), 'r-', linewidth=2, label=r'$y = \cos x$')
    x1 = np.linspace(0, np.pi/4, 200)
    x2 = np.linspace(np.pi/4, np.pi/2, 200)
    ax.fill_between(x1, np.sin(x1), np.cos(x1), alpha=0.3, color='green', label='Area 1')
    ax.fill_between(x2, np.cos(x2), np.sin(x2), alpha=0.3, color='orange', label='Area 2')
    ax.axvline(x=np.pi/4, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Area Between sin(x) and cos(x)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex06_area_between_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [Plot saved: ex06_area_between_curves.png]")


# ============================================================
# Problem 2: Volume by Disk/Washer
# ============================================================
def exercise_2():
    """
    Region bounded by y = sqrt(x), y = 0, x = 4, rotated about x-axis.
    (a) Disk method
    (b) Verify with Python
    """
    print("\n" + "=" * 60)
    print("Problem 2: Volume by Disk/Washer")
    print("=" * 60)

    x = sp.Symbol('x')

    # (a) Disk method: V = pi * integral_0^4 [sqrt(x)]^2 dx = pi * integral_0^4 x dx
    integrand = sp.sqrt(x)**2  # = x
    volume = sp.pi * sp.integrate(integrand, (x, 0, 4))
    print(f"\n(a) Disk method:")
    print(f"    Region: y = sqrt(x), y = 0, x = 4, rotated about x-axis")
    print(f"    V = pi * integral_0^4 [sqrt(x)]^2 dx")
    print(f"      = pi * integral_0^4 x dx")
    print(f"      = pi * [x^2/2]_0^4")
    print(f"      = pi * (16/2 - 0)")
    print(f"      = {volume}")
    print(f"      = {float(volume):.10f}")

    # (b) Numerical verification
    vol_num, _ = sci_int.quad(lambda xv: np.pi * xv, 0, 4)
    print(f"\n(b) Numerical verification:")
    print(f"    scipy.quad result: {vol_num:.10f}")
    print(f"    Exact 8*pi:       {8*np.pi:.10f}")
    print(f"    Match: {abs(vol_num - 8*np.pi) < 1e-10}")


# ============================================================
# Problem 3: Volume by Shell Method
# ============================================================
def exercise_3():
    """
    Region bounded by y = x - x^2 and y = 0, rotated about y-axis.
    Shell method, then verify with washer method.
    """
    print("\n" + "=" * 60)
    print("Problem 3: Volume by Shell Method")
    print("=" * 60)

    x, y = sp.symbols('x y', positive=True)

    # y = x - x^2 = x(1-x), roots at x=0 and x=1
    # Shell method: V = 2*pi * integral_0^1 x * (x - x^2) dx
    shell_integrand = x * (x - x**2)
    V_shell = 2 * sp.pi * sp.integrate(shell_integrand, (x, 0, 1))
    print(f"\n  Shell method (rotating about y-axis):")
    print(f"  y = x - x^2, roots at x = 0 and x = 1")
    print(f"  V = 2*pi * integral_0^1 x*(x - x^2) dx")
    print(f"    = 2*pi * integral_0^1 (x^2 - x^3) dx")
    print(f"    = 2*pi * [x^3/3 - x^4/4]_0^1")
    print(f"    = 2*pi * (1/3 - 1/4)")
    print(f"    = 2*pi * 1/12")
    print(f"    = {V_shell}")
    print(f"    = {float(V_shell):.10f}")

    # Washer method (about y-axis, integrate wrt y):
    # y = x - x^2 => x^2 - x + y = 0 => x = (1 +/- sqrt(1 - 4y))/2
    # Outer radius: R(y) = (1 + sqrt(1-4y))/2
    # Inner radius: r(y) = (1 - sqrt(1-4y))/2
    # Max y: vertex at x = 1/2, y = 1/4
    R = (1 + sp.sqrt(1 - 4*y)) / 2
    r = (1 - sp.sqrt(1 - 4*y)) / 2
    washer_integrand = R**2 - r**2
    V_washer = sp.pi * sp.integrate(washer_integrand, (y, 0, sp.Rational(1, 4)))
    V_washer_simplified = sp.simplify(V_washer)

    print(f"\n  Washer method (rotating about y-axis, integrating wrt y):")
    print(f"  Outer: R(y) = (1 + sqrt(1-4y))/2")
    print(f"  Inner: r(y) = (1 - sqrt(1-4y))/2")
    print(f"  y ranges from 0 to 1/4 (vertex of parabola)")
    print(f"  V = pi * integral_0^{{1/4}} (R^2 - r^2) dy")
    print(f"    = {V_washer_simplified}")
    print(f"    = {float(V_washer_simplified):.10f}")
    print(f"\n  Both methods agree: {sp.simplify(V_shell - V_washer_simplified) == 0}")


# ============================================================
# Problem 4: Arc Length
# ============================================================
def exercise_4():
    """
    Arc length of y = x^2/2 - ln(x)/4 from x = 1 to x = 2.
    Show that the integral simplifies.
    """
    print("\n" + "=" * 60)
    print("Problem 4: Arc Length")
    print("=" * 60)

    x = sp.Symbol('x', positive=True)

    f = x**2 / 2 - sp.ln(x) / 4
    f_prime = sp.diff(f, x)
    print(f"\n  f(x) = x^2/2 - ln(x)/4")
    print(f"  f'(x) = {f_prime}")

    # Simplify 1 + [f'(x)]^2
    integrand_squared = 1 + f_prime**2
    integrand_squared_simplified = sp.simplify(sp.expand(integrand_squared))
    print(f"\n  1 + [f'(x)]^2 = 1 + (x - 1/(4x))^2")
    print(f"    = 1 + x^2 - 1/2 + 1/(16x^2)")
    print(f"    = x^2 + 1/2 + 1/(16x^2)")
    print(f"    = (x + 1/(4x))^2")
    print(f"\n  This is a perfect square!")
    print(f"  sqrt(1 + [f']^2) = x + 1/(4x)")

    # So the arc length integral simplifies:
    # L = integral_1^2 (x + 1/(4x)) dx = [x^2/2 + ln(x)/4]_1^2
    arc_length = sp.integrate(x + 1/(4*x), (x, 1, 2))
    print(f"\n  L = integral_1^2 (x + 1/(4x)) dx")
    print(f"    = [x^2/2 + ln(x)/4]_1^2")
    print(f"    = (4/2 + ln(2)/4) - (1/2 + 0)")
    print(f"    = 3/2 + ln(2)/4")
    print(f"    = {arc_length}")
    print(f"    = {float(arc_length):.10f}")

    # Numerical verification
    from scipy import integrate as sci
    num_length, _ = sci.quad(
        lambda xv: np.sqrt(1 + (xv - 1/(4*xv))**2), 1, 2
    )
    print(f"\n  Numerical verification: {num_length:.10f}")


# ============================================================
# Problem 5: Physical Application (Work)
# ============================================================
def exercise_5():
    """
    Conical tank (point down): height 6 m, top radius 3 m.
    Filled with water (rho = 1000 kg/m^3).
    Compute work to pump all water out the top.
    """
    print("\n" + "=" * 60)
    print("Problem 5: Physical Application (Work)")
    print("=" * 60)

    y = sp.Symbol('y')

    # Cone with vertex at bottom (y=0), top at y=6, radius 3 at top
    # At height y: radius r(y) = y/2 (by similar triangles: r/y = 3/6)
    # Volume of thin slice at height y: dV = pi * r(y)^2 * dy = pi*(y/2)^2 dy
    # Weight of slice: dW = rho * g * dV = 1000 * 9.8 * pi * y^2/4 * dy
    # Distance to lift to top: (6 - y)
    # Work for slice: rho * g * pi * (y^2/4) * (6 - y) * dy

    rho = 1000  # kg/m^3
    g = sp.Rational(98, 10)  # 9.8 m/s^2
    H = 6  # height in meters

    # r(y) = y/2, so cross-section area = pi*(y/2)^2 = pi*y^2/4
    # Work = integral_0^6 rho*g*pi*(y^2/4)*(6-y) dy

    integrand = rho * g * sp.pi * (y**2 / 4) * (H - y)
    work = sp.integrate(integrand, (y, 0, H))

    print(f"\n  Cone: height H = {H} m, top radius R = 3 m")
    print(f"  At height y: radius r(y) = y/2 (similar triangles)")
    print(f"  Slice volume: dV = pi*(y/2)^2 dy = pi*y^2/4 dy")
    print(f"  Distance to lift: (6 - y)")
    print(f"\n  W = integral_0^6 rho*g*pi*(y^2/4)*(6-y) dy")
    print(f"    = rho*g*pi/4 * integral_0^6 (6y^2 - y^3) dy")

    # Evaluate the polynomial integral
    poly_integral = sp.integrate(6*y**2 - y**3, (y, 0, 6))
    print(f"    = rho*g*pi/4 * [2y^3 - y^4/4]_0^6")
    print(f"    = rho*g*pi/4 * (2*216 - 1296/4)")
    print(f"    = rho*g*pi/4 * (432 - 324)")
    print(f"    = rho*g*pi/4 * 108")
    print(f"    = {rho} * {float(g)} * pi * 27")

    work_simplified = sp.simplify(work)
    print(f"\n  W = {work_simplified}")
    print(f"    = {float(work_simplified):.2f} J")
    print(f"    = {float(work_simplified)/1000:.2f} kJ")

    # Numerical verification
    work_num, _ = sci_int.quad(
        lambda yv: 1000 * 9.8 * np.pi * (yv**2 / 4) * (6 - yv), 0, 6
    )
    print(f"\n  Numerical verification: {work_num:.2f} J")


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
    print("All exercises for Lesson 06 completed.")
    print("=" * 60)
