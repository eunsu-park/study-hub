"""
Exercise Solutions: Lesson 08 - Parametric Curves and Polar Coordinates
Calculus and Differential Equations

Topics covered:
- Parametric velocity, speed, and arc length (exponential spiral)
- Epicycloid tangent lines and arc length
- Rose curve area in polar coordinates
- Logarithmic spiral arc length
- Satellite orbit (conic section in polar form)
"""

import numpy as np
import sympy as sp
from scipy import integrate as sci_int
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# Problem 1: Parametric Velocity and Arc Length
# ============================================================
def exercise_1():
    """
    Particle on x = e^t * cos(t), y = e^t * sin(t) for 0 <= t <= 2*pi.
    (a) Velocity and speed at t = 0
    (b) Show speed = sqrt(2)*e^t
    (c) Total arc length
    """
    print("=" * 60)
    print("Problem 1: Parametric Velocity and Arc Length")
    print("=" * 60)

    t = sp.Symbol('t')

    x_t = sp.exp(t) * sp.cos(t)
    y_t = sp.exp(t) * sp.sin(t)

    dx_dt = sp.diff(x_t, t)
    dy_dt = sp.diff(y_t, t)

    # (a) Velocity at t = 0
    vx_0 = dx_dt.subs(t, 0)
    vy_0 = dy_dt.subs(t, 0)
    speed_0 = sp.sqrt(vx_0**2 + vy_0**2)

    print(f"\n  x(t) = e^t * cos(t), y(t) = e^t * sin(t)")
    print(f"  dx/dt = {sp.expand(dx_dt)}")
    print(f"        = e^t*(cos(t) - sin(t))")
    print(f"  dy/dt = {sp.expand(dy_dt)}")
    print(f"        = e^t*(sin(t) + cos(t))")

    print(f"\n(a) At t = 0:")
    print(f"    velocity = ({vx_0}, {vy_0})")
    print(f"    speed = sqrt({vx_0}^2 + {vy_0}^2) = {sp.simplify(speed_0)}")

    # (b) Show speed = sqrt(2)*e^t
    speed_sq = sp.expand(dx_dt**2 + dy_dt**2)
    speed_sq_simplified = sp.simplify(speed_sq)
    speed_general = sp.sqrt(speed_sq_simplified)

    print(f"\n(b) Speed = sqrt((dx/dt)^2 + (dy/dt)^2)")
    print(f"    (dx/dt)^2 = e^(2t)*(cos(t) - sin(t))^2 = e^(2t)*(1 - sin(2t))")
    print(f"    (dy/dt)^2 = e^(2t)*(sin(t) + cos(t))^2 = e^(2t)*(1 + sin(2t))")
    print(f"    Sum = e^(2t) * 2 = 2*e^(2t)")
    print(f"    Speed = sqrt(2*e^(2t)) = sqrt(2)*e^t")
    print(f"    SymPy: speed^2 = {speed_sq_simplified}")
    print(f"    speed = {sp.simplify(speed_general)}")

    # (c) Arc length
    arc_length = sp.integrate(sp.sqrt(2) * sp.exp(t), (t, 0, 2*sp.pi))
    print(f"\n(c) Arc length = integral_0^{{2pi}} sqrt(2)*e^t dt")
    print(f"    = sqrt(2) * [e^t]_0^{{2pi}}")
    print(f"    = sqrt(2) * (e^(2pi) - 1)")
    print(f"    = {arc_length}")
    print(f"    = {float(arc_length):.6f}")

    # Visualization
    t_vals = np.linspace(0, 2*np.pi, 1000)
    x_vals = np.exp(t_vals) * np.cos(t_vals)
    y_vals = np.exp(t_vals) * np.sin(t_vals)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x_vals, y_vals, 'b-', linewidth=2, label='Exponential spiral')
    ax.plot(x_vals[0], y_vals[0], 'go', markersize=10, label='t=0')
    ax.plot(x_vals[-1], y_vals[-1], 'ro', markersize=10, label=f't=2pi')
    # Velocity arrow at t=0
    ax.arrow(1, 0, 0.5*1, 0.5*1, head_width=5, head_length=3, fc='green', ec='green')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Exponential Spiral $x = e^t\\cos t$, $y = e^t\\sin t$', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex08_exponential_spiral.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [Plot saved: ex08_exponential_spiral.png]")


# ============================================================
# Problem 2: Epicycloid
# ============================================================
def exercise_2():
    """
    Epicycloid: x = 2*cos(t) + cos(2t), y = 2*sin(t) + sin(2t)
    (a) dy/dx at t = pi/4
    (b) Horizontal tangent values on [0, 2pi)
    (c) Numerically compute arc length
    """
    print("\n" + "=" * 60)
    print("Problem 2: Epicycloid")
    print("=" * 60)

    t = sp.Symbol('t')

    x_t = 2*sp.cos(t) + sp.cos(2*t)
    y_t = 2*sp.sin(t) + sp.sin(2*t)

    dx_dt = sp.diff(x_t, t)
    dy_dt = sp.diff(y_t, t)

    print(f"\n  x(t) = 2*cos(t) + cos(2t)")
    print(f"  y(t) = 2*sin(t) + sin(2t)")
    print(f"  dx/dt = {dx_dt}")
    print(f"  dy/dt = {dy_dt}")

    # (a) dy/dx at t = pi/4
    dy_dx = dy_dt / dx_dt
    dy_dx_val = dy_dx.subs(t, sp.pi/4)
    print(f"\n(a) dy/dx = (dy/dt) / (dx/dt)")
    print(f"    At t = pi/4: dy/dx = {sp.simplify(dy_dx_val)}")
    print(f"    = {float(dy_dx_val):.10f}")

    # (b) Horizontal tangent: dy/dt = 0
    # dy/dt = 2*cos(t) + 2*cos(2t) = 0
    # cos(t) + cos(2t) = 0
    # cos(t) + 2*cos^2(t) - 1 = 0  (double angle formula)
    # 2*cos^2(t) + cos(t) - 1 = 0
    # (2*cos(t) - 1)(cos(t) + 1) = 0
    # cos(t) = 1/2 or cos(t) = -1
    horiz_t = sp.solve(dy_dt, t)
    print(f"\n(b) Horizontal tangent: dy/dt = 0")
    print(f"    2*cos(t) + 2*cos(2t) = 0")
    print(f"    cos(t) + cos(2t) = 0")
    print(f"    Using cos(2t) = 2*cos^2(t) - 1:")
    print(f"    2*cos^2(t) + cos(t) - 1 = 0")
    print(f"    (2*cos(t) - 1)(cos(t) + 1) = 0")
    print(f"    cos(t) = 1/2 => t = pi/3, 5*pi/3")
    print(f"    cos(t) = -1  => t = pi")

    # Verify that dx/dt != 0 at these points
    for t_val in [sp.pi/3, sp.pi, 5*sp.pi/3]:
        dx_val = dx_dt.subs(t, t_val)
        dy_val = dy_dt.subs(t, t_val)
        print(f"    t = {t_val}: dx/dt = {dx_val}, dy/dt = {dy_val}", end="")
        if dx_val != 0 and dy_val == 0:
            print(" => horizontal tangent")
        elif dx_val == 0 and dy_val == 0:
            print(" => singular point (cusp)")
        else:
            print()

    # (c) Arc length numerically
    speed_func = lambda tv: np.sqrt(
        (-2*np.sin(tv) - 2*np.sin(2*tv))**2 +
        (2*np.cos(tv) + 2*np.cos(2*tv))**2
    )
    arc_len_num, _ = sci_int.quad(speed_func, 0, 2*np.pi)
    print(f"\n(c) Arc length (numerical):")
    print(f"    L = {arc_len_num:.10f}")
    print(f"    L = {arc_len_num:.4f}")

    # The exact arc length of epicycloid with R=1, r=1 is 16
    print(f"    Exact (for this 3-cusped epicycloid with a=1): 16")

    # Visualization
    t_vals = np.linspace(0, 2*np.pi, 1000)
    x_vals = 2*np.cos(t_vals) + np.cos(2*t_vals)
    y_vals = 2*np.sin(t_vals) + np.sin(2*t_vals)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x_vals, y_vals, 'b-', linewidth=2, label='Epicycloid')
    for tv_sym in [sp.pi/3, sp.pi, 5*sp.pi/3]:
        tv = float(tv_sym)
        xp = 2*np.cos(tv) + np.cos(2*tv)
        yp = 2*np.sin(tv) + np.sin(2*tv)
        ax.plot(xp, yp, 'ro', markersize=8)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Epicycloid: $x = 2\\cos t + \\cos 2t$, $y = 2\\sin t + \\sin 2t$', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('ex08_epicycloid.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [Plot saved: ex08_epicycloid.png]")


# ============================================================
# Problem 3: Rose Curve Area
# ============================================================
def exercise_3():
    """
    Area enclosed by one petal of r = sin(3*theta).
    One petal spans 0 <= theta <= pi/3.
    """
    print("\n" + "=" * 60)
    print("Problem 3: Rose Curve Area")
    print("=" * 60)

    theta = sp.Symbol('theta')

    # Area = (1/2) * integral_0^{pi/3} r^2 d(theta)
    r = sp.sin(3*theta)
    area = sp.Rational(1, 2) * sp.integrate(r**2, (theta, 0, sp.pi/3))

    print(f"\n  r = sin(3*theta)")
    print(f"  One petal: 0 <= theta <= pi/3 (where r >= 0)")
    print(f"\n  Area = (1/2) integral_0^{{pi/3}} sin^2(3*theta) d(theta)")
    print(f"       = (1/2) integral_0^{{pi/3}} (1 - cos(6*theta))/2 d(theta)")
    print(f"       = (1/4) [theta - sin(6*theta)/6]_0^{{pi/3}}")
    print(f"       = (1/4) * (pi/3 - 0)")
    print(f"       = pi/12")
    print(f"\n  SymPy result: {area} = {float(area):.10f}")

    # Numerical verification
    area_num, _ = sci_int.quad(
        lambda th: 0.5 * np.sin(3*th)**2, 0, np.pi/3
    )
    print(f"  Numerical: {area_num:.10f}")

    # Plot the rose curve
    theta_vals = np.linspace(0, 2*np.pi, 1000)
    r_vals = np.sin(3*theta_vals)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    ax.plot(theta_vals, np.abs(r_vals), 'b-', linewidth=2)
    # Highlight one petal
    theta_petal = np.linspace(0, np.pi/3, 200)
    r_petal = np.sin(3*theta_petal)
    ax.fill(theta_petal, r_petal, alpha=0.3, color='orange', label=f'Area = $\\pi/12$')
    ax.set_title('Rose curve $r = \\sin(3\\theta)$', fontsize=14, pad=20)
    ax.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    plt.savefig('ex08_rose_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [Plot saved: ex08_rose_curve.png]")


# ============================================================
# Problem 4: Logarithmic Spiral Arc Length
# ============================================================
def exercise_4():
    """
    Arc length of r = e^(0.2*theta) from theta = 0 to theta = 4*pi.
    """
    print("\n" + "=" * 60)
    print("Problem 4: Logarithmic Spiral Arc Length")
    print("=" * 60)

    theta = sp.Symbol('theta')
    a = sp.Rational(1, 5)  # 0.2

    r = sp.exp(a * theta)
    dr_dtheta = sp.diff(r, theta)

    # Arc length in polar: L = integral sqrt(r^2 + (dr/dtheta)^2) d(theta)
    integrand = sp.sqrt(r**2 + dr_dtheta**2)
    integrand_simplified = sp.simplify(integrand)

    print(f"\n  r = e^(0.2*theta), dr/d(theta) = 0.2*e^(0.2*theta)")
    print(f"\n  L = integral_0^{{4pi}} sqrt(r^2 + (dr/dtheta)^2) d(theta)")
    print(f"    = integral_0^{{4pi}} sqrt(e^(0.4*theta) + 0.04*e^(0.4*theta)) d(theta)")
    print(f"    = integral_0^{{4pi}} e^(0.2*theta) * sqrt(1 + 0.04) d(theta)")
    print(f"    = sqrt(1.04) * integral_0^{{4pi}} e^(0.2*theta) d(theta)")
    print(f"    = sqrt(1.04) * [e^(0.2*theta)/0.2]_0^{{4pi}}")
    print(f"    = sqrt(1.04) * 5 * (e^(0.8*pi) - 1)")

    arc_length = sp.integrate(integrand_simplified, (theta, 0, 4*sp.pi))
    print(f"\n  SymPy exact: {arc_length}")
    print(f"  Numerical:   {float(arc_length):.10f}")

    # Verify with scipy
    arc_num, _ = sci_int.quad(
        lambda th: np.sqrt(np.exp(0.4*th) + 0.04*np.exp(0.4*th)),
        0, 4*np.pi
    )
    print(f"  scipy.quad:  {arc_num:.10f}")
    print(f"  Agreement:   {abs(float(arc_length) - arc_num):.2e}")

    # Visualization
    theta_vals = np.linspace(0, 4*np.pi, 1000)
    r_vals = np.exp(0.2 * theta_vals)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    ax.plot(theta_vals, r_vals, 'b-', linewidth=2, label=r'$r = e^{0.2\theta}$')
    ax.set_title(f'Log Spiral, Arc Length = {float(arc_length):.2f}', fontsize=14, pad=20)
    ax.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    plt.savefig('ex08_log_spiral.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [Plot saved: ex08_log_spiral.png]")


# ============================================================
# Problem 5: Satellite Orbit
# ============================================================
def exercise_5():
    """
    Polar orbit r = p / (1 + e*cos(theta)) with p=7000 km, e=0.1.
    (a) Periapsis and apoapsis
    (b) Plot the orbit
    (c) Area swept from theta=0 to theta=pi
    """
    print("\n" + "=" * 60)
    print("Problem 5: Satellite Orbit")
    print("=" * 60)

    theta = sp.Symbol('theta')
    p = 7000  # km (semi-latus rectum)
    ecc = 0.1  # eccentricity

    # (a) Periapsis (closest) and apoapsis (farthest)
    # r = p / (1 + e*cos(theta))
    # Periapsis: cos(theta) = 1 (max denominator) => r_min = p/(1+e)
    # Apoapsis: cos(theta) = -1 (min denominator) => r_max = p/(1-e)
    r_peri = p / (1 + ecc)
    r_apo = p / (1 - ecc)

    print(f"\n  Orbit: r = {p} / (1 + {ecc}*cos(theta))")
    print(f"\n(a) Periapsis and Apoapsis:")
    print(f"    Periapsis (theta=0, closest): r_min = p/(1+e) = {p}/(1+{ecc})")
    print(f"    = {r_peri:.4f} km")
    print(f"    Apoapsis (theta=pi, farthest): r_max = p/(1-e) = {p}/(1-{ecc})")
    print(f"    = {r_apo:.4f} km")

    # Semi-major axis
    a_orbit = p / (1 - ecc**2)
    print(f"\n    Semi-major axis: a = p/(1-e^2) = {a_orbit:.4f} km")

    # (b) Plot
    theta_vals = np.linspace(0, 2*np.pi, 1000)
    r_vals = p / (1 + ecc * np.cos(theta_vals))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Polar plot
    ax1 = fig.add_subplot(121, projection='polar')
    ax1.plot(theta_vals, r_vals, 'b-', linewidth=2)
    ax1.plot(0, r_peri, 'ro', markersize=8, label=f'Periapsis ({r_peri:.0f} km)')
    ax1.plot(np.pi, r_apo, 'go', markersize=8, label=f'Apoapsis ({r_apo:.0f} km)')
    ax1.set_title('Satellite Orbit (Polar)', fontsize=12, pad=20)
    ax1.legend(loc='lower left', fontsize=9)

    # Cartesian plot
    ax2 = fig.add_subplot(122)
    x_orbit = r_vals * np.cos(theta_vals)
    y_orbit = r_vals * np.sin(theta_vals)
    ax2.plot(x_orbit, y_orbit, 'b-', linewidth=2, label='Orbit')
    ax2.plot(0, 0, 'k*', markersize=15, label='Focus (planet)')
    ax2.plot(r_peri, 0, 'ro', markersize=8, label=f'Periapsis')
    ax2.plot(-r_apo, 0, 'go', markersize=8, label=f'Apoapsis')
    ax2.set_xlabel('x (km)', fontsize=12)
    ax2.set_ylabel('y (km)', fontsize=12)
    ax2.set_title('Satellite Orbit (Cartesian)', fontsize=12)
    ax2.set_aspect('equal')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex08_satellite_orbit.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n(b) [Plot saved: ex08_satellite_orbit.png]")

    # (c) Area swept from theta=0 to theta=pi (Kepler's second law)
    # A = (1/2) integral_0^pi r^2 d(theta)
    theta_sym = sp.Symbol('theta')
    r_sym = p / (1 + ecc * sp.cos(theta_sym))
    area = sp.Rational(1, 2) * sp.integrate(r_sym**2, (theta_sym, 0, sp.pi))
    area_simplified = sp.simplify(area)

    print(f"\n(c) Area swept from theta=0 to theta=pi:")
    print(f"    A = (1/2) integral_0^pi [p/(1+e*cos(theta))]^2 d(theta)")
    print(f"    = {float(area_simplified):.2f} km^2")

    # Numerical verification
    area_num, _ = sci_int.quad(
        lambda th: 0.5 * (p / (1 + ecc * np.cos(th)))**2, 0, np.pi
    )
    print(f"    Numerical: {area_num:.2f} km^2")

    # Total orbital area for comparison
    total_area = np.pi * a_orbit * a_orbit * np.sqrt(1 - ecc**2)
    print(f"\n    Total orbital area (pi*a*b): {total_area:.2f} km^2")
    print(f"    Half-orbit area fraction: {float(area_simplified)/total_area:.4f}")
    print(f"    (If e=0 this would be exactly 0.5; with e>0 the fraction differs)")


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
    print("All exercises for Lesson 08 completed.")
    print("=" * 60)
