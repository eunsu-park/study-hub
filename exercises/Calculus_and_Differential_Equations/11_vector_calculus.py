"""
Exercise Solutions: Lesson 11 - Vector Calculus
Calculus and Differential Equations

Topics covered:
- Green's theorem verification
- Conservative fields and potential functions
- Divergence theorem
- Stokes' theorem
- Gauss's law for inverse-square field
"""

import numpy as np
import sympy as sp
from scipy import integrate as sci_int
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# Problem 1: Green's Theorem
# ============================================================
def exercise_1():
    """
    F = (x^2*y, x*y^2). Verify Green's theorem on the triangle
    with vertices (0,0), (1,0), (0,1).
    """
    print("=" * 60)
    print("Problem 1: Green's Theorem Verification")
    print("=" * 60)

    x, y, t = sp.symbols('x y t')

    P = x**2 * y
    Q = x * y**2

    # Green's theorem: oint_C P dx + Q dy = double_integral_D (dQ/dx - dP/dy) dA
    dQ_dx = sp.diff(Q, x)
    dP_dy = sp.diff(P, y)
    integrand = dQ_dx - dP_dy

    print(f"\n  F = (P, Q) = (x^2*y, x*y^2)")
    print(f"  dQ/dx = {dQ_dx}, dP/dy = {dP_dy}")
    print(f"  dQ/dx - dP/dy = {integrand}")

    # Double integral over the triangle: 0 <= x <= 1, 0 <= y <= 1-x
    double_int = sp.integrate(
        sp.integrate(integrand, (y, 0, 1 - x)),
        (x, 0, 1)
    )
    print(f"\n  Double integral:")
    print(f"  integral_0^1 integral_0^{{1-x}} ({integrand}) dy dx = {double_int}")

    # Line integral: three sides of the triangle
    # Side 1: (0,0) -> (1,0), parametrize x=t, y=0, 0<=t<=1
    P1 = P.subs([(x, t), (y, 0)]) * 1 + Q.subs([(x, t), (y, 0)]) * 0
    side1 = sp.integrate(P1, (t, 0, 1))

    # Side 2: (1,0) -> (0,1), parametrize x=1-t, y=t, 0<=t<=1
    dx2 = -1  # dx/dt
    dy2 = 1   # dy/dt
    P2 = P.subs([(x, 1 - t), (y, t)]) * dx2 + Q.subs([(x, 1 - t), (y, t)]) * dy2
    side2 = sp.integrate(P2, (t, 0, 1))

    # Side 3: (0,1) -> (0,0), parametrize x=0, y=1-t, 0<=t<=1
    P3 = P.subs([(x, 0), (y, 1 - t)]) * 0 + Q.subs([(x, 0), (y, 1 - t)]) * (-1)
    side3 = sp.integrate(P3, (t, 0, 1))

    line_int = side1 + side2 + side3

    print(f"\n  Line integral:")
    print(f"    Side 1 [(0,0)->(1,0)]: {side1}")
    print(f"    Side 2 [(1,0)->(0,1)]: {side2}")
    print(f"    Side 3 [(0,1)->(0,0)]: {side3}")
    print(f"    Total: {line_int}")

    print(f"\n  Green's theorem verified: {double_int} = {line_int} => {sp.simplify(double_int - line_int) == 0}")


# ============================================================
# Problem 2: Conservative Field and Potential
# ============================================================
def exercise_2():
    """
    F = (2xy + z^2, x^2 + 2yz, 2xz + y^2).
    Check if conservative, find potential, evaluate line integral.
    """
    print("\n" + "=" * 60)
    print("Problem 2: Conservative Field and Potential")
    print("=" * 60)

    x, y, z = sp.symbols('x y z')

    F1 = 2*x*y + z**2
    F2 = x**2 + 2*y*z
    F3 = 2*x*z + y**2

    # Check curl F = 0
    curl_x = sp.diff(F3, y) - sp.diff(F2, z)
    curl_y = sp.diff(F1, z) - sp.diff(F3, x)
    curl_z = sp.diff(F2, x) - sp.diff(F1, y)

    print(f"\n  F = ({F1}, {F2}, {F3})")
    print(f"\n  curl(F):")
    print(f"    (dF3/dy - dF2/dz) = {sp.diff(F3, y)} - {sp.diff(F2, z)} = {curl_x}")
    print(f"    (dF1/dz - dF3/dx) = {sp.diff(F1, z)} - {sp.diff(F3, x)} = {curl_y}")
    print(f"    (dF2/dx - dF1/dy) = {sp.diff(F2, x)} - {sp.diff(F1, y)} = {curl_z}")
    print(f"    curl(F) = ({curl_x}, {curl_y}, {curl_z}) = (0, 0, 0)")
    print(f"    F is CONSERVATIVE")

    # Find potential: phi_x = F1, phi_y = F2, phi_z = F3
    # phi = integral F1 dx = x^2*y + x*z^2 + g(y,z)
    # phi_y = x^2 + g_y(y,z) = x^2 + 2yz => g_y = 2yz => g = y^2*z + h(z)
    # phi_z = 2xz + y^2 + h'(z) = 2xz + y^2 => h'(z) = 0 => h = C
    # phi = x^2*y + x*z^2 + y^2*z
    phi = x**2 * y + x*z**2 + y**2 * z

    print(f"\n  Finding potential phi:")
    print(f"    phi_x = {F1} => phi = x^2*y + x*z^2 + g(y,z)")
    print(f"    phi_y = x^2 + g_y = {F2} => g_y = 2yz => g = y^2*z + h(z)")
    print(f"    phi_z = 2xz + y^2 + h'(z) = {F3} => h'(z) = 0")
    print(f"    phi(x,y,z) = {phi}")

    # Verify
    assert sp.diff(phi, x) == F1
    assert sp.diff(phi, y) == F2
    assert sp.diff(phi, z) == F3
    print(f"    Verification: grad(phi) = F [PASSED]")

    # Line integral from (0,0,0) to (1,2,3)
    result = phi.subs([(x, 1), (y, 2), (z, 3)]) - phi.subs([(x, 0), (y, 0), (z, 0)])
    print(f"\n  Line integral from (0,0,0) to (1,2,3):")
    print(f"    = phi(1,2,3) - phi(0,0,0)")
    print(f"    = {phi.subs([(x, 1), (y, 2), (z, 3)])} - {phi.subs([(x, 0), (y, 0), (z, 0)])}")
    print(f"    = {result}")


# ============================================================
# Problem 3: Divergence Theorem
# ============================================================
def exercise_3():
    """
    F = (x^3, y^3, z^3), S = sphere x^2+y^2+z^2 = 4.
    Use divergence theorem.
    """
    print("\n" + "=" * 60)
    print("Problem 3: Divergence Theorem")
    print("=" * 60)

    x, y, z = sp.symbols('x y z')
    rho, phi, theta = sp.symbols('rho phi theta', positive=True)

    F1, F2, F3 = x**3, y**3, z**3

    div_F = sp.diff(F1, x) + sp.diff(F2, y) + sp.diff(F3, z)
    print(f"\n  F = (x^3, y^3, z^3)")
    print(f"  div(F) = {div_F}")
    print(f"         = 3(x^2 + y^2 + z^2) = 3*rho^2 in spherical")

    # By divergence theorem: flux = triple_integral div(F) dV
    # In spherical: div(F) = 3*rho^2, dV = rho^2*sin(phi)*d(rho)*d(phi)*d(theta)
    # Sphere of radius 2: rho from 0 to 2
    integrand = 3 * rho**2 * rho**2 * sp.sin(phi)
    inner = sp.integrate(integrand, (rho, 0, 2))
    middle = sp.integrate(inner, (phi, 0, sp.pi))
    result = sp.integrate(middle, (theta, 0, 2*sp.pi))

    print(f"\n  Divergence theorem: flux = triple_integral div(F) dV")
    print(f"  In spherical coordinates:")
    print(f"  = integral_0^{{2pi}} integral_0^pi integral_0^2 3*rho^4*sin(phi) d(rho) d(phi) d(theta)")
    print(f"\n  Inner (rho): integral_0^2 3*rho^4 d(rho) = {inner}")
    print(f"  Middle (phi): integral_0^pi {inner}*sin(phi) d(phi) = {middle}")
    print(f"  Outer (theta): integral_0^{{2pi}} {middle} d(theta) = {result}")
    print(f"\n  Flux = {result} = {float(result):.4f}")


# ============================================================
# Problem 4: Stokes' Theorem
# ============================================================
def exercise_4():
    """
    F = (y^2, z^2, x^2), C = triangle (1,0,0),(0,1,0),(0,0,1).
    Use Stokes' theorem: oint F.dr = double_integral curl(F).dS.
    """
    print("\n" + "=" * 60)
    print("Problem 4: Stokes' Theorem")
    print("=" * 60)

    x, y, z, u, v = sp.symbols('x y z u v')

    F1, F2, F3 = y**2, z**2, x**2

    # curl(F) = (dF3/dy - dF2/dz, dF1/dz - dF3/dx, dF2/dx - dF1/dy)
    curl = (sp.diff(F3, y) - sp.diff(F2, z),
            sp.diff(F1, z) - sp.diff(F3, x),
            sp.diff(F2, x) - sp.diff(F1, y))

    print(f"\n  F = (y^2, z^2, x^2)")
    print(f"  curl(F) = ({curl[0]}, {curl[1]}, {curl[2]})")
    print(f"          = (-2z, -2x, -2y)")

    # Triangle on plane x + y + z = 1
    # Normal to plane: n = (1, 1, 1)/sqrt(3) (outward/upward)
    # Parametrize: r(u,v) = (1-u-v, u, v), 0 <= u, v, u+v <= 1
    # r_u = (-1, 1, 0), r_v = (-1, 0, 1)
    # r_u x r_v = (1, 1, 1) (points upward -- correct orientation)

    # curl(F) . (r_u x r_v) = (-2z)*(1) + (-2x)*(1) + (-2y)*(1)
    # = -2(x + y + z) = -2*1 = -2 (on the plane x+y+z=1)
    print(f"\n  Surface: plane x+y+z = 1 (triangle vertices)")
    print(f"  Parametrize: r(u,v) = (1-u-v, u, v), 0<=u, v<=1, u+v<=1")
    print(f"  r_u x r_v = (1, 1, 1)")
    print(f"  curl(F).(r_u x r_v) = -2z - 2x - 2y = -2(x+y+z) = -2")

    # Surface integral
    result = sp.integrate(
        sp.integrate(-2, (v, 0, 1 - u)),
        (u, 0, 1)
    )

    print(f"\n  Surface integral = integral_0^1 integral_0^{{1-u}} (-2) dv du")
    print(f"  = -2 * (area of triangle) = -2 * 1/2 = {result}")

    # Numerical verification via line integral
    # Side 1: (1,0,0) -> (0,1,0): r = (1-t, t, 0), t in [0,1]
    t = sp.Symbol('t')
    s1 = sp.integrate(
        (0)**2 * (-1) + (0)**2 * (1) + (1-t)**2 * 0,
        # F.dr = F1*dx + F2*dy + F3*dz
        # = t^2*(-1) + 0^2*(1) + (1-t)^2*0
        (t, 0, 1)
    )
    # Actually: F at (1-t, t, 0): F1 = t^2, F2 = 0, F3 = (1-t)^2
    # dr = (-1, 1, 0) dt
    # F.dr = t^2*(-1) + 0*1 + (1-t)^2*0 = -t^2
    s1 = sp.integrate(-t**2, (t, 0, 1))

    # Side 2: (0,1,0) -> (0,0,1): r = (0, 1-t, t)
    # F: F1 = (1-t)^2, F2 = t^2, F3 = 0
    # dr = (0, -1, 1) dt
    # F.dr = 0 + t^2*(-1) + 0 = -t^2
    s2 = sp.integrate(-t**2, (t, 0, 1))

    # Side 3: (0,0,1) -> (1,0,0): r = (t, 0, 1-t)
    # F: F1 = 0, F2 = (1-t)^2, F3 = t^2
    # dr = (1, 0, -1) dt
    # F.dr = 0 + 0 + t^2*(-1) = -t^2
    s3 = sp.integrate(-t**2, (t, 0, 1))

    line_total = s1 + s2 + s3
    print(f"\n  Line integral verification:")
    print(f"    Side 1: {s1}")
    print(f"    Side 2: {s2}")
    print(f"    Side 3: {s3}")
    print(f"    Total:  {line_total}")
    print(f"\n  Stokes' verified: surface integral = line integral = {result}")


# ============================================================
# Problem 5: Gauss's Law (Inverse-Square Field)
# ============================================================
def exercise_5():
    """
    Show flux of F = r/|r|^3 through any closed surface enclosing
    the origin is 4*pi.
    """
    print("\n" + "=" * 60)
    print("Problem 5: Gauss's Law (Inverse-Square Field)")
    print("=" * 60)

    x, y, z = sp.symbols('x y z')
    rho, phi, theta = sp.symbols('rho phi theta', positive=True)

    print(f"\n  F = r/|r|^3 = (x, y, z) / (x^2+y^2+z^2)^(3/2)")
    print(f"\n  Key insight: div(F) = 0 for r != 0")

    # Verify div(F) = 0
    r_sq = x**2 + y**2 + z**2
    F1 = x / r_sq**sp.Rational(3, 2)
    div_check = sp.diff(F1, x)
    div_check_simplified = sp.simplify(div_check + sp.diff(y/r_sq**sp.Rational(3, 2), y) +
                                        sp.diff(z/r_sq**sp.Rational(3, 2), z))
    print(f"  div(F) = {div_check_simplified} (for r != 0)")

    print(f"\n  Since div(F) = 0 everywhere except the origin, by the")
    print(f"  divergence theorem, the flux through ANY closed surface")
    print(f"  enclosing the origin equals the flux through a small sphere.")

    # Compute flux through sphere of radius R
    # On sphere: |r| = R, outward normal n = r/|r| = r/R
    # F.n = (r/R^3).(r/R) = |r|^2/(R^4) = R^2/R^4 = 1/R^2
    # Flux = integral F.n dS = (1/R^2) * 4*pi*R^2 = 4*pi
    print(f"\n  On sphere of radius R:")
    print(f"    F = r/R^3")
    print(f"    n = r/R (outward normal)")
    print(f"    F . n = (r/R^3) . (r/R) = R^2 / R^4 = 1/R^2")
    print(f"    Flux = (1/R^2) * surface area = (1/R^2) * 4*pi*R^2 = 4*pi")

    # Numerical verification: flux through unit sphere
    # F.n = 1/R^2 = 1 on unit sphere
    # Flux = integral_0^{2pi} integral_0^pi 1 * R^2*sin(phi) d(phi) d(theta)
    # = integral_0^{2pi} integral_0^pi sin(phi) d(phi) d(theta) = 4*pi
    R = 1
    flux_num, _ = sci_int.dblquad(
        lambda phi_val, theta_val: np.sin(phi_val),
        0, 2*np.pi,
        0, np.pi
    )
    print(f"\n  Numerical verification (unit sphere):")
    print(f"    Flux = {flux_num:.10f}")
    print(f"    4*pi = {4*np.pi:.10f}")
    print(f"    Match: {abs(flux_num - 4*np.pi) < 1e-8}")

    print(f"\n  If the surface does NOT enclose the origin:")
    print(f"    div(F) = 0 everywhere inside => by divergence theorem, flux = 0.")
    print(f"    This is analogous to Gauss's law in electrostatics:")
    print(f"    the electric flux through a surface depends only on the")
    print(f"    enclosed charge.")


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
    print("All exercises for Lesson 11 completed.")
    print("=" * 60)
