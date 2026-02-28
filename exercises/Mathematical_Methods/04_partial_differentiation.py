"""
Exercise Solutions: Lesson 04 - Partial Differentiation
Mathematical Methods for Physical Sciences

Covers: partial derivatives, Laplacian in polar coordinates,
        critical points, Lagrange multipliers, van der Waals, Gaussian integral
"""

import numpy as np
import sympy as sp


def exercise_1_partial_derivatives():
    """
    Problem 1: For f(x,y) = x^3*y^2 + sin(xy),
    verify f_xy = f_yx.
    """
    print("=" * 60)
    print("Problem 1: Partial Derivatives and Mixed Partials")
    print("=" * 60)

    x, y = sp.symbols('x y')
    f = x**3 * y**2 + sp.sin(x * y)

    f_x = sp.diff(f, x)
    f_y = sp.diff(f, y)
    f_xy = sp.diff(f_x, y)
    f_yx = sp.diff(f_y, x)

    print(f"\nf(x,y) = x^3*y^2 + sin(xy)")
    print(f"\nf_x = {f_x}")
    print(f"f_y = {f_y}")
    print(f"\nf_xy = {sp.simplify(f_xy)}")
    print(f"f_yx = {sp.simplify(f_yx)}")
    print(f"\nf_xy == f_yx: {sp.simplify(f_xy - f_yx) == 0}")
    print("(Clairaut's theorem: mixed partials are equal for C^2 functions)")


def exercise_2_laplacian_polar():
    """
    Problem 2: Show that the Laplacian in polar coordinates is:
    nabla^2 f = f_rr + (1/r)*f_r + (1/r^2)*f_{theta theta}
    Verify with f = r^n * cos(n*theta).
    """
    print("\n" + "=" * 60)
    print("Problem 2: Laplacian in Polar Coordinates")
    print("=" * 60)

    r, theta, n = sp.symbols('r theta n', positive=True)

    # f = r^n * cos(n*theta) should be harmonic (nabla^2 f = 0)
    f = r**n * sp.cos(n * theta)

    f_r = sp.diff(f, r)
    f_rr = sp.diff(f, r, 2)
    f_tt = sp.diff(f, theta, 2)

    laplacian = f_rr + f_r / r + f_tt / r**2
    laplacian_simplified = sp.simplify(laplacian)

    print(f"\nf(r, theta) = r^n * cos(n*theta)")
    print(f"\nf_r  = {f_r}")
    print(f"f_rr = {f_rr}")
    print(f"f_{'{theta theta}'} = {f_tt}")
    print(f"\nnabla^2 f = f_rr + (1/r)*f_r + (1/r^2)*f_tt")
    print(f"         = {laplacian_simplified}")
    print(f"\nSince nabla^2 f = 0, r^n*cos(n*theta) is HARMONIC")

    # Verify for specific n values
    print("\nVerification for specific n:")
    for n_val in [1, 2, 3]:
        val = laplacian_simplified.subs(n, n_val)
        print(f"  n={n_val}: nabla^2 f = {sp.simplify(val)}")


def exercise_3_critical_points():
    """
    Problem 3: Find and classify critical points of f(x,y) = x^3 + y^3 - 3xy.
    """
    print("\n" + "=" * 60)
    print("Problem 3: Critical Points Classification")
    print("=" * 60)

    x, y = sp.symbols('x y')
    f = x**3 + y**3 - 3 * x * y

    f_x = sp.diff(f, x)
    f_y = sp.diff(f, y)

    print(f"\nf(x,y) = x^3 + y^3 - 3xy")
    print(f"f_x = {f_x}")
    print(f"f_y = {f_y}")

    # Find critical points: f_x = f_y = 0
    critical = sp.solve([f_x, f_y], [x, y])
    print(f"\nCritical points (f_x = f_y = 0):")

    # Second derivatives for classification
    f_xx = sp.diff(f, x, 2)
    f_yy = sp.diff(f, y, 2)
    f_xy = sp.diff(f, x, y)

    print(f"\nf_xx = {f_xx}, f_yy = {f_yy}, f_xy = {f_xy}")

    for point in critical:
        px, py = point
        D = f_xx.subs([(x, px), (y, py)]) * f_yy.subs([(x, px), (y, py)]) \
            - f_xy.subs([(x, px), (y, py)])**2
        fxx_val = f_xx.subs([(x, px), (y, py)])
        f_val = f.subs([(x, px), (y, py)])

        print(f"\n  Point ({px}, {py}):")
        print(f"    f = {f_val}")
        print(f"    D = f_xx*f_yy - f_xy^2 = {D}")
        print(f"    f_xx = {fxx_val}")

        if D > 0:
            if fxx_val > 0:
                print(f"    D > 0, f_xx > 0 => LOCAL MINIMUM")
            else:
                print(f"    D > 0, f_xx < 0 => LOCAL MAXIMUM")
        elif D < 0:
            print(f"    D < 0 => SADDLE POINT")
        else:
            print(f"    D = 0 => INCONCLUSIVE")


def exercise_4_lagrange_multipliers():
    """
    Problem 4: Use Lagrange multipliers to find the extremum of
    f(x,y,z) = x + y + z on the sphere x^2 + y^2 + z^2 = 1.
    """
    print("\n" + "=" * 60)
    print("Problem 4: Lagrange Multipliers")
    print("=" * 60)

    x, y, z, lam = sp.symbols('x y z lambda')
    f = x + y + z
    g = x**2 + y**2 + z**2 - 1

    print(f"\nMaximize f = x + y + z")
    print(f"Subject to g = x^2 + y^2 + z^2 - 1 = 0")

    # nabla f = lambda * nabla g
    eq1 = sp.Eq(sp.diff(f, x), lam * sp.diff(g, x))  # 1 = 2*lambda*x
    eq2 = sp.Eq(sp.diff(f, y), lam * sp.diff(g, y))  # 1 = 2*lambda*y
    eq3 = sp.Eq(sp.diff(f, z), lam * sp.diff(g, z))  # 1 = 2*lambda*z
    eq4 = sp.Eq(g, 0)

    print(f"\nLagrange conditions:")
    print(f"  1 = 2*lambda*x")
    print(f"  1 = 2*lambda*y")
    print(f"  1 = 2*lambda*z")
    print(f"  x^2 + y^2 + z^2 = 1")

    solutions = sp.solve([eq1, eq2, eq3, eq4], [x, y, z, lam])
    print(f"\nSolutions:")
    for sol in solutions:
        sx, sy, sz, sl = sol
        fval = f.subs([(x, sx), (y, sy), (z, sz)])
        print(f"  (x, y, z) = ({sx}, {sy}, {sz}), lambda = {sl}")
        print(f"  f = {fval} = {float(fval):.6f}")

    print(f"\n  Maximum of f = sqrt(3) at (1/sqrt(3), 1/sqrt(3), 1/sqrt(3))")
    print(f"  Minimum of f = -sqrt(3) at (-1/sqrt(3), -1/sqrt(3), -1/sqrt(3))")
    print(f"  Geometrically: the gradient of f is along (1,1,1)/sqrt(3),")
    print(f"  so extrema occur where the sphere normal points in that direction.")


def exercise_5_van_der_waals():
    """
    Problem 5: Van der Waals equation: (P + a/V^2)(V - b) = nRT
    Find (dP/dT)_V, (dV/dT)_P, and verify the cyclic relation
    (dP/dT)_V * (dT/dV)_P * (dV/dP)_T = -1
    """
    print("\n" + "=" * 60)
    print("Problem 5: Van der Waals Partial Derivatives")
    print("=" * 60)

    P, V, T, a, b, n, R = sp.symbols('P V T a b n R', positive=True)

    # Van der Waals: P = nRT/(V-b) - a/V^2
    P_expr = n * R * T / (V - b) - a / V**2

    print(f"\nVan der Waals equation: P = nRT/(V-b) - a/V^2")
    print(f"P = {P_expr}")

    # (dP/dT)_V
    dP_dT = sp.diff(P_expr, T)
    print(f"\n(dP/dT)_V = {dP_dT}")

    # For (dV/dT)_P, use implicit differentiation
    # F(P,V,T) = P - nRT/(V-b) + a/V^2 = 0
    # (dV/dT)_P = -(dF/dT)/(dF/dV)
    F = P - P_expr
    dF_dT = sp.diff(F, T)
    dF_dV = sp.diff(F, V)
    dV_dT = -dF_dT / dF_dV
    dV_dT_simplified = sp.simplify(dV_dT)

    print(f"\n(dV/dT)_P = {dV_dT_simplified}")

    # (dV/dP)_T
    dF_dP = sp.diff(F, P)  # = 1
    dV_dP = -dF_dP / dF_dV
    dV_dP_simplified = sp.simplify(dV_dP)

    print(f"(dV/dP)_T = {dV_dP_simplified}")

    # Verify cyclic relation: (dP/dT)_V * (dT/dV)_P * (dV/dP)_T = -1
    dT_dV = 1 / dV_dT
    product = sp.simplify(dP_dT * dT_dV * dV_dP_simplified)
    print(f"\nCyclic relation check:")
    print(f"  (dP/dT)_V * (dT/dV)_P * (dV/dP)_T = {product}")
    print(f"  Expected: -1")


def exercise_6_gaussian_integral():
    """
    Problem 6: Evaluate the 3D Gaussian integral
    I = integral of exp(-(x^2 + y^2 + z^2)) dV over all space
    using spherical coordinates.
    """
    print("\n" + "=" * 60)
    print("Problem 6: 3D Gaussian Integral (Spherical Coordinates)")
    print("=" * 60)

    r, theta, phi = sp.symbols('r theta phi', nonneg=True)

    # In spherical coordinates:
    # dV = r^2 * sin(theta) * dr * dtheta * dphi
    # x^2 + y^2 + z^2 = r^2

    integrand = sp.exp(-r**2) * r**2 * sp.sin(theta)

    print("\nI = integral exp(-(x^2+y^2+z^2)) dxdydz")
    print("  = integral exp(-r^2) * r^2 * sin(theta) dr dtheta dphi")

    # phi integral: 0 to 2*pi
    I_phi = sp.integrate(1, (phi, 0, 2 * sp.pi))
    print(f"\n  phi integral: {I_phi}")

    # theta integral: 0 to pi
    I_theta = sp.integrate(sp.sin(theta), (theta, 0, sp.pi))
    print(f"  theta integral: {I_theta}")

    # r integral: 0 to infinity
    I_r = sp.integrate(r**2 * sp.exp(-r**2), (r, 0, sp.oo))
    print(f"  r integral: {I_r}")

    # Total
    I_total = I_phi * I_theta * I_r
    print(f"\n  I = {I_phi} * {I_theta} * {I_r} = {I_total}")
    print(f"    = pi^(3/2) = {float(I_total):.10f}")
    print(f"    = {np.pi**(3/2):.10f}")

    print(f"\n  Note: This equals (sqrt(pi))^3 = [integral e^(-x^2) dx]^3")
    print(f"  Confirms: integral e^(-x^2) dx = sqrt(pi)")

    # Numerical verification
    from scipy import integrate as sci_integrate
    result, error = sci_integrate.tplquad(
        lambda z, y, x: np.exp(-(x**2 + y**2 + z**2)),
        -5, 5,  # x limits
        lambda x: -5, lambda x: 5,  # y limits
        lambda x, y: -5, lambda x, y: 5  # z limits
    )
    print(f"\n  Numerical (truncated [-5,5]^3): {result:.10f}")
    print(f"  Exact:                          {np.pi**1.5:.10f}")


if __name__ == "__main__":
    exercise_1_partial_derivatives()
    exercise_2_laplacian_polar()
    exercise_3_critical_points()
    exercise_4_lagrange_multipliers()
    exercise_5_van_der_waals()
    exercise_6_gaussian_integral()
