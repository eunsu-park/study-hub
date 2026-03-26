"""
Exercise Solutions: Lesson 09 - Multivariable Functions
Calculus and Differential Equations

Topics covered:
- Gradient and directional derivative
- Multivariable chain rule
- Critical points and classification (second derivative test)
- Ideal gas law cyclic relation
- Linear approximation in two variables
"""

import numpy as np
import sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# Problem 1: Gradient and Directional Derivative
# ============================================================
def exercise_1():
    """
    f(x,y) = ln(x^2 + y^2):
    (a) Find grad(f) and show it points radially outward
    (b) Directional derivative at (1,1) in direction (3,4)
    (c) Where is grad(f) undefined?
    """
    print("=" * 60)
    print("Problem 1: Gradient and Directional Derivative")
    print("=" * 60)

    x, y = sp.symbols('x y', real=True)

    f = sp.ln(x**2 + y**2)
    fx = sp.diff(f, x)
    fy = sp.diff(f, y)

    print(f"\n  f(x,y) = ln(x^2 + y^2)")
    print(f"  f_x = {fx}")
    print(f"  f_y = {fy}")

    # (a) Show gradient points radially outward
    print(f"\n(a) grad(f) = ({fx}, {fy})")
    print(f"           = (2x/(x^2+y^2), 2y/(x^2+y^2))")
    print(f"           = 2/(x^2+y^2) * (x, y)")
    print(f"    The vector (x, y) points radially outward from the origin.")
    print(f"    The scalar 2/(x^2+y^2) > 0, so grad(f) points radially outward.")
    print(f"    This makes sense: ln(r^2) = 2*ln(r) increases as r increases.")

    # (b) Directional derivative at (1,1) in direction (3,4)
    # Unit direction vector: u = (3,4)/5
    grad_at_11 = (fx.subs([(x, 1), (y, 1)]), fy.subs([(x, 1), (y, 1)]))
    u = (sp.Rational(3, 5), sp.Rational(4, 5))  # unit vector
    D_u_f = grad_at_11[0] * u[0] + grad_at_11[1] * u[1]

    print(f"\n(b) At (1, 1):")
    print(f"    grad(f)(1,1) = ({grad_at_11[0]}, {grad_at_11[1]})")
    print(f"    Direction (3, 4), unit vector u = (3/5, 4/5)")
    print(f"    D_u f = grad(f) . u = {grad_at_11[0]}*3/5 + {grad_at_11[1]}*4/5")
    print(f"          = {D_u_f} = {float(D_u_f):.10f}")

    # (c) Where is grad(f) undefined?
    print(f"\n(c) grad(f) undefined where:")
    print(f"    x^2 + y^2 = 0, i.e., at the origin (0, 0)")
    print(f"    (Also f itself is undefined at the origin: ln(0) = -inf)")


# ============================================================
# Problem 2: Multivariable Chain Rule
# ============================================================
def exercise_2():
    """
    w = xy + yz + zx, with x = t, y = t^2, z = t^3.
    Find dw/dt via chain rule, verify by direct substitution.
    """
    print("\n" + "=" * 60)
    print("Problem 2: Multivariable Chain Rule")
    print("=" * 60)

    x, y, z, t = sp.symbols('x y z t')

    w = x*y + y*z + z*x

    # Partial derivatives
    w_x = sp.diff(w, x)
    w_y = sp.diff(w, y)
    w_z = sp.diff(w, z)

    # Parametric: x=t, y=t^2, z=t^3
    x_t = t
    y_t = t**2
    z_t = t**3

    dx_dt = sp.diff(x_t, t)
    dy_dt = sp.diff(y_t, t)
    dz_dt = sp.diff(z_t, t)

    # Chain rule: dw/dt = w_x * dx/dt + w_y * dy/dt + w_z * dz/dt
    dw_dt_chain = (w_x * dx_dt + w_y * dy_dt + w_z * dz_dt)
    dw_dt_chain_sub = dw_dt_chain.subs([(x, x_t), (y, y_t), (z, z_t)])
    dw_dt_chain_expanded = sp.expand(dw_dt_chain_sub)

    print(f"\n  w = xy + yz + zx")
    print(f"  w_x = {w_x}, w_y = {w_y}, w_z = {w_z}")
    print(f"\n  x = t, y = t^2, z = t^3")
    print(f"  dx/dt = {dx_dt}, dy/dt = {dy_dt}, dz/dt = {dz_dt}")
    print(f"\n  Chain rule:")
    print(f"  dw/dt = w_x*dx/dt + w_y*dy/dt + w_z*dz/dt")
    print(f"        = {dw_dt_chain}")
    print(f"  Substituting x=t, y=t^2, z=t^3:")
    print(f"  dw/dt = {dw_dt_chain_expanded}")

    # Direct verification: substitute first, then differentiate
    w_direct = x_t * y_t + y_t * z_t + z_t * x_t
    w_direct_expanded = sp.expand(w_direct)
    dw_dt_direct = sp.diff(w_direct, t)
    dw_dt_direct_expanded = sp.expand(dw_dt_direct)

    print(f"\n  Direct method:")
    print(f"  w(t) = t*t^2 + t^2*t^3 + t^3*t = {w_direct_expanded}")
    print(f"  dw/dt = {dw_dt_direct_expanded}")
    print(f"\n  Match: {sp.simplify(dw_dt_chain_expanded - dw_dt_direct_expanded) == 0}")


# ============================================================
# Problem 3: Critical Points Classification
# ============================================================
def exercise_3():
    """
    Find and classify all critical points of f(x,y) = 2x^3 + 6xy^2 - 3y^3 - 150x.
    """
    print("\n" + "=" * 60)
    print("Problem 3: Critical Points Classification")
    print("=" * 60)

    x, y = sp.symbols('x y', real=True)

    f = 2*x**3 + 6*x*y**2 - 3*y**3 - 150*x

    fx = sp.diff(f, x)
    fy = sp.diff(f, y)
    fxx = sp.diff(f, x, 2)
    fyy = sp.diff(f, y, 2)
    fxy = sp.diff(f, x, y)

    print(f"\n  f(x,y) = 2x^3 + 6xy^2 - 3y^3 - 150x")
    print(f"  f_x = {fx}")
    print(f"  f_y = {fy}")

    # Solve f_x = 0, f_y = 0
    critical_pts = sp.solve([fx, fy], [x, y])
    print(f"\n  Critical points (f_x = 0, f_y = 0):")
    print(f"  {critical_pts}")

    # Second derivatives
    print(f"\n  f_xx = {fxx}, f_yy = {fyy}, f_xy = {fxy}")

    # Classify each critical point
    print(f"\n  Classification:")
    for pt in critical_pts:
        xp, yp = pt
        D = fxx.subs([(x, xp), (y, yp)]) * fyy.subs([(x, xp), (y, yp)]) - \
            fxy.subs([(x, xp), (y, yp)])**2
        fxx_val = fxx.subs([(x, xp), (y, yp)])
        f_val = f.subs([(x, xp), (y, yp)])

        if D > 0 and fxx_val > 0:
            label = "LOCAL MINIMUM"
        elif D > 0 and fxx_val < 0:
            label = "LOCAL MAXIMUM"
        elif D < 0:
            label = "SADDLE POINT"
        else:
            label = "INCONCLUSIVE"

        print(f"    ({xp}, {yp}): f = {f_val}, D = {D}, f_xx = {fxx_val} => {label}")


# ============================================================
# Problem 4: Ideal Gas Law Cyclic Relation
# ============================================================
def exercise_4():
    """
    PV = nRT. Find dP/dV, dP/dT using implicit diff.
    Verify (dP/dV)(dV/dT)(dT/dP) = -1.
    """
    print("\n" + "=" * 60)
    print("Problem 4: Ideal Gas Law Cyclic Relation")
    print("=" * 60)

    P, V, T, n, R = sp.symbols('P V T n R', positive=True)

    # PV = nRT => P = nRT/V
    P_expr = n * R * T / V

    dP_dV = sp.diff(P_expr, V)  # holding T constant
    dP_dT = sp.diff(P_expr, T)  # holding V constant

    print(f"\n  Ideal gas law: PV = nRT")
    print(f"  P = nRT/V")
    print(f"\n(a) Partial derivatives:")
    print(f"    (dP/dV)_T = {dP_dV}")
    print(f"    (dP/dT)_V = {dP_dT}")

    # (b) Cyclic relation
    # V = nRT/P => dV/dT = nR/P
    V_expr = n * R * T / P
    dV_dT = sp.diff(V_expr, T)

    # T = PV/(nR) => dT/dP = V/(nR)
    T_expr = P * V / (n * R)
    dT_dP = sp.diff(T_expr, P)

    print(f"\n(b) Other partial derivatives:")
    print(f"    V = nRT/P => (dV/dT)_P = {dV_dT}")
    print(f"    T = PV/(nR) => (dT/dP)_V = {dT_dP}")

    # Product: (dP/dV)(dV/dT)(dT/dP)
    # = (-nRT/V^2)(nR/P)(V/(nR))
    product = sp.simplify(dP_dV * dV_dT * dT_dP)
    # Substitute P = nRT/V to simplify
    product_sub = product.subs(P, n*R*T/V)
    product_simplified = sp.simplify(product_sub)

    print(f"\n    Product: (dP/dV)*(dV/dT)*(dT/dP)")
    print(f"    = ({dP_dV})*({dV_dT})*({dT_dP})")
    print(f"    = {product}")
    print(f"    Substituting P = nRT/V: {product_simplified}")
    print(f"\n    VERIFIED: The cyclic relation gives -1")
    print(f"    This is a general result: for any equation F(P,V,T) = 0,")
    print(f"    (dP/dV)_T * (dV/dT)_P * (dT/dP)_V = -1")


# ============================================================
# Problem 5: Linear Approximation
# ============================================================
def exercise_5():
    """
    f(x,y) = sqrt(x)*e^y at (4,0).
    Estimate f(4.1, -0.05) using linear approximation.
    """
    print("\n" + "=" * 60)
    print("Problem 5: Linear Approximation")
    print("=" * 60)

    x, y = sp.symbols('x y')

    f = sp.sqrt(x) * sp.exp(y)
    fx = sp.diff(f, x)
    fy = sp.diff(f, y)

    # At (4, 0)
    a, b = 4, 0
    f_val = f.subs([(x, a), (y, b)])
    fx_val = fx.subs([(x, a), (y, b)])
    fy_val = fy.subs([(x, a), (y, b)])

    print(f"\n  f(x,y) = sqrt(x)*e^y")
    print(f"  f_x = {fx}")
    print(f"  f_y = {fy}")
    print(f"\n  At (4, 0):")
    print(f"    f(4,0) = sqrt(4)*e^0 = {f_val}")
    print(f"    f_x(4,0) = {fx_val} = {float(fx_val):.6f}")
    print(f"    f_y(4,0) = {fy_val} = {float(fy_val):.6f}")

    # Linear approximation:
    # L(x,y) = f(a,b) + f_x(a,b)*(x-a) + f_y(a,b)*(y-b)
    print(f"\n  Linear approximation:")
    print(f"  L(x,y) = {f_val} + {fx_val}*(x-4) + {fy_val}*(y-0)")
    print(f"         = 2 + (1/4)*(x-4) + 2*(y)")

    # Estimate f(4.1, -0.05)
    dx, dy = 0.1, -0.05
    L_approx = float(f_val) + float(fx_val) * dx + float(fy_val) * dy
    exact = np.sqrt(4.1) * np.exp(-0.05)

    print(f"\n  Estimating f(4.1, -0.05):")
    print(f"    L(4.1, -0.05) = 2 + (1/4)*(0.1) + 2*(-0.05)")
    print(f"                  = 2 + 0.025 - 0.1")
    print(f"                  = {L_approx:.10f}")
    print(f"\n    Exact value: sqrt(4.1)*e^(-0.05) = {exact:.10f}")
    print(f"    Error: {abs(L_approx - exact):.6e}")
    print(f"    Relative error: {abs(L_approx - exact)/exact * 100:.4f}%")


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
    print("All exercises for Lesson 09 completed.")
    print("=" * 60)
