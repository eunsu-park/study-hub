"""
Exercises for Lesson 18: Tensor Analysis
Topic: Mathematical_Methods
Solutions to practice problems from the lesson.
"""

import numpy as np
import sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def exercise_1():
    """
    Problem 1: Tensor transformation under Lorentz boost.
    x' = x*cosh(a) + y*sinh(a), y' = x*sinh(a) + y*cosh(a)
    (a) Transformation matrix
    (b) Transform vector A^i = (3, 4) with alpha = 0.5
    (c) Transform tensor T^{ij}
    """
    print("=" * 60)
    print("Problem 1: Tensor Transformation (Lorentz Boost)")
    print("=" * 60)

    alpha = 0.5

    # (a) Transformation matrix
    print("\n(a) Transformation matrix dx'^i/dx^j:")
    Lambda = np.array([
        [np.cosh(alpha), np.sinh(alpha)],
        [np.sinh(alpha), np.cosh(alpha)]
    ])
    print(f"  Lambda = [[cosh(a), sinh(a)],")
    print(f"            [sinh(a), cosh(a)]]")
    print(f"  = [[{Lambda[0, 0]:.6f}, {Lambda[0, 1]:.6f}],")
    print(f"     [{Lambda[1, 0]:.6f}, {Lambda[1, 1]:.6f}]]")

    # (b) Transform vector
    print("\n(b) Transform A^i = (3, 4):")
    A = np.array([3.0, 4.0])
    A_prime = Lambda @ A
    print(f"  A'^i = Lambda^i_j * A^j")
    print(f"  A' = ({A_prime[0]:.6f}, {A_prime[1]:.6f})")

    # Verify: magnitude should be preserved (Lorentz-like)
    # For this metric (hyperbolic), the invariant is x^2 - y^2 (not x^2 + y^2)
    inv_orig = A[0]**2 - A[1]**2
    inv_prime = A_prime[0]**2 - A_prime[1]**2
    print(f"\n  Lorentz invariant check: A^2 - B^2")
    print(f"    Original: {inv_orig:.6f}")
    print(f"    Transformed: {inv_prime:.6f}")

    # (c) Transform rank-2 tensor
    print("\n(c) Transform T^{ij}:")
    T = np.array([[1.0, 2.0], [3.0, 4.0]])
    T_prime = Lambda @ T @ Lambda.T
    print(f"  T'^{{ij}} = Lambda^i_k * Lambda^j_l * T^{{kl}}")
    print(f"  T' = [[{T_prime[0, 0]:.4f}, {T_prime[0, 1]:.4f}],")
    print(f"        [{T_prime[1, 0]:.4f}, {T_prime[1, 1]:.4f}]]")

    # Verify trace is invariant for this transformation
    print(f"\n  Trace(T) = {np.trace(T):.4f}")
    print(f"  Trace(T') = {np.trace(T_prime):.4f}")


def exercise_2():
    """
    Problem 2: Einstein summation convention.
    (a) Dot product A^i B_i
    (b) Matrix multiplication M^i_k N^k_j
    (c) Reduce eps_{ijk} eps_{imn} using delta-identity
    """
    print("\n" + "=" * 60)
    print("Problem 2: Einstein Summation Convention")
    print("=" * 60)

    # (a) Dot product
    print("\n(a) Dot product: sum_i A^i B_i  =>  A^i B_i  (summation implied)")
    A = np.array([1, 2, 3])
    B = np.array([4, 5, 6])
    dot = np.einsum('i,i', A, B)
    print(f"  A = {A}, B = {B}")
    print(f"  A^i B_i = {dot}  (expanded: 1*4 + 2*5 + 3*6 = {1*4+2*5+3*6})")

    # (b) Matrix multiplication
    print("\n(b) Matrix product: sum_k M^i_k N^k_j  =>  M^i_k N^k_j")
    M = np.array([[1, 2], [3, 4]])
    N = np.array([[5, 6], [7, 8]])
    MN = np.einsum('ik,kj->ij', M, N)
    print(f"  M = [[1,2],[3,4]], N = [[5,6],[7,8]]")
    print(f"  M^i_k N^k_j = [[{MN[0, 0]}, {MN[0, 1]}], [{MN[1, 0]}, {MN[1, 1]}]]")
    print(f"  Verify: M @ N = {(M @ N).tolist()}")

    # (c) Levi-Civita contraction identity
    print("\n(c) eps_{ijk} eps_{imn} = delta_{jm}*delta_{kn} - delta_{jn}*delta_{km}")

    # Build Levi-Civita tensor
    eps = np.zeros((3, 3, 3))
    eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1
    eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1

    delta = np.eye(3)

    # Verify identity for all j, k, m, n
    max_error = 0.0
    for j in range(3):
        for k in range(3):
            for m_idx in range(3):
                for n in range(3):
                    lhs = np.einsum('i', eps[:, j, k] * eps[:, m_idx, n])
                    rhs = delta[j, m_idx] * delta[k, n] - delta[j, n] * delta[k, m_idx]
                    max_error = max(max_error, abs(lhs - rhs))

    print(f"  Verification: max|LHS - RHS| = {max_error:.2e}")
    print(f"  Identity verified: {max_error < 1e-10}")

    # Example: cross product via Levi-Civita
    u = np.array([1, 0, 0])
    v = np.array([0, 1, 0])
    cross_einsum = np.einsum('ijk,j,k->i', eps, u, v)
    print(f"\n  Application: u x v via eps_ijk")
    print(f"  u = {u}, v = {v}")
    print(f"  u x v = {cross_einsum}  (expected: [0, 0, 1])")


def exercise_3():
    """
    Problem 3: Metric tensor and index raising/lowering in polar coordinates.
    ds^2 = dr^2 + r^2 dtheta^2
    """
    print("\n" + "=" * 60)
    print("Problem 3: Metric Tensor in Polar Coordinates")
    print("=" * 60)

    r_val = 2.0  # specific radius for numerical example

    # (a) Metric tensor
    print("\n(a) ds^2 = dr^2 + r^2*dtheta^2")
    g = np.diag([1.0, r_val**2])
    g_inv = np.diag([1.0, 1.0 / r_val**2])
    print(f"  g_ij = diag(1, r^2) = diag(1, {r_val**2})")
    print(f"  g^ij = diag(1, 1/r^2) = diag(1, {1/r_val**2})")
    print(f"  Verify: g_ij * g^jk = delta_i^k: {np.allclose(g @ g_inv, np.eye(2))}")

    # (b) Lower index: A_i = g_{ij} A^j
    print(f"\n(b) Contravariant vector: A^i = (A^r, A^theta) = (2, 1/r)")
    A_contra = np.array([2.0, 1.0 / r_val])
    A_cov = g @ A_contra
    print(f"  A^i = ({A_contra[0]}, {A_contra[1]:.4f})")
    print(f"  A_i = g_ij A^j = ({A_cov[0]:.4f}, {A_cov[1]:.4f})")
    print(f"  A_r = g_{'{rr}'} A^r = 1 * 2 = {A_cov[0]}")
    print(f"  A_theta = g_{{theta theta}} A^theta = r^2 * (1/r) = r = {A_cov[1]}")

    # (c) Scalar invariant A^i A_i
    print(f"\n(c) Invariant: A^i A_i = A^r*A_r + A^theta*A_theta")
    invariant = np.dot(A_contra, A_cov)
    print(f"  A^i A_i = {A_contra[0]}*{A_cov[0]} + {A_contra[1]:.4f}*{A_cov[1]:.4f}")
    print(f"          = {invariant:.6f}")

    # Verify in Cartesian: A = (2, 1/r) in polar means
    # A_x = A^r cos(theta) - r*A^theta sin(theta)
    # A_y = A^r sin(theta) + r*A^theta cos(theta)
    theta_test = np.pi / 4
    Ax = A_contra[0] * np.cos(theta_test) - r_val * A_contra[1] * np.sin(theta_test)
    Ay = A_contra[0] * np.sin(theta_test) + r_val * A_contra[1] * np.cos(theta_test)
    inv_cart = Ax**2 + Ay**2
    print(f"\n  Cartesian verification (theta=pi/4):")
    print(f"  A_x = {Ax:.6f}, A_y = {Ay:.6f}")
    print(f"  |A|^2 = A_x^2 + A_y^2 = {inv_cart:.6f}")
    print(f"  Matches polar result: {np.isclose(invariant, inv_cart)}")


def exercise_4():
    """
    Problem 4: Christoffel symbols for polar coordinates.
    g = diag(1, r^2)
    (a) All Christoffel symbols
    (b) Covariant divergence of A^r = cos(theta), A^theta = -sin(theta)/r
    (c) Verify with Cartesian divergence
    """
    print("\n" + "=" * 60)
    print("Problem 4: Christoffel Symbols (Polar Coordinates)")
    print("=" * 60)

    print("\n(a) Christoffel symbols for g = diag(1, r^2)")
    print("  Gamma^k_{ij} = (1/2) g^{kl} (g_{li,j} + g_{lj,i} - g_{ij,l})")

    # Symbolic computation
    r, theta = sp.symbols('r theta', positive=True)
    g_sym = sp.Matrix([[1, 0], [0, r**2]])
    g_inv_sym = g_sym.inv()
    coords = [r, theta]

    # Compute Christoffel symbols
    Gamma = [[[sp.Rational(0)] * 2 for _ in range(2)] for _ in range(2)]
    for k in range(2):
        for i in range(2):
            for j in range(2):
                val = sp.Rational(0)
                for l in range(2):
                    val += sp.Rational(1, 2) * g_inv_sym[k, l] * (
                        sp.diff(g_sym[l, i], coords[j]) +
                        sp.diff(g_sym[l, j], coords[i]) -
                        sp.diff(g_sym[i, j], coords[l])
                    )
                Gamma[k][i][j] = sp.simplify(val)

    coord_names = ['r', 'theta']
    for k in range(2):
        for i in range(2):
            for j in range(i, 2):
                if Gamma[k][i][j] != 0:
                    print(f"  Gamma^{coord_names[k]}_{{{coord_names[i]},{coord_names[j]}}} = {Gamma[k][i][j]}")

    # (b) Covariant divergence
    print("\n(b) Covariant divergence: nabla_i A^i = d_i A^i + Gamma^i_{ik} A^k")
    print("  A^r = cos(theta), A^theta = -sin(theta)/r")

    Ar = sp.cos(theta)
    Atheta = -sp.sin(theta) / r

    # d_r A^r + d_theta A^theta + Gamma^r_{rr}*A^r + Gamma^r_{r,theta}*A^theta
    # + Gamma^theta_{theta,r}*A^r + Gamma^theta_{theta,theta}*A^theta
    # Simplification: Gamma^i_{ik} A^k = (1/sqrt(g)) * d_k(sqrt(g)*A^k)
    # sqrt(g) = r for polar coordinates

    div = (1 / r) * (sp.diff(r * Ar, r) + sp.diff(r * Atheta, theta))
    div_simplified = sp.simplify(div)
    print(f"  div(A) = (1/r)*d/dr(r*A^r) + (1/r)*d/dtheta(A^theta)")
    print(f"         = {div_simplified}")

    # Alternative: direct computation
    div_direct = sp.diff(Ar, r) + sp.diff(Atheta, theta) + Ar / r
    print(f"  Direct: d_r(A^r) + d_theta(A^theta) + Gamma^r_{'{r,theta}'}*0 + (1/r)*A^r")
    print(f"        = {sp.simplify(div_direct)}")

    # (c) Verify in Cartesian
    print("\n(c) Cartesian verification:")
    print("  A^r = cos(theta), A^theta = -sin(theta)/r")
    print("  A_x = A^r*cos(theta) - r*A^theta*sin(theta)")
    print("      = cos^2(theta) + sin^2(theta) = 1")
    print("  A_y = A^r*sin(theta) + r*A^theta*cos(theta)")
    print("      = sin(theta)*cos(theta) - sin(theta)*cos(theta) = 0")
    print("  div(A) = d(A_x)/dx + d(A_y)/dy = 0 + 0 = 0")
    print(f"  Matches polar result: div = {div_simplified}")


def exercise_5():
    """
    Problem 5: Pseudosphere metric ds^2 = du^2 + e^{-2u}dv^2.
    (a) Christoffel symbols
    (b) Gaussian curvature K = -1
    (c) Geodesic equations
    """
    print("\n" + "=" * 60)
    print("Problem 5: Geodesics on a Pseudosphere")
    print("=" * 60)

    u, v = sp.symbols('u v')
    g = sp.Matrix([[1, 0], [0, sp.exp(-2 * u)]])
    g_inv = g.inv()
    coords = [u, v]

    # (a) Christoffel symbols
    print("\n(a) Christoffel symbols for ds^2 = du^2 + e^{-2u}dv^2")
    Gamma = [[[sp.Rational(0)] * 2 for _ in range(2)] for _ in range(2)]
    for k in range(2):
        for i in range(2):
            for j in range(2):
                val = sp.Rational(0)
                for l in range(2):
                    val += sp.Rational(1, 2) * g_inv[k, l] * (
                        sp.diff(g[l, i], coords[j]) +
                        sp.diff(g[l, j], coords[i]) -
                        sp.diff(g[i, j], coords[l])
                    )
                Gamma[k][i][j] = sp.simplify(val)

    coord_names = ['u', 'v']
    for k in range(2):
        for i in range(2):
            for j in range(i, 2):
                if Gamma[k][i][j] != 0:
                    print(f"  Gamma^{coord_names[k]}_{{{coord_names[i]},{coord_names[j]}}} = {Gamma[k][i][j]}")

    # (b) Gaussian curvature
    print("\n(b) Gaussian curvature:")
    # For 2D: K = R^1_{212} / det(g)
    # R^k_{lij} = d_i Gamma^k_{jl} - d_j Gamma^k_{il} + Gamma^k_{im} Gamma^m_{jl} - Gamma^k_{jm} Gamma^m_{il}
    # K = (1/sqrt(g)) * [-d/du(Gamma^u_{vv}/sqrt(g)) ...] simplified formula:
    # K = -(1/sqrt(g_{vv})) * d^2(sqrt(g_{vv}))/du^2
    sqrt_gvv = sp.sqrt(g[1, 1])
    K = -(1 / sqrt_gvv) * sp.diff(sp.diff(sqrt_gvv, u), u)
    K_simplified = sp.simplify(K)
    print(f"  K = -(1/sqrt(g_vv)) * d^2(sqrt(g_vv))/du^2")
    print(f"  sqrt(g_vv) = e^{{-u}}")
    print(f"  d^2(e^{{-u}})/du^2 = e^{{-u}}")
    print(f"  K = -(1/e^{{-u}})*e^{{-u}} = {K_simplified}")

    # (c) Geodesic equations
    print("\n(c) Geodesic equations:")
    print("  u'' + Gamma^u_{vv}*(v')^2 = 0")
    print(f"  u'' + ({Gamma[0][1][1]})*(v')^2 = 0")
    print(f"  => u'' - e^{{-2u}}*(v')^2 = 0")
    print()
    print("  v'' + 2*Gamma^v_{uv}*u'*v' = 0")
    print(f"  v'' + 2*({Gamma[1][0][1]})*u'*v' = 0")
    print(f"  => v'' - 2*u'*v' = 0")
    print()
    print("  For u = const: u'=0, u''=0")
    print("    First eq: -e^{-2u}*(v')^2 = 0 => v' = 0 => v = const")
    print("    So u=const curves are NOT geodesics (they are circles of latitude)")
    print("    Only the trivial case u=const, v=const (a point) satisfies both.")


def exercise_6():
    """
    Problem 6: Curvature of a torus.
    ds^2 = a^2 dtheta^2 + (R + a*cos(theta))^2 dphi^2
    (a) Gaussian curvature K(theta)
    (b) Regions K > 0, K = 0, K < 0
    (c) Gauss-Bonnet: int K dA = 2*pi*chi = 0
    """
    print("\n" + "=" * 60)
    print("Problem 6: Curvature of a Torus")
    print("=" * 60)

    theta_sym = sp.Symbol('theta')
    R_sym, a_sym = sp.symbols('R a', positive=True)

    # (a) Gaussian curvature
    print("\n(a) ds^2 = a^2 dtheta^2 + (R + a*cos(theta))^2 dphi^2")
    print("  For diagonal metric ds^2 = E dtheta^2 + G dphi^2:")
    print("  K = -(1/(2*sqrt(EG))) * [d/dtheta((G_theta)/(2*sqrt(EG)))]")

    E = a_sym**2
    G = (R_sym + a_sym * sp.cos(theta_sym))**2
    sqrt_EG = sp.sqrt(E * G)

    # K = -(1/(2*sqrt(EG))) * d/dtheta(G_theta / (2*sqrt(EG)))
    # Simplified: K = -(1/sqrt(G)) * d^2(sqrt(G))/dtheta^2 / sqrt(E)
    # But more standard: for ds^2 = a^2 dtheta^2 + f(theta)^2 dphi^2
    # K = -(1/a) * f''(theta) / f(theta) where f = R + a*cos(theta)
    # Wait, K = -(1/(a*f)) * d^2 f/dtheta^2 ... let me use the formula directly

    f = R_sym + a_sym * sp.cos(theta_sym)
    f_tt = sp.diff(f, theta_sym, 2)
    K_expr = -f_tt / (a_sym * f)  # K = -f''/( a * f ) where E = a^2
    # Actually K = -(1/sqrt(E)) * (1/sqrt(G)) * d^2(sqrt(G))/dtheta^2
    # sqrt(G) = f, sqrt(E) = a
    K_expr_alt = -(1 / a_sym) * f_tt / f
    K_simplified = sp.simplify(K_expr_alt)

    print(f"\n  f(theta) = R + a*cos(theta)")
    print(f"  f''(theta) = -a*cos(theta)")
    print(f"  K = -f''/(a*f) = cos(theta)/(a*(R + a*cos(theta)))")
    print(f"  Symbolic: K = {K_simplified}")

    # (b) Sign of K
    print("\n(b) Sign of K depends on cos(theta) and (R + a*cos(theta)):")
    print("  Assuming R > a (standard torus):")
    print("  K > 0 when cos(theta) > 0  =>  -pi/2 < theta < pi/2  (outer half)")
    print("  K = 0 when cos(theta) = 0  =>  theta = +/- pi/2  (top and bottom)")
    print("  K < 0 when cos(theta) < 0  =>  pi/2 < theta < 3pi/2  (inner half)")

    # (c) Gauss-Bonnet theorem
    print("\n(c) Gauss-Bonnet: int K dA = 2*pi*chi")
    print("  dA = a*(R + a*cos(theta)) dtheta dphi")
    print("  int_0^{2pi} int_0^{2pi} K * a*(R+a*cos(theta)) dtheta dphi")
    print("  = int_0^{2pi} dphi * int_0^{2pi} cos(theta) dtheta")
    print("  = 2*pi * [sin(theta)]_0^{2pi} = 2*pi * 0 = 0")
    print("  => chi = 0 (Euler characteristic of torus)")

    # Numerical verification
    R_val, a_val = 3.0, 1.0
    theta_num = np.linspace(0, 2 * np.pi, 1000)
    K_num = np.cos(theta_num) / (a_val * (R_val + a_val * np.cos(theta_num)))
    dA = a_val * (R_val + a_val * np.cos(theta_num))
    integral = 2 * np.pi * np.trapz(K_num * dA, theta_num)
    print(f"\n  Numerical (R={R_val}, a={a_val}): int K dA = {integral:.6f}")

    # Plot K(theta)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.degrees(theta_num), K_num, 'b-', linewidth=2)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.fill_between(np.degrees(theta_num), K_num, 0,
                    where=K_num > 0, alpha=0.3, color='green', label='K > 0')
    ax.fill_between(np.degrees(theta_num), K_num, 0,
                    where=K_num < 0, alpha=0.3, color='red', label='K < 0')
    ax.set_xlabel('theta (degrees)')
    ax.set_ylabel('K (Gaussian curvature)')
    ax.set_title(f'Gaussian Curvature of Torus (R={R_val}, a={a_val})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex18_torus_curvature.png', dpi=150)
    plt.close()
    print("Plot saved to ex18_torus_curvature.png")


def exercise_7():
    """
    Problem 7: Electromagnetic field tensor.
    E = E0 x_hat, B = B0 z_hat
    (a) Construct F_{mu nu}
    (b) Lorentz invariants
    (c) Lorentz boost v = 0.8c
    """
    print("\n" + "=" * 60)
    print("Problem 7: Electromagnetic Field Tensor")
    print("=" * 60)

    E0, B0 = 1.0, 0.5  # in natural units (c = 1)
    c = 1.0

    # (a) Construct F_{mu nu}
    # Convention: F^{0i} = E^i/c, F^{ij} = -eps_{ijk} B^k
    # F_{mu nu} is antisymmetric
    print("\n(a) E = E0 x_hat, B = B0 z_hat")
    print("  Metric: eta = diag(-1, 1, 1, 1)")

    # F_{\mu\nu} in (+---) or (-+++) convention
    # Using (-+++) (standard particle physics):
    # F^{0i} = E^i, F^{12} = -B^3, F^{23} = -B^1, F^{31} = -B^2
    F_up = np.zeros((4, 4))
    # E = (E0, 0, 0)
    F_up[0, 1] = E0
    F_up[1, 0] = -E0
    # B = (0, 0, B0)
    F_up[1, 2] = -B0
    F_up[2, 1] = B0

    eta = np.diag([-1.0, 1.0, 1.0, 1.0])
    F_down = eta @ F_up @ eta

    print(f"\n  F^{{mu nu}} =")
    for i in range(4):
        print(f"    [{F_up[i, 0]:6.2f} {F_up[i, 1]:6.2f} {F_up[i, 2]:6.2f} {F_up[i, 3]:6.2f}]")

    print(f"\n  F_{{mu nu}} =")
    for i in range(4):
        print(f"    [{F_down[i, 0]:6.2f} {F_down[i, 1]:6.2f} {F_down[i, 2]:6.2f} {F_down[i, 3]:6.2f}]")

    # (b) Lorentz invariants
    print("\n(b) Lorentz invariants:")
    inv1 = np.einsum('ij,ij', F_down, F_up)
    print(f"  F_{{mu nu}} F^{{mu nu}} = 2(B^2 - E^2) = {inv1:.6f}")
    print(f"  Expected: 2*({B0}^2 - {E0}^2) = {2*(B0**2 - E0**2):.6f}")

    # Dual tensor: *F^{mu nu} = (1/2)*eps^{mu nu rho sigma}*F_{rho sigma}
    # Second invariant: (1/2)*eps^{mu nu rho sigma}*F_{mu nu}*F_{rho sigma} = -4*E.B
    inv2 = -4 * np.dot([E0, 0, 0], [0, 0, B0])  # E.B = 0
    print(f"  (1/2)*eps^{{mu nu rho sigma}}*F_{{mu nu}}*F_{{rho sigma}} = -4*E.B = {inv2:.6f}")

    # (c) Lorentz boost in x-direction with v = 0.8c
    print("\n(c) Lorentz boost (v = 0.8c in x-direction):")
    v = 0.8
    gamma = 1.0 / np.sqrt(1 - v**2)
    beta = v

    # Boost matrix
    Lambda_boost = np.eye(4)
    Lambda_boost[0, 0] = gamma
    Lambda_boost[0, 1] = -gamma * beta
    Lambda_boost[1, 0] = -gamma * beta
    Lambda_boost[1, 1] = gamma

    F_prime_up = Lambda_boost @ F_up @ Lambda_boost.T

    # Extract E' and B' from F'^{mu nu}
    Ex_p = F_prime_up[0, 1]
    Ey_p = F_prime_up[0, 2]
    Ez_p = F_prime_up[0, 3]
    Bx_p = F_prime_up[2, 1]  # -F'^{12} no, B' extraction:
    # F'^{12} = -B'_z, F'^{23} = -B'_x, F'^{31} = -B'_y
    Bz_p = -F_prime_up[1, 2]
    Bx_p_actual = -F_prime_up[2, 3]
    By_p = F_prime_up[3, 1]  # F'^{31} = -B'_y => no, let me be careful

    # Standard: F^{0i} = E^i, and B^k = -(1/2)*eps_{kij}*F^{ij}
    E_prime = np.array([F_prime_up[0, 1], F_prime_up[0, 2], F_prime_up[0, 3]])
    B_prime = np.array([
        F_prime_up[3, 2],   # -F^{23} -> wait: F^{23} = -eps_{231}*B^1 = -B^1
        # Actually F^{ij} and B: F^{12} = -B^3, F^{13} = B^2, F^{23} = -B^1
        -F_prime_up[3, 1],  # F^{31} = -F^{13} = -B^2 => B^2 = F^{13}
        0
    ])
    # Let me just compute properly
    B_prime_x = F_prime_up[3, 2]   # This is -(-B'^1) ... let me use direct formula
    # Easier: use the transformation rules
    # E'_parallel = E_parallel
    # E'_perp = gamma*(E_perp + v x B)_perp
    # B'_parallel = B_parallel
    # B'_perp = gamma*(B_perp - (v/c^2) x E)_perp

    E = np.array([E0, 0, 0])
    B = np.array([0, 0, B0])
    v_vec = np.array([v, 0, 0])

    # Parallel and perpendicular to v (x-direction)
    E_par = np.array([E[0], 0, 0])
    E_perp = E - E_par
    B_par = np.array([B[0], 0, 0])
    B_perp = B - B_par

    E_prime_dir = E_par + gamma * (E_perp + np.cross(v_vec, B))
    B_prime_dir = B_par + gamma * (B_perp - np.cross(v_vec, E))

    print(f"  gamma = {gamma:.6f}, beta = {beta}")
    print(f"  E' = ({E_prime_dir[0]:.4f}, {E_prime_dir[1]:.4f}, {E_prime_dir[2]:.4f})")
    print(f"  B' = ({B_prime_dir[0]:.4f}, {B_prime_dir[1]:.4f}, {B_prime_dir[2]:.4f})")

    # Verify invariants are preserved
    inv1_prime = 2 * (np.dot(B_prime_dir, B_prime_dir) - np.dot(E_prime_dir, E_prime_dir))
    inv2_prime = -4 * np.dot(E_prime_dir, B_prime_dir)
    print(f"\n  Invariant check:")
    print(f"  F_{{mu nu}}F^{{mu nu}}: original = {inv1:.4f}, boosted = {inv1_prime:.4f}")
    print(f"  E.B: original = {np.dot(E, B):.4f}, boosted = {np.dot(E_prime_dir, B_prime_dir):.4f}")


def exercise_8():
    """
    Problem 8: Principal axes of stress tensor.
    sigma = [[100, 30, 0], [30, 50, 20], [0, 20, 80]] MPa
    """
    print("\n" + "=" * 60)
    print("Problem 8: Principal Axes of Stress Tensor")
    print("=" * 60)

    sigma = np.array([
        [100, 30, 0],
        [30, 50, 20],
        [0, 20, 80]
    ], dtype=float)

    print("\nsigma_ij (MPa):")
    for i in range(3):
        print(f"  [{sigma[i, 0]:6.1f} {sigma[i, 1]:6.1f} {sigma[i, 2]:6.1f}]")

    # (a) Eigenvalues (principal stresses) and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(sigma)
    # Sort by descending eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    print("\n(a) Principal stresses (eigenvalues):")
    for i in range(3):
        print(f"  sigma_{i + 1} = {eigenvalues[i]:.4f} MPa")

    print("\n  Principal directions (eigenvectors):")
    for i in range(3):
        v = eigenvectors[:, i]
        print(f"  n_{i + 1} = ({v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f})")

    # Verify: sigma * n = lambda * n
    for i in range(3):
        residual = np.linalg.norm(sigma @ eigenvectors[:, i] - eigenvalues[i] * eigenvectors[:, i])
        print(f"  |sigma*n_{i + 1} - sigma_{i + 1}*n_{i + 1}| = {residual:.2e}")

    # (b) Maximum shear stress
    print("\n(b) Maximum shear stress:")
    s1, s2, s3 = eigenvalues
    tau_max = (s1 - s3) / 2
    print(f"  tau_max = (sigma_1 - sigma_3)/2 = ({s1:.4f} - {s3:.4f})/2 = {tau_max:.4f} MPa")

    # All three shear stresses
    tau_12 = abs(s1 - s2) / 2
    tau_23 = abs(s2 - s3) / 2
    tau_13 = abs(s1 - s3) / 2
    print(f"  tau_12 = |sigma_1 - sigma_2|/2 = {tau_12:.4f} MPa")
    print(f"  tau_23 = |sigma_2 - sigma_3|/2 = {tau_23:.4f} MPa")
    print(f"  tau_13 = |sigma_1 - sigma_3|/2 = {tau_13:.4f} MPa")

    # (c) von Mises stress
    print("\n(c) von Mises stress:")
    sigma_vm = np.sqrt(0.5 * ((s1 - s2)**2 + (s2 - s3)**2 + (s3 - s1)**2))
    print(f"  sigma_v = sqrt((1/2)*[(s1-s2)^2 + (s2-s3)^2 + (s3-s1)^2])")
    print(f"          = {sigma_vm:.4f} MPa")

    # Alternative formula using original components
    sigma_vm_alt = np.sqrt(0.5 * ((sigma[0, 0] - sigma[1, 1])**2 +
                                   (sigma[1, 1] - sigma[2, 2])**2 +
                                   (sigma[2, 2] - sigma[0, 0])**2 +
                                   6 * (sigma[0, 1]**2 + sigma[1, 2]**2 + sigma[0, 2]**2)))
    print(f"  Alternative (from components): {sigma_vm_alt:.4f} MPa")
    print(f"  Match: {np.isclose(sigma_vm, sigma_vm_alt)}")

    # Stress invariants
    I1 = np.trace(sigma)
    I2 = 0.5 * (np.trace(sigma)**2 - np.trace(sigma @ sigma))
    I3 = np.linalg.det(sigma)
    print(f"\n  Stress invariants:")
    print(f"    I1 (trace) = {I1:.4f} MPa")
    print(f"    I2 = {I2:.4f} MPa^2")
    print(f"    I3 (det) = {I3:.4f} MPa^3")

    # Mohr's circle visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    theta_plot = np.linspace(0, 2 * np.pi, 200)

    # Three Mohr's circles
    for (sa, sb), color, label in [
        ((s1, s2), 'b', f'1-2 plane'),
        ((s2, s3), 'r', f'2-3 plane'),
        ((s1, s3), 'g', f'1-3 plane')
    ]:
        center = (sa + sb) / 2
        radius = abs(sa - sb) / 2
        ax.plot(center + radius * np.cos(theta_plot),
                radius * np.sin(theta_plot),
                color=color, linewidth=2, label=label)

    ax.plot(eigenvalues, [0, 0, 0], 'ko', markersize=8)
    for i, s in enumerate(eigenvalues):
        ax.annotate(f'$\\sigma_{i + 1}$={s:.1f}', (s, 0), textcoords='offset points',
                    xytext=(0, 10), ha='center')

    ax.set_xlabel('Normal Stress (MPa)')
    ax.set_ylabel('Shear Stress (MPa)')
    ax.set_title("Mohr's Circles for 3D Stress State")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.axhline(y=0, color='k', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('ex18_mohrs_circles.png', dpi=150)
    plt.close()
    print("\nPlot saved to ex18_mohrs_circles.png")


if __name__ == "__main__":
    print("=== Exercise 1 ===")
    exercise_1()
    print("\n=== Exercise 2 ===")
    exercise_2()
    print("\n=== Exercise 3 ===")
    exercise_3()
    print("\n=== Exercise 4 ===")
    exercise_4()
    print("\n=== Exercise 5 ===")
    exercise_5()
    print("\n=== Exercise 6 ===")
    exercise_6()
    print("\n=== Exercise 7 ===")
    exercise_7()
    print("\n=== Exercise 8 ===")
    exercise_8()
    print("\nAll exercises completed!")
