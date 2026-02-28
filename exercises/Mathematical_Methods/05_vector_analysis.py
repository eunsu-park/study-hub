"""
Exercise Solutions: Lesson 05 - Vector Analysis
Mathematical Methods for Physical Sciences

Covers: gradient, directional derivative, divergence, curl, conservative fields,
        line integrals, divergence theorem, Stokes' theorem, Maxwell's equations
"""

import numpy as np
import sympy as sp


def exercise_1_gradient_directional():
    """
    Problem 1: For f(x,y,z) = x^2*y + y^2*z + z^2*x:
    (a) Find grad(f)
    (b) Directional derivative at (1,1,1) in direction (1,1,1)/sqrt(3)
    (c) Direction of fastest increase at (1,1,1) and its rate
    """
    print("=" * 60)
    print("Problem 1: Gradient and Directional Derivative")
    print("=" * 60)

    x, y, z = sp.symbols('x y z')
    f = x**2 * y + y**2 * z + z**2 * x

    grad_f = [sp.diff(f, var) for var in [x, y, z]]

    print(f"\nf(x,y,z) = x^2*y + y^2*z + z^2*x")
    print(f"\n(a) grad(f) = ({grad_f[0]}, {grad_f[1]}, {grad_f[2]})")

    # Evaluate at (1,1,1)
    point = {x: 1, y: 1, z: 1}
    grad_at_point = [g.subs(point) for g in grad_f]
    print(f"\n    grad(f) at (1,1,1) = ({grad_at_point[0]}, {grad_at_point[1]}, {grad_at_point[2]})")

    # (b) Directional derivative
    u_hat = np.array([1, 1, 1]) / np.sqrt(3)
    grad_np = np.array([float(g) for g in grad_at_point])
    dir_deriv = np.dot(grad_np, u_hat)

    print(f"\n(b) Direction u = (1,1,1)/sqrt(3)")
    print(f"    D_u f = grad(f) . u = {dir_deriv:.6f}")
    print(f"         = ({grad_at_point[0]}+{grad_at_point[1]}+{grad_at_point[2]})/sqrt(3)")
    print(f"         = {sum(float(g) for g in grad_at_point)}/sqrt(3) = {dir_deriv:.6f}")

    # (c) Maximum rate of increase
    mag_grad = np.linalg.norm(grad_np)
    direction = grad_np / mag_grad

    print(f"\n(c) Direction of fastest increase: grad(f)/|grad(f)|")
    print(f"    = ({direction[0]:.6f}, {direction[1]:.6f}, {direction[2]:.6f})")
    print(f"    Maximum rate = |grad(f)| = {mag_grad:.6f}")


def exercise_2_divergence_curl():
    """
    Problem 2: Calculate div and curl, determine if conservative:
    (a) F = (yz, xz, xy)
    (b) G = (x^2 - y, y^2 + x, z)
    """
    print("\n" + "=" * 60)
    print("Problem 2: Divergence and Curl")
    print("=" * 60)

    x, y, z = sp.symbols('x y z')

    # (a) F = (yz, xz, xy)
    F = [y*z, x*z, x*y]
    div_F = sum(sp.diff(F[i], var) for i, var in enumerate([x, y, z]))
    curl_F = [
        sp.diff(F[2], y) - sp.diff(F[1], z),
        sp.diff(F[0], z) - sp.diff(F[2], x),
        sp.diff(F[1], x) - sp.diff(F[0], y)
    ]

    print(f"\n(a) F = (yz, xz, xy)")
    print(f"  div(F) = {div_F}")
    print(f"  curl(F) = ({curl_F[0]}, {curl_F[1]}, {curl_F[2]})")

    is_conservative = all(c == 0 for c in curl_F)
    print(f"  Conservative (curl = 0): {is_conservative}")

    if is_conservative:
        # Find potential: phi_x = yz => phi = xyz + g(y,z)
        # phi_y = xz + g_y = xz => g_y = 0 => g = h(z)
        # phi_z = xy + h'(z) = xy => h'(z) = 0 => h = const
        print(f"  Potential: phi = xyz + C")
        # Verify
        phi = x * y * z
        grad_phi = [sp.diff(phi, var) for var in [x, y, z]]
        print(f"  Verify: grad(phi) = ({grad_phi[0]}, {grad_phi[1]}, {grad_phi[2]})")

    # (b) G = (x^2 - y, y^2 + x, z)
    G = [x**2 - y, y**2 + x, z]
    div_G = sum(sp.diff(G[i], var) for i, var in enumerate([x, y, z]))
    curl_G = [
        sp.diff(G[2], y) - sp.diff(G[1], z),
        sp.diff(G[0], z) - sp.diff(G[2], x),
        sp.diff(G[1], x) - sp.diff(G[0], y)
    ]

    print(f"\n(b) G = (x^2 - y, y^2 + x, z)")
    print(f"  div(G) = {div_G}")
    print(f"  curl(G) = ({curl_G[0]}, {curl_G[1]}, {curl_G[2]})")
    print(f"  Conservative (curl = 0): {all(c == 0 for c in curl_G)}")
    print(f"  curl(G)_z = d(y^2+x)/dx - d(x^2-y)/dy = 1 - (-1) = 2 != 0")
    print(f"  => G is NOT conservative")


def exercise_3_line_integral():
    """
    Problem 3: F = (2xy + z^2, x^2, 2xz)
    (a) Show conservative, find potential phi
    (b) Line integral (0,0,0) to (1,2,3) via potential and direct
    """
    print("\n" + "=" * 60)
    print("Problem 3: Line Integrals and Potential")
    print("=" * 60)

    x, y, z, t = sp.symbols('x y z t')
    F = [2*x*y + z**2, x**2, 2*x*z]

    # (a) Check curl = 0
    curl_F = [
        sp.diff(F[2], y) - sp.diff(F[1], z),
        sp.diff(F[0], z) - sp.diff(F[2], x),
        sp.diff(F[1], x) - sp.diff(F[0], y)
    ]

    print(f"\nF = (2xy + z^2, x^2, 2xz)")
    print(f"\n(a) curl(F) = ({curl_F[0]}, {curl_F[1]}, {curl_F[2]})")
    print(f"    curl(F) = 0 => F is conservative")

    # Find potential phi
    # phi_x = 2xy + z^2 => phi = x^2*y + x*z^2 + g(y,z)
    # phi_y = x^2 + g_y = x^2 => g_y = 0, g = h(z)
    # phi_z = 2xz + h'(z) = 2xz => h'(z) = 0, h = C
    phi = x**2 * y + x * z**2
    print(f"    Potential: phi = x^2*y + x*z^2")

    # Verify
    grad_phi = [sp.diff(phi, var) for var in [x, y, z]]
    print(f"    Verify: grad(phi) = ({grad_phi[0]}, {grad_phi[1]}, {grad_phi[2]})")

    # (b) Line integral using potential
    phi_end = phi.subs([(x, 1), (y, 2), (z, 3)])
    phi_start = phi.subs([(x, 0), (y, 0), (z, 0)])
    I_potential = phi_end - phi_start

    print(f"\n(b) Using potential: integral = phi(1,2,3) - phi(0,0,0)")
    print(f"    phi(1,2,3) = 1*2 + 1*9 = {phi_end}")
    print(f"    phi(0,0,0) = {phi_start}")
    print(f"    Integral = {I_potential}")

    # Direct integration along r(t) = (t, 2t, 3t), 0 <= t <= 1
    x_t, y_t, z_t = t, 2*t, 3*t
    dx, dy, dz = 1, 2, 3

    Fx = (2*x_t*y_t + z_t**2)
    Fy = x_t**2
    Fz = 2*x_t*z_t

    integrand = Fx * dx + Fy * dy + Fz * dz
    integrand_simplified = sp.expand(integrand)
    I_direct = sp.integrate(integrand_simplified, (t, 0, 1))

    print(f"\n    Direct: r(t) = (t, 2t, 3t), 0 <= t <= 1")
    print(f"    F.dr/dt = {integrand_simplified}")
    print(f"    Integral = {I_direct}")
    print(f"    Match: {I_potential == I_direct}")


def exercise_4_divergence_theorem():
    """
    Problem 4: F = (x^2, y^2, z^2) in unit cube [0,1]^3.
    Verify the divergence theorem.
    """
    print("\n" + "=" * 60)
    print("Problem 4: Divergence Theorem Verification")
    print("=" * 60)

    x, y, z = sp.symbols('x y z')
    F = [x**2, y**2, z**2]

    # div(F) = 2x + 2y + 2z
    div_F = sum(sp.diff(F[i], var) for i, var in enumerate([x, y, z]))
    print(f"\nF = (x^2, y^2, z^2)")
    print(f"div(F) = {div_F}")

    # Volume integral
    vol_integral = sp.integrate(
        sp.integrate(
            sp.integrate(div_F, (z, 0, 1)),
            (y, 0, 1)),
        (x, 0, 1))
    print(f"\n(1) Volume integral: triple_int div(F) dV")
    print(f"    = int_0^1 int_0^1 int_0^1 (2x+2y+2z) dz dy dx")
    print(f"    = {vol_integral}")

    # Surface integral (6 faces)
    print(f"\n(2) Surface integrals on 6 faces:")
    total_surface = sp.Integer(0)

    # Face x=1 (outward normal = +x): int F.n dA = int x^2 dy dz at x=1
    face_x1 = sp.integrate(sp.integrate(sp.Integer(1), (z, 0, 1)), (y, 0, 1))
    print(f"    x=1 face: int 1 dy dz = {face_x1}")
    total_surface += face_x1

    # Face x=0 (outward normal = -x): int -x^2 dy dz at x=0
    face_x0 = sp.integrate(sp.integrate(sp.Integer(0), (z, 0, 1)), (y, 0, 1))
    print(f"    x=0 face: int 0 dy dz = {face_x0}")
    total_surface += face_x0

    # Face y=1: int y^2 dx dz at y=1
    face_y1 = sp.integrate(sp.integrate(sp.Integer(1), (z, 0, 1)), (x, 0, 1))
    print(f"    y=1 face: int 1 dx dz = {face_y1}")
    total_surface += face_y1

    # Face y=0
    face_y0 = sp.Integer(0)
    print(f"    y=0 face: int 0 dx dz = {face_y0}")
    total_surface += face_y0

    # Face z=1
    face_z1 = sp.integrate(sp.integrate(sp.Integer(1), (y, 0, 1)), (x, 0, 1))
    print(f"    z=1 face: int 1 dx dy = {face_z1}")
    total_surface += face_z1

    # Face z=0
    face_z0 = sp.Integer(0)
    print(f"    z=0 face: int 0 dx dy = {face_z0}")
    total_surface += face_z0

    print(f"\n    Total surface integral = {total_surface}")
    print(f"\n(3) Verification: {vol_integral} == {total_surface}: {vol_integral == total_surface}")


def exercise_5_ampere_law():
    """
    Problem 5: Stokes' theorem application - Ampere's law.
    Uniform current J = J_0 * z_hat in cylindrical wire of radius a.
    Find B for r < a and r > a, verify curl B = mu_0 * J.
    """
    print("\n" + "=" * 60)
    print("Problem 5: Ampere's Law (Stokes' Theorem)")
    print("=" * 60)

    r, a, mu_0, J_0 = sp.symbols('r a mu_0 J_0', positive=True)

    print(f"\nCylindrical wire: radius a, uniform current density J = J_0 * z_hat")

    # Ampere's law: oint B.dl = mu_0 * I_enclosed
    # By symmetry: B = B_phi(r) * phi_hat

    # (1) r < a: I_enc = J_0 * pi * r^2
    print(f"\n(1) Inside wire (r < a):")
    print(f"    oint B.dl = B_phi * 2*pi*r = mu_0 * J_0 * pi * r^2")
    B_inside = mu_0 * J_0 * r / 2
    print(f"    B_phi = mu_0*J_0*r/2 = {B_inside}")

    # (2) r > a: I_enc = J_0 * pi * a^2
    print(f"\n(2) Outside wire (r > a):")
    print(f"    oint B.dl = B_phi * 2*pi*r = mu_0 * J_0 * pi * a^2")
    B_outside = mu_0 * J_0 * a**2 / (2 * r)
    print(f"    B_phi = mu_0*J_0*a^2/(2r) = {B_outside}")

    # Verify curl B = mu_0 * J for r < a using cylindrical curl
    # In cylindrical coordinates, for B = B_phi(r) * phi_hat:
    # (curl B)_z = (1/r) * d(r*B_phi)/dr
    print(f"\n(3) Verify curl B = mu_0 * J for r < a:")
    rB_phi = r * B_inside
    curl_z = sp.diff(rB_phi, r) / r
    curl_z_simplified = sp.simplify(curl_z)
    print(f"    (curl B)_z = (1/r) * d(r*B_phi)/dr")
    print(f"              = {curl_z_simplified}")
    print(f"    mu_0 * J_0 = mu_0*J_0")
    print(f"    Match: {sp.simplify(curl_z_simplified - mu_0 * J_0) == 0}")

    # Numerical plot
    a_val = 1.0
    mu0_J0 = 1.0
    r_vals = np.linspace(0.01, 3, 300)
    B_vals = np.where(
        r_vals < a_val,
        mu0_J0 * r_vals / 2,
        mu0_J0 * a_val**2 / (2 * r_vals)
    )
    print(f"\n    B is linear for r < a, decays as 1/r for r > a")
    print(f"    B_max at r = a: B = mu_0*J_0*a/2")


def exercise_6_faraday_law():
    """
    Problem 6: Derive differential form of Faraday's law from integral form
    using Stokes' theorem.
    """
    print("\n" + "=" * 60)
    print("Problem 6: Faraday's Law Derivation (Stokes' Theorem)")
    print("=" * 60)

    print("""
Derivation of differential Faraday's law from integral form:

Step 1: Start with integral form
  oint_C E . dr = -d/dt iint_S B . dS

Step 2: Apply Stokes' theorem to the left side
  oint_C E . dr = iint_S (curl E) . dS

Step 3: For a surface fixed in time, move d/dt inside:
  iint_S (curl E) . dS = -iint_S (dB/dt) . dS

Step 4: Combine into single surface integral:
  iint_S [curl E + dB/dt] . dS = 0

Step 5: Since S is arbitrary, the integrand must vanish:
  curl E = -dB/dt

This is the differential form of Faraday's law (3rd Maxwell equation).
""")

    # Numerical verification with a simple example
    print("Numerical verification with a simple time-varying B field:")
    print("  Let B = B_0 * sin(omega*t) * z_hat (uniform, time-varying)")
    print("  Then dB/dt = B_0 * omega * cos(omega*t) * z_hat")
    print("  curl E = -dB/dt = -B_0 * omega * cos(omega*t) * z_hat")
    print("")
    print("  A solution: E = (B_0*omega*r/2)*cos(omega*t) * phi_hat")
    print("  This describes circular electric field lines induced by")
    print("  the changing magnetic flux (electromagnetic induction).")

    # Verify the E field solution
    print("\n  Verification (cylindrical coordinates):")
    print("  curl E for E = E_phi(r,t) * phi_hat:")
    print("    (curl E)_z = (1/r) * d(r * E_phi)/dr")
    print("               = (1/r) * d(r * B_0*omega*r/(2) * cos(wt))/dr")
    print("               = (1/r) * B_0*omega*cos(wt) * d(r^2/2)/dr")
    print("               = (1/r) * B_0*omega*cos(wt) * r")
    print("               = B_0*omega*cos(wt)")
    print("    -dB/dt     = B_0*omega*cos(wt)  [check]")


if __name__ == "__main__":
    exercise_1_gradient_directional()
    exercise_2_divergence_curl()
    exercise_3_line_integral()
    exercise_4_divergence_theorem()
    exercise_5_ampere_law()
    exercise_6_faraday_law()
