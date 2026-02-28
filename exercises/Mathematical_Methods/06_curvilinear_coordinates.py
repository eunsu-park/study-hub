"""
Exercise Solutions: Lesson 06 - Curvilinear Coordinates and Multiple Integrals
Mathematical Methods for Physical Sciences

Covers: Jacobians, integration order, cylindrical/spherical integrals,
        toroidal coordinates, surface charge distributions
"""

import numpy as np
import sympy as sp
from scipy import integrate as sci_integrate


def exercise_1_jacobian():
    """
    Problem 1: Calculate the Jacobian for:
    (a) Parabolic coordinates: x = u*v, y = (u^2 - v^2)/2
    (b) Log-polar: x = e^u * cos(v), y = e^u * sin(v)
    """
    print("=" * 60)
    print("Problem 1: Jacobian Calculation")
    print("=" * 60)

    u, v = sp.symbols('u v')

    # (a) Parabolic coordinates
    x_a = u * v
    y_a = (u**2 - v**2) / 2

    J_a = sp.Matrix([[sp.diff(x_a, u), sp.diff(x_a, v)],
                     [sp.diff(y_a, u), sp.diff(y_a, v)]])
    det_J_a = sp.simplify(J_a.det())

    print(f"\n(a) Parabolic: x = uv, y = (u^2 - v^2)/2")
    print(f"  Jacobian matrix:\n{J_a}")
    print(f"  |J| = {det_J_a}")
    print(f"  |J| = -(u^2 + v^2)")
    print(f"  |det J| = u^2 + v^2")

    # (b) Log-polar coordinates
    x_b = sp.exp(u) * sp.cos(v)
    y_b = sp.exp(u) * sp.sin(v)

    J_b = sp.Matrix([[sp.diff(x_b, u), sp.diff(x_b, v)],
                     [sp.diff(y_b, u), sp.diff(y_b, v)]])
    det_J_b = sp.simplify(J_b.det())

    print(f"\n(b) Log-polar: x = e^u*cos(v), y = e^u*sin(v)")
    print(f"  Jacobian matrix:\n{J_b}")
    print(f"  |J| = {det_J_b}")
    print(f"  Note: r = e^u, so this is just r^2 in disguise (e^{2u})")


def exercise_2_integration_order():
    """
    Problem 2: Change the order of integration for:
    integral_0^1 integral_0^{sqrt(x)} f(x,y) dy dx
    """
    print("\n" + "=" * 60)
    print("Problem 2: Changing Order of Integration")
    print("=" * 60)

    print(f"\nOriginal: int_0^1 int_0^sqrt(x) f(x,y) dy dx")
    print(f"\nRegion description:")
    print(f"  0 <= x <= 1")
    print(f"  0 <= y <= sqrt(x), i.e., y^2 <= x")
    print(f"\nSketching the region:")
    print(f"  Bounded by y = 0, y = sqrt(x) (i.e., x = y^2), and x = 1")
    print(f"\nChanging order: fix y first, then x")
    print(f"  0 <= y <= 1")
    print(f"  y^2 <= x <= 1")
    print(f"\nNew integral: int_0^1 int_{y^2}^1 f(x,y) dx dy")

    # Verify with f = 1 (area computation)
    x, y = sp.symbols('x y')

    area_orig = sp.integrate(sp.integrate(1, (y, 0, sp.sqrt(x))), (x, 0, 1))
    area_new = sp.integrate(sp.integrate(1, (x, y**2, 1)), (y, 0, 1))

    print(f"\nVerification with f = 1 (area):")
    print(f"  Original order: A = {area_orig}")
    print(f"  New order:      A = {area_new}")
    print(f"  Match: {area_orig == area_new}")

    # Another verification: f = x*y
    f_test = x * y
    I_orig = sp.integrate(sp.integrate(f_test, (y, 0, sp.sqrt(x))), (x, 0, 1))
    I_new = sp.integrate(sp.integrate(f_test, (x, y**2, 1)), (y, 0, 1))

    print(f"\n  With f = x*y:")
    print(f"    Original order: I = {I_orig}")
    print(f"    New order:      I = {I_new}")
    print(f"    Match: {sp.simplify(I_orig - I_new) == 0}")


def exercise_3_cylindrical_integral():
    """
    Problem 3: Find the volume and moment of inertia (about z-axis) of the cone
    z = sqrt(x^2 + y^2), 0 <= z <= h using cylindrical coordinates.
    """
    print("\n" + "=" * 60)
    print("Problem 3: Cone Volume and Moment of Inertia (Cylindrical)")
    print("=" * 60)

    r, theta, z, h, rho_0 = sp.symbols('r theta z h rho_0', positive=True)

    # Cone: z = r (in cylindrical), 0 <= z <= h
    # => r <= z, so for fixed z: 0 <= r <= z
    # For z: 0 <= z <= h

    print(f"\nCone: z = sqrt(x^2+y^2) = r, 0 <= z <= h")
    print(f"In cylindrical: 0 <= theta <= 2*pi, r <= z <= h, 0 <= r <= h")
    print(f"(or equivalently: 0 <= z <= h, 0 <= r <= z)")

    # Volume
    # V = int_0^{2pi} int_0^h int_r^h r dz dr dtheta
    #   = 2*pi * int_0^h r*(h-r) dr
    #   = 2*pi * [h*r^2/2 - r^3/3]_0^h
    #   = 2*pi * (h^3/2 - h^3/3) = 2*pi*h^3/6 = pi*h^3/3

    V = sp.integrate(
        sp.integrate(
            sp.integrate(r, (z, r, h)),
            (r, 0, h)),
        (theta, 0, 2 * sp.pi))

    print(f"\nVolume = {V}")
    print(f"       = pi*h^3/3 (matches (1/3)*pi*R^2*H with R = H = h)")

    # Moment of inertia about z-axis
    # I_z = int rho * r^2 dV (with uniform density rho_0)
    I_z = sp.integrate(
        sp.integrate(
            sp.integrate(rho_0 * r**2 * r, (z, r, h)),  # r^2 * r dr dz dtheta
            (r, 0, h)),
        (theta, 0, 2 * sp.pi))

    I_z_simplified = sp.simplify(I_z)
    mass = rho_0 * V

    print(f"\nMoment of inertia I_z = {I_z_simplified}")
    print(f"Mass M = rho_0 * V = {sp.simplify(mass)}")
    print(f"I_z / (M*h^2) = {sp.simplify(I_z / (mass * h**2))}")
    print(f"I_z = (3/10)*M*h^2")


def exercise_4_spherical_center_of_mass():
    """
    Problem 4: Find the center of mass of a uniform solid hemisphere
    x^2 + y^2 + z^2 <= R^2, z >= 0 using spherical coordinates.
    """
    print("\n" + "=" * 60)
    print("Problem 4: Center of Mass of Hemisphere (Spherical)")
    print("=" * 60)

    r, theta, phi, R = sp.symbols('r theta phi R', positive=True)

    # By symmetry: x_cm = y_cm = 0, need only z_cm
    # z = r*cos(theta)
    # dV = r^2 * sin(theta) dr dtheta dphi

    print(f"\nUniform hemisphere: r <= R, 0 <= theta <= pi/2, 0 <= phi <= 2*pi")
    print(f"By symmetry: x_cm = y_cm = 0")

    # Volume
    V = sp.integrate(
        sp.integrate(
            sp.integrate(r**2 * sp.sin(theta), (r, 0, R)),
            (theta, 0, sp.pi / 2)),
        (phi, 0, 2 * sp.pi))
    print(f"\nVolume = {V} = (2/3)*pi*R^3")

    # z_cm = (1/V) * integral z dV
    z_integral = sp.integrate(
        sp.integrate(
            sp.integrate(r * sp.cos(theta) * r**2 * sp.sin(theta), (r, 0, R)),
            (theta, 0, sp.pi / 2)),
        (phi, 0, 2 * sp.pi))

    z_cm = sp.simplify(z_integral / V)
    print(f"\nint z dV = {z_integral}")
    print(f"z_cm = {z_cm}")
    print(f"     = 3R/8")
    print(f"\nCenter of mass is at (0, 0, 3R/8)")
    print(f"(Below the geometric center R/2, shifted toward the flat face)")


def exercise_5_toroidal_coordinates():
    """
    Problem 5: For toroidal coordinates:
    x = (R + a*cos(theta))*cos(phi)
    y = (R + a*cos(theta))*sin(phi)
    z = a*sin(theta)
    Find the scale factors and surface area element.
    """
    print("\n" + "=" * 60)
    print("Problem 5: Toroidal Coordinate Scale Factors")
    print("=" * 60)

    theta, phi, R, a = sp.symbols('theta phi R a', positive=True)

    # Torus parameterization
    x = (R + a * sp.cos(theta)) * sp.cos(phi)
    y = (R + a * sp.cos(theta)) * sp.sin(phi)
    z = a * sp.sin(theta)

    print(f"\nTorus parameterization:")
    print(f"  x = (R + a*cos(theta))*cos(phi)")
    print(f"  y = (R + a*cos(theta))*sin(phi)")
    print(f"  z = a*sin(theta)")

    # Scale factors: h_i = |dr/dq_i|
    # h_theta
    dx_dtheta = sp.diff(x, theta)
    dy_dtheta = sp.diff(y, theta)
    dz_dtheta = sp.diff(z, theta)
    h_theta = sp.sqrt(dx_dtheta**2 + dy_dtheta**2 + dz_dtheta**2)
    h_theta = sp.simplify(h_theta)

    # h_phi
    dx_dphi = sp.diff(x, phi)
    dy_dphi = sp.diff(y, phi)
    dz_dphi = sp.diff(z, phi)
    h_phi = sp.sqrt(dx_dphi**2 + dy_dphi**2 + dz_dphi**2)
    h_phi = sp.simplify(h_phi)

    print(f"\nScale factors:")
    print(f"  h_theta = {h_theta}")
    print(f"  h_phi   = {h_phi}")

    # Surface area element
    dA = h_theta * h_phi
    dA_simplified = sp.simplify(sp.trigsimp(dA))
    print(f"\n  dA = h_theta * h_phi * dtheta * dphi")
    print(f"     = {dA_simplified} * dtheta * dphi")

    # Total surface area
    A = sp.integrate(
        sp.integrate(dA_simplified, (theta, 0, 2 * sp.pi)),
        (phi, 0, 2 * sp.pi))
    A_simplified = sp.simplify(A)

    print(f"\nSurface area = {A_simplified}")
    print(f"             = 4*pi^2*a*R")
    print(f"             = (2*pi*a) * (2*pi*R)")
    print(f"  [circumference of tube cross-section] * [path length of center]")


def exercise_6_surface_charge():
    """
    Problem 6: A hemisphere of radius R has surface charge sigma = sigma_0 * cos(theta).
    Find the total charge and the electric dipole moment.
    """
    print("\n" + "=" * 60)
    print("Problem 6: Surface Charge Distribution")
    print("=" * 60)

    theta, phi, R, sigma_0 = sp.symbols('theta phi R sigma_0', positive=True)

    # Surface element on sphere: dA = R^2 * sin(theta) dtheta dphi
    sigma = sigma_0 * sp.cos(theta)

    print(f"\nHemisphere: r = R, 0 <= theta <= pi/2, 0 <= phi <= 2*pi")
    print(f"Surface charge: sigma = sigma_0 * cos(theta)")
    print(f"Surface element: dA = R^2 * sin(theta) dtheta dphi")

    # Total charge
    Q = sp.integrate(
        sp.integrate(sigma * R**2 * sp.sin(theta), (theta, 0, sp.pi / 2)),
        (phi, 0, 2 * sp.pi))

    print(f"\nTotal charge Q = integral sigma dA")
    print(f"  = int_0^2pi int_0^{sp.pi}/2 sigma_0*cos(theta)*R^2*sin(theta) dtheta dphi")
    print(f"  = {Q}")
    print(f"  = pi*R^2*sigma_0")

    # Electric dipole moment p = integral r * sigma dA
    # By symmetry, p_x = p_y = 0 (azimuthal symmetry)
    # p_z = integral z * sigma dA = integral R*cos(theta) * sigma dA

    p_z = sp.integrate(
        sp.integrate(
            R * sp.cos(theta) * sigma * R**2 * sp.sin(theta),
            (theta, 0, sp.pi / 2)),
        (phi, 0, 2 * sp.pi))

    print(f"\nDipole moment (by symmetry, only z-component):")
    print(f"  p_z = integral z * sigma dA")
    print(f"      = integral R*cos(theta) * sigma_0*cos(theta) * R^2*sin(theta) dtheta dphi")
    print(f"      = {p_z}")
    print(f"      = (2/3)*pi*R^3*sigma_0")

    # Numerical verification
    R_val = 1.0
    sigma_0_val = 1.0

    Q_numerical, _ = sci_integrate.dblquad(
        lambda theta, phi: sigma_0_val * np.cos(theta) * R_val**2 * np.sin(theta),
        0, 2 * np.pi,
        0, np.pi / 2
    )
    print(f"\nNumerical verification (R=1, sigma_0=1):")
    print(f"  Q = {Q_numerical:.8f} (exact: pi = {np.pi:.8f})")

    p_z_numerical, _ = sci_integrate.dblquad(
        lambda theta, phi: R_val * np.cos(theta) * sigma_0_val * np.cos(theta) * R_val**2 * np.sin(theta),
        0, 2 * np.pi,
        0, np.pi / 2
    )
    print(f"  p_z = {p_z_numerical:.8f} (exact: 2*pi/3 = {2*np.pi/3:.8f})")


if __name__ == "__main__":
    exercise_1_jacobian()
    exercise_2_integration_order()
    exercise_3_cylindrical_integral()
    exercise_4_spherical_center_of_mass()
    exercise_5_toroidal_coordinates()
    exercise_6_surface_charge()
