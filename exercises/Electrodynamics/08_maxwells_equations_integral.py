"""
Exercises for Lesson 08: Maxwell's Equations -- Integral Form
Topic: Electrodynamics
Solutions to practice problems from the lesson.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Constants
epsilon_0 = 8.854e-12
mu_0 = 4.0 * np.pi * 1e-7
c = 1.0 / np.sqrt(mu_0 * epsilon_0)


def exercise_1():
    """
    Exercise 1: Energy in a Capacitor
    Parallel plate capacitor. Verify W = (1/2)CV^2 equals W = (eps_0/2)*int(E^2 dV).
    """
    A = 100e-4     # 100 cm^2
    d = 1e-3       # 1 mm
    V_cap = 1000   # 1000 V

    # Method (a): W = (1/2)*C*V^2
    C = epsilon_0 * A / d
    W_cv = 0.5 * C * V_cap**2

    # Method (b): W = (eps_0/2)*int(E^2 dV)
    E = V_cap / d       # uniform field between plates
    volume = A * d       # volume of the gap
    W_field = 0.5 * epsilon_0 * E**2 * volume

    print(f"  Capacitor: A = {A*1e4:.0f} cm^2, d = {d*1e3:.0f} mm, V = {V_cap} V")
    print(f"  Capacitance C = {C*1e12:.4f} pF")
    print(f"  Electric field E = V/d = {E:.2e} V/m")
    print()
    print(f"  Method (a): W = (1/2)CV^2 = {W_cv:.6e} J")
    print(f"  Method (b): W = (eps_0/2)*E^2*Vol = {W_field:.6e} J")
    print(f"  Agreement: {abs(W_cv - W_field)/W_cv:.2e} relative error")
    print(f"  Energy density: u = {0.5*epsilon_0*E**2:.4e} J/m^3")


def exercise_2():
    """
    Exercise 2: Poynting Vector of a Charging Capacitor
    During charging, S points radially inward at the edge.
    Verify total inward flux = rate of energy increase.
    """
    R = 0.05     # plate radius (5 cm)
    d = 0.002    # gap (2 mm)
    I = 0.5      # charging current (A)
    A = np.pi * R**2

    # E between plates: E = sigma/eps_0 = Q/(eps_0*A)
    # dE/dt = I/(eps_0*A) = J_d/eps_0 where J_d = I/A
    dE_dt = I / (epsilon_0 * A)

    # B at the edge (r = R): B = mu_0*I/(2*pi*R)
    B_edge = mu_0 * I / (2 * np.pi * R)

    # E at a specific instant: assume t=0 has Q=0, so E = I*t/(eps_0*A)
    t = 1e-6  # 1 microsecond
    E_val = I * t / (epsilon_0 * A)

    # Poynting vector at the edge: S = E x H = E x (B/mu_0)
    # S points radially inward: S = E * B/mu_0
    # But actually S = (1/mu_0)*(E x B), and at the edge both E and B are perpendicular
    # E is along z (between plates), B is along phi (circling the axis)
    # S = ExB/mu_0 points radially inward (-r_hat direction)
    S_edge = E_val * B_edge / mu_0

    # Total inward flux through the cylindrical side surface:
    # Flux = S * (lateral area) = S * (2*pi*R*d)
    P_inward = S_edge * 2 * np.pi * R * d

    # Rate of energy increase: d/dt[(eps_0/2)*E^2*Volume]
    # = eps_0*E*(dE/dt)*A*d
    P_increase = epsilon_0 * E_val * dE_dt * A * d

    print(f"  Charging capacitor: R = {R*100:.0f} cm, d = {d*1e3:.0f} mm, I = {I} A")
    print(f"  At t = {t*1e6:.0f} us:")
    print(f"    E between plates: {E_val:.4f} V/m")
    print(f"    B at edge: {B_edge*1e6:.4f} uT")
    print(f"    Poynting magnitude at edge: {S_edge:.4e} W/m^2")
    print(f"    Total inward power (Poynting): {P_inward:.6e} W")
    print(f"    Rate of energy increase: {P_increase:.6e} W")
    print(f"    Ratio: {P_inward/P_increase:.6f} (should be 1)")


def exercise_3():
    """
    Exercise 3: Solar Radiation Pressure
    I = 1361 W/m^2 at 1 AU.
    Compute radiation pressure, force on solar sail, field amplitudes.
    """
    I_sun = 1361.0  # solar irradiance (W/m^2)
    A_sail = 100.0  # sail area (m^2, 10x10 m)

    # (a) Radiation pressure
    P_absorb = I_sun / c                # absorbing surface
    P_reflect = 2 * I_sun / c           # reflecting surface

    # (b) Force on the sail
    F_absorb = P_absorb * A_sail
    F_reflect = P_reflect * A_sail

    # (c) Field amplitudes: I = (1/2)*eps_0*c*E_0^2 = c*B_0^2/(2*mu_0)
    E_0 = np.sqrt(2 * I_sun / (epsilon_0 * c))
    B_0 = E_0 / c

    print(f"  Solar irradiance at 1 AU: I = {I_sun} W/m^2")
    print(f"  Sail area: A = {A_sail} m^2")
    print()
    print("  (a) Radiation pressure:")
    print(f"    Absorbing: P = I/c = {P_absorb:.4e} Pa = {P_absorb*1e6:.4f} uPa")
    print(f"    Reflecting: P = 2I/c = {P_reflect:.4e} Pa = {P_reflect*1e6:.4f} uPa")
    print()
    print("  (b) Force on solar sail (reflecting):")
    print(f"    F = {F_reflect:.6e} N = {F_reflect*1e6:.4f} uN")
    print(f"    For 1 kg sail, a = {F_reflect/1.0*1e6:.4f} um/s^2")
    print()
    print("  (c) Field amplitudes:")
    print(f"    E_0 = {E_0:.2f} V/m")
    print(f"    B_0 = E_0/c = {B_0*1e6:.4f} uT")

    # Compare with gravity
    G = 6.674e-11
    M_sun = 1.989e30
    r_AU = 1.496e11
    m_sail = 1.0
    F_grav = G * M_sun * m_sail / r_AU**2
    print(f"\n  Gravitational force on 1 kg at 1 AU: {F_grav:.4e} N")
    print(f"  Ratio F_rad/F_grav: {F_reflect/F_grav:.4e}")


def exercise_4():
    """
    Exercise 4: Stress Tensor Force
    Force between parallel wires using Maxwell stress tensor.
    Verify F/L = mu_0*I1*I2/(2*pi*d).
    """
    I1 = 10.0   # current in wire 1 (A)
    I2 = 5.0    # current in wire 2 (A)
    d = 0.1     # separation (m)

    # Analytic: F/L = mu_0*I1*I2 / (2*pi*d)
    F_per_L = mu_0 * I1 * I2 / (2 * np.pi * d)

    # Maxwell stress tensor approach:
    # Choose a cylindrical surface around wire 2
    # T_ij = (1/mu_0)(B_i*B_j - 0.5*delta_ij*B^2) + eps_0(E_i*E_j - 0.5*delta_ij*E^2)
    # For DC currents, E = 0 between the wires (assuming neutral wires)
    # B at surface of wire 2 (distance r from wire 2, a from wire 1):
    # We integrate the stress tensor over a cylindrical surface around wire 2

    # Numerical integration: integrate T_rr * dA over cylinder around wire 2
    # Use a cylinder of radius a << d around wire 2
    a_cyl = 0.01  # cylinder radius around wire 2
    N_phi = 1000
    phi = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
    dphi = 2 * np.pi / N_phi

    # Points on the cylinder surface (centered on wire 2)
    x_surf = a_cyl * np.cos(phi)
    y_surf = a_cyl * np.sin(phi) + d  # wire 2 is at (0, d)

    # B field at surface points: superposition of both wires
    # Wire 1 at origin: B = mu_0*I1/(2*pi*r) in phi_hat direction
    # Wire 2 at (0, d): B = mu_0*I2/(2*pi*a_cyl) in phi_hat direction
    # For force on wire 2, we only need the field from wire 1 at wire 2's location
    # The self-field of wire 2 contributes no net force on itself

    # Field from wire 1 at the surface points
    r1 = np.sqrt(x_surf**2 + y_surf**2)  # distance from wire 1
    B1_x = -mu_0 * I1 / (2 * np.pi) * y_surf / r1**2
    B1_y = mu_0 * I1 / (2 * np.pi) * x_surf / r1**2

    # Field from wire 2 (self-field) at surface points
    # Wire 2 at (0, d); surface points relative to wire 2:
    dx2 = x_surf - 0
    dy2 = (y_surf - d)
    r2 = np.sqrt(dx2**2 + dy2**2)
    r2 = np.maximum(r2, 1e-10)
    B2_x = -mu_0 * I2 / (2 * np.pi) * dy2 / r2**2
    B2_y = mu_0 * I2 / (2 * np.pi) * dx2 / r2**2

    # Total B
    Bx = B1_x + B2_x
    By = B1_y + B2_y
    B_sq = Bx**2 + By**2

    # Outward normal on the cylinder: n = (cos(phi), sin(phi))
    nx = np.cos(phi)
    ny = np.sin(phi)

    # Maxwell stress tensor force per unit length:
    # f_x = integral of (T_xx*nx + T_xy*ny) * a*dphi
    # T_xx = (1/mu_0)(Bx^2 - 0.5*B^2), T_xy = (1/mu_0)(Bx*By)
    T_xx = (1 / mu_0) * (Bx**2 - 0.5 * B_sq)
    T_xy = (1 / mu_0) * Bx * By
    T_yx = T_xy
    T_yy = (1 / mu_0) * (By**2 - 0.5 * B_sq)

    fx_per_L = np.sum((T_xx * nx + T_xy * ny) * a_cyl * dphi)
    fy_per_L = np.sum((T_yx * nx + T_yy * ny) * a_cyl * dphi)

    print(f"  Wire 1: I1 = {I1} A at origin")
    print(f"  Wire 2: I2 = {I2} A at y = {d} m")
    print()
    print(f"  Analytic: F/L = mu_0*I1*I2/(2*pi*d) = {F_per_L*1e6:.4f} uN/m")
    print(f"  Stress tensor: F_x/L = {fx_per_L*1e6:.4f} uN/m")
    print(f"  Stress tensor: F_y/L = {fy_per_L*1e6:.4f} uN/m")
    print(f"  |F|/L = {np.sqrt(fx_per_L**2 + fy_per_L**2)*1e6:.4f} uN/m")
    print(f"  (Force should be along y-axis, attracting for parallel currents)")


def exercise_5():
    """
    Exercise 5: Numerical Poynting Theorem Verification
    1D FDTD: verify energy conservation using Poynting theorem.
    """
    Nz = 500
    dz = 1e-3
    dt = 0.9 * dz / c

    Ex = np.zeros(Nz)
    By = np.zeros(Nz)

    # Initial Gaussian pulse in Ex
    z = np.arange(Nz) * dz
    z_center = Nz * dz / 2
    width = 15 * dz
    Ex = np.exp(-(z - z_center)**2 / width**2)

    # Initial energy
    u_E = 0.5 * epsilon_0 * Ex**2
    u_B = 0.5 * By**2 / mu_0
    E_total_initial = np.sum((u_E + u_B) * dz)

    # Also initialize By for a right-moving wave: By = Ex/c
    By = Ex / c

    E_total_initial = np.sum((0.5 * epsilon_0 * Ex**2 + 0.5 * By**2 / mu_0) * dz)

    Nt = 200
    energy_history = [E_total_initial]
    energy_left_boundary = 0.0
    energy_right_boundary = 0.0

    for n in range(Nt):
        # Track Poynting flux at boundaries
        S_left = Ex[0] * By[0] / mu_0 if len(Ex) > 0 else 0
        S_right = Ex[-1] * By[-1] / mu_0 if len(Ex) > 0 else 0
        energy_left_boundary += S_left * dt  # energy leaving from left (actually inward)
        energy_right_boundary += S_right * dt  # energy leaving from right

        # FDTD update
        By[:-1] += dt / dz * (Ex[1:] - Ex[:-1])
        Ex[1:] += dt / (epsilon_0 * mu_0 * dz) * (By[1:] - By[:-1])

        # Current energy
        u_E = 0.5 * epsilon_0 * Ex**2
        u_B = 0.5 * By**2 / mu_0
        E_now = np.sum((u_E + u_B) * dz)
        energy_history.append(E_now)

    E_final = energy_history[-1]
    E_leaked = energy_right_boundary - energy_left_boundary

    print(f"  1D FDTD: {Nz} cells, {Nt} time steps")
    print(f"  Initial energy: {E_total_initial:.6e} J/m^2")
    print(f"  Final energy in domain: {E_final:.6e} J/m^2")
    print(f"  Energy leaked through boundaries: {E_leaked:.6e} J/m^2")
    print(f"  Total (domain + leaked): {E_final + E_leaked:.6e} J/m^2")
    print(f"  Conservation error: {abs(E_total_initial - E_final - E_leaked)/E_total_initial:.4e}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(energy_history)), energy_history, 'b-', linewidth=2)
    ax.axhline(y=E_total_initial, color='red', linestyle='--', label='Initial energy')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Energy in domain (J/m^2)')
    ax.set_title('Poynting Theorem: Energy Conservation in 1D FDTD')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex08_poynting_conservation.png', dpi=150)
    plt.close()
    print("  Plot saved: ex08_poynting_conservation.png")


if __name__ == "__main__":
    print("=== Exercise 1: Energy in a Capacitor ===")
    exercise_1()
    print("\n=== Exercise 2: Poynting Vector of Charging Capacitor ===")
    exercise_2()
    print("\n=== Exercise 3: Solar Radiation Pressure ===")
    exercise_3()
    print("\n=== Exercise 4: Stress Tensor Force ===")
    exercise_4()
    print("\n=== Exercise 5: Poynting Theorem Verification ===")
    exercise_5()
    print("\nAll exercises completed!")
