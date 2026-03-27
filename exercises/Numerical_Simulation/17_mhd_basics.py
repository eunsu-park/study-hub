"""
Exercises for Lesson 17: MHD Basics
Topic: Numerical_Simulation

Solutions to practice problems from the lesson.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Physical constants
mu0 = 4 * np.pi * 1e-7    # Vacuum permeability [H/m]
kB = 1.381e-23             # Boltzmann constant [J/K]
mp = 1.673e-27             # Proton mass [kg]


# === Exercise 1: Alfven Velocity and Plasma Beta ===
# Problem: For solar corona B=10 G, n=1e14 m^-3, T=1e6 K, compute
# v_A, c_s, and plasma beta.

def exercise_1():
    """Alfven velocity, sound speed, and plasma beta for solar corona."""
    B = 10e-4          # 10 G = 1e-3 T
    n = 1e14           # m^-3
    T = 1e6            # K
    gamma = 5 / 3      # Adiabatic index

    rho = n * mp       # Mass density (proton plasma)
    p = n * kB * T     # Thermal pressure

    # (a) Alfven velocity
    v_A = B / np.sqrt(mu0 * rho)

    # (b) Sound speed
    c_s = np.sqrt(gamma * p / rho)

    # (c) Plasma beta
    p_mag = B**2 / (2 * mu0)
    beta = 2 * mu0 * p / B**2  # = p_thermal / p_magnetic

    print("Solar Corona Parameters:")
    print(f"  B = {B*1e4:.0f} G = {B:.3e} T")
    print(f"  n = {n:.1e} m^-3")
    print(f"  T = {T:.1e} K")
    print(f"  rho = n*mp = {rho:.3e} kg/m^3")
    print(f"  p_thermal = {p:.3e} Pa")
    print(f"  p_magnetic = {p_mag:.3e} Pa")
    print()
    print(f"  (a) Alfven velocity: v_A = {v_A/1e3:.1f} km/s")
    print(f"  (b) Sound speed:     c_s = {c_s/1e3:.1f} km/s")
    print(f"  (c) Plasma beta:     beta = {beta:.4f}")
    print()

    if beta < 1:
        print("  Classification: Magnetically dominated (beta < 1)")
        print("  The magnetic field controls the dynamics; magnetic pressure")
        print("  exceeds thermal pressure. MHD waves are dominated by")
        print("  the Alfven mode (v_A > c_s).")
    else:
        print("  Classification: Thermally dominated (beta > 1)")

    print(f"\n  v_A/c_s = {v_A/c_s:.2f}")
    print(f"  Fast magnetosonic speed (perp): v_f = sqrt(v_A^2 + c_s^2) = "
          f"{np.sqrt(v_A**2 + c_s**2)/1e3:.1f} km/s")


# === Exercise 2: MHD Wave Phase Velocities ===
# Problem: For v_A = 2*c_s, compute fast/Alfven/slow wave speeds at various angles.
# Plot Friedrichs diagram.

def exercise_2():
    """MHD wave phase velocities and Friedrichs diagram."""
    c_s = 1.0       # Normalized sound speed
    v_A = 2.0 * c_s  # Alfven speed = 2 * sound speed

    theta = np.linspace(0, 2 * np.pi, 360)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Dispersion relations (phase velocities):
    # v_fast^2 = 0.5*(v_A^2 + c_s^2 + sqrt((v_A^2 + c_s^2)^2 - 4*v_A^2*c_s^2*cos^2(theta)))
    # v_slow^2 = 0.5*(v_A^2 + c_s^2 - sqrt((v_A^2 + c_s^2)^2 - 4*v_A^2*c_s^2*cos^2(theta)))
    # v_Alfven = v_A * |cos(theta)|

    discriminant = np.sqrt((v_A**2 + c_s**2)**2 - 4 * v_A**2 * c_s**2 * cos_t**2)

    v_fast = np.sqrt(0.5 * (v_A**2 + c_s**2 + discriminant))
    v_slow = np.sqrt(0.5 * (v_A**2 + c_s**2 - discriminant))
    v_alfven = v_A * np.abs(cos_t)

    # Print values at specific angles
    angles_deg = [0, 30, 60, 90]
    print(f"MHD wave speeds for v_A = {v_A}, c_s = {c_s}:")
    print(f"{'Angle':>8s}  {'v_fast':>8s}  {'v_Alfven':>10s}  {'v_slow':>8s}")
    print("-" * 42)
    for deg in angles_deg:
        rad = np.radians(deg)
        disc = np.sqrt((v_A**2 + c_s**2)**2 - 4 * v_A**2 * c_s**2 * np.cos(rad)**2)
        vf = np.sqrt(0.5 * (v_A**2 + c_s**2 + disc))
        vs = np.sqrt(0.5 * (v_A**2 + c_s**2 - disc))
        va = v_A * abs(np.cos(rad))
        print(f"  {deg:3d} deg  {vf:8.4f}  {va:10.4f}  {vs:8.4f}")

    # Maximum fast wave speed
    max_fast_idx = np.argmax(v_fast[:90])
    max_angle = np.degrees(theta[max_fast_idx])
    print(f"\n  Maximum fast wave speed at theta = {max_angle:.0f} deg: "
          f"v_f = {v_fast[max_fast_idx]:.4f}")
    print(f"  (At 90 deg: v_f = sqrt(v_A^2 + c_s^2) = {np.sqrt(v_A**2+c_s**2):.4f})")

    # Friedrichs diagram (polar plot of phase velocity surfaces)
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.plot(theta, v_fast, 'b-', linewidth=2, label='Fast magnetosonic')
    ax.plot(theta, v_alfven, 'g--', linewidth=2, label='Alfven')
    ax.plot(theta, v_slow, 'r-', linewidth=2, label='Slow magnetosonic')

    ax.set_title(f'Friedrichs Diagram (v_A/c_s = {v_A/c_s:.1f})', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    plt.savefig('ex17_friedrichs.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Plot saved: ex17_friedrichs.png")


# === Exercise 3: Magnetic Pressure Equilibrium ===
# Problem: Calculate pressure balance across a current sheet.

def exercise_3():
    """Magnetic pressure equilibrium at a current sheet."""
    B1 = 10e-4       # 10 G in T
    p1 = 0.5e-3      # Pa
    B2 = 0.0         # Field-free region

    # Total pressure: p + B^2/(2*mu0) = const
    p_mag_1 = B1**2 / (2 * mu0)
    total_1 = p1 + p_mag_1

    # In field-free region: p2 + B2^2/(2*mu0) = total_1
    p2 = total_1 - B2**2 / (2 * mu0)
    p_mag_2 = B2**2 / (2 * mu0)
    total_2 = p2 + p_mag_2

    print("Magnetic pressure equilibrium across current sheet:")
    print(f"  Region 1: B = {B1*1e4:.0f} G, p = {p1:.3e} Pa")
    print(f"    p_mag = B^2/(2 mu0) = {p_mag_1:.3e} Pa")
    print(f"    p_total = {total_1:.3e} Pa")
    print()
    print(f"  Region 2: B = {B2:.0f} G")
    print(f"    p2 = p_total - B2^2/(2 mu0) = {p2:.3e} Pa")
    print(f"    p_total = {total_2:.3e} Pa")
    print()
    print(f"  Pressure jump: |total_1 - total_2| = {abs(total_1-total_2):.2e} Pa")
    print(f"  Equilibrium verified: jump = 0")
    print()
    print(f"  The thermal pressure in the field-free region must increase by")
    print(f"  {p_mag_1:.3e} Pa to compensate for the loss of magnetic pressure.")


# === Exercise 4: Frozen-in Theorem and Magnetic Diffusion ===
# Problem: Compute magnetic diffusivity, diffusion time, and Rm.

def exercise_4():
    """Frozen-in theorem analysis: diffusion time and magnetic Reynolds number."""
    sigma = 1e6      # Conductivity [S/m]
    L = 1e6          # Length scale [m] = 1 Mm
    v = 1e5          # Flow velocity [m/s] = 100 km/s
    B = 10e-4        # 10 G in T
    n = 1e14         # m^-3

    # (a) Magnetic diffusivity
    eta = 1 / (mu0 * sigma)

    # (b) Magnetic diffusion timescale
    tau_diff = mu0 * sigma * L**2

    # (c) Magnetic Reynolds number
    Rm = mu0 * sigma * v * L

    # Alfven crossing time
    rho = n * mp
    v_A = B / np.sqrt(mu0 * rho)
    tau_A = L / v_A

    print("Frozen-in Theorem and Magnetic Diffusion:")
    print(f"  sigma = {sigma:.1e} S/m")
    print(f"  L = {L/1e6:.0f} Mm")
    print(f"  v = {v/1e3:.0f} km/s")
    print()
    print(f"  (a) Magnetic diffusivity: eta = 1/(mu0 sigma) = {eta:.4f} m^2/s")
    print(f"  (b) Diffusion time: tau_diff = mu0 sigma L^2 = {tau_diff:.3e} s "
          f"= {tau_diff/(3600*24*365):.1f} years")
    print(f"  (c) Magnetic Reynolds number: Rm = mu0 sigma v L = {Rm:.3e}")
    print()
    print(f"  Rm >> 1: Frozen-in condition is {'VALID' if Rm > 100 else 'NOT valid'}")
    print(f"  (Rm >> 1 means advection dominates over diffusion)")
    print()
    print(f"  Alfven speed: v_A = {v_A/1e3:.1f} km/s")
    print(f"  Alfven crossing time: tau_A = L/v_A = {tau_A:.1f} s")
    print(f"  tau_diff / tau_A = {tau_diff/tau_A:.1e}")
    print(f"  (Diffusion operates on much longer timescales than wave crossing)")


# === Exercise 5: Ideal MHD Conservative Form ===
# Problem: Write out conservative variable and flux vectors, verify dimensions.

def exercise_5():
    """Ideal MHD conservative form: variables and fluxes."""
    print("1D Ideal MHD Conservative Form: dU/dt + dF/dx = 0")
    print()
    print("Conservative variable vector U (7 components):")
    print("  U = [rho, rho*vx, rho*vy, rho*vz, By, Bz, E]^T")
    print()
    print("  where E = p/(gamma-1) + 0.5*rho*v^2 + B^2/(2*mu0)")
    print("  (total energy density)")
    print()
    print("Flux vector F(U):")
    print("  F[0] = rho*vx                          [mass flux]")
    print("  F[1] = rho*vx^2 + p_total - Bx^2/mu0   [x-momentum flux]")
    print("  F[2] = rho*vx*vy - Bx*By/mu0           [y-momentum flux]")
    print("  F[3] = rho*vx*vz - Bx*Bz/mu0           [z-momentum flux]")
    print("  F[4] = By*vx - Bx*vy                   [By flux = induction]")
    print("  F[5] = Bz*vx - Bx*vz                   [Bz flux = induction]")
    print("  F[6] = (E + p_total)*vx - Bx*(v.B)/mu0 [energy flux]")
    print()
    print("  where p_total = p + B^2/(2*mu0)  [total pressure]")
    print("  Note: Bx = const in 1D (from div B = 0)")
    print()
    print("Dimensional verification (SI):")
    print("  [rho] = kg/m^3")
    print("  [rho*vx] = kg/(m^2 s)")
    print("  [E] = J/m^3 = Pa")
    print("  [F_mass] = [rho*vx] = kg/(m^2 s)")
    print("  [F_momentum] = [rho*vx^2 + p] = Pa = kg/(m s^2)")
    print("  [F_induction] = [B*v] = T m/s = V/m")
    print("  [F_energy] = [(E+p)*v] = W/m^2")
    print()

    # Verify induction equation equivalence
    print("Induction equation equivalence:")
    print("  dB/dt = curl(v x B)")
    print("  In 1D with Bx = const:")
    print("    dBy/dt = d/dx(By*vx - Bx*vy)")
    print("    dBz/dt = d/dx(Bz*vx - Bx*vz)")
    print("  This is exactly the flux form F[4] and F[5] above.")
    print("  The equivalence holds because v x B has components:")
    print("    (v x B)_y = vz*Bx - vx*Bz")
    print("    (v x B)_z = vx*By - vy*Bx")
    print("  and curl(v x B) in 1D gives d/dx of the z-component for By, etc.")


if __name__ == "__main__":
    print("=== Exercise 1: Alfven Velocity and Plasma Beta ===")
    exercise_1()
    print("\n=== Exercise 2: MHD Wave Phase Velocities ===")
    exercise_2()
    print("\n=== Exercise 3: Magnetic Pressure Equilibrium ===")
    exercise_3()
    print("\n=== Exercise 4: Frozen-in Theorem and Magnetic Diffusion ===")
    exercise_4()
    print("\n=== Exercise 5: Ideal MHD Conservative Form ===")
    exercise_5()
    print("\nAll exercises completed!")
