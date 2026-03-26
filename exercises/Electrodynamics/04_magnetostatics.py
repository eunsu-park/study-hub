"""
Exercises for Lesson 04: Magnetostatics
Topic: Electrodynamics
Solutions to practice problems from the lesson.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Constants
mu_0 = 4.0 * np.pi * 1e-7   # permeability of free space (T*m/A)
epsilon_0 = 8.854e-12
k_e = 1.0 / (4.0 * np.pi * epsilon_0)


def exercise_1():
    """
    Exercise 1: Cyclotron Simulation with E x B Drift
    Include uniform E = E0 x_hat with B = B z_hat.
    Observe and verify E x B drift velocity v_d = E/B.
    """
    q = 1.6e-19     # proton charge
    m = 1.67e-27    # proton mass
    B_val = 1e-3    # B field magnitude (1 mT)
    E_val = 1.0     # E field magnitude (1 V/m)

    B = np.array([0, 0, B_val])
    E = np.array([E_val, 0, 0])

    # Expected drift velocity: v_d = E x B / B^2
    v_drift_expected = np.cross(E, B) / np.dot(B, B)

    omega_c = q * B_val / m
    T_c = 2 * np.pi / omega_c
    dt = T_c / 200
    N_steps = 2000

    # Initial velocity: zero (will develop cyclotron + drift)
    v0 = np.array([0.0, 0.0, 0.0])

    pos = np.zeros((N_steps, 3))
    vel = np.zeros((N_steps, 3))
    vel[0] = v0

    for i in range(N_steps - 1):
        # Lorentz force: F = q(E + v x B)
        a = (q / m) * (E + np.cross(vel[i], B))
        vel[i + 1] = vel[i] + a * dt
        pos[i + 1] = pos[i] + vel[i + 1] * dt

    # Measure drift velocity from average y-displacement over time
    t_total = N_steps * dt
    # Average velocity in y-direction (drift direction)
    v_drift_measured = pos[-1, 1] / t_total

    print(f"  B = {B_val*1e3:.1f} mT (z-direction)")
    print(f"  E = {E_val:.1f} V/m (x-direction)")
    print(f"  Cyclotron frequency: {omega_c:.4e} rad/s")
    print(f"  Cyclotron period: {T_c:.4e} s")
    print()
    print(f"  Expected E x B drift: v_d = E/B = {E_val/B_val:.2f} m/s (y-direction)")
    print(f"  Expected: ({v_drift_expected[0]:.2f}, {v_drift_expected[1]:.2f}, "
          f"{v_drift_expected[2]:.2f}) m/s")
    print(f"  Measured drift (avg v_y): {v_drift_measured:.2f} m/s")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].plot(pos[:, 0], pos[:, 1], 'b-', linewidth=0.5)
    axes[0].set_xlabel('x (m)')
    axes[0].set_ylabel('y (m)')
    axes[0].set_title('Particle Trajectory (E x B Drift)')
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)

    t = np.arange(N_steps) * dt
    axes[1].plot(t * 1e6, pos[:, 1], 'b-', linewidth=1)
    axes[1].plot(t * 1e6, v_drift_expected[1] * t, 'r--', linewidth=1.5,
                 label=f'v_d = {v_drift_expected[1]:.0f} m/s')
    axes[1].set_xlabel('Time (us)')
    axes[1].set_ylabel('y (m)')
    axes[1].set_title('y-Displacement (Drift)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('E x B Drift', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ex04_exb_drift.png', dpi=150)
    plt.close()
    print("  Plot saved: ex04_exb_drift.png")


def exercise_2():
    """
    Exercise 2: Helmholtz Coils
    Two coaxial loops of radius R separated by R, each carrying current I.
    Compute B_z along axis and show uniformity near the midpoint.
    """
    R = 0.1   # loop radius (10 cm)
    I = 1.0   # current (A)
    d = R     # separation = R (Helmholtz condition)

    z = np.linspace(-0.3, 0.3, 1000)

    # On-axis field of a single loop at position z0:
    # B_z = (mu_0 * I * R^2) / (2 * (R^2 + (z-z0)^2)^(3/2))
    def B_loop(z_arr, z0):
        return mu_0 * I * R**2 / (2.0 * (R**2 + (z_arr - z0)**2)**1.5)

    # Two loops at z = +d/2 and z = -d/2
    B_total = B_loop(z, d / 2) + B_loop(z, -d / 2)

    # Field at center
    B_center = B_total[len(z) // 2]

    # Derivatives at center (numerical)
    dz = z[1] - z[0]
    dB_dz = np.gradient(B_total, dz)
    d2B_dz2 = np.gradient(dB_dz, dz)

    center_idx = len(z) // 2
    print(f"  Helmholtz coils: R = {R*100:.0f} cm, separation = R = {R*100:.0f} cm")
    print(f"  Current I = {I} A")
    print(f"  B at center: {B_center*1e6:.4f} uT")
    print(f"  dB/dz at center: {dB_dz[center_idx]*1e6:.4e} uT/m (should be ~0)")
    print(f"  d^2B/dz^2 at center: {d2B_dz2[center_idx]*1e6:.4e} uT/m^2 (should be ~0)")

    # Uniformity: how much does B vary in the central 20% of the gap?
    mask = np.abs(z) < d * 0.1
    B_variation = (np.max(B_total[mask]) - np.min(B_total[mask])) / B_center
    print(f"  B variation in central 10%: {B_variation*100:.4f}%")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(z * 100, B_total * 1e6, 'b-', linewidth=2, label='Helmholtz')
    axes[0].plot(z * 100, B_loop(z, 0) * 1e6, 'g--', linewidth=1, label='Single loop')
    axes[0].axhline(y=B_center * 1e6, color='red', linestyle=':', alpha=0.5)
    axes[0].axvline(x=d / 2 * 100, color='gray', linestyle='--', alpha=0.3)
    axes[0].axvline(x=-d / 2 * 100, color='gray', linestyle='--', alpha=0.3)
    axes[0].set_xlabel('z (cm)')
    axes[0].set_ylabel('B_z (uT)')
    axes[0].set_title('Helmholtz Coil Field')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(z * 100, (B_total / B_center - 1) * 1e6, 'r-', linewidth=2)
    axes[1].set_xlabel('z (cm)')
    axes[1].set_ylabel('(B - B_center) / B_center (ppm)')
    axes[1].set_title('Field Uniformity')
    axes[1].set_xlim(-d * 100, d * 100)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Helmholtz Coils', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ex04_helmholtz.png', dpi=150)
    plt.close()
    print("  Plot saved: ex04_helmholtz.png")


def exercise_3():
    """
    Exercise 3: Biot-Savart for a Finite Wire
    Straight wire of length 2L carrying current I.
    Derive and compute B at perpendicular distance s.
    Show it approaches mu_0*I/(2*pi*s) for L -> inf.
    """
    I = 1.0     # current (A)
    L = 0.5     # half-length (m)

    s_vals = np.linspace(0.01, 1.0, 200)

    # Analytic result for finite wire (Biot-Savart):
    # B = (mu_0 * I) / (4 * pi * s) * 2L / sqrt(s^2 + L^2)
    # = (mu_0 * I * L) / (2 * pi * s * sqrt(s^2 + L^2))
    B_finite = mu_0 * I * L / (2 * np.pi * s_vals * np.sqrt(s_vals**2 + L**2))

    # Infinite wire: B = mu_0 * I / (2 * pi * s)
    B_infinite = mu_0 * I / (2 * np.pi * s_vals)

    # Numerical Biot-Savart: discretize wire into segments
    N_seg = 10000
    z_seg = np.linspace(-L, L, N_seg)
    dz = z_seg[1] - z_seg[0]

    B_numerical = np.zeros_like(s_vals)
    for i, s in enumerate(s_vals):
        # Field point at (s, 0, 0), wire along z-axis
        # dl = dz z_hat, r_vec = (s, 0, -z_seg)
        # dl x r_hat has only phi component: = dz * s / r^2 (in magnitude)
        r_mag = np.sqrt(s**2 + z_seg**2)
        dB = mu_0 * I / (4 * np.pi) * dz * s / r_mag**3
        B_numerical[i] = np.sum(dB)

    print(f"  Finite wire: half-length L = {L} m, current I = {I} A")
    print(f"\n  At s = 0.1 m:")
    idx = np.argmin(np.abs(s_vals - 0.1))
    print(f"    B (finite analytic):  {B_finite[idx]*1e6:.4f} uT")
    print(f"    B (numerical):        {B_numerical[idx]*1e6:.4f} uT")
    print(f"    B (infinite wire):    {B_infinite[idx]*1e6:.4f} uT")

    # Show convergence to infinite wire as s << L
    print(f"\n  Ratio B_finite/B_infinite:")
    for s_test in [0.01, 0.05, 0.1, 0.5]:
        idx = np.argmin(np.abs(s_vals - s_test))
        ratio = B_finite[idx] / B_infinite[idx]
        print(f"    s = {s_test} m: {ratio:.6f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(s_vals * 100, B_finite * 1e6, 'b-', linewidth=2, label=f'Finite (L={L} m)')
    ax.plot(s_vals * 100, B_infinite * 1e6, 'r--', linewidth=1.5, label='Infinite wire')
    ax.plot(s_vals * 100, B_numerical * 1e6, 'g:', linewidth=1.5, label='Numerical Biot-Savart')
    ax.set_xlabel('s (cm)')
    ax.set_ylabel('B (uT)')
    ax.set_title('Magnetic Field of a Finite Wire')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex04_finite_wire.png', dpi=150)
    plt.close()
    print("  Plot saved: ex04_finite_wire.png")


def exercise_4():
    """
    Exercise 4: Force Between Current Loops
    Two coaxial circular loops of radius R with currents I1, I2.
    Compute force as a function of separation d.
    """
    R = 0.1     # loop radius
    I1 = 1.0    # current in loop 1
    I2 = 1.0    # current in loop 2

    N_seg = 500
    phi = np.linspace(0, 2 * np.pi, N_seg, endpoint=False)
    dphi = 2 * np.pi / N_seg

    def biot_savart_Bz_on_axis(R_loop, I_loop, z_point):
        """On-axis B_z from a single loop at z=0."""
        return mu_0 * I_loop * R_loop**2 / (2.0 * (R_loop**2 + z_point**2)**1.5)

    separations = np.linspace(0.02, 0.5, 50)
    forces = np.zeros_like(separations)

    for idx, d in enumerate(separations):
        # Force on loop 2 (at z=d) due to field of loop 1 (at z=0)
        # Use F = I2 * integral(dl2 x B1)
        # Loop 2 at z=d
        total_Fz = 0.0
        for p2 in phi:
            x2 = R * np.cos(p2)
            y2 = R * np.sin(p2)
            z2 = d

            # dl2 = R*dphi * (-sin(phi2), cos(phi2), 0)
            dl2x = -R * np.sin(p2) * dphi
            dl2y = R * np.cos(p2) * dphi

            # B at (x2, y2, z2) from loop 1
            Bx = 0.0
            By = 0.0
            Bz = 0.0
            for p1 in phi:
                x1 = R * np.cos(p1)
                y1 = R * np.sin(p1)
                z1 = 0.0

                dl1x = -R * np.sin(p1) * dphi
                dl1y = R * np.cos(p1) * dphi

                rx = x2 - x1
                ry = y2 - y1
                rz = z2 - z1
                r_mag = np.sqrt(rx**2 + ry**2 + rz**2)

                # dl1 x r_hat / r^2
                cx = dl1y * rz / r_mag**3
                cy = -dl1x * rz / r_mag**3
                cz = (dl1x * ry - dl1y * rx) / r_mag**3

                Bx += cx
                By += cy
                Bz += cz

            Bx *= mu_0 * I1 / (4 * np.pi)
            By *= mu_0 * I1 / (4 * np.pi)
            Bz *= mu_0 * I1 / (4 * np.pi)

            # F = I2 * dl2 x B
            Fz = I2 * (dl2x * By - dl2y * Bx)
            total_Fz += Fz

        forces[idx] = total_Fz

    # Dipole approximation for large d: F ~ 3*mu_0*m1*m2/(2*pi*d^4)
    m1 = I1 * np.pi * R**2
    m2 = I2 * np.pi * R**2
    F_dipole = 3 * mu_0 * m1 * m2 / (2 * np.pi * separations**4)

    print(f"  Two coaxial loops: R = {R*100:.0f} cm, I1 = I2 = {I1} A")
    print(f"  Magnetic moments: m = I*pi*R^2 = {m1:.4e} A*m^2")
    print(f"\n  Force at d = R = {R*100:.0f} cm:")
    idx_R = np.argmin(np.abs(separations - R))
    print(f"    Numerical: {forces[idx_R]:.4e} N")
    print(f"    Dipole:    {F_dipole[idx_R]:.4e} N")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(separations * 100, np.abs(forces) * 1e6, 'b-', linewidth=2,
                label='Full numerical')
    ax.semilogy(separations * 100, F_dipole * 1e6, 'r--', linewidth=1.5,
                label='Dipole approx')
    ax.set_xlabel('Separation d (cm)')
    ax.set_ylabel('|Force| (uN)')
    ax.set_title('Force Between Coaxial Current Loops')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex04_loop_force.png', dpi=150)
    plt.close()
    print("  Plot saved: ex04_loop_force.png")


def exercise_5():
    """
    Exercise 5: Magnetic Field of a Solenoid (Numerical)
    Model solenoid as N loops. Compute B_z along axis.
    Verify interior field approaches mu_0*n*I.
    """
    R = 0.05     # solenoid radius (5 cm)
    L = 0.4      # solenoid length (40 cm)
    N_turns = 200
    I = 1.0      # current (A)
    n = N_turns / L  # turns per unit length

    # Expected interior field
    B_ideal = mu_0 * n * I

    # Position of each loop along z-axis
    z_loops = np.linspace(-L / 2, L / 2, N_turns)

    # Compute B_z along the axis
    z_axis = np.linspace(-0.4, 0.4, 500)
    B_z = np.zeros_like(z_axis)

    for z0 in z_loops:
        # On-axis field from a loop at z0:
        # B_z = mu_0*I*R^2 / (2*(R^2 + (z-z0)^2)^(3/2))
        B_z += mu_0 * I * R**2 / (2.0 * (R**2 + (z_axis - z0)**2)**1.5)

    # Interior field (central region)
    mask_center = np.abs(z_axis) < L / 4
    B_center_avg = np.mean(B_z[mask_center])

    print(f"  Solenoid: R = {R*100:.0f} cm, L = {L*100:.0f} cm")
    print(f"  N = {N_turns} turns, I = {I} A, n = {n:.0f} turns/m")
    print(f"  Ideal B = mu_0*n*I = {B_ideal*1e3:.4f} mT")
    print(f"  Numerical B (center): {B_z[len(z_axis)//2]*1e3:.4f} mT")
    print(f"  Numerical B (avg center): {B_center_avg*1e3:.4f} mT")
    print(f"  Relative error: {abs(B_center_avg - B_ideal)/B_ideal:.4e}")

    # Fringing field at the ends
    idx_end = np.argmin(np.abs(z_axis - L / 2))
    B_end = B_z[idx_end]
    print(f"\n  B at the end (z=L/2): {B_end*1e3:.4f} mT")
    print(f"  Ratio B_end/B_center: {B_end/B_ideal:.4f} (should be ~0.5)")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(z_axis * 100, B_z * 1e3, 'b-', linewidth=2, label='Numerical')
    ax.axhline(y=B_ideal * 1e3, color='red', linestyle='--',
               label=f'mu_0*n*I = {B_ideal*1e3:.3f} mT')
    ax.axhline(y=B_ideal * 1e3 / 2, color='green', linestyle=':',
               label='B_ideal / 2')
    ax.axvline(x=-L / 2 * 100, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=L / 2 * 100, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('z (cm)')
    ax.set_ylabel('B_z (mT)')
    ax.set_title(f'Solenoid: {N_turns} turns, R={R*100:.0f} cm, L={L*100:.0f} cm')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex04_solenoid.png', dpi=150)
    plt.close()
    print("  Plot saved: ex04_solenoid.png")


if __name__ == "__main__":
    print("=== Exercise 1: Cyclotron with E x B Drift ===")
    exercise_1()
    print("\n=== Exercise 2: Helmholtz Coils ===")
    exercise_2()
    print("\n=== Exercise 3: Biot-Savart for Finite Wire ===")
    exercise_3()
    print("\n=== Exercise 4: Force Between Current Loops ===")
    exercise_4()
    print("\n=== Exercise 5: Solenoid Field (Numerical) ===")
    exercise_5()
    print("\nAll exercises completed!")
