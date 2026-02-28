"""
Exercises for Lesson 16: FDTD Implementation
Topic: Numerical_Simulation

Solutions to practice problems from the lesson.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Physical constants
eps0 = 8.854187817e-12
mu0 = 4 * np.pi * 1e-7
c0 = 1 / np.sqrt(mu0 * eps0)


# === Exercise 1: Hard Source vs Soft Source Reflected Field ===
# Problem: Compare hard and soft source behavior when a reflected wave
# returns to the source position.

def exercise_1():
    """Hard vs soft source: artifact comparison after reflection from PEC."""
    Nx = 300
    dx = 1e-3
    dt = 0.99 * dx / c0
    n_steps = 500

    source_pos = 50
    reflector_pos = 200  # PEC reflector

    t0 = 30 * dt
    tau = 10 * dt

    def gaussian_pulse(t):
        return np.exp(-((t - t0) / tau)**2)

    results = {}

    for source_type in ['hard', 'soft']:
        Ey = np.zeros(Nx)
        Hz = np.zeros(Nx)
        sigma = np.zeros(Nx)
        sigma[reflector_pos:reflector_pos + 5] = 1e7  # PEC reflector

        Ca = dt / (mu0 * dx)
        eps_eff = eps0 * np.ones(Nx)
        Cb_arr = dt / (eps_eff * dx) / (1 + sigma * dt / (2 * eps_eff))
        Ca_arr = (1 - sigma * dt / (2 * eps_eff)) / (1 + sigma * dt / (2 * eps_eff))

        Ey_at_source = []

        # Mur ABC state
        Ey_left_prev = [0.0, 0.0]
        Ey_right_prev = [0.0, 0.0]

        for n in range(n_steps):
            Hz[:-1] = Hz[:-1] - Ca * (Ey[1:] - Ey[:-1])

            t = n * dt
            source = gaussian_pulse(t)
            if source_type == 'hard':
                Ey[source_pos] = source
            else:
                Ey[source_pos] += source

            Ey[1:-1] = Ca_arr[1:-1] * Ey[1:-1] - Cb_arr[1:-1] * (Hz[1:-1] - Hz[:-2])

            # Mur ABC
            coeff = (c0 * dt - dx) / (c0 * dt + dx)
            Ey_new = Ey_left_prev[1] + coeff * (Ey[1] - Ey_left_prev[0])
            Ey_left_prev = [Ey_new, Ey[1]]
            Ey[0] = Ey_new

            Ey_new_r = Ey_right_prev[1] + coeff * (Ey[-2] - Ey_right_prev[0])
            Ey_right_prev = [Ey_new_r, Ey[-2]]
            Ey[-1] = Ey_new_r

            Ey_at_source.append(Ey[source_pos])

        results[source_type] = np.array(Ey_at_source)

    fig, ax = plt.subplots(figsize=(10, 5))
    t_arr = np.arange(n_steps) * dt * 1e9
    ax.plot(t_arr, results['hard'], 'r-', linewidth=1.5, label='Hard source')
    ax.plot(t_arr, results['soft'], 'b-', linewidth=1.5, label='Soft source')
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('Ey at source position')
    ax.set_title('Hard vs Soft Source: Field at Source After Reflection')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex16_source_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Hard vs Soft source comparison:")
    print("  Hard source: Ey[source] is forced to the source value each step.")
    print("    When the reflected wave arrives back, it reflects again at the")
    print("    source position (artificial secondary reflection).")
    print("  Soft source: Ey[source] += source_value allows the reflected wave")
    print("    to pass through the source location without spurious reflection.")
    print("  Plot saved: ex16_source_comparison.png")


# === Exercise 2: Mur ABC Angle Dependence ===
# Problem: 2D FDTD with Mur ABC on 3 sides, observe angle-dependent absorption.

def exercise_2():
    """Mur ABC angle-dependent absorption in 2D FDTD TM mode."""
    Nx = Ny = 100
    dx = dy = 1e-3
    dt = 0.9 / (c0 * np.sqrt(1/dx**2 + 1/dy**2))
    n_steps = 200

    Ez = np.zeros((Ny, Nx))
    Hx = np.zeros((Ny, Nx))
    Hy = np.zeros((Ny, Nx))

    # Source at center
    sx, sy = Nx // 2, Ny // 2

    # Store previous boundary values for Mur ABC
    Ez_left_prev = np.zeros(Ny)
    Ez_left_curr = np.zeros(Ny)
    Ez_right_prev = np.zeros(Ny)
    Ez_right_curr = np.zeros(Ny)
    Ez_top_prev = np.zeros(Nx)
    Ez_top_curr = np.zeros(Nx)

    coeff_x = (c0 * dt - dx) / (c0 * dt + dx)
    coeff_y = (c0 * dt - dy) / (c0 * dt + dy)

    # Probes at 5 cells from boundaries
    probe_left = (Ny // 2, 5)      # ~normal incidence
    probe_top = (5, Nx // 2)       # ~normal incidence
    probe_corner = (5, 5)          # ~45 degree incidence

    probe_data = {'left': [], 'top': [], 'corner': []}

    for n in range(n_steps):
        # H update
        Hx[:, :-1] -= dt / (mu0 * dy) * (Ez[:, 1:] - Ez[:, :-1])
        Hy[:-1, :] += dt / (mu0 * dx) * (Ez[1:, :] - Ez[:-1, :])

        # Source
        t = n * dt
        t0 = 30 * dt
        tau = 10 * dt
        Ez[sy, sx] += np.exp(-((t - t0) / tau)**2)

        # E update
        Ez[1:-1, 1:-1] += dt / eps0 * (
            (Hy[1:-1, 1:-1] - Hy[:-2, 1:-1]) / dx -
            (Hx[1:-1, 1:-1] - Hx[1:-1, :-2]) / dy
        )

        # Mur ABC on left, right, top; PEC on bottom
        # Left boundary
        for j in range(1, Ny - 1):
            Ez[j, 0] = Ez_left_prev[j] + coeff_x * (Ez[j, 1] - Ez_left_curr[j])
        Ez_left_prev = Ez_left_curr.copy()
        Ez_left_curr = Ez[:, 0].copy()

        # Right boundary
        for j in range(1, Ny - 1):
            Ez[j, -1] = Ez_right_prev[j] + coeff_x * (Ez[j, -2] - Ez_right_curr[j])
        Ez_right_prev = Ez_right_curr.copy()
        Ez_right_curr = Ez[:, -1].copy()

        # Top boundary
        for i in range(1, Nx - 1):
            Ez[0, i] = Ez_top_prev[i] + coeff_y * (Ez[1, i] - Ez_top_curr[i])
        Ez_top_prev = Ez_top_curr.copy()
        Ez_top_curr = Ez[0, :].copy()

        # Bottom: PEC
        Ez[-1, :] = 0

        probe_data['left'].append(Ez[probe_left])
        probe_data['top'].append(Ez[probe_top])
        probe_data['corner'].append(Ez[probe_corner])

    # Measure reflection
    for name in ['left', 'top', 'corner']:
        data = np.array(probe_data[name])
        peak = np.max(np.abs(data[:100]))
        reflected = np.max(np.abs(data[120:]))
        ratio = reflected / peak if peak > 0 else 0
        angle = "~0 deg" if name != 'corner' else "~45 deg"
        print(f"  Probe at {name} ({angle}): peak={peak:.4f}, "
              f"reflected={reflected:.4f}, ratio={ratio:.4f}")

    print("\n  Mur 1st-order ABC absorbs normal incidence well but has")
    print("  higher reflection at oblique angles (45 deg).")
    print("  This is the main limitation motivating PML.")


# === Exercise 3: PML Thickness Optimization ===
# Problem: Test PML with different layer counts and measure reflection.

def exercise_3():
    """PML thickness optimization study."""
    Nx_base = 80
    pml_counts = [4, 8, 12, 16]

    print("PML thickness optimization:")

    reflection_levels = []

    for pml_layers in pml_counts:
        Nx = Ny = Nx_base + 2 * pml_layers
        dx = dy = 1e-3
        dt = 0.9 / (c0 * np.sqrt(1/dx**2 + 1/dy**2))
        n_steps = 300

        Ez = np.zeros((Ny, Nx))
        Hx = np.zeros((Ny, Nx))
        Hy = np.zeros((Ny, Nx))

        # PML conductivity profile
        eta0 = np.sqrt(mu0 / eps0)
        sigma_max = 0.8 * 4 / (eta0 * dx)
        order = 3

        sigma_x = np.zeros(Nx)
        sigma_y = np.zeros(Ny)

        for i in range(pml_layers):
            d = (pml_layers - i) / pml_layers
            s = sigma_max * d**order
            sigma_x[i] = s
            sigma_x[-(i + 1)] = s
            sigma_y[i] = s
            sigma_y[-(i + 1)] = s

        # Simplified PML: add loss to E update
        eps_arr = eps0 * np.ones((Ny, Nx))
        sigma_2d = np.zeros((Ny, Nx))
        for j in range(Ny):
            for i in range(Nx):
                sigma_2d[j, i] = sigma_x[i] + sigma_y[j]

        Ca = (1 - sigma_2d * dt / (2 * eps_arr)) / (1 + sigma_2d * dt / (2 * eps_arr))
        Cb = (dt / eps_arr) / (1 + sigma_2d * dt / (2 * eps_arr))

        sx, sy = Nx // 2, Ny // 2
        probe_pos = (sy, sx)

        Ez_probe = []

        for n in range(n_steps):
            Hx[:, :-1] -= dt / (mu0 * dy) * (Ez[:, 1:] - Ez[:, :-1])
            Hy[:-1, :] += dt / (mu0 * dx) * (Ez[1:, :] - Ez[:-1, :])

            t = n * dt
            t0 = 30 * dt
            tau = 10 * dt
            Ez[sy, sx] += np.exp(-((t - t0) / tau)**2)

            Ez[1:-1, 1:-1] = (
                Ca[1:-1, 1:-1] * Ez[1:-1, 1:-1] +
                Cb[1:-1, 1:-1] * (
                    (Hy[1:-1, 1:-1] - Hy[:-2, 1:-1]) / dx -
                    (Hx[1:-1, 1:-1] - Hx[1:-1, :-2]) / dy
                )
            )

            # PEC outer boundary
            Ez[0, :] = Ez[-1, :] = Ez[:, 0] = Ez[:, -1] = 0

            Ez_probe.append(Ez[probe_pos])

        Ez_probe = np.array(Ez_probe)
        # Main pulse passes probe by step ~50. Reflected arrives ~step 150+
        peak = np.max(np.abs(Ez_probe[:80]))
        reflected = np.max(np.abs(Ez_probe[150:]))
        ratio = reflected / peak if peak > 0 else 1e-10
        dB = 20 * np.log10(ratio) if ratio > 0 else -100

        reflection_levels.append(dB)
        print(f"  PML={pml_layers:2d} layers: peak={peak:.4f}, "
              f"reflected={reflected:.6f}, level={dB:.1f} dB")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pml_counts, reflection_levels, 'bo-', markersize=8, linewidth=2)
    ax.axhline(y=-40, color='r', linestyle='--', label='-40 dB target')
    ax.set_xlabel('PML layers')
    ax.set_ylabel('Reflection level [dB]')
    ax.set_title('PML Reflection vs Layer Thickness')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex16_pml_optimization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Plot saved: ex16_pml_optimization.png")


# === Exercise 4: Waveguide Mode Cutoff ===
# Problem: Simulate waveguide below, at, and above cutoff frequency.

def exercise_4():
    """Waveguide mode propagation below and above cutoff."""
    a = 30  # waveguide width in cells
    Nx = 150
    Ny = 40
    dx = dy = 1e-3

    dt = 0.9 / (c0 * np.sqrt(1/dx**2 + 1/dy**2))

    f_c = c0 / (2 * a * dx)  # TE10 cutoff frequency
    print(f"TE10 cutoff frequency: f_c = {f_c/1e9:.2f} GHz")
    print(f"Waveguide width: a = {a*dx*1000:.0f} mm")

    freq_factors = [0.8, 1.0, 1.2]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    wall_y1 = (Ny - a) // 2
    wall_y2 = wall_y1 + a

    for idx, factor in enumerate(freq_factors):
        f_op = factor * f_c
        print(f"\n  f = {factor}*f_c = {f_op/1e9:.2f} GHz:")

        Ez = np.zeros((Ny, Nx))
        Hx = np.zeros((Ny, Nx))
        Hy = np.zeros((Ny, Nx))

        sigma = np.zeros((Ny, Nx))
        sigma[:wall_y1, :] = 1e7
        sigma[wall_y2:, :] = 1e7

        eps = eps0 * np.ones((Ny, Nx))
        Ca = (1 - sigma * dt / (2 * eps)) / (1 + sigma * dt / (2 * eps))
        Cb = (dt / eps) / (1 + sigma * dt / (2 * eps))

        source_x = 10
        source_y = Ny // 2
        n_steps = 300

        for n in range(n_steps):
            Hx[:, :-1] -= dt / (mu0 * dy) * (Ez[:, 1:] - Ez[:, :-1])
            Hy[:-1, :] += dt / (mu0 * dx) * (Ez[1:, :] - Ez[:-1, :])

            t = n * dt
            t0 = 0.2e-9
            tau = 0.05e-9
            ramp = min(1.0, t / t0) if t < t0 else 1.0
            Ez[source_y, source_x] += ramp * np.sin(2 * np.pi * f_op * t)

            Ez[1:-1, 1:-1] = (
                Ca[1:-1, 1:-1] * Ez[1:-1, 1:-1] +
                Cb[1:-1, 1:-1] * (
                    (Hy[1:-1, 1:-1] - Hy[:-2, 1:-1]) / dx -
                    (Hx[1:-1, 1:-1] - Hx[1:-1, :-2]) / dy
                )
            )

            Ez[0, :] = Ez[-1, :] = Ez[:, 0] = Ez[:, -1] = 0

        # Check if wave propagated
        energy_near = np.sum(Ez[wall_y1:wall_y2, 20:60]**2)
        energy_far = np.sum(Ez[wall_y1:wall_y2, 100:140]**2)
        propagates = energy_far > 0.01 * energy_near

        x = np.arange(Nx) * dx * 1000
        y = np.arange(Ny) * dy * 1000
        X, Y = np.meshgrid(x, y)

        ax = axes[idx]
        vmax = np.max(np.abs(Ez)) * 0.5
        if vmax < 1e-10:
            vmax = 1
        ax.pcolormesh(X, Y, Ez, cmap='RdBu_r', shading='auto',
                      vmin=-vmax, vmax=vmax)
        ax.axhline(y=wall_y1 * dy * 1000, color='k', linewidth=2)
        ax.axhline(y=wall_y2 * dy * 1000, color='k', linewidth=2)
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        title = f'f = {factor}*fc'
        title += ' (evanescent)' if not propagates else ' (propagating)'
        ax.set_title(title)

        print(f"    Propagating: {propagates}")

        if propagates and factor > 1:
            # Measure guide wavelength
            centerline = Ez[Ny // 2, 40:130]
            crossings = []
            for i in range(len(centerline) - 1):
                if centerline[i] * centerline[i + 1] < 0:
                    crossings.append(i)
            if len(crossings) >= 3:
                half_wl = np.mean(np.diff(crossings)) * dx
                lambda_g_meas = 2 * half_wl
                lambda_0 = c0 / f_op
                lambda_g_theory = lambda_0 / np.sqrt(1 - (f_c / f_op)**2)
                print(f"    Guide wavelength: measured={lambda_g_meas*1000:.1f} mm, "
                      f"theory={lambda_g_theory*1000:.1f} mm")

    plt.suptitle('Waveguide Mode Cutoff Study', fontsize=13)
    plt.tight_layout()
    plt.savefig('ex16_waveguide_cutoff.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  Plot saved: ex16_waveguide_cutoff.png")


if __name__ == "__main__":
    print("=== Exercise 1: Hard Source vs Soft Source ===")
    exercise_1()
    print("\n=== Exercise 2: Mur ABC Angle Dependence ===")
    exercise_2()
    print("\n=== Exercise 3: PML Thickness Optimization ===")
    exercise_3()
    print("\n=== Exercise 4: Waveguide Mode Cutoff ===")
    exercise_4()
    print("\nAll exercises completed!")
