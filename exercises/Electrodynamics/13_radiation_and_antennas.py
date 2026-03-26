"""
Exercises for Lesson 13: Radiation and Antennas
Topic: Electrodynamics
Solutions to practice problems from the lesson.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Constants
c = 2.998e8
mu_0 = 4.0 * np.pi * 1e-7
epsilon_0 = 8.854e-12
k_B = 1.38e-23   # Boltzmann constant


def exercise_1():
    """
    Exercise 1: Retarded Potential of a Switched Current
    Wire carries 0 current for t<0, I0 for t>0. Compute B(s,t).
    """
    I0 = 1.0     # current (A)
    s = 0.1      # perpendicular distance (m)

    print("  Long wire: I = 0 for t < 0, I = I0 for t > 0")
    print(f"  I0 = {I0} A, observation distance s = {s} m")
    print()

    # The field "turns on" from the nearest point:
    # At time t after the switch, only the portion of the wire within
    # a distance z_max = sqrt((ct)^2 - s^2) contributes.
    # B(s,t) = (mu_0*I0)/(2*pi*s) * (ct)/sqrt((ct)^2 - s^2) for ct > s
    # B = 0 for ct < s (information hasn't arrived yet)

    t = np.linspace(0, 2e-9, 1000)
    B = np.zeros_like(t)

    for i, ti in enumerate(t):
        ct = c * ti
        if ct > s:
            # Partial contribution from the wire
            # B = (mu_0*I0/(2*pi*s)) * ct/sqrt(ct^2 - s^2)
            # Actually for a suddenly switched-on infinite wire:
            # B(s,t) = (mu_0*I0)/(2*pi*s) for ct >> s (steady state)
            # The transition happens near ct = s
            z_max = np.sqrt(ct**2 - s**2)
            # Biot-Savart integral from -z_max to +z_max:
            # B = (mu_0*I0)/(4*pi) * 2*z_max*s / (s^2 * sqrt(s^2 + z_max^2))
            # Simplified: B = (mu_0*I0)/(2*pi*s) * z_max/ct
            B[i] = (mu_0 * I0 / (2 * np.pi * s)) * z_max / ct

    # Steady-state value
    B_steady = mu_0 * I0 / (2 * np.pi * s)
    t_arrival = s / c

    print(f"  Light travel time to s: t = s/c = {t_arrival*1e9:.4f} ns")
    print(f"  Steady-state B: {B_steady*1e6:.4f} uT")
    print(f"  B at t = 2*t_arrival: {B[np.argmin(np.abs(t - 2*t_arrival))]*1e6:.4f} uT")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t * 1e9, B * 1e6, 'b-', linewidth=2)
    ax.axhline(y=B_steady * 1e6, color='red', linestyle='--', label='Steady state')
    ax.axvline(x=t_arrival * 1e9, color='green', linestyle=':', label='t = s/c')
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('B (uT)')
    ax.set_title('Switched Current: B Field Turn-On')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex13_switched_current.png', dpi=150)
    plt.close()
    print("  Plot saved: ex13_switched_current.png")


def exercise_2():
    """
    Exercise 2: Magnetic Dipole Radiation
    Oscillating m(t) = m0*cos(wt) z_hat. Far-field E, total power.
    """
    m0 = 1e-3     # magnetic dipole moment amplitude (A*m^2)
    f = 1e9        # frequency (1 GHz)
    omega = 2 * np.pi * f
    k = omega / c

    # Total radiated power for magnetic dipole:
    # P = mu_0 * m0^2 * omega^4 / (12 * pi * c^3)
    P_mag = mu_0 * m0**2 * omega**4 / (12 * np.pi * c**3)

    # For comparison, electric dipole with p0:
    # P_elec = p0^2 * omega^4 / (12 * pi * eps_0 * c^3)
    # Ratio: P_mag/P_elec = (m0*mu_0*eps_0*c^3)^2 / (p0*c^3)^2 = (m0/(p0*c))^2
    # if m0 = p0*c: P_mag = P_elec (equivalent radiation)

    print(f"  Magnetic dipole: m0 = {m0} A*m^2, f = {f/1e9:.0f} GHz")
    print(f"  omega = {omega:.4e} rad/s, k = {k:.4f} m^-1")
    print()
    print(f"  (a) Far-field E: |E| ~ (mu_0*m0*omega^2)/(4*pi*c*r) * sin(theta)")
    E_far = mu_0 * m0 * omega**2 / (4 * np.pi * c)
    print(f"      At r = 1 m, theta = pi/2: |E| = {E_far:.4e} V/m")
    print()
    print(f"  (b) Total radiated power: P = {P_mag:.4e} W")
    print(f"      = {10*np.log10(P_mag*1e3):.2f} dBm")
    print()
    print(f"  (c) Ratio P_mag/P_elec = (m0*omega/(p0*c^2))^2")
    print(f"      For equal moments: P_mag/P_elec = (omega/(c/m0*p0))^2")
    print("      Magnetic dipole radiation is generally much weaker than")
    print("      electric dipole radiation for non-relativistic sources.")

    # Radiation pattern
    theta = np.linspace(0, np.pi, 200)
    pattern = np.sin(theta)**2  # same as electric dipole

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
    ax.plot(theta, pattern, 'b-', linewidth=2, label='Magnetic dipole')
    ax.plot(theta + np.pi, pattern, 'b-', linewidth=2)
    ax.set_title('Magnetic Dipole Radiation Pattern', pad=20)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('ex13_magnetic_dipole.png', dpi=150)
    plt.close()
    print("  Plot saved: ex13_magnetic_dipole.png")


def exercise_3():
    """
    Exercise 3: Phased Array Design
    8-element array at 10 GHz, half-wavelength spacing.
    Plot array factor for various steering angles.
    """
    N_elements = 8
    f = 10e9
    lam = c / f
    d_spacing = lam / 2   # half-wavelength spacing

    print(f"  Phased array: {N_elements} elements at {f/1e9:.0f} GHz")
    print(f"  Wavelength: {lam*1e3:.2f} mm, spacing: {d_spacing*1e3:.2f} mm")

    theta = np.linspace(-90, 90, 1000)
    theta_rad = np.radians(theta)

    steering_angles = [0, 30, 45, 60]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, steer_deg in enumerate(steering_angles):
        ax = axes[idx // 2, idx % 2]
        steer_rad = np.radians(steer_deg)

        # Phase shift per element for beam steering
        delta_phase = 2 * np.pi * d_spacing / lam * np.sin(steer_rad)

        # Array factor: AF = sum_{n=0}^{N-1} exp(j*n*(k*d*sin(theta) - delta_phase))
        psi = 2 * np.pi * d_spacing / lam * np.sin(theta_rad) - delta_phase
        AF = np.zeros_like(theta, dtype=complex)
        for n in range(N_elements):
            AF += np.exp(1j * n * psi)

        AF_mag = np.abs(AF) / N_elements  # normalize
        AF_dB = 20 * np.log10(AF_mag + 1e-10)

        # 3dB beamwidth
        half_power = np.where(AF_mag >= 1 / np.sqrt(2))[0]
        if len(half_power) > 0:
            bw_3dB = theta[half_power[-1]] - theta[half_power[0]]
        else:
            bw_3dB = 0

        ax.plot(theta, AF_dB, 'b-', linewidth=2)
        ax.axvline(x=steer_deg, color='red', linestyle='--', alpha=0.5,
                   label=f'Steer: {steer_deg} deg')
        ax.axhline(y=-3, color='green', linestyle=':', alpha=0.5, label='-3 dB')
        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('AF (dB)')
        ax.set_title(f'Steering = {steer_deg} deg, BW_3dB = {bw_3dB:.1f} deg')
        ax.set_ylim(-30, 2)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        print(f"  Steering {steer_deg} deg: 3dB beamwidth = {bw_3dB:.1f} deg")

    # Grating lobe appears when d*sin(theta) = lambda
    theta_grating = np.degrees(np.arcsin(lam / d_spacing - 1)) if d_spacing >= lam / 2 else 90
    print(f"\n  Max steering before grating lobe: {theta_grating:.1f} deg")

    plt.suptitle('8-Element Phased Array at 10 GHz', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ex13_phased_array.png', dpi=150)
    plt.close()
    print("  Plot saved: ex13_phased_array.png")


def exercise_4():
    """
    Exercise 4: Satellite Link Budget
    Geostationary satellite at 12 GHz.
    """
    f = 12e9             # frequency
    P_t = 20.0           # transmit power (W)
    G_t_dBi = 40.0       # transmit antenna gain
    G_r_dBi = 50.0       # receive antenna gain
    R = 36000e3           # distance (36000 km)
    BW = 30e6             # receiver bandwidth (30 MHz)
    T_sys = 290.0         # system temperature (K)
    lam = c / f

    # (a) Free-space path loss: FSPL = (4*pi*R/lambda)^2
    FSPL = (4 * np.pi * R / lam)**2
    FSPL_dB = 10 * np.log10(FSPL)

    # (b) Received power: Friis equation
    # P_r = P_t * G_t * G_r * (lambda/(4*pi*R))^2
    G_t = 10**(G_t_dBi / 10)
    G_r = 10**(G_r_dBi / 10)

    P_r = P_t * G_t * G_r * (lam / (4 * np.pi * R))**2
    P_r_dBm = 10 * np.log10(P_r * 1e3)

    # (c) Thermal noise floor: N = k_B * T * BW
    N_floor = k_B * T_sys * BW
    N_floor_dBm = 10 * np.log10(N_floor * 1e3)
    SNR_dB = P_r_dBm - N_floor_dBm

    print(f"  Satellite link budget:")
    print(f"  Frequency: {f/1e9:.0f} GHz, wavelength: {lam*100:.2f} cm")
    print(f"  Distance: {R/1e3:.0f} km")
    print(f"  Tx power: {P_t} W = {10*np.log10(P_t*1e3):.1f} dBm")
    print(f"  Tx gain: {G_t_dBi:.0f} dBi, Rx gain: {G_r_dBi:.0f} dBi")
    print()
    print(f"  (a) Free-space path loss: {FSPL_dB:.2f} dB")
    print(f"  (b) Received power: {P_r:.4e} W = {P_r_dBm:.2f} dBm")
    print(f"  (c) Noise floor (BW={BW/1e6:.0f} MHz, T={T_sys} K): {N_floor_dBm:.2f} dBm")
    print(f"      SNR = {SNR_dB:.2f} dB")
    print(f"      {'Signal above noise floor' if SNR_dB > 0 else 'Signal below noise floor'}")


if __name__ == "__main__":
    print("=== Exercise 1: Retarded Potential of Switched Current ===")
    exercise_1()
    print("\n=== Exercise 2: Magnetic Dipole Radiation ===")
    exercise_2()
    print("\n=== Exercise 3: Phased Array Design ===")
    exercise_3()
    print("\n=== Exercise 4: Satellite Link Budget ===")
    exercise_4()
    print("\nAll exercises completed!")
