"""
Exercises for Lesson 10: EM Waves in Matter
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
    Exercise 1: Drude Model for Aluminum
    omega_p = 2.29e16 rad/s, gamma = 1.22e14 rad/s.
    Plot eps_r(omega), find transparency frequency, compute skin depth.
    """
    omega_p = 2.29e16   # plasma frequency
    gamma = 1.22e14     # damping rate

    omega = np.linspace(0.01 * omega_p, 3 * omega_p, 5000)

    # Drude dielectric function: eps_r = 1 - omega_p^2 / (omega^2 + i*gamma*omega)
    eps_r = 1 - omega_p**2 / (omega**2 + 1j * gamma * omega)

    # (a) Plot eps_r(omega)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(omega / omega_p, eps_r.real, 'b-', linewidth=2, label=r"$\epsilon'$")
    axes[0].plot(omega / omega_p, eps_r.imag, 'r-', linewidth=2, label=r"$\epsilon''$")
    axes[0].axhline(y=0, color='gray', linestyle='--')
    axes[0].axvline(x=1, color='green', linestyle=':', label=r'$\omega_p$')
    axes[0].set_xlabel(r'$\omega / \omega_p$')
    axes[0].set_ylabel(r'$\epsilon_r$')
    axes[0].set_title('Drude Model: Aluminum')
    axes[0].legend()
    axes[0].set_ylim(-5, 5)
    axes[0].grid(True, alpha=0.3)

    # (b) Transparency: where eps' crosses zero (eps' = 0 => omega ~ omega_p)
    # eps' = 1 - omega_p^2/(omega^2 + gamma^2) for real part
    # At transparency: 1 = omega_p^2/(omega_t^2 + gamma^2)
    omega_t = np.sqrt(omega_p**2 - gamma**2)
    lambda_t = 2 * np.pi * c / omega_t
    print(f"  Aluminum: omega_p = {omega_p:.2e} rad/s, gamma = {gamma:.2e} rad/s")
    print(f"  (b) Transparency frequency: omega_t = {omega_t:.4e} rad/s")
    print(f"      Transparency wavelength: lambda_t = {lambda_t*1e9:.1f} nm")
    print(f"      (In the UV range)")

    # (c) Skin depth: delta = c / (omega * kappa), where n + i*kappa = sqrt(eps_r)
    # At 1 GHz
    omega_1GHz = 2 * np.pi * 1e9
    eps_1GHz = 1 - omega_p**2 / (omega_1GHz**2 + 1j * gamma * omega_1GHz)
    n_complex_1GHz = np.sqrt(eps_1GHz)
    kappa_1GHz = n_complex_1GHz.imag
    delta_1GHz = c / (omega_1GHz * kappa_1GHz)

    # At visible (500 nm)
    omega_vis = 2 * np.pi * c / 500e-9
    eps_vis = 1 - omega_p**2 / (omega_vis**2 + 1j * gamma * omega_vis)
    n_complex_vis = np.sqrt(eps_vis)
    kappa_vis = n_complex_vis.imag
    delta_vis = c / (omega_vis * kappa_vis)

    print(f"\n  (c) Skin depths:")
    print(f"      At 1 GHz: delta = {delta_1GHz*1e6:.2f} um")
    print(f"      At 500 nm (visible): delta = {delta_vis*1e9:.2f} nm")

    # Skin depth vs frequency
    omega_range = np.logspace(8, 17, 500) * 2 * np.pi
    eps_range = 1 - omega_p**2 / (omega_range**2 + 1j * gamma * omega_range)
    n_range = np.sqrt(eps_range)
    kappa_range = n_range.imag
    delta_range = c / (omega_range * np.maximum(kappa_range, 1e-10))

    axes[1].loglog(omega_range / (2 * np.pi), delta_range * 1e6, 'b-', linewidth=2)
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel(r'Skin depth $\delta$ ($\mu$m)')
    axes[1].set_title('Aluminum Skin Depth')
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=1e9, color='red', linestyle='--', alpha=0.5, label='1 GHz')
    axes[1].axvline(x=c / 500e-9, color='green', linestyle='--', alpha=0.5, label='500 nm')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('ex10_drude_aluminum.png', dpi=150)
    plt.close()
    print("  Plot saved: ex10_drude_aluminum.png")


def exercise_2():
    """
    Exercise 2: Multi-Resonance Lorentz Model for Glass
    Two oscillators (UV and IR). Plot n and kappa in visible range.
    """
    omega_p = 2e16    # effective plasma frequency

    # Oscillator 1 (UV)
    omega_1 = 1.5e16
    gamma_1 = 1e14
    f_1 = 0.6

    # Oscillator 2 (IR)
    omega_2 = 6e13
    gamma_2 = 5e12
    f_2 = 0.4

    # Visible range: 380 nm to 780 nm
    lambda_vis = np.linspace(380e-9, 780e-9, 500)
    omega_vis = 2 * np.pi * c / lambda_vis

    # Lorentz model: eps = 1 + sum_j (f_j * omega_p^2) / (omega_j^2 - omega^2 - i*gamma_j*omega)
    eps = np.ones(len(omega_vis), dtype=complex)
    eps += f_1 * omega_p**2 / (omega_1**2 - omega_vis**2 - 1j * gamma_1 * omega_vis)
    eps += f_2 * omega_p**2 / (omega_2**2 - omega_vis**2 - 1j * gamma_2 * omega_vis)

    n_complex = np.sqrt(eps)
    n = n_complex.real
    kappa = n_complex.imag

    print("  Multi-resonance Lorentz model for glass:")
    print(f"  UV oscillator: omega_1 = {omega_1:.1e}, gamma_1 = {gamma_1:.0e}, f_1 = {f_1}")
    print(f"  IR oscillator: omega_2 = {omega_2:.1e}, gamma_2 = {gamma_2:.0e}, f_2 = {f_2}")
    print()
    print(f"  Refractive index at 500 nm: n = {n[np.argmin(np.abs(lambda_vis-500e-9))]:.4f}")
    print(f"  Extinction at 500 nm: kappa = {kappa[np.argmin(np.abs(lambda_vis-500e-9))]:.4e}")
    print(f"  n range in visible: [{np.min(n):.4f}, {np.max(n):.4f}]")
    print(f"  Glass is transparent (kappa << 1) in the visible range: "
          f"max kappa = {np.max(np.abs(kappa)):.4e}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(lambda_vis * 1e9, n, 'b-', linewidth=2)
    axes[0].set_xlabel('Wavelength (nm)')
    axes[0].set_ylabel('n (refractive index)')
    axes[0].set_title('Refractive Index of Glass')
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(lambda_vis * 1e9, np.abs(kappa), 'r-', linewidth=2)
    axes[1].set_xlabel('Wavelength (nm)')
    axes[1].set_ylabel(r'$\kappa$ (extinction coefficient)')
    axes[1].set_title('Extinction Coefficient')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Lorentz Model: Glass in Visible Range', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ex10_lorentz_glass.png', dpi=150)
    plt.close()
    print("  Plot saved: ex10_lorentz_glass.png")


def exercise_3():
    """
    Exercise 3: Kramers-Kronig Numerical Verification for Silver
    Compute eps'' analytically, reconstruct eps' via KK integral.
    """
    omega_p = 14.0e15   # silver plasma frequency
    gamma_s = 3.2e13    # silver damping

    # eps = 1 - omega_p^2/(omega^2 + i*gamma*omega)
    # eps' = 1 - omega_p^2*omega^2 / ((omega^2)^2 + (gamma*omega)^2)
    #      = 1 - omega_p^2 / (omega^2 + gamma^2)
    # eps'' = omega_p^2*gamma / (omega*(omega^2 + gamma^2))

    omega = np.linspace(1e12, 5e16, 50000)
    domega = omega[1] - omega[0]

    eps_exact_real = 1 - omega_p**2 / (omega**2 + gamma_s**2)
    eps_exact_imag = omega_p**2 * gamma_s / (omega * (omega**2 + gamma_s**2))

    # Kramers-Kronig: eps'(omega) - 1 = (2/pi) * P.V. integral of omega'*eps''(omega')/(omega'^2 - omega^2) domega'
    # Numerical KK for a few sample frequencies
    N_sample = 50
    omega_sample = np.linspace(1e13, 4e16, N_sample)
    eps_kk_real = np.zeros(N_sample)

    for i, w0 in enumerate(omega_sample):
        # Cauchy principal value integral
        integrand = omega * eps_exact_imag / (omega**2 - w0**2)
        # Exclude the singularity by masking
        mask = np.abs(omega - w0) > 3 * domega
        eps_kk_real[i] = 1 + (2 / np.pi) * np.sum(integrand[mask] * domega)

    # Compare
    eps_exact_at_samples = 1 - omega_p**2 / (omega_sample**2 + gamma_s**2)
    rel_error = np.abs(eps_kk_real - eps_exact_at_samples) / np.abs(eps_exact_at_samples + 1e-10)

    print(f"  Silver: omega_p = {omega_p:.1e} rad/s, gamma = {gamma_s:.1e} rad/s")
    print(f"  KK integral with {len(omega)} frequency points")
    print(f"  Mean relative error: {np.mean(rel_error):.4f}")
    print(f"  Max relative error: {np.max(rel_error):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(omega / omega_p, eps_exact_real, 'b-', linewidth=2, label=r"Exact $\epsilon'$")
    axes[0].plot(omega_sample / omega_p, eps_kk_real, 'ro', markersize=5, label='KK reconstructed')
    axes[0].set_xlabel(r'$\omega / \omega_p$')
    axes[0].set_ylabel(r"$\epsilon'$")
    axes[0].set_title(r"Real Part of $\epsilon$: Exact vs KK")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-5, 2)

    axes[1].plot(omega_sample / omega_p, rel_error * 100, 'r-', linewidth=2)
    axes[1].set_xlabel(r'$\omega / \omega_p$')
    axes[1].set_ylabel('Relative Error (%)')
    axes[1].set_title('KK Reconstruction Error')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Kramers-Kronig Verification: Silver', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ex10_kramers_kronig.png', dpi=150)
    plt.close()
    print("  Plot saved: ex10_kramers_kronig.png")


def exercise_4():
    """
    Exercise 4: Pulse Broadening
    10 fs Gaussian pulse at 800 nm through 1 cm BK7 glass (GVD ~ 44.7 fs^2/mm).
    """
    tau_in = 10e-15     # input pulse duration (10 fs, FWHM)
    lambda_0 = 800e-9   # center wavelength
    GVD = 44.7e-30      # 44.7 fs^2/mm = 44.7e-30 s^2/m
    L = 0.01            # propagation distance (1 cm)

    # (a) Analytical estimate:
    # tau_out = tau_in * sqrt(1 + (4*ln(2)*GVD*L / tau_in^2)^2)
    factor = 4 * np.log(2) * GVD * L / tau_in**2
    tau_out = tau_in * np.sqrt(1 + factor**2)

    print(f"  Input pulse: tau = {tau_in*1e15:.0f} fs (FWHM), lambda = {lambda_0*1e9:.0f} nm")
    print(f"  BK7 glass: GVD = {GVD*1e30:.1f} fs^2/mm, L = {L*100:.0f} cm")
    print(f"  Broadening factor: 4*ln2*GVD*L/tau^2 = {factor:.4f}")
    print(f"  Output pulse duration: tau_out = {tau_out*1e15:.2f} fs")
    print(f"  Broadening ratio: {tau_out/tau_in:.4f}")

    # (b) Numerical simulation using Fourier transform
    N = 2**14
    dt = 0.5e-15  # time step (0.5 fs)
    t = (np.arange(N) - N // 2) * dt

    # Input Gaussian pulse (field envelope)
    sigma_t = tau_in / (2 * np.sqrt(2 * np.log(2)))  # convert FWHM to sigma
    E_in = np.exp(-t**2 / (2 * sigma_t**2))

    # FFT to frequency domain
    E_freq = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(E_in)))
    freq = np.fft.fftshift(np.fft.fftfreq(N, dt))
    omega_freq = 2 * np.pi * freq

    # Apply GVD phase: phi(omega) = (1/2)*GVD*L*omega^2 (relative to center)
    phase_GVD = 0.5 * GVD * L * omega_freq**2
    E_freq_out = E_freq * np.exp(1j * phase_GVD)

    # Back to time domain
    E_out = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(E_freq_out)))
    I_out = np.abs(E_out)**2
    I_in = np.abs(E_in)**2

    # Measure output FWHM
    I_out_norm = I_out / np.max(I_out)
    half_max_indices = np.where(I_out_norm >= 0.5)[0]
    if len(half_max_indices) > 0:
        tau_out_numerical = (half_max_indices[-1] - half_max_indices[0]) * dt
    else:
        tau_out_numerical = 0

    print(f"\n  Numerical simulation:")
    print(f"    Output FWHM: {tau_out_numerical*1e15:.2f} fs")
    print(f"    Analytical: {tau_out*1e15:.2f} fs")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t * 1e15, I_in / np.max(I_in), 'b-', linewidth=2, label=f'Input ({tau_in*1e15:.0f} fs)')
    ax.plot(t * 1e15, I_out_norm, 'r-', linewidth=2,
            label=f'Output ({tau_out_numerical*1e15:.1f} fs)')
    ax.set_xlabel('Time (fs)')
    ax.set_ylabel('Normalized Intensity')
    ax.set_title('Gaussian Pulse Broadening in BK7 Glass')
    ax.set_xlim(-50, 50)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex10_pulse_broadening.png', dpi=150)
    plt.close()
    print("  Plot saved: ex10_pulse_broadening.png")


if __name__ == "__main__":
    print("=== Exercise 1: Drude Model for Aluminum ===")
    exercise_1()
    print("\n=== Exercise 2: Multi-Resonance Lorentz Model ===")
    exercise_2()
    print("\n=== Exercise 3: Kramers-Kronig Verification ===")
    exercise_3()
    print("\n=== Exercise 4: Pulse Broadening ===")
    exercise_4()
    print("\nAll exercises completed!")
