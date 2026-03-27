"""
Exercises for Lesson 08: MHD Turbulence
Topic: MHD
Solutions to practice problems from the lesson.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def exercise_1():
    """
    Problem 1: Kolmogorov Scaling

    epsilon = 1e-3 m^2/s^3, L = 1 m, ell = 0.01 m, nu = 1e-5 m^2/s
    """
    epsilon = 1e-3  # m^2/s^3
    L = 1.0         # m (largest eddy)
    ell = 0.01      # m (scale of interest)
    nu = 1e-5       # m^2/s (kinematic viscosity)

    # (a) Velocity at scale ell
    # K41: v(ell) ~ (epsilon * ell)^(1/3)
    v_ell = (epsilon * ell)**(1.0 / 3)
    print(f"(a) Velocity at scale ell = {ell} m:")
    print(f"    v(ell) ~ (epsilon * ell)^(1/3)")
    print(f"    = ({epsilon} * {ell})^(1/3)")
    print(f"    = {v_ell:.4f} m/s")

    # (b) Eddy turnover time
    tau_ell = ell / v_ell
    print(f"\n(b) Eddy turnover time:")
    print(f"    tau(ell) = ell / v(ell) = {ell} / {v_ell:.4f}")
    print(f"    = {tau_ell:.4f} s")

    # (c) Kolmogorov scale
    eta_K = (nu**3 / epsilon)**0.25
    print(f"\n(c) Kolmogorov dissipation scale:")
    print(f"    eta_K = (nu^3/epsilon)^(1/4)")
    print(f"    = ({nu}^3 / {epsilon})^(1/4)")
    print(f"    = {eta_K:.6f} m = {eta_K*1e3:.4f} mm")

    # Verify inertial range
    print(f"\n    Inertial range: eta_K << ell << L")
    print(f"    {eta_K:.2e} << {ell} << {L}")
    Re_L = (L / eta_K)**(4.0/3)  # Reynolds number
    print(f"    Reynolds number: Re ~ (L/eta_K)^(4/3) = {Re_L:.2e}")


def exercise_2():
    """
    Problem 2: Reynolds Number

    Earth's atmosphere: L=1000 km, v=10 m/s, nu=1.5e-5 m^2/s
    """
    L = 1000e3     # m
    v = 10.0       # m/s
    nu = 1.5e-5    # m^2/s
    epsilon_est = v**3 / L  # rough estimate

    # (a) Reynolds number
    Re = v * L / nu
    print(f"(a) Reynolds number:")
    print(f"    Re = v*L/nu = {v}*{L:.1e}/{nu:.1e}")
    print(f"    Re = {Re:.4e}")

    # (b) L/eta ratio
    # eta_K ~ L * Re^(-3/4)
    eta_K = L * Re**(-3.0/4)
    ratio = L / eta_K
    print(f"\n(b) Kolmogorov scale: eta_K = L * Re^(-3/4)")
    print(f"    = {L:.1e} * ({Re:.2e})^(-3/4)")
    print(f"    = {eta_K:.4e} m = {eta_K*1e3:.4f} mm")
    print(f"    L/eta_K = {ratio:.4e}")

    # (c) Decades of inertial range
    decades = np.log10(ratio)
    print(f"\n(c) Decades of inertial range:")
    print(f"    log10(L/eta_K) = {decades:.1f}")
    print(f"    The inertial range spans ~{decades:.0f} decades in scale")
    print(f"    This is why atmospheric turbulence is so complex --")
    print(f"    structures range from ~{eta_K*1e3:.1f} mm to ~{L/1e3:.0f} km")


def exercise_3():
    """
    Problem 3: IK vs K41 Spectra

    Plot E(k) for k^(-3/2) (IK) and k^(-5/3) (K41) on log-log
    """
    k = np.logspace(-1, 2, 500)

    # Normalize both to equal at k=1
    E_K41 = k**(-5.0 / 3)
    E_IK = k**(-3.0 / 2)

    # (a) Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.loglog(k, E_K41, 'b-', linewidth=2, label='K41: $k^{-5/3}$')
    ax.loglog(k, E_IK, 'r-', linewidth=2, label='IK: $k^{-3/2}$')
    ax.set_xlabel('Wavenumber k', fontsize=12)
    ax.set_ylabel('Energy spectrum E(k)', fontsize=12)
    ax.set_title('IK vs K41 Energy Spectra', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig('/tmp/ex08_ik_vs_k41.png', dpi=100)
    plt.close()
    print("(a) Plot saved to /tmp/ex08_ik_vs_k41.png")

    # (b) Where they differ by factor of 2
    # E_K41 / E_IK = k^(-5/3 + 3/2) = k^(-1/6)
    # k^(-1/6) = 2 => k = 2^(-6) = 1/64
    # k^(-1/6) = 0.5 => k = 0.5^(-6) = 64
    ratio = E_K41 / E_IK
    # Find where ratio = 2 (K41 > IK, happens for k < 1)
    k_ratio2_below = 2**(-6)
    # Find where ratio = 0.5 (K41 < IK, happens for k > 1)
    k_ratio2_above = 2**6
    print(f"\n(b) Ratio E_K41/E_IK = k^(-1/6)")
    print(f"    Ratio = 2 (K41 2x larger) at k = 2^(-6) = {k_ratio2_below:.6f}")
    print(f"    Ratio = 0.5 (IK 2x larger) at k = 2^6 = {k_ratio2_above}")
    print(f"    The spectra are very similar (differ by factor 2 only over ~4 decades)")

    # (c) Total energy over two decades
    # integral of k^(-5/3) from 1 to 100 = [k^(-2/3) / (-2/3)]_1^100
    E_total_K41 = (100**(-2.0/3) - 1) / (-2.0/3)
    E_total_IK = (100**(-1.0/2) - 1) / (-1.0/2)
    print(f"\n(c) Total energy from k=1 to k=100:")
    print(f"    K41: integral of k^(-5/3) dk = {E_total_K41:.4f}")
    print(f"    IK:  integral of k^(-3/2) dk = {E_total_IK:.4f}")
    if E_total_IK > E_total_K41:
        print(f"    IK has more energy ({E_total_IK/E_total_K41:.2f}x)")
        print(f"    IK's shallower slope retains more energy at small scales")
    else:
        print(f"    K41 has more energy ({E_total_K41/E_total_IK:.2f}x)")


def exercise_4():
    """
    Problem 4: Goldreich-Sridhar Anisotropy

    k_perp = 100 m^-1, GS95: k_par ~ k_perp^(2/3)
    """
    k_perp = 100.0  # m^-1

    # (a) k_parallel
    # GS95: k_par ~ k_perp^(2/3) with normalization: k_par = k_perp at outer scale
    # At outer scale k_perp = k_par = 1 (e.g.)
    # k_par = k_perp^(2/3) when outer scale normalization is k0 = 1
    k_par = k_perp**(2.0 / 3)
    print(f"(a) GS95 anisotropy: k_par ~ k_perp^(2/3)")
    print(f"    k_perp = {k_perp} m^-1")
    print(f"    k_par = ({k_perp})^(2/3) = {k_par:.2f} m^-1")

    # (b) Aspect ratio
    ell_perp = 2 * np.pi / k_perp
    ell_par = 2 * np.pi / k_par
    ratio = ell_par / ell_perp
    print(f"\n(b) Corresponding length scales:")
    print(f"    ell_perp = 2*pi/k_perp = {ell_perp:.4f} m")
    print(f"    ell_par = 2*pi/k_par = {ell_par:.4f} m")
    print(f"    Aspect ratio: ell_par/ell_perp = {ratio:.2f}")
    print(f"    Eddies are {ratio:.0f}x longer parallel to B than perpendicular")

    # (c) Sketch
    print(f"\n(c) Eddy shape at k_perp = {k_perp} m^-1:")
    print(f"    The eddy is highly elongated along the magnetic field:")
    print(f"    Perpendicular size: {ell_perp*100:.1f} cm")
    print(f"    Parallel size: {ell_par*100:.1f} cm")
    print(f"    Like a thin cigar aligned with B0")
    print(f"    At smaller scales, anisotropy increases further")

    # Plot anisotropy vs scale
    k_perp_vals = np.logspace(0, 4, 100)
    k_par_vals = k_perp_vals**(2.0 / 3)
    aspect = k_perp_vals / k_par_vals

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.loglog(k_perp_vals, k_par_vals, 'b-', linewidth=2, label='$k_\\parallel \\propto k_\\perp^{2/3}$')
    ax1.loglog(k_perp_vals, k_perp_vals, 'k--', linewidth=1, label='Isotropic: $k_\\parallel = k_\\perp$')
    ax1.set_xlabel('$k_\\perp$', fontsize=12)
    ax1.set_ylabel('$k_\\parallel$', fontsize=12)
    ax1.set_title('GS95 Anisotropy', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3, which='both')

    ax2.semilogx(k_perp_vals, aspect, 'r-', linewidth=2)
    ax2.set_xlabel('$k_\\perp$', fontsize=12)
    ax2.set_ylabel('$k_\\perp / k_\\parallel$ (aspect ratio)', fontsize=12)
    ax2.set_title('Eddy Aspect Ratio vs Scale', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/ex08_gs_anisotropy.png', dpi=100)
    plt.close()
    print("    Plot saved to /tmp/ex08_gs_anisotropy.png")


def exercise_5():
    """
    Problem 5: Elsasser Variables

    v = (1, 0, 0) m/s, B = (0, 0.01, 0) T, rho = 1e-12 kg/m^3
    """
    v = np.array([1.0, 0.0, 0.0])
    B = np.array([0.0, 0.01, 0.0])
    rho = 1e-12
    mu0 = 4 * np.pi * 1e-7

    # (a) Elsasser variables: z+ = v + B/sqrt(mu0*rho), z- = v - B/sqrt(mu0*rho)
    b = B / np.sqrt(mu0 * rho)  # Alfven velocity units
    z_plus = v + b
    z_minus = v - b

    print(f"(a) Elsasser variables:")
    print(f"    v = {v}")
    print(f"    B = {B} T")
    print(f"    b = B/sqrt(mu0*rho) = {b}")
    print(f"    |b| = {np.linalg.norm(b):.4f} m/s")
    print(f"    z+ = v + b = {z_plus}")
    print(f"    z- = v - b = {z_minus}")

    # (b) Energies
    E_kin = 0.5 * rho * np.dot(v, v)
    E_mag = np.dot(B, B) / (2 * mu0)
    print(f"\n(b) Kinetic energy density: E_kin = 0.5*rho*v^2 = {E_kin:.4e} J/m^3")
    print(f"    Magnetic energy density: E_mag = B^2/(2*mu0) = {E_mag:.4e} J/m^3")
    print(f"    Total: E_kin + E_mag = {E_kin + E_mag:.4e} J/m^3")

    # (c) Verify Elsasser energy identity
    E_elsasser = rho / 4 * (np.dot(z_plus, z_plus) + np.dot(z_minus, z_minus))
    print(f"\n(c) Elsasser energy: (rho/4)*(|z+|^2 + |z-|^2)")
    print(f"    |z+|^2 = {np.dot(z_plus, z_plus):.4e}")
    print(f"    |z-|^2 = {np.dot(z_minus, z_minus):.4e}")
    print(f"    (rho/4)*(|z+|^2 + |z-|^2) = {E_elsasser:.4e} J/m^3")
    print(f"    E_kin + E_mag = {E_kin + E_mag:.4e} J/m^3")
    if abs(E_elsasser - (E_kin + E_mag)) / (E_kin + E_mag) < 1e-10:
        print(f"    VERIFIED: Identity holds (difference < 1e-10)")
    else:
        print(f"    Difference: {abs(E_elsasser - (E_kin + E_mag)):.4e}")


def exercise_6():
    """
    Problem 6: Critical Balance

    Solar wind: v_A = 50 km/s, L = 1e6 km
    """
    v_A = 50e3       # m/s
    L = 1e6 * 1e3    # m (10^9 m)

    # (a) Velocity fluctuation at injection scale
    # Critical balance: v_L ~ v_A at outer scale (nonlinear and wave timescales equal)
    v_L = v_A
    print(f"(a) At injection scale L = {L:.1e} m:")
    print(f"    Critical balance implies: v_L ~ v_A = {v_A/1e3:.0f} km/s")
    print(f"    (Nonlinear timescale L/v_L equals Alfven crossing time L/v_A)")

    # (b) At ell_perp = 100 km, Kolmogorov scaling
    ell_perp = 100e3  # m
    # v(ell) ~ v_L * (ell_perp/L)^(1/3) (Kolmogorov)
    v_ell = v_L * (ell_perp / L)**(1.0 / 3)
    print(f"\n(b) At ell_perp = {ell_perp/1e3:.0f} km:")
    print(f"    v(ell) ~ v_L * (ell_perp/L)^(1/3)")
    print(f"    = {v_L/1e3:.0f} * ({ell_perp/L:.1e})^(1/3)")
    print(f"    = {v_ell:.2f} m/s = {v_ell/1e3:.4f} km/s")

    # (c) Parallel scale from critical balance
    # Critical balance: v(ell) / ell_perp ~ v_A / ell_par
    # ell_par = v_A * ell_perp / v(ell)
    ell_par = v_A * ell_perp / v_ell
    print(f"\n(c) Parallel scale from critical balance:")
    print(f"    v(ell)/ell_perp = v_A/ell_par")
    print(f"    ell_par = v_A * ell_perp / v(ell)")
    print(f"    = {v_A:.0f} * {ell_perp:.0f} / {v_ell:.2f}")
    print(f"    = {ell_par:.4e} m = {ell_par/1e3:.0f} km")
    print(f"    Anisotropy ratio: ell_par/ell_perp = {ell_par/ell_perp:.0f}")
    print(f"    At this scale, eddies are {ell_par/ell_perp:.0f}x longer than wide")


def exercise_7():
    """
    Problem 7: Structure Function

    Generate synthetic velocity field with k^(-5/3) spectrum
    Compute S2(ell) and fit power law
    """
    np.random.seed(42)
    N = 4096
    dx = 1.0 / N

    # Generate synthetic field with k^(-5/3) spectrum
    k = np.fft.rfftfreq(N, d=dx)
    k[0] = 1e-10  # avoid division by zero

    # Power spectrum P(k) ~ k^(-5/3)
    # Energy spectrum E(k) ~ k^(-5/3), so amplitude ~ k^(-5/6)
    amplitude = k**(-5.0/6)
    amplitude[0] = 0  # zero mean
    # Apply high-k cutoff
    amplitude[k > N/4] = 0

    phase = 2 * np.pi * np.random.random(len(k))
    v_hat = amplitude * np.exp(1j * phase)
    v = np.fft.irfft(v_hat, n=N)
    v = v / np.std(v)  # normalize

    # Compute structure function S2(ell)
    lags = np.unique(np.logspace(0, np.log10(N/4), 50).astype(int))
    lags = lags[lags > 0]
    S2 = np.zeros(len(lags))

    for i, lag in enumerate(lags):
        dv = v[lag:] - v[:-lag]
        S2[i] = np.mean(dv**2)

    ell = lags * dx

    # Fit power law in inertial range
    fit_mask = (ell > 5 * dx) & (ell < 0.1)
    if np.sum(fit_mask) > 2:
        log_ell = np.log10(ell[fit_mask])
        log_S2 = np.log10(S2[fit_mask])
        coeffs = np.polyfit(log_ell, log_S2, 1)
        zeta_2 = coeffs[0]
    else:
        zeta_2 = 0.0

    print(f"(a) Generated synthetic velocity field with E(k) ~ k^(-5/3)")
    print(f"    N = {N} points")

    print(f"\n(b) Structure function S2(ell) = <|delta v(ell)|^2>")
    print(f"    Computed for {len(lags)} lag values")

    print(f"\n(c) Power law fit: S2 ~ ell^zeta_2")
    print(f"    Measured: zeta_2 = {zeta_2:.4f}")
    print(f"    K41 prediction: zeta_2 = 2/3 = {2.0/3:.4f}")
    print(f"    Difference: {abs(zeta_2 - 2.0/3):.4f}")
    if abs(zeta_2 - 2.0/3) < 0.1:
        print(f"    Good agreement with K41!")
    else:
        print(f"    Deviation from K41 (finite size effects or intermittency)")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Power spectrum
    k_plot = np.fft.rfftfreq(N, d=dx)[1:N//4]
    v_hat_plot = np.abs(np.fft.rfft(v))**2
    ax1.loglog(k_plot, v_hat_plot[1:N//4], 'b-', alpha=0.5, linewidth=0.5)
    # Smooth spectrum
    from numpy import convolve
    window = 5
    smoothed = np.convolve(np.log10(v_hat_plot[1:N//4]), np.ones(window)/window, mode='valid')
    k_smooth = k_plot[window//2:window//2+len(smoothed)]
    ax1.loglog(k_smooth, 10**smoothed, 'r-', linewidth=2, label='Smoothed')
    ax1.loglog(k_plot, 0.1 * k_plot**(-5.0/3), 'k--', linewidth=2, label='$k^{-5/3}$')
    ax1.set_xlabel('k', fontsize=12)
    ax1.set_ylabel('Power', fontsize=12)
    ax1.set_title('Power Spectrum', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3, which='both')

    # Structure function
    ax2.loglog(ell, S2, 'bo-', markersize=3, linewidth=1)
    ell_fit = np.logspace(np.log10(ell[fit_mask][0]), np.log10(ell[fit_mask][-1]), 50)
    ax2.loglog(ell_fit, 10**np.polyval(coeffs, np.log10(ell_fit)), 'r-', linewidth=2,
               label=f'Fit: $\\zeta_2$ = {zeta_2:.3f}')
    ax2.loglog(ell_fit, ell_fit**(2.0/3) * S2[fit_mask][0] / ell[fit_mask][0]**(2.0/3),
               'k--', linewidth=2, label='K41: $\\ell^{2/3}$')
    ax2.set_xlabel('Lag $\\ell$', fontsize=12)
    ax2.set_ylabel('$S_2(\\ell)$', fontsize=12)
    ax2.set_title('Second-Order Structure Function', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('/tmp/ex08_structure_function.png', dpi=100)
    plt.close()
    print("    Plot saved to /tmp/ex08_structure_function.png")


def exercise_8():
    """
    Problem 8: Solar Wind Spectral Break

    n = 1e7 m^-3, B = 5 nT, v_sw = 400 km/s
    """
    n = 1e7          # m^-3
    B = 5e-9         # T
    v_sw = 400e3     # m/s
    c = 3e8          # m/s
    e = 1.602e-19
    mp = 1.67e-27
    eps0 = 8.854e-12
    mu0 = 4 * np.pi * 1e-7

    # (a) Ion inertial length
    omega_pi = np.sqrt(n * e**2 / (eps0 * mp))
    d_i = c / omega_pi
    print(f"(a) Ion inertial length:")
    print(f"    omega_pi = sqrt(n*e^2/(eps0*mp)) = {omega_pi:.4e} rad/s")
    print(f"    d_i = c/omega_pi = {d_i:.4e} m = {d_i/1e3:.2f} km")

    # (b) Break frequency using Taylor's hypothesis
    # f_break = v_sw / (2*pi*d_i)
    f_break = v_sw / (2 * np.pi * d_i)
    print(f"\n(b) Break frequency (Taylor's hypothesis):")
    print(f"    f_break = v_sw / (2*pi*d_i)")
    print(f"    = {v_sw:.0f} / (2*pi*{d_i:.0f})")
    print(f"    = {f_break:.4f} Hz")

    # (c) Compare to observed
    f_obs = 0.5  # Hz (typical observed break)
    print(f"\n(c) Observed break frequency: ~{f_obs} Hz")
    print(f"    Calculated break frequency: {f_break:.4f} Hz")
    ratio = f_break / f_obs
    print(f"    Ratio: f_calc/f_obs = {ratio:.2f}")
    if 0.1 < ratio < 10:
        print(f"    Order-of-magnitude agreement!")
        print(f"    The ion inertial length scale is consistent with the")
        print(f"    observed spectral break in the solar wind")
    else:
        print(f"    Significant discrepancy")
    print(f"    Note: The exact break scale depends on local conditions,")
    print(f"    and may involve ion gyroradius rho_i as well as d_i")


def exercise_9():
    """
    Problem 9: Turbulent Heating Rate

    epsilon = 1e-16 erg/g/s in solar wind
    """
    epsilon_cgs = 1e-16   # erg/g/s
    epsilon_SI = epsilon_cgs * 1e-4  # J/kg/s = m^2/s^3
    mp = 1.67e-27          # kg
    kB = 1.38e-23          # J/K
    day = 86400            # s

    # (a) Energy per proton per second
    E_per_proton = epsilon_SI * mp  # J/s per proton
    print(f"(a) Turbulent cascade rate: epsilon = {epsilon_cgs} erg/g/s")
    print(f"    = {epsilon_SI:.2e} J/kg/s")
    print(f"    Energy per proton per second:")
    print(f"    = epsilon * mp = {E_per_proton:.4e} J/s")
    print(f"    = {E_per_proton/1.602e-19:.4e} eV/s")

    # (b) Temperature increase over 1 day
    # dT/dt = (2/3) * epsilon * mp / kB (for 3/2 kB T per proton)
    dTdt = (2.0 / 3) * epsilon_SI * mp / kB
    delta_T = dTdt * day
    print(f"\n(b) Heating rate: dT/dt = (2/3)*epsilon*mp/kB")
    print(f"    = {dTdt:.4e} K/s")
    print(f"    Temperature increase over 1 day:")
    print(f"    Delta T = {delta_T:.4f} K")
    print(f"    = {delta_T:.2e} K")

    # (c) Sufficiency for explaining solar wind temperature
    print(f"\n(c) Solar wind temperature at 1 AU: T ~ 10^5 K")
    print(f"    Adiabatic cooling (no heating): T ~ T_0 * (r/r_0)^(-4/3)")
    print(f"    Without heating, T would drop to ~10^3-10^4 K at 1 AU")
    print(f"    Daily heating: {delta_T:.2e} K")
    print(f"    Over the ~4 day transit time (Sun to 1 AU):")
    print(f"    Total heating: ~{4*delta_T:.2e} K")
    print(f"    This is very small compared to the observed temperature,")
    print(f"    but the relevant question is whether it compensates for")
    print(f"    adiabatic cooling. The total dissipated energy over the")
    print(f"    transit is epsilon * M_p * t ~ {epsilon_SI * mp * 4*day:.2e} J per proton")
    print(f"    = {epsilon_SI * mp * 4*day / 1.602e-19:.4f} eV per proton")
    print(f"    Compared to thermal energy 3/2*kB*T ~ {1.5*kB*1e5/1.602e-19:.1f} eV at 10^5 K")
    print(f"    Turbulent heating provides a fraction of the needed energy,")
    print(f"    suggesting other mechanisms (e.g., wave damping) also contribute.")


def exercise_10():
    """
    Problem 10: Anisotropic Energy Spectrum

    Sketch GS95 energy contours in (k_perp, k_par) space
    """
    # (a) Energy density contours
    print("(a) GS95 anisotropic energy spectrum:")
    print("    E(k_perp, k_par) is concentrated near the critical balance surface")
    print("    In k-space, energy contours are NOT circular but elongated")
    print("    along k_perp (perpendicular to B0)")
    print()
    print("    The energy is distributed along the curve:")
    print("    k_par ~ k_perp^(2/3)")
    print("    Most energy is at perpendicular scales (cascade is perpendicular)")

    # (b) Critical balance surface
    print("\n(b) Critical balance surface in k-space:")
    print("    k_par * v_A = k_perp * v(k_perp)")
    print("    With v(k_perp) ~ k_perp^(-1/3):")
    print("    k_par = k_perp^(2/3) * const")
    print("    This is the locus where wave and nonlinear timescales are equal")

    # (c) Comparison to isotropic turbulence
    print("\n(c) Comparison to isotropic turbulence:")
    print("    Isotropic: E contours are circles (k_perp^2 + k_par^2 = const)")
    print("    GS95: E contours are elongated ellipses in k_perp direction")
    print("    At small scales, the anisotropy becomes extreme:")
    print("    k_perp >> k_par, meaning perpendicular cascade dominates")
    print("    This is observed in solar wind turbulence measurements")

    # Plot
    k_perp = np.logspace(-1, 3, 200)
    k_par = np.logspace(-1, 3, 200)
    KP, KA = np.meshgrid(k_perp, k_par)

    # GS95 energy model: E ~ k_perp^(-10/3) * f(k_par / k_perp^(2/3))
    # where f is peaked near k_par / k_perp^(2/3) ~ 1
    sigma_par = 0.3  # width of critical balance region in log space
    E = KP**(-10.0/3) * np.exp(-0.5 * (np.log10(KA) - (2.0/3)*np.log10(KP))**2 / sigma_par**2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # GS95 anisotropic spectrum
    im1 = ax1.pcolormesh(np.log10(KP), np.log10(KA), np.log10(E + 1e-30),
                          cmap='hot', shading='auto')
    # Critical balance line
    ax1.plot(np.log10(k_perp), (2.0/3) * np.log10(k_perp), 'c--', linewidth=2,
             label='Critical balance: $k_\\parallel \\propto k_\\perp^{2/3}$')
    ax1.set_xlabel('$\\log_{10} k_\\perp$', fontsize=12)
    ax1.set_ylabel('$\\log_{10} k_\\parallel$', fontsize=12)
    ax1.set_title('GS95 Anisotropic Spectrum', fontsize=14)
    ax1.legend(loc='upper left')
    plt.colorbar(im1, ax=ax1, label='$\\log_{10} E(k_\\perp, k_\\parallel)$')

    # Isotropic for comparison
    E_iso = (KP**2 + KA**2)**(-5.0/6)  # isotropic k^(-5/3) spectrum
    im2 = ax2.pcolormesh(np.log10(KP), np.log10(KA), np.log10(E_iso),
                          cmap='hot', shading='auto')
    # Isotropic contour (circle in log space would be straight line at 45 deg)
    ax2.plot(np.log10(k_perp), np.log10(k_perp), 'c--', linewidth=2,
             label='$k_\\parallel = k_\\perp$')
    ax2.set_xlabel('$\\log_{10} k_\\perp$', fontsize=12)
    ax2.set_ylabel('$\\log_{10} k_\\parallel$', fontsize=12)
    ax2.set_title('Isotropic Spectrum (comparison)', fontsize=14)
    ax2.legend(loc='upper left')
    plt.colorbar(im2, ax=ax2, label='$\\log_{10} E(k_\\perp, k_\\parallel)$')

    plt.tight_layout()
    plt.savefig('/tmp/ex08_anisotropic_spectrum.png', dpi=100)
    plt.close()
    print("    Plot saved to /tmp/ex08_anisotropic_spectrum.png")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Kolmogorov Scaling ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Reynolds Number ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: IK vs K41 Spectra ===")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("=== Exercise 4: GS Anisotropy ===")
    print("=" * 60)
    exercise_4()

    print("\n" + "=" * 60)
    print("=== Exercise 5: Elsasser Variables ===")
    print("=" * 60)
    exercise_5()

    print("\n" + "=" * 60)
    print("=== Exercise 6: Critical Balance ===")
    print("=" * 60)
    exercise_6()

    print("\n" + "=" * 60)
    print("=== Exercise 7: Structure Function ===")
    print("=" * 60)
    exercise_7()

    print("\n" + "=" * 60)
    print("=== Exercise 8: Solar Wind Spectral Break ===")
    print("=" * 60)
    exercise_8()

    print("\n" + "=" * 60)
    print("=== Exercise 9: Turbulent Heating Rate ===")
    print("=" * 60)
    exercise_9()

    print("\n" + "=" * 60)
    print("=== Exercise 10: Anisotropic Energy Spectrum ===")
    print("=" * 60)
    exercise_10()

    print("\nAll exercises completed!")
