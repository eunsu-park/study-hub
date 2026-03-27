"""
Lesson 17: Spectral and Advanced Methods
Topic: MHD
Description: Exercises on pseudo-spectral methods, Fourier derivatives,
             dealiasing rules, Chebyshev polynomials, energy spectra,
             AMR efficiency, hybrid MHD-PIC, SPH kernels, and code
             comparison (Athena++ vs PLUTO).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def exercise_1():
    """Fourier Derivative.

    For f(x) = sin(3x) on [0, 2*pi] with N = 16 points, compute f'(x)
    using FFT. Compare to exact f'(x) = 3*cos(3x).
    """
    N = 16
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    dx = x[1] - x[0]

    f = np.sin(3 * x)
    f_exact_deriv = 3 * np.cos(3 * x)

    # FFT-based derivative:
    # 1. Compute FFT: f_hat = FFT(f)
    # 2. Multiply by ik: f'_hat = i*k * f_hat
    # 3. Inverse FFT: f' = IFFT(f'_hat)

    f_hat = np.fft.fft(f)

    # Wavenumbers
    k = np.fft.fftfreq(N, d=dx / (2 * np.pi))  # wavenumbers scaled to 2*pi domain
    # More explicitly: k_n = n for n = 0, ..., N/2-1 and n-N for n = N/2, ..., N-1
    k = np.fft.fftfreq(N) * N  # integer wavenumbers

    # Spectral derivative
    f_deriv_hat = 1j * k * f_hat
    f_deriv = np.real(np.fft.ifft(f_deriv_hat))

    # Error
    error = np.max(np.abs(f_deriv - f_exact_deriv))

    print(f"  f(x) = sin(3x) on [0, 2*pi], N = {N} points")
    print(f"  FFT-based derivative:")
    print(f"    Max error |f'_FFT - f'_exact| = {error:.2e}")
    print()

    if error < 1e-10:
        print(f"  Machine-precision accuracy! ({error:.2e})")
        print(f"  This is because sin(3x) has only one Fourier mode (k=3),")
        print(f"  and N=16 > 2*3 = 6, so the mode is exactly representable.")
    else:
        print(f"  Error: {error:.6f}")

    # Verify with higher mode
    for k_test in [3, 5, 7, 8]:
        f_test = np.sin(k_test * x)
        f_hat_test = np.fft.fft(f_test)
        k_arr = np.fft.fftfreq(N) * N
        f_deriv_test = np.real(np.fft.ifft(1j * k_arr * f_hat_test))
        f_exact_test = k_test * np.cos(k_test * x)
        err = np.max(np.abs(f_deriv_test - f_exact_test))
        status = "EXACT" if err < 1e-10 else f"error={err:.2e}"
        nyquist = "at Nyquist" if k_test == N // 2 else ("above Nyquist" if k_test > N // 2 else "")
        print(f"  k={k_test}: {status} {nyquist}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x_fine = np.linspace(0, 2 * np.pi, 200)
    ax1.plot(x_fine, 3 * np.cos(3 * x_fine), 'b-', linewidth=2, label="Exact: 3cos(3x)")
    ax1.plot(x, f_deriv, 'ro', markersize=8, label="FFT derivative")
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel("f'(x)", fontsize=12)
    ax1.set_title('Fourier Derivative of sin(3x)', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, np.abs(f_deriv - f_exact_deriv), 'r-o', markersize=6)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('|Error|', fontsize=12)
    ax2.set_title('Pointwise Error', fontsize=13)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('17_fourier_derivative.png', dpi=150)
    plt.close()
    print("  Plot saved to 17_fourier_derivative.png")


def exercise_2():
    """2/3 Dealiasing.

    f(x) = sin(5x), g(x) = sin(7x). Product has max wavenumber 12.
    N = 16 (Nyquist k_N = 8). Will aliasing occur? What is the aliased mode?
    """
    N = 16
    k_N = N // 2  # Nyquist wavenumber = 8

    k_f = 5
    k_g = 7

    # Product: f*g = sin(5x)*sin(7x) = (1/2)[cos((5-7)x) - cos((5+7)x)]
    #         = (1/2)[cos(2x) - cos(12x)]
    # The product contains modes at k = 2 and k = 12.
    k_product_low = abs(k_f - k_g)  # = 2
    k_product_high = k_f + k_g       # = 12

    print(f"  f(x) = sin({k_f}x), g(x) = sin({k_g}x)")
    print(f"  Product: f*g = (1/2)[cos({k_product_low}x) - cos({k_product_high}x)]")
    print(f"  N = {N}, Nyquist wavenumber k_N = N/2 = {k_N}")
    print()

    # Aliasing: mode k > k_N is aliased to k_alias = 2*k_N - k (for k_N < k < 2*k_N)
    if k_product_high > k_N:
        # Aliased wavenumber
        k_alias = 2 * k_N - k_product_high  # wrap around: 2*8 - 12 = 4
        print(f"  k = {k_product_high} > k_N = {k_N} => ALIASING OCCURS!")
        print(f"  Aliased wavenumber: k_alias = 2*k_N - k = {k_alias}")
        print(f"  The cos(12x) mode will appear as cos({abs(k_alias)}x) in the discrete spectrum!")
        print(f"  This corrupts the k={abs(k_alias)} mode with spurious energy.")
    else:
        print(f"  k = {k_product_high} <= k_N = {k_N}: No aliasing")

    print()
    print(f"  2/3 Dealiasing Rule:")
    print(f"  - Keep only modes with |k| <= (2/3)*k_N = {2 * k_N / 3:.1f}")
    print(f"  - Zero out modes with |k| > {2 * k_N / 3:.1f}")
    print(f"  - Maximum product wavenumber of kept modes: 2*(2/3)*k_N = {4 * k_N / 3:.1f}")
    print(f"  - This is < 2*k_N = {2 * k_N}, so aliased modes don't contaminate kept modes")

    # Verify with FFT
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    f = np.sin(k_f * x)
    g = np.sin(k_g * x)
    fg = f * g

    # Exact product spectrum
    fg_hat = np.fft.fft(fg) / N  # normalized
    k_arr = np.fft.fftfreq(N) * N

    # Expected: coefficient at k=2 and k=12 (aliased to k=4)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Without dealiasing
    ax1.stem(k_arr[:k_N + 1], np.abs(fg_hat[:k_N + 1]), 'b-', markerfmt='bo',
             basefmt='gray', label='Without dealiasing')
    ax1.axvline(2 * k_N / 3, color='red', linestyle='--', label=f'2/3 cutoff (k={2 * k_N / 3:.0f})')
    ax1.set_xlabel('Wavenumber k', fontsize=12)
    ax1.set_ylabel('|Amplitude|', fontsize=12)
    ax1.set_title('Spectrum of f*g (aliased)', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # With 2/3 dealiasing
    f_hat = np.fft.fft(f)
    g_hat = np.fft.fft(g)
    k_cutoff = int(2 * N / 3)
    f_hat_dealiased = f_hat.copy()
    g_hat_dealiased = g_hat.copy()
    # Zero out modes above 2/3 * N/2
    for i in range(N):
        ki = k_arr[i]
        if abs(ki) > 2 * k_N / 3:
            f_hat_dealiased[i] = 0
            g_hat_dealiased[i] = 0

    f_dealiased = np.real(np.fft.ifft(f_hat_dealiased))
    g_dealiased = np.real(np.fft.ifft(g_hat_dealiased))
    fg_dealiased = f_dealiased * g_dealiased
    fg_hat_dealiased = np.fft.fft(fg_dealiased) / N

    ax2.stem(k_arr[:k_N + 1], np.abs(fg_hat_dealiased[:k_N + 1]), 'g-', markerfmt='go',
             basefmt='gray', label='With 2/3 dealiasing')
    ax2.axvline(2 * k_N / 3, color='red', linestyle='--', label=f'2/3 cutoff')
    ax2.set_xlabel('Wavenumber k', fontsize=12)
    ax2.set_ylabel('|Amplitude|', fontsize=12)
    ax2.set_title('Spectrum of f*g (dealiased)', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('17_dealiasing.png', dpi=150)
    plt.close()
    print("  Plot saved to 17_dealiasing.png")
    print(f"  Note: With dealiasing, modes k={k_f} and k={k_g} are above the 2/3 cutoff")
    print(f"  ({2 * k_N / 3:.0f}), so they are zeroed out. The product is trivially zero.")
    print(f"  In practice, dealiasing affects the nonlinear terms in the MHD equations,")
    print(f"  not individual fields. The spectral code computes products in physical space")
    print(f"  and then truncates the result.")


def exercise_3():
    """Chebyshev Points.

    Compute N = 8 Gauss-Lobatto Chebyshev points. Plot their distribution.
    Explain why they cluster at endpoints.
    """
    N = 8
    # Gauss-Lobatto Chebyshev points: x_j = cos(pi * j / N), j = 0, ..., N
    j = np.arange(N + 1)
    x_cheb = np.cos(np.pi * j / N)

    # For comparison: uniform points
    x_uniform = np.linspace(-1, 1, N + 1)

    print(f"  N = {N} Gauss-Lobatto Chebyshev points:")
    print(f"  x_j = cos(pi * j / N), j = 0, ..., {N}")
    print()
    print(f"  {'j':>3s}  {'x_j':>10s}  {'spacing':>10s}")
    print(f"  {'-'*3}  {'-'*10}  {'-'*10}")
    for i in range(N + 1):
        spacing = abs(x_cheb[i] - x_cheb[i - 1]) if i > 0 else "-"
        if isinstance(spacing, float):
            print(f"  {i:3d}  {x_cheb[i]:10.6f}  {spacing:10.6f}")
        else:
            print(f"  {i:3d}  {x_cheb[i]:10.6f}  {spacing:>10s}")

    print()
    print(f"  Why Chebyshev points cluster at endpoints:")
    print(f"  - They are projections of equally spaced points on a semicircle")
    print(f"    onto the x-axis. Near the edges (x = +/-1), the semicircle is")
    print(f"    nearly horizontal, so projections are closer together.")
    print(f"  - This clustering PREVENTS the Runge phenomenon (oscillations")
    print(f"    in polynomial interpolation near boundaries).")
    print(f"  - The maximum interpolation error is minimized (near-optimal")
    print(f"    polynomial approximation).")
    print(f"  - Endpoint spacing ~ pi/(2*N^2) vs interior ~ pi/N")
    print(f"    Ratio: 1/(2N) ~ {1/(2*N):.4f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    # Chebyshev points on number line
    ax1.plot(x_cheb, np.zeros(N + 1), 'bo', markersize=10, label='Chebyshev')
    ax1.plot(x_uniform, 0.3 * np.ones(N + 1), 'rs', markersize=8, label='Uniform')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_title(f'Chebyshev vs Uniform Points (N = {N})', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.set_ylim(-0.5, 0.8)
    ax1.set_yticks([])
    ax1.axhline(0, color='blue', linestyle='-', alpha=0.2)
    ax1.axhline(0.3, color='red', linestyle='-', alpha=0.2)
    ax1.grid(True, alpha=0.3, axis='x')

    # Semicircle projection
    theta = np.linspace(0, np.pi, N + 1)
    theta_fine = np.linspace(0, np.pi, 200)
    ax2.plot(np.cos(theta_fine), np.sin(theta_fine), 'k-', linewidth=1)
    ax2.plot(np.cos(theta), np.sin(theta), 'go', markersize=8, label='Equal angles on semicircle')
    for i in range(N + 1):
        ax2.plot([np.cos(theta[i]), np.cos(theta[i])], [np.sin(theta[i]), 0],
                 'b--', alpha=0.4, linewidth=1)
    ax2.plot(x_cheb, np.zeros(N + 1), 'bo', markersize=10, label='Projected Chebyshev points')
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.set_title('Chebyshev Points as Semicircle Projections', fontsize=13)
    ax2.set_aspect('equal')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('17_chebyshev_points.png', dpi=150)
    plt.close()
    print("  Plot saved to 17_chebyshev_points.png")


def exercise_4():
    """Energy Spectrum Slope.

    Generate synthetic 2D MHD turbulence spectra for different viscosities.
    Identify the inertial range and measure the spectral slope.
    """
    # Synthetic energy spectrum model:
    # E(k) = C * k^alpha * exp(-(k/k_diss)^2) * (1 + (k_L/k)^2)^(-1)
    # where alpha ~ -5/3 (Kolmogorov) or -3/2 (Iroshnikov-Kraichnan)

    k = np.logspace(-1, 3, 1000)

    # Injection scale
    k_L = 1.0

    # Different viscosities -> different dissipation scales
    nu_values = [1e-2, 5e-3, 1e-3, 5e-4]
    Re_approx = [100, 200, 1000, 2000]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

    for nu, Re, color in zip(nu_values, Re_approx, colors):
        # Dissipation wavenumber: k_diss ~ (epsilon/nu^3)^(1/4) ~ Re^(3/4) * k_L
        k_diss = Re**(3.0 / 4.0) * k_L

        # Kolmogorov spectrum
        E_k53 = k**(-5.0 / 3.0) * np.exp(-(k / k_diss)**2)
        E_k53 *= (k / k_L)**2 / (1 + (k / k_L)**2)  # low-k forcing

        ax.loglog(k, E_k53, color=color, linewidth=2,
                  label=f'nu={nu}, Re~{Re}')

    # Reference slopes
    k_ref = np.logspace(0, 1.5, 50)
    ax.loglog(k_ref, 0.5 * k_ref**(-5.0 / 3.0), 'k--', linewidth=2, alpha=0.5,
              label=r'$k^{-5/3}$ (Kolmogorov)')
    ax.loglog(k_ref, 0.5 * k_ref**(-3.0 / 2.0), 'k:', linewidth=2, alpha=0.5,
              label=r'$k^{-3/2}$ (IK)')

    ax.set_xlabel('Wavenumber k', fontsize=12)
    ax.set_ylabel('E(k)', fontsize=12)
    ax.set_title('MHD Energy Spectra for Different Viscosities', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.1, 1000)
    ax.set_ylim(1e-8, 10)

    plt.tight_layout()
    plt.savefig('17_energy_spectrum.png', dpi=150)
    plt.close()

    print("  Energy spectrum analysis:")
    print("  ========================")
    for nu, Re in zip(nu_values, Re_approx):
        k_diss = Re**(3.0 / 4.0)
        k_inertial_max = k_diss / 10  # roughly
        inertial_decades = np.log10(k_inertial_max / k_L)
        has_inertial = "YES" if inertial_decades > 0.5 else "NO"
        print(f"  nu = {nu:.0e}, Re ~ {Re}: k_diss ~ {k_diss:.0f}, "
              f"inertial range: {inertial_decades:.1f} decades, clear k^-5/3: {has_inertial}")

    print()
    print("  Key observations:")
    print("  - Need Re > ~1000 for a clear k^{-5/3} inertial range")
    print("  - MHD turbulence may show k^{-3/2} (Iroshnikov-Kraichnan) at moderate Re")
    print("  - Spectral methods can achieve higher effective Re for same grid size")
    print("  - Dealiasing is essential for correct cascade behavior")
    print("  Plot saved to 17_energy_spectrum.png")


def exercise_5():
    """AMR Efficiency.

    Estimate grid points needed for uniform vs 2-level AMR to resolve
    a 1D shock to 1% accuracy.
    """
    print("  AMR Efficiency for 1D Shock Resolution:")
    print("  =========================================")
    print()

    # Assumptions
    L = 1.0              # domain length
    shock_width = 0.001  # physical shock width (controlled by dissipation)
    target_accuracy = 0.01  # 1% accuracy

    # For a well-resolved shock, need ~10 cells across the shock width
    cells_per_shock = 10

    # Uniform grid
    dx_uniform = shock_width / cells_per_shock
    N_uniform = int(L / dx_uniform)

    print(f"  Domain length: L = {L}")
    print(f"  Shock width: {shock_width}")
    print(f"  Cells needed across shock: {cells_per_shock}")
    print()

    # Uniform grid
    print(f"  Uniform grid:")
    print(f"    dx = shock_width / {cells_per_shock} = {dx_uniform:.1e}")
    print(f"    N = L / dx = {N_uniform}")
    print(f"    Total cells: {N_uniform}")
    print()

    # AMR: 2 levels, refinement factor 4
    ref_factor = 4
    # Coarse grid: resolve features at ~10x shock width
    dx_coarse = 10 * shock_width  # 0.01
    N_coarse = int(L / dx_coarse)

    # Fine grid (level 2): covers only the shock region
    # Shock region: ~20 * shock_width (buffer around shock)
    shock_region = 20 * shock_width
    dx_fine = dx_coarse / ref_factor
    N_fine = int(shock_region / dx_fine)

    # May need further refinement
    if dx_fine > dx_uniform:
        # Need level 3
        dx_fine2 = dx_fine / ref_factor
        N_fine2 = int(shock_region / dx_fine2)
        N_total_amr = N_coarse + N_fine + N_fine2
        print(f"  2-level AMR (refinement factor {ref_factor}):")
        print(f"    Level 0 (coarse): dx = {dx_coarse:.1e}, N = {N_coarse}")
        print(f"    Level 1: dx = {dx_fine:.1e}, N = {N_fine} (covers shock region)")
        print(f"    Level 2: dx = {dx_fine2:.1e}, N = {N_fine2} (covers shock region)")
        print(f"    Total cells: {N_total_amr}")
    else:
        N_total_amr = N_coarse + N_fine
        print(f"  2-level AMR (refinement factor {ref_factor}):")
        print(f"    Level 0 (coarse): dx = {dx_coarse:.1e}, N = {N_coarse}")
        print(f"    Level 1 (fine):   dx = {dx_fine:.1e}, N = {N_fine} (covers shock region)")
        print(f"    Total cells: {N_total_amr}")

    # Efficiency ratio
    efficiency = N_uniform / N_total_amr
    print()
    print(f"  Efficiency: Uniform / AMR = {N_uniform} / {N_total_amr} = {efficiency:.1f}x savings")
    print(f"  AMR uses {N_total_amr / N_uniform * 100:.1f}% of the uniform grid cells")
    print()
    print(f"  In 2D: savings scale as {efficiency:.0f}^2 ~ {efficiency**2:.0f}x (approximate)")
    print(f"  In 3D: savings scale as {efficiency:.0f}^3 ~ {efficiency**3:.0f}x (approximate)")
    print(f"  AMR is essential for problems with localized features (shocks, current sheets)")


def exercise_6():
    """Hybrid MHD-PIC.

    Cosmic ray diffusion with D = 10^28 cm^2/s, MHD grid dx = 10^10 cm.
    Estimate diffusion CFL timestep.
    """
    D = 1e28             # cm^2/s (cosmic ray diffusion coefficient)
    dx = 1e10            # cm (MHD grid spacing)

    # Diffusion CFL: dt < dx^2 / (2*D) (1D)
    # or dt < dx^2 / (2*N_dim*D) (N_dim dimensions)
    dt_1D = dx**2 / (2 * D)
    dt_2D = dx**2 / (4 * D)
    dt_3D = dx**2 / (6 * D)

    print(f"  Hybrid MHD-PIC: Cosmic Ray Diffusion")
    print(f"  ======================================")
    print(f"  Diffusion coefficient D = {D:.1e} cm^2/s")
    print(f"  MHD grid spacing dx = {dx:.1e} cm")
    print()
    print(f"  Diffusion CFL condition: dt < dx^2 / (2*N_dim*D)")
    print(f"    1D: dt < {dt_1D:.3e} s = {dt_1D:.1f} s")
    print(f"    2D: dt < {dt_2D:.3e} s")
    print(f"    3D: dt < {dt_3D:.3e} s")
    print()

    # Compare to MHD advection CFL
    # Typical Alfven speed in ISM: v_A ~ 10 km/s = 10^6 cm/s
    v_A = 1e6  # cm/s
    dt_MHD = dx / v_A
    print(f"  Compare to MHD advection CFL:")
    print(f"    v_A ~ {v_A:.0e} cm/s")
    print(f"    dt_MHD = dx / v_A = {dt_MHD:.1e} s")
    print()

    ratio = dt_MHD / dt_1D
    print(f"  Ratio dt_MHD / dt_diff = {ratio:.1f}")
    if ratio > 1:
        print(f"  Diffusion CFL is MORE restrictive by factor {ratio:.0f}")
        print(f"  => Need subcycling: take {int(ratio)+1} diffusion steps per MHD step")
    else:
        print(f"  MHD CFL is more restrictive.")

    print()
    print(f"  In hybrid MHD-PIC codes:")
    print(f"  - MHD fluid evolves on the advection CFL timestep")
    print(f"  - Cosmic ray particles are advanced with the diffusion CFL")
    print(f"  - Subcycling bridges the two timescales")
    print(f"  - Particle push: use Boris integrator for magnetic field")


def exercise_7():
    """SPH Kernel.

    For the cubic spline kernel W(r, h) ~ (1 - r/h)^3 for r < h,
    verify normalization and compute the constant in 2D.
    """
    print("  SPH Cubic Spline Kernel Normalization (2D):")
    print("  =============================================")
    print()

    # Standard cubic spline (Monaghan 1992):
    # W(q) = sigma_d * { (2-q)^3 - 4*(1-q)^3,  0 <= q <= 1
    #                     (2-q)^3,                1 < q <= 2
    #                     0,                       q > 2       }
    # where q = r/h, sigma_d = normalization constant

    # Simplified kernel from the problem: W ~ (1-r/h)^3 for r < h, 0 otherwise
    # Normalization: integral over 2D = 1
    # integral_0^h W(r) * 2*pi*r dr = 1
    # integral_0^h (1-r/h)^3 * 2*pi*r dr = 1

    # Let u = r/h, dr = h*du
    # integral_0^1 (1-u)^3 * 2*pi*(u*h) * h*du = 2*pi*h^2 * integral_0^1 u*(1-u)^3 du

    # Compute integral_0^1 u*(1-u)^3 du
    # Expand: u*(1-u)^3 = u - 3u^2 + 3u^3 - u^4
    # Integral: [u^2/2 - u^3 + 3u^4/4 - u^5/5]_0^1
    #         = 1/2 - 1 + 3/4 - 1/5
    #         = 10/20 - 20/20 + 15/20 - 4/20
    #         = 1/20

    I = 1.0 / 20.0
    print(f"  Kernel: W(r,h) = C * (1 - r/h)^3 for r < h, 0 otherwise")
    print(f"  Normalization in 2D: int W(r,h) d^2r = 1")
    print(f"  = int_0^h W(r) * 2*pi*r * dr = 1")
    print(f"  = C * 2*pi*h^2 * int_0^1 u*(1-u)^3 du = 1  (u = r/h)")
    print()
    print(f"  Computing integral:")
    print(f"  int_0^1 u*(1-u)^3 du = int_0^1 (u - 3u^2 + 3u^3 - u^4) du")
    print(f"  = [u^2/2 - u^3 + 3u^4/4 - u^5/5]_0^1")
    print(f"  = 1/2 - 1 + 3/4 - 1/5 = 1/20")
    print()

    C_2D = 1.0 / (2 * np.pi * I)  # in units of h^(-2)
    print(f"  C * 2*pi*h^2 * (1/20) = 1")
    print(f"  C = 10 / (pi * h^2)")
    print(f"  C = {C_2D / 1:.4f} / h^2  =  {10.0 / np.pi:.6f} / h^2")
    print()

    # Verify numerically
    h = 1.0
    N_pts = 10000
    r_sample = np.linspace(0, h, N_pts)
    dr = r_sample[1] - r_sample[0]
    W_sample = C_2D / h**2 * (1 - r_sample / h)**3
    integral_check = np.sum(W_sample * 2 * np.pi * r_sample * dr)
    print(f"  Numerical verification: int W * 2*pi*r*dr = {integral_check:.6f} (should be 1.0)")

    # Also compute in 3D for comparison
    # 3D: int_0^h W * 4*pi*r^2 dr = 1
    # int_0^1 u^2*(1-u)^3 du = 1/2 - 3/4 + 3/5 - 1/6 + ... = 1/60
    I_3D = 1.0 / 60.0
    C_3D = 1.0 / (4 * np.pi * I_3D)  # in units of h^(-3)
    print(f"  For comparison, 3D normalization: C = 15 / (pi * h^3) = {C_3D:.4f} / h^3")

    # Plot kernel
    fig, ax = plt.subplots(figsize=(8, 5))
    r = np.linspace(0, 1.5, 200)
    W = np.where(r < 1, C_2D * (1 - r)**3, 0)
    dW_dr = np.where(r < 1, -3 * C_2D * (1 - r)**2, 0)
    ax.plot(r, W, 'b-', linewidth=2, label='W(r/h)')
    ax.plot(r, -dW_dr / 10, 'r--', linewidth=1.5, label='-dW/dr / 10')
    ax.set_xlabel('r / h', fontsize=12)
    ax.set_ylabel('W', fontsize=12)
    ax.set_title('Cubic Spline Kernel (2D)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('17_sph_kernel.png', dpi=150)
    plt.close()
    print("  Plot saved to 17_sph_kernel.png")


def exercise_8():
    """Athena++ vs PLUTO.

    Compare default Riemann solvers and diffusivity.
    """
    print("  Athena++ vs PLUTO: Riemann Solver Comparison")
    print("  ==============================================")
    print()
    print("  Athena++ (Stone et al. 2020, ApJS):")
    print("    Default MHD Riemann solver: HLLD (Miyoshi & Kusano 2005)")
    print("    - Resolves all 7 MHD waves (including Alfven)")
    print("    - Contacts and rotational discontinuities are resolved")
    print("    - Less diffusive than HLL")
    print("    - Default reconstruction: PLM (piecewise linear) with slope limiters")
    print("    - Higher-order option: PPM (piecewise parabolic)")
    print("    - CT (Constrained Transport) for div(B) = 0")
    print()
    print("  PLUTO (Mignone et al. 2007, ApJS):")
    print("    Default MHD Riemann solver: HLLD")
    print("    - Same HLLD as Athena++ (same algorithm)")
    print("    - Also supports: Roe, HLL, HLLC")
    print("    - Default reconstruction: Linear (PLM) with van Leer or MC limiter")
    print("    - Higher-order options: PPM, WENO3, LIMO3")
    print("    - Both CT and GLM (divergence cleaning) for div(B)")
    print()
    print("  Diffusivity comparison (same Riemann solver = HLLD):")
    print("    - Both HLLD: similar numerical diffusion")
    print("    - Diffusivity depends more on reconstruction + limiter choice")
    print("    - MC limiter (PLUTO default) is less diffusive than minmod")
    print("    - PPM/WENO reconstruction reduces diffusion further")
    print()
    print("  Key differences:")
    print("  | Feature          | Athena++              | PLUTO                |")
    print("  |------------------|-----------------------|----------------------|")
    print("  | Default solver   | HLLD                  | HLLD                 |")
    print("  | Language         | C++                   | C                    |")
    print("  | AMR              | Built-in (block-based)| via CHOMBO interface |")
    print("  | Parallelism      | MPI + OpenMP          | MPI + OpenMP         |")
    print("  | Special physics  | Cosmic rays, GR       | MHD, RMHD, radiation |")
    print("  | Reconstruction   | PLM, PPM, WENOZ       | PLM, PPM, WENO3      |")
    print("  | div(B) control   | CT only               | CT or GLM            |")
    print()
    print("  Which is more diffusive?")
    print("  - With identical settings (HLLD + PLM + same limiter): similar")
    print("  - PLUTO's GLM option can be more diffusive than CT for div(B)")
    print("  - In practice, differences are small; both are well-tested production codes")
    print("  - Choice depends on problem-specific features (AMR, GRMHD, etc.)")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Fourier Derivative", exercise_1),
        ("Exercise 2: 2/3 Dealiasing", exercise_2),
        ("Exercise 3: Chebyshev Points", exercise_3),
        ("Exercise 4: Energy Spectrum Slope", exercise_4),
        ("Exercise 5: AMR Efficiency", exercise_5),
        ("Exercise 6: Hybrid MHD-PIC", exercise_6),
        ("Exercise 7: SPH Kernel", exercise_7),
        ("Exercise 8: Athena++ vs PLUTO", exercise_8),
    ]
    for title, func in exercises:
        print(f"\n{'=' * 60}")
        print(f" {title}")
        print(f"{'=' * 60}")
        func()
