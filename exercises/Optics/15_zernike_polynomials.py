"""
Exercises for Lesson 15: Zernike Polynomials
Topic: Optics
Solutions to practice problems from the lesson.
"""

import numpy as np
from scipy.special import comb


def zernike_radial(n, m, rho):
    """Compute the radial polynomial R_n^m(rho) using the explicit formula."""
    m_abs = abs(m)
    R = np.zeros_like(rho, dtype=float)
    for s in range(int((n - m_abs) / 2) + 1):
        coeff = ((-1)**s * comb(n - s, s, exact=True) *
                 comb(n - 2*s, (n - m_abs)/2 - s, exact=True))
        R += coeff * rho**(n - 2*s)
    return R


def zernike_noll(j, rho, theta):
    """Compute Zernike polynomial for Noll index j."""
    # Convert Noll index to (n, m)
    n = 0
    while (n + 1) * (n + 2) / 2 < j:
        n += 1
    m_options = list(range(-n, n + 1, 2))
    # Noll ordering
    idx = j - int(n * (n + 1) / 2) - 1
    if n % 2 == 0:
        m_sorted = sorted(m_options, key=lambda x: (abs(x), -x))
    else:
        m_sorted = sorted(m_options, key=lambda x: (abs(x), x))

    if idx < len(m_sorted):
        m = m_sorted[idx]
    else:
        m = m_options[0]

    R = zernike_radial(n, m, rho)

    if m > 0:
        return R * np.cos(m * theta) * np.sqrt(2 * (n + 1))
    elif m < 0:
        return R * np.sin(-m * theta) * np.sqrt(2 * (n + 1))
    else:
        return R * np.sqrt(n + 1)


def exercise_1():
    """
    Exercise 1: Radial Polynomial Verification
    Verify the Zernike radial polynomials by direct computation and
    check orthogonality numerically.
    """
    print("Zernike Radial Polynomial Verification:")

    # (a) R_4^0(rho) = 6*rho^4 - 6*rho^2 + 1
    print("\n(a) Verify R_4^0(rho) = 6*rho^4 - 6*rho^2 + 1:")
    rho_test = np.array([0, 0.25, 0.5, 0.75, 1.0])

    R40_formula = 6*rho_test**4 - 6*rho_test**2 + 1
    R40_computed = zernike_radial(4, 0, rho_test)

    print(f"{'rho':>6} {'Formula':>10} {'Computed':>10} {'Match':>8}")
    print("-" * 36)
    for i, r in enumerate(rho_test):
        match = np.isclose(R40_formula[i], R40_computed[i])
        print(f"{r:>6.2f} {R40_formula[i]:>10.4f} {R40_computed[i]:>10.4f} {str(match):>8}")

    # (b) Orthogonality: integral of R_2^0 * R_4^0 * rho drho = 0
    print(f"\n(b) Orthogonality check:")
    N_quad = 10000
    rho = np.linspace(0, 1, N_quad)
    dr = rho[1] - rho[0]

    R20 = zernike_radial(2, 0, rho)
    R40 = zernike_radial(4, 0, rho)

    # Integral of R_2^0 * R_4^0 * rho drho from 0 to 1
    integral_20_40 = np.trapezoid(R20 * R40 * rho, rho)
    print(f"  <R_2^0, R_4^0> = integral(R_2^0 * R_4^0 * rho * drho) = {integral_20_40:.6e}")
    print(f"  Orthogonal: {abs(integral_20_40) < 1e-6}")

    # Self-overlap (normalization)
    integral_20_20 = np.trapezoid(R20 * R20 * rho, rho)
    integral_40_40 = np.trapezoid(R40 * R40 * rho, rho)
    print(f"  <R_2^0, R_2^0> = {integral_20_20:.6f}")
    print(f"  <R_4^0, R_4^0> = {integral_40_40:.6f}")

    # (c) R_3^1 and R_5^1
    print(f"\n(c) Orthogonality of R_3^1 and R_5^1:")
    R31 = zernike_radial(3, 1, rho)
    R51 = zernike_radial(5, 1, rho)

    integral_31_51 = np.trapezoid(R31 * R51 * rho, rho)
    print(f"  <R_3^1, R_5^1> = {integral_31_51:.6e}")
    print(f"  Orthogonal: {abs(integral_31_51) < 1e-6}")

    # (d) R_6^0 computation and plot
    print(f"\n(d) R_6^0(rho):")
    R60 = zernike_radial(6, 0, rho)

    print(f"  R_6^0(rho) = 20*rho^6 - 30*rho^4 + 12*rho^2 - 1")
    R60_explicit = 20*rho**6 - 30*rho**4 + 12*rho**2 - 1

    rho_check = np.array([0, 0.25, 0.5, 0.75, 1.0])
    R60_check = zernike_radial(6, 0, rho_check)
    R60_exp_check = 20*rho_check**6 - 30*rho_check**4 + 12*rho_check**2 - 1

    print(f"{'rho':>6} {'R_6^0':>10}")
    for r, val in zip(rho_check, R60_check):
        print(f"{r:>6.2f} {val:>10.4f}")

    print(f"  Physical interpretation: Secondary spherical aberration")
    print(f"  Has 3 zero crossings within [0,1]")


def exercise_2():
    """
    Exercise 2: Wavefront Analysis from Zernike Coefficients
    Analyze a telescope wavefront from measured Noll Zernike coefficients.
    """
    # Given coefficients (in waves)
    coefficients = {
        2: 0.05,     # Tip (x-tilt)
        3: -0.08,    # Tilt (y-tilt)
        4: 0.30,     # Defocus
        5: -0.15,    # Astigmatism (oblique)
        6: 0.10,     # Astigmatism (vertical)
        7: 0.12,     # Coma (vertical)
        8: -0.06,    # Coma (horizontal)
        11: 0.20,    # Spherical aberration
    }

    names = {
        2: "Tip (x-tilt)", 3: "Tilt (y-tilt)", 4: "Defocus",
        5: "Astigmatism (oblique)", 6: "Astigmatism (vertical)",
        7: "Coma (vertical)", 8: "Coma (horizontal)",
        11: "Spherical"
    }

    print("Wavefront Analysis from Zernike Coefficients:")
    print(f"\n{'j':>4} {'a_j (waves)':>14} {'a_j^2':>12} {'Name':>25}")
    print("-" * 57)

    for j, a in sorted(coefficients.items()):
        print(f"{j:>4} {a:>14.4f} {a**2:>12.6f} {names[j]:>25}")

    # (a) Total RMS wavefront error
    a_values = np.array(list(coefficients.values()))
    sigma_W = np.sqrt(np.sum(a_values**2))
    print(f"\n(a) Total RMS wavefront error:")
    print(f"  sigma_W = sqrt(sum(a_j^2)) = {sigma_W:.6f} waves")
    print(f"  = {sigma_W * 550:.2f} nm (at 550 nm)")

    # (b) Dominant mode
    j_dominant = max(coefficients.keys(), key=lambda j: abs(coefficients[j]))
    print(f"\n(b) Dominant mode: j = {j_dominant} ({names[j_dominant]})")
    print(f"  |a_{j_dominant}| = {abs(coefficients[j_dominant]):.4f} waves")
    print(f"  Contribution to variance: {coefficients[j_dominant]**2/sigma_W**2*100:.1f}%")

    # (c) Strehl ratio
    S = np.exp(-(2 * np.pi * sigma_W)**2)
    print(f"\n(c) Marechal Strehl ratio:")
    print(f"  S = exp(-(2*pi*sigma)^2) = exp(-{(2*np.pi*sigma_W)**2:.4f})")
    print(f"  S = {S:.6f} = {S*100:.4f}%")

    if S > 0.8:
        print(f"  Marechal criterion: S > 0.8 -> Diffraction limited")
    else:
        print(f"  Marechal criterion: S < 0.8 -> Aberration limited")

    # (d) Correct three worst modes
    print(f"\n(d) Correcting three worst modes:")
    sorted_modes = sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)

    for n_correct in [1, 2, 3]:
        modes_to_correct = [j for j, _ in sorted_modes[:n_correct]]
        remaining = {j: a for j, a in coefficients.items() if j not in modes_to_correct}
        sigma_new = np.sqrt(np.sum(np.array(list(remaining.values()))**2))
        S_new = np.exp(-(2 * np.pi * sigma_new)**2)

        corrected_names = [names[j] for j in modes_to_correct]
        print(f"\n  After correcting {n_correct} mode(s): {', '.join(corrected_names)}")
        print(f"    New sigma = {sigma_new:.6f} waves")
        print(f"    New Strehl = {S_new:.6f} = {S_new*100:.4f}%")
        print(f"    Strehl improvement: {(S_new-S)*100:.4f}%")

    # (e) PSF computation concept
    print(f"\n(e) PSF from wavefront (FFT method):")
    print(f"  Pupil function: P(rho,theta) = A(rho) * exp(i * 2*pi * W(rho,theta))")
    print(f"  PSF = |FT[P]|^2")
    print(f"  Strehl = max(PSF_aberrated) / max(PSF_perfect)")

    # Quick numerical PSF check
    N = 256
    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)
    pupil_mask = R <= 1.0

    # Build wavefront (simplified - using analytic forms)
    W = np.zeros((N, N))
    W += coefficients[4] * np.sqrt(3) * (2*R**2 - 1)
    W += coefficients[11] * np.sqrt(5) * (6*R**4 - 6*R**2 + 1)
    W += coefficients[7] * np.sqrt(8) * (3*R**3 - 2*R) * np.sin(Theta)
    W[~pupil_mask] = 0

    # Pupil function
    P_aberrated = pupil_mask * np.exp(1j * 2 * np.pi * W)
    P_perfect = pupil_mask.astype(complex)

    # PSF
    psf_aberr = np.abs(np.fft.fftshift(np.fft.fft2(P_aberrated, s=[512, 512])))**2
    psf_perfect = np.abs(np.fft.fftshift(np.fft.fft2(P_perfect, s=[512, 512])))**2

    S_numeric = psf_aberr.max() / psf_perfect.max()
    print(f"  Numerical Strehl (from FFT): {S_numeric:.6f}")
    print(f"  Marechal approximation:      {S:.6f}")
    print(f"  (Marechal is accurate for S > ~0.1)")


def exercise_3():
    """
    Exercise 3: Atmospheric Phase Screen Generation
    Generate a Kolmogorov phase screen and verify its statistics.
    """
    N = 512
    D = 4.0         # Telescope diameter (m)
    r0 = 0.15       # Fried parameter (m) at 500 nm
    lam = 500e-9    # Wavelength (m)
    dx = D / N      # Grid spacing (m)

    print("Kolmogorov Atmospheric Phase Screen:")
    print(f"Telescope: D = {D} m")
    print(f"Fried parameter: r0 = {r0*100:.0f} cm at {lam*1e9:.0f} nm")
    print(f"D/r0 = {D/r0:.1f}")
    print(f"Grid: {N}x{N}, dx = {dx*1e3:.2f} mm")

    # Generate phase screen using FFT method
    # PSD: Phi(kappa) = 0.023 * r0^(-5/3) * kappa^(-11/3)
    np.random.seed(42)

    fx = np.fft.fftfreq(N, dx)
    FX, FY = np.meshgrid(fx, fx)
    F_rad = np.sqrt(FX**2 + FY**2)
    F_rad[0, 0] = 1e-10  # Avoid division by zero

    # Kolmogorov PSD
    PSD = 0.023 * r0**(-5./3.) * (F_rad * 2 * np.pi)**(-11./3.)
    PSD[0, 0] = 0  # Remove piston

    # Generate random realization
    noise = np.random.randn(N, N) + 1j * np.random.randn(N, N)
    phase_spectrum = noise * np.sqrt(PSD) * (2 * np.pi / dx)

    phase_screen = np.real(np.fft.ifft2(phase_spectrum))
    phase_screen -= np.mean(phase_screen)

    print(f"\nPhase screen statistics:")
    print(f"  RMS phase: {np.std(phase_screen):.4f} rad")
    print(f"  PV phase: {phase_screen.max() - phase_screen.min():.2f} rad")

    # (a) Phase structure function
    print(f"\n(a) Phase Structure Function D_phi(r):")
    print(f"  Theory: D_phi(r) = 6.88 * (r/r0)^(5/3)")
    print(f"\n{'r/r0':>8} {'D_measured':>12} {'D_theory':>12} {'Ratio':>8}")
    print("-" * 42)

    separations = [1, 2, 5, 10, 20, 50]
    for sep_pixels in separations:
        r = sep_pixels * dx
        r_over_r0 = r / r0

        # Compute structure function from screen
        D_meas = np.mean((phase_screen[:, sep_pixels:] - phase_screen[:, :-sep_pixels])**2)

        # Theoretical
        D_theory = 6.88 * (r / r0)**(5./3.)

        ratio = D_meas / D_theory if D_theory > 0 else 0
        print(f"{r_over_r0:>8.2f} {D_meas:>12.4f} {D_theory:>12.4f} {ratio:>8.3f}")

    # (b) Zernike decomposition
    print(f"\n(b) Zernike coefficient magnitudes (first 15 modes):")

    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)
    mask = R <= 1.0
    area = np.sum(mask)

    # Simple Zernike projection (approximate)
    print(f"{'j':>4} {'Name':>20} {'|a_j| (rad)':>14}")
    print("-" * 40)

    zernike_names = {
        2: "Tip", 3: "Tilt", 4: "Defocus",
        5: "Astigmatism-1", 6: "Astigmatism-2",
        7: "Coma-Y", 8: "Coma-X",
        9: "Trefoil-Y", 10: "Trefoil-X",
        11: "Spherical",
    }

    a_tip_tilt = 0
    a_higher = 0

    for j in range(2, 12):
        Z_j = zernike_noll(j, R, Theta)
        Z_j[~mask] = 0
        a_j = np.sum(phase_screen * Z_j * mask) / np.sum(Z_j**2 * mask)
        name = zernike_names.get(j, f"Z{j}")
        print(f"{j:>4} {name:>20} {abs(a_j):>14.4f}")

        if j <= 3:
            a_tip_tilt += a_j**2
        else:
            a_higher += a_j**2

    # (c) Tip-tilt dominance
    total_var = np.var(phase_screen[mask])
    tt_frac = a_tip_tilt / total_var if total_var > 0 else 0
    print(f"\n(c) Tip-tilt variance fraction: {tt_frac*100:.1f}%")
    print(f"  Theory predicts ~87% of total variance in tip-tilt")

    # (d) Residual variance after correction
    print(f"\n(d) Residual variance (Noll formula):")
    print(f"  sigma_J^2 = 0.2944 * J^(-sqrt(3)/2) * (D/r0)^(5/3)")
    D_over_r0 = D / r0

    for J in [1, 3, 10, 21, 36]:
        sigma2_residual = 0.2944 * J**(-np.sqrt(3)/2) * D_over_r0**(5./3.)
        S_residual = np.exp(-(2*np.pi/(2*np.pi))**2 * sigma2_residual)
        # Strehl from residual (sigma in radians, need to convert)
        print(f"  J = {J:>3}: sigma^2 = {sigma2_residual:.4f} rad^2, "
              f"sigma = {np.sqrt(sigma2_residual):.4f} rad")


def exercise_4():
    """
    Exercise 4: Annular Aperture Zernike Extension
    Investigate Zernike polynomials on an annular aperture and
    the need for annular Zernike polynomials.
    """
    N = 128
    eps = 0.3  # Obscuration ratio

    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)
    annulus = (R <= 1.0) & (R >= eps)
    full_circle = R <= 1.0

    print("Annular Aperture Zernike Analysis:")
    print(f"Obscuration ratio: epsilon = {eps}")

    # (a) Gram matrix for standard Zernike on annulus
    n_modes = 10
    print(f"\n(a) Gram matrix G_jj' = <Z_j, Z_j'>_annulus for first {n_modes} modes:")

    Z_modes = []
    for j in range(2, 2 + n_modes):
        Z_j = zernike_noll(j, R, Theta)
        Z_j[~annulus] = 0
        Z_modes.append(Z_j)

    area_annulus = np.sum(annulus)
    G = np.zeros((n_modes, n_modes))
    for i in range(n_modes):
        for j_idx in range(n_modes):
            G[i, j_idx] = np.sum(Z_modes[i] * Z_modes[j_idx] * annulus) / area_annulus

    # Check off-diagonal elements
    max_offdiag = 0
    for i in range(n_modes):
        for j_idx in range(n_modes):
            if i != j_idx:
                max_offdiag = max(max_offdiag, abs(G[i, j_idx]))

    print(f"  Diagonal elements (should be ~1 if orthogonal):")
    for i in range(min(5, n_modes)):
        print(f"    G[{i+2},{i+2}] = {G[i,i]:.4f}")

    print(f"  Maximum off-diagonal: {max_offdiag:.4f}")
    print(f"  Standard Zernike are {'NOT ' if max_offdiag > 0.01 else ''}orthogonal on annulus")

    # (b) Gram-Schmidt orthogonalization
    print(f"\n(b) Gram-Schmidt orthogonalization:")
    orthogonal_modes = []

    for i in range(n_modes):
        v = Z_modes[i].copy()
        for j_idx in range(len(orthogonal_modes)):
            # Project out previous modes
            proj = np.sum(v * orthogonal_modes[j_idx] * annulus) / np.sum(orthogonal_modes[j_idx]**2 * annulus)
            v -= proj * orthogonal_modes[j_idx]

        # Normalize
        norm = np.sqrt(np.sum(v**2 * annulus) / area_annulus)
        if norm > 1e-10:
            v /= norm
        orthogonal_modes.append(v)

    # Verify orthogonality
    G_new = np.zeros((n_modes, n_modes))
    for i in range(n_modes):
        for j_idx in range(n_modes):
            G_new[i, j_idx] = np.sum(orthogonal_modes[i] * orthogonal_modes[j_idx] * annulus) / area_annulus

    max_offdiag_new = 0
    for i in range(n_modes):
        for j_idx in range(n_modes):
            if i != j_idx:
                max_offdiag_new = max(max_offdiag_new, abs(G_new[i, j_idx]))

    print(f"  After Gram-Schmidt:")
    print(f"  Maximum off-diagonal: {max_offdiag_new:.6e}")
    print(f"  Orthogonality achieved: {max_offdiag_new < 1e-6}")

    # (c) Decomposition comparison
    print(f"\n(c) Decomposition of test wavefront (coma + spherical):")

    # Create test wavefront: coma + spherical
    W_test = (0.5 * np.sqrt(8) * (3*R**3 - 2*R) * np.sin(Theta) +
              0.3 * np.sqrt(5) * (6*R**4 - 6*R**2 + 1))
    W_test[~annulus] = 0

    # Standard Zernike coefficients
    print(f"\n  {'Mode':>6} {'Standard a_j':>14} {'Annular a_j':>14}")
    print(f"  " + "-" * 36)

    for i in range(min(5, n_modes)):
        a_std = np.sum(W_test * Z_modes[i] * annulus) / np.sum(Z_modes[i]**2 * annulus)
        a_ann = np.sum(W_test * orthogonal_modes[i] * annulus) / np.sum(orthogonal_modes[i]**2 * annulus)
        print(f"  {i+2:>6} {a_std:>14.6f} {a_ann:>14.6f}")

    # (d) When annular Zernike are necessary
    print(f"\n(d) When annular Zernike polynomials are necessary:")
    print(f"  - Large central obscuration (epsilon > 0.2-0.3)")
    print(f"  - Cross-coupling between modes is significant")
    print(f"  - Quantitative wavefront error budget needed")
    print(f"  - For epsilon = {eps}: standard Zernike have cross-talk up to {max_offdiag:.3f}")
    print(f"  - For epsilon < 0.1: standard Zernike are usually sufficient")
    print(f"  - HST (epsilon ~ 0.33), JWST (epsilon ~ 0.16)")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Radial Polynomial Verification", exercise_1),
        ("Exercise 2: Wavefront Analysis", exercise_2),
        ("Exercise 3: Atmospheric Phase Screen", exercise_3),
        ("Exercise 4: Annular Aperture Extension", exercise_4),
    ]
    for title, func in exercises:
        print(f"\n{'='*60}")
        print(f"=== {title} ===")
        print(f"{'='*60}")
        func()

    print(f"\n{'='*60}")
    print("All exercises completed!")
