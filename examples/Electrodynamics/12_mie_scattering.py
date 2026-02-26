"""
Mie Scattering: Exact Solution for Spherical Particle Scattering
================================================================

Topics covered:
  1. Mie theory: scattering coefficients a_n, b_n
  2. Scattering, extinction, and absorption cross sections vs size parameter
  3. Rayleigh limit verification (small particle limit)
  4. Angular scattering pattern (phase function)

Why Mie theory?
  When an electromagnetic wave hits a spherical particle, the exact
  solution can be expressed as an infinite series of partial waves
  (multipoles). This is Mie theory, named after Gustav Mie (1908).
  It's one of the few EXACT solutions in electrodynamics and has
  enormous practical importance:
    - Atmospheric optics (why the sky is blue, why sunsets are red)
    - Aerosol science and pollution monitoring
    - Biomedical optics (tissue scattering)
    - Nanophotonics (plasmonic nanoparticles)
    - Radar meteorology (rain drops)

Physics background:
  - Size parameter: x = 2*pi*a/lambda  (a = sphere radius)
  - Mie coefficients:
      a_n = [m*psi_n(mx)*psi_n'(x) - psi_n(x)*psi_n'(mx)] /
            [m*psi_n(mx)*xi_n'(x) - xi_n(x)*psi_n'(mx)]
      b_n = [psi_n(mx)*psi_n'(x) - m*psi_n(x)*psi_n'(mx)] /
            [psi_n(mx)*xi_n'(x) - m*xi_n(x)*psi_n'(mx)]
    where m = n_sphere / n_medium (relative refractive index)
    psi_n, xi_n are Riccati-Bessel functions
  - Cross sections:
      Q_sca = (2/x^2) * sum_n (2n+1)(|a_n|^2 + |b_n|^2)
      Q_ext = (2/x^2) * sum_n (2n+1)*Re(a_n + b_n)
      Q_abs = Q_ext - Q_sca
  - Rayleigh limit (x << 1):
      Q_sca ~ (8/3) * x^4 * |(m^2-1)/(m^2+2)|^2
      This is the famous lambda^{-4} dependence that explains blue sky.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, spherical_yn


# ===========================
# Riccati-Bessel Functions
# ===========================

def riccati_bessel_jn(n, z):
    """
    Riccati-Bessel function of the first kind: psi_n(z) = z * j_n(z)

    Why Riccati-Bessel instead of plain Bessel?
      The Mie coefficients are naturally expressed in terms of
      Riccati-Bessel functions, which combine the spherical Bessel
      function with a factor of z. This simplifies the formulas
      and avoids dividing by z (which would be singular at z=0).
    """
    return z * spherical_jn(n, z)


def riccati_bessel_yn(n, z):
    """
    Riccati-Bessel function of the second kind: chi_n(z) = -z * y_n(z)

    Note the sign convention: chi_n = -z*y_n so that the outgoing wave
    function xi_n = psi_n - i*chi_n = z*(j_n + i*y_n) = z*h_n^(1).
    """
    return -z * spherical_yn(n, z)


def riccati_bessel_derivative(n, z, func_type='psi'):
    """
    Compute derivatives of Riccati-Bessel functions using recurrence.

    d[psi_n(z)]/dz = psi_{n-1}(z) - n*psi_n(z)/z

    Why use recurrence rather than finite differences?
      Recurrence is exact (to machine precision), while finite
      differences introduce truncation error. For oscillatory functions
      like Bessel functions, finite differences can be particularly
      inaccurate.
    """
    if func_type == 'psi':
        if n == 0:
            return np.cos(z)  # d[z*j_0(z)]/dz = d[sin(z)]/dz = cos(z)
        psi_n = riccati_bessel_jn(n, z)
        psi_nm1 = riccati_bessel_jn(n - 1, z)
        return psi_nm1 - n * psi_n / z
    elif func_type == 'chi':
        if n == 0:
            return np.sin(z)  # d[-z*y_0(z)]/dz = d[cos(z)]/dz = -sin(z)... check
        chi_n = riccati_bessel_yn(n, z)
        chi_nm1 = riccati_bessel_yn(n - 1, z)
        return chi_nm1 - n * chi_n / z


def xi_n(n, z):
    """
    Outgoing spherical wave Riccati-Bessel function:
    xi_n(z) = psi_n(z) - i * chi_n(z)

    Why complex?
      xi_n represents outgoing spherical waves (like e^{ikr}/r in the
      far field). The combination j_n + i*y_n gives the spherical
      Hankel function of the first kind, which has the correct
      outgoing-wave asymptotic behavior.
    """
    return riccati_bessel_jn(n, z) - 1j * riccati_bessel_yn(n, z)


def xi_n_derivative(n, z):
    """Derivative of xi_n."""
    return riccati_bessel_derivative(n, z, 'psi') - \
           1j * riccati_bessel_derivative(n, z, 'chi')


# ===========================
# 1. Mie Coefficients
# ===========================

def mie_coefficients(m, x, n_max=None):
    """
    Compute Mie scattering coefficients a_n and b_n.

    Parameters
    ----------
    m : complex
        Relative refractive index (n_sphere / n_medium).
    x : float
        Size parameter (2*pi*a/lambda).
    n_max : int or None
        Maximum multipole order. If None, use Wiscombe criterion.

    Returns
    -------
    a_n, b_n : arrays of complex
        Mie coefficients for n = 1, 2, ..., n_max.

    Why sum only a finite number of terms?
      The Mie series converges: for large n, a_n and b_n become
      exponentially small. The Wiscombe criterion n_max ~ x + 4*x^{1/3} + 2
      ensures we include enough terms without wasting computation.
    """
    if n_max is None:
        # Wiscombe (1980) criterion for truncation
        # Why this formula?
        #   It's an empirically validated rule that guarantees the
        #   series error is below machine precision for most cases.
        n_max = int(x + 4 * x**(1.0 / 3) + 2)
        n_max = max(n_max, 3)  # at least 3 terms for very small x

    a = np.zeros(n_max, dtype=complex)
    b = np.zeros(n_max, dtype=complex)

    mx = m * x

    for n in range(1, n_max + 1):
        # Riccati-Bessel functions and their derivatives
        psi_x = riccati_bessel_jn(n, x)
        psi_mx = riccati_bessel_jn(n, mx)
        dpsi_x = riccati_bessel_derivative(n, x, 'psi')
        dpsi_mx = riccati_bessel_derivative(n, mx, 'psi')

        xi_x_val = xi_n(n, x)
        dxi_x = xi_n_derivative(n, x)

        # Mie coefficients
        # Why these particular combinations?
        #   They arise from matching boundary conditions (continuity of
        #   tangential E and H) at the sphere surface r = a. The
        #   numerator and denominator come from the determinant of the
        #   boundary condition matrix.
        a[n - 1] = (m * psi_mx * dpsi_x - psi_x * dpsi_mx) / \
                   (m * psi_mx * dxi_x - xi_x_val * dpsi_mx)

        b[n - 1] = (psi_mx * dpsi_x - m * psi_x * dpsi_mx) / \
                   (psi_mx * dxi_x - m * xi_x_val * dpsi_mx)

    return a, b


# ===========================
# 2. Cross Sections vs Size Parameter
# ===========================

def cross_sections(m, x_values):
    """
    Compute scattering, extinction, and absorption efficiency factors
    Q_sca, Q_ext, Q_abs as functions of size parameter.

    Q = sigma / (pi * a^2)  (efficiency = cross section / geometric area)

    Why efficiency Q instead of cross section sigma?
      Q is dimensionless and depends only on m and x (not on the absolute
      size). This makes comparisons between different particles easier.
      Q > 1 means the particle interacts with more light than its
      geometric shadow would suggest (diffraction contribution).
    """
    Q_sca = np.zeros_like(x_values)
    Q_ext = np.zeros_like(x_values)

    for i, x in enumerate(x_values):
        if x < 1e-6:
            continue  # skip x=0

        a_n, b_n = mie_coefficients(m, x)
        n_vals = np.arange(1, len(a_n) + 1)

        # Scattering efficiency
        Q_sca[i] = (2.0 / x**2) * np.sum(
            (2 * n_vals + 1) * (np.abs(a_n)**2 + np.abs(b_n)**2)
        )

        # Extinction efficiency (extinction = scattering + absorption)
        Q_ext[i] = (2.0 / x**2) * np.sum(
            (2 * n_vals + 1) * np.real(a_n + b_n)
        )

    Q_abs = Q_ext - Q_sca

    return Q_sca, Q_ext, Q_abs


def plot_cross_sections():
    """
    Plot Q_sca, Q_ext, Q_abs vs size parameter for a dielectric sphere.

    Key features to observe:
      - Rayleigh region (x << 1): Q_sca ~ x^4 (very weak scattering)
      - Mie resonances (x ~ 1-10): oscillatory behavior (interference)
      - Geometric optics limit (x >> 1): Q_ext -> 2 (extinction paradox)
    """
    x = np.linspace(0.01, 25, 500)

    # Non-absorbing sphere (real m)
    m_real = 1.5 + 0j
    Q_sca_r, Q_ext_r, Q_abs_r = cross_sections(m_real, x)

    # Absorbing sphere (complex m)
    m_complex = 1.5 + 0.1j
    Q_sca_c, Q_ext_c, Q_abs_c = cross_sections(m_complex, x)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Non-absorbing
    ax = axes[0]
    ax.plot(x, Q_ext_r, 'b-', linewidth=2, label=r'$Q_{ext}$')
    ax.plot(x, Q_sca_r, 'r--', linewidth=2, label=r'$Q_{sca}$')
    ax.axhline(y=2, color='gray', linestyle=':', alpha=0.5,
               label=r'$Q_{ext} \to 2$ (geometric limit)')
    ax.set_xlabel('Size parameter x = 2$\\pi$a/$\\lambda$')
    ax.set_ylabel('Efficiency Q')
    ax.set_title(f'Non-absorbing sphere (m = {m_real})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 5)

    # Absorbing
    ax = axes[1]
    ax.plot(x, Q_ext_c, 'b-', linewidth=2, label=r'$Q_{ext}$')
    ax.plot(x, Q_sca_c, 'r--', linewidth=2, label=r'$Q_{sca}$')
    ax.plot(x, Q_abs_c, 'g:', linewidth=2, label=r'$Q_{abs}$')
    ax.axhline(y=2, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Size parameter x')
    ax.set_ylabel('Efficiency Q')
    ax.set_title(f'Absorbing sphere (m = {m_complex})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 5)

    plt.tight_layout()
    plt.savefig('12_mie_cross_sections.png', dpi=150)
    plt.close()
    print("[Saved] 12_mie_cross_sections.png")

    # Why does Q_ext -> 2 for large x?
    #   This is the "extinction paradox": a large sphere blocks its
    #   geometric shadow (Q = 1 from blocking) AND diffracts an equal
    #   amount of light around its edges (another Q = 1), giving Q = 2.
    #   This is counterintuitive but confirmed experimentally.


# ===========================
# 3. Rayleigh Limit Verification
# ===========================

def rayleigh_verification():
    """
    Verify that Mie theory reduces to the Rayleigh approximation for x << 1.

    Rayleigh scattering:
      Q_sca = (8/3) * x^4 * |(m^2 - 1)/(m^2 + 2)|^2

    Why is the lambda^{-4} dependence important?
      Short wavelengths (blue light) scatter much more than long
      wavelengths (red light). This is why:
        - The sky is blue (scattered sunlight reaching your eyes)
        - Sunsets are red (blue light scattered away, red passes through)
        - Distant mountains appear blue (Rayleigh scattering of sunlight)
    """
    m = 1.5 + 0j
    x = np.logspace(-3, 0, 200)  # x from 0.001 to 1

    Q_sca_mie, _, _ = cross_sections(m, x)

    # Rayleigh formula
    K = (m**2 - 1) / (m**2 + 2)
    Q_sca_rayleigh = (8.0 / 3.0) * x**4 * np.abs(K)**2

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Log-log comparison
    ax = axes[0]
    ax.loglog(x, Q_sca_mie, 'b-', linewidth=2, label='Mie (exact)')
    ax.loglog(x, Q_sca_rayleigh, 'r--', linewidth=2, label='Rayleigh approx')
    ax.set_xlabel('Size parameter x')
    ax.set_ylabel(r'$Q_{sca}$')
    ax.set_title('Mie vs Rayleigh Scattering')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Relative error
    ax = axes[1]
    valid = Q_sca_mie > 1e-20
    rel_error = np.abs(Q_sca_mie[valid] - Q_sca_rayleigh[valid]) / Q_sca_mie[valid]
    ax.loglog(x[valid], rel_error, 'g-', linewidth=2)
    ax.axhline(y=0.01, color='red', linestyle='--', label='1% error')
    ax.axhline(y=0.1, color='orange', linestyle='--', label='10% error')
    ax.set_xlabel('Size parameter x')
    ax.set_ylabel('Relative error |Mie - Rayleigh| / Mie')
    ax.set_title('Rayleigh Approximation Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Find x where Rayleigh becomes 10% inaccurate
    idx_10pct = np.where(rel_error > 0.1)[0]
    if len(idx_10pct) > 0:
        x_10pct = x[valid][idx_10pct[0]]
        print(f"  Rayleigh error exceeds 10% at x = {x_10pct:.3f}")
        print(f"  (particle radius ~ {x_10pct/(2*np.pi):.4f} * wavelength)")

    plt.tight_layout()
    plt.savefig('12_rayleigh_verification.png', dpi=150)
    plt.close()
    print("[Saved] 12_rayleigh_verification.png")

    # Verify lambda^{-4} scaling
    # Q_sca ~ x^4 ~ (a/lambda)^4 ~ lambda^{-4}
    slope = np.polyfit(np.log10(x[:50]), np.log10(Q_sca_mie[:50] + 1e-30), 1)[0]
    print(f"  Power law slope in Rayleigh regime: {slope:.2f} (should be ~4.0)")


# ===========================
# 4. Angular Scattering Pattern (Phase Function)
# ===========================

def angular_scattering(m, x):
    """
    Compute the angular scattering pattern (phase function).

    The far-field scattering amplitudes are:
      S1(theta) = sum_n (2n+1)/(n(n+1)) * [a_n*pi_n(cos theta) + b_n*tau_n(cos theta)]
      S2(theta) = sum_n (2n+1)/(n(n+1)) * [a_n*tau_n(cos theta) + b_n*pi_n(cos theta)]

    where pi_n and tau_n are angular functions derived from associated
    Legendre polynomials:
      pi_n(cos theta) = P_n^1(cos theta) / sin(theta)
      tau_n(cos theta) = d[P_n^1(cos theta)] / d(theta)

    The scattered intensity for unpolarized light:
      I(theta) ~ |S1|^2 + |S2|^2

    Why two scattering amplitudes?
      S1 and S2 correspond to the two polarization components of the
      scattered wave (perpendicular and parallel to the scattering plane).
      Their ratio determines the polarization of scattered light.
    """
    a_n, b_n = mie_coefficients(m, x)
    n_max = len(a_n)

    theta = np.linspace(0, np.pi, 500)
    mu = np.cos(theta)

    S1 = np.zeros(len(theta), dtype=complex)
    S2 = np.zeros(len(theta), dtype=complex)

    for n in range(1, n_max + 1):
        # Compute pi_n and tau_n using recurrence relations
        # Why recurrence?
        #   Direct computation of associated Legendre polynomials is
        #   numerically unstable for large n. The recurrence:
        #     pi_n = (2n-1)/(n-1) * mu * pi_{n-1} - n/(n-1) * pi_{n-2}
        #     tau_n = n * mu * pi_n - (n+1) * pi_{n-1}
        #   is stable and efficient.
        pi_n_arr = np.zeros(len(theta))
        pi_nm1 = np.zeros(len(theta))
        pi_nm2 = np.zeros(len(theta))

        # Initial conditions: pi_0 = 0, pi_1 = 1
        if n == 1:
            pi_n_arr = np.ones(len(theta))
        else:
            # Build up from pi_1 using recurrence
            pi_prev_prev = np.zeros(len(theta))
            pi_prev = np.ones(len(theta))
            for nn in range(2, n + 1):
                pi_curr = ((2 * nn - 1) / (nn - 1)) * mu * pi_prev - \
                          (nn / (nn - 1)) * pi_prev_prev
                pi_prev_prev = pi_prev
                pi_prev = pi_curr
            pi_n_arr = pi_curr

        # tau_n = n*mu*pi_n - (n+1)*pi_{n-1}
        if n == 1:
            tau_n_arr = mu * pi_n_arr
        else:
            tau_n_arr = n * mu * pi_n_arr - (n + 1) * pi_prev_prev

        # Accumulate scattering amplitudes
        factor = (2 * n + 1) / (n * (n + 1))
        S1 += factor * (a_n[n - 1] * pi_n_arr + b_n[n - 1] * tau_n_arr)
        S2 += factor * (a_n[n - 1] * tau_n_arr + b_n[n - 1] * pi_n_arr)

    # Intensity for unpolarized light
    I_unpol = 0.5 * (np.abs(S1)**2 + np.abs(S2)**2)

    return theta, S1, S2, I_unpol


def plot_phase_functions():
    """
    Plot angular scattering patterns for different size parameters.

    Key observations:
      - x << 1 (Rayleigh): nearly symmetric front-back (1 + cos^2(theta))
      - x ~ 1 (Mie): forward scattering begins to dominate
      - x >> 1 (geometric): strong forward peak, complex side lobes
    """
    m = 1.5 + 0j
    x_values = [0.1, 1.0, 5.0, 10.0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, x_val in zip(axes, x_values):
        theta, S1, S2, I_unpol = angular_scattering(m, x_val)
        theta_deg = np.degrees(theta)

        # Normalize
        I_norm = I_unpol / np.max(I_unpol)

        # Polar plot (upper) in a new axes
        ax.semilogy(theta_deg, I_norm, 'b-', linewidth=2)
        ax.set_xlabel(r'$\theta$ (degrees)')
        ax.set_ylabel(r'$I(\theta) / I_{max}$')
        ax.set_title(f'x = {x_val}')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(1e-4, 2)

        # Mark forward and backward
        ax.axvline(x=0, color='green', linestyle=':', alpha=0.5, label='Forward')
        ax.axvline(x=180, color='red', linestyle=':', alpha=0.5, label='Backward')

        # Compute asymmetry parameter g = <cos(theta)>
        # Why the asymmetry parameter?
        #   g quantifies how forward-peaked the scattering is.
        #   g = 0: isotropic,  g = 1: all forward,  g = -1: all backward
        #   It's crucial for radiative transfer (how light diffuses
        #   through clouds, tissue, etc.)
        dtheta = theta[1] - theta[0]
        g = np.sum(I_unpol * np.cos(theta) * np.sin(theta) * dtheta) / \
            np.sum(I_unpol * np.sin(theta) * dtheta)
        ax.legend(title=f'g = {g:.3f}', fontsize=9)

    plt.suptitle(f'Angular Scattering Patterns (m = {m})', fontsize=14)
    plt.tight_layout()
    plt.savefig('12_mie_phase_functions.png', dpi=150)
    plt.close()
    print("[Saved] 12_mie_phase_functions.png")


def plot_polar_scattering():
    """Create polar plots showing the angular scattering pattern."""
    m = 1.5 + 0j
    x_values = [0.5, 2.0, 5.0, 10.0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 11),
                              subplot_kw={'projection': 'polar'})
    axes = axes.flatten()

    for ax, x_val in zip(axes, x_values):
        theta, _, _, I_unpol = angular_scattering(m, x_val)

        # Use log scale for visibility
        I_log = np.log10(I_unpol / np.max(I_unpol) + 1e-10) + 4
        I_log = np.maximum(I_log, 0)

        # Plot both forward and backward hemispheres
        ax.plot(theta, I_log, 'b-', linewidth=2)
        ax.plot(-theta, I_log, 'b-', linewidth=2)  # mirror for full pattern
        ax.set_title(f'x = {x_val}', pad=15)
        ax.set_rlabel_position(0)

    plt.suptitle(f'Mie Scattering Polar Patterns (m = {m}, log scale)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('12_mie_polar.png', dpi=150)
    plt.close()
    print("[Saved] 12_mie_polar.png")


# ===========================
# 5. Wavelength-Dependent Scattering (Color of Sky)
# ===========================

def sky_color_demo():
    """
    Demonstrate why the sky is blue using Mie/Rayleigh scattering.

    The key: air molecules (N2, O2) have radius ~ 0.1-0.2 nm,
    so x = 2*pi*a/lambda ~ 10^{-3} for visible light.
    This puts us firmly in the Rayleigh regime where Q_sca ~ lambda^{-4}.

    Why compute this with Mie theory?
      To show that the exact theory agrees with the Rayleigh approximation
      for small particles, and to illustrate the dramatic wavelength
      dependence that makes the sky blue.
    """
    # Visible spectrum
    lam_nm = np.linspace(380, 700, 200)
    lam_m = lam_nm * 1e-9

    # Effective air molecule radius
    a = 0.15e-9  # ~0.15 nm

    # Refractive index of air (very close to 1)
    # Why not exactly 1?
    #   If m = 1, there's no scattering at all! The tiny deviation
    #   from 1 (about 0.0003) is what causes Rayleigh scattering.
    m = 1.0003 + 0j

    x_values = 2 * np.pi * a / lam_m

    # Compute Q_sca for each wavelength
    Q_sca_arr = np.zeros_like(lam_nm)
    for i, x in enumerate(x_values):
        Q_sca_arr[i], _, _ = cross_sections(m, np.array([x]))

    # Rayleigh prediction
    K = (m**2 - 1) / (m**2 + 2)
    Q_rayleigh = (8.0 / 3.0) * x_values**4 * np.abs(K)**2

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Q_sca vs wavelength
    ax = axes[0]
    Q_norm = Q_sca_arr / np.max(Q_sca_arr)
    ax.plot(lam_nm, Q_norm, 'b-', linewidth=2, label='Mie')
    Q_rayleigh_norm = Q_rayleigh / np.max(Q_rayleigh)
    ax.plot(lam_nm, Q_rayleigh_norm, 'r--', linewidth=2, label='Rayleigh')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel(r'$Q_{sca}$ (normalized)')
    ax.set_title('Scattering Efficiency vs Wavelength')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Color spectrum visualization
    ax = axes[1]
    # Why use actual colors?
    #   This makes the physics visceral: blue light scatters MUCH more
    #   than red light. The color bar shows the visible spectrum.
    colors = wavelength_to_rgb(lam_nm)
    for i in range(len(lam_nm) - 1):
        ax.bar(lam_nm[i], Q_norm[i], width=lam_nm[1] - lam_nm[0],
               color=colors[i], edgecolor='none')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel(r'Relative scattering')
    ax.set_title('Why the Sky is Blue: Blue Scatters ~5x More Than Red')
    ax.grid(True, alpha=0.3)

    ratio_blue_red = Q_sca_arr[0] / Q_sca_arr[-1]
    ax.annotate(f'Blue/Red ratio: {ratio_blue_red:.1f}x',
                xy=(450, 0.8), fontsize=12, color='blue')

    plt.tight_layout()
    plt.savefig('12_sky_color.png', dpi=150)
    plt.close()
    print("[Saved] 12_sky_color.png")
    print(f"  Blue (400nm) / Red (700nm) scattering ratio: {ratio_blue_red:.1f}x")
    print(f"  Theoretical (700/400)^4 = {(700/400)**4:.1f}x")


def wavelength_to_rgb(lam_nm):
    """
    Approximate conversion from wavelength (nm) to RGB color.

    Why approximate?
      The human eye's response is complex (CIE color matching functions).
      This simplified mapping gives visually reasonable colors for
      demonstration purposes.
    """
    colors = []
    for lam in lam_nm:
        if 380 <= lam < 440:
            r = -(lam - 440) / (440 - 380)
            g = 0.0
            b = 1.0
        elif 440 <= lam < 490:
            r = 0.0
            g = (lam - 440) / (490 - 440)
            b = 1.0
        elif 490 <= lam < 510:
            r = 0.0
            g = 1.0
            b = -(lam - 510) / (510 - 490)
        elif 510 <= lam < 580:
            r = (lam - 510) / (580 - 510)
            g = 1.0
            b = 0.0
        elif 580 <= lam < 645:
            r = 1.0
            g = -(lam - 645) / (645 - 580)
            b = 0.0
        elif 645 <= lam <= 700:
            r = 1.0
            g = 0.0
            b = 0.0
        else:
            r = g = b = 0.0
        colors.append((r, g, b))
    return colors


# ===========================
# Main
# ===========================

if __name__ == '__main__':
    print("=== Mie Cross Sections ===")
    plot_cross_sections()

    print("\n=== Rayleigh Limit Verification ===")
    rayleigh_verification()

    print("\n=== Angular Scattering Patterns ===")
    plot_phase_functions()

    print("\n=== Polar Scattering Plots ===")
    plot_polar_scattering()

    print("\n=== Why the Sky is Blue ===")
    sky_color_demo()
