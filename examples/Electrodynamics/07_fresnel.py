"""
Fresnel Equations: Reflection and Transmission at Interfaces
============================================================

Topics covered:
  1. Fresnel coefficients vs angle (s- and p-polarization)
  2. Brewster angle identification
  3. Total internal reflection and critical angle
  4. Anti-reflection coating thickness optimization

Why Fresnel equations matter:
  When an EM wave hits an interface between two media (e.g., air-glass),
  part of it reflects and part transmits. The Fresnel equations give
  the exact reflection and transmission coefficients as functions of
  angle and polarization. They explain everyday phenomena (glare on
  water, fiber optics) and are essential for optical coating design.

Physics background:
  - s-polarization (TE): E perpendicular to plane of incidence
  - p-polarization (TM): E in the plane of incidence
  - Fresnel reflection coefficients:
      r_s = (n1*cos(theta_i) - n2*cos(theta_t)) / (n1*cos(theta_i) + n2*cos(theta_t))
      r_p = (n2*cos(theta_i) - n1*cos(theta_t)) / (n2*cos(theta_i) + n1*cos(theta_t))
  - Snell's law: n1*sin(theta_i) = n2*sin(theta_t)
  - Brewster angle: tan(theta_B) = n2/n1  (r_p = 0)
  - Critical angle: sin(theta_c) = n2/n1  (total internal reflection, n1 > n2)
"""

import numpy as np
import matplotlib.pyplot as plt


def fresnel_coefficients(n1, n2, theta_i):
    """
    Compute Fresnel reflection and transmission coefficients.

    Parameters
    ----------
    n1, n2 : float
        Refractive indices of media 1 (incident) and 2 (transmitted).
    theta_i : array
        Angle of incidence in radians.

    Returns
    -------
    r_s, r_p : arrays
        Reflection amplitude coefficients (can be complex for TIR).
    t_s, t_p : arrays
        Transmission amplitude coefficients.
    R_s, R_p : arrays
        Reflectance (power reflection coefficient) = |r|^2.
    T_s, T_p : arrays
        Transmittance (power transmission coefficient).

    Why return both amplitude and power coefficients?
      Amplitude coefficients (r, t) can be negative or complex, encoding
      phase shifts. Power coefficients (R, T) are always real and
      non-negative, representing measurable energy fractions.
      R + T = 1 (energy conservation) at every angle.
    """
    # Snell's law to get transmission angle
    # Why use complex arithmetic?
    #   When n1 > n2 and theta_i > theta_c, sin(theta_t) > 1, making
    #   cos(theta_t) imaginary. This is total internal reflection.
    #   Complex arithmetic handles this naturally without special cases.
    sin_t = (n1 / n2) * np.sin(theta_i)
    cos_t = np.sqrt(1 - sin_t**2 + 0j)  # +0j forces complex
    cos_i = np.cos(theta_i)

    # Fresnel amplitude coefficients
    # Why two separate formulas for s and p?
    #   s and p polarizations have different boundary conditions at the
    #   interface. For s-pol, E is continuous and H_parallel is continuous.
    #   For p-pol, the roles of E and H swap. This leads to different
    #   formulas even though the physics (Maxwell's BCs) is the same.
    r_s = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
    r_p = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)

    t_s = 2 * n1 * cos_i / (n1 * cos_i + n2 * cos_t)
    t_p = 2 * n1 * cos_i / (n2 * cos_i + n1 * cos_t)

    # Power coefficients
    R_s = np.abs(r_s)**2
    R_p = np.abs(r_p)**2

    # Why the factor (n2*cos_t)/(n1*cos_i) in transmittance?
    #   The transmitted beam has a different cross-section and propagation
    #   speed than the incident beam. Energy conservation requires:
    #   T = |t|^2 * Re(n2*cos_t) / (n1*cos_i)
    #   This factor accounts for the change in beam width (cos ratio)
    #   and wave impedance (n ratio).
    T_s = np.abs(t_s)**2 * np.real(n2 * cos_t) / (n1 * cos_i)
    T_p = np.abs(t_p)**2 * np.real(n2 * cos_t) / (n1 * cos_i)

    return r_s, r_p, t_s, t_p, R_s, R_p, T_s, T_p


# ===========================
# 1. Fresnel Coefficients vs Angle
# ===========================

def plot_fresnel_external(n1=1.0, n2=1.5):
    """
    Plot Fresnel coefficients for external reflection (n1 < n2).
    Example: air to glass.

    Why start with external reflection?
      This is the most common case in everyday optics: light hitting
      a window, lens, or water surface. No total internal reflection
      occurs (that requires n1 > n2).
    """
    theta_i = np.linspace(0, np.pi / 2 - 0.001, 500)
    _, _, _, _, R_s, R_p, T_s, T_p = fresnel_coefficients(n1, n2, theta_i)

    theta_B = np.arctan(n2 / n1)  # Brewster angle

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Reflectance
    ax = axes[0]
    ax.plot(np.degrees(theta_i), R_s, 'b-', linewidth=2, label=r'$R_s$ (s-pol)')
    ax.plot(np.degrees(theta_i), R_p, 'r--', linewidth=2, label=r'$R_p$ (p-pol)')
    ax.axvline(x=np.degrees(theta_B), color='green', linestyle=':',
               label=rf'Brewster: $\theta_B$ = {np.degrees(theta_B):.1f} deg')
    ax.set_xlabel('Angle of incidence (degrees)')
    ax.set_ylabel('Reflectance R')
    ax.set_title(f'Reflectance: n1={n1}, n2={n2} (air to glass)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Transmittance (verify R + T = 1)
    ax = axes[1]
    ax.plot(np.degrees(theta_i), T_s, 'b-', linewidth=2, label=r'$T_s$')
    ax.plot(np.degrees(theta_i), T_p, 'r--', linewidth=2, label=r'$T_p$')
    ax.plot(np.degrees(theta_i), R_s + T_s, 'k:', linewidth=1, alpha=0.5,
            label=r'$R_s + T_s$ (should = 1)')
    ax.set_xlabel('Angle of incidence (degrees)')
    ax.set_ylabel('Transmittance T')
    ax.set_title('Transmittance (energy conservation: R + T = 1)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig('07_fresnel_external.png', dpi=150)
    plt.close()
    print("[Saved] 07_fresnel_external.png")
    print(f"  Brewster angle: {np.degrees(theta_B):.2f} degrees")
    print(f"  Normal incidence R = {R_s[0]:.4f} = ((n1-n2)/(n1+n2))^2 = "
          f"{((n1-n2)/(n1+n2))**2:.4f}")


# ===========================
# 2. Brewster Angle
# ===========================

def brewster_demo():
    """
    Demonstrate the Brewster angle for multiple refractive index ratios.

    Why is Brewster's angle physically significant?
      At the Brewster angle, reflected light is perfectly s-polarized
      (no p-component). This happens because the reflected and
      transmitted rays are perpendicular (90 degrees apart). The
      oscillating dipoles in the medium radiate perpendicular to their
      oscillation direction, so p-polarized dipoles can't radiate in
      the reflection direction.

    Practical applications:
      - Polarizing windows (Brewster windows in lasers)
      - Polarizing sunglasses (reduce glare from horizontal surfaces)
      - Measuring refractive index (ellipsometry)
    """
    theta_i = np.linspace(0, 89.9, 500)
    n_values = [1.3, 1.5, 1.8, 2.0, 2.5]

    fig, ax = plt.subplots(figsize=(10, 6))

    for n2 in n_values:
        theta_rad = np.radians(theta_i)
        _, _, _, _, _, R_p, _, _ = fresnel_coefficients(1.0, n2, theta_rad)
        theta_B = np.degrees(np.arctan(n2))
        ax.plot(theta_i, R_p, linewidth=2, label=f'n = {n2} (Brewster = {theta_B:.1f} deg)')
        # Mark Brewster angle
        ax.plot(theta_B, 0, 'ko', markersize=5)

    ax.set_xlabel('Angle of incidence (degrees)')
    ax.set_ylabel(r'$R_p$ (p-polarization reflectance)')
    ax.set_title("Brewster Angle: p-Reflectance Vanishes")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 0.5)

    plt.tight_layout()
    plt.savefig('07_brewster.png', dpi=150)
    plt.close()
    print("[Saved] 07_brewster.png")


# ===========================
# 3. Total Internal Reflection
# ===========================

def total_internal_reflection(n1=1.5, n2=1.0):
    """
    Demonstrate total internal reflection (TIR) for n1 > n2.
    Example: glass to air.

    Why does TIR occur?
      Snell's law: sin(theta_t) = (n1/n2) * sin(theta_i).
      When n1 > n2, there exists a critical angle theta_c where
      sin(theta_t) = 1 (theta_t = 90 deg). Beyond this angle,
      there is no real transmitted angle -- ALL light reflects.

    Why is TIR perfect (R = 1 exactly)?
      Unlike partial reflection (where energy splits), TIR reflects
      100% of incident power. The transmitted "wave" becomes evanescent
      (exponentially decaying), carrying no energy on average.
      This is the basis of fiber optics and prism-based optics.
    """
    theta_c = np.arcsin(n2 / n1)
    print(f"  Critical angle: {np.degrees(theta_c):.2f} degrees")

    theta_i = np.linspace(0, np.pi / 2 - 0.001, 500)
    r_s, r_p, _, _, R_s, R_p, T_s, T_p = fresnel_coefficients(n1, n2, theta_i)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Reflectance
    ax = axes[0]
    ax.plot(np.degrees(theta_i), R_s, 'b-', linewidth=2, label=r'$R_s$')
    ax.plot(np.degrees(theta_i), R_p, 'r--', linewidth=2, label=r'$R_p$')
    ax.axvline(x=np.degrees(theta_c), color='green', linestyle=':',
               linewidth=2, label=rf'$\theta_c$ = {np.degrees(theta_c):.1f} deg')
    ax.fill_between(np.degrees(theta_i), 0, 1,
                    where=(theta_i > theta_c), alpha=0.1, color='yellow',
                    label='TIR region')
    ax.set_xlabel('Angle of incidence (degrees)')
    ax.set_ylabel('Reflectance R')
    ax.set_title(f'Total Internal Reflection: n1={n1}, n2={n2}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)

    # Phase shift on reflection during TIR
    ax = axes[1]
    # Why plot the phase?
    #   During TIR, the reflection coefficient is complex with |r| = 1.
    #   The phase tells us how much the reflected wave is shifted in time.
    #   This phase shift is the Goos-Hanchen effect's origin and is used
    #   in phase-matching thin film designs.
    phase_s = np.angle(r_s)
    phase_p = np.angle(r_p)

    ax.plot(np.degrees(theta_i), np.degrees(phase_s), 'b-', linewidth=2,
            label=r'$\phi_s$')
    ax.plot(np.degrees(theta_i), np.degrees(phase_p), 'r--', linewidth=2,
            label=r'$\phi_p$')
    ax.axvline(x=np.degrees(theta_c), color='green', linestyle=':',
               linewidth=2, label=rf'$\theta_c$')
    ax.set_xlabel('Angle of incidence (degrees)')
    ax.set_ylabel('Phase shift (degrees)')
    ax.set_title('Phase Shift on Reflection')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('07_tir.png', dpi=150)
    plt.close()
    print("[Saved] 07_tir.png")


# ===========================
# 4. Anti-Reflection Coating
# ===========================

def anti_reflection_coating():
    """
    Optimize single-layer anti-reflection (AR) coating thickness.

    An AR coating works by destructive interference between reflections
    from the coating's top and bottom surfaces.

    Optimal conditions:
      1. Coating thickness = lambda / (4 * n_coating)  (quarter-wave)
      2. n_coating = sqrt(n_substrate)  (impedance matching)

    Why quarter-wave thickness?
      Light reflecting from the bottom surface travels an extra path
      of 2*d*n_coating (down and back up through the coating). For
      destructive interference, this must equal lambda/2:
        2 * d * n_coating = lambda / 2  -->  d = lambda / (4 * n_coating)
      The two reflected beams (top and bottom of coating) then have
      opposite phase and cancel.

    Why n_coating = sqrt(n_substrate)?
      For perfect cancellation, the amplitudes of the two reflections
      must be equal. The reflection at air-coating interface has
      r1 = (1 - n_c)/(1 + n_c), and at coating-substrate interface
      r2 = (n_c - n_s)/(n_c + n_s). Setting |r1| = |r2| gives n_c = sqrt(n_s).
    """
    n_air = 1.0
    n_glass = 1.52  # BK7 glass

    # Optimal coating: n_c = sqrt(n_glass) ~ 1.233 (MgF2 ~ 1.38 is close)
    n_coating_optimal = np.sqrt(n_glass)
    n_MgF2 = 1.38  # practical coating material

    # Wavelength range (visible light)
    lam = np.linspace(380, 780, 500)  # nm

    # Design wavelength (center of visible)
    lam_design = 550  # nm

    def reflectance_coating(n_c, d_nm, lam_nm):
        """
        Compute reflectance of a single-layer coating using the
        transfer matrix method.

        Why transfer matrix instead of simple formula?
          The simple interference formula assumes normal incidence and
          ignores multiple internal reflections. The transfer matrix
          handles all of these correctly with minimal extra complexity.
        """
        # Phase accumulated in coating layer
        delta = 2 * np.pi * n_c * d_nm / lam_nm

        # Fresnel coefficients at each interface
        r12 = (n_air - n_c) / (n_air + n_c)
        r23 = (n_c - n_glass) / (n_c + n_glass)

        # Total reflection (including multiple reflections via Airy formula)
        # r_total = (r12 + r23 * exp(-2j*delta)) / (1 + r12*r23*exp(-2j*delta))
        r_total = (r12 + r23 * np.exp(-2j * delta)) / \
                  (1 + r12 * r23 * np.exp(-2j * delta))
        R = np.abs(r_total)**2
        return R

    # Quarter-wave thickness at design wavelength
    d_optimal = lam_design / (4 * n_coating_optimal)
    d_MgF2 = lam_design / (4 * n_MgF2)

    # Compute reflectance spectra
    R_bare = ((n_air - n_glass) / (n_air + n_glass))**2 * np.ones_like(lam)
    R_optimal = reflectance_coating(n_coating_optimal, d_optimal, lam)
    R_MgF2 = reflectance_coating(n_MgF2, d_MgF2, lam)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Spectral reflectance
    ax = axes[0]
    ax.plot(lam, R_bare * 100, 'gray', linewidth=2, label='Bare glass')
    ax.plot(lam, R_optimal * 100, 'b-', linewidth=2,
            label=f'Optimal n={n_coating_optimal:.3f}')
    ax.plot(lam, R_MgF2 * 100, 'r--', linewidth=2,
            label=f'MgF2 n={n_MgF2}')
    ax.axvline(x=lam_design, color='green', linestyle=':', alpha=0.5,
               label=f'Design wavelength {lam_design} nm')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Reflectance (%)')
    ax.set_title('Anti-Reflection Coating Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 5)

    # Thickness optimization at design wavelength
    ax = axes[1]
    d_range = np.linspace(0, 300, 500)
    R_vs_d_opt = reflectance_coating(n_coating_optimal, d_range, lam_design)
    R_vs_d_MgF2 = reflectance_coating(n_MgF2, d_range, lam_design)

    ax.plot(d_range, R_vs_d_opt * 100, 'b-', linewidth=2,
            label=f'n = {n_coating_optimal:.3f} (optimal)')
    ax.plot(d_range, R_vs_d_MgF2 * 100, 'r--', linewidth=2,
            label=f'n = {n_MgF2} (MgF2)')
    ax.axvline(x=d_optimal, color='blue', linestyle=':', alpha=0.5)
    ax.axvline(x=d_MgF2, color='red', linestyle=':', alpha=0.5)
    ax.set_xlabel('Coating thickness (nm)')
    ax.set_ylabel('Reflectance (%)')
    ax.set_title(f'Reflectance vs Thickness (at {lam_design} nm)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('07_ar_coating.png', dpi=150)
    plt.close()
    print("[Saved] 07_ar_coating.png")

    print(f"\n  Bare glass reflectance: {R_bare[0]*100:.2f}%")
    print(f"  Optimal coating (n={n_coating_optimal:.3f}, d={d_optimal:.1f} nm): "
          f"R = {R_optimal[len(lam)//2]*100:.4f}%")
    print(f"  MgF2 coating   (n={n_MgF2}, d={d_MgF2:.1f} nm): "
          f"R = {R_MgF2[len(lam)//2]*100:.2f}%")


# ===========================
# Main
# ===========================

if __name__ == '__main__':
    print("=== Fresnel Coefficients (External Reflection) ===")
    plot_fresnel_external()

    print("\n=== Brewster Angle ===")
    brewster_demo()

    print("\n=== Total Internal Reflection ===")
    total_internal_reflection()

    print("\n=== Anti-Reflection Coating ===")
    anti_reflection_coating()
