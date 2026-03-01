#!/usr/bin/env python3
"""Snell's Law Visualization: Refraction, Total Internal Reflection, and Prism Dispersion.

This module demonstrates the fundamental law of refraction (Snell's law):
    n1 * sin(theta1) = n2 * sin(theta2)

We explore:
1. Ray tracing through a flat interface at various incidence angles
2. Total internal reflection (TIR) and the critical angle
3. Prism refraction with wavelength-dependent refractive index (dispersion)

Physics background:
- Snell's law governs how light bends when crossing a boundary between
  two media with different refractive indices.
- When light travels from a denser medium to a less dense medium (n1 > n2),
  there exists a critical angle beyond which all light is reflected (TIR).
- Dispersion arises because the refractive index depends on wavelength,
  described empirically by the Cauchy or Sellmeier equations. This causes
  white light to separate into its constituent colors in a prism.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection


# ---------------------------------------------------------------------------
# Refractive index models
# ---------------------------------------------------------------------------

def cauchy_index(wavelength_um: float, A: float = 1.5046, B: float = 0.00420) -> float:
    """Compute refractive index using the Cauchy equation.

    The Cauchy equation is an empirical relation:
        n(lambda) = A + B / lambda^2

    This is a good approximation for visible wavelengths in many glasses.
    The default coefficients approximate BK7 borosilicate glass.

    Args:
        wavelength_um: Wavelength in micrometers.
        A: First Cauchy coefficient (dimensionless).
        B: Second Cauchy coefficient (um^2).

    Returns:
        Refractive index at the given wavelength.
    """
    return A + B / wavelength_um**2


def sellmeier_index(wavelength_um: float) -> float:
    """Compute refractive index using the Sellmeier equation for BK7 glass.

    The Sellmeier equation is more accurate than Cauchy, especially in the UV/IR:
        n^2 - 1 = sum_i [ B_i * lambda^2 / (lambda^2 - C_i) ]

    The coefficients below are for Schott BK7, one of the most common optical glasses.

    Args:
        wavelength_um: Wavelength in micrometers.

    Returns:
        Refractive index at the given wavelength.
    """
    # Sellmeier coefficients for BK7 (Schott catalog)
    B = [1.03961212, 0.231792344, 1.01046945]
    C = [0.00600069867, 0.0200179144, 103.560653]
    lam2 = wavelength_um**2
    n_sq = 1.0
    for b, c in zip(B, C):
        n_sq += b * lam2 / (lam2 - c)
    return np.sqrt(n_sq)


# ---------------------------------------------------------------------------
# Snell's law utilities
# ---------------------------------------------------------------------------

def snell_refraction(theta_i: float, n1: float, n2: float) -> float | None:
    """Apply Snell's law to compute the refraction angle.

    Args:
        theta_i: Incidence angle in radians (measured from surface normal).
        n1: Refractive index of the incident medium.
        n2: Refractive index of the transmitted medium.

    Returns:
        Refraction angle in radians, or None if total internal reflection occurs.
    """
    sin_theta_t = n1 * np.sin(theta_i) / n2
    # TIR occurs when sin(theta_t) > 1 -- no real solution exists
    if abs(sin_theta_t) > 1.0:
        return None
    return np.arcsin(sin_theta_t)


def critical_angle(n1: float, n2: float) -> float | None:
    """Compute the critical angle for total internal reflection.

    TIR is only possible when light goes from a denser to a less dense medium.

    Args:
        n1: Refractive index of the incident (denser) medium.
        n2: Refractive index of the transmitted (less dense) medium.

    Returns:
        Critical angle in radians, or None if n1 <= n2.
    """
    if n1 <= n2:
        return None
    return np.arcsin(n2 / n1)


def reflection_angle(theta_i: float) -> float:
    """Compute the reflection angle (trivially equal to incidence angle).

    By the law of reflection, theta_r = theta_i. We include this function
    for clarity in the ray-tracing code.
    """
    return theta_i


# ---------------------------------------------------------------------------
# Visualization 1: Refraction at a flat interface
# ---------------------------------------------------------------------------

def plot_refraction_at_interface():
    """Visualize Snell's law at a flat air-glass interface for multiple angles.

    We draw incident, refracted, and reflected rays at a horizontal boundary.
    The upper half is air (n=1.0), the lower half is glass (n=1.52).
    """
    n_air = 1.0
    n_glass = 1.52
    angles_deg = [10, 20, 30, 40, 50, 60, 70, 80]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw the interface and media labels
    ax.axhline(0, color='gray', linewidth=2, zorder=1)
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--', label='Normal')
    ax.fill_between([-3, 3], 0, 3, color='lightyellow', alpha=0.3)
    ax.fill_between([-3, 3], -3, 0, color='lightcyan', alpha=0.3)
    ax.text(-2.8, 2.5, f'Air (n={n_air:.2f})', fontsize=12)
    ax.text(-2.8, -2.8, f'Glass (n={n_glass:.2f})', fontsize=12)

    # Use a colormap to distinguish different angles
    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=0, vmax=90)

    ray_length = 2.5

    for theta_deg in angles_deg:
        theta_i = np.radians(theta_deg)
        color = cmap(norm(theta_deg))

        # Incident ray: comes from upper-left toward the origin
        # Direction vector points downward and to the right
        x_inc = -ray_length * np.sin(theta_i)
        y_inc = ray_length * np.cos(theta_i)
        ax.annotate('', xy=(0, 0), xytext=(x_inc, y_inc),
                     arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

        # Refracted ray
        theta_t = snell_refraction(theta_i, n_air, n_glass)
        if theta_t is not None:
            x_ref = ray_length * np.sin(theta_t)
            y_ref = -ray_length * np.cos(theta_t)
            ax.annotate('', xy=(x_ref, y_ref), xytext=(0, 0),
                         arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

        # Label the angle
        ax.text(x_inc * 0.45, y_inc * 0.45 + 0.15,
                f'{theta_deg}', fontsize=8, color=color, ha='center')

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_xlabel('x (arbitrary units)')
    ax.set_ylabel('y (arbitrary units)')
    ax.set_title("Snell's Law: Refraction at Air-Glass Interface")
    fig.tight_layout()
    fig.savefig('01_snell_refraction_interface.png', dpi=150)
    plt.close(fig)
    print("[Saved] 01_snell_refraction_interface.png")


# ---------------------------------------------------------------------------
# Visualization 2: Total internal reflection
# ---------------------------------------------------------------------------

def plot_total_internal_reflection():
    """Demonstrate total internal reflection (TIR) at a glass-air interface.

    Light travels from glass (n=1.52) into air (n=1.0). Beyond the critical
    angle, no refracted ray exists and all energy is reflected.
    """
    n_glass = 1.52
    n_air = 1.0
    theta_c = critical_angle(n_glass, n_air)
    theta_c_deg = np.degrees(theta_c)
    print(f"\nCritical angle (glass->air): {theta_c_deg:.2f} deg")

    # Sweep angles from 0 to 90 and compute refracted angle (or TIR)
    angles_deg = np.arange(5, 85, 5)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: ray diagram showing transition to TIR
    ax1.axhline(0, color='gray', linewidth=2)
    ax1.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    ax1.fill_between([-3, 3], 0, 3, color='lightcyan', alpha=0.3)
    ax1.fill_between([-3, 3], -3, 0, color='lightyellow', alpha=0.3)
    ax1.text(-2.8, 2.5, f'Glass (n={n_glass:.2f})', fontsize=11)
    ax1.text(-2.8, -2.8, f'Air (n={n_air:.2f})', fontsize=11)

    ray_length = 2.5
    selected_angles = [10, 20, 30, 40, 41, 42, 50, 60, 70]

    for theta_deg in selected_angles:
        theta_i = np.radians(theta_deg)
        is_tir = theta_deg > theta_c_deg
        color = 'red' if is_tir else 'blue'

        # Incident ray from below (glass side)
        x_inc = -ray_length * np.sin(theta_i)
        y_inc = ray_length * np.cos(theta_i)
        ax1.annotate('', xy=(0, 0), xytext=(x_inc, -y_inc),
                      arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

        theta_t = snell_refraction(theta_i, n_glass, n_air)
        if theta_t is not None:
            # Refracted ray exits into air (upward)
            x_ref = ray_length * np.sin(theta_t)
            y_ref = ray_length * np.cos(theta_t)
            ax1.annotate('', xy=(x_ref, y_ref), xytext=(0, 0),
                          arrowprops=dict(arrowstyle='->', color=color, lw=1.2, ls='--'))
        else:
            # TIR: reflected ray stays in glass (downward)
            theta_r = reflection_angle(theta_i)
            x_refl = ray_length * np.sin(theta_r)
            y_refl = -ray_length * np.cos(theta_r)
            ax1.annotate('', xy=(x_refl, y_refl), xytext=(0, 0),
                          arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    # Mark the critical angle
    arc_theta = np.linspace(np.pi / 2, np.pi / 2 + theta_c, 50)
    arc_r = 1.0
    ax1.plot(arc_r * np.cos(arc_theta), -arc_r * np.sin(arc_theta),
             'g-', linewidth=2, label=f'Critical angle = {theta_c_deg:.1f} deg')
    ax1.legend(loc='lower right', fontsize=10)

    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    ax1.set_aspect('equal')
    ax1.set_title('Total Internal Reflection (Glass to Air)')

    # Right panel: refracted angle vs incidence angle
    theta_i_range = np.linspace(0, np.pi / 2, 500)
    theta_t_list = []
    reflectance = []
    for th in theta_i_range:
        sin_t = n_glass * np.sin(th) / n_air
        if abs(sin_t) <= 1.0:
            theta_t_list.append(np.degrees(np.arcsin(sin_t)))
            # Fresnel reflectance (s-polarization) for reference
            cos_t = np.sqrt(1 - sin_t**2)
            rs = ((n_glass * np.cos(th) - n_air * cos_t) /
                  (n_glass * np.cos(th) + n_air * cos_t))**2
            reflectance.append(rs)
        else:
            theta_t_list.append(np.nan)
            reflectance.append(1.0)  # TIR: 100% reflection

    ax2.plot(np.degrees(theta_i_range), theta_t_list, 'b-', linewidth=2,
             label='Refraction angle')
    ax2.axvline(theta_c_deg, color='green', linestyle='--', linewidth=1.5,
                label=f'Critical angle = {theta_c_deg:.1f} deg')
    ax2.set_xlabel('Incidence angle (deg)')
    ax2.set_ylabel('Refraction angle (deg)', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_xlim(0, 90)
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper left')
    ax2.set_title('Refraction Angle and Reflectance vs Incidence Angle')

    # Overlay reflectance on twin axis
    ax2b = ax2.twinx()
    ax2b.plot(np.degrees(theta_i_range), reflectance, 'r-', linewidth=1.5,
              label='Reflectance (s-pol)')
    ax2b.set_ylabel('Reflectance', color='red')
    ax2b.tick_params(axis='y', labelcolor='red')
    ax2b.set_ylim(0, 1.1)
    ax2b.legend(loc='center right')

    fig.tight_layout()
    fig.savefig('01_snell_tir.png', dpi=150)
    plt.close(fig)
    print("[Saved] 01_snell_tir.png")


# ---------------------------------------------------------------------------
# Visualization 3: Prism dispersion
# ---------------------------------------------------------------------------

def trace_prism_ray(theta_i: float, n: float, apex_angle: float) -> dict | None:
    """Trace a ray through a triangular prism using Snell's law.

    The prism has an apex angle 'alpha' at the top. Light enters the left face
    and exits the right face. We compute the deviation angle.

    Args:
        theta_i: Incidence angle on the first face (radians).
        n: Refractive index of the prism material.
        apex_angle: Apex angle of the prism (radians).

    Returns:
        Dictionary with intermediate angles and total deviation, or None if TIR.
    """
    # Refraction at first surface (air -> prism)
    theta_r1 = snell_refraction(theta_i, 1.0, n)
    if theta_r1 is None:
        return None

    # Angle of incidence on the second surface
    # Geometry: theta_i2 = apex_angle - theta_r1
    theta_i2 = apex_angle - theta_r1
    if theta_i2 < 0:
        return None  # Ray doesn't hit the second face

    # Refraction at second surface (prism -> air)
    theta_r2 = snell_refraction(theta_i2, n, 1.0)
    if theta_r2 is None:
        return None  # TIR at second surface

    # Total deviation: delta = (theta_i - theta_r1) + (theta_r2 - theta_i2)
    # Simplified: delta = theta_i + theta_r2 - apex_angle
    deviation = theta_i + theta_r2 - apex_angle

    return {
        'theta_i': theta_i,
        'theta_r1': theta_r1,
        'theta_i2': theta_i2,
        'theta_r2': theta_r2,
        'deviation': deviation,
    }


def plot_prism_dispersion():
    """Visualize dispersion of white light through a glass prism.

    White light (multiple wavelengths) enters a prism. Because the refractive
    index varies with wavelength (dispersion), each color is deviated by a
    different amount, producing the familiar rainbow spectrum.
    """
    apex_deg = 60.0
    apex_rad = np.radians(apex_deg)
    theta_i = np.radians(45.0)

    # Visible spectrum: wavelengths from 380 nm (violet) to 780 nm (red)
    wavelengths_nm = np.linspace(380, 780, 50)
    wavelengths_um = wavelengths_nm / 1000.0

    # Map wavelength to approximate RGB color
    def wavelength_to_rgb(wl_nm):
        """Convert wavelength (nm) to approximate RGB color.

        Uses a piecewise linear approximation across the visible spectrum.
        """
        if 380 <= wl_nm < 440:
            r, g, b = -(wl_nm - 440) / 60, 0, 1
        elif 440 <= wl_nm < 490:
            r, g, b = 0, (wl_nm - 440) / 50, 1
        elif 490 <= wl_nm < 510:
            r, g, b = 0, 1, -(wl_nm - 510) / 20
        elif 510 <= wl_nm < 580:
            r, g, b = (wl_nm - 510) / 70, 1, 0
        elif 580 <= wl_nm < 645:
            r, g, b = 1, -(wl_nm - 645) / 65, 0
        elif 645 <= wl_nm <= 780:
            r, g, b = 1, 0, 0
        else:
            r, g, b = 0, 0, 0
        # Intensity adjustment at spectrum edges
        if 380 <= wl_nm < 420:
            factor = 0.3 + 0.7 * (wl_nm - 380) / 40
        elif 700 < wl_nm <= 780:
            factor = 0.3 + 0.7 * (780 - wl_nm) / 80
        else:
            factor = 1.0
        return (r * factor, g * factor, b * factor)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left panel: ray through prism ---
    # Draw the prism as a triangle
    h = 3.0  # half-size
    # Equilateral prism vertices (apex at top)
    prism_verts = np.array([
        [-h * np.tan(apex_rad / 2), 0],  # bottom-left
        [h * np.tan(apex_rad / 2), 0],   # bottom-right
        [0, h],                            # apex
    ])
    prism = patches.Polygon(prism_verts, closed=True, fill=True,
                             facecolor='lightblue', edgecolor='black',
                             alpha=0.3, linewidth=2)
    ax1.add_patch(prism)

    # Trace rays for several wavelengths
    for wl_nm, wl_um in zip(wavelengths_nm, wavelengths_um):
        n = sellmeier_index(wl_um)
        result = trace_prism_ray(theta_i, n, apex_rad)
        if result is None:
            continue

        color = wavelength_to_rgb(wl_nm)
        dev = result['deviation']

        # Simplified visualization: incoming ray from left, exiting ray to right
        # Entry point on left face of prism
        entry_y = 1.5  # midpoint of left face
        entry_x = -entry_y * np.tan(apex_rad / 2)

        # Incoming ray
        ray_len = 2.0
        ax1.plot([entry_x - ray_len * np.cos(theta_i), entry_x],
                 [entry_y + ray_len * np.sin(theta_i) * 0.3, entry_y],
                 color=color, linewidth=0.8, alpha=0.8)

        # Outgoing ray (direction depends on deviation)
        # The exit angle relative to horizontal varies with deviation
        exit_y = 1.5
        exit_x = exit_y * np.tan(apex_rad / 2)
        out_angle = dev - 0.1  # simplified geometric projection
        ax1.plot([exit_x, exit_x + ray_len * np.cos(out_angle)],
                 [exit_y, exit_y - ray_len * np.sin(out_angle)],
                 color=color, linewidth=0.8, alpha=0.8)

    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-1, 5)
    ax1.set_aspect('equal')
    ax1.set_title(f'Prism Dispersion (apex={apex_deg} deg, incidence={np.degrees(theta_i):.0f} deg)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    # --- Right panel: deviation angle vs wavelength ---
    deviations = []
    indices = []
    for wl_um in wavelengths_um:
        n = sellmeier_index(wl_um)
        indices.append(n)
        result = trace_prism_ray(theta_i, n, apex_rad)
        if result:
            deviations.append(np.degrees(result['deviation']))
        else:
            deviations.append(np.nan)

    colors = [wavelength_to_rgb(wl) for wl in wavelengths_nm]

    # Plot deviation as colored scatter
    for wl, dev, c in zip(wavelengths_nm, deviations, colors):
        ax2.scatter(wl, dev, color=c, s=30, edgecolors='none')

    ax2.plot(wavelengths_nm, deviations, 'k-', linewidth=0.5, alpha=0.3)
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Deviation angle (deg)')
    ax2.set_title('Dispersion: Deviation vs Wavelength')

    # Overlay refractive index on twin axis
    ax2b = ax2.twinx()
    ax2b.plot(wavelengths_nm, indices, 'k--', linewidth=1, label='n(lambda)')
    ax2b.set_ylabel('Refractive index n')
    ax2b.legend(loc='upper right')

    # Print some key values
    n_red = sellmeier_index(0.656)
    n_blue = sellmeier_index(0.486)
    n_yellow = sellmeier_index(0.589)
    abbe_number = (n_yellow - 1) / (n_blue - n_red)
    print(f"\nBK7 Sellmeier refractive indices:")
    print(f"  n(656.3 nm, red)   = {n_red:.6f}")
    print(f"  n(589.3 nm, yellow)= {n_yellow:.6f}")
    print(f"  n(486.1 nm, blue)  = {n_blue:.6f}")
    print(f"  Abbe number V_d    = {abbe_number:.2f}")

    fig.tight_layout()
    fig.savefig('01_snell_prism_dispersion.png', dpi=150)
    plt.close(fig)
    print("[Saved] 01_snell_prism_dispersion.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Snell's Law: Refraction, TIR, and Prism Dispersion")
    print("=" * 60)

    # Demo 1: Refraction at flat interface
    print("\n--- Demo 1: Refraction at air-glass interface ---")
    n1, n2 = 1.0, 1.52
    for angle_deg in [0, 15, 30, 45, 60, 75]:
        theta_t = snell_refraction(np.radians(angle_deg), n1, n2)
        if theta_t is not None:
            print(f"  theta_i={angle_deg:3d} deg -> theta_t={np.degrees(theta_t):6.2f} deg")
    plot_refraction_at_interface()

    # Demo 2: Total internal reflection
    print("\n--- Demo 2: Total Internal Reflection ---")
    plot_total_internal_reflection()

    # Demo 3: Prism dispersion
    print("\n--- Demo 3: Prism Dispersion ---")
    plot_prism_dispersion()

    print("\nDone. All plots saved.")
