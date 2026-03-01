#!/usr/bin/env python3
"""Thin Lens Ray Tracing: Image Formation, Magnification, and Aberrations.

This module demonstrates the thin lens equation and geometrical ray tracing:
    1/f = 1/do + 1/di

where f is the focal length, do is the object distance, and di is the image
distance (all measured from the lens).

We explore:
1. Ray tracing through converging (positive f) and diverging (negative f) lenses
2. Image formation for various object distances (real vs virtual images)
3. Magnification and its sign (inverted vs upright)
4. Simple models of spherical and chromatic aberration

Physics background:
- A thin lens approximation treats the lens as having zero thickness.
- Three principal rays fully determine the image:
  (a) Parallel ray -> passes through focal point
  (b) Central ray  -> passes straight through the lens center
  (c) Focal ray    -> exits parallel after passing through the front focal point
- Real images form where refracted rays converge (di > 0).
- Virtual images form where refracted rays appear to diverge from (di < 0).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ---------------------------------------------------------------------------
# Thin lens equation utilities
# ---------------------------------------------------------------------------

def thin_lens_image(f: float, do: float) -> tuple[float, float]:
    """Compute image distance and magnification using the thin lens equation.

    Convention: distances are positive to the right of the lens.
    Object distance do is positive when the object is to the left.

    Args:
        f: Focal length (positive for converging, negative for diverging).
        do: Object distance (positive, measured from lens).

    Returns:
        Tuple of (image distance di, magnification m).
        di > 0 means real image (right of lens); di < 0 means virtual (left).
        m < 0 means inverted image; |m| > 1 means magnified.
    """
    # Guard against object at focal point (image at infinity)
    if abs(do - f) < 1e-10:
        return float('inf'), float('inf')

    di = 1.0 / (1.0 / f - 1.0 / do)
    m = -di / do  # Negative sign: inverted image convention
    return di, m


def lensmakers_equation(n: float, R1: float, R2: float) -> float:
    """Compute focal length from lens geometry using the lensmaker's equation.

    1/f = (n - 1) * [1/R1 - 1/R2]

    Sign convention: R > 0 if center of curvature is to the right.

    Args:
        n: Refractive index of the lens material.
        R1: Radius of curvature of the first surface.
        R2: Radius of curvature of the second surface.

    Returns:
        Focal length f.
    """
    return 1.0 / ((n - 1) * (1.0 / R1 - 1.0 / R2))


# ---------------------------------------------------------------------------
# Ray tracing visualization helpers
# ---------------------------------------------------------------------------

def draw_lens(ax, x_pos: float, height: float, f: float):
    """Draw a thin lens as a vertical line with arrowheads indicating type.

    Converging lenses get outward-pointing arrowheads (biconvex shape).
    Diverging lenses get inward-pointing arrowheads (biconcave shape).

    Args:
        ax: Matplotlib axes.
        x_pos: Horizontal position of the lens.
        height: Half-height of the lens symbol.
        f: Focal length (sign determines lens type).
    """
    ax.plot([x_pos, x_pos], [-height, height], 'b-', linewidth=2)

    arrow_size = height * 0.15
    if f > 0:  # Converging: arrows point outward
        for sign in [1, -1]:
            y = sign * height
            ax.annotate('', xy=(x_pos - arrow_size, y),
                         xytext=(x_pos, y - sign * arrow_size * 0.8),
                         arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
            ax.annotate('', xy=(x_pos + arrow_size, y),
                         xytext=(x_pos, y - sign * arrow_size * 0.8),
                         arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
    else:  # Diverging: arrows point inward
        for sign in [1, -1]:
            y = sign * height
            ax.annotate('', xy=(x_pos, y - sign * arrow_size * 0.8),
                         xytext=(x_pos - arrow_size, y),
                         arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
            ax.annotate('', xy=(x_pos, y - sign * arrow_size * 0.8),
                         xytext=(x_pos + arrow_size, y),
                         arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))


def draw_object_arrow(ax, x: float, height: float, color='green'):
    """Draw an upright arrow representing the object."""
    ax.annotate('', xy=(x, height), xytext=(x, 0),
                 arrowprops=dict(arrowstyle='->', color=color, lw=2.5))


def draw_image_arrow(ax, x: float, height: float, color='red', style='-'):
    """Draw an arrow representing the image (possibly inverted/virtual)."""
    ls = '--' if style == '--' else '-'
    ax.annotate('', xy=(x, height), xytext=(x, 0),
                 arrowprops=dict(arrowstyle='->', color=color, lw=2.5, ls=ls))


# ---------------------------------------------------------------------------
# Visualization 1: Converging lens ray tracing
# ---------------------------------------------------------------------------

def plot_converging_lens_cases():
    """Show image formation for a converging lens at different object distances.

    Three principal rays are traced for each case:
    1. Object beyond 2f (real, reduced, inverted)
    2. Object at 2f (real, same size, inverted)
    3. Object between f and 2f (real, magnified, inverted)
    4. Object inside f (virtual, magnified, upright)
    """
    f = 3.0  # focal length
    cases = [
        (8.0, "Object beyond 2f: real, reduced, inverted"),
        (6.0, "Object at 2f: real, same size, inverted"),
        (4.0, "Object between f and 2f: real, magnified, inverted"),
        (2.0, "Object inside f: virtual, magnified, upright"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for ax, (do, title) in zip(axes.flat, cases):
        di, m = thin_lens_image(f, do)
        obj_height = 1.0
        img_height = m * obj_height

        # Determine plot extents
        x_min = min(-do - 2, di - 2 if np.isfinite(di) else -do - 2, -2 * f - 1)
        x_max = max(do + 2, di + 2 if np.isfinite(di) and di > 0 else 2, 2 * f + 1)
        y_lim = max(abs(img_height) + 1 if np.isfinite(img_height) else 3, 3)

        # Optical axis
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.plot([-f], [0], 'kx', markersize=10, label='F')
        ax.plot([f], [0], 'kx', markersize=10)
        ax.text(-f, -0.3, 'F', ha='center', fontsize=9)
        ax.text(f, -0.3, "F'", ha='center', fontsize=9)

        # Draw lens at x=0
        draw_lens(ax, 0, y_lim * 0.8, f)

        # Object arrow
        draw_object_arrow(ax, -do, obj_height)
        ax.text(-do, obj_height + 0.2, 'Object', ha='center', fontsize=9, color='green')

        # Ray 1: Parallel to axis -> through back focal point
        # From object tip, go horizontal to the lens, then toward F'
        ax.plot([-do, 0], [obj_height, obj_height], 'r-', linewidth=1)
        if np.isfinite(di):
            # Extend to image or beyond
            ray1_slope = (obj_height - 0) / (0 - f)  # slope from lens to F'
            if di > 0:
                ray1_end_y = obj_height + ray1_slope * max(di, 2 * f)
                ax.plot([0, max(di, 2 * f)],
                        [obj_height, ray1_end_y], 'r-', linewidth=1)
            else:
                # Virtual image: extend forward but also trace backward (dashed)
                ray1_end_y = obj_height + ray1_slope * 5
                ax.plot([0, 5], [obj_height, ray1_end_y], 'r-', linewidth=1)
                # Virtual extension backward
                ax.plot([0, di], [obj_height, obj_height - ray1_slope * abs(di)],
                        'r--', linewidth=0.8, alpha=0.5)

        # Ray 2: Through center of lens (undeviated)
        central_slope = -obj_height / do  # slope = -h / do (downward from object to center)
        if np.isfinite(di):
            if di > 0:
                x_end = max(di, 2 * f)
            else:
                x_end = -di + 2
            ray2_end_y = central_slope * (x_end + do) + obj_height
            ax.plot([-do, x_end], [obj_height, obj_height + central_slope * (x_end + do)],
                    'orange', linewidth=1)
            if di < 0:
                ax.plot([0, di], [0, central_slope * (di + do) + obj_height],
                        'orange', linewidth=0.8, linestyle='--', alpha=0.5)

        # Ray 3: Through front focal point -> exits parallel
        # From object tip toward front focal point F(-f, 0)
        ray3_slope = (obj_height - 0) / (-do - (-f))  # slope from object to F
        ray3_y_at_lens = obj_height + ray3_slope * do  # y when x=0
        ax.plot([-do, 0], [obj_height, ray3_y_at_lens], 'm-', linewidth=1)
        # After lens, exits parallel to axis
        if np.isfinite(di):
            if di > 0:
                ax.plot([0, max(di, 2 * f)],
                        [ray3_y_at_lens, ray3_y_at_lens], 'm-', linewidth=1)
            else:
                ax.plot([0, 5],
                        [ray3_y_at_lens, ray3_y_at_lens], 'm-', linewidth=1)
                ax.plot([0, di],
                        [ray3_y_at_lens, ray3_y_at_lens], 'm--', linewidth=0.8, alpha=0.5)

        # Image arrow
        if np.isfinite(di) and np.isfinite(img_height):
            is_virtual = di < 0
            draw_image_arrow(ax, di, img_height, color='red',
                            style='--' if is_virtual else '-')
            label = 'Virtual Image' if is_virtual else 'Real Image'
            y_offset = 0.2 if img_height > 0 else -0.4
            ax.text(di, img_height + y_offset, label, ha='center', fontsize=9, color='red')

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-y_lim, y_lim)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Position (units of f)')
        ax.grid(True, alpha=0.2)

        # Print numerical results
        if np.isfinite(di):
            print(f"  do={do:.1f}f: di={di:.2f}, m={m:.2f} "
                  f"({'virtual' if di < 0 else 'real'}, "
                  f"{'upright' if m > 0 else 'inverted'})")

    fig.suptitle(f'Converging Lens (f = {f:.1f}) — Image Formation Cases', fontsize=14)
    fig.tight_layout()
    fig.savefig('02_thin_lens_converging.png', dpi=150)
    plt.close(fig)
    print("[Saved] 02_thin_lens_converging.png")


# ---------------------------------------------------------------------------
# Visualization 2: Diverging lens
# ---------------------------------------------------------------------------

def plot_diverging_lens():
    """Demonstrate a diverging lens always produces a virtual, upright, reduced image.

    For a diverging lens (f < 0), the image is always:
    - Virtual (di < 0, same side as object)
    - Upright (m > 0)
    - Reduced (|m| < 1)
    """
    f = -3.0
    do_values = [2, 4, 6, 8, 10]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.axhline(0, color='gray', linewidth=0.5)
    draw_lens(ax, 0, 3, f)
    ax.plot([f], [0], 'kx', markersize=10)
    ax.plot([-f], [0], 'kx', markersize=10)
    ax.text(f, -0.4, 'F (virtual)', ha='center', fontsize=8)
    ax.text(-f, -0.4, "F'", ha='center', fontsize=8)

    cmap = plt.cm.viridis
    for i, do in enumerate(do_values):
        color = cmap(i / len(do_values))
        di, m = thin_lens_image(f, do)
        obj_h = 1.0
        img_h = m * obj_h

        # Object
        ax.annotate('', xy=(-do, obj_h), xytext=(-do, 0),
                     arrowprops=dict(arrowstyle='->', color=color, lw=2))

        # Image (always virtual for diverging lens)
        ax.annotate('', xy=(di, img_h), xytext=(di, 0),
                     arrowprops=dict(arrowstyle='->', color=color, lw=2, ls='--'))

        ax.text(-do, obj_h + 0.15, f'do={do}', ha='center', fontsize=8, color=color)
        ax.text(di, img_h - 0.3, f'di={di:.1f}\nm={m:.2f}', ha='center',
                fontsize=7, color=color)

    ax.set_xlim(-14, 4)
    ax.set_ylim(-2, 3)
    ax.set_aspect('equal')
    ax.set_title(f'Diverging Lens (f = {f:.1f}): Always Virtual, Upright, Reduced')
    ax.set_xlabel('Position')
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig('02_thin_lens_diverging.png', dpi=150)
    plt.close(fig)
    print("[Saved] 02_thin_lens_diverging.png")


# ---------------------------------------------------------------------------
# Visualization 3: Magnification curve
# ---------------------------------------------------------------------------

def plot_magnification_curve():
    """Plot image distance and magnification as functions of object distance.

    This reveals the singularity at do = f (image at infinity) and the
    transition between real and virtual images.
    """
    f = 5.0
    do_range = np.linspace(0.5, 30, 1000)

    di_values = []
    m_values = []
    for do in do_range:
        di, m = thin_lens_image(f, do)
        di_values.append(di if np.isfinite(di) else np.nan)
        m_values.append(m if np.isfinite(m) else np.nan)

    di_values = np.array(di_values)
    m_values = np.array(m_values)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Image distance vs object distance
    ax1.plot(do_range, di_values, 'b-', linewidth=2)
    ax1.axhline(0, color='gray', linewidth=0.5)
    ax1.axvline(f, color='red', linewidth=1, linestyle='--', label=f'do = f = {f}')
    ax1.axhline(f, color='green', linewidth=1, linestyle=':', label=f'di = f')
    ax1.set_ylabel('Image distance di')
    ax1.set_ylim(-30, 30)
    ax1.set_title(f'Thin Lens (f = {f}): Image Distance and Magnification')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Annotate regions
    ax1.fill_betweenx([-30, 30], 0, f, alpha=0.05, color='red',
                       label='Virtual image region')
    ax1.text(f / 2, -20, 'Virtual\nimage\n(di < 0)', ha='center', fontsize=10, color='red')
    ax1.text(f * 2, 20, 'Real image\n(di > 0)', ha='center', fontsize=10, color='blue')

    # Magnification vs object distance
    ax2.plot(do_range, m_values, 'r-', linewidth=2)
    ax2.axhline(0, color='gray', linewidth=0.5)
    ax2.axhline(-1, color='green', linewidth=1, linestyle=':', label='m = -1 (same size)')
    ax2.axhline(1, color='purple', linewidth=1, linestyle=':', label='m = +1')
    ax2.axvline(f, color='red', linewidth=1, linestyle='--', label=f'do = f')
    ax2.set_xlabel('Object distance do')
    ax2.set_ylabel('Magnification m')
    ax2.set_ylim(-5, 5)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig('02_thin_lens_magnification.png', dpi=150)
    plt.close(fig)
    print("[Saved] 02_thin_lens_magnification.png")


# ---------------------------------------------------------------------------
# Visualization 4: Simple aberration model
# ---------------------------------------------------------------------------

def plot_spherical_aberration():
    """Demonstrate spherical aberration in a simple lens model.

    Spherical aberration occurs because rays far from the optical axis are
    focused at a slightly different point than paraxial rays. We model this
    by adding a height-dependent correction to the focal length:
        f_eff(h) = f * (1 - k * h^2)
    where h is the ray height at the lens and k is the aberration coefficient.

    This is a simplified model -- real aberrations depend on lens shape (Coddington
    equations), but the qualitative behavior is correct.
    """
    f = 10.0  # paraxial focal length
    k = 0.005  # aberration coefficient (spherical)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left panel: ray diagram with aberrated focus ---
    n_rays = 20
    heights = np.linspace(-3, 3, n_rays)

    # Draw lens
    ax1.plot([0, 0], [-3.5, 3.5], 'b-', linewidth=2)
    ax1.axhline(0, color='gray', linewidth=0.5)

    for h in heights:
        # Aberrated focal length: rays far from axis focus closer
        f_eff = f * (1 - k * h**2)

        # Incoming ray: horizontal at height h
        ax1.plot([-5, 0], [h, h], 'g-', linewidth=0.5, alpha=0.5)

        # After lens: converges toward aberrated focal point
        ax1.plot([0, f_eff], [h, 0], 'r-', linewidth=0.5, alpha=0.5)
        # Extend past focus
        overshoot = 2
        slope = -h / f_eff
        ax1.plot([f_eff, f_eff + overshoot],
                 [0, slope * overshoot], 'r-', linewidth=0.3, alpha=0.3)

    # Mark paraxial and marginal focus
    f_marginal = f * (1 - k * 3.0**2)
    ax1.plot(f, 0, 'ko', markersize=6)
    ax1.text(f, 0.3, 'Paraxial\nfocus', ha='center', fontsize=8)
    ax1.plot(f_marginal, 0, 'ro', markersize=6)
    ax1.text(f_marginal, -0.5, 'Marginal\nfocus', ha='center', fontsize=8)

    # Longitudinal spherical aberration (LSA)
    lsa = f - f_marginal
    ax1.annotate('', xy=(f_marginal, -0.2), xytext=(f, -0.2),
                  arrowprops=dict(arrowstyle='<->', color='purple', lw=1.5))
    ax1.text((f + f_marginal) / 2, -0.6, f'LSA = {lsa:.2f}',
             ha='center', fontsize=9, color='purple')

    ax1.set_xlim(-6, 14)
    ax1.set_ylim(-4, 4)
    ax1.set_aspect('equal')
    ax1.set_title('Spherical Aberration: Marginal Rays Focus Closer')
    ax1.set_xlabel('z (optical axis)')
    ax1.set_ylabel('y (ray height)')

    # --- Right panel: spot diagram at paraxial focal plane ---
    n_spot_rays = 500
    # Random rays uniformly filling the lens aperture
    rng = np.random.default_rng(42)
    ray_h = rng.uniform(-3, 3, n_spot_rays)
    ray_phi = rng.uniform(0, 2 * np.pi, n_spot_rays)

    # Convert to 2D heights (hy, hz) at the lens
    hy = ray_h * np.cos(ray_phi)
    hz = ray_h * np.sin(ray_phi)
    r = np.sqrt(hy**2 + hz**2)

    # At the paraxial focal plane (x = f), aberrated rays miss the axis
    # Transverse aberration: y_err = h * (1 - f/f_eff) approximately = h * k * h^2
    # More precisely: at x = f, y = h * (1 - f / f_eff(r))
    f_eff = f * (1 - k * r**2)
    y_err = hy * (1 - f / f_eff)
    z_err = hz * (1 - f / f_eff)

    ax2.scatter(y_err, z_err, s=1, c='red', alpha=0.5)
    ax2.set_aspect('equal')
    ax2.set_title('Spot Diagram at Paraxial Focal Plane')
    ax2.set_xlabel('y error')
    ax2.set_ylabel('z error')
    ax2.grid(True, alpha=0.3)

    # RMS spot size
    rms = np.sqrt(np.mean(y_err**2 + z_err**2))
    ax2.text(0.05, 0.95, f'RMS spot radius = {rms:.4f}',
             transform=ax2.transAxes, fontsize=10, va='top',
             bbox=dict(boxstyle='round', facecolor='wheat'))

    fig.tight_layout()
    fig.savefig('02_thin_lens_aberration.png', dpi=150)
    plt.close(fig)
    print(f"[Saved] 02_thin_lens_aberration.png")
    print(f"  Longitudinal spherical aberration: {lsa:.4f}")
    print(f"  RMS spot radius: {rms:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Thin Lens Ray Tracing: Image Formation and Aberrations")
    print("=" * 60)

    # Lensmaker's equation demo
    print("\n--- Lensmaker's Equation ---")
    n_glass = 1.52
    R1, R2 = 20.0, -20.0  # Symmetric biconvex
    f_calc = lensmakers_equation(n_glass, R1, R2)
    print(f"  Biconvex lens: n={n_glass}, R1={R1}, R2={R2}")
    print(f"  Focal length f = {f_calc:.2f}")

    # Converging lens cases
    print("\n--- Converging Lens (f=3.0) ---")
    plot_converging_lens_cases()

    # Diverging lens
    print("\n--- Diverging Lens (f=-3.0) ---")
    f_div = -3.0
    for do in [2, 4, 6, 8, 10]:
        di, m = thin_lens_image(f_div, do)
        print(f"  do={do}: di={di:.2f}, m={m:.2f}")
    plot_diverging_lens()

    # Magnification curve
    print("\n--- Magnification Curve ---")
    plot_magnification_curve()

    # Spherical aberration
    print("\n--- Spherical Aberration ---")
    plot_spherical_aberration()

    print("\nDone. All plots saved.")
