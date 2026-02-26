#!/usr/bin/env python3
"""Sequential Ray Tracing: Multi-Element Lens Systems and Aberration Analysis.

This module implements a sequential ray tracer for rotationally symmetric
optical systems composed of spherical surfaces:

1. Ray-surface intersection and refraction at spherical interfaces
2. Tracing rays through multi-element lens systems
3. Computing system focal length, principal planes, and f-number
4. Spot diagram and ray fan plots for aberration analysis
5. Simple Seidel aberration classification

Physics background:
- Sequential ray tracing follows rays through surfaces in order (S1, S2, ..., SN).
  At each surface, we compute the intersection point and apply Snell's law.
- For a spherical surface of radius R centered on the optical axis, the
  surface equation is: x^2 + y^2 + (z - z_center)^2 = R^2
- The focal length, principal planes, and cardinal points fully characterize
  the imaging properties of a thick lens or multi-element system.
- Aberrations cause the image to deviate from the paraxial (small-angle)
  prediction. The five Seidel aberrations are: spherical, coma, astigmatism,
  field curvature, and distortion.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Surface:
    """An optical surface (spherical or flat).

    Convention:
    - The optical axis is the z-axis.
    - Radius R > 0 means the center of curvature is to the right.
    - R = float('inf') represents a flat surface (e.g., aperture stop).

    Attributes:
        radius: Radius of curvature (meters). inf for flat.
        z_position: Axial position of the surface vertex.
        aperture: Semi-diameter (clear aperture radius).
        n_after: Refractive index of the medium after this surface.
        label: Optional descriptive label.
    """
    radius: float
    z_position: float
    aperture: float
    n_after: float
    label: str = ""


@dataclass
class Ray:
    """A geometrical ray defined by position and direction.

    The ray is parameterized as: P(t) = origin + t * direction
    We store the ray in (y, z) space for 2D meridional tracing,
    where y is the height and z is the axial coordinate.

    Attributes:
        y: Height above optical axis.
        z: Axial position.
        u: Slope angle (tan(u) = dy/dz for paraxial, sin(u) for exact).
        wavelength: Wavelength (for chromatic analysis).
    """
    y: float
    z: float
    u: float  # ray slope (angle with optical axis)
    wavelength: float = 550e-9


@dataclass
class RayTrace:
    """Record of a ray's path through the system.

    Stores the height, position, and slope at each surface for plotting
    and aberration analysis.
    """
    y_history: list = field(default_factory=list)
    z_history: list = field(default_factory=list)
    u_history: list = field(default_factory=list)
    valid: bool = True  # False if ray is vignetted or undergoes TIR


# ---------------------------------------------------------------------------
# Ray tracing engine
# ---------------------------------------------------------------------------

def trace_ray_through_surface(ray: Ray, surface: Surface, n_before: float) -> Ray | None:
    """Trace a ray through a single spherical surface.

    Uses the exact (non-paraxial) ray-surface intersection and Snell's law.
    This handles real rays accurately, unlike the paraxial approximation.

    The algorithm:
    1. Find where the ray intersects the spherical surface
    2. Compute the surface normal at the intersection point
    3. Apply Snell's law to find the refracted ray direction

    Args:
        ray: Input ray.
        surface: The optical surface.
        n_before: Refractive index before the surface.

    Returns:
        New refracted ray, or None if the ray is vignetted or TIR occurs.
    """
    n_after = surface.n_after
    R = surface.radius
    z_s = surface.z_position

    if abs(R) == float('inf'):
        # Flat surface: intersection is simply at z = z_s
        dz = z_s - ray.z
        if abs(np.cos(ray.u)) < 1e-15:
            return None  # Ray parallel to surface
        y_int = ray.y + dz * np.tan(ray.u)
        z_int = z_s

        # Check aperture
        if abs(y_int) > surface.aperture:
            return None  # Vignetted

        # Normal is along z-axis for flat surface
        # Snell's law: n_before * sin(u_in) = n_after * sin(u_out)
        sin_u_out = n_before * np.sin(ray.u) / n_after
        if abs(sin_u_out) > 1:
            return None  # TIR
        u_out = np.arcsin(sin_u_out)

        return Ray(y_int, z_int, u_out, ray.wavelength)

    # Spherical surface: center of curvature at (0, z_s + R)
    z_c = z_s + R

    # Ray direction: (sin(u), cos(u)) in (y, z) space
    dy = np.sin(ray.u)
    dz = np.cos(ray.u)

    # Solve for intersection with sphere: |P(t) - C|^2 = R^2
    # P(t) = (ray.y + t*dy, ray.z + t*dz)
    # C = (0, z_c)
    oy = ray.y
    oz = ray.z - z_c

    a = dy**2 + dz**2  # = 1 (direction is unit vector)
    b = 2 * (oy * dy + oz * dz)
    c = oy**2 + oz**2 - R**2

    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return None  # Ray misses the surface

    sqrt_disc = np.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)

    # Choose the correct intersection (closest to the surface vertex)
    # For R > 0, the vertex is at (0, z_s), and we want the intersection
    # on the side facing the incoming ray
    if R > 0:
        t = t1 if t1 > 1e-10 else t2
    else:
        t = t2 if t2 > 1e-10 else t1

    if t < 1e-10:
        return None  # No valid intersection

    # Intersection point
    y_int = ray.y + t * dy
    z_int = ray.z + t * dz

    # Check aperture
    if abs(y_int) > surface.aperture:
        return None

    # Surface normal at intersection (pointing toward center of curvature)
    ny = -y_int / R
    nz = -(z_int - z_c) / R

    # Ensure normal points in the direction of the incoming ray
    dot_in = dy * ny + dz * nz
    if dot_in > 0:
        ny, nz = -ny, -nz
        dot_in = -dot_in

    # Snell's law in vector form:
    # n1 * (d x n) = n2 * (d' x n)
    # d' = (n1/n2)*d + (n1/n2*cos_i - cos_t)*n
    cos_i = -dot_in
    sin_i_sq = 1 - cos_i**2
    sin_t_sq = (n_before / n_after)**2 * sin_i_sq

    if sin_t_sq > 1:
        return None  # TIR

    cos_t = np.sqrt(1 - sin_t_sq)
    ratio = n_before / n_after

    # Refracted direction
    dy_out = ratio * dy + (ratio * cos_i - cos_t) * ny
    dz_out = ratio * dz + (ratio * cos_i - cos_t) * nz

    # Normalize (should already be close to unit)
    norm = np.sqrt(dy_out**2 + dz_out**2)
    dy_out /= norm
    dz_out /= norm

    u_out = np.arctan2(dy_out, dz_out)

    return Ray(y_int, z_int, u_out, ray.wavelength)


def trace_ray_through_system(ray: Ray, surfaces: list[Surface],
                              n_initial: float = 1.0) -> RayTrace:
    """Trace a ray sequentially through all surfaces in the system.

    Args:
        ray: Input ray.
        surfaces: Ordered list of optical surfaces.
        n_initial: Refractive index of the medium before the first surface.

    Returns:
        RayTrace record with the ray's path through the system.
    """
    trace = RayTrace()
    trace.y_history.append(ray.y)
    trace.z_history.append(ray.z)
    trace.u_history.append(ray.u)

    current_ray = ray
    n_before = n_initial

    for surface in surfaces:
        result = trace_ray_through_surface(current_ray, surface, n_before)
        if result is None:
            trace.valid = False
            break

        trace.y_history.append(result.y)
        trace.z_history.append(result.z)
        trace.u_history.append(result.u)

        current_ray = result
        n_before = surface.n_after

    return trace


# ---------------------------------------------------------------------------
# Lens system definitions
# ---------------------------------------------------------------------------

def cooke_triplet() -> tuple[list[Surface], float]:
    """Define a Cooke triplet (three-element anastigmat lens).

    The Cooke triplet is one of the simplest lens designs that can correct
    all five Seidel aberrations simultaneously. It consists of:
    - Positive crown glass element (front)
    - Negative flint glass element (middle, also the aperture stop)
    - Positive crown glass element (rear)

    Returns:
        Tuple of (surfaces list, initial refractive index).
    """
    # Simplified Cooke triplet (scaled to ~100mm focal length)
    # Units: millimeters
    n_crown = 1.517  # BK7
    n_flint = 1.620  # SF2

    surfaces = [
        Surface(radius=28.0, z_position=0.0, aperture=12.0, n_after=n_crown, label='S1'),
        Surface(radius=-120.0, z_position=4.0, aperture=12.0, n_after=1.0, label='S2'),
        Surface(radius=-25.0, z_position=6.0, aperture=8.0, n_after=n_flint, label='S3 (stop)'),
        Surface(radius=28.0, z_position=7.5, aperture=8.0, n_after=1.0, label='S4'),
        Surface(radius=200.0, z_position=11.0, aperture=12.0, n_after=n_crown, label='S5'),
        Surface(radius=-40.0, z_position=14.0, aperture=12.0, n_after=1.0, label='S6'),
    ]
    return surfaces, 1.0


def simple_doublet() -> tuple[list[Surface], float]:
    """Define a cemented achromatic doublet.

    An achromatic doublet combines a positive crown glass element with
    a negative flint glass element to correct chromatic aberration.
    This is the workhorse lens element in most optical instruments.

    Returns:
        Tuple of (surfaces list, initial refractive index).
    """
    n_crown = 1.517  # BK7
    n_flint = 1.620  # SF2

    surfaces = [
        Surface(radius=60.0, z_position=0.0, aperture=15.0, n_after=n_crown, label='S1 (crown)'),
        Surface(radius=-45.0, z_position=5.0, aperture=15.0, n_after=n_flint, label='S2 (cemented)'),
        Surface(radius=-200.0, z_position=7.0, aperture=15.0, n_after=1.0, label='S3 (flint)'),
    ]
    return surfaces, 1.0


# ---------------------------------------------------------------------------
# System analysis
# ---------------------------------------------------------------------------

def find_focal_length(surfaces: list[Surface], n_initial: float = 1.0) -> float:
    """Compute the effective focal length using a paraxial marginal ray.

    A paraxial ray entering parallel to the axis (u=0, y=h) crosses the
    axis after the last surface at the back focal point. The focal length
    is: f = -h / u_final (where h is the initial height and u_final is
    the final slope).

    Args:
        surfaces: Lens system surfaces.
        n_initial: Initial refractive index.

    Returns:
        Effective focal length.
    """
    h = 0.001  # Very small height for paraxial approximation
    ray = Ray(y=h, z=surfaces[0].z_position - 50.0, u=0.0)
    trace = trace_ray_through_system(ray, surfaces, n_initial)

    if not trace.valid or len(trace.u_history) < 2:
        return float('inf')

    u_final = trace.u_history[-1]
    y_final = trace.y_history[-1]

    if abs(u_final) < 1e-15:
        return float('inf')

    # Back focal distance from last surface
    bfd = -y_final / np.tan(u_final)
    # Effective focal length: f = -h / tan(u_final)
    efl = -h / np.tan(u_final)
    return efl


def compute_f_number(surfaces: list[Surface], efl: float) -> float:
    """Compute the f-number (f/D) from the maximum aperture.

    The f-number is the ratio of focal length to aperture diameter.
    Lower f-number means more light gathering and shallower depth of field.

    Args:
        surfaces: Lens system surfaces.
        efl: Effective focal length.

    Returns:
        f-number.
    """
    # Find the aperture stop (smallest clear aperture)
    max_aperture = min(s.aperture for s in surfaces)
    D = 2 * max_aperture
    return abs(efl) / D


# ---------------------------------------------------------------------------
# Visualization 1: Ray trace through lens system
# ---------------------------------------------------------------------------

def plot_ray_trace(surfaces: list[Surface], n_initial: float, title: str,
                   filename: str, max_field_angle: float = 0.05):
    """Trace and plot rays through a lens system.

    Shows ray paths for on-axis and off-axis bundles, the lens elements,
    and marks the focal point.

    Args:
        surfaces: Lens system.
        n_initial: Initial refractive index.
        title: Plot title.
        filename: Output filename.
        max_field_angle: Maximum field angle in radians.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Draw optical axis
    z_min = surfaces[0].z_position - 30
    z_max = surfaces[-1].z_position + 50
    ax.axhline(0, color='gray', linewidth=0.5)

    # Draw lens surfaces
    for i, s in enumerate(surfaces):
        if abs(s.radius) == float('inf'):
            # Flat surface
            ax.plot([s.z_position, s.z_position], [-s.aperture, s.aperture],
                    'b-', linewidth=2)
        else:
            # Curved surface: draw arc
            theta_max = np.arcsin(min(s.aperture / abs(s.radius), 0.99))
            theta = np.linspace(-theta_max, theta_max, 100)
            z_c = s.z_position + s.radius  # center of curvature
            z_arc = z_c - s.radius * np.cos(theta)
            y_arc = s.radius * np.sin(theta)
            ax.plot(z_arc, y_arc, 'b-', linewidth=1.5)

    # Find focal length
    efl = find_focal_length(surfaces, n_initial)
    f_num = compute_f_number(surfaces, efl)
    print(f"\n  {title}:")
    print(f"  Effective focal length: {efl:.2f} mm")
    print(f"  f-number: f/{f_num:.1f}")

    # Trace rays for different field angles and heights
    field_angles = [0, max_field_angle / 2, max_field_angle]
    colors = ['red', 'green', 'blue']
    n_rays = 7  # rays per field angle

    max_aperture = min(s.aperture for s in surfaces)

    for angle, color in zip(field_angles, colors):
        heights = np.linspace(-max_aperture * 0.9, max_aperture * 0.9, n_rays)
        for h in heights:
            # Ray starts far to the left, at height determined by field angle
            z_start = z_min
            y_start = h + (surfaces[0].z_position - z_start) * np.tan(angle)
            ray = Ray(y=y_start, z=z_start, u=angle)
            trace = trace_ray_through_system(ray, surfaces, n_initial)

            if trace.valid:
                # Extend ray to image plane (find where it crosses axis or goes to z_max)
                z_list = trace.z_history
                y_list = trace.y_history

                # Extend last segment
                if len(z_list) >= 2:
                    u_final = trace.u_history[-1]
                    z_extend = z_max
                    y_extend = y_list[-1] + (z_extend - z_list[-1]) * np.tan(u_final)
                    z_list = list(z_list) + [z_extend]
                    y_list = list(y_list) + [y_extend]

                ax.plot(z_list, y_list, color=color, linewidth=0.5, alpha=0.6)

    ax.set_xlim(z_min, z_max)
    y_max = max(s.aperture for s in surfaces) * 1.5
    ax.set_ylim(-y_max, y_max)
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title(f'{title} (EFL = {efl:.1f} mm, f/{f_num:.1f})')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', label='On-axis'),
        Line2D([0], [0], color='green', label='Half-field'),
        Line2D([0], [0], color='blue', label='Full field'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"[Saved] {filename}")


# ---------------------------------------------------------------------------
# Visualization 2: Spot diagram
# ---------------------------------------------------------------------------

def plot_spot_diagram(surfaces: list[Surface], n_initial: float, title: str):
    """Generate spot diagrams for on-axis and off-axis field points.

    A spot diagram shows where rays from a single object point land on
    the image plane. A perfect lens would produce a single point; aberrations
    spread the spots into patterns characteristic of each aberration type.

    - Spherical aberration: circular spot, size increases with aperture
    - Coma: comet-shaped spot
    - Astigmatism: line-shaped spots in different orientations
    """
    efl = find_focal_length(surfaces, n_initial)

    # Image plane position (approximate: back focal point)
    # Trace a paraxial ray to find where it crosses the axis
    ray_par = Ray(y=0.001, z=surfaces[0].z_position - 50, u=0)
    trace_par = trace_ray_through_system(ray_par, surfaces, n_initial)
    if trace_par.valid:
        u_f = trace_par.u_history[-1]
        y_f = trace_par.y_history[-1]
        z_f = trace_par.z_history[-1]
        z_image = z_f - y_f / np.tan(u_f)
    else:
        z_image = surfaces[-1].z_position + abs(efl)

    field_angles = [0, 0.02, 0.04, 0.06]
    max_aperture = min(s.aperture for s in surfaces)

    fig, axes = plt.subplots(1, len(field_angles), figsize=(16, 4))

    rng = np.random.default_rng(42)

    for ax, angle in zip(axes, field_angles):
        # Generate rays uniformly filling the entrance pupil
        n_rays = 500
        r = max_aperture * 0.9 * np.sqrt(rng.uniform(0, 1, n_rays))
        phi = rng.uniform(0, 2 * np.pi, n_rays)
        hy = r * np.cos(phi)
        hz = r * np.sin(phi)  # out-of-plane (we approximate in meridional)

        spot_y = []
        spot_z_transverse = []

        for h, h_z in zip(hy, hz):
            y_start = h + (surfaces[0].z_position - (surfaces[0].z_position - 50)) * np.tan(angle)
            ray = Ray(y=y_start, z=surfaces[0].z_position - 50, u=angle)
            trace = trace_ray_through_system(ray, surfaces, n_initial)

            if trace.valid and len(trace.y_history) >= 2:
                # Propagate to image plane
                u_final = trace.u_history[-1]
                y_final = trace.y_history[-1]
                z_final = trace.z_history[-1]
                dz = z_image - z_final
                y_at_image = y_final + dz * np.tan(u_final)

                # Approximate sagittal component (using initial out-of-plane height)
                # In a full 3D tracer this would be computed exactly
                z_sagittal = h_z * (1 - dz / max(abs(efl), 1))

                spot_y.append(y_at_image)
                spot_z_transverse.append(z_sagittal * 0.01)  # Scale for visualization

        spot_y = np.array(spot_y)
        spot_z_transverse = np.array(spot_z_transverse)

        if len(spot_y) > 0:
            # Center on the chief ray position
            y_chief = np.median(spot_y)
            spot_y -= y_chief

            ax.scatter(spot_y * 1000, spot_z_transverse * 1000, s=1, c='red', alpha=0.3)

            rms = np.sqrt(np.mean(spot_y**2 + spot_z_transverse**2)) * 1000
            ax.text(0.05, 0.95, f'RMS = {rms:.2f} um',
                    transform=ax.transAxes, fontsize=9, va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat'))

        ax.set_aspect('equal')
        ax.set_xlabel('Tangential (um)')
        ax.set_ylabel('Sagittal (um)')
        ax.set_title(f'Field = {np.degrees(angle):.1f} deg')
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Spot Diagram: {title}', fontsize=14)
    fig.tight_layout()
    fig.savefig('09_ray_tracing_spot.png', dpi=150)
    plt.close(fig)
    print("[Saved] 09_ray_tracing_spot.png")


# ---------------------------------------------------------------------------
# Visualization 3: Ray fan plot
# ---------------------------------------------------------------------------

def plot_ray_fan(surfaces: list[Surface], n_initial: float, title: str):
    """Generate ray fan (ray aberration) plots.

    A ray fan shows the transverse ray error as a function of pupil height.
    - x-axis: normalized pupil coordinate (entrance pupil height / max)
    - y-axis: transverse ray aberration (distance from ideal image point)

    The shape of the ray fan curve identifies the aberration type:
    - Linear: defocus or field curvature
    - Cubic: spherical aberration (S-shaped for 3rd order)
    - Asymmetric: coma
    """
    efl = find_focal_length(surfaces, n_initial)

    # Find paraxial image plane
    ray_par = Ray(y=0.001, z=surfaces[0].z_position - 50, u=0)
    trace_par = trace_ray_through_system(ray_par, surfaces, n_initial)
    if trace_par.valid:
        u_f = trace_par.u_history[-1]
        y_f = trace_par.y_history[-1]
        z_f = trace_par.z_history[-1]
        z_image = z_f - y_f / np.tan(u_f)
    else:
        z_image = surfaces[-1].z_position + abs(efl)

    max_aperture = min(s.aperture for s in surfaces)
    field_angles = [0, 0.02, 0.04]

    fig, axes = plt.subplots(1, len(field_angles), figsize=(15, 5))

    for ax, angle in zip(axes, field_angles):
        # Normalized pupil heights
        n_rays = 100
        pupil_heights = np.linspace(-0.95, 0.95, n_rays)
        heights = pupil_heights * max_aperture

        aberration = []

        for h in heights:
            y_start = h + (surfaces[0].z_position - (surfaces[0].z_position - 50)) * np.tan(angle)
            ray = Ray(y=y_start, z=surfaces[0].z_position - 50, u=angle)
            trace = trace_ray_through_system(ray, surfaces, n_initial)

            if trace.valid and len(trace.y_history) >= 2:
                u_final = trace.u_history[-1]
                y_final = trace.y_history[-1]
                z_final = trace.z_history[-1]
                dz = z_image - z_final
                y_at_image = y_final + dz * np.tan(u_final)

                # Ideal image position (from chief ray)
                ray_chief = Ray(y=0, z=surfaces[0].z_position - 50, u=angle)
                trace_chief = trace_ray_through_system(ray_chief, surfaces, n_initial)
                if trace_chief.valid:
                    y_ideal = (trace_chief.y_history[-1] +
                               (z_image - trace_chief.z_history[-1]) *
                               np.tan(trace_chief.u_history[-1]))
                else:
                    y_ideal = z_image * np.tan(angle)

                aberration.append((y_at_image - y_ideal) * 1000)  # um
            else:
                aberration.append(np.nan)

        aberration = np.array(aberration)

        ax.plot(pupil_heights, aberration, 'b-', linewidth=2)
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5)
        ax.set_xlabel('Normalized pupil height')
        ax.set_ylabel('Transverse aberration (um)')
        ax.set_title(f'Field = {np.degrees(angle):.1f} deg')
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Ray Fan Plot: {title}', fontsize=14)
    fig.tight_layout()
    fig.savefig('09_ray_tracing_rayfan.png', dpi=150)
    plt.close(fig)
    print("[Saved] 09_ray_tracing_rayfan.png")


# ---------------------------------------------------------------------------
# Visualization 4: Chromatic focal shift
# ---------------------------------------------------------------------------

def plot_chromatic_focal_shift(surfaces: list[Surface], n_initial: float, title: str):
    """Plot focal length variation with wavelength (chromatic aberration).

    Chromatic aberration occurs because the refractive index depends on
    wavelength. An achromatic doublet minimizes this by using two glass
    types with different dispersion.

    We approximate the wavelength dependence using the Cauchy equation.
    """
    wavelengths = np.linspace(400, 700, 50)  # nm

    # For each wavelength, modify refractive indices using Cauchy approximation
    # and recompute the focal length
    focal_lengths = []

    for wl in wavelengths:
        wl_um = wl / 1000
        # Scale each glass refractive index with wavelength
        modified_surfaces = []
        for s in surfaces:
            if s.n_after > 1.01:  # Glass (not air)
                # Cauchy model: n(lambda) = n_d + B/lambda^2
                # We estimate B from the glass type
                n_d = s.n_after
                B = 0.004 if n_d < 1.55 else 0.008  # Crown vs flint
                n_wl = n_d + B * (1 / wl_um**2 - 1 / 0.5893**2)
                modified_surfaces.append(Surface(
                    s.radius, s.z_position, s.aperture, n_wl, s.label
                ))
            else:
                modified_surfaces.append(s)

        efl = find_focal_length(modified_surfaces, n_initial)
        focal_lengths.append(efl)

    focal_lengths = np.array(focal_lengths)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(wavelengths, focal_lengths, 'b-', linewidth=2)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Focal length (mm)')
    ax.set_title(f'Chromatic Focal Shift: {title}')
    ax.grid(True, alpha=0.3)

    # Mark key wavelengths
    for wl, name, color in [(486, 'F (blue)', 'blue'),
                             (589, 'd (yellow)', 'orange'),
                             (656, 'C (red)', 'red')]:
        idx = np.argmin(np.abs(wavelengths - wl))
        ax.axvline(wl, color=color, linestyle=':', alpha=0.5)
        ax.plot(wl, focal_lengths[idx], 'o', color=color, markersize=8)
        ax.text(wl + 5, focal_lengths[idx], f'{name}\nf={focal_lengths[idx]:.1f}',
                fontsize=8, color=color)

    # Longitudinal chromatic aberration
    idx_F = np.argmin(np.abs(wavelengths - 486))
    idx_C = np.argmin(np.abs(wavelengths - 656))
    lca = focal_lengths[idx_F] - focal_lengths[idx_C]
    ax.text(0.05, 0.95, f'LCA (F-C) = {lca:.2f} mm',
            transform=ax.transAxes, fontsize=11, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat'))

    print(f"\n  Chromatic aberration (LCA): {lca:.3f} mm")

    fig.tight_layout()
    fig.savefig('09_ray_tracing_chromatic.png', dpi=150)
    plt.close(fig)
    print("[Saved] 09_ray_tracing_chromatic.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Sequential Ray Tracing: Multi-Element Lens Systems")
    print("=" * 60)

    # Achromatic doublet
    print("\n--- Achromatic Doublet ---")
    surfaces_d, n0_d = simple_doublet()
    plot_ray_trace(surfaces_d, n0_d, 'Achromatic Doublet', '09_ray_tracing_doublet.png')

    # Cooke triplet
    print("\n--- Cooke Triplet ---")
    surfaces_t, n0_t = cooke_triplet()
    plot_ray_trace(surfaces_t, n0_t, 'Cooke Triplet', '09_ray_tracing_triplet.png')

    # Spot diagram (using doublet)
    print("\n--- Spot Diagram ---")
    plot_spot_diagram(surfaces_d, n0_d, 'Achromatic Doublet')

    # Ray fan plot
    print("\n--- Ray Fan Plot ---")
    plot_ray_fan(surfaces_d, n0_d, 'Achromatic Doublet')

    # Chromatic focal shift
    print("\n--- Chromatic Focal Shift ---")
    plot_chromatic_focal_shift(surfaces_d, n0_d, 'Achromatic Doublet')

    print("\nDone. All plots saved.")
