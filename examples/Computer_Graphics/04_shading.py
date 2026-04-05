"""
Shading Models: Phong, Blinn-Phong, and PBR (Cook-Torrance)
============================================================

Implements three shading models of increasing physical accuracy:

1. **Phong**: Classic empirical model (ambient + diffuse + specular)
2. **Blinn-Phong**: Cheaper half-vector variant (used by OpenGL fixed pipeline)
3. **PBR Cook-Torrance**: Physically-based with GGX NDF, Schlick Fresnel,
   and Smith geometry term

Renders spheres side by side to visually compare the models.

Why these three?  Phong is the historical foundation, Blinn-Phong was the
GPU standard for decades, and PBR (Cook-Torrance) is the modern standard
used by Unreal, Unity, Disney, and most film production pipelines.

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Utility functions
# ---------------------------------------------------------------------------


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector (or array of vectors along last axis).

    Why a helper?  Division by zero on zero-length vectors causes NaN
    propagation.  This function handles that edge case.
    """
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    # Avoid division by zero: return zero vector if norm is zero
    return np.where(norm > 1e-10, v / norm, 0)


def reflect(incident: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Reflect incident vector about normal.

    Formula: R = I - 2(I . N)N
    This is the standard reflection formula used in Phong specular and
    in ray tracing for mirror reflections.
    """
    dot = np.sum(incident * normal, axis=-1, keepdims=True)
    return incident - 2 * dot * normal


def generate_sphere_normals(width: int, height: int) -> tuple:
    """Generate a sphere's surface normals for a front-facing hemisphere.

    Returns (normals, mask) where normals has shape (H, W, 3) and mask
    is a boolean array indicating which pixels are on the sphere.

    Why a hemisphere?  We render the sphere as seen from the front --
    only front-facing normals are visible.  This is the standard
    "sphere in a box" technique for shading demos.
    """
    # Create a grid of normalized device coordinates [-1, 1]
    y_coords = np.linspace(1, -1, height)
    x_coords = np.linspace(-1, 1, width)
    X, Y = np.meshgrid(x_coords, y_coords)

    # Points on the sphere satisfy x^2 + y^2 + z^2 = 1
    # So z = sqrt(1 - x^2 - y^2) for the front hemisphere
    R_sq = X**2 + Y**2
    mask = R_sq <= 1.0  # Inside the sphere's silhouette

    Z = np.zeros_like(X)
    Z[mask] = np.sqrt(1.0 - R_sq[mask])

    normals = np.stack([X, Y, Z], axis=-1)
    # Normals on the sphere surface are just the position (for a unit sphere)
    # Already normalized by construction (x^2 + y^2 + z^2 = 1)

    return normals, mask


# ---------------------------------------------------------------------------
# 2. Phong Illumination Model
# ---------------------------------------------------------------------------

def phong_shading(normal: np.ndarray, light_dir: np.ndarray,
                  view_dir: np.ndarray, material: dict) -> np.ndarray:
    """Compute Phong illumination: ambient + diffuse + specular.

    The Phong model (Bui Tuong Phong, 1975) is the first widely-used
    local illumination model.  It's empirical (not physically based)
    but captures the three main visual components of surface appearance:

    - Ambient: constant base illumination (fakes global illumination)
    - Diffuse: Lambert's cosine law (brightness proportional to N.L)
    - Specular: mirror-like highlight (R.V)^shininess

    Why is it not physically based?  It doesn't conserve energy -- the
    sum of reflected light can exceed incoming light.  It also doesn't
    handle Fresnel effects or microfacet roughness correctly.

    Parameters
    ----------
    normal    : Surface normals (H, W, 3)
    light_dir : Direction TO the light (H, W, 3), normalized
    view_dir  : Direction TO the viewer (H, W, 3), normalized
    material  : Dict with keys: ambient, diffuse, specular (RGB),
                shininess (float), light_color (RGB)
    """
    ambient_color = material.get('ambient', np.array([0.1, 0.1, 0.1]))
    diffuse_color = material.get('diffuse', np.array([0.7, 0.2, 0.2]))
    specular_color = material.get('specular', np.array([1.0, 1.0, 1.0]))
    shininess = material.get('shininess', 32.0)
    light_color = material.get('light_color', np.array([1.0, 1.0, 1.0]))

    # Ambient: constant contribution regardless of geometry
    ambient = ambient_color

    # Diffuse: Lambert's cosine law
    # Why max(0, N.L)?  Negative dot product means light is behind the
    # surface -- it shouldn't contribute negative light.
    NdotL = np.sum(normal * light_dir, axis=-1, keepdims=True)
    NdotL = np.maximum(NdotL, 0)
    diffuse = diffuse_color * NdotL * light_color

    # Specular: reflected light direction compared to view direction
    # R = 2(N.L)N - L  (reflection of light about normal)
    R = 2 * NdotL * normal - light_dir
    R = normalize(R)
    RdotV = np.sum(R * view_dir, axis=-1, keepdims=True)
    RdotV = np.maximum(RdotV, 0)
    specular = specular_color * (RdotV ** shininess) * light_color

    return ambient + diffuse + specular


# ---------------------------------------------------------------------------
# 3. Blinn-Phong Model
# ---------------------------------------------------------------------------

def blinn_phong_shading(normal: np.ndarray, light_dir: np.ndarray,
                         view_dir: np.ndarray, material: dict) -> np.ndarray:
    """Compute Blinn-Phong illumination using the half-vector.

    Blinn (1977) replaced the reflection vector R with the half-vector H:
      H = normalize(L + V)
      specular = (N.H)^shininess

    Why is this better?
    1. Cheaper to compute: no reflection calculation needed
    2. More physically plausible: the highlight shape is more realistic
       for grazing angles
    3. It was the default in OpenGL's fixed-function pipeline for decades

    The half-vector H represents the hypothetical microfacet normal that
    would reflect light from L toward V -- this connects to PBR's
    microfacet theory.
    """
    ambient_color = material.get('ambient', np.array([0.1, 0.1, 0.1]))
    diffuse_color = material.get('diffuse', np.array([0.7, 0.2, 0.2]))
    specular_color = material.get('specular', np.array([1.0, 1.0, 1.0]))
    shininess = material.get('shininess', 32.0)
    light_color = material.get('light_color', np.array([1.0, 1.0, 1.0]))

    ambient = ambient_color

    NdotL = np.sum(normal * light_dir, axis=-1, keepdims=True)
    NdotL = np.maximum(NdotL, 0)
    diffuse = diffuse_color * NdotL * light_color

    # Half-vector: bisector of light and view directions
    H = normalize(light_dir + view_dir)
    NdotH = np.sum(normal * H, axis=-1, keepdims=True)
    NdotH = np.maximum(NdotH, 0)

    # Why higher shininess for Blinn-Phong?  The N.H falloff is broader
    # than R.V, so equivalent visual sharpness requires ~4x the exponent.
    specular = specular_color * (NdotH ** shininess) * light_color

    return ambient + diffuse + specular


# ---------------------------------------------------------------------------
# 4. PBR Cook-Torrance BRDF
# ---------------------------------------------------------------------------

def ggx_ndf(NdotH: np.ndarray, roughness: float) -> np.ndarray:
    """GGX/Trowbridge-Reitz Normal Distribution Function.

    Models the statistical distribution of microfacet orientations on
    a rough surface.  GGX has longer tails than Beckmann, producing
    more realistic highlights that fade gradually instead of cutting off.

    Why GGX over Beckmann?  GGX matches measured material data better
    and is now the industry standard (used in Disney's principled BRDF,
    Unreal Engine, Unity HDRP).
    """
    a = roughness * roughness
    a2 = a * a
    denom = NdotH * NdotH * (a2 - 1.0) + 1.0
    # Add epsilon to prevent division by zero at grazing angles
    return a2 / (np.pi * denom * denom + 1e-10)


def schlick_fresnel(cosTheta: np.ndarray, F0: np.ndarray) -> np.ndarray:
    """Schlick's approximation of the Fresnel equations.

    At normal incidence, reflectance = F0 (material-dependent).
    At grazing angles, reflectance -> 1.0 (all materials become mirrors).

    Why Schlick instead of exact Fresnel?  The approximation is very
    accurate and much cheaper to compute.  The key insight: Fresnel
    reflectance is mostly determined by F0 and the viewing angle.

    F0 for common materials:
    - Water:   0.02
    - Plastic: 0.04
    - Gold:    (1.0, 0.71, 0.29)  -- colored!
    - Silver:  (0.95, 0.93, 0.88)
    """
    return F0 + (1.0 - F0) * (1.0 - cosTheta) ** 5


def smith_geometry(NdotV: np.ndarray, NdotL: np.ndarray,
                   roughness: float) -> np.ndarray:
    """Smith's geometry function (Schlick-GGX approximation).

    Models self-shadowing and self-masking of microfacets.  Rough
    surfaces have more microfacets that block each other, reducing
    the visible reflected light -- this is why rough surfaces appear
    darker at grazing angles.

    Why Smith's separable form?  It factors into G(V) * G(L), which
    is physically motivated (masking and shadowing are approximately
    independent) and computationally convenient.
    """
    r = roughness + 1.0
    k = (r * r) / 8.0  # Remapping for direct lighting (not IBL)

    def G_SchlickGGX(NdotX):
        return NdotX / (NdotX * (1.0 - k) + k + 1e-10)

    return G_SchlickGGX(NdotV) * G_SchlickGGX(NdotL)


def cook_torrance_shading(normal: np.ndarray, light_dir: np.ndarray,
                           view_dir: np.ndarray, material: dict) -> np.ndarray:
    """Cook-Torrance PBR BRDF with Lambert diffuse.

    The rendering equation for a single direct light:
      Lo = (kd * diffuse/pi + ks * D*F*G / (4*NdotV*NdotL)) * Li * NdotL

    Where:
    - D = Normal Distribution Function (GGX)
    - F = Fresnel term (Schlick)
    - G = Geometry term (Smith)
    - kd = 1 - ks (energy conservation)

    Why /pi for diffuse?  A Lambertian surface reflects uniformly over
    the hemisphere.  Integrating cos(theta) over the hemisphere gives pi,
    so dividing by pi normalizes the BRDF to conserve energy.

    Parameters
    ----------
    material : Dict with keys: albedo (RGB), metallic (float 0-1),
               roughness (float 0-1), light_color (RGB)
    """
    albedo = material.get('albedo', np.array([0.7, 0.2, 0.2]))
    metallic = material.get('metallic', 0.0)
    roughness = material.get('roughness', 0.5)
    light_color = material.get('light_color', np.array([1.0, 1.0, 1.0]))
    light_intensity = material.get('light_intensity', 3.0)

    H = normalize(light_dir + view_dir)

    NdotL = np.maximum(np.sum(normal * light_dir, axis=-1, keepdims=True), 0)
    NdotV = np.maximum(np.sum(normal * view_dir, axis=-1, keepdims=True), 0)
    NdotH = np.maximum(np.sum(normal * H, axis=-1, keepdims=True), 0)
    VdotH = np.maximum(np.sum(view_dir * H, axis=-1, keepdims=True), 0)

    # F0: base reflectivity
    # Dielectrics have F0 ~ 0.04, metals use albedo as F0
    # Why this interpolation?  Metallic surfaces reflect their albedo color
    # as specular (colored reflections), while dielectrics reflect white.
    F0 = np.full_like(albedo, 0.04)
    F0 = F0 * (1 - metallic) + albedo * metallic

    # BRDF components
    D = ggx_ndf(NdotH, roughness)
    F = schlick_fresnel(VdotH, F0)
    G = smith_geometry(NdotV, NdotL, roughness)

    # Specular BRDF
    numerator = D * F * G
    denominator = 4.0 * NdotV * NdotL + 1e-10
    specular = numerator / denominator

    # Energy conservation: diffuse + specular <= 1
    kS = F  # Specular fraction = Fresnel reflectance
    kD = (1.0 - kS) * (1.0 - metallic)  # Metals have no diffuse

    # Final radiance
    Lo = (kD * albedo / np.pi + specular) * light_color * light_intensity * NdotL

    # Simple ambient term (would be IBL in a full PBR pipeline)
    ambient = 0.03 * albedo
    color = ambient + Lo

    return color


# ---------------------------------------------------------------------------
# 5. Rendering a sphere with a given shading function
# ---------------------------------------------------------------------------

def render_sphere(shading_fn, material: dict, light_pos: np.ndarray,
                  size: int = 256) -> np.ndarray:
    """Render a shaded sphere using the given shading function.

    Why per-pixel shading?  This simulates what a fragment shader does:
    for each visible fragment (pixel), compute the illumination from
    the surface normal, light direction, and view direction.
    """
    normals, mask = generate_sphere_normals(size, size)

    # View direction: camera looking along -Z, so view dir is +Z
    view_dir = np.zeros_like(normals)
    view_dir[..., 2] = 1.0  # Looking from +Z toward -Z

    # Light direction: from surface point toward light
    # We compute this per-pixel for positional lights
    light_dir = normalize(light_pos - normals * 0.0)  # Directional approximation
    light_dir = np.broadcast_to(normalize(light_pos), normals.shape).copy()

    # Call the shading function
    color = shading_fn(normals, light_dir, view_dir, material)

    # Apply mask: set background to dark
    image = np.zeros((size, size, 3))
    image[mask] = np.clip(color[mask], 0, 1)

    # Simple tone mapping: Reinhard operator
    # Why tone map?  PBR can produce HDR values > 1.  Tone mapping
    # compresses the range while preserving relative brightness.
    image = image / (image + 1.0)  # Reinhard

    # Gamma correction: linear -> sRGB
    # Why gamma?  Human vision is non-linear.  Without gamma correction,
    # mid-tones appear too dark on standard monitors.
    image = np.power(np.clip(image, 0, 1), 1.0 / 2.2)

    return image


# ---------------------------------------------------------------------------
# 6. Comparison demo: Phong vs Blinn-Phong vs PBR
# ---------------------------------------------------------------------------

def demo_model_comparison():
    """Compare all three shading models side by side on identical spheres."""
    light_pos = np.array([2.0, 2.0, 3.0])

    # Matching material parameters for fair comparison
    classic_material = {
        'ambient': np.array([0.1, 0.02, 0.02]),
        'diffuse': np.array([0.8, 0.2, 0.2]),
        'specular': np.array([1.0, 1.0, 1.0]),
        'shininess': 64.0,
        'light_color': np.array([1.0, 1.0, 1.0]),
    }

    pbr_material = {
        'albedo': np.array([0.8, 0.2, 0.2]),
        'metallic': 0.0,
        'roughness': 0.3,
        'light_color': np.array([1.0, 1.0, 1.0]),
        'light_intensity': 3.0,
    }

    phong_img = render_sphere(phong_shading, classic_material, light_pos)
    blinn_img = render_sphere(blinn_phong_shading, classic_material, light_pos)
    pbr_img = render_sphere(cook_torrance_shading, pbr_material, light_pos)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Shading Model Comparison", fontsize=14, fontweight='bold')

    for ax, img, title in zip(axes,
                               [phong_img, blinn_img, pbr_img],
                               ["Phong", "Blinn-Phong", "PBR Cook-Torrance"]):
        ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("/opt/projects/01_Personal/03_Study/examples/Computer_Graphics/"
                "output_04_model_comparison.png", dpi=100)
    plt.show()


# ---------------------------------------------------------------------------
# 7. PBR material gallery
# ---------------------------------------------------------------------------

def demo_pbr_gallery():
    """Show how roughness and metallic parameters affect PBR appearance.

    Why a gallery?  PBR's power is in its two intuitive parameters:
    roughness (0=mirror, 1=matte) and metallic (0=dielectric, 1=metal).
    This grid shows the full range of appearances from just two sliders.
    """
    light_pos = np.array([2.0, 2.5, 3.0])

    roughness_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    metallic_values = [0.0, 0.5, 1.0]

    fig, axes = plt.subplots(len(metallic_values), len(roughness_values),
                              figsize=(16, 10))
    fig.suptitle("PBR Material Gallery: Roughness vs Metallic",
                 fontsize=14, fontweight='bold')

    for i, metallic in enumerate(metallic_values):
        for j, roughness in enumerate(roughness_values):
            material = {
                'albedo': np.array([0.9, 0.6, 0.2]),  # Gold-ish base
                'metallic': metallic,
                'roughness': roughness,
                'light_color': np.array([1.0, 1.0, 1.0]),
                'light_intensity': 3.0,
            }
            img = render_sphere(cook_torrance_shading, material, light_pos,
                                size=128)
            ax = axes[i, j]
            ax.imshow(img)
            ax.axis('off')

            if i == 0:
                ax.set_title(f"rough={roughness}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"metal={metallic}", fontsize=9)

    plt.tight_layout()
    plt.savefig("/opt/projects/01_Personal/03_Study/examples/Computer_Graphics/"
                "output_04_pbr_gallery.png", dpi=100)
    plt.show()


# ---------------------------------------------------------------------------
# 8. Multi-light scene
# ---------------------------------------------------------------------------

def demo_multi_light():
    """Render a sphere with multiple colored lights.

    Why multiple lights?  Real scenes have many light sources.
    The rendering equation is linear, so we just sum contributions
    from each light.  This is the basis for multi-pass and deferred
    rendering techniques.
    """
    size = 300
    normals, mask = generate_sphere_normals(size, size)
    view_dir = np.zeros_like(normals)
    view_dir[..., 2] = 1.0

    lights = [
        {'pos': np.array([3.0, 2.0, 2.0]), 'color': np.array([1.0, 0.3, 0.3]),
         'intensity': 3.0},
        {'pos': np.array([-3.0, 1.0, 2.0]), 'color': np.array([0.3, 0.3, 1.0]),
         'intensity': 2.5},
        {'pos': np.array([0.0, -2.0, 3.0]), 'color': np.array([0.3, 1.0, 0.3]),
         'intensity': 2.0},
    ]

    total_color = np.zeros_like(normals)

    for light in lights:
        light_dir = normalize(np.broadcast_to(
            normalize(light['pos']), normals.shape).copy())

        material = {
            'albedo': np.array([0.8, 0.8, 0.8]),
            'metallic': 0.1,
            'roughness': 0.4,
            'light_color': light['color'],
            'light_intensity': light['intensity'],
        }

        total_color += cook_torrance_shading(normals, light_dir, view_dir, material)

    image = np.zeros((size, size, 3))
    image[mask] = np.clip(total_color[mask], 0, None)
    image = image / (image + 1.0)
    image = np.power(np.clip(image, 0, 1), 1.0 / 2.2)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    ax.set_title("PBR with Multiple Colored Lights (R+G+B)", fontsize=12)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig("/opt/projects/01_Personal/03_Study/examples/Computer_Graphics/"
                "output_04_multi_light.png", dpi=100)
    plt.show()


# ---------------------------------------------------------------------------
# 9. Fresnel effect visualization
# ---------------------------------------------------------------------------

def demo_fresnel():
    """Visualize the Fresnel effect: reflectance increases at grazing angles.

    This is visible in everyday life -- look at a table at a shallow
    angle and you see reflections; look straight down and you see the
    surface color.  The Fresnel equations govern this transition.
    """
    angles = np.linspace(0, 90, 200)
    cos_theta = np.cos(np.radians(angles))

    F0_values = {
        'Water (0.02)': 0.02,
        'Plastic (0.04)': 0.04,
        'Glass (0.08)': 0.08,
        'Diamond (0.17)': 0.17,
        'Iron (0.56)': 0.56,
        'Gold (0.95)': 0.95,
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for name, F0 in F0_values.items():
        F = F0 + (1 - F0) * (1 - cos_theta) ** 5
        ax.plot(angles, F, label=name, linewidth=2)

    ax.set_xlabel("Angle from Normal (degrees)", fontsize=11)
    ax.set_ylabel("Fresnel Reflectance", fontsize=11)
    ax.set_title("Schlick Fresnel Approximation for Various Materials",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig("/opt/projects/01_Personal/03_Study/examples/Computer_Graphics/"
                "output_04_fresnel.png", dpi=100)
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Shading Models: Phong, Blinn-Phong, and PBR")
    print("=" * 60)

    print("\n[1/4] Model comparison (Phong vs Blinn-Phong vs PBR)...")
    demo_model_comparison()

    print("\n[2/4] PBR material gallery...")
    demo_pbr_gallery()

    print("\n[3/4] Multi-light PBR scene...")
    demo_multi_light()

    print("\n[4/4] Fresnel effect visualization...")
    demo_fresnel()

    print("\nDone!")


if __name__ == "__main__":
    main()
