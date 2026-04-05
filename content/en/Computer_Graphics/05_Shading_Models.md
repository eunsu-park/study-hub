# 05. Shading Models

[&larr; Previous: Rasterization](04_Rasterization.md) | [Next: Texture Mapping &rarr;](06_Texture_Mapping.md)

---

## Learning Objectives

1. Understand the components of empirical shading: ambient, diffuse (Lambert), and specular (Phong)
2. Derive and implement the Blinn-Phong modification for specular highlights
3. Distinguish between flat, Gouraud, and Phong shading interpolation methods
4. Understand the physical basis of Physically Based Rendering (PBR)
5. Implement the Cook-Torrance BRDF with GGX normal distribution, Schlick's Fresnel, and Smith's geometry function
6. Explain the metallic-roughness workflow used in modern game engines
7. Compare Phong and PBR shading results on a sphere
8. Build intuition for how surface properties (roughness, metallicness) affect appearance

---

## Why This Matters

Shading is what makes 3D objects look *real* -- or at least convincing. Without shading, a sphere is just a flat circle. The difference between a plastic ball, a chrome sphere, and a rough stone is entirely determined by how the surface interacts with light. Understanding shading models -- from the classic Phong model taught since the 1970s to the physically based rendering (PBR) used in every modern game engine -- is essential for creating believable digital imagery. PBR has become the industry standard because it produces consistent, predictable results under any lighting condition, guided by actual physics.

---

## 1. Light-Surface Interaction

When light hits a surface, three things can happen:
- **Absorption**: Light energy is converted to heat (the surface appears darker)
- **Reflection**: Light bounces off the surface
- **Transmission**: Light passes through (transparency, refraction)

For opaque surfaces, we focus on reflection, which has two components:

- **Diffuse reflection**: Light scatters equally in all directions (matte surfaces like chalk, rubber)
- **Specular reflection**: Light reflects in a preferred direction (shiny surfaces like metal, glass)

```
         Incoming      Specular        Incoming      Diffuse
          light        reflection       light       reflection
            \         /                  \         / | \
             \       /                    \       /  |  \
              \     /                      \     /   |   \
    ───────────●───────────     ───────────●───────────
              surface                     surface
```

---

## 2. The Phong Reflection Model

The **Phong model** (Bui Tuong Phong, 1975) is an empirical model that approximates surface appearance using three additive components.

### 2.1 Ambient Component

Ambient light represents indirect illumination from the environment -- light that has bounced off many surfaces before reaching this point. It is approximated as a constant:

$$I_{\text{ambient}} = k_a \cdot I_a$$

Where:
- $k_a$: ambient reflectance coefficient (material property, $[0, 1]$)
- $I_a$: ambient light intensity

> **Limitation**: Real indirect illumination varies across the scene. Global illumination techniques (radiosity, path tracing) model this accurately, but ambient is a cheap approximation.

### 2.2 Diffuse Component (Lambert's Law)

A perfectly diffuse (Lambertian) surface scatters incident light equally in all directions. The perceived brightness depends only on the angle between the surface normal $\mathbf{N}$ and the light direction $\mathbf{L}$:

$$I_{\text{diffuse}} = k_d \cdot I_l \cdot \max(\mathbf{N} \cdot \mathbf{L}, 0)$$

Where:
- $k_d$: diffuse reflectance coefficient (material color, typically an RGB tuple)
- $I_l$: light intensity (color)
- $\mathbf{N}$: unit surface normal
- $\mathbf{L}$: unit vector from surface point toward the light
- $\max(\ldots, 0)$: clamp to prevent negative contributions (surface facing away from light)

**Physical interpretation**: $\mathbf{N} \cdot \mathbf{L} = \cos\theta_i$ where $\theta_i$ is the angle of incidence. At $\theta_i = 0$ (light hits perpendicularly), the surface receives maximum illumination. At $\theta_i = 90°$, the surface is edge-on to the light and receives none.

### 2.3 Specular Component (Phong)

Specular reflection creates bright highlights on shiny surfaces. In the Phong model, specular intensity depends on the angle between the reflection vector $\mathbf{R}$ and the view direction $\mathbf{V}$:

$$I_{\text{specular}} = k_s \cdot I_l \cdot \max(\mathbf{R} \cdot \mathbf{V}, 0)^n$$

Where:
- $k_s$: specular reflectance coefficient
- $\mathbf{R}$: reflection of $\mathbf{L}$ about $\mathbf{N}$: $\mathbf{R} = 2(\mathbf{N} \cdot \mathbf{L})\mathbf{N} - \mathbf{L}$
- $\mathbf{V}$: unit vector from surface point toward the camera
- $n$: shininess exponent (higher = sharper, smaller highlight)

| Shininess $n$ | Surface Appearance |
|---------------|-------------------|
| 1-10 | Very rough, wide highlight (rubber) |
| 30-50 | Moderate shine (plastic) |
| 100-500 | Highly glossy (polished metal) |

### 2.4 Complete Phong Model

$$I = k_a I_a + \sum_{\text{lights}} \left[ k_d I_l (\mathbf{N} \cdot \mathbf{L}) + k_s I_l (\mathbf{R} \cdot \mathbf{V})^n \right]$$

```python
import numpy as np

def normalize(v):
    """Normalize a vector to unit length."""
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v

def reflect(incident, normal):
    """
    Compute the reflection of a vector about a normal.

    The reflection formula: R = 2(N.L)N - L
    This mirrors the incident direction across the surface normal.
    """
    return 2.0 * np.dot(normal, incident) * normal - incident

def phong_shading(point, normal, view_pos, light_pos, light_color,
                  ka=0.1, kd=0.7, ks=0.3, shininess=32.0,
                  ambient_color=None, object_color=None):
    """
    Compute Phong shading at a surface point.

    Parameters:
        point: surface position (3D)
        normal: surface normal (unit vector)
        view_pos: camera position (3D)
        light_pos: light position (3D)
        light_color: light RGB color
        ka, kd, ks: ambient, diffuse, specular coefficients
        shininess: specular exponent (higher = sharper highlight)
        ambient_color: color of ambient light (defaults to light_color)
        object_color: diffuse material color (defaults to white)

    Returns:
        RGB color (numpy array)
    """
    if ambient_color is None:
        ambient_color = np.array([1.0, 1.0, 1.0])
    if object_color is None:
        object_color = np.array([1.0, 1.0, 1.0])

    N = normalize(normal)
    L = normalize(light_pos - point)     # Direction to light
    V = normalize(view_pos - point)      # Direction to camera
    R = reflect(L, N)                    # Reflection of light direction

    # Ambient: constant base illumination
    ambient = ka * ambient_color * object_color

    # Diffuse: Lambert's cosine law
    diff_factor = max(np.dot(N, L), 0.0)
    diffuse = kd * diff_factor * light_color * object_color

    # Specular: shiny highlight
    spec_factor = max(np.dot(R, V), 0.0) ** shininess
    specular = ks * spec_factor * light_color

    return np.clip(ambient + diffuse + specular, 0.0, 1.0)
```

---

## 3. Blinn-Phong Model

Jim Blinn (1977) proposed a modification that replaces the reflection vector $\mathbf{R}$ with the **halfway vector** $\mathbf{H}$:

$$\mathbf{H} = \text{normalize}(\mathbf{L} + \mathbf{V})$$

The specular term becomes:

$$I_{\text{specular}} = k_s \cdot I_l \cdot \max(\mathbf{N} \cdot \mathbf{H}, 0)^n$$

### 3.1 Why Blinn-Phong?

1. **Efficiency**: Computing $\mathbf{H}$ is cheaper than computing $\mathbf{R}$ (one addition + normalize vs. three multiplications)
2. **Better physics**: The halfway vector model more closely approximates microfacet theory (the basis of PBR)
3. **Stability**: For directional lights (where $\mathbf{L}$ is constant), $\mathbf{H}$ depends only on $\mathbf{V}$
4. **Industry standard**: OpenGL's fixed-function pipeline used Blinn-Phong

> **Note**: Blinn-Phong produces slightly wider highlights than Phong for the same exponent $n$. To match Phong's appearance, use roughly $n_{\text{Blinn}} \approx 4 \times n_{\text{Phong}}$.

```python
def blinn_phong_shading(point, normal, view_pos, light_pos, light_color,
                         ka=0.1, kd=0.7, ks=0.3, shininess=64.0,
                         object_color=None):
    """
    Blinn-Phong shading model.

    The key difference from Phong: we use the halfway vector H instead
    of the reflection vector R. This is both faster to compute and
    more physically plausible.
    """
    if object_color is None:
        object_color = np.array([1.0, 1.0, 1.0])

    N = normalize(normal)
    L = normalize(light_pos - point)
    V = normalize(view_pos - point)

    # The halfway vector: the "average" of light and view directions
    # Physically, N.H measures how well the surface is oriented to
    # reflect light from L toward V
    H = normalize(L + V)

    # Ambient
    ambient = ka * object_color

    # Diffuse (same as Phong)
    diff = max(np.dot(N, L), 0.0)
    diffuse = kd * diff * light_color * object_color

    # Specular: N.H instead of R.V
    spec = max(np.dot(N, H), 0.0) ** shininess
    specular = ks * spec * light_color

    return np.clip(ambient + diffuse + specular, 0.0, 1.0)
```

---

## 4. Shading Interpolation Methods

The shading model computes color at specific points. But where exactly should we evaluate it -- per triangle, per vertex, or per pixel?

### 4.1 Flat Shading

Compute the lighting **once per triangle** using the face normal. The entire triangle gets one uniform color.

- **Pros**: Fast, very cheap
- **Cons**: Faceted appearance -- individual triangles are clearly visible
- **Use case**: Low-poly art style, extremely performance-constrained scenarios

### 4.2 Gouraud Shading

Compute lighting **at each vertex** using vertex normals, then **interpolate colors** across the triangle using barycentric coordinates.

- **Pros**: Smooth appearance for diffuse lighting, runs lighting calculation only per vertex
- **Cons**: Specular highlights can be missed or distorted if they fall between vertices
- **Use case**: Early GPUs (fixed-function era), very simple shading

### 4.3 Phong Shading (Interpolation)

Interpolate the **normals** across the triangle, then compute lighting **at each pixel** (fragment) using the interpolated normal.

> **Naming clarification**: "Phong shading" can refer to both the *reflection model* (Section 2) and this *interpolation method*. They were introduced in the same paper but are independent concepts. You can use Phong interpolation with Blinn-Phong's reflection model, for instance.

- **Pros**: Accurate specular highlights, smooth appearance even with coarse meshes
- **Cons**: More expensive (lighting computed per pixel instead of per vertex)
- **Use case**: All modern rendering (fragment shaders compute per-pixel lighting)

```python
def shade_triangle_flat(v0, v1, v2, light_pos, view_pos, light_color):
    """
    Flat shading: compute one color for the entire triangle.

    The face normal is the cross product of two edges.
    """
    edge1 = v1['pos'] - v0['pos']
    edge2 = v2['pos'] - v0['pos']
    face_normal = normalize(np.cross(edge1, edge2))
    centroid = (v0['pos'] + v1['pos'] + v2['pos']) / 3.0

    color = blinn_phong_shading(centroid, face_normal, view_pos,
                                 light_pos, light_color)
    return color  # Same color for every pixel in the triangle


def shade_triangle_gouraud(alpha, beta, gamma,
                            color_v0, color_v1, color_v2):
    """
    Gouraud shading: interpolate pre-computed vertex colors.

    Lighting is computed ONCE per vertex (before rasterization).
    During rasterization, we just interpolate the resulting colors.
    """
    return alpha * color_v0 + beta * color_v1 + gamma * color_v2


def shade_triangle_phong_interp(alpha, beta, gamma,
                                 normal_v0, normal_v1, normal_v2,
                                 frag_pos, view_pos, light_pos, light_color):
    """
    Phong interpolation: interpolate normals, compute lighting per pixel.

    This is more expensive than Gouraud but produces much better
    specular highlights. All modern GPUs do this in the fragment shader.
    """
    # Interpolate normal (must re-normalize after interpolation!)
    # Why re-normalize? Linear interpolation of unit vectors does NOT
    # produce a unit vector in general (the result is slightly shorter).
    interp_normal = alpha * normal_v0 + beta * normal_v1 + gamma * normal_v2
    interp_normal = normalize(interp_normal)

    return blinn_phong_shading(frag_pos, interp_normal, view_pos,
                                light_pos, light_color)
```

### 4.4 Visual Comparison

```
Flat Shading:           Gouraud Shading:        Phong Interpolation:
┌───┐                   ┌───────┐               ┌───────────┐
│   │ Each face         │ ░░▓▓▓ │ Colors        │  ░░░▓▓▓   │ Normals
│   │ = one color       │ ░░▓▓▓ │ smoothly      │ ░░▓▓▓▓▓░  │ smoothly
├───┤                   │ ░░░▓▓ │ interpolated  │ ░░▓▓●▓▓░  │ interpolated,
│   │ Visible           │ ░░░░▓ │               │ ░░▓▓▓▓▓░  │ lighting per
│   │ facets            └───────┘               │  ░░░▓▓▓   │ pixel -- sharp
└───┘                   Highlight may           └───────────┘ specular dot!
                        be missed!
```

---

## 5. Physically Based Rendering (PBR)

### 5.1 Motivation

The Phong/Blinn-Phong model has served well for decades, but it has fundamental limitations:
- The parameters ($k_a$, $k_d$, $k_s$, $n$) are not physically meaningful
- Materials do not conserve energy (reflected light can exceed incoming light)
- It cannot accurately represent metals, rough surfaces, or Fresnel effects
- Artists must re-tune parameters for every lighting setup

**PBR** addresses these issues by grounding the shading model in physics:
- **Energy conservation**: Reflected light never exceeds incident light
- **Fresnel effect**: All materials become more reflective at grazing angles
- **Microfacet theory**: Surface roughness modeled as statistical distribution of tiny mirrors
- **Two intuitive parameters**: roughness and metallicness

### 5.2 The Rendering Equation

PBR is based on the **rendering equation** (Kajiya, 1986):

$$L_o(\mathbf{p}, \omega_o) = L_e(\mathbf{p}, \omega_o) + \int_\Omega f_r(\mathbf{p}, \omega_i, \omega_o) \cdot L_i(\mathbf{p}, \omega_i) \cdot (\omega_i \cdot \mathbf{n}) \, d\omega_i$$

Where:
- $L_o$: outgoing radiance (what we see)
- $L_e$: emitted radiance (for light sources)
- $f_r$: the **BRDF** (Bidirectional Reflectance Distribution Function)
- $L_i$: incoming radiance from direction $\omega_i$
- $\omega_i \cdot \mathbf{n} = \cos\theta_i$: Lambert's cosine factor

For real-time rendering, we approximate the integral by summing over discrete light sources:

$$L_o \approx \sum_{\text{lights}} f_r(\omega_i, \omega_o) \cdot L_i \cdot (\mathbf{N} \cdot \mathbf{L})$$

### 5.3 The Cook-Torrance BRDF

The standard real-time PBR BRDF splits reflectance into diffuse and specular:

$$f_r = k_d \cdot f_{\text{Lambert}} + k_s \cdot f_{\text{Cook-Torrance}}$$

**Diffuse term** (Lambertian):

$$f_{\text{Lambert}} = \frac{\text{albedo}}{\pi}$$

The $\frac{1}{\pi}$ factor ensures energy conservation (a Lambertian surface distributes light over a hemisphere of solid angle $2\pi$, and the $\cos\theta$ weighting integrates to $\pi$).

**Specular term** (Cook-Torrance):

$$f_{\text{Cook-Torrance}} = \frac{D(\mathbf{H}) \cdot F(\mathbf{V}, \mathbf{H}) \cdot G(\mathbf{L}, \mathbf{V}, \mathbf{H})}{4 \cdot (\mathbf{N} \cdot \mathbf{L}) \cdot (\mathbf{N} \cdot \mathbf{V})}$$

Where:
- $D$: **Normal Distribution Function** (NDF) -- how many microfacets point toward $\mathbf{H}$
- $F$: **Fresnel function** -- reflectance that varies with viewing angle
- $G$: **Geometry function** -- microfacet self-shadowing and masking
- The denominator is a normalization factor

### 5.4 Normal Distribution Function: GGX / Trowbridge-Reitz

The GGX distribution (also called Trowbridge-Reitz) models the statistical distribution of microfacet orientations:

$$D_{\text{GGX}}(\mathbf{H}) = \frac{\alpha^2}{\pi \left[(\mathbf{N} \cdot \mathbf{H})^2 (\alpha^2 - 1) + 1\right]^2}$$

Where $\alpha = \text{roughness}^2$ (roughness is the artist-facing parameter in $[0, 1]$).

- At $\alpha \rightarrow 0$ (perfectly smooth): $D$ becomes a sharp spike (mirror)
- At $\alpha \rightarrow 1$ (perfectly rough): $D$ becomes a broad dome (matte)

```python
def distribution_ggx(n_dot_h, roughness):
    """
    GGX / Trowbridge-Reitz Normal Distribution Function.

    Models the probability that a microfacet's normal aligns with
    the halfway vector H. Rougher surfaces have more spread-out
    microfacet orientations, producing wider specular highlights.

    The alpha = roughness^2 remapping provides a more perceptually
    linear roughness slider for artists.
    """
    alpha = roughness * roughness
    a2 = alpha * alpha
    n_dot_h2 = n_dot_h * n_dot_h

    denom = n_dot_h2 * (a2 - 1.0) + 1.0
    denom = np.pi * denom * denom

    return a2 / max(denom, 1e-10)
```

### 5.5 Fresnel Equation: Schlick's Approximation

The **Fresnel effect** describes how surface reflectance increases at grazing angles. Look at a lake: directly below you, you see through the water; at a shallow angle, the surface becomes a mirror.

The exact Fresnel equations are complex. Schlick's approximation is accurate and fast:

$$F(\mathbf{V}, \mathbf{H}) = F_0 + (1 - F_0)(1 - \mathbf{V} \cdot \mathbf{H})^5$$

Where $F_0$ is the **base reflectance** at normal incidence (looking straight at the surface):
- **Dielectrics** (non-metals): $F_0 \approx 0.04$ (4% reflective head-on)
- **Metals**: $F_0$ = the metal's color (gold: $(1.0, 0.765, 0.336)$, iron: $(0.56, 0.57, 0.58)$)

```python
def fresnel_schlick(cos_theta, f0):
    """
    Schlick's approximation of the Fresnel equation.

    At normal incidence (cos_theta = 1): returns F0 (base reflectance)
    At grazing angle (cos_theta -> 0): returns 1.0 (fully reflective)

    This captures the universal physical phenomenon where all
    surfaces become more reflective when viewed at shallow angles.
    """
    return f0 + (1.0 - f0) * (1.0 - cos_theta) ** 5
```

### 5.6 Geometry Function: Smith GGX

The geometry function accounts for microfacet **self-shadowing** (incoming light blocked by other microfacets) and **masking** (reflected light blocked before reaching the viewer).

The Smith formulation separates these into two independent terms:

$$G(\mathbf{L}, \mathbf{V}) = G_1(\mathbf{L}) \cdot G_1(\mathbf{V})$$

Using the Schlick-GGX approximation:

$$G_1(\mathbf{X}) = \frac{\mathbf{N} \cdot \mathbf{X}}{(\mathbf{N} \cdot \mathbf{X})(1 - k) + k}$$

Where:
- For direct lighting: $k = \frac{(\alpha + 1)^2}{8}$
- For IBL (image-based lighting): $k = \frac{\alpha^2}{2}$

```python
def geometry_schlick_ggx(n_dot_v, roughness):
    """
    Schlick-GGX geometry function (one direction).

    Models self-occlusion of microfacets: on rough surfaces,
    microscopic peaks block light from reaching or leaving
    nearby valleys.
    """
    r = roughness + 1.0
    k = (r * r) / 8.0  # k for direct lighting

    return n_dot_v / (n_dot_v * (1.0 - k) + k)

def geometry_smith(n_dot_v, n_dot_l, roughness):
    """
    Smith's geometry function: combines shadowing and masking.

    G(L,V) = G1(L) * G1(V)
    Both the incoming light direction and outgoing view direction
    can be partially blocked by microfacet geometry.
    """
    ggx_v = geometry_schlick_ggx(max(n_dot_v, 0.0), roughness)
    ggx_l = geometry_schlick_ggx(max(n_dot_l, 0.0), roughness)
    return ggx_v * ggx_l
```

### 5.7 The Metallic-Roughness Workflow

Modern PBR uses two intuitive parameters:

| Parameter | Range | Effect |
|-----------|-------|--------|
| **Roughness** | $[0, 1]$ | 0 = mirror smooth, 1 = completely matte |
| **Metallic** | $[0, 1]$ | 0 = dielectric (plastic, wood), 1 = metal (gold, iron) |

The metallic parameter determines:
- $F_0$: `lerp(0.04, albedo, metallic)` -- metals use their color as base reflectance
- Diffuse contribution: `albedo * (1 - metallic)` -- metals have no diffuse component

```python
def pbr_shading(point, normal, view_pos, light_pos, light_color,
                albedo, metallic, roughness):
    """
    Complete PBR shading using the Cook-Torrance BRDF.

    This is the shading model used by Unity, Unreal Engine, Godot,
    and virtually every modern rendering engine.

    Parameters:
        point: surface position
        normal: surface normal (unit)
        view_pos: camera position
        light_pos: light position
        light_color: light RGB color/intensity
        albedo: base color of the surface (RGB)
        metallic: 0 = dielectric, 1 = metal
        roughness: 0 = mirror, 1 = matte
    """
    N = normalize(normal)
    V = normalize(view_pos - point)
    L = normalize(light_pos - point)
    H = normalize(V + L)

    # Dot products (clamped to non-negative)
    n_dot_l = max(np.dot(N, L), 0.0)
    n_dot_v = max(np.dot(N, V), 0.0)
    n_dot_h = max(np.dot(N, H), 0.0)
    v_dot_h = max(np.dot(V, H), 0.001)

    # Base reflectance: dielectrics use 0.04, metals use albedo
    f0 = np.full(3, 0.04)
    f0 = f0 * (1.0 - metallic) + albedo * metallic

    # Cook-Torrance specular BRDF components
    D = distribution_ggx(n_dot_h, roughness)
    F = fresnel_schlick(v_dot_h, f0)
    G = geometry_smith(n_dot_v, n_dot_l, roughness)

    # Specular contribution
    numerator = D * F * G
    denominator = 4.0 * n_dot_v * n_dot_l + 0.0001  # Prevent division by zero
    specular = numerator / denominator

    # Energy conservation: what's not reflected is refracted (diffuse)
    # Metals have no diffuse component (all light is reflected)
    ks = F  # Specular coefficient = Fresnel
    kd = (1.0 - ks) * (1.0 - metallic)

    # Diffuse: Lambertian (divided by pi for energy conservation)
    diffuse = kd * albedo / np.pi

    # Final color: (diffuse + specular) * light * cos_theta
    lo = (diffuse + specular) * light_color * n_dot_l

    # Tone mapping (Reinhard) -- compress HDR to displayable range
    lo = lo / (lo + 1.0)

    # Gamma correction (linear to sRGB)
    lo = np.power(np.clip(lo, 0.0, 1.0), 1.0 / 2.2)

    return lo
```

---

## 6. Comparing Phong and PBR on a Sphere

```python
"""
Render a sphere with both Phong and PBR shading for comparison.

The sphere is rendered analytically (ray-sphere intersection) rather
than via rasterization, to focus purely on the shading computation.
"""

import numpy as np

def render_sphere_comparison(width=400, height=200):
    """
    Render a sphere with Phong (left half) and PBR (right half).
    """
    image = np.zeros((height, width, 3), dtype=float)

    # Scene setup
    sphere_center = np.array([0.0, 0.0, -3.0])
    sphere_radius = 1.0
    light_pos = np.array([2.0, 3.0, 0.0])
    light_color = np.array([1.0, 1.0, 1.0]) * 3.0  # Bright white
    view_pos = np.array([0.0, 0.0, 0.0])

    # PBR material: slightly rough, non-metallic
    albedo = np.array([0.8, 0.2, 0.2])  # Red
    metallic = 0.0
    roughness = 0.4

    aspect = width / height

    for y in range(height):
        for x in range(width):
            # Ray from camera through pixel
            # Map pixel to [-1, 1] range
            u = (2.0 * x / width - 1.0) * aspect
            v = 1.0 - 2.0 * y / height
            ray_dir = normalize(np.array([u, v, -1.0]))

            # Ray-sphere intersection
            oc = view_pos - sphere_center
            a = np.dot(ray_dir, ray_dir)
            b = 2.0 * np.dot(oc, ray_dir)
            c = np.dot(oc, oc) - sphere_radius ** 2
            discriminant = b * b - 4 * a * c

            if discriminant < 0:
                image[y, x] = [0.05, 0.05, 0.08]  # Background
                continue

            t = (-b - np.sqrt(discriminant)) / (2 * a)
            if t < 0:
                image[y, x] = [0.05, 0.05, 0.08]
                continue

            # Hit point and normal
            hit_point = view_pos + t * ray_dir
            normal = normalize(hit_point - sphere_center)

            # Left half: Phong shading
            if x < width // 2:
                color = phong_shading(
                    hit_point, normal, view_pos, light_pos,
                    light_color, ka=0.1, kd=0.7, ks=0.5,
                    shininess=64.0, object_color=albedo
                )
            # Right half: PBR shading
            else:
                color = pbr_shading(
                    hit_point, normal, view_pos, light_pos,
                    light_color, albedo=albedo,
                    metallic=metallic, roughness=roughness
                )

            image[y, x] = color

    return image


def render_material_grid(width=600, height=400):
    """
    Render a grid of spheres with varying metallic and roughness values.

    This demonstrates how PBR's two parameters intuitively control appearance:
    - Rows: roughness from 0.1 (top, smooth) to 0.9 (bottom, rough)
    - Columns: metallic from 0 (left, dielectric) to 1 (right, metal)
    """
    image = np.zeros((height, width, 3), dtype=float)

    rows, cols = 5, 5
    sphere_radius = 0.35
    light_pos = np.array([5.0, 5.0, 5.0])
    light_color = np.array([1.0, 1.0, 1.0]) * 5.0
    view_pos = np.array([0.0, 0.0, 0.0])
    albedo = np.array([0.9, 0.6, 0.2])  # Gold-like base color

    for row in range(rows):
        for col in range(cols):
            roughness = 0.1 + 0.8 * row / (rows - 1)
            metallic = col / (cols - 1)

            # Center of this sphere on screen
            cx = (col + 0.5) / cols * width
            cy = (row + 0.5) / rows * height
            pix_radius = min(width / cols, height / rows) * 0.4

            # Render this sphere
            for dy in range(int(-pix_radius), int(pix_radius) + 1):
                for dx in range(int(-pix_radius), int(pix_radius) + 1):
                    px = int(cx + dx)
                    py = int(cy + dy)
                    if not (0 <= px < width and 0 <= py < height):
                        continue

                    # Map pixel offset to sphere surface
                    nx = dx / pix_radius
                    ny = -dy / pix_radius
                    r2 = nx * nx + ny * ny
                    if r2 > 1.0:
                        continue

                    nz = np.sqrt(1.0 - r2)
                    normal = np.array([nx, ny, nz])
                    point = np.array([nx, ny, nz - 3.0])

                    color = pbr_shading(
                        point, normal, view_pos, light_pos,
                        light_color, albedo=albedo,
                        metallic=metallic, roughness=roughness
                    )
                    image[py, px] = color

    return image


if __name__ == "__main__":
    print("Rendering Phong vs PBR comparison...")
    comparison = render_sphere_comparison()
    print(f"Comparison image shape: {comparison.shape}")

    print("Rendering material grid (roughness x metallic)...")
    grid = render_material_grid()
    print(f"Grid image shape: {grid.shape}")

    try:
        from PIL import Image
        img1 = Image.fromarray((comparison * 255).astype(np.uint8))
        img1.save('phong_vs_pbr.png')
        print("Saved phong_vs_pbr.png")

        img2 = Image.fromarray((grid * 255).astype(np.uint8))
        img2.save('material_grid.png')
        print("Saved material_grid.png")
    except ImportError:
        print("Install Pillow for image output: pip install Pillow")
```

---

## 7. Tone Mapping and Gamma Correction

### 7.1 Why Tone Mapping?

PBR operates in **linear, High Dynamic Range (HDR)** space. Light intensities can be arbitrarily large (the sun is millions of times brighter than a candle). But displays can only show values in $[0, 1]$. **Tone mapping** compresses HDR values to the displayable range.

**Reinhard tone mapping** (simple):

$$L_{\text{display}} = \frac{L}{L + 1}$$

**ACES filmic tone mapping** (industry standard):

$$f(x) = \frac{x(2.51x + 0.03)}{x(2.43x + 0.59) + 0.14}$$

### 7.2 Gamma Correction

Displays are non-linear: they apply a gamma curve ($\text{output} = \text{input}^{2.2}$). To compensate, we apply the inverse gamma before sending colors to the display:

$$C_{\text{sRGB}} = C_{\text{linear}}^{1/2.2}$$

**PBR calculations must always be done in linear space**. Textures stored in sRGB must be converted to linear on input, and the result must be converted back to sRGB on output.

---

## 8. Multiple Lights and Light Types

### 8.1 Summing Light Contributions

For multiple lights, simply sum the contributions (the rendering equation is linear in incoming radiance):

$$L_o = \sum_{i=1}^{n} f_r(\omega_i, \omega_o) \cdot L_i \cdot (\mathbf{N} \cdot \mathbf{L}_i)$$

### 8.2 Light Types

| Light Type | Description | Attenuation |
|-----------|-------------|-------------|
| **Directional** | Infinitely far (sun) | None (constant) |
| **Point** | Emits in all directions | $\frac{1}{d^2}$ (inverse square law) |
| **Spot** | Point with angular cutoff | $\frac{1}{d^2} \cdot \text{cone\_factor}$ |
| **Area** | Finite surface (realistic) | Complex (requires sampling or LTC) |

```python
def point_light_attenuation(light_pos, frag_pos):
    """
    Inverse-square attenuation for point lights.

    This follows the physical law: light intensity decreases with
    the square of distance. A light twice as far away is 1/4 as bright.
    """
    distance = np.linalg.norm(light_pos - frag_pos)
    return 1.0 / (distance * distance + 0.0001)
```

---

## Summary

| Model | Type | Parameters | Energy Conserving? | Use Case |
|-------|------|------------|-------------------|----------|
| **Phong** | Empirical | $k_a, k_d, k_s, n$ | No | Learning, simple applications |
| **Blinn-Phong** | Empirical | Same, uses $\mathbf{H}$ | No | Legacy engines |
| **Cook-Torrance PBR** | Physical | albedo, roughness, metallic | Yes | All modern engines |

| Interpolation | Computes Lighting | Quality | Cost |
|--------------|-------------------|---------|------|
| **Flat** | Once per face | Faceted | Cheapest |
| **Gouraud** | Per vertex | Smooth (misses specular) | Moderate |
| **Phong interp.** | Per pixel | Best (accurate specular) | Highest |

**Key takeaways**:
- The Phong model's three components (ambient + diffuse + specular) capture the essential visual elements
- Blinn-Phong's halfway vector is both more efficient and more physically motivated
- PBR (Cook-Torrance) uses microfacet theory with three functions: D (distribution), F (Fresnel), G (geometry)
- The metallic-roughness workflow gives artists two intuitive parameters for all materials
- All PBR calculations must be done in linear color space with proper tone mapping and gamma correction
- Phong interpolation (per-pixel normals) is standard for all modern per-pixel lighting

---

## Exercises

1. **Phong Components**: Render a sphere three times, showing only the ambient, diffuse, and specular components separately. Then show the combined result. How does changing the shininess exponent $n$ affect each component?

2. **Blinn vs Phong Specular**: Render a sphere with both Phong and Blinn-Phong specular at the same shininess. Observe the difference in highlight size. What Blinn-Phong exponent approximately matches a Phong exponent of 64?

3. **Shading Comparison**: Render a low-polygon sphere (20 faces) with flat, Gouraud, and Phong interpolation. At what polygon count does the Gouraud-shaded sphere begin to look acceptably smooth?

4. **PBR Material Exploration**: Create a 5x5 grid of spheres varying roughness (0.0 to 1.0, rows) and metallic (0.0 to 1.0, columns). Use a gold-colored albedo. Explain why fully metallic, fully smooth spheres look mirror-like while fully metallic, fully rough spheres look like brushed metal.

5. **Fresnel Effect**: Plot the Schlick Fresnel function for $F_0 = 0.04$ (dielectric) and $F_0 = 0.9$ (metal) for $\cos\theta$ from 0 to 1. At what angle does a dielectric surface become 50% reflective?

6. **Energy Conservation**: For the Phong model with $k_d = 0.8$ and $k_s = 0.5$, show that the total reflected light can exceed the incoming light (violating energy conservation). Then show that the PBR model does not have this problem.

---

## Further Reading

1. Phong, B.T. "Illumination for Computer Generated Pictures" (1975) -- The original paper
2. Burley, B. "Physically Based Shading at Disney" (SIGGRAPH 2012) -- Foundation of modern PBR
3. [Learn OpenGL - PBR Theory](https://learnopengl.com/PBR/Theory) -- Excellent practical PBR tutorial
4. [Filament PBR Documentation](https://google.github.io/filament/Filament.html) -- Google's comprehensive PBR reference
5. Hoffman, N. "Background: Physics and Math of Shading" in *Real-Time Rendering* (4th ed.), Ch. 9
