# 11. Path Tracing and Global Illumination

[← Previous: Ray Tracing Basics](10_Ray_Tracing_Basics.md) | [Next: Animation and Skeletal Systems →](12_Animation_and_Skeletal_Systems.md)

---

## Learning Objectives

1. Understand the rendering equation (Kajiya 1986) as the fundamental equation of light transport
2. Apply Monte Carlo integration to estimate integrals with random sampling
3. Implement naive path tracing as random walks of light through a scene
4. Use importance sampling (cosine-weighted, BRDF-proportional) to reduce variance
5. Explain Russian roulette as an unbiased path termination strategy
6. Implement next event estimation (direct light sampling) for faster convergence
7. Understand multiple importance sampling (MIS) and its benefits
8. Recognize denoising as a practical necessity and survey current approaches

---

## Why This Matters

Whitted-style ray tracing (L10) handles mirror reflections and sharp shadows beautifully, but the real world is full of **soft** effects: color bleeding from a red wall onto a white ceiling, diffuse inter-reflections that brighten corners, caustics from light focused through a glass. These effects arise from light bouncing off *all* surfaces in *all* directions -- what we call **global illumination**.

Path tracing is the algorithm that faithfully simulates this full light transport. It is the foundation of every major offline renderer used in film (Arnold, RenderMan, Cycles, Manuka) and increasingly appears in real-time engines. When you see a photorealistic CG image, path tracing almost certainly produced it.

---

## 1. The Rendering Equation

### 1.1 Formulation

James Kajiya introduced the **rendering equation** in 1986. It describes the total outgoing radiance $L_o$ from a point $\mathbf{p}$ in direction $\omega_o$:

$$L_o(\mathbf{p}, \omega_o) = L_e(\mathbf{p}, \omega_o) + \int_{\Omega} f_r(\mathbf{p}, \omega_i, \omega_o) \, L_i(\mathbf{p}, \omega_i) \, (\omega_i \cdot \mathbf{n}) \, d\omega_i$$

where:
- $L_e(\mathbf{p}, \omega_o)$ is the **emitted** radiance (non-zero only for light sources)
- $f_r(\mathbf{p}, \omega_i, \omega_o)$ is the **BRDF** (Bidirectional Reflectance Distribution Function) at $\mathbf{p}$
- $L_i(\mathbf{p}, \omega_i)$ is the **incoming** radiance from direction $\omega_i$
- $\omega_i \cdot \mathbf{n} = \cos\theta_i$ is the cosine foreshortening term
- $\Omega$ is the hemisphere of directions above the surface at $\mathbf{p}$

**The key insight**: $L_i$ at one point depends on $L_o$ at another point, making this a recursive integral equation. There is no closed-form solution for general scenes -- we must solve it numerically.

### 1.2 Understanding Each Term

**Emission** $L_e$: Light sources emit radiance. For non-emissive surfaces, $L_e = 0$.

**BRDF** $f_r$: Describes how light arriving from $\omega_i$ is scattered toward $\omega_o$. Examples:
- **Lambertian (diffuse)**: $f_r = \frac{\rho}{\pi}$ where $\rho$ is the albedo
- **Perfect mirror**: A Dirac delta -- all light goes in the reflection direction
- **Glossy**: Concentrated around the reflection direction (e.g., Cook-Torrance microfacet model)

**Cosine term** $\cos\theta_i$: Light arriving at a grazing angle illuminates less area, so its contribution is reduced.

### 1.3 Why Not Just Integrate?

The integral is over a hemisphere of directions, each of which requires tracing a ray to find what is visible, which gives $L_i$ -- which is itself defined by the same integral at the next surface. This infinite recursion is what makes global illumination hard. Path tracing handles it via **Monte Carlo estimation**.

---

## 2. Monte Carlo Integration

### 2.1 Basic Idea

To estimate an integral $I = \int_a^b f(x)\,dx$, draw $N$ random samples $x_k$ uniformly from $[a, b]$:

$$I \approx \frac{b - a}{N} \sum_{k=1}^{N} f(x_k)$$

This **Monte Carlo estimator** converges to the true integral as $N \to \infty$, with error proportional to $\frac{1}{\sqrt{N}}$. Doubling the accuracy requires quadrupling the samples.

### 2.2 General Estimator with PDF

For samples drawn from a probability density function $p(x)$ (not necessarily uniform):

$$I = \int f(x)\,dx \approx \frac{1}{N} \sum_{k=1}^{N} \frac{f(x_k)}{p(x_k)}$$

This is valid for any $p(x) > 0$ wherever $f(x) \neq 0$. The art is choosing $p$ to reduce **variance** (noise).

### 2.3 Variance and Convergence

The variance of the Monte Carlo estimator is:

$$\text{Var}\left[\frac{f(X)}{p(X)}\right] = \int \left(\frac{f(x)}{p(x)} - I\right)^2 p(x)\,dx$$

Variance is minimized when $p(x) \propto f(x)$ -- this is the principle behind **importance sampling**. If $p$ matches $f$ perfectly, the variance is zero (one sample suffices!).

In practice, we cannot perfectly match $p$ to the unknown integrand, but we can use analytical parts of the integrand (e.g., the cosine term or the BRDF) to guide sampling.

---

## 3. Naive Path Tracing

### 3.1 Algorithm

The simplest path tracer estimates the rendering equation by randomly bouncing rays through the scene:

```
function path_trace(ray, depth):
    if depth > MAX_DEPTH:
        return BLACK

    hit = find_nearest_intersection(ray)
    if no hit:
        return BACKGROUND

    p, n, material = hit

    // Emission: if we hit a light, collect it
    color = material.emission

    // Sample a random direction on the hemisphere
    wi = random_hemisphere_direction(n)

    // Evaluate the rendering equation estimator
    // f(x)/p(x) where f = BRDF * Li * cos(theta), p = 1/(2*pi)
    cos_theta = dot(wi, n)
    brdf = material.brdf(wi, wo)

    // Recurse: trace the bounced ray
    Li = path_trace(Ray(p + epsilon*n, wi), depth + 1)

    color += 2 * pi * brdf * Li * cos_theta

    return color
```

For a Lambertian surface with albedo $\rho$, the BRDF is $\frac{\rho}{\pi}$. With uniform hemisphere sampling ($p(\omega) = \frac{1}{2\pi}$):

$$\text{color} \approx L_e + \frac{f_r \cdot L_i \cdot \cos\theta}{p(\omega)} = L_e + \frac{\frac{\rho}{\pi} \cdot L_i \cdot \cos\theta}{\frac{1}{2\pi}} = L_e + 2\rho \cdot L_i \cdot \cos\theta$$

### 3.2 The Noise Problem

Naive path tracing works but converges slowly. At each bounce, a single random direction is chosen. Many of these directions point toward dark parts of the scene, contributing little information. The result is an extremely **noisy** image that requires hundreds or thousands of samples per pixel to clean up.

---

## 4. Importance Sampling

### 4.1 Cosine-Weighted Hemisphere Sampling

Since the rendering equation includes a $\cos\theta$ factor, we can build this into our sampling PDF. Instead of sampling uniformly over the hemisphere, we sample proportionally to $\cos\theta$:

$$p(\omega) = \frac{\cos\theta}{\pi}$$

This is generated by sampling two uniform random numbers $(\xi_1, \xi_2) \in [0, 1)^2$:

$$\phi = 2\pi\xi_1, \quad \theta = \arccos\sqrt{1 - \xi_2}$$

Or equivalently, using the disk-to-hemisphere mapping:

$$x = \cos\phi\sqrt{\xi_2}, \quad y = \sin\phi\sqrt{\xi_2}, \quad z = \sqrt{1 - \xi_2}$$

The $z$ component is aligned with the surface normal.

With this PDF, the Monte Carlo estimator for Lambertian surfaces simplifies:

$$\frac{f_r \cdot L_i \cdot \cos\theta}{p(\omega)} = \frac{\frac{\rho}{\pi} \cdot L_i \cdot \cos\theta}{\frac{\cos\theta}{\pi}} = \rho \cdot L_i$$

The cosine term cancels, reducing variance dramatically.

### 4.2 BRDF-Proportional Sampling

For glossy BRDFs (e.g., GGX microfacet), sample directions proportional to the BRDF itself. The GGX distribution function has known analytical sampling formulas, so the heavy lobe of the BRDF is sampled more frequently.

### 4.3 Comparison

| Sampling Strategy | PDF $p(\omega)$ | Variance |
|-------------------|-----------------|----------|
| Uniform hemisphere | $1/(2\pi)$ | High |
| Cosine-weighted | $\cos\theta / \pi$ | Lower (cancels cosine in estimator) |
| BRDF-proportional | $\propto f_r \cdot \cos\theta$ | Lowest for BRDF term |

---

## 5. Russian Roulette

### 5.1 Motivation

Path tracing paths could bounce infinitely. Setting a fixed maximum depth introduces **bias** (we miss energy from deeper bounces). **Russian roulette** provides an unbiased termination:

At each bounce, terminate the path with probability $q$. If the path survives (probability $1 - q$), boost the contribution by $\frac{1}{1 - q}$:

$$L \approx \begin{cases} \frac{L_{\text{bounce}}}{1 - q} & \text{with probability } 1 - q \\ 0 & \text{with probability } q \end{cases}$$

The expected value is:

$$E[L] = (1 - q) \cdot \frac{L_{\text{bounce}}}{1 - q} + q \cdot 0 = L_{\text{bounce}}$$

This is **unbiased**: on average, we get the correct answer. Paths that contribute less (e.g., after hitting dark surfaces) can be terminated more aggressively.

### 5.2 Choosing $q$

A common heuristic: set survival probability proportional to the surface albedo (or maximum BRDF response):

$$q_{\text{terminate}} = 1 - \min(\max(\rho_r, \rho_g, \rho_b), 0.95)$$

This means bright surfaces almost always continue bouncing (they carry significant energy), while dark surfaces terminate early.

---

## 6. Next Event Estimation (Direct Light Sampling)

### 6.1 Problem

In naive path tracing, the only way to collect light is if a randomly bounced ray happens to hit a light source. For small lights, this is extremely unlikely -- the image is noisy.

### 6.2 Solution

At each hit point, explicitly sample the light sources:

1. Choose a point $\mathbf{q}$ on a light source (e.g., random point on an area light)
2. Cast a **shadow ray** from $\mathbf{p}$ to $\mathbf{q}$
3. If unoccluded, add the direct illumination contribution

This is called **Next Event Estimation (NEE)** or **direct light sampling**. The contribution from a light with area $A$ is:

$$L_{\text{direct}} \approx \frac{f_r(\omega_i, \omega_o) \cdot L_e(\mathbf{q}) \cdot \cos\theta_i \cdot \cos\theta_q}{|\mathbf{p} - \mathbf{q}|^2 \cdot p(\mathbf{q})}$$

where $\cos\theta_q$ accounts for the light's orientation and $p(\mathbf{q}) = 1/A$ for uniform sampling.

**Important**: When using NEE, the indirect bounce should **not** collect emission from lights directly (to avoid double-counting). The bounce handles only indirect illumination.

---

## 7. Multiple Importance Sampling (MIS)

### 7.1 The Problem

BRDF sampling is good when the material is glossy (narrow lobe), but bad for small lights. Light sampling is good for small lights, but bad for sharp specular surfaces. Neither strategy is universally best.

### 7.2 MIS Solution

Veach and Guibas (1995) showed how to combine multiple sampling strategies with **balance heuristic** weights:

$$F \approx \sum_{k=1}^{N_{\text{BRDF}}} \frac{w_{\text{BRDF}}(\omega_k) \cdot f_r \cdot L_i \cdot \cos\theta}{p_{\text{BRDF}}(\omega_k)} + \sum_{k=1}^{N_{\text{light}}} \frac{w_{\text{light}}(\omega_k) \cdot f_r \cdot L_i \cdot \cos\theta}{p_{\text{light}}(\omega_k)}$$

The **balance heuristic** weight for BRDF sampling is:

$$w_{\text{BRDF}}(\omega) = \frac{p_{\text{BRDF}}(\omega)}{p_{\text{BRDF}}(\omega) + p_{\text{light}}(\omega)}$$

MIS is provably close to optimal -- it automatically favors whichever strategy is better for each particular sample. It is used in virtually all production path tracers.

### 7.3 Power Heuristic

In practice, the **power heuristic** (with exponent $\beta = 2$) performs even better:

$$w_{\text{BRDF}}(\omega) = \frac{p_{\text{BRDF}}(\omega)^2}{p_{\text{BRDF}}(\omega)^2 + p_{\text{light}}(\omega)^2}$$

---

## 8. Convergence and Noise

### 8.1 Error Analysis

Monte Carlo path tracing converges as $O(1/\sqrt{N})$ where $N$ is the number of samples per pixel. This means:

| Samples per Pixel (spp) | Relative Noise |
|--------------------------|----------------|
| 1 | 1.00 |
| 4 | 0.50 |
| 16 | 0.25 |
| 64 | 0.125 |
| 256 | 0.0625 |
| 1024 | 0.03125 |

Halving the noise requires $4\times$ the samples. Production renders typically use 128-4096 spp.

### 8.2 Sources of Noise

- **Low probability paths**: Caustics (light focused through glass) are notoriously difficult because the light reaches the camera via unlikely paths
- **Small lights**: A small bright light contributes high variance (large value when hit, zero otherwise)
- **Glossy inter-reflections**: Multiple bounces between glossy surfaces compound variance

### 8.3 Variance Reduction Techniques Summary

| Technique | Mechanism |
|-----------|-----------|
| Importance sampling | Match sampling PDF to integrand |
| Russian roulette | Unbiased path termination |
| Next event estimation | Explicitly sample lights |
| MIS | Combine multiple strategies optimally |
| Stratified sampling | Divide sample space into strata |
| Quasi-Monte Carlo | Low-discrepancy sequences (Halton, Sobol) |

---

## 9. Denoising

### 9.1 Why Denoise?

Even with all the variance reduction techniques above, path-traced images at low sample counts are noisy. **Denoising** filters recover a clean image from noisy input, effectively allowing production-quality results at lower sample counts.

### 9.2 Classical Approaches

**Bilateral filter**: Averages nearby pixels, but weights by both spatial distance and color similarity. Preserves edges while smoothing noise:

$$\hat{I}(\mathbf{p}) = \frac{1}{W} \sum_{\mathbf{q}} I(\mathbf{q}) \cdot G_{\sigma_s}(\|\mathbf{p} - \mathbf{q}\|) \cdot G_{\sigma_r}(\|I(\mathbf{p}) - I(\mathbf{q})\|)$$

**A-trous wavelet filter**: Multi-scale bilateral filter using increasing dilation. Used in real-time denoisers (SVGF).

### 9.3 Machine Learning Denoisers

Modern denoisers use neural networks trained on pairs of (noisy, reference) images:

- **NVIDIA OptiX AI Denoiser**: Trained CNN; takes noisy color, albedo, and normal buffers as input
- **Intel Open Image Denoise (OIDN)**: Open-source ML denoiser; works without GPU ray tracing
- **SVGF** (Spatiotemporal Variance-Guided Filtering): Uses temporal reprojection and per-pixel variance estimation for real-time denoising

These denoisers use **auxiliary buffers** (normals, albedo, depth) to guide filtering:
- Noise-free normals tell the denoiser where geometry edges are
- Albedo separates texture detail from lighting noise
- Depth helps distinguish foreground from background

### 9.4 Denoising in Production

Film renderers (Arnold, Manuka) render at 64-256 spp and apply ML denoising, saving hours of compute time. Real-time ray tracing (1-4 spp) relies even more heavily on temporal accumulation and ML denoising.

---

## 10. Python Implementation: Monte Carlo Path Tracer

```python
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

# --- Utilities ---

def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def random_cosine_hemisphere(normal):
    """
    Sample a direction on the hemisphere proportional to cos(theta).
    Uses Malley's method: sample uniform disk, then project to hemisphere.
    This ensures more samples near the normal (where cos(theta) is large),
    reducing variance compared to uniform hemisphere sampling.
    """
    # Build local coordinate frame around the normal
    if abs(normal[0]) > 0.9:
        tangent = normalize(np.cross(np.array([0, 1, 0]), normal))
    else:
        tangent = normalize(np.cross(np.array([1, 0, 0]), normal))
    bitangent = np.cross(normal, tangent)

    # Random point on unit disk
    xi1, xi2 = np.random.random(), np.random.random()
    phi = 2.0 * np.pi * xi1
    r = np.sqrt(xi2)  # Why sqrt: ensures uniform distribution on disk
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    z = np.sqrt(max(0.0, 1.0 - xi2))  # Project up to hemisphere

    # Transform to world space
    return x * tangent + y * bitangent + z * normal


# --- Scene Objects ---

@dataclass
class PTMaterial:
    albedo: np.ndarray          # Diffuse color
    emission: np.ndarray = None  # Emissive color (for lights)

    def __post_init__(self):
        if self.emission is None:
            self.emission = np.zeros(3)


@dataclass
class PTSphere:
    center: np.ndarray
    radius: float
    material: PTMaterial

    def intersect(self, origin, direction):
        oc = origin - self.center
        a = np.dot(direction, direction)
        b = 2.0 * np.dot(oc, direction)
        c = np.dot(oc, oc) - self.radius ** 2
        disc = b * b - 4 * a * c
        if disc < 0:
            return float('inf'), None
        sq = np.sqrt(disc)
        t = (-b - sq) / (2 * a)
        if t < 1e-4:
            t = (-b + sq) / (2 * a)
        if t < 1e-4:
            return float('inf'), None
        hit = origin + t * direction
        n = (hit - self.center) / self.radius
        return t, n


@dataclass
class PTPlane:
    point: np.ndarray
    normal: np.ndarray
    material: PTMaterial

    def intersect(self, origin, direction):
        denom = np.dot(self.normal, direction)
        if abs(denom) < 1e-8:
            return float('inf'), None
        t = np.dot(self.point - origin, self.normal) / denom
        if t < 1e-4:
            return float('inf'), None
        return t, self.normal.copy()


# --- Path Tracer ---

class PathTracer:
    """
    Monte Carlo path tracer with:
    - Cosine-weighted importance sampling
    - Russian roulette for unbiased termination
    - Next Event Estimation (direct light sampling)
    """

    def __init__(self, objects, max_depth=10):
        self.objects = objects
        self.max_depth = max_depth
        # Identify emissive objects for direct light sampling
        self.lights = [obj for obj in objects
                       if np.any(obj.material.emission > 0)]

    def find_nearest(self, origin, direction):
        nearest_t = float('inf')
        nearest_obj = None
        nearest_normal = None
        for obj in self.objects:
            t, n = obj.intersect(origin, direction)
            if t < nearest_t:
                nearest_t = t
                nearest_obj = obj
                nearest_normal = n
        return nearest_t, nearest_obj, nearest_normal

    def sample_light(self, hit_point, normal):
        """
        Next Event Estimation: sample a point on a light source and
        compute the direct illumination contribution.
        """
        if not self.lights:
            return np.zeros(3)

        # Randomly pick one light (uniform selection)
        light = self.lights[np.random.randint(len(self.lights))]

        # Sample a random point on the light sphere
        # Why random direction: approximates area light sampling
        rand_dir = normalize(np.random.randn(3))
        light_point = light.center + light.radius * rand_dir

        # Direction from hit point to light sample
        to_light = light_point - hit_point
        dist2 = np.dot(to_light, to_light)
        dist = np.sqrt(dist2)
        light_dir = to_light / dist

        # Check if light is above the surface
        cos_theta = np.dot(normal, light_dir)
        if cos_theta <= 0:
            return np.zeros(3)

        # Cosine at the light surface
        light_normal = normalize(light_point - light.center)
        cos_light = abs(np.dot(light_normal, -light_dir))

        # Shadow test
        shadow_t, shadow_obj, _ = self.find_nearest(
            hit_point + 1e-4 * normal, light_dir
        )
        if shadow_obj is not light or shadow_t > dist + 0.01:
            if shadow_t < dist - 0.01:
                return np.zeros(3)  # Occluded

        # Light area (sphere surface = 4*pi*r^2)
        light_area = 4.0 * np.pi * light.radius ** 2

        # Contribution: Le * BRDF * cos_theta * cos_light * area / dist^2
        # BRDF for Lambertian = albedo / pi
        # Multiplied by number of lights for unbiased estimator
        brdf = 1.0 / np.pi
        contribution = (light.material.emission * brdf * cos_theta
                       * cos_light * light_area / dist2)

        return contribution * len(self.lights)

    def trace(self, origin, direction, depth=0):
        """
        Trace a single path through the scene.
        Returns the estimated radiance along this path.
        """
        t, obj, normal = self.find_nearest(origin, direction)
        if obj is None:
            return np.zeros(3)  # No hit -> black background

        hit_point = origin + t * direction

        # Ensure normal faces the incoming ray
        if np.dot(normal, direction) > 0:
            normal = -normal

        # Collect emission (only on first bounce to avoid double-counting with NEE)
        if depth == 0:
            radiance = obj.material.emission.copy()
        else:
            radiance = np.zeros(3)

        # Russian roulette termination
        albedo = obj.material.albedo
        # Why max component: ensures bright channels survive with high probability
        survival_prob = min(max(albedo[0], albedo[1], albedo[2]), 0.95)
        if depth > 2:
            if np.random.random() > survival_prob:
                return radiance
        else:
            survival_prob = 1.0

        # Direct illumination via Next Event Estimation
        direct = self.sample_light(hit_point, normal)
        radiance += albedo * direct

        # Indirect illumination: sample a bounce direction
        bounce_dir = random_cosine_hemisphere(normal)

        # Why we don't include cos/pi here: cosine-weighted sampling cancels them
        # For Lambertian: (albedo/pi) * Li * cos(theta) / (cos(theta)/pi) = albedo * Li
        bounce_radiance = self.trace(
            hit_point + 1e-4 * normal, bounce_dir, depth + 1
        )

        # Add indirect contribution, compensated by Russian roulette survival
        radiance += albedo * bounce_radiance / survival_prob

        return radiance

    def render(self, width, height, spp=32, fov_deg=60.0,
               eye=np.array([0.0, 1.0, 4.0]),
               target=np.array([0.0, 0.5, 0.0])):
        """Render the scene with multiple samples per pixel."""
        image = np.zeros((height, width, 3))
        fov = np.radians(fov_deg)
        aspect = width / height

        forward = normalize(target - eye)
        right = normalize(np.cross(forward, np.array([0, 1, 0])))
        up = np.cross(right, forward)
        half_h = np.tan(fov / 2)
        half_w = half_h * aspect

        for j in range(height):
            for i in range(width):
                pixel_color = np.zeros(3)

                for s in range(spp):
                    # Jittered sampling: random offset within the pixel
                    u = (2 * (i + np.random.random()) / width - 1) * half_w
                    v = (1 - 2 * (j + np.random.random()) / height) * half_h
                    direction = normalize(forward + u * right + v * up)

                    pixel_color += self.trace(eye, direction)

                image[j, i] = pixel_color / spp

            if (j + 1) % max(1, height // 10) == 0:
                print(f"  Row {j+1}/{height} ({100*(j+1)//height}%)")

        return image


# --- Build Cornell-Box-Like Scene ---

white   = PTMaterial(albedo=np.array([0.73, 0.73, 0.73]))
red     = PTMaterial(albedo=np.array([0.65, 0.05, 0.05]))
green   = PTMaterial(albedo=np.array([0.12, 0.45, 0.15]))
light_m = PTMaterial(albedo=np.array([0.0, 0.0, 0.0]),
                     emission=np.array([15.0, 15.0, 15.0]))

objects = [
    # Floor
    PTPlane(np.array([0, 0, 0]), np.array([0, 1, 0]), white),
    # Ceiling
    PTPlane(np.array([0, 3, 0]), np.array([0, -1, 0]), white),
    # Back wall
    PTPlane(np.array([0, 0, -2]), np.array([0, 0, 1]), white),
    # Left wall (red)
    PTPlane(np.array([-2, 0, 0]), np.array([1, 0, 0]), red),
    # Right wall (green)
    PTPlane(np.array([2, 0, 0]), np.array([-1, 0, 0]), green),
    # Spheres
    PTSphere(np.array([-0.7, 0.5, -0.5]), 0.5, white),
    PTSphere(np.array([0.7, 0.8, 0.2]), 0.8, white),
    # Light source (small sphere on ceiling)
    PTSphere(np.array([0, 2.8, -0.5]), 0.3, light_m),
]

# --- Render ---
pt = PathTracer(objects, max_depth=8)
print("Path tracing 200x150 @ 64 spp...")
image = pt.render(200, 150, spp=64,
                  eye=np.array([0.0, 1.5, 5.0]),
                  target=np.array([0.0, 1.0, 0.0]))

# Tone mapping: simple Reinhard + gamma correction
# Why tone mapping: path tracer output is HDR; must compress for display
image = image / (1.0 + image)          # Reinhard tone mapping
image = np.power(np.clip(image, 0, 1), 1.0 / 2.2)  # Gamma correction

try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 7.5))
    plt.imshow(image)
    plt.axis('off')
    plt.title('Path Tracer (64 spp, cosine sampling + NEE + Russian roulette)')
    plt.tight_layout()
    plt.savefig('path_trace_result.png', dpi=150)
    plt.close()
    print("Saved path_trace_result.png")
except ImportError:
    print("Install matplotlib to save the image")
```

---

## 11. Advanced Topics (Brief Overview)

### 11.1 Bidirectional Path Tracing (BDPT)

Trace paths from both the camera and the light, then connect them. Efficiently handles scenes where light follows complex paths (e.g., indoor scenes lit by a small window).

### 11.2 Metropolis Light Transport (MLT)

Uses Markov Chain Monte Carlo (MCMC) to explore path space. Excels at finding and exploiting difficult light paths (caustics, light through a keyhole).

### 11.3 Photon Mapping

First pass: trace photons from lights and store them in a spatial hash. Second pass: gather nearby photons at camera-visible hit points. Good for caustics and volumetric effects.

### 11.4 Spectral Rendering

Instead of RGB, simulate light at individual wavelengths. Required for accurate dispersion (rainbow through a prism), fluorescence, and polarization effects.

---

## Summary

| Concept | Key Idea |
|---------|----------|
| Rendering equation | $L_o = L_e + \int f_r L_i \cos\theta\,d\omega$ -- the complete description of light transport |
| Monte Carlo | Estimate integrals by random sampling; error $\propto 1/\sqrt{N}$ |
| Path tracing | Random walk: bounce rays through the scene; each path samples the rendering equation |
| Importance sampling | Sample where the integrand is large; cosine-weighted for diffuse, BRDF for glossy |
| Russian roulette | Unbiased path termination; compensate by $1/(1-q)$ |
| Next event estimation | Explicitly sample lights at each bounce; dramatically reduces noise |
| MIS | Combine multiple strategies optimally; balance heuristic |
| Denoising | Filter noisy low-sample images; ML denoisers use auxiliary buffers |

## Exercises

1. **Monte Carlo pi estimation**: Estimate $\pi$ by randomly throwing points at a unit square and counting how many land inside the inscribed circle. Plot convergence vs. sample count. Verify the $1/\sqrt{N}$ rate.

2. **Importance sampling comparison**: Estimate $\int_0^1 x^{10}\,dx$ using (a) uniform sampling and (b) importance sampling with $p(x) = 11x^{10}$. Compare variance after 1000 samples.

3. **Noise vs. spp**: Render the Cornell box scene at 1, 4, 16, 64, and 256 spp. Measure the per-pixel RMSE against a 4096-spp reference. Verify the $1/\sqrt{N}$ convergence.

4. **NEE ablation**: Render the scene with and without Next Event Estimation at 64 spp. Compare noise levels. Explain why NEE helps more when lights are small.

5. **Russian roulette**: Implement a version without Russian roulette (fixed max depth = 5) and compare the brightness of the result against the version with Russian roulette. Which is more accurate?

6. **Simple denoiser**: Implement a bilateral filter as a post-process. Apply it to a 16-spp render and compare against raw 64-spp output. Experiment with the spatial and range sigma parameters.

## Further Reading

- Kajiya, J. "The Rendering Equation." *SIGGRAPH*, 1986. (The foundational paper that started it all)
- Veach, E. "Robust Monte Carlo Methods for Light Transport Simulation." PhD thesis, Stanford, 1997. (Introduced MIS, BDPT, and MLT; the most influential thesis in rendering)
- Pharr, M., Jakob, W., Humphreys, G. *Physically Based Rendering*, 4th ed. MIT Press, 2023. (The reference for path tracing implementation)
- Schied, C. et al. "Spatiotemporal Variance-Guided Filtering: Real-Time Reconstruction for Path-Traced Global Illumination." *HPG*, 2017. (SVGF denoiser)
- Kulla, C. and Conty, A. "Revisiting Physically Based Shading at Imageworks." *SIGGRAPH Course*, 2017. (Production path tracing at Sony Pictures)
