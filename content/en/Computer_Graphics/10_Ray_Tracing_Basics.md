# 10. Ray Tracing Basics

[← Previous: Scene Graphs and Spatial Data Structures](09_Scene_Graphs_and_Spatial_Data_Structures.md) | [Next: Path Tracing and Global Illumination →](11_Path_Tracing_and_Global_Illumination.md)

---

## Learning Objectives

1. Define rays mathematically and generate primary rays from a virtual camera
2. Derive and implement ray-sphere, ray-plane, and ray-triangle intersection tests
3. Understand the Moller-Trumbore algorithm for efficient ray-triangle intersection
4. Implement recursive (Whitted-style) ray tracing with reflection and refraction
5. Apply Snell's law for refraction and the Fresnel equations for reflection intensity
6. Explain how shadow rays determine visibility of light sources
7. Connect acceleration structures (BVH from L09) to practical ray tracing performance
8. Build a complete software ray tracer in Python that renders spheres with shadows and reflections

---

## Why This Matters

Think of ray tracing as **tracing light backwards from your eye**. In the physical world, light leaves sources, bounces around the scene, and eventually enters your eye. Simulating this forward process is hopelessly wasteful because the vast majority of light rays never reach the viewer. Ray tracing reverses the process: it shoots rays *from* the camera *into* the scene, asking "what did this pixel see?"

This elegant reversal, first formalized by Turner Whitted in 1980, produces stunningly realistic images with proper shadows, reflections, and refractions -- effects that are difficult or impossible with rasterization alone. Today, hardware ray tracing (NVIDIA RTX, AMD RDNA) has made real-time ray tracing practical, and understanding the underlying algorithms is essential for anyone working with modern graphics.

---

## 1. Ray Definition and Primary Ray Generation

### 1.1 Mathematical Ray

A **ray** is a half-line defined by an origin $\mathbf{o}$ and a direction $\mathbf{d}$:

$$\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}, \quad t \ge 0$$

Any point on the ray can be found by choosing a value of $t$. The parameter $t$ acts as a "distance" along the ray (exactly equal to distance when $\|\mathbf{d}\| = 1$).

### 1.2 Camera Model

To generate primary rays, we define a virtual camera with:
- **Eye position** $\mathbf{e}$
- **Look-at direction** (or target point)
- **Up vector** $\mathbf{up}$
- **Field of view** (FOV) angle $\theta$
- **Image resolution** $W \times H$

We construct an orthonormal camera basis:

$$\mathbf{w} = \frac{\mathbf{e} - \mathbf{target}}{\|\mathbf{e} - \mathbf{target}\|}, \quad
\mathbf{u} = \frac{\mathbf{up} \times \mathbf{w}}{\|\mathbf{up} \times \mathbf{w}\|}, \quad
\mathbf{v} = \mathbf{w} \times \mathbf{u}$$

where $\mathbf{w}$ points away from the scene (right-hand convention), $\mathbf{u}$ is the camera's right, and $\mathbf{v}$ is the camera's true up.

### 1.3 Generating Pixel Rays

For pixel $(i, j)$ in an image of size $W \times H$:

$$\text{aspect} = W / H$$

$$s = 2 \tan(\theta / 2)$$

$$u_{\text{pixel}} = s \cdot \text{aspect} \cdot \left(\frac{i + 0.5}{W} - 0.5\right)$$

$$v_{\text{pixel}} = s \cdot \left(0.5 - \frac{j + 0.5}{H}\right)$$

The ray direction (not yet normalized) is:

$$\mathbf{d} = u_{\text{pixel}} \cdot \mathbf{u} + v_{\text{pixel}} \cdot \mathbf{v} - \mathbf{w}$$

The $+0.5$ offset places the ray at the **center** of the pixel rather than its corner, reducing aliasing.

---

## 2. Ray-Object Intersection

### 2.1 Ray-Sphere Intersection

A sphere with center $\mathbf{c}$ and radius $r$ satisfies:

$$\|\mathbf{p} - \mathbf{c}\|^2 = r^2$$

Substituting the ray equation $\mathbf{p} = \mathbf{o} + t\mathbf{d}$:

$$\|\mathbf{o} + t\mathbf{d} - \mathbf{c}\|^2 = r^2$$

Let $\boldsymbol{\ell} = \mathbf{o} - \mathbf{c}$. Expanding:

$$(\mathbf{d} \cdot \mathbf{d})t^2 + 2(\boldsymbol{\ell} \cdot \mathbf{d})t + (\boldsymbol{\ell} \cdot \boldsymbol{\ell} - r^2) = 0$$

This is a quadratic $at^2 + bt + c = 0$ with:
- $a = \mathbf{d} \cdot \mathbf{d}$
- $b = 2(\boldsymbol{\ell} \cdot \mathbf{d})$
- $c = \boldsymbol{\ell} \cdot \boldsymbol{\ell} - r^2$

Discriminant $\Delta = b^2 - 4ac$:
- $\Delta < 0$: Ray misses the sphere
- $\Delta = 0$: Ray is tangent (one hit)
- $\Delta > 0$: Ray passes through (two hits)

The nearest positive $t$ gives the visible intersection. The surface normal at hit point $\mathbf{p}$ is:

$$\mathbf{n} = \frac{\mathbf{p} - \mathbf{c}}{r}$$

### 2.2 Ray-Plane Intersection

A plane defined by normal $\mathbf{n}$ and point $\mathbf{q}$ (or equivalently, $\mathbf{n} \cdot \mathbf{p} = d$ where $d = \mathbf{n} \cdot \mathbf{q}$):

$$\mathbf{n} \cdot (\mathbf{o} + t\mathbf{d}) = d$$

$$t = \frac{d - \mathbf{n} \cdot \mathbf{o}}{\mathbf{n} \cdot \mathbf{d}}$$

If $\mathbf{n} \cdot \mathbf{d} = 0$, the ray is parallel to the plane (no intersection). If $t < 0$, the plane is behind the ray origin.

### 2.3 Ray-Triangle Intersection: Moller-Trumbore Algorithm

A triangle is defined by vertices $\mathbf{v}_0, \mathbf{v}_1, \mathbf{v}_2$. Any point on the triangle can be written using **barycentric coordinates**:

$$\mathbf{p} = (1 - u - v)\mathbf{v}_0 + u\mathbf{v}_1 + v\mathbf{v}_2, \quad u \ge 0, \; v \ge 0, \; u + v \le 1$$

Setting this equal to the ray equation $\mathbf{o} + t\mathbf{d}$:

$$\mathbf{o} + t\mathbf{d} = (1 - u - v)\mathbf{v}_0 + u\mathbf{v}_1 + v\mathbf{v}_2$$

Rearranging into a linear system:

$$\begin{bmatrix} -\mathbf{d} & \mathbf{v}_1 - \mathbf{v}_0 & \mathbf{v}_2 - \mathbf{v}_0 \end{bmatrix} \begin{bmatrix} t \\ u \\ v \end{bmatrix} = \mathbf{o} - \mathbf{v}_0$$

The **Moller-Trumbore algorithm** solves this using Cramer's rule with cross products, avoiding explicit matrix construction:

Let $\mathbf{e}_1 = \mathbf{v}_1 - \mathbf{v}_0$, $\mathbf{e}_2 = \mathbf{v}_2 - \mathbf{v}_0$, $\mathbf{T} = \mathbf{o} - \mathbf{v}_0$:

$$\mathbf{P} = \mathbf{d} \times \mathbf{e}_2, \quad \text{det} = \mathbf{e}_1 \cdot \mathbf{P}$$

$$u = \frac{\mathbf{T} \cdot \mathbf{P}}{\text{det}}, \quad \mathbf{Q} = \mathbf{T} \times \mathbf{e}_1$$

$$v = \frac{\mathbf{d} \cdot \mathbf{Q}}{\text{det}}, \quad t = \frac{\mathbf{e}_2 \cdot \mathbf{Q}}{\text{det}}$$

The intersection is valid if $u \ge 0$, $v \ge 0$, $u + v \le 1$, and $t > 0$.

**Why Moller-Trumbore?** It uses only 1 division, 2 cross products, and several dot products -- no matrix inverse needed. It also provides barycentric coordinates $(u, v)$ for free, which are essential for interpolating vertex attributes (normals, texture coordinates).

### 2.4 Implementation

```python
import numpy as np

def ray_sphere(origin, direction, center, radius):
    """
    Ray-sphere intersection.
    Returns (hit, t, normal) where t is the nearest positive intersection.
    """
    oc = origin - center
    a = np.dot(direction, direction)
    b = 2.0 * np.dot(oc, direction)
    c = np.dot(oc, oc) - radius * radius

    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return False, float('inf'), None

    sqrt_disc = np.sqrt(discriminant)
    # Why we try the smaller root first: it's the nearer intersection
    t = (-b - sqrt_disc) / (2.0 * a)
    if t < 1e-4:  # Avoid self-intersection
        t = (-b + sqrt_disc) / (2.0 * a)
    if t < 1e-4:
        return False, float('inf'), None

    hit_point = origin + t * direction
    normal = (hit_point - center) / radius
    return True, t, normal


def ray_plane(origin, direction, plane_normal, plane_d):
    """
    Ray-plane intersection.
    Plane equation: dot(normal, p) = plane_d.
    """
    denom = np.dot(plane_normal, direction)
    if abs(denom) < 1e-8:  # Ray parallel to plane
        return False, float('inf'), None

    t = (plane_d - np.dot(plane_normal, origin)) / denom
    if t < 1e-4:
        return False, float('inf'), None

    return True, t, plane_normal


def ray_triangle_moller_trumbore(origin, direction, v0, v1, v2):
    """
    Moller-Trumbore ray-triangle intersection.
    Returns (hit, t, u, v) where u, v are barycentric coordinates.
    """
    EPSILON = 1e-8
    e1 = v1 - v0
    e2 = v2 - v0

    # P = d x e2 -- this vector is reused for both det and u
    P = np.cross(direction, e2)
    det = np.dot(e1, P)

    # Why we check det near zero: ray is parallel to triangle plane
    if abs(det) < EPSILON:
        return False, float('inf'), 0, 0

    inv_det = 1.0 / det
    T = origin - v0

    # u is the first barycentric coordinate
    u = np.dot(T, P) * inv_det
    if u < 0.0 or u > 1.0:
        return False, float('inf'), 0, 0

    Q = np.cross(T, e1)

    # v is the second barycentric coordinate
    v = np.dot(direction, Q) * inv_det
    if v < 0.0 or u + v > 1.0:
        return False, float('inf'), 0, 0

    t = np.dot(e2, Q) * inv_det
    if t < EPSILON:
        return False, float('inf'), 0, 0

    return True, t, u, v
```

---

## 3. Shadow Rays

To determine if a surface point $\mathbf{p}$ is in shadow, cast a **shadow ray** from $\mathbf{p}$ toward each light source:

$$\mathbf{r}_{\text{shadow}}(t) = \mathbf{p} + t\mathbf{L}$$

where $\mathbf{L}$ is the direction to the light. If the shadow ray hits any object before reaching the light, the point is in shadow and receives no direct illumination from that light.

**Self-intersection problem**: The shadow ray origin $\mathbf{p}$ lies on a surface. Due to floating-point imprecision, the ray may immediately intersect the same surface. The standard fix is to offset the origin slightly along the normal:

$$\mathbf{p}_{\text{offset}} = \mathbf{p} + \epsilon \cdot \mathbf{n}$$

where $\epsilon \approx 10^{-4}$ is a small bias. Alternatively, use a minimum $t$ threshold.

---

## 4. Recursive Ray Tracing (Whitted-Style)

### 4.1 The Whitted Model

Turner Whitted's 1980 paper introduced **recursive ray tracing**, where each ray hit can spawn additional rays:

1. **Primary ray**: From camera through each pixel
2. **Shadow ray**: From hit point toward each light (determines shadows)
3. **Reflection ray**: If the surface is reflective, spawn a ray in the reflection direction
4. **Refraction ray**: If the surface is transparent, spawn a ray in the refracted direction

The color at a pixel is computed recursively:

$$L(\mathbf{p}) = L_{\text{ambient}} + L_{\text{direct}} + k_r \cdot L(\mathbf{p}_{\text{reflect}}) + k_t \cdot L(\mathbf{p}_{\text{refract}})$$

where $k_r$ is the reflection coefficient and $k_t$ is the transmission (refraction) coefficient.

Recursion terminates when:
- Maximum depth is reached (e.g., 5 bounces)
- The ray misses all objects (returns background color)
- The contribution becomes negligibly small

### 4.2 Reflection Direction

For an incoming ray direction $\mathbf{d}$ hitting a surface with normal $\mathbf{n}$, the **perfect reflection** direction is:

$$\mathbf{r} = \mathbf{d} - 2(\mathbf{d} \cdot \mathbf{n})\mathbf{n}$$

This formula "mirrors" $\mathbf{d}$ around $\mathbf{n}$. Note that $\mathbf{d}$ points *toward* the surface, so $\mathbf{d} \cdot \mathbf{n} < 0$ for a front-facing hit.

### 4.3 Refraction: Snell's Law

When light passes from a medium with refractive index $\eta_1$ into a medium with index $\eta_2$, the refracted direction obeys **Snell's law**:

$$\eta_1 \sin\theta_1 = \eta_2 \sin\theta_2$$

where $\theta_1$ is the angle of incidence and $\theta_2$ is the angle of refraction.

The refracted direction vector is:

$$\mathbf{t} = \frac{\eta_1}{\eta_2}\mathbf{d} + \left(\frac{\eta_1}{\eta_2}\cos\theta_1 - \cos\theta_2\right)\mathbf{n}$$

where:
$$\cos\theta_1 = -\mathbf{d} \cdot \mathbf{n}$$
$$\cos^2\theta_2 = 1 - \left(\frac{\eta_1}{\eta_2}\right)^2(1 - \cos^2\theta_1)$$

If $\cos^2\theta_2 < 0$, **total internal reflection** occurs (no refracted ray).

Common refractive indices: air $\approx 1.0$, water $= 1.33$, glass $= 1.5$, diamond $= 2.42$.

### 4.4 Fresnel Equations

The **Fresnel equations** determine how much light is reflected vs. refracted at an interface. The Schlick approximation is commonly used in graphics:

$$R(\theta) \approx R_0 + (1 - R_0)(1 - \cos\theta)^5$$

where $R_0 = \left(\frac{\eta_1 - \eta_2}{\eta_1 + \eta_2}\right)^2$ is the reflectance at normal incidence.

At grazing angles ($\theta \to 90°$), $R \to 1$ -- nearly all light is reflected. This is why you see clear reflections on water when looking at a shallow angle.

---

## 5. Acceleration Structures for Ray Tracing

Testing every ray against every object is $O(N)$ per ray. For a scene with millions of triangles and millions of pixels, this is far too slow.

**Acceleration structures** (detailed in [Lesson 9](09_Scene_Graphs_and_Spatial_Data_Structures.md)) reduce this to $O(\log N)$ per ray:

- **BVH (Bounding Volume Hierarchy)**: The most common choice. Build a tree of AABBs; traverse top-down, skipping subtrees whose boxes the ray misses.
- **kd-tree**: Split space with axis-aligned planes. Historically popular but BVH is generally preferred now.
- **Uniform grid**: Simple to build, but performance degrades with non-uniform object distributions.

Modern GPU ray tracing (RTX, DXR, Vulkan RT) uses hardware-accelerated BVH traversal.

---

## 6. Python Implementation: Simple Ray Tracer

```python
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple

# --- Scene Definition ---

@dataclass
class Material:
    """Surface material properties."""
    color: np.ndarray           # Diffuse color (RGB, 0-1)
    ambient: float = 0.1        # Ambient coefficient
    diffuse: float = 0.7        # Diffuse coefficient
    specular: float = 0.3       # Specular coefficient
    shininess: float = 50.0     # Specular exponent
    reflectivity: float = 0.0   # 0 = no reflection, 1 = perfect mirror
    transparency: float = 0.0   # 0 = opaque, 1 = fully transparent
    ior: float = 1.5            # Index of refraction (glass)


@dataclass
class SceneSphere:
    """A sphere in the scene."""
    center: np.ndarray
    radius: float
    material: Material

    def intersect(self, origin, direction):
        """Ray-sphere intersection. Returns (t, normal) or (inf, None)."""
        oc = origin - self.center
        a = np.dot(direction, direction)
        b = 2.0 * np.dot(oc, direction)
        c = np.dot(oc, oc) - self.radius ** 2
        disc = b * b - 4 * a * c
        if disc < 0:
            return float('inf'), None
        sqrt_disc = np.sqrt(disc)
        t = (-b - sqrt_disc) / (2.0 * a)
        if t < 1e-4:
            t = (-b + sqrt_disc) / (2.0 * a)
        if t < 1e-4:
            return float('inf'), None
        hit = origin + t * direction
        normal = (hit - self.center) / self.radius
        return t, normal


@dataclass
class Plane:
    """An infinite plane."""
    point: np.ndarray       # A point on the plane
    normal: np.ndarray      # Surface normal (unit)
    material: Material

    def intersect(self, origin, direction):
        """Ray-plane intersection."""
        denom = np.dot(self.normal, direction)
        if abs(denom) < 1e-8:
            return float('inf'), None
        t = np.dot(self.point - origin, self.normal) / denom
        if t < 1e-4:
            return float('inf'), None
        return t, self.normal.copy()


@dataclass
class Light:
    """A point light source."""
    position: np.ndarray
    color: np.ndarray       # Light color/intensity
    intensity: float = 1.0


def normalize(v):
    """Safely normalize a vector."""
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def reflect(d, n):
    """Compute reflection direction: d is incoming (toward surface)."""
    return d - 2 * np.dot(d, n) * n


def refract(d, n, eta_ratio):
    """
    Compute refraction direction using Snell's law.
    d: incoming direction (toward surface, unit)
    n: surface normal (unit, pointing outward)
    eta_ratio: eta_i / eta_t
    Returns refracted direction or None (total internal reflection).
    """
    cos_i = -np.dot(d, n)
    sin2_t = eta_ratio ** 2 * (1.0 - cos_i ** 2)
    if sin2_t > 1.0:
        # Total internal reflection
        return None
    cos_t = np.sqrt(1.0 - sin2_t)
    return eta_ratio * d + (eta_ratio * cos_i - cos_t) * n


def fresnel_schlick(cos_theta, eta1, eta2):
    """
    Schlick's approximation for Fresnel reflectance.
    Returns fraction of light that is reflected.
    """
    r0 = ((eta1 - eta2) / (eta1 + eta2)) ** 2
    return r0 + (1 - r0) * (1 - cos_theta) ** 5


# --- Ray Tracer Core ---

class RayTracer:
    def __init__(self, objects, lights, bg_color=np.array([0.1, 0.1, 0.2]),
                 max_depth=5):
        self.objects = objects
        self.lights = lights
        self.bg_color = bg_color
        self.max_depth = max_depth

    def find_nearest(self, origin, direction):
        """Find the nearest intersection among all objects."""
        nearest_t = float('inf')
        nearest_obj = None
        nearest_normal = None

        for obj in self.objects:
            t, normal = obj.intersect(origin, direction)
            if t < nearest_t:
                nearest_t = t
                nearest_obj = obj
                nearest_normal = normal

        return nearest_t, nearest_obj, nearest_normal

    def shade(self, origin, direction, depth=0):
        """
        Recursively trace a ray and compute the color.
        This is the heart of Whitted-style ray tracing.
        """
        if depth >= self.max_depth:
            return self.bg_color.copy()

        t, obj, normal = self.find_nearest(origin, direction)
        if obj is None:
            return self.bg_color.copy()

        hit_point = origin + t * direction
        mat = obj.material

        # Ensure normal faces the incoming ray
        # Why: for transparent objects, we may be inside the surface
        if np.dot(normal, direction) > 0:
            normal = -normal

        color = np.zeros(3)

        # --- Ambient term ---
        color += mat.ambient * mat.color

        # --- Direct illumination (diffuse + specular) for each light ---
        for light in self.lights:
            light_dir = normalize(light.position - hit_point)
            light_dist = np.linalg.norm(light.position - hit_point)

            # Shadow test: cast ray toward light, check for occlusion
            shadow_origin = hit_point + 1e-4 * normal
            shadow_t, shadow_obj, _ = self.find_nearest(shadow_origin, light_dir)

            if shadow_t < light_dist:
                # Point is in shadow for this light
                continue

            # Diffuse (Lambertian)
            n_dot_l = max(0.0, np.dot(normal, light_dir))
            color += mat.diffuse * n_dot_l * mat.color * light.color * light.intensity

            # Specular (Blinn-Phong)
            view_dir = normalize(-direction)
            half_vec = normalize(light_dir + view_dir)
            n_dot_h = max(0.0, np.dot(normal, half_vec))
            color += mat.specular * (n_dot_h ** mat.shininess) * light.color * light.intensity

        # --- Reflection ---
        if mat.reflectivity > 0.0 and depth < self.max_depth:
            reflect_dir = normalize(reflect(direction, normal))
            reflect_origin = hit_point + 1e-4 * normal
            reflect_color = self.shade(reflect_origin, reflect_dir, depth + 1)
            color += mat.reflectivity * reflect_color

        # --- Refraction (transparency) ---
        if mat.transparency > 0.0 and depth < self.max_depth:
            # Determine if we're entering or leaving the object
            entering = np.dot(direction, normal) < 0
            if entering:
                eta_ratio = 1.0 / mat.ior  # Air -> Material
                refract_normal = normal
            else:
                eta_ratio = mat.ior / 1.0   # Material -> Air
                refract_normal = -normal

            cos_i = abs(np.dot(direction, refract_normal))
            kr = fresnel_schlick(cos_i, 1.0, mat.ior)

            refract_dir = refract(normalize(direction), refract_normal, eta_ratio)
            if refract_dir is not None:
                refract_origin = hit_point - 1e-4 * refract_normal
                refract_color = self.shade(refract_origin, normalize(refract_dir),
                                           depth + 1)
                # Mix reflection and refraction using Fresnel
                color = color * (1 - mat.transparency) \
                      + mat.transparency * (kr * reflect(direction, normal) is not None
                                            and self.shade(hit_point + 1e-4 * normal,
                                                          normalize(reflect(direction, normal)),
                                                          depth + 1) * kr
                                            + (1 - kr) * refract_color
                                            if True else refract_color)
                # Simplified: blend refracted and reflected
                reflect_dir = normalize(reflect(direction, normal))
                reflect_origin = hit_point + 1e-4 * normal
                reflect_color = self.shade(reflect_origin, reflect_dir, depth + 1)

                color = color * (1 - mat.transparency) \
                      + mat.transparency * (kr * reflect_color + (1 - kr) * refract_color)

        # Clamp to [0, 1]
        return np.clip(color, 0.0, 1.0)

    def render(self, width, height, fov_deg=60.0,
               eye=np.array([0, 1, 5.0]),
               target=np.array([0, 0, 0])):
        """
        Render the scene to an image array (H, W, 3).
        """
        image = np.zeros((height, width, 3))
        fov_rad = np.radians(fov_deg)
        aspect = width / height

        # Build camera coordinate frame
        forward = normalize(target - eye)
        right = normalize(np.cross(forward, np.array([0, 1, 0])))
        up = np.cross(right, forward)

        # Image plane half-dimensions
        half_h = np.tan(fov_rad / 2)
        half_w = half_h * aspect

        for j in range(height):
            for i in range(width):
                # Map pixel to [-1, 1] range
                u = (2 * (i + 0.5) / width - 1) * half_w
                v = (1 - 2 * (j + 0.5) / height) * half_h

                direction = normalize(forward + u * right + v * up)
                color = self.shade(eye, direction)
                image[j, i] = color

            # Progress indicator
            if (j + 1) % (height // 10) == 0:
                print(f"  Row {j+1}/{height} ({100*(j+1)//height}%)")

        return image


# --- Build Scene ---

# Materials
red_mat = Material(color=np.array([0.9, 0.1, 0.1]), reflectivity=0.2)
green_mat = Material(color=np.array([0.1, 0.9, 0.1]), reflectivity=0.1)
blue_mat = Material(color=np.array([0.1, 0.1, 0.9]), reflectivity=0.3)
mirror_mat = Material(color=np.array([0.9, 0.9, 0.9]),
                      reflectivity=0.8, diffuse=0.2, specular=0.8)
floor_mat = Material(color=np.array([0.5, 0.5, 0.5]), reflectivity=0.1)

# Objects
objects = [
    SceneSphere(center=np.array([-2.0, 0.5, -1.0]), radius=1.0, material=red_mat),
    SceneSphere(center=np.array([0.0, 0.7, 0.0]),   radius=1.2, material=mirror_mat),
    SceneSphere(center=np.array([2.0, 0.5, -0.5]),   radius=1.0, material=blue_mat),
    SceneSphere(center=np.array([0.5, 0.3, 2.0]),    radius=0.6, material=green_mat),
    Plane(point=np.array([0, -0.5, 0]), normal=np.array([0, 1, 0]),
          material=floor_mat),
]

# Lights
lights = [
    Light(position=np.array([-5, 8, 5]),  color=np.array([1, 1, 1]),    intensity=0.8),
    Light(position=np.array([5, 6, -3]),   color=np.array([0.8, 0.9, 1]), intensity=0.6),
]

# --- Render ---
tracer = RayTracer(objects, lights, max_depth=4)
print("Rendering 320x240 image...")
image = tracer.render(320, 240, fov_deg=60,
                      eye=np.array([0, 2, 6]),
                      target=np.array([0, 0, 0]))

# Save result (requires matplotlib or PIL)
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 7.5))
    plt.imshow(image)
    plt.axis('off')
    plt.title('Whitted-Style Ray Tracer')
    plt.tight_layout()
    plt.savefig('ray_trace_result.png', dpi=150)
    plt.close()
    print("Saved ray_trace_result.png")
except ImportError:
    # Fallback: save as PPM
    with open('ray_trace_result.ppm', 'w') as f:
        f.write(f'P3\n{320} {240}\n255\n')
        for j in range(240):
            for i in range(320):
                r, g, b = (image[j, i] * 255).astype(int)
                f.write(f'{r} {g} {b} ')
            f.write('\n')
    print("Saved ray_trace_result.ppm")
```

---

## 7. Ray Tracing vs. Rasterization

| Aspect | Rasterization | Ray Tracing |
|--------|---------------|-------------|
| Approach | Object-order: project each triangle onto screen | Image-order: trace ray for each pixel |
| Shadows | Requires shadow maps (approximate) | Natural (shadow rays are exact) |
| Reflections | Screen-space or cube maps (approximate) | Natural (reflection rays are exact) |
| Refractions | Very difficult | Natural (refraction rays) |
| Performance | $O(N)$ in triangles; GPU-optimized | $O(N \log N)$ with BVH; historically slower |
| Real-time | Default for games | Now feasible with RTX hardware |
| Global illumination | Approximate (SSAO, probes) | Extended to path tracing (L11) |

Modern renderers often combine both: rasterize for primary visibility, then ray trace for shadows, reflections, and ambient occlusion.

---

## 8. Practical Considerations

### 8.1 Numerical Robustness

Floating-point errors cause artifacts:
- **Shadow acne**: Self-intersection of shadow rays. Fix: offset ray origin by $\epsilon \cdot \mathbf{n}$
- **Peter Panning**: Over-biasing moves shadows away from the object. Fix: use tight $\epsilon$
- **Watertight intersections**: Gaps between triangles sharing edges. Modern algorithms (e.g., Woop et al. 2013) guarantee watertight tests

### 8.2 Anti-Aliasing

A single ray per pixel produces aliased (jagged) edges. **Supersampling** casts multiple rays per pixel with small random offsets (stratified or jittered sampling) and averages the results:

$$\text{pixel color} = \frac{1}{N}\sum_{k=1}^{N} \text{shade}(\mathbf{r}_k)$$

Even $N = 4$ (2x2 grid) significantly reduces jaggies.

### 8.3 Gamma Correction

The raw linear-space colors must be **gamma corrected** before display:

$$c_{\text{display}} = c_{\text{linear}}^{1/\gamma}, \quad \gamma = 2.2$$

Without gamma correction, the image will look too dark.

---

## Summary

| Concept | Key Formula / Idea |
|---------|--------------------|
| Ray | $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$ |
| Ray-sphere | Quadratic: $at^2 + bt + c = 0$; discriminant determines hit count |
| Ray-plane | $t = (d - \mathbf{n} \cdot \mathbf{o}) / (\mathbf{n} \cdot \mathbf{d})$ |
| Moller-Trumbore | Cramer's rule with cross products; gives $t$, $u$, $v$ |
| Shadow ray | Cast from hit point toward light; any closer intersection = shadow |
| Reflection | $\mathbf{r} = \mathbf{d} - 2(\mathbf{d} \cdot \mathbf{n})\mathbf{n}$ |
| Snell's law | $\eta_1 \sin\theta_1 = \eta_2 \sin\theta_2$ |
| Fresnel (Schlick) | $R(\theta) \approx R_0 + (1 - R_0)(1 - \cos\theta)^5$ |
| Whitted tracing | Recursive: primary + shadow + reflection + refraction rays |
| BVH acceleration | Reduces per-ray cost from $O(N)$ to $O(\log N)$ |

## Exercises

1. **Ray-sphere by hand**: Given ray origin $(0, 0, 5)$, direction $(0, 0, -1)$, and sphere center $(0, 0, 0)$ with radius 1, solve for both intersection $t$-values. What is the surface normal at the nearer hit point?

2. **Moller-Trumbore**: Implement the algorithm and test it with triangle $\mathbf{v}_0 = (0, 0, 0)$, $\mathbf{v}_1 = (1, 0, 0)$, $\mathbf{v}_2 = (0, 1, 0)$ and ray origin $(0.2, 0.2, 1)$, direction $(0, 0, -1)$. Verify the barycentric coordinates.

3. **Refraction visualization**: Modify the ray tracer to add a glass sphere (IOR = 1.5). Observe how background objects are distorted when viewed through the sphere. Experiment with different IOR values.

4. **Shadow comparison**: Render the same scene with 1 light, then 3 lights at different positions. Compare the shadow patterns. Why do multiple lights create softer-looking shadows?

5. **Anti-aliasing**: Implement $4\times$ supersampling (2x2 jittered grid per pixel). Compare the result with single-sample rendering. Measure the performance difference.

6. **BVH integration**: Connect the BVH from [Lesson 9](09_Scene_Graphs_and_Spatial_Data_Structures.md) to this ray tracer. Generate 100 random spheres and compare rendering time with and without BVH acceleration.

## Further Reading

- Whitted, T. "An Improved Illumination Model for Shaded Display." *Communications of the ACM*, 1980. (The seminal recursive ray tracing paper)
- Pharr, M., Jakob, W., Humphreys, G. *Physically Based Rendering*, 4th ed. MIT Press, 2023. (Chapters 2-4: ray-object intersection, acceleration)
- Shirley, P. *Ray Tracing in One Weekend*. Online, 2020. (Excellent hands-on tutorial)
- Moller, T., Trumbore, B. "Fast, Minimum Storage Ray/Triangle Intersection." *Journal of Graphics Tools*, 1997. (The standard ray-triangle algorithm)
- Akenine-Moller, T. et al. *Real-Time Rendering*, 4th ed. CRC Press, 2018. (Chapter 26: real-time ray tracing)
