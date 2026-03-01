"""
Monte Carlo Path Tracer
========================

Implements a physically-based path tracer with:
1. Monte Carlo integration of the rendering equation
2. Cosine-weighted hemisphere sampling
3. Cornell box scene
4. Next Event Estimation (direct light sampling)
5. Russian roulette path termination
6. Progressive rendering (accumulate samples over time)

Unlike the Whitted ray tracer (08), which only follows perfect
reflection/refraction, a path tracer traces random light paths to
solve the full rendering equation.  This produces correct soft shadows,
color bleeding (indirect illumination), and glossy reflections --
all emergent from the physics, not special-cased.

The tradeoff: path tracing requires many samples per pixel (typically
hundreds to thousands) to converge to a noise-free image.

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import time

# ---------------------------------------------------------------------------
# 1. Core structures
# ---------------------------------------------------------------------------


@dataclass
class Ray:
    origin: np.ndarray
    direction: np.ndarray


@dataclass
class Material:
    """Diffuse Lambertian material.

    Why only diffuse?  A Lambertian material is the simplest physically-
    based BRDF: it reflects light equally in all directions.  This
    keeps the code focused on the path tracing algorithm itself.
    Extensions to glossy/specular materials would use a different
    sampling distribution but the same overall framework.
    """
    emission: np.ndarray = field(default_factory=lambda: np.zeros(3))
    albedo: np.ndarray = field(default_factory=lambda: np.array([0.7, 0.7, 0.7]))


@dataclass
class HitRecord:
    t: float
    point: np.ndarray
    normal: np.ndarray
    material: Material


# ---------------------------------------------------------------------------
# 2. Sampling utilities
# ---------------------------------------------------------------------------

def cosine_weighted_hemisphere(normal: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Sample a direction from a cosine-weighted hemisphere distribution.

    Why cosine-weighted?  The rendering equation contains a cos(theta)
    term (Lambert's cosine law).  By sampling proportional to cos(theta),
    we importance-sample the integrand, which dramatically reduces
    variance compared to uniform hemisphere sampling.

    The technique: generate a uniformly distributed point on a disk,
    then project it onto the hemisphere.  This naturally produces
    the cos(theta) distribution because the disk area element maps
    to cos(theta) * solid angle on the hemisphere.

    Math: if (u1, u2) are uniform [0, 1]:
      r = sqrt(u1),  theta = 2*pi*u2
      x = r*cos(theta),  y = r*sin(theta),  z = sqrt(1 - u1)
    This is in the local frame where z = normal direction.
    """
    u1, u2 = rng.random(), rng.random()
    r = np.sqrt(u1)
    phi = 2.0 * np.pi * u2

    # Local coordinates (z = normal direction)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    z = np.sqrt(max(0, 1.0 - u1))

    # Build a tangent-space basis from the normal
    # Why this particular construction?  We need an orthonormal frame
    # where one axis is aligned with the normal.  The "Frisvad" trick
    # of choosing an arbitrary perpendicular avoids a branch.
    if abs(normal[0]) > 0.9:
        tangent = np.array([0, 1, 0])
    else:
        tangent = np.array([1, 0, 0])

    bitangent = np.cross(normal, tangent)
    bitangent = bitangent / np.linalg.norm(bitangent)
    tangent = np.cross(bitangent, normal)

    # Transform from local to world coordinates
    world_dir = x * tangent + y * bitangent + z * normal
    return world_dir / np.linalg.norm(world_dir)


# ---------------------------------------------------------------------------
# 3. Scene geometry (axis-aligned boxes for Cornell box)
# ---------------------------------------------------------------------------

class Sphere:
    def __init__(self, center, radius, material):
        self.center = np.asarray(center, dtype=float)
        self.radius = radius
        self.material = material

    def intersect(self, ray: Ray) -> Optional[HitRecord]:
        oc = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius ** 2
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        sqrt_d = np.sqrt(disc)
        t = (-b - sqrt_d) / (2 * a)
        if t < 1e-4:
            t = (-b + sqrt_d) / (2 * a)
            if t < 1e-4:
                return None
        point = ray.origin + t * ray.direction
        normal = (point - self.center) / self.radius
        if np.dot(ray.direction, normal) > 0:
            normal = -normal
        return HitRecord(t, point, normal, self.material)


class AAPlane:
    """An axis-aligned rectangle (quad) in 3D space.

    Why quads instead of triangles?  The Cornell box is made of axis-
    aligned walls, so quads are the natural primitive.  Each quad is
    defined by which axis it's perpendicular to, its position along
    that axis, and the min/max bounds in the other two axes.
    """

    def __init__(self, axis: int, value: float,
                 min0: float, max0: float, min1: float, max1: float,
                 material: Material, flip_normal: bool = False):
        """
        axis: 0=x, 1=y, 2=z  (which axis is constant)
        value: position along that axis
        min0, max0: bounds along the first non-constant axis
        min1, max1: bounds along the second non-constant axis
        """
        self.axis = axis
        self.value = value
        self.min0 = min0
        self.max0 = max0
        self.min1 = min1
        self.max1 = max1
        self.material = material
        self.flip_normal = flip_normal

        # Determine the two varying axes
        self.ax0 = (axis + 1) % 3
        self.ax1 = (axis + 2) % 3

    def intersect(self, ray: Ray) -> Optional[HitRecord]:
        # Ray-plane intersection for axis-aligned plane
        if abs(ray.direction[self.axis]) < 1e-10:
            return None

        t = (self.value - ray.origin[self.axis]) / ray.direction[self.axis]
        if t < 1e-4:
            return None

        point = ray.origin + t * ray.direction

        # Check if hit point is within the rectangle bounds
        c0 = point[self.ax0]
        c1 = point[self.ax1]
        if c0 < self.min0 or c0 > self.max0 or c1 < self.min1 or c1 > self.max1:
            return None

        normal = np.zeros(3)
        normal[self.axis] = -1.0 if self.flip_normal else 1.0
        # Make sure normal faces the ray
        if np.dot(ray.direction, normal) > 0:
            normal = -normal

        return HitRecord(t, point, normal, self.material)

    def area(self) -> float:
        """Surface area of the rectangle (for light sampling)."""
        return (self.max0 - self.min0) * (self.max1 - self.min1)

    def sample_point(self, rng: np.random.RandomState) -> np.ndarray:
        """Sample a uniform random point on this rectangle.

        Why do we need to sample light surfaces?  Next Event Estimation
        requires picking a random point on each light source to create
        a shadow ray.  Uniform sampling on the light area is the
        simplest strategy.
        """
        p = np.zeros(3)
        p[self.axis] = self.value
        p[self.ax0] = self.min0 + rng.random() * (self.max0 - self.min0)
        p[self.ax1] = self.min1 + rng.random() * (self.max1 - self.min1)
        return p


# ---------------------------------------------------------------------------
# 4. Scene and intersection
# ---------------------------------------------------------------------------

class Scene:
    def __init__(self):
        self.objects: List = []
        self.lights: List = []  # Emissive objects (for NEE sampling)

    def add(self, obj, is_light=False):
        self.objects.append(obj)
        if is_light:
            self.lights.append(obj)

    def intersect(self, ray: Ray) -> Optional[HitRecord]:
        closest = None
        min_t = float('inf')
        for obj in self.objects:
            hit = obj.intersect(ray)
            if hit and hit.t < min_t:
                min_t = hit.t
                closest = hit
        return closest


# ---------------------------------------------------------------------------
# 5. Cornell Box scene
# ---------------------------------------------------------------------------

def create_cornell_box() -> Scene:
    """Build the classic Cornell box scene.

    Why Cornell box?  It's the standard test scene for global illumination
    algorithms.  The enclosed room with colored walls makes indirect
    illumination (color bleeding) clearly visible -- the red wall tints
    nearby surfaces red, the green wall tints them green.

    Dimensions: a box from (0,0,0) to (5,5,5) with:
    - White floor, ceiling, back wall
    - Red left wall
    - Green right wall
    - Area light on ceiling
    - Two spheres inside
    """
    scene = Scene()

    white = Material(albedo=np.array([0.73, 0.73, 0.73]))
    red = Material(albedo=np.array([0.65, 0.05, 0.05]))
    green = Material(albedo=np.array([0.12, 0.45, 0.15]))
    light_mat = Material(
        emission=np.array([15.0, 15.0, 12.0]),  # Warm white light
        albedo=np.array([0.0, 0.0, 0.0])
    )

    # Walls (axis-aligned planes)
    # Floor (y = 0)
    scene.add(AAPlane(axis=1, value=0, min0=0, max0=5, min1=0, max1=5,
                      material=white))
    # Ceiling (y = 5)
    scene.add(AAPlane(axis=1, value=5, min0=0, max0=5, min1=0, max1=5,
                      material=white, flip_normal=True))
    # Back wall (z = 0)
    scene.add(AAPlane(axis=2, value=0, min0=0, max0=5, min1=0, max1=5,
                      material=white))
    # Left wall (x = 0) - RED
    scene.add(AAPlane(axis=0, value=0, min0=0, max0=5, min1=0, max1=5,
                      material=red))
    # Right wall (x = 5) - GREEN
    scene.add(AAPlane(axis=0, value=5, min0=0, max0=5, min1=0, max1=5,
                      material=green, flip_normal=True))

    # Area light on ceiling (slightly below ceiling to be visible)
    light_panel = AAPlane(axis=1, value=4.99,
                          min0=1.5, max0=3.5, min1=1.5, max1=3.5,
                          material=light_mat, flip_normal=True)
    scene.add(light_panel, is_light=True)

    # Two spheres
    scene.add(Sphere(
        center=np.array([1.5, 1.0, 1.5]),
        radius=1.0,
        material=Material(albedo=np.array([0.73, 0.73, 0.73]))
    ))
    scene.add(Sphere(
        center=np.array([3.5, 0.7, 3.0]),
        radius=0.7,
        material=Material(albedo=np.array([0.73, 0.73, 0.73]))
    ))

    return scene


# ---------------------------------------------------------------------------
# 6. Path tracing core
# ---------------------------------------------------------------------------

def trace_path(scene: Scene, ray: Ray, rng: np.random.RandomState,
               max_depth: int = 8) -> np.ndarray:
    """Trace a single random light path and return its radiance.

    The rendering equation (Kajiya, 1986):
      Lo(p, wo) = Le(p, wo) + integral[ fr(p, wi, wo) * Li(p, wi) * cos(theta) dwi ]

    Monte Carlo integration estimates this integral by averaging random
    samples.  Each path represents one sample of the full light transport.

    This implementation uses:
    - Cosine-weighted importance sampling (reduces variance for Lambertian)
    - Next Event Estimation (directly sample lights at each bounce)
    - Russian roulette (probabilistic path termination for efficiency)
    """
    throughput = np.ones(3)  # Accumulated path weight
    radiance = np.zeros(3)  # Accumulated radiance
    current_ray = ray

    for depth in range(max_depth):
        hit = scene.intersect(current_ray)
        if hit is None:
            break  # Ray escaped the scene

        # Add emission from directly hitting a light
        # Only on the first bounce (for NEE, subsequent bounces handle
        # direct lighting via shadow rays)
        if depth == 0:
            radiance += throughput * hit.material.emission

        # --- Next Event Estimation (NEE) / Direct Light Sampling ---
        # Why NEE?  Pure path tracing relies on randomly hitting lights,
        # which is extremely unlikely for small light sources.  NEE
        # explicitly samples the light at each bounce, dramatically
        # reducing noise for direct illumination.
        for light_obj in scene.lights:
            # Sample a random point on the light surface
            light_point = light_obj.sample_point(rng)
            to_light = light_point - hit.point
            light_dist = np.linalg.norm(to_light)
            light_dir = to_light / light_dist

            # Check visibility (shadow ray)
            shadow_ray = Ray(hit.point + hit.normal * 1e-4, light_dir)
            shadow_hit = scene.intersect(shadow_ray)

            if shadow_hit and abs(shadow_hit.t - light_dist) < 1e-3:
                # We hit the light! Compute the contribution.

                # Geometry term: accounts for the angle of the light surface
                # and the distance (inverse square law)
                cos_light = abs(np.dot(shadow_hit.normal, -light_dir))
                cos_surface = max(np.dot(hit.normal, light_dir), 0)

                if cos_light > 1e-6 and cos_surface > 1e-6:
                    # PDF of sampling this point on the light
                    # = 1 / light_area (uniform sampling)
                    light_area = light_obj.area()

                    # Convert area PDF to solid angle PDF
                    # pdf_solid_angle = pdf_area * dist^2 / cos_light
                    # The contribution is: Le * BRDF * cos / pdf
                    # For Lambertian BRDF = albedo / pi
                    # Simplification: many terms cancel
                    weight = (cos_surface * cos_light * light_area
                              / (light_dist * light_dist))

                    brdf = hit.material.albedo / np.pi
                    direct = shadow_hit.material.emission * brdf * weight

                    radiance += throughput * direct

        # --- Russian Roulette ---
        # Why Russian roulette?  Instead of hard-capping at max_depth,
        # we probabilistically terminate paths based on their throughput.
        # Paths that have lost most of their energy are terminated early,
        # while bright paths continue.  This is unbiased because we
        # compensate by dividing the surviving paths' throughput by
        # the survival probability.
        if depth > 2:
            survival_prob = min(0.95, np.max(throughput))
            if rng.random() > survival_prob:
                break
            # Compensate: divide by survival probability to keep the
            # estimator unbiased
            throughput /= survival_prob

        # --- Indirect bounce ---
        # Sample a new direction from the cosine-weighted distribution
        new_dir = cosine_weighted_hemisphere(hit.normal, rng)
        current_ray = Ray(hit.point + hit.normal * 1e-4, new_dir)

        # Update throughput
        # For cosine-weighted sampling of Lambertian:
        #   BRDF = albedo / pi
        #   PDF = cos(theta) / pi
        #   throughput *= BRDF * cos(theta) / PDF = albedo
        # The cos and pi terms cancel -- this is the beauty of
        # importance sampling!
        throughput *= hit.material.albedo

    return radiance


# ---------------------------------------------------------------------------
# 7. Renderer with progressive accumulation
# ---------------------------------------------------------------------------

def render_progressive(scene: Scene, width: int = 300, height: int = 225,
                        samples_per_pixel: int = 64,
                        fov_deg: float = 40.0,
                        camera_pos: np.ndarray = None,
                        camera_target: np.ndarray = None) -> np.ndarray:
    """Render the scene with progressive sample accumulation.

    Why progressive?  Instead of computing all samples before showing
    anything, we accumulate samples incrementally.  This lets the user
    see a noisy preview quickly, which progressively cleans up.  Film
    production renderers (Arnold, RenderMan) use this approach.
    """
    if camera_pos is None:
        camera_pos = np.array([2.5, 2.5, 8.0])
    if camera_target is None:
        camera_target = np.array([2.5, 2.5, 0.0])

    rng = np.random.RandomState(42)

    # Build camera
    forward = camera_target - camera_pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.array([0, 1, 0]))
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    aspect = width / height
    half_h = np.tan(np.radians(fov_deg) / 2)
    half_w = half_h * aspect

    accumulator = np.zeros((height, width, 3))
    total_samples = 0

    start_time = time.time()

    for sample in range(samples_per_pixel):
        if (sample + 1) % 8 == 0 or sample == 0:
            elapsed = time.time() - start_time
            print(f"  Sample {sample + 1}/{samples_per_pixel} "
                  f"({elapsed:.1f}s)", end='\r')

        for j in range(height):
            for i in range(width):
                # Jittered sampling: add random sub-pixel offset
                # Why jitter?  Shooting rays through exact pixel centers
                # produces aliasing.  Random jitter turns aliasing into
                # noise, which is perceptually less objectionable and
                # averages out with more samples.
                jx = rng.random()
                jy = rng.random()

                u = (2 * (i + jx) / width - 1) * half_w
                v = (1 - 2 * (j + jy) / height) * half_h

                direction = forward + u * right + v * up
                direction = direction / np.linalg.norm(direction)

                ray = Ray(camera_pos.copy(), direction)
                color = trace_path(scene, ray, rng)

                accumulator[j, i] += color

        total_samples += 1

    elapsed = time.time() - start_time
    print(f"  Completed {samples_per_pixel} samples in {elapsed:.1f}s      ")

    # Average by number of samples
    image = accumulator / total_samples

    return image


def tonemap_aces(color: np.ndarray) -> np.ndarray:
    """ACES filmic tone mapping.

    Why ACES?  Simple Reinhard (x/(1+x)) desaturates highlights.
    ACES preserves color better and has a pleasant S-curve that
    mimics film response.  It's the industry standard in VFX/games.
    """
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    x = color
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)


# ---------------------------------------------------------------------------
# 8. Demo functions
# ---------------------------------------------------------------------------

def demo_path_tracer():
    """Render the Cornell box with the path tracer."""
    print("Creating Cornell box scene...")
    scene = create_cornell_box()

    print("Path tracing (this will take a few minutes)...")
    image = render_progressive(
        scene,
        width=200, height=150,
        samples_per_pixel=32,
        fov_deg=40,
        camera_pos=np.array([2.5, 2.5, 9.5]),
        camera_target=np.array([2.5, 2.5, 0.0])
    )

    # Tone mapping and gamma correction
    image = tonemap_aces(image)
    image = np.power(np.clip(image, 0, 1), 1.0 / 2.2)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image)
    ax.set_title("Monte Carlo Path Tracer: Cornell Box (32 spp)",
                 fontsize=13, fontweight='bold')
    ax.axis('off')
    ax.text(0.02, 0.02,
            "Features: cosine-weighted sampling, NEE, Russian roulette\n"
            "Notice: soft shadows, color bleeding (red/green walls onto spheres)",
            transform=ax.transAxes, fontsize=8, color='white',
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))

    plt.tight_layout()
    plt.savefig("/opt/projects/01_Personal/03_Study/examples/Computer_Graphics/"
                "output_09_path_tracer.png", dpi=150)
    plt.show()


def demo_convergence():
    """Show how image quality improves with more samples.

    This visualizes the fundamental noise-vs-time tradeoff of Monte
    Carlo rendering: more samples = less noise = more time.
    Variance decreases as 1/N (standard deviation as 1/sqrt(N)),
    so quadrupling render time halves the noise.
    """
    scene = create_cornell_box()
    rng = np.random.RandomState(42)

    # Camera setup
    camera_pos = np.array([2.5, 2.5, 9.5])
    camera_target = np.array([2.5, 2.5, 0.0])
    forward = camera_target - camera_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0, 1, 0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    w, h = 150, 112
    aspect = w / h
    half_h = np.tan(np.radians(40) / 2)
    half_w = half_h * aspect

    accumulator = np.zeros((h, w, 3))
    sample_counts = [1, 4, 16, 64]
    snapshots = []

    total_samples = 0
    for s in range(max(sample_counts)):
        if (s + 1) in sample_counts or s == 0:
            print(f"  Rendering sample {s + 1}...", end='\r')

        for j in range(h):
            for i in range(w):
                jx, jy = rng.random(), rng.random()
                u_val = (2 * (i + jx) / w - 1) * half_w
                v_val = (1 - 2 * (j + jy) / h) * half_h
                direction = forward + u_val * right + v_val * up
                direction /= np.linalg.norm(direction)
                ray = Ray(camera_pos.copy(), direction)
                color = trace_path(scene, ray, rng)
                accumulator[j, i] += color

        total_samples += 1

        if total_samples in sample_counts:
            snapshot = accumulator / total_samples
            snapshot = tonemap_aces(snapshot)
            snapshot = np.power(np.clip(snapshot, 0, 1), 1.0 / 2.2)
            snapshots.append((total_samples, snapshot.copy()))
            print(f"  Captured snapshot at {total_samples} spp")

    fig, axes = plt.subplots(1, len(snapshots), figsize=(16, 4))
    fig.suptitle("Path Tracer Convergence: Noise Decreases with More Samples",
                 fontsize=13, fontweight='bold')

    for ax, (count, img) in zip(axes, snapshots):
        ax.imshow(img)
        ax.set_title(f"{count} sample{'s' if count > 1 else ''} per pixel")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("/opt/projects/01_Personal/03_Study/examples/Computer_Graphics/"
                "output_09_convergence.png", dpi=100)
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Monte Carlo Path Tracer")
    print("=" * 60)

    print("\n[1/2] Cornell box render...")
    demo_path_tracer()

    print("\n[2/2] Convergence visualization...")
    demo_convergence()

    print("\nDone!")


if __name__ == "__main__":
    main()
