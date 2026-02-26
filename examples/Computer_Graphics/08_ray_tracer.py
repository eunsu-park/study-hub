"""
Recursive Ray Tracer
=====================

Implements a classic Whitted-style ray tracer with:
1. Ray-sphere and ray-plane intersection
2. Phong shading at hit points
3. Recursive reflection and refraction (transmission)
4. Shadow rays for hard shadows
5. Multiple light sources
6. Scene with reflective sphere, glass sphere, and textured ground

Ray tracing answers the question "what color is this pixel?" by
shooting a ray from the camera through each pixel and tracing its
path as it bounces around the scene.  Unlike rasterization (which
asks "which pixels does this triangle cover?"), ray tracing naturally
handles reflection, refraction, and shadows.

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import time

# ---------------------------------------------------------------------------
# 1. Core data structures
# ---------------------------------------------------------------------------


@dataclass
class Ray:
    """A ray defined by origin + direction.

    Why a dataclass?  Rays are the fundamental data flowing through the
    entire pipeline.  Having a named type makes code self-documenting
    and catches bugs where you'd accidentally swap origin and direction.
    """
    origin: np.ndarray
    direction: np.ndarray  # Should be normalized


@dataclass
class Material:
    """Surface material properties.

    Why separate ambient/diffuse/specular AND reflectivity/transparency?
    The first three control local illumination (Phong model).
    Reflectivity and transparency control how much light is recursively
    traced via reflection/refraction rays -- these are what make the
    ray tracer fundamentally more powerful than a rasterizer.
    """
    color: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.5]))
    ambient: float = 0.1
    diffuse: float = 0.7
    specular: float = 0.5
    shininess: float = 50.0
    reflectivity: float = 0.0    # 0 = matte, 1 = perfect mirror
    transparency: float = 0.0    # 0 = opaque, 1 = fully transparent
    ior: float = 1.5             # Index of refraction (glass ~ 1.5)


@dataclass
class HitRecord:
    """Information about a ray-object intersection.

    Why record all this?  The shading computation needs the hit point
    (where to shade), the normal (surface orientation for lighting),
    the distance (for finding the closest hit), and a flag for whether
    we're inside the object (important for refraction direction).
    """
    t: float                       # Distance along the ray
    point: np.ndarray             # World-space hit point
    normal: np.ndarray            # Surface normal at hit point
    material: Material            # Material at hit point
    inside: bool = False          # True if ray is inside the object


@dataclass
class Light:
    """A point light source.

    Why point lights?  They're the simplest to implement and produce
    hard shadows.  Area lights (which produce soft shadows) require
    many sample rays per pixel -- that's what the path tracer does.
    """
    position: np.ndarray
    color: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0]))
    intensity: float = 1.0


# ---------------------------------------------------------------------------
# 2. Utility functions
# ---------------------------------------------------------------------------

def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector, returning zero vector if length is near zero."""
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else np.zeros_like(v)


def reflect(I: np.ndarray, N: np.ndarray) -> np.ndarray:
    """Reflect incident vector I about normal N.

    Standard reflection formula: R = I - 2(I.N)N
    I should point TOWARD the surface for correct reflection.
    """
    return I - 2 * np.dot(I, N) * N


def refract(I: np.ndarray, N: np.ndarray, eta: float) -> Optional[np.ndarray]:
    """Compute refraction direction using Snell's law.

    eta = n1/n2 (ratio of indices of refraction)

    Returns None if total internal reflection occurs (angle too steep).
    Total internal reflection happens when light tries to exit a dense
    medium at a shallow angle -- this is why fiber optics work and why
    you see reflections on the underside of water surfaces.
    """
    cos_i = -np.dot(I, N)
    sin2_t = eta * eta * (1.0 - cos_i * cos_i)

    if sin2_t > 1.0:
        return None  # Total internal reflection

    cos_t = np.sqrt(1.0 - sin2_t)
    return eta * I + (eta * cos_i - cos_t) * N


def schlick_reflectance(cos_theta: float, ior: float) -> float:
    """Schlick's approximation for Fresnel reflectance.

    Why Schlick here too?  Even transparent objects reflect some light.
    The amount depends on the viewing angle (more reflection at grazing
    angles).  This is crucial for realistic glass rendering.
    """
    r0 = ((1 - ior) / (1 + ior)) ** 2
    return r0 + (1 - r0) * (1 - cos_theta) ** 5


# ---------------------------------------------------------------------------
# 3. Scene objects
# ---------------------------------------------------------------------------

class Sphere:
    """A sphere defined by center and radius.

    Why spheres?  They have the simplest ray intersection formula
    (a quadratic equation) and produce beautiful reflections/refractions.
    Spheres are the "hydrogen atom" of ray tracing.
    """

    def __init__(self, center: np.ndarray, radius: float, material: Material):
        self.center = np.asarray(center, dtype=float)
        self.radius = radius
        self.material = material

    def intersect(self, ray: Ray) -> Optional[HitRecord]:
        """Ray-sphere intersection using the quadratic formula.

        The ray P(t) = O + t*D intersects the sphere |P - C|^2 = r^2
        when substituting gives: a*t^2 + b*t + c = 0
        where a = D.D, b = 2*D.(O-C), c = (O-C).(O-C) - r^2

        The discriminant b^2 - 4ac tells us:
          < 0: ray misses the sphere
          = 0: ray is tangent (grazes the surface)
          > 0: ray pierces the sphere at two points
        """
        oc = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius * self.radius

        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return None

        sqrt_disc = np.sqrt(discriminant)

        # Try the closer intersection first (smaller t)
        t = (-b - sqrt_disc) / (2 * a)
        if t < 1e-4:
            # If the closer hit is behind us, try the farther one
            # (this happens when the ray origin is inside the sphere)
            t = (-b + sqrt_disc) / (2 * a)
            if t < 1e-4:
                return None

        point = ray.origin + t * ray.direction
        normal = normalize(point - self.center)

        # Check if we hit from inside the sphere
        inside = np.dot(ray.direction, normal) > 0
        if inside:
            normal = -normal  # Flip normal to face the ray

        return HitRecord(t=t, point=point, normal=normal,
                         material=self.material, inside=inside)


class Plane:
    """An infinite plane defined by a point and normal.

    Why a plane?  It serves as a ground/floor in most ray tracing demos.
    The intersection test is extremely simple: one dot product.
    """

    def __init__(self, point: np.ndarray, normal: np.ndarray,
                 material: Material, checker: bool = False):
        self.point = np.asarray(point, dtype=float)
        self.normal = normalize(np.asarray(normal, dtype=float))
        self.material = material
        self.checker = checker  # Enable checkerboard pattern

    def intersect(self, ray: Ray) -> Optional[HitRecord]:
        """Ray-plane intersection.

        A point P is on the plane if (P - P0).N = 0.
        Substituting the ray equation: t = (P0 - O).N / (D.N)

        If D.N ~ 0, the ray is parallel to the plane (no intersection).
        If t < 0, the intersection is behind the ray origin.
        """
        denom = np.dot(ray.direction, self.normal)
        if abs(denom) < 1e-8:
            return None  # Ray is parallel to the plane

        t = np.dot(self.point - ray.origin, self.normal) / denom
        if t < 1e-4:
            return None  # Behind the ray

        point = ray.origin + t * ray.direction

        # Create a copy of the material so we can modify the color
        mat = Material(
            color=self.material.color.copy(),
            ambient=self.material.ambient,
            diffuse=self.material.diffuse,
            specular=self.material.specular,
            shininess=self.material.shininess,
            reflectivity=self.material.reflectivity,
            transparency=self.material.transparency,
            ior=self.material.ior,
        )

        # Checkerboard pattern: alternate colors based on floor position
        # Why checkerboard?  It makes the ground plane visually interesting
        # and clearly shows reflections, shadows, and refraction distortion.
        if self.checker:
            u = int(np.floor(point[0] * 0.5))
            v = int(np.floor(point[2] * 0.5))
            if (u + v) % 2 == 0:
                mat.color = np.array([0.9, 0.9, 0.9])
            else:
                mat.color = np.array([0.2, 0.2, 0.2])

        return HitRecord(t=t, point=point, normal=self.normal, material=mat)


# ---------------------------------------------------------------------------
# 4. Scene definition
# ---------------------------------------------------------------------------

class Scene:
    """Container for all objects and lights in the scene."""

    def __init__(self):
        self.objects: List = []
        self.lights: List[Light] = []
        self.background_color = np.array([0.1, 0.1, 0.2])

    def add_object(self, obj):
        self.objects.append(obj)

    def add_light(self, light: Light):
        self.lights.append(light)

    def closest_hit(self, ray: Ray) -> Optional[HitRecord]:
        """Find the closest intersection of a ray with any scene object.

        Why iterate over ALL objects?  The closest hit determines what
        you see.  Unlike rasterization which uses a Z-buffer, ray tracing
        explicitly finds the nearest intersection by comparing distances.
        In production, spatial acceleration structures (BVH, KD-tree)
        avoid testing every object.
        """
        closest: Optional[HitRecord] = None
        min_t = float('inf')

        for obj in self.objects:
            hit = obj.intersect(ray)
            if hit and hit.t < min_t:
                min_t = hit.t
                closest = hit

        return closest


# ---------------------------------------------------------------------------
# 5. Shading and recursive tracing
# ---------------------------------------------------------------------------

def shade(scene: Scene, ray: Ray, hit: HitRecord) -> np.ndarray:
    """Compute local illumination (Phong) at the hit point.

    This handles the ambient + diffuse + specular components for each
    light, with shadow testing.
    """
    mat = hit.material
    color = mat.ambient * mat.color  # Start with ambient

    for light in scene.lights:
        # Direction from hit point to light
        light_dir = normalize(light.position - hit.point)
        light_dist = np.linalg.norm(light.position - hit.point)

        # Shadow ray: check if any object blocks the path to the light
        # Why offset the origin?  Without the small epsilon offset along
        # the normal, the shadow ray would immediately hit the surface
        # it starts from (self-intersection, a classic ray tracing bug).
        shadow_origin = hit.point + hit.normal * 1e-4
        shadow_ray = Ray(shadow_origin, light_dir)
        shadow_hit = scene.closest_hit(shadow_ray)

        # If something blocks the light and is closer than the light, we're in shadow
        if shadow_hit and shadow_hit.t < light_dist:
            continue  # This light doesn't contribute

        # Diffuse (Lambert)
        NdotL = max(np.dot(hit.normal, light_dir), 0)
        diffuse = mat.diffuse * NdotL * mat.color * light.color * light.intensity

        # Specular (Phong)
        reflect_dir = reflect(-light_dir, hit.normal)
        view_dir = normalize(-ray.direction)
        RdotV = max(np.dot(reflect_dir, view_dir), 0)
        specular = mat.specular * (RdotV ** mat.shininess) * light.color * light.intensity

        color = color + diffuse + specular

    return color


def trace_ray(scene: Scene, ray: Ray, depth: int = 0,
              max_depth: int = 5) -> np.ndarray:
    """Recursively trace a ray through the scene.

    This is the heart of the Whitted ray tracer (1980).  At each hit:
    1. Compute local shading (Phong + shadows)
    2. If reflective, spawn a reflection ray and recurse
    3. If transparent, spawn a refraction ray and recurse
    4. Blend the results using Fresnel reflectance

    Why limit depth?  Without a maximum recursion depth, a ray bouncing
    between two mirrors would recurse forever.  In practice, each bounce
    diminishes the contribution, so 4-6 bounces suffice.
    """
    if depth >= max_depth:
        return scene.background_color

    hit = scene.closest_hit(ray)
    if hit is None:
        return scene.background_color

    mat = hit.material

    # Local shading (Phong + shadows)
    local_color = shade(scene, ray, hit)

    # If the material is neither reflective nor transparent, we're done
    if mat.reflectivity < 1e-4 and mat.transparency < 1e-4:
        return local_color

    # Fresnel: determines how much light is reflected vs refracted
    # Even glass surfaces reflect some light (especially at grazing angles)
    cos_theta = max(-np.dot(ray.direction, hit.normal), 0)
    fresnel = schlick_reflectance(cos_theta, mat.ior) if mat.transparency > 0 else 1.0

    result = local_color * (1 - mat.reflectivity) * (1 - mat.transparency)

    # Reflection
    if mat.reflectivity > 0 or mat.transparency > 0:
        reflect_dir = reflect(ray.direction, hit.normal)
        reflect_origin = hit.point + hit.normal * 1e-4
        reflect_ray = Ray(reflect_origin, normalize(reflect_dir))
        reflect_color = trace_ray(scene, reflect_ray, depth + 1, max_depth)

        if mat.transparency > 0:
            result += fresnel * mat.transparency * reflect_color
        else:
            result += mat.reflectivity * reflect_color

    # Refraction (only for transparent objects)
    if mat.transparency > 0:
        # Determine the ratio of indices of refraction
        # If we're inside the object, swap the ratio
        if hit.inside:
            eta = mat.ior  # Exiting: n_object / n_air
            refract_normal = -hit.normal
        else:
            eta = 1.0 / mat.ior  # Entering: n_air / n_object
            refract_normal = hit.normal

        refract_dir = refract(ray.direction, refract_normal, eta)

        if refract_dir is not None:
            # Offset origin slightly INSIDE the surface to avoid self-intersection
            refract_origin = hit.point - refract_normal * 1e-4
            refract_ray_obj = Ray(refract_origin, normalize(refract_dir))
            refract_color = trace_ray(scene, refract_ray_obj, depth + 1, max_depth)

            result += (1 - fresnel) * mat.transparency * refract_color
        else:
            # Total internal reflection: all light is reflected
            # (already handled by the reflection branch above)
            pass

    return result


# ---------------------------------------------------------------------------
# 6. Camera and rendering
# ---------------------------------------------------------------------------

def render_scene(scene: Scene, width: int = 400, height: int = 300,
                 fov_deg: float = 60.0,
                 camera_pos: np.ndarray = None,
                 camera_target: np.ndarray = None) -> np.ndarray:
    """Render the scene by shooting a ray through each pixel.

    Why one ray per pixel?  This is the simplest approach.  For
    anti-aliasing, you'd shoot multiple rays per pixel (supersampling)
    and average the results.  The path tracer in 09 does this.

    The camera model: we construct a ray direction for each pixel by
    computing where that pixel is on the near plane (the virtual film).
    """
    if camera_pos is None:
        camera_pos = np.array([0, 2, 6])
    if camera_target is None:
        camera_target = np.array([0, 0.5, 0])

    # Build camera coordinate system
    forward = normalize(camera_target - camera_pos)
    right = normalize(np.cross(forward, np.array([0, 1, 0])))
    up = np.cross(right, forward)

    aspect = width / height
    fov_rad = np.radians(fov_deg)
    half_height = np.tan(fov_rad / 2)
    half_width = half_height * aspect

    image = np.zeros((height, width, 3))

    start_time = time.time()
    total_pixels = width * height

    for j in range(height):
        if j % 50 == 0:
            elapsed = time.time() - start_time
            progress = (j * width) / total_pixels * 100
            print(f"  Rendering: {progress:.0f}% ({elapsed:.1f}s)", end='\r')

        for i in range(width):
            # Map pixel coordinates to normalized device coordinates [-1, 1]
            # Why +0.5?  We sample at the center of each pixel, not the corner.
            u = (2 * (i + 0.5) / width - 1) * half_width
            v = (1 - 2 * (j + 0.5) / height) * half_height

            direction = normalize(forward + u * right + v * up)
            ray = Ray(camera_pos, direction)

            color = trace_ray(scene, ray)
            image[j, i] = np.clip(color, 0, 1)

    elapsed = time.time() - start_time
    print(f"  Rendering: 100% ({elapsed:.1f}s)         ")

    return image


# ---------------------------------------------------------------------------
# 7. Scene setup and demo
# ---------------------------------------------------------------------------

def create_demo_scene() -> Scene:
    """Build a scene with reflective, glass, and matte objects.

    Why this particular scene?  It demonstrates all the ray tracer's
    capabilities in a single image:
    - The mirror sphere shows reflection
    - The glass sphere shows refraction + Fresnel + total internal reflection
    - The matte sphere shows basic Phong shading
    - The ground plane shows shadows and checkerboard pattern
    - Two lights show multiple shadow casting
    """
    scene = Scene()

    # Ground plane with checkerboard pattern
    ground_mat = Material(
        color=np.array([0.5, 0.5, 0.5]),
        ambient=0.1, diffuse=0.6, specular=0.2, shininess=10,
        reflectivity=0.15
    )
    scene.add_object(Plane(
        point=np.array([0, 0, 0]),
        normal=np.array([0, 1, 0]),
        material=ground_mat,
        checker=True
    ))

    # Reflective (mirror) sphere -- left
    mirror_mat = Material(
        color=np.array([0.8, 0.8, 0.9]),
        ambient=0.05, diffuse=0.1, specular=0.8, shininess=200,
        reflectivity=0.85
    )
    scene.add_object(Sphere(
        center=np.array([-2, 1, -1]),
        radius=1.0,
        material=mirror_mat
    ))

    # Glass (refractive) sphere -- center
    glass_mat = Material(
        color=np.array([0.95, 0.95, 1.0]),
        ambient=0.02, diffuse=0.05, specular=0.9, shininess=300,
        reflectivity=0.0,
        transparency=0.9,
        ior=1.5  # Glass-like index of refraction
    )
    scene.add_object(Sphere(
        center=np.array([0.5, 1.2, 0.5]),
        radius=1.2,
        material=glass_mat
    ))

    # Matte red sphere -- right
    matte_mat = Material(
        color=np.array([0.85, 0.2, 0.2]),
        ambient=0.1, diffuse=0.8, specular=0.3, shininess=30,
        reflectivity=0.05
    )
    scene.add_object(Sphere(
        center=np.array([2.5, 0.7, -0.5]),
        radius=0.7,
        material=matte_mat
    ))

    # Small metallic gold sphere -- far back
    gold_mat = Material(
        color=np.array([0.9, 0.7, 0.2]),
        ambient=0.08, diffuse=0.4, specular=0.8, shininess=100,
        reflectivity=0.5
    )
    scene.add_object(Sphere(
        center=np.array([-0.5, 0.5, -3]),
        radius=0.5,
        material=gold_mat
    ))

    # Lights
    scene.add_light(Light(
        position=np.array([5, 8, 5]),
        color=np.array([1.0, 0.95, 0.9]),
        intensity=1.0
    ))
    scene.add_light(Light(
        position=np.array([-4, 6, 3]),
        color=np.array([0.7, 0.8, 1.0]),
        intensity=0.6
    ))

    return scene


def demo_ray_tracer():
    """Render the demo scene and display the result."""
    print("Building scene...")
    scene = create_demo_scene()

    print("Rendering (this may take 1-2 minutes at 400x300)...")
    image = render_scene(
        scene,
        width=400, height=300,
        fov_deg=55,
        camera_pos=np.array([0, 3, 7]),
        camera_target=np.array([0, 0.5, -1])
    )

    # Gamma correction
    image = np.power(np.clip(image, 0, 1), 1.0 / 2.2)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    ax.set_title("Recursive Ray Tracer: Reflection, Refraction, Shadows",
                 fontsize=13, fontweight='bold')
    ax.axis('off')

    # Annotate the objects
    ax.text(0.02, 0.02, "Mirror sphere | Glass sphere | Matte sphere | Gold sphere\n"
            "Checkerboard ground | 2 point lights | Max 5 bounces",
            transform=ax.transAxes, fontsize=9, color='white',
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))

    plt.tight_layout()
    plt.savefig("/opt/projects/01_Personal/03_Study/examples/Computer_Graphics/"
                "output_08_ray_tracer.png", dpi=150)
    plt.show()


def demo_depth_comparison():
    """Show how max recursion depth affects reflection/refraction quality.

    Lower depth = reflections cut off sooner = darker/less detailed
    Higher depth = more bounces traced = richer visual result
    """
    scene = create_demo_scene()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Effect of Max Recursion Depth on Ray Tracing Quality",
                 fontsize=13, fontweight='bold')

    for ax, max_d in zip(axes, [1, 3, 6]):
        print(f"\n  Rendering with max depth = {max_d}...")

        # Smaller resolution for the comparison
        img = np.zeros((150, 200, 3))
        camera_pos = np.array([0, 3, 7])
        camera_target = np.array([0, 0.5, -1])

        forward = normalize(camera_target - camera_pos)
        right_vec = normalize(np.cross(forward, np.array([0, 1, 0])))
        up_vec = np.cross(right_vec, forward)

        aspect = 200 / 150
        hh = np.tan(np.radians(55) / 2)
        hw = hh * aspect

        for j in range(150):
            for i in range(200):
                u = (2 * (i + 0.5) / 200 - 1) * hw
                v = (1 - 2 * (j + 0.5) / 150) * hh
                direction = normalize(forward + u * right_vec + v * up_vec)
                ray = Ray(camera_pos, direction)
                color = trace_ray(scene, ray, max_depth=max_d)
                img[j, i] = np.clip(color, 0, 1)

        img = np.power(np.clip(img, 0, 1), 1.0 / 2.2)
        ax.imshow(img)
        ax.set_title(f"Max Depth = {max_d}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("/opt/projects/01_Personal/03_Study/examples/Computer_Graphics/"
                "output_08_depth_comparison.png", dpi=100)
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Recursive Ray Tracer")
    print("=" * 60)

    print("\n[1/2] Main render...")
    demo_ray_tracer()

    print("\n[2/2] Recursion depth comparison...")
    demo_depth_comparison()

    print("\nDone!")


if __name__ == "__main__":
    main()
