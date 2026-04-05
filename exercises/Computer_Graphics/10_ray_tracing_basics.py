"""
Exercises for Lesson 10: Ray Tracing Basics
Topic: Computer_Graphics
Solutions to practice problems from the lesson.
"""

import numpy as np

matplotlib_available = False
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib_available = True
except ImportError:
    pass


def normalize(v):
    """Safely normalize a vector."""
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


# ---------------------------------------------------------------------------
# Exercise 1 -- Ray-Sphere by hand
# ---------------------------------------------------------------------------

def exercise_1():
    """
    Given ray origin (0,0,5), direction (0,0,-1), and sphere center (0,0,0)
    with radius 1, solve for both intersection t-values.  What is the surface
    normal at the nearer hit point?

    Derivation:
      oc = o - c = (0,0,5)
      a  = d.d = 1
      b  = 2*(oc.d) = 2*(0*0 + 0*0 + 5*(-1)) = -10
      c  = oc.oc - r^2 = 25 - 1 = 24
      disc = b^2 - 4ac = 100 - 96 = 4
      t1 = (-b - sqrt(disc)) / 2a = (10 - 2) / 2 = 4
      t2 = (-b + sqrt(disc)) / 2a = (10 + 2) / 2 = 6
      Nearer: t = 4 -> hit point = (0, 0, 5) + 4*(0, 0, -1) = (0, 0, 1)
      Normal = (hit - center) / r = (0, 0, 1)
    """
    origin = np.array([0.0, 0.0, 5.0])
    direction = np.array([0.0, 0.0, -1.0])
    center = np.array([0.0, 0.0, 0.0])
    radius = 1.0

    oc = origin - center
    a = np.dot(direction, direction)
    b = 2.0 * np.dot(oc, direction)
    c = np.dot(oc, oc) - radius * radius
    disc = b * b - 4 * a * c

    print(f"  Quadratic coefficients: a={a}, b={b}, c={c}")
    print(f"  Discriminant: {disc}")

    if disc < 0:
        print("  No intersection (unexpected!)")
        return

    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)
    print(f"  t1 (near) = {t1:.4f}")
    print(f"  t2 (far)  = {t2:.4f}")

    hit_point = origin + t1 * direction
    normal = (hit_point - center) / radius
    print(f"  Near hit point: ({hit_point[0]:.2f}, {hit_point[1]:.2f}, {hit_point[2]:.2f})")
    print(f"  Surface normal: ({normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f})")


# ---------------------------------------------------------------------------
# Exercise 2 -- Moller-Trumbore
# ---------------------------------------------------------------------------

def exercise_2():
    """
    Implement Moller-Trumbore and test with triangle v0=(0,0,0), v1=(1,0,0),
    v2=(0,1,0) and ray origin (0.2,0.2,1), direction (0,0,-1).
    Verify barycentric coordinates.
    """
    def ray_triangle_mt(origin, direction, v0, v1, v2):
        EPSILON = 1e-8
        e1 = v1 - v0
        e2 = v2 - v0
        P = np.cross(direction, e2)
        det = np.dot(e1, P)
        if abs(det) < EPSILON:
            return False, 0, 0, 0
        inv_det = 1.0 / det
        T = origin - v0
        u = np.dot(T, P) * inv_det
        if u < 0.0 or u > 1.0:
            return False, 0, 0, 0
        Q = np.cross(T, e1)
        v = np.dot(direction, Q) * inv_det
        if v < 0.0 or u + v > 1.0:
            return False, 0, 0, 0
        t = np.dot(e2, Q) * inv_det
        if t < EPSILON:
            return False, 0, 0, 0
        return True, t, u, v

    v0 = np.array([0.0, 0.0, 0.0])
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    origin = np.array([0.2, 0.2, 1.0])
    direction = np.array([0.0, 0.0, -1.0])

    hit, t, u, v = ray_triangle_mt(origin, direction, v0, v1, v2)
    w = 1.0 - u - v

    print(f"  Triangle: v0=(0,0,0), v1=(1,0,0), v2=(0,1,0)")
    print(f"  Ray: origin=(0.2, 0.2, 1), dir=(0, 0, -1)")
    print(f"  Hit: {hit}")
    print(f"  t = {t:.4f}")
    print(f"  Barycentric: u={u:.4f}, v={v:.4f}, w(1-u-v)={w:.4f}")

    # Verify: reconstructed point should equal origin + t*direction
    hit_point = origin + t * direction
    bary_point = w * v0 + u * v1 + v * v2
    print(f"  Hit point (ray):  ({hit_point[0]:.4f}, {hit_point[1]:.4f}, {hit_point[2]:.4f})")
    print(f"  Hit point (bary): ({bary_point[0]:.4f}, {bary_point[1]:.4f}, {bary_point[2]:.4f})")
    print(f"  Points match: {np.allclose(hit_point, bary_point)}")

    # Also test a miss
    miss_origin = np.array([2.0, 2.0, 1.0])
    hit_miss, _, _, _ = ray_triangle_mt(miss_origin, direction, v0, v1, v2)
    print(f"  Ray from (2,2,1) hits triangle: {hit_miss} (expected False)")


# ---------------------------------------------------------------------------
# Exercise 3 -- Refraction Visualization (glass sphere)
# ---------------------------------------------------------------------------

def exercise_3():
    """
    Add a glass sphere (IOR=1.5) to a simple ray tracer. Render and observe
    distortion through the sphere. Experiment with different IOR values.
    """
    from dataclasses import dataclass

    @dataclass
    class Material:
        color: np.ndarray
        ambient: float = 0.1
        diffuse: float = 0.7
        specular: float = 0.3
        shininess: float = 50.0
        reflectivity: float = 0.0
        transparency: float = 0.0
        ior: float = 1.5

    @dataclass
    class Sphere:
        center: np.ndarray
        radius: float
        material: Material

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
            normal = (hit - self.center) / self.radius
            return t, normal

    @dataclass
    class Plane:
        point: np.ndarray
        normal: np.ndarray
        material: Material

        def intersect(self, origin, direction):
            denom = np.dot(self.normal, direction)
            if abs(denom) < 1e-8:
                return float('inf'), None
            t = np.dot(self.point - origin, self.normal) / denom
            if t < 1e-4:
                return float('inf'), None
            return t, self.normal.copy()

    def reflect(d, n):
        return d - 2 * np.dot(d, n) * n

    def refract(d, n, eta_ratio):
        cos_i = -np.dot(d, n)
        sin2_t = eta_ratio ** 2 * (1.0 - cos_i ** 2)
        if sin2_t > 1.0:
            return None
        cos_t = np.sqrt(1.0 - sin2_t)
        return eta_ratio * d + (eta_ratio * cos_i - cos_t) * n

    def fresnel_schlick(cos_theta, eta1, eta2):
        r0 = ((eta1 - eta2) / (eta1 + eta2)) ** 2
        return r0 + (1 - r0) * (1 - cos_theta) ** 5

    # Checkerboard floor
    checker_a = Material(color=np.array([0.9, 0.9, 0.9]))
    checker_b = Material(color=np.array([0.2, 0.2, 0.2]))

    glass = Material(color=np.array([1.0, 1.0, 1.0]),
                     ambient=0.0, diffuse=0.0, specular=0.5,
                     reflectivity=0.1, transparency=0.9, ior=1.5)
    red = Material(color=np.array([0.9, 0.2, 0.2]))

    objects = [
        Sphere(np.array([0.0, 0.5, 0.0]), 1.0, glass),
        Sphere(np.array([-2.0, 0.3, -1.0]), 0.6, red),
        Sphere(np.array([2.0, 0.3, 1.0]), 0.6, red),
        Plane(np.array([0, -0.5, 0]), np.array([0, 1, 0]), checker_a),
    ]

    light_pos = np.array([-3, 5, 5])
    width, height = 160, 120
    image = np.zeros((height, width, 3))

    eye = np.array([0, 1.5, 5])
    target = np.array([0, 0, 0])
    forward = normalize(target - eye)
    right = normalize(np.cross(forward, np.array([0, 1, 0])))
    up = np.cross(right, forward)
    fov = np.radians(60)
    half_h = np.tan(fov / 2)
    half_w = half_h * width / height

    def find_nearest(origin, direction):
        nearest_t = float('inf')
        nearest_obj = None
        nearest_n = None
        for obj in objects:
            t, n = obj.intersect(origin, direction)
            if t < nearest_t:
                nearest_t = t
                nearest_obj = obj
                nearest_n = n
        return nearest_t, nearest_obj, nearest_n

    def shade(origin, direction, depth=0):
        if depth >= 5:
            return np.array([0.1, 0.1, 0.2])
        t, obj, n = find_nearest(origin, direction)
        if obj is None:
            return np.array([0.1, 0.1, 0.2])
        hit = origin + t * direction

        # Floor checkerboard
        if isinstance(obj, Plane):
            cx = int(np.floor(hit[0])) % 2
            cz = int(np.floor(hit[2])) % 2
            mat = checker_a if (cx + cz) % 2 == 0 else checker_b
        else:
            mat = obj.material

        if np.dot(n, direction) > 0:
            n = -n

        color = mat.ambient * mat.color

        # Direct lighting
        to_light = normalize(light_pos - hit)
        s_t, s_obj, _ = find_nearest(hit + 1e-4 * n, to_light)
        if s_t > np.linalg.norm(light_pos - hit):
            ndl = max(0, np.dot(n, to_light))
            color += mat.diffuse * ndl * mat.color
            h_vec = normalize(to_light + normalize(-direction))
            ndh = max(0, np.dot(n, h_vec))
            color += mat.specular * ndh ** mat.shininess

        # Reflection
        if mat.reflectivity > 0:
            r_dir = normalize(reflect(direction, n))
            r_col = shade(hit + 1e-4 * n, r_dir, depth + 1)
            color += mat.reflectivity * r_col

        # Refraction (transparency)
        if mat.transparency > 0:
            entering = np.dot(direction, n) < 0
            eta = 1.0 / mat.ior if entering else mat.ior
            ref_n = n if entering else -n
            cos_i = abs(np.dot(direction, ref_n))
            kr = fresnel_schlick(cos_i, 1.0, mat.ior)
            t_dir = refract(normalize(direction), ref_n, eta)
            if t_dir is not None:
                t_col = shade(hit - 1e-4 * ref_n, normalize(t_dir), depth + 1)
                refl_dir = normalize(reflect(direction, n))
                refl_col = shade(hit + 1e-4 * n, refl_dir, depth + 1)
                color = color * (1 - mat.transparency) + mat.transparency * (
                    kr * refl_col + (1 - kr) * t_col)
            else:
                # Total internal reflection
                refl_dir = normalize(reflect(direction, n))
                refl_col = shade(hit + 1e-4 * n, refl_dir, depth + 1)
                color = color * (1 - mat.transparency) + mat.transparency * refl_col

        return np.clip(color, 0, 1)

    print("  Rendering 160x120 scene with glass sphere (IOR=1.5)...")
    for j in range(height):
        for i in range(width):
            u = (2 * (i + 0.5) / width - 1) * half_w
            v = (1 - 2 * (j + 0.5) / height) * half_h
            d = normalize(forward + u * right + v * up)
            image[j, i] = shade(eye, d)

    # Gamma correct
    image = np.power(np.clip(image, 0, 1), 1.0 / 2.2)
    print(f"  Render complete. Image range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"  Glass sphere distorts background objects visible through it.")
    print(f"  Different IOR values produce different distortion magnitudes.")


# ---------------------------------------------------------------------------
# Exercise 4 -- Shadow Comparison (1 vs 3 lights)
# ---------------------------------------------------------------------------

def exercise_4():
    """
    Render a simple scene with 1 light and then 3 lights at different
    positions. Compare shadow patterns. Multiple lights create overlapping
    penumbra regions giving the appearance of softer shadows.
    """

    class ShadowSphere:
        def __init__(self, center, radius, color):
            self.center = np.asarray(center, dtype=float)
            self.radius = float(radius)
            self.color = np.asarray(color, dtype=float)

        def intersect(self, o, d):
            oc = o - self.center
            a = np.dot(d, d)
            b = 2 * np.dot(oc, d)
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
            return t, normalize(o + t * d - self.center)

    objects = [
        ShadowSphere([0, 0.5, 0], 1.0, [0.8, 0.3, 0.3]),
        ShadowSphere([-1.5, 0.3, 1.5], 0.6, [0.3, 0.8, 0.3]),
    ]

    def render_scene(lights, label, width=100, height=75):
        eye = np.array([0, 2, 5.0])
        target = np.array([0, 0, 0])
        forward = normalize(target - eye)
        right = normalize(np.cross(forward, np.array([0, 1, 0])))
        up = np.cross(right, forward)
        fov = np.radians(60)
        hh = np.tan(fov / 2)
        hw = hh * width / height

        shadow_count = 0
        lit_count = 0

        for j in range(height):
            for i in range(width):
                u = (2 * (i + 0.5) / width - 1) * hw
                v = (1 - 2 * (j + 0.5) / height) * hh
                d = normalize(forward + u * right + v * up)

                # Find nearest hit
                best_t = float('inf')
                best_n = None
                for obj in objects:
                    t, n = obj.intersect(eye, d)
                    if t < best_t:
                        best_t = t
                        best_n = n

                # Floor plane
                floor_denom = d[1]
                if abs(floor_denom) > 1e-8:
                    floor_t = (-0.5 - eye[1]) / floor_denom
                    if 1e-4 < floor_t < best_t:
                        best_t = floor_t
                        best_n = np.array([0, 1, 0])

                if best_t < 1e8 and best_n is not None:
                    hit = eye + best_t * d
                    # Count how many lights illuminate this point
                    visible = 0
                    for lp in lights:
                        to_l = normalize(lp - hit)
                        so = hit + 1e-3 * best_n
                        blocked = False
                        for obj in objects:
                            st, _ = obj.intersect(so, to_l)
                            if st < np.linalg.norm(lp - hit):
                                blocked = True
                                break
                        if not blocked:
                            visible += 1
                    if visible == 0:
                        shadow_count += 1
                    else:
                        lit_count += 1

        total = shadow_count + lit_count
        shadow_pct = shadow_count / total * 100 if total > 0 else 0
        print(f"  {label}: {shadow_count} shadowed pixels ({shadow_pct:.1f}%), "
              f"{lit_count} lit pixels")
        return shadow_pct

    lights_1 = [np.array([3.0, 5.0, 3.0])]
    lights_3 = [
        np.array([3.0, 5.0, 3.0]),
        np.array([-3.0, 4.0, 2.0]),
        np.array([0.0, 6.0, -2.0]),
    ]

    s1 = render_scene(lights_1, "1 light")
    s3 = render_scene(lights_3, "3 lights")
    print(f"  With 3 lights, shadow area is smaller ({s3:.1f}% vs {s1:.1f}%).")
    print(f"  Multiple lights create overlapping partial illumination,")
    print(f"  giving the appearance of softer shadows.")


# ---------------------------------------------------------------------------
# Exercise 5 -- Anti-aliasing (4x supersampling)
# ---------------------------------------------------------------------------

def exercise_5():
    """
    Implement 4x supersampling (2x2 jittered grid per pixel). Compare the
    result with single-sample rendering. Measure the performance difference.
    """
    import time

    class AASphere:
        def __init__(self, center, radius, color):
            self.center = np.asarray(center, dtype=float)
            self.radius = float(radius)
            self.color = np.asarray(color, dtype=float)

        def intersect(self, o, d):
            oc = o - self.center
            a = np.dot(d, d)
            b = 2 * np.dot(oc, d)
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
            return t, normalize(o + t * d - self.center)

    sphere = AASphere([0, 0, 0], 1.0, [0.8, 0.2, 0.2])
    light = np.array([-3, 5, 5.0])
    eye = np.array([0, 0, 3.0])
    width, height = 80, 60
    fov = np.radians(60)
    hh = np.tan(fov / 2)
    hw = hh * width / height

    def shade_pixel(u, v):
        d = normalize(np.array([u, v, -1.0]))
        t, n = sphere.intersect(eye, d)
        if n is None:
            return np.array([0.1, 0.1, 0.2])
        hit = eye + t * d
        to_l = normalize(light - hit)
        diff = max(0, np.dot(n, to_l))
        return sphere.color * (0.1 + 0.9 * diff)

    # 1-sample rendering
    t0 = time.perf_counter()
    img_1spp = np.zeros((height, width, 3))
    for j in range(height):
        for i in range(width):
            u = (2 * (i + 0.5) / width - 1) * hw
            v = (1 - 2 * (j + 0.5) / height) * hh
            img_1spp[j, i] = shade_pixel(u, v)
    t_1spp = time.perf_counter() - t0

    # 4-sample (2x2 jittered) rendering
    t0 = time.perf_counter()
    img_4spp = np.zeros((height, width, 3))
    offsets = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
    for j in range(height):
        for i in range(width):
            color = np.zeros(3)
            for ox, oy in offsets:
                u = (2 * (i + ox) / width - 1) * hw
                v = (1 - 2 * (j + oy) / height) * hh
                color += shade_pixel(u, v)
            img_4spp[j, i] = color / 4.0
    t_4spp = time.perf_counter() - t0

    # Compare edge pixels
    diff = np.abs(img_1spp - img_4spp)
    edge_diff = diff.sum(axis=2)
    num_edge_diffs = np.sum(edge_diff > 0.01)

    print(f"  1 spp render time: {t_1spp:.4f}s")
    print(f"  4 spp render time: {t_4spp:.4f}s  ({t_4spp/t_1spp:.1f}x slower)")
    print(f"  Pixels with visible difference: {num_edge_diffs} / {width*height}")
    print(f"  4x supersampling smooths edges at sphere silhouette.")


# ---------------------------------------------------------------------------
# Exercise 6 -- BVH Integration
# ---------------------------------------------------------------------------

def exercise_6():
    """
    Connect a BVH to the ray tracer. Generate 100 random spheres and compare
    rendering time with and without BVH acceleration.
    """
    import time

    class RTSphere:
        def __init__(self, center, radius, color):
            self.center = np.asarray(center, dtype=float)
            self.radius = float(radius)
            self.color = np.asarray(color, dtype=float)

        def intersect(self, o, d):
            oc = o - self.center
            a = np.dot(d, d)
            b = 2 * np.dot(oc, d)
            c = np.dot(oc, oc) - self.radius ** 2
            disc = b * b - 4 * a * c
            if disc < 0:
                return False, float('inf')
            sq = np.sqrt(disc)
            t = (-b - sq) / (2 * a)
            if t < 1e-4:
                t = (-b + sq) / (2 * a)
            if t < 1e-4:
                return False, float('inf')
            return True, t

        def aabb(self):
            r = np.full(3, self.radius)
            return (self.center - r, self.center + r)

    class BVHNode:
        def __init__(self, bmin, bmax, left=None, right=None, prims=None):
            self.bmin = bmin
            self.bmax = bmax
            self.left = left
            self.right = right
            self.prims = prims or []

    def aabb_intersect(bmin, bmax, origin, inv_dir):
        t1 = (bmin - origin) * inv_dir
        t2 = (bmax - origin) * inv_dir
        tmin = np.minimum(t1, t2)
        tmax = np.maximum(t1, t2)
        return np.max(tmin) <= np.min(tmax) and np.min(tmax) >= 0

    def build_simple_bvh(spheres, max_leaf=4):
        if not spheres:
            return None
        aabbs = [s.aabb() for s in spheres]
        bmin = np.min([a[0] for a in aabbs], axis=0)
        bmax = np.max([a[1] for a in aabbs], axis=0)
        if len(spheres) <= max_leaf:
            return BVHNode(bmin, bmax, prims=spheres)
        # Split on longest axis at midpoint
        d = bmax - bmin
        axis = int(np.argmax(d))
        mid = (bmin[axis] + bmax[axis]) / 2
        left_s = [s for s in spheres if s.center[axis] < mid]
        right_s = [s for s in spheres if s.center[axis] >= mid]
        if not left_s or not right_s:
            return BVHNode(bmin, bmax, prims=spheres)
        return BVHNode(bmin, bmax,
                       left=build_simple_bvh(left_s, max_leaf),
                       right=build_simple_bvh(right_s, max_leaf))

    def bvh_trace(node, origin, direction, inv_dir):
        if node is None:
            return False, float('inf'), None
        if not aabb_intersect(node.bmin, node.bmax, origin, inv_dir):
            return False, float('inf'), None
        if node.prims:
            best_t = float('inf')
            best_s = None
            for s in node.prims:
                h, t = s.intersect(origin, direction)
                if h and t < best_t:
                    best_t = t
                    best_s = s
            return best_s is not None, best_t, best_s
        hl, tl, sl = bvh_trace(node.left, origin, direction, inv_dir)
        hr, tr, sr = bvh_trace(node.right, origin, direction, inv_dir)
        if hl and hr:
            return (True, tl, sl) if tl <= tr else (True, tr, sr)
        if hl:
            return True, tl, sl
        if hr:
            return True, tr, sr
        return False, float('inf'), None

    # Generate 100 random spheres
    np.random.seed(99)
    spheres = [RTSphere(np.random.uniform(-5, 5, 3),
                        np.random.uniform(0.2, 0.6),
                        np.random.uniform(0.2, 1.0, 3))
               for _ in range(100)]

    bvh = build_simple_bvh(spheres)

    eye = np.array([0, 0, 10.0])
    width, height = 80, 60
    fov = np.radians(60)
    hh = np.tan(fov / 2)
    hw = hh * width / height

    # Without BVH (brute force)
    t0 = time.perf_counter()
    for j in range(height):
        for i in range(width):
            u = (2 * (i + 0.5) / width - 1) * hw
            v = (1 - 2 * (j + 0.5) / height) * hh
            d = normalize(np.array([u, v, -1.0]))
            for s in spheres:
                s.intersect(eye, d)
    t_brute = time.perf_counter() - t0

    # With BVH
    t0 = time.perf_counter()
    for j in range(height):
        for i in range(width):
            u = (2 * (i + 0.5) / width - 1) * hw
            v = (1 - 2 * (j + 0.5) / height) * hh
            d = normalize(np.array([u, v, -1.0]))
            inv_d = np.where(np.abs(d) > 1e-10, 1.0 / d, 1e10)
            bvh_trace(bvh, eye, d, inv_d)
    t_bvh = time.perf_counter() - t0

    print(f"  100 spheres, {width}x{height} image:")
    print(f"    Brute force: {t_brute:.3f}s")
    print(f"    With BVH:    {t_bvh:.3f}s")
    speedup = t_brute / t_bvh if t_bvh > 0 else float('inf')
    print(f"    Speedup:     {speedup:.1f}x")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Exercise 1: Ray-Sphere by Hand ===")
    exercise_1()

    print("\n=== Exercise 2: Moller-Trumbore ===")
    exercise_2()

    print("\n=== Exercise 3: Refraction Visualization ===")
    exercise_3()

    print("\n=== Exercise 4: Shadow Comparison ===")
    exercise_4()

    print("\n=== Exercise 5: Anti-aliasing (4x Supersampling) ===")
    exercise_5()

    print("\n=== Exercise 6: BVH Integration ===")
    exercise_6()

    print("\nAll exercises completed!")
