"""
Exercises for Lesson 11: Path Tracing and Global Illumination
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


# ---------------------------------------------------------------------------
# Exercise 1 -- Monte Carlo pi estimation
# ---------------------------------------------------------------------------

def exercise_1():
    """
    Estimate pi by randomly throwing points at a unit square and counting how
    many land inside the inscribed circle.  Plot convergence vs sample count.
    Verify the 1/sqrt(N) error rate.
    """
    np.random.seed(42)
    max_n = 10000
    x = np.random.uniform(-1, 1, max_n)
    y = np.random.uniform(-1, 1, max_n)
    inside = (x ** 2 + y ** 2) <= 1.0

    sample_counts = [10, 50, 100, 500, 1000, 5000, 10000]
    print("  N       pi_est     error      1/sqrt(N)")
    print("  ------- ---------- ---------- ----------")
    for n in sample_counts:
        pi_est = 4.0 * np.sum(inside[:n]) / n
        error = abs(pi_est - np.pi)
        theory = 1.0 / np.sqrt(n)
        print(f"  {n:7d} {pi_est:10.6f} {error:10.6f} {theory:10.6f}")

    if matplotlib_available:
        # Convergence plot
        ns = np.arange(1, max_n + 1)
        cumulative = np.cumsum(inside)
        pi_estimates = 4.0 * cumulative / ns
        errors = np.abs(pi_estimates - np.pi)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(ns, pi_estimates, alpha=0.7)
        ax1.axhline(np.pi, color='r', linestyle='--', label=f'pi = {np.pi:.4f}')
        ax1.set_xlabel('Samples')
        ax1.set_ylabel('Estimated pi')
        ax1.set_title('Monte Carlo Pi Estimation')
        ax1.legend()

        ax2.loglog(ns[9::10], errors[9::10], '.', alpha=0.3, label='Actual error')
        ax2.loglog(ns[9::10], 2.0 / np.sqrt(ns[9::10]), 'r-', label='O(1/sqrt(N))')
        ax2.set_xlabel('Samples')
        ax2.set_ylabel('Absolute error')
        ax2.set_title('Convergence Rate')
        ax2.legend()

        plt.tight_layout()
        plt.savefig('mc_pi_convergence.png', dpi=100)
        plt.close()
        print("  Saved mc_pi_convergence.png")


# ---------------------------------------------------------------------------
# Exercise 2 -- Importance sampling comparison
# ---------------------------------------------------------------------------

def exercise_2():
    """
    Estimate integral_0^1 x^10 dx using (a) uniform sampling and
    (b) importance sampling with p(x) = 11*x^10.  Compare variance
    after 1000 samples.

    Exact value: integral_0^1 x^10 dx = 1/11 = 0.09090909...
    """
    np.random.seed(7)
    exact = 1.0 / 11.0
    N = 1000

    # (a) Uniform sampling: f(x)/p(x) = x^10 / 1 = x^10, scaled by (b-a)=1
    uniform_samples = np.random.uniform(0, 1, N)
    uniform_estimates = uniform_samples ** 10
    uniform_mean = np.mean(uniform_estimates)
    uniform_var = np.var(uniform_estimates)

    # (b) Importance sampling with p(x) = 11*x^10
    # CDF: F(x) = x^11, so x = u^(1/11) where u ~ Uniform(0,1)
    u = np.random.uniform(0, 1, N)
    is_samples = u ** (1.0 / 11.0)
    # f(x)/p(x) = x^10 / (11*x^10) = 1/11 (constant!)
    is_estimates = np.full(N, 1.0 / 11.0)
    is_mean = np.mean(is_estimates)
    is_var = np.var(is_estimates)

    print(f"  Exact value: {exact:.10f}")
    print(f"  (a) Uniform sampling:")
    print(f"      Mean: {uniform_mean:.10f}, Error: {abs(uniform_mean-exact):.2e}")
    print(f"      Variance: {uniform_var:.2e}")
    print(f"  (b) Importance sampling (p(x) = 11x^10):")
    print(f"      Mean: {is_mean:.10f}, Error: {abs(is_mean-exact):.2e}")
    print(f"      Variance: {is_var:.2e}")
    if uniform_var > 0:
        print(f"  Variance reduction: {uniform_var / max(is_var, 1e-30):.0f}x")
    print(f"  IS gives zero variance because f/p is constant (perfect IS).")


# ---------------------------------------------------------------------------
# Exercise 3 -- Noise vs spp
# ---------------------------------------------------------------------------

def exercise_3():
    """
    Render a simple Cornell-box-like scene at 1, 4, 16, 64, and 256 spp.
    Measure the per-pixel RMSE against a high-spp reference.
    Verify the 1/sqrt(N) convergence.
    """

    def normalize(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-10 else v

    class PTSphere:
        def __init__(self, center, radius, albedo, emission=None):
            self.center = np.asarray(center, dtype=float)
            self.radius = float(radius)
            self.albedo = np.asarray(albedo, dtype=float)
            self.emission = np.zeros(3) if emission is None else np.asarray(emission)

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

    class PTPlane:
        def __init__(self, point, normal, albedo):
            self.point = np.asarray(point, dtype=float)
            self.normal = np.asarray(normal, dtype=float)
            self.albedo = np.asarray(albedo, dtype=float)
            self.emission = np.zeros(3)

        def intersect(self, o, d):
            dn = np.dot(self.normal, d)
            if abs(dn) < 1e-8:
                return float('inf'), None
            t = np.dot(self.point - o, self.normal) / dn
            if t < 1e-4:
                return float('inf'), None
            return t, self.normal.copy()

    objects = [
        PTPlane([0, 0, 0], [0, 1, 0], [0.73, 0.73, 0.73]),   # floor
        PTPlane([0, 3, 0], [0, -1, 0], [0.73, 0.73, 0.73]),  # ceiling
        PTPlane([0, 0, -2], [0, 0, 1], [0.73, 0.73, 0.73]),  # back
        PTPlane([-2, 0, 0], [1, 0, 0], [0.65, 0.05, 0.05]),  # left red
        PTPlane([2, 0, 0], [-1, 0, 0], [0.12, 0.45, 0.15]),  # right green
        PTSphere([0, 0.6, -0.5], 0.6, [0.73, 0.73, 0.73]),   # sphere
        PTSphere([0, 2.8, -0.5], 0.3, [0, 0, 0], [12, 12, 12]),  # light
    ]
    lights = [obj for obj in objects if np.any(obj.emission > 0)]

    def random_cos_hemi(n):
        if abs(n[0]) < 0.9:
            t = normalize(np.cross(np.array([1, 0, 0]), n))
        else:
            t = normalize(np.cross(np.array([0, 1, 0]), n))
        b = np.cross(n, t)
        r1, r2 = np.random.random(), np.random.random()
        phi = 2 * np.pi * r1
        r = np.sqrt(r2)
        return normalize(r * np.cos(phi) * t + r * np.sin(phi) * b + np.sqrt(max(0, 1 - r2)) * n)

    def find_near(o, d):
        bt = float('inf')
        bo = None
        bn = None
        for ob in objects:
            t, n = ob.intersect(o, d)
            if t < bt:
                bt = t
                bo = ob
                bn = n
        return bt, bo, bn

    def trace(o, d, depth=0):
        if depth > 5:
            return np.zeros(3)
        t, obj, n = find_near(o, d)
        if obj is None:
            return np.zeros(3)
        hit = o + t * d
        if np.dot(n, d) > 0:
            n = -n
        rad = obj.emission.copy() if depth == 0 else np.zeros(3)
        alb = obj.albedo
        surv = min(max(alb[0], alb[1], alb[2]), 0.95)
        if depth > 2 and np.random.random() > surv:
            return rad
        if depth <= 2:
            surv = 1.0
        bd = random_cos_hemi(n)
        br = trace(hit + 1e-4 * n, bd, depth + 1)
        rad += alb * br / surv
        return rad

    width, height = 40, 30
    eye = np.array([0, 1.5, 5.0])
    target = np.array([0, 1.0, 0])
    forward = normalize(target - eye)
    right = normalize(np.cross(forward, np.array([0, 1, 0])))
    up = np.cross(right, forward)
    fov = np.radians(60)
    hh = np.tan(fov / 2)
    hw = hh * width / height

    def render(spp):
        img = np.zeros((height, width, 3))
        for j in range(height):
            for i in range(width):
                c = np.zeros(3)
                for _ in range(spp):
                    u = (2 * (i + np.random.random()) / width - 1) * hw
                    v = (1 - 2 * (j + np.random.random()) / height) * hh
                    d = normalize(forward + u * right + v * up)
                    c += trace(eye, d)
                img[j, i] = c / spp
        return img

    # Reference render
    print("  Rendering reference (256 spp) ...")
    np.random.seed(0)
    ref = render(256)

    spp_list = [1, 4, 16, 64, 256]
    print(f"\n  {'spp':>5s}  {'RMSE':>10s}  {'1/sqrt(N)':>10s}")
    print(f"  {'---':>5s}  {'---':>10s}  {'---':>10s}")
    for spp in spp_list:
        np.random.seed(0)
        img = render(spp)
        rmse = np.sqrt(np.mean((img - ref) ** 2))
        theory = 1.0 / np.sqrt(spp)
        print(f"  {spp:5d}  {rmse:10.6f}  {theory:10.6f}")

    print("  Error decreases roughly as 1/sqrt(spp), confirming MC convergence.")


# ---------------------------------------------------------------------------
# Exercise 4 -- NEE ablation
# ---------------------------------------------------------------------------

def exercise_4():
    """
    Render a scene with and without Next Event Estimation at low spp.
    Compare noise levels. NEE helps more when lights are small because
    random bounces rarely hit small lights.
    """

    def normalize(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-10 else v

    class Sphere:
        def __init__(self, center, radius, albedo, emission=None):
            self.center = np.asarray(center, dtype=float)
            self.radius = float(radius)
            self.albedo = np.asarray(albedo, dtype=float)
            self.emission = np.zeros(3) if emission is None else np.asarray(emission)

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

    class Plane:
        def __init__(self, point, normal, albedo):
            self.point = np.asarray(point, dtype=float)
            self.normal = np.asarray(normal, dtype=float)
            self.albedo = np.asarray(albedo, dtype=float)
            self.emission = np.zeros(3)

        def intersect(self, o, d):
            dn = np.dot(self.normal, d)
            if abs(dn) < 1e-8:
                return float('inf'), None
            t = np.dot(self.point - o, self.normal) / dn
            if t < 1e-4:
                return float('inf'), None
            return t, self.normal.copy()

    # Small light source makes NEE especially helpful
    light_obj = Sphere([0, 2.8, -0.5], 0.15, [0, 0, 0], [40, 40, 40])
    objects = [
        Plane([0, 0, 0], [0, 1, 0], [0.73, 0.73, 0.73]),
        Plane([0, 3, 0], [0, -1, 0], [0.73, 0.73, 0.73]),
        Plane([0, 0, -2], [0, 0, 1], [0.73, 0.73, 0.73]),
        Sphere([0, 0.6, -0.5], 0.6, [0.7, 0.7, 0.7]),
        light_obj,
    ]

    def rcos(n):
        if abs(n[0]) < 0.9:
            t = normalize(np.cross(np.array([1, 0, 0]), n))
        else:
            t = normalize(np.cross(np.array([0, 1, 0]), n))
        b = np.cross(n, t)
        r1, r2 = np.random.random(), np.random.random()
        phi = 2 * np.pi * r1
        r = np.sqrt(r2)
        return normalize(r * np.cos(phi) * t + r * np.sin(phi) * b +
                         np.sqrt(max(0, 1 - r2)) * n)

    def fnear(o, d):
        bt, bo, bn = float('inf'), None, None
        for ob in objects:
            t, n = ob.intersect(o, d)
            if t < bt:
                bt, bo, bn = t, ob, n
        return bt, bo, bn

    def trace_no_nee(o, d, depth=0):
        if depth > 5:
            return np.zeros(3)
        t, obj, n = fnear(o, d)
        if obj is None:
            return np.zeros(3)
        hit = o + t * d
        if np.dot(n, d) > 0:
            n = -n
        rad = obj.emission.copy()
        alb = obj.albedo
        surv = min(max(alb[0], alb[1], alb[2]), 0.9)
        if depth > 2 and np.random.random() > surv:
            return rad
        if depth <= 2:
            surv = 1.0
        bd = rcos(n)
        rad += alb * trace_no_nee(hit + 1e-4 * n, bd, depth + 1) / surv
        return rad

    def trace_with_nee(o, d, depth=0):
        if depth > 5:
            return np.zeros(3)
        t, obj, n = fnear(o, d)
        if obj is None:
            return np.zeros(3)
        hit = o + t * d
        if np.dot(n, d) > 0:
            n = -n
        rad = obj.emission.copy() if depth == 0 else np.zeros(3)
        alb = obj.albedo
        surv = min(max(alb[0], alb[1], alb[2]), 0.9)
        if depth > 2 and np.random.random() > surv:
            return rad
        if depth <= 2:
            surv = 1.0
        # NEE: sample light directly
        lp = light_obj.center + light_obj.radius * normalize(np.random.randn(3))
        to_l = lp - hit
        ld = np.linalg.norm(to_l)
        ldir = to_l / ld
        cos_t = np.dot(n, ldir)
        if cos_t > 0:
            st, so, _ = fnear(hit + 1e-4 * n, ldir)
            if so is light_obj or st > ld - 0.01:
                ln = normalize(lp - light_obj.center)
                cos_l = abs(np.dot(ln, -ldir))
                area = 4 * np.pi * light_obj.radius ** 2
                brdf = 1.0 / np.pi
                contrib = light_obj.emission * brdf * cos_t * cos_l * area / (ld * ld)
                rad += alb * contrib
        bd = rcos(n)
        rad += alb * trace_with_nee(hit + 1e-4 * n, bd, depth + 1) / surv
        return rad

    width, height, spp = 30, 22, 32
    eye = np.array([0, 1.5, 4.0])
    fw = normalize(np.array([0, 1, 0]) - eye)
    rt = normalize(np.cross(fw, np.array([0, 1, 0])))
    up = np.cross(rt, fw)
    fov = np.radians(60)
    hh = np.tan(fov / 2)
    hw = hh * width / height

    def render(trace_fn, label):
        np.random.seed(42)
        img = np.zeros((height, width, 3))
        for j in range(height):
            for i in range(width):
                c = np.zeros(3)
                for _ in range(spp):
                    u = (2 * (i + np.random.random()) / width - 1) * hw
                    v = (1 - 2 * (j + np.random.random()) / height) * hh
                    d = normalize(fw + u * rt + v * up)
                    c += trace_fn(eye, d)
                img[j, i] = c / spp
        var = np.var(img)
        mean_val = np.mean(img)
        print(f"  {label}: mean={mean_val:.4f}, var={var:.6f}")
        return img

    img_no = render(trace_no_nee, "Without NEE")
    img_yes = render(trace_with_nee, "With NEE   ")
    var_no = np.var(img_no)
    var_yes = np.var(img_yes)
    if var_yes > 0:
        print(f"  Variance ratio (no_NEE / NEE): {var_no / var_yes:.1f}x")
    print(f"  NEE greatly reduces noise for small lights because it")
    print(f"  explicitly samples the light rather than waiting for random")
    print(f"  bounces to accidentally hit it.")


# ---------------------------------------------------------------------------
# Exercise 5 -- Russian Roulette comparison
# ---------------------------------------------------------------------------

def exercise_5():
    """
    Implement a version without Russian roulette (fixed max depth = 5) and
    compare brightness against the version with Russian roulette.
    Russian roulette is unbiased and preserves more energy.
    """

    def normalize(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-10 else v

    class Sphere:
        def __init__(self, center, radius, albedo, emission=None):
            self.center = np.asarray(center, dtype=float)
            self.radius = float(radius)
            self.albedo = np.asarray(albedo, dtype=float)
            self.emission = np.zeros(3) if emission is None else np.asarray(emission)

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

    class Plane:
        def __init__(self, pt, nm, alb):
            self.point = np.asarray(pt, dtype=float)
            self.normal = np.asarray(nm, dtype=float)
            self.albedo = np.asarray(alb, dtype=float)
            self.emission = np.zeros(3)

        def intersect(self, o, d):
            dn = np.dot(self.normal, d)
            if abs(dn) < 1e-8:
                return float('inf'), None
            t = np.dot(self.point - o, self.normal) / dn
            if t < 1e-4:
                return float('inf'), None
            return t, self.normal.copy()

    objects = [
        Plane([0, 0, 0], [0, 1, 0], [0.73, 0.73, 0.73]),
        Plane([0, 3, 0], [0, -1, 0], [0.73, 0.73, 0.73]),
        Plane([0, 0, -2], [0, 0, 1], [0.73, 0.73, 0.73]),
        Plane([-2, 0, 0], [1, 0, 0], [0.65, 0.05, 0.05]),
        Plane([2, 0, 0], [-1, 0, 0], [0.12, 0.45, 0.15]),
        Sphere([0, 0.6, -0.5], 0.6, [0.73, 0.73, 0.73]),
        Sphere([0, 2.8, -0.5], 0.3, [0, 0, 0], [12, 12, 12]),
    ]

    def rcos(n):
        if abs(n[0]) < 0.9:
            t = normalize(np.cross(np.array([1, 0, 0]), n))
        else:
            t = normalize(np.cross(np.array([0, 1, 0]), n))
        b = np.cross(n, t)
        r1, r2 = np.random.random(), np.random.random()
        phi = 2 * np.pi * r1
        r = np.sqrt(r2)
        return normalize(r * np.cos(phi) * t + r * np.sin(phi) * b +
                         np.sqrt(max(0, 1 - r2)) * n)

    def fnear(o, d):
        bt, bo, bn = float('inf'), None, None
        for ob in objects:
            t, n = ob.intersect(o, d)
            if t < bt:
                bt, bo, bn = t, ob, n
        return bt, bo, bn

    def trace_fixed(o, d, depth=0):
        """Fixed max depth, no Russian roulette -- biased."""
        if depth >= 5:
            return np.zeros(3)
        t, obj, n = fnear(o, d)
        if obj is None:
            return np.zeros(3)
        hit = o + t * d
        if np.dot(n, d) > 0:
            n = -n
        rad = obj.emission.copy()
        bd = rcos(n)
        rad += obj.albedo * trace_fixed(hit + 1e-4 * n, bd, depth + 1)
        return rad

    def trace_rr(o, d, depth=0):
        """Russian roulette -- unbiased."""
        if depth > 20:
            return np.zeros(3)
        t, obj, n = fnear(o, d)
        if obj is None:
            return np.zeros(3)
        hit = o + t * d
        if np.dot(n, d) > 0:
            n = -n
        rad = obj.emission.copy()
        alb = obj.albedo
        surv = min(max(alb[0], alb[1], alb[2]), 0.95)
        if depth > 2:
            if np.random.random() > surv:
                return rad
        else:
            surv = 1.0
        bd = rcos(n)
        rad += alb * trace_rr(hit + 1e-4 * n, bd, depth + 1) / surv
        return rad

    width, height, spp = 30, 22, 64
    eye = np.array([0, 1.5, 5.0])
    fw = normalize(np.array([0, 1, 0]) - eye)
    rt = normalize(np.cross(fw, np.array([0, 1, 0])))
    up = np.cross(rt, fw)
    hh = np.tan(np.radians(30))
    hw = hh * width / height

    def render(tfn, label):
        np.random.seed(42)
        img = np.zeros((height, width, 3))
        for j in range(height):
            for i in range(width):
                c = np.zeros(3)
                for _ in range(spp):
                    u = (2 * (i + np.random.random()) / width - 1) * hw
                    v = (1 - 2 * (j + np.random.random()) / height) * hh
                    d = normalize(fw + u * rt + v * up)
                    c += tfn(eye, d)
                img[j, i] = c / spp
        mean_brightness = np.mean(img)
        print(f"  {label}: mean brightness = {mean_brightness:.6f}")
        return img, mean_brightness

    _, b_fixed = render(trace_fixed, "Fixed depth=5 (biased)")
    _, b_rr = render(trace_rr, "Russian roulette (unbiased)")

    if b_rr > 0:
        ratio = b_fixed / b_rr
        print(f"  Fixed/RR brightness ratio: {ratio:.4f}")
        print(f"  Fixed depth loses energy from truncated paths -> darker image.")
        print(f"  Russian roulette preserves expected energy -> more accurate.")


# ---------------------------------------------------------------------------
# Exercise 6 -- Simple bilateral denoiser
# ---------------------------------------------------------------------------

def exercise_6():
    """
    Implement a bilateral filter as a post-process denoiser.
    Apply it to a noisy image and compare with a higher-spp render.
    """

    def bilateral_filter(image, sigma_s=2.0, sigma_r=0.1, radius=3):
        """
        Bilateral filter: smooths noise while preserving edges.
        sigma_s: spatial Gaussian sigma
        sigma_r: range (color) Gaussian sigma
        radius: kernel radius in pixels
        """
        h, w, c = image.shape
        output = np.zeros_like(image)

        for y in range(h):
            for x in range(w):
                center = image[y, x]
                weight_sum = 0.0
                color_sum = np.zeros(c)

                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            neighbor = image[ny, nx]
                            # Spatial weight
                            spatial = np.exp(-(dx*dx + dy*dy) / (2 * sigma_s**2))
                            # Range (color) weight
                            color_diff = np.linalg.norm(center - neighbor)
                            range_w = np.exp(-color_diff**2 / (2 * sigma_r**2))
                            w_total = spatial * range_w
                            weight_sum += w_total
                            color_sum += w_total * neighbor

                output[y, x] = color_sum / max(weight_sum, 1e-10)

        return output

    # Generate a simple test: ground truth + noise
    np.random.seed(42)
    size = 32
    # Create a simple gradient image as ground truth
    gt = np.zeros((size, size, 3))
    for y in range(size):
        for x in range(size):
            gt[y, x] = np.array([x / size, y / size, 0.5])
    # Add a disc
    for y in range(size):
        for x in range(size):
            if (x - size//2)**2 + (y - size//2)**2 < (size//4)**2:
                gt[y, x] = np.array([0.8, 0.2, 0.2])

    # Simulate noisy (low spp) render
    noise_level = 0.15
    noisy = gt + np.random.randn(size, size, 3) * noise_level
    noisy = np.clip(noisy, 0, 1)

    # Apply bilateral filter
    denoised = bilateral_filter(noisy, sigma_s=2.0, sigma_r=0.1, radius=2)

    # Measure quality
    rmse_noisy = np.sqrt(np.mean((noisy - gt) ** 2))
    rmse_denoised = np.sqrt(np.mean((denoised - gt) ** 2))

    print(f"  Image size: {size}x{size}")
    print(f"  Noise level (sigma): {noise_level}")
    print(f"  RMSE noisy:    {rmse_noisy:.6f}")
    print(f"  RMSE denoised: {rmse_denoised:.6f}")
    print(f"  Improvement:   {(1 - rmse_denoised/rmse_noisy)*100:.1f}% reduction in error")
    print(f"  Bilateral filter smooths noise while preserving the disc edge.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Exercise 1: Monte Carlo Pi Estimation ===")
    exercise_1()

    print("\n=== Exercise 2: Importance Sampling Comparison ===")
    exercise_2()

    print("\n=== Exercise 3: Noise vs SPP ===")
    exercise_3()

    print("\n=== Exercise 4: NEE Ablation ===")
    exercise_4()

    print("\n=== Exercise 5: Russian Roulette Comparison ===")
    exercise_5()

    print("\n=== Exercise 6: Simple Bilateral Denoiser ===")
    exercise_6()

    print("\nAll exercises completed!")
