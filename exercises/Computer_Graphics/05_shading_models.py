"""
Exercises for Lesson 05: Shading Models
Topic: Computer_Graphics
Solutions to practice problems from the lesson.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ===================================================================
# Helper functions (from the lesson)
# ===================================================================

def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def reflect(incident, normal):
    return 2.0 * np.dot(normal, incident) * normal - incident


def distribution_ggx(n_dot_h, roughness):
    alpha = roughness * roughness
    a2 = alpha * alpha
    denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0
    denom = np.pi * denom * denom
    return a2 / max(denom, 1e-10)


def fresnel_schlick(cos_theta, f0):
    return f0 + (1.0 - f0) * (1.0 - cos_theta) ** 5


def geometry_schlick_ggx(n_dot_v, roughness):
    r = roughness + 1.0
    k = (r * r) / 8.0
    return n_dot_v / (n_dot_v * (1.0 - k) + k)


def geometry_smith(n_dot_v, n_dot_l, roughness):
    return geometry_schlick_ggx(max(n_dot_v, 0.0), roughness) * \
           geometry_schlick_ggx(max(n_dot_l, 0.0), roughness)


def render_sphere(width, height, shade_func, **kwargs):
    """Render a sphere using ray-sphere intersection with a given shading function."""
    image = np.zeros((height, width, 3), dtype=float)
    sphere_center = np.array([0.0, 0.0, -3.0])
    sphere_radius = 1.0
    view_pos = np.array([0.0, 0.0, 0.0])
    aspect = width / height

    for y in range(height):
        for x in range(width):
            u = (2.0 * x / width - 1.0) * aspect
            v = 1.0 - 2.0 * y / height
            ray_dir = normalize(np.array([u, v, -1.0]))

            oc = view_pos - sphere_center
            a = np.dot(ray_dir, ray_dir)
            b = 2.0 * np.dot(oc, ray_dir)
            c = np.dot(oc, oc) - sphere_radius ** 2
            disc = b * b - 4 * a * c

            if disc < 0:
                image[y, x] = [0.05, 0.05, 0.08]
                continue

            t = (-b - np.sqrt(disc)) / (2 * a)
            if t < 0:
                image[y, x] = [0.05, 0.05, 0.08]
                continue

            hit = view_pos + t * ray_dir
            normal = normalize(hit - sphere_center)
            color = shade_func(hit, normal, view_pos, **kwargs)
            image[y, x] = np.clip(color, 0, 1)

    return image


def exercise_1():
    """
    Phong Components: Render a sphere showing ambient, diffuse, and specular
    components separately, then combined. Show effect of changing shininess.
    """
    light_pos = np.array([2.0, 3.0, 0.0])
    light_color = np.array([1.0, 1.0, 1.0])
    obj_color = np.array([0.7, 0.3, 0.3])
    w, h = 150, 150

    def ambient_only(point, normal, view_pos, **kw):
        return 0.15 * obj_color

    def diffuse_only(point, normal, view_pos, **kw):
        N = normalize(normal)
        L = normalize(light_pos - point)
        diff = max(np.dot(N, L), 0.0)
        return 0.7 * diff * light_color * obj_color

    def specular_only(point, normal, view_pos, shininess=64, **kw):
        N = normalize(normal)
        L = normalize(light_pos - point)
        V = normalize(view_pos - point)
        R = reflect(L, N)
        spec = max(np.dot(R, V), 0.0) ** shininess
        return 0.5 * spec * light_color

    def phong_combined(point, normal, view_pos, shininess=64, **kw):
        return (ambient_only(point, normal, view_pos) +
                diffuse_only(point, normal, view_pos) +
                specular_only(point, normal, view_pos, shininess=shininess))

    print("Rendering Phong components separately...")
    img_ambient = render_sphere(w, h, ambient_only)
    img_diffuse = render_sphere(w, h, diffuse_only)
    img_spec = render_sphere(w, h, specular_only)
    img_combined = render_sphere(w, h, phong_combined)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(img_ambient)
    axes[0, 0].set_title('Ambient Only')
    axes[0, 1].imshow(img_diffuse)
    axes[0, 1].set_title('Diffuse Only')
    axes[0, 2].imshow(img_spec)
    axes[0, 2].set_title('Specular Only (n=64)')

    # Different shininess values
    for idx, n in enumerate([8, 64, 256]):
        img = render_sphere(w, h, phong_combined, shininess=n)
        axes[1, idx].imshow(img)
        axes[1, idx].set_title(f'Combined (shininess={n})')

    for ax in axes.flat:
        ax.axis('off')
    plt.suptitle('Phong Shading Components', fontsize=14)
    plt.tight_layout()
    plt.savefig('ex05_phong_components.png', dpi=150, bbox_inches='tight')
    print("Saved ex05_phong_components.png")
    print()
    print("Effect of shininess on specular:")
    print("  n=8:   Wide, soft highlight (rubber-like)")
    print("  n=64:  Moderate highlight (plastic-like)")
    print("  n=256: Tight, sharp highlight (polished metal-like)")
    print("  Ambient and diffuse are NOT affected by shininess.")


def exercise_2():
    """
    Blinn vs Phong Specular: Render a sphere with both models at the same
    shininess. Find matching Blinn exponent for Phong n=64.
    """
    light_pos = np.array([2.0, 3.0, 0.0])
    light_color = np.array([1.0, 1.0, 1.0])
    obj_color = np.array([0.8, 0.8, 0.8])
    w, h = 150, 150

    def phong_spec(point, normal, view_pos, shininess=64, **kw):
        N = normalize(normal)
        L = normalize(light_pos - point)
        V = normalize(view_pos - point)
        R = reflect(L, N)
        diff = max(np.dot(N, L), 0.0)
        spec = max(np.dot(R, V), 0.0) ** shininess
        return 0.1 * obj_color + 0.6 * diff * light_color * obj_color + 0.5 * spec * light_color

    def blinn_spec(point, normal, view_pos, shininess=64, **kw):
        N = normalize(normal)
        L = normalize(light_pos - point)
        V = normalize(view_pos - point)
        H = normalize(L + V)
        diff = max(np.dot(N, L), 0.0)
        spec = max(np.dot(N, H), 0.0) ** shininess
        return 0.1 * obj_color + 0.6 * diff * light_color * obj_color + 0.5 * spec * light_color

    print("Blinn-Phong vs Phong Specular Comparison")
    print("=" * 50)

    # Render comparison
    img_phong = render_sphere(w, h, phong_spec, shininess=64)
    img_blinn_same = render_sphere(w, h, blinn_spec, shininess=64)
    img_blinn_match = render_sphere(w, h, blinn_spec, shininess=256)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_phong)
    axes[0].set_title('Phong (n=64)')
    axes[1].imshow(img_blinn_same)
    axes[1].set_title('Blinn-Phong (n=64)')
    axes[2].imshow(img_blinn_match)
    axes[2].set_title('Blinn-Phong (n=256)\n(matches Phong n=64)')

    for ax in axes:
        ax.axis('off')
    plt.suptitle('Blinn-Phong vs Phong Specular', fontsize=14)
    plt.tight_layout()
    plt.savefig('ex05_blinn_vs_phong.png', dpi=150, bbox_inches='tight')
    print("Saved ex05_blinn_vs_phong.png")
    print()
    print("Observation:")
    print("  At the same exponent (n=64), Blinn-Phong produces a WIDER highlight")
    print("  than Phong. To approximately match Phong n=64, use Blinn n~256.")
    print("  Rule of thumb: n_Blinn ~= 4 * n_Phong")


def exercise_3():
    """
    Shading Comparison: Conceptual comparison of flat, Gouraud, and Phong
    interpolation on a low-polygon sphere.
    """
    print("Shading Interpolation Comparison (Low-Poly Sphere)")
    print("=" * 50)
    print()

    polygon_counts = [8, 20, 80, 320]

    print(f"{'Faces':<8} {'Flat':<20} {'Gouraud':<20} {'Phong Interp.':<20}")
    print("-" * 68)

    for faces in polygon_counts:
        flat_quality = "Very faceted" if faces < 20 else "Faceted" if faces < 100 else "Smooth-ish"
        gouraud_quality = "Visible facets" if faces < 20 else "Smooth" if faces < 100 else "Very smooth"
        phong_quality = "Some facets" if faces < 20 else "Smooth" if faces < 100 else "Very smooth"

        gouraud_spec = "Missed" if faces < 80 else "OK"
        phong_spec = "Accurate" if faces >= 8 else "N/A"

        print(f"  {faces:<6} {flat_quality:<20} "
              f"{gouraud_quality + ' (spec:' + gouraud_spec + ')':<20} "
              f"{phong_quality + ' (spec:' + phong_spec + ')':<20}")

    print()
    print("Key findings:")
    print("  - Flat shading always shows facets (one color per triangle)")
    print("  - Gouraud becomes acceptably smooth around 80-100 faces")
    print("  - Gouraud MISSES specular highlights on low-poly meshes because")
    print("    the highlight may fall between vertices where lighting is computed")
    print("  - Phong interpolation produces accurate specular even on low-poly meshes")
    print("    because lighting is computed per-pixel with interpolated normals")
    print()
    print("  For a sphere, approximately 80 faces is where Gouraud shading")
    print("  begins to look acceptably smooth for diffuse lighting.")
    print("  For accurate specular highlights, Phong interpolation is needed")
    print("  regardless of polygon count.")


def exercise_4():
    """
    PBR Material Exploration: Create a 5x5 grid of spheres varying roughness
    and metallic. Use a gold-colored albedo.
    """
    rows, cols = 5, 5
    sphere_pixel_radius = 30
    padding = 10
    w = cols * (2 * sphere_pixel_radius + padding) + padding
    h = rows * (2 * sphere_pixel_radius + padding) + padding

    image = np.zeros((h, w, 3), dtype=float)
    image[:] = [0.02, 0.02, 0.03]

    light_pos = np.array([5.0, 5.0, 5.0])
    light_color = np.array([1.0, 1.0, 1.0]) * 5.0
    view_pos = np.array([0.0, 0.0, 0.0])
    albedo = np.array([0.9, 0.6, 0.2])

    print("PBR Material Grid (roughness x metallic)")
    print("=" * 50)
    print(f"  Rows: roughness 0.1 (top) to 0.9 (bottom)")
    print(f"  Cols: metallic 0.0 (left) to 1.0 (right)")
    print(f"  Albedo: gold ({albedo})")
    print()

    for row in range(rows):
        for col in range(cols):
            roughness = 0.1 + 0.8 * row / (rows - 1)
            metallic = col / (cols - 1)

            cx = padding + col * (2 * sphere_pixel_radius + padding) + sphere_pixel_radius
            cy = padding + row * (2 * sphere_pixel_radius + padding) + sphere_pixel_radius

            for dy in range(-sphere_pixel_radius, sphere_pixel_radius + 1):
                for dx in range(-sphere_pixel_radius, sphere_pixel_radius + 1):
                    px = cx + dx
                    py = cy + dy
                    if not (0 <= px < w and 0 <= py < h):
                        continue

                    nx = dx / sphere_pixel_radius
                    ny = -dy / sphere_pixel_radius
                    r2 = nx * nx + ny * ny
                    if r2 > 1.0:
                        continue

                    nz = np.sqrt(1.0 - r2)
                    normal = np.array([nx, ny, nz])
                    point = np.array([nx, ny, nz - 3.0])

                    # PBR shading (Cook-Torrance)
                    N = normalize(normal)
                    V = normalize(view_pos - point)
                    L = normalize(light_pos - point)
                    H = normalize(V + L)

                    n_dot_l = max(np.dot(N, L), 0.0)
                    n_dot_v = max(np.dot(N, V), 0.0)
                    n_dot_h = max(np.dot(N, H), 0.0)
                    v_dot_h = max(np.dot(V, H), 0.001)

                    f0 = np.full(3, 0.04) * (1.0 - metallic) + albedo * metallic
                    D = distribution_ggx(n_dot_h, roughness)
                    F = fresnel_schlick(v_dot_h, f0)
                    G = geometry_smith(n_dot_v, n_dot_l, roughness)

                    spec = D * F * G / (4.0 * n_dot_v * n_dot_l + 0.0001)
                    ks = F
                    kd = (1.0 - ks) * (1.0 - metallic)
                    diffuse = kd * albedo / np.pi

                    lo = (diffuse + spec) * light_color * n_dot_l
                    lo = lo / (lo + 1.0)  # Reinhard
                    lo = np.power(np.clip(lo, 0.0, 1.0), 1.0 / 2.2)  # Gamma

                    image[py, px] = lo

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)
    ax.set_title('PBR Material Grid: Roughness (rows) x Metallic (columns)')
    ax.set_xticks([padding + col * (2 * sphere_pixel_radius + padding) + sphere_pixel_radius
                    for col in range(cols)])
    ax.set_xticklabels([f'{col / (cols - 1):.1f}' for col in range(cols)])
    ax.set_yticks([padding + row * (2 * sphere_pixel_radius + padding) + sphere_pixel_radius
                    for row in range(rows)])
    ax.set_yticklabels([f'{0.1 + 0.8 * row / (rows - 1):.1f}' for row in range(rows)])
    ax.set_xlabel('Metallic')
    ax.set_ylabel('Roughness')
    plt.tight_layout()
    plt.savefig('ex05_pbr_material_grid.png', dpi=150, bbox_inches='tight')
    print("Saved ex05_pbr_material_grid.png")
    print()
    print("Explanation:")
    print("  Top-left (low roughness, non-metallic): shiny plastic")
    print("  Top-right (low roughness, metallic): mirror-like metal (gold mirror)")
    print("  Bottom-left (high roughness, non-metallic): matte clay")
    print("  Bottom-right (high roughness, metallic): brushed/matte metal")
    print()
    print("  Metals have no diffuse component (all reflected light is specular)")
    print("  so they appear darker when rough (less focused specular).")


def exercise_5():
    """
    Fresnel Effect: Plot Schlick Fresnel for dielectric (F0=0.04) and metal
    (F0=0.9). Find angle where dielectric becomes 50% reflective.
    """
    cos_theta = np.linspace(0, 1, 500)

    f0_dielectric = 0.04
    f0_metal = 0.9

    fresnel_diel = f0_dielectric + (1.0 - f0_dielectric) * (1.0 - cos_theta) ** 5
    fresnel_metal = f0_metal + (1.0 - f0_metal) * (1.0 - cos_theta) ** 5

    # Find where dielectric hits 50%
    idx_50 = np.argmin(np.abs(fresnel_diel - 0.5))
    cos_50 = cos_theta[idx_50]
    angle_50 = np.degrees(np.arccos(cos_50))

    print("Fresnel Effect Analysis")
    print("=" * 50)
    print(f"  Dielectric F0 = {f0_dielectric}")
    print(f"  Metal F0 = {f0_metal}")
    print()
    print(f"  Dielectric becomes 50% reflective at:")
    print(f"    cos(theta) = {cos_50:.4f}")
    print(f"    theta = {angle_50:.1f} degrees from surface normal")
    print(f"    grazing angle = {90 - angle_50:.1f} degrees from surface")
    print()

    # Sample values
    print(f"  {'cos(theta)':<12} {'Angle':<10} {'Dielectric':<15} {'Metal':<15}")
    print("  " + "-" * 52)
    for ct in [1.0, 0.8, 0.5, 0.3, 0.1, 0.0]:
        fd = f0_dielectric + (1.0 - f0_dielectric) * (1.0 - ct) ** 5
        fm = f0_metal + (1.0 - f0_metal) * (1.0 - ct) ** 5
        ang = np.degrees(np.arccos(min(ct, 1.0)))
        print(f"  {ct:<12.1f} {ang:>6.1f} deg  {fd:<15.4f} {fm:<15.4f}")

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(np.degrees(np.arccos(cos_theta)), fresnel_diel, 'b-', linewidth=2,
            label=f'Dielectric (F0={f0_dielectric})')
    ax.plot(np.degrees(np.arccos(cos_theta)), fresnel_metal, 'r-', linewidth=2,
            label=f'Metal (F0={f0_metal})')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% reflectance')
    ax.axvline(x=angle_50, color='blue', linestyle=':', alpha=0.5,
               label=f'Dielectric 50% at {angle_50:.1f} deg')
    ax.set_xlabel('Angle from normal (degrees)')
    ax.set_ylabel('Reflectance')
    ax.set_title('Schlick Fresnel Approximation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig('ex05_fresnel.png', dpi=150, bbox_inches='tight')
    print("\nSaved ex05_fresnel.png")


def exercise_6():
    """
    Energy Conservation: Show Phong model can exceed incoming light intensity,
    while PBR model conserves energy.
    """
    print("Energy Conservation Analysis")
    print("=" * 50)
    print()

    # Phong model: kd + ks can exceed 1
    kd = 0.8
    ks = 0.5
    ka = 0.1

    print("Phong model with kd=0.8, ks=0.5, ka=0.1:")
    print(f"  Total reflection coefficients: ka + kd + ks = {ka + kd + ks}")
    print(f"  This exceeds 1.0, meaning reflected > incoming!")
    print()

    # At the specular peak (R.V = 1) with N.L = 1:
    max_phong = ka + kd * 1.0 + ks * (1.0 ** 64)
    print(f"  Maximum possible output (per channel): {max_phong:.2f}")
    print(f"  With incoming light intensity = 1.0, output = {max_phong:.2f}")
    print(f"  Energy violation: {max_phong:.2f} > 1.0")
    print()

    # PBR model: energy is conserved by design
    print("PBR Cook-Torrance model:")
    albedo = np.array([0.8, 0.8, 0.8])
    roughness = 0.5
    metallic = 0.0

    # Integrate reflected energy over hemisphere (Monte Carlo)
    N = np.array([0, 0, 1.0])
    n_samples = 10000
    total_energy = 0

    np.random.seed(42)
    for _ in range(n_samples):
        # Random direction in upper hemisphere
        phi = np.random.uniform(0, 2 * np.pi)
        cos_theta = np.random.uniform(0, 1)
        sin_theta = np.sqrt(1 - cos_theta ** 2)
        L = np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])

        V = np.array([0, 0, 1.0])  # Looking straight at surface
        H = normalize(V + L)

        n_dot_l = max(np.dot(N, L), 0.0)
        n_dot_v = max(np.dot(N, V), 0.0)
        n_dot_h = max(np.dot(N, H), 0.0)
        v_dot_h = max(np.dot(V, H), 0.001)

        f0 = np.full(3, 0.04) * (1 - metallic) + albedo * metallic
        D = distribution_ggx(n_dot_h, roughness)
        F = fresnel_schlick(v_dot_h, f0)
        G = geometry_smith(n_dot_v, n_dot_l, roughness)

        spec = D * F * G / (4.0 * n_dot_v * n_dot_l + 0.0001)
        ks_val = F
        kd_val = (1.0 - ks_val) * (1.0 - metallic)
        diffuse = kd_val * albedo / np.pi

        brdf = diffuse + spec
        # Hemispherical integral: brdf * cos(theta) * 2pi
        total_energy += np.mean(brdf) * n_dot_l * 2 * np.pi

    total_energy /= n_samples
    print(f"  Albedo: {albedo[0]}, roughness: {roughness}, metallic: {metallic}")
    print(f"  Hemispherical reflectance (Monte Carlo, {n_samples} samples): {total_energy:.4f}")
    print(f"  Energy conserved: {total_energy <= 1.0 + 0.05}")
    print(f"  (value should be <= 1.0; small excess is Monte Carlo noise)")
    print()
    print("Conclusion:")
    print("  Phong model's kd + ks can easily exceed 1, producing unrealistic")
    print("  'glowing' surfaces that emit more light than they receive.")
    print("  PBR's Cook-Torrance BRDF is designed so that the integral over")
    print("  the hemisphere never exceeds incoming energy (Fresnel + diffuse")
    print("  sum to <= 1 at every point).")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Phong Components ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Blinn vs Phong ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Shading Comparison ===")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("=== Exercise 4: PBR Material Grid ===")
    print("=" * 60)
    exercise_4()

    print("\n" + "=" * 60)
    print("=== Exercise 5: Fresnel Effect ===")
    print("=" * 60)
    exercise_5()

    print("\n" + "=" * 60)
    print("=== Exercise 6: Energy Conservation ===")
    print("=" * 60)
    exercise_6()

    print("\nAll exercises completed!")
