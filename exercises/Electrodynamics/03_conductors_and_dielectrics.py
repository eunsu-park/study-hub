"""
Exercises for Lesson 03: Conductors and Dielectrics
Topic: Electrodynamics
Solutions to practice problems from the lesson.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Constants
epsilon_0 = 8.854e-12
k_e = 1.0 / (4.0 * np.pi * epsilon_0)


def exercise_1():
    """
    Exercise 1: Image Charge -- Grounded Sphere
    q = 5 nC at distance a = 0.5 m from center of a grounded sphere R = 0.2 m.
    Find image charge magnitude and position. Plot potential.
    """
    q = 5e-9     # 5 nC
    a = 0.5      # distance from center (m)
    R = 0.2      # sphere radius (m)

    # Image charge: q' = -(R/a)*q at distance b = R^2/a from center
    q_image = -(R / a) * q
    b = R**2 / a

    print(f"  Real charge: q = {q*1e9:.1f} nC at distance a = {a} m")
    print(f"  Grounded sphere: R = {R} m")
    print(f"  Image charge: q' = {q_image*1e9:.3f} nC")
    print(f"  Image position: b = R^2/a = {b:.4f} m from center")

    # Plot potential in the xz-plane (charge on z-axis)
    x = np.linspace(-0.8, 0.8, 400)
    z = np.linspace(-0.8, 0.8, 400)
    X, Z = np.meshgrid(x, z)

    # Real charge at (0, 0, a), image at (0, 0, b)
    r_real = np.sqrt(X**2 + (Z - a)**2)
    r_image = np.sqrt(X**2 + (Z - b)**2)
    r_real = np.maximum(r_real, 1e-4)
    r_image = np.maximum(r_image, 1e-4)

    V = k_e * q / r_real + k_e * q_image / r_image

    # Mask inside the sphere
    r_from_center = np.sqrt(X**2 + Z**2)
    V[r_from_center < R] = 0.0

    V_clipped = np.clip(V, -200, 200)

    fig, ax = plt.subplots(figsize=(8, 8))
    levels = np.linspace(-150, 150, 31)
    cs = ax.contour(X, Z, V_clipped, levels=levels, cmap='RdBu_r')
    circle = plt.Circle((0, 0), R, fill=True, color='gray', alpha=0.5, label=f'Sphere R={R}')
    ax.add_patch(circle)
    ax.plot(0, a, 'ro', markersize=10, label=f'+q at z={a}')
    ax.plot(0, b, 'bx', markersize=10, label=f"q' at z={b:.3f}")
    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)')
    ax.set_title('Image Charge: Grounded Sphere')
    ax.legend()
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('ex03_image_sphere.png', dpi=150)
    plt.close()
    print("  Plot saved: ex03_image_sphere.png")

    # Verify V = 0 on sphere surface
    theta_check = np.linspace(0, 2 * np.pi, 100)
    x_surf = R * np.sin(theta_check)
    z_surf = R * np.cos(theta_check)
    r_real_surf = np.sqrt(x_surf**2 + (z_surf - a)**2)
    r_image_surf = np.sqrt(x_surf**2 + (z_surf - b)**2)
    V_surf = k_e * q / r_real_surf + k_e * q_image / r_image_surf
    print(f"  V on sphere surface: max |V| = {np.max(np.abs(V_surf)):.6e} V (should be ~0)")


def exercise_2():
    """
    Exercise 2: Dielectric Sphere
    Dielectric sphere of radius R in uniform field E0.
    Field inside: E_inside = 3*E0 / (eps_r + 2). Plot field lines.
    """
    R = 0.3        # sphere radius
    E0 = 100.0     # external field (V/m)
    eps_r = 4.0    # dielectric constant

    E_inside = 3.0 * E0 / (eps_r + 2.0)

    print(f"  Sphere R = {R} m, eps_r = {eps_r}")
    print(f"  External field E0 = {E0} V/m")
    print(f"  Field inside sphere: E_inside = 3*E0/(eps_r+2) = {E_inside:.2f} V/m")
    print(f"  Reduction factor: {E_inside/E0:.4f}")

    # Potential solution:
    # Inside: V_in = -E_inside * r * cos(theta) = -E_inside * z
    # Outside: V_out = -E0*r*cos(theta) + (eps_r-1)/(eps_r+2) * E0 * R^3 * cos(theta)/r^2
    x = np.linspace(-1.0, 1.0, 200)
    z = np.linspace(-1.0, 1.0, 200)
    X, Z = np.meshgrid(x, z)

    r = np.sqrt(X**2 + Z**2)
    r = np.maximum(r, 1e-6)
    cos_theta = Z / r

    # Potential
    factor = (eps_r - 1.0) / (eps_r + 2.0)
    V_outside = -E0 * Z + factor * E0 * R**3 * cos_theta / r**2
    V_inside = -E_inside * Z

    V = np.where(r <= R, V_inside, V_outside)

    # Electric field: E = -grad V (numerical)
    Ez, Ex = np.gradient(-V, z[1] - z[0], x[1] - x[0])

    fig, ax = plt.subplots(figsize=(8, 8))
    E_mag = np.sqrt(Ex**2 + Ez**2)
    ax.streamplot(X, Z, Ex, Ez, color=np.log10(E_mag + 1), cmap='viridis',
                  density=2, linewidth=0.8)
    circle = plt.Circle((0, 0), R, fill=False, color='red', linewidth=2,
                         label=f'Dielectric (eps_r={eps_r})')
    ax.add_patch(circle)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)')
    ax.set_title(f'Dielectric Sphere in Uniform Field (E_in/E_0 = {E_inside/E0:.3f})')
    ax.legend()
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('ex03_dielectric_sphere.png', dpi=150)
    plt.close()
    print("  Plot saved: ex03_dielectric_sphere.png")


def exercise_3():
    """
    Exercise 3: Multi-Layer Capacitor
    Three dielectric slabs stacked between parallel plates.
    Effective capacitance = series combination.
    """
    A = 0.01   # plate area (m^2)

    # Layer parameters
    d = np.array([1e-3, 1e-3, 1e-3])     # thicknesses (1 mm each)
    eps_r = np.array([2.0, 5.0, 10.0])    # dielectric constants

    # Each layer acts as a capacitor in series:
    # C_i = eps_0 * eps_r_i * A / d_i
    C_layers = epsilon_0 * eps_r * A / d

    # Series combination: 1/C_total = sum(1/C_i)
    C_total = 1.0 / np.sum(1.0 / C_layers)

    # Compare with single dielectric of same total thickness
    d_total = np.sum(d)
    C_vacuum = epsilon_0 * A / d_total

    print("  Multi-layer capacitor:")
    print(f"  Plate area A = {A*1e4:.0f} cm^2")
    for i in range(3):
        print(f"  Layer {i+1}: d = {d[i]*1e3:.1f} mm, eps_r = {eps_r[i]:.1f}, "
              f"C = {C_layers[i]*1e12:.3f} pF")
    print()
    print(f"  Total capacitance (series): C = {C_total*1e12:.4f} pF")
    print(f"  Vacuum capacitance (same thickness): C_vac = {C_vacuum*1e12:.4f} pF")
    print(f"  Effective eps_r = {C_total/C_vacuum:.4f}")

    # Derivation: for series capacitors
    # 1/C_eff = d1/(eps_0*eps1*A) + d2/(eps_0*eps2*A) + d3/(eps_0*eps3*A)
    # = (1/(eps_0*A)) * sum(d_i/eps_i)
    # C_eff = eps_0*A / sum(d_i/eps_i)
    C_formula = epsilon_0 * A / np.sum(d / eps_r)
    print(f"  Formula verification: C = {C_formula*1e12:.4f} pF")
    print(f"  Agreement: {abs(C_total - C_formula)/C_total:.2e}")


def exercise_4():
    """
    Exercise 4: Energy of a Charged Conductor
    Conducting sphere of radius R with total charge Q.
    Compare with uniformly charged insulating sphere.
    """
    R = 0.05    # radius (5 cm)
    Q = 1e-9    # 1 nC

    # Conducting sphere: all charge on surface, E only outside
    # E = kQ/r^2 for r > R, E = 0 for r < R
    # W_conductor = (1/2) * Q^2 / (4*pi*eps_0*R) = kQ^2/(2R)
    W_conductor = k_e * Q**2 / (2.0 * R)

    # Uniformly charged insulating sphere: E = kQr/R^3 inside, kQ/r^2 outside
    # W_insulator = (3/5) * kQ^2/R
    W_insulator = (3.0 / 5.0) * k_e * Q**2 / R

    print(f"  Sphere radius R = {R*100:.0f} cm, charge Q = {Q*1e9:.1f} nC")
    print()
    print(f"  Conducting sphere energy:  W = kQ^2/(2R) = {W_conductor:.6e} J")
    print(f"  Insulating sphere energy:  W = (3/5)kQ^2/R = {W_insulator:.6e} J")
    print(f"  Ratio W_insulator/W_conductor = {W_insulator/W_conductor:.4f}")
    print("  (The insulating sphere has more energy because field exists inside too)")

    # Numerical verification for conducting sphere
    N = 100000
    r = np.linspace(R, 100 * R, N)
    dr = r[1] - r[0]
    E = k_e * Q / r**2
    W_numerical = 0.5 * epsilon_0 * np.sum(E**2 * 4 * np.pi * r**2 * dr)
    print(f"\n  Numerical W_conductor: {W_numerical:.6e} J")
    print(f"  Relative error: {abs(W_numerical - W_conductor)/W_conductor:.4e}")


def exercise_5():
    """
    Exercise 5: Polarization Bound Charges
    Dielectric cylinder with uniform P = P0 z_hat.
    Compute bound volume and surface charges. Show total = 0.
    """
    P0 = 1e-6    # polarization (C/m^2)
    R_cyl = 0.05  # cylinder radius (m)
    L = 0.2       # cylinder length (m)

    # Bound volume charge: rho_b = -div(P)
    # P = P0 z_hat (uniform) => div(P) = dP_z/dz = 0
    # So rho_b = 0 everywhere inside
    rho_b = 0.0

    # Bound surface charge: sigma_b = P dot n_hat
    # Top face (z = L/2): n = +z_hat => sigma_b = +P0
    # Bottom face (z = -L/2): n = -z_hat => sigma_b = -P0
    # Curved surface: n = r_hat (perpendicular to z) => sigma_b = P dot r_hat = 0
    sigma_top = P0
    sigma_bottom = -P0
    sigma_curved = 0.0

    # Total bound charge
    A_face = np.pi * R_cyl**2
    Q_top = sigma_top * A_face
    Q_bottom = sigma_bottom * A_face
    Q_curved = sigma_curved * 2 * np.pi * R_cyl * L
    Q_total = Q_top + Q_bottom + Q_curved

    print(f"  Dielectric cylinder: R = {R_cyl*100:.0f} cm, L = {L*100:.0f} cm")
    print(f"  Uniform polarization P = {P0*1e6:.1f} uC/m^2 in z-direction")
    print()
    print(f"  Bound volume charge density: rho_b = -div(P) = {rho_b} C/m^3")
    print(f"  Bound surface charge (top):    sigma_b = +P0 = {sigma_top*1e6:.1f} uC/m^2")
    print(f"  Bound surface charge (bottom): sigma_b = -P0 = {sigma_bottom*1e6:.1f} uC/m^2")
    print(f"  Bound surface charge (curved): sigma_b = 0")
    print()
    print(f"  Total bound charge on top:    Q = {Q_top*1e9:.4f} nC")
    print(f"  Total bound charge on bottom: Q = {Q_bottom*1e9:.4f} nC")
    print(f"  Total bound charge:           Q = {Q_total:.2e} C  (zero, as expected)")

    # Sketch
    fig, ax = plt.subplots(figsize=(6, 8))
    # Draw cylinder outline
    rect = plt.Rectangle((-R_cyl * 100, -L / 2 * 100), 2 * R_cyl * 100, L * 100,
                          fill=False, edgecolor='blue', linewidth=2)
    ax.add_patch(rect)

    # Arrows for P
    for z_pos in np.linspace(-L/2*100*0.8, L/2*100*0.8, 5):
        ax.annotate('', xy=(0, z_pos + 3), xytext=(0, z_pos - 3),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2))

    # Show surface charges
    ax.plot([-R_cyl*100, R_cyl*100], [L/2*100, L/2*100], 'r-', linewidth=4,
            label=r'$\sigma_b = +P_0$')
    ax.plot([-R_cyl*100, R_cyl*100], [-L/2*100, -L/2*100], 'b-', linewidth=4,
            label=r'$\sigma_b = -P_0$')
    ax.text(R_cyl*100 + 1, 0, r'$\mathbf{P} = P_0\hat{z}$', fontsize=12, color='green')
    ax.text(R_cyl*100 + 1, L/2*100, '+', fontsize=16, color='red')
    ax.text(R_cyl*100 + 1, -L/2*100, '-', fontsize=16, color='blue')

    ax.set_xlim(-10, 15)
    ax.set_ylim(-15, 15)
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('z (cm)')
    ax.set_title('Bound Charges on Polarized Cylinder')
    ax.legend(loc='lower left')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('ex03_bound_charges.png', dpi=150)
    plt.close()
    print("  Sketch saved: ex03_bound_charges.png")


if __name__ == "__main__":
    print("=== Exercise 1: Image Charge -- Grounded Sphere ===")
    exercise_1()
    print("\n=== Exercise 2: Dielectric Sphere ===")
    exercise_2()
    print("\n=== Exercise 3: Multi-Layer Capacitor ===")
    exercise_3()
    print("\n=== Exercise 4: Energy of a Charged Conductor ===")
    exercise_4()
    print("\n=== Exercise 5: Polarization Bound Charges ===")
    exercise_5()
    print("\nAll exercises completed!")
