"""
Exercises for Lesson 07: Polarization
Topic: Optics
Solutions to practice problems from the lesson.
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Malus's Law Chain
    Calculate the transmitted intensity through a chain of polarizers
    at various angles, including the paradox of adding a polarizer
    to increase transmission.
    """
    I_0 = 1.0  # Unpolarized input intensity (normalized)

    print("Malus's Law: I = I_0 * cos^2(theta)")

    # Part (a): Two crossed polarizers
    print("\n--- Part (a): Two crossed polarizers ---")
    theta_12 = 90  # degrees
    I_after_1 = I_0 / 2  # First polarizer: half of unpolarized
    I_after_2 = I_after_1 * np.cos(np.radians(theta_12))**2
    print(f"After polarizer 1 (vertical): I = {I_after_1:.4f}")
    print(f"After polarizer 2 (horizontal, 90 deg): I = {I_after_2:.6f}")
    print(f"No light gets through crossed polarizers!")

    # Part (b): Insert 45-degree polarizer between crossed pair
    print("\n--- Part (b): Insert 45-degree polarizer between crossed pair ---")
    I_after_1 = I_0 / 2
    theta_12 = 45
    I_after_mid = I_after_1 * np.cos(np.radians(theta_12))**2
    theta_23 = 45  # 90 - 45
    I_after_2 = I_after_mid * np.cos(np.radians(theta_23))**2
    print(f"After polarizer 1 (vertical): I = {I_after_1:.4f}")
    print(f"After middle (45 deg): I = {I_after_mid:.4f}")
    print(f"After polarizer 2 (horizontal): I = {I_after_2:.4f}")
    print(f"Adding a polarizer INCREASED transmission from 0 to {I_after_2:.4f}!")

    # Part (c): N equally spaced polarizers from 0 to 90 degrees
    print("\n--- Part (c): N polarizers spanning 0 to 90 degrees ---")
    print(f"{'N polarizers':>14} {'I_out/I_in':>12} {'Transmission':>14}")
    print("-" * 42)
    for N in [2, 3, 4, 5, 10, 20, 50, 100]:
        delta_theta = 90.0 / (N - 1)  # Angle between adjacent polarizers
        I = I_0 / 2  # After first polarizer
        for k in range(1, N):
            I *= np.cos(np.radians(delta_theta))**2
        print(f"{N:>14} {I:>12.6f} {I*100:>13.3f}%")

    # In the limit N -> infinity, I -> I_0/2 (all light gets through!)
    print(f"Limit (N->inf): cos^2(90/N) -> 1, so I -> I_0/2 = 0.5")


def exercise_2():
    """
    Exercise 2: Brewster's Angle
    Calculate Brewster's angle for various interfaces and verify
    using the Fresnel equations.
    """
    print("Brewster's Angle: tan(theta_B) = n2/n1")
    print(f"\n{'Interface':>25} {'n1':>6} {'n2':>6} {'theta_B (deg)':>14}")
    print("-" * 53)

    interfaces = [
        ("Air-Glass (BK7)", 1.0, 1.5168),
        ("Air-Water", 1.0, 1.333),
        ("Air-Diamond", 1.0, 2.42),
        ("Glass-Water", 1.5168, 1.333),
        ("Water-Glass", 1.333, 1.5168),
    ]

    for name, n1, n2 in interfaces:
        theta_B = np.degrees(np.arctan(n2 / n1))
        print(f"{name:>25} {n1:>6.4f} {n2:>6.4f} {theta_B:>14.2f}")

    # Fresnel equations at and near Brewster's angle for air-glass
    n1, n2 = 1.0, 1.5168
    theta_B = np.arctan(n2 / n1)

    print(f"\nFresnel coefficients near Brewster's angle ({np.degrees(theta_B):.2f} deg):")
    print(f"{'theta_i (deg)':>14} {'r_s':>8} {'r_p':>8} {'R_s':>8} {'R_p':>8}")
    print("-" * 48)

    for theta_deg in np.arange(50, 62, 1):
        theta_i = np.radians(theta_deg)
        sin_t = n1 * np.sin(theta_i) / n2
        if abs(sin_t) > 1:
            continue
        theta_t = np.arcsin(sin_t)

        r_s = (n1*np.cos(theta_i) - n2*np.cos(theta_t)) / (n1*np.cos(theta_i) + n2*np.cos(theta_t))
        r_p = (n2*np.cos(theta_i) - n1*np.cos(theta_t)) / (n2*np.cos(theta_i) + n1*np.cos(theta_t))
        R_s = r_s**2
        R_p = r_p**2

        print(f"{theta_deg:>14.0f} {r_s:>8.4f} {r_p:>8.4f} {R_s:>8.4f} {R_p:>8.4f}")

    print(f"\nAt Brewster's angle: r_p = 0 (reflected light is purely s-polarized)")
    print(f"theta_B + theta_t = 90 deg (reflected and refracted rays are perpendicular)")


def exercise_3():
    """
    Exercise 3: Wave Plate Analysis Using Jones Calculus
    Analyze the polarization state after passing through quarter-wave
    and half-wave plates at various orientations.
    """
    print("Jones Calculus: Wave Plate Analysis")

    # Jones matrices for wave plates with fast axis at angle theta
    def hwp_matrix(theta):
        """Half-wave plate with fast axis at angle theta (radians)."""
        c, s = np.cos(2*theta), np.sin(2*theta)
        return np.array([[c, s], [s, -c]])

    def qwp_matrix(theta):
        """Quarter-wave plate with fast axis at angle theta (radians)."""
        c, s = np.cos(theta), np.sin(theta)
        return np.exp(-1j*np.pi/4) * np.array([
            [c**2 + 1j*s**2, (1 - 1j)*c*s],
            [(1 - 1j)*c*s, s**2 + 1j*c**2]
        ])

    def describe_polarization(jones):
        """Describe the polarization state from a Jones vector."""
        jones = jones / np.abs(jones[0]) if np.abs(jones[0]) > 1e-10 else jones / np.abs(jones[1])
        ratio = jones[1] / jones[0] if np.abs(jones[0]) > 1e-10 else np.inf
        if np.abs(np.imag(ratio)) < 1e-6:
            angle = np.degrees(np.arctan(np.real(ratio)))
            return f"Linear at {angle:.1f} deg"
        elif np.abs(np.abs(ratio) - 1) < 0.1:
            phase = np.angle(ratio)
            if abs(phase - np.pi/2) < 0.1:
                return "Right circular"
            elif abs(phase + np.pi/2) < 0.1:
                return "Left circular"
            else:
                return f"Elliptical (|ratio|={np.abs(ratio):.2f}, phase={np.degrees(phase):.1f} deg)"
        else:
            return f"Elliptical (ratio={ratio:.2f})"

    # Input states
    H = np.array([1, 0], dtype=complex)   # Horizontal
    V = np.array([0, 1], dtype=complex)   # Vertical
    D = np.array([1, 1], dtype=complex) / np.sqrt(2)  # +45 deg
    R = np.array([1, -1j], dtype=complex) / np.sqrt(2)  # Right circular

    # Part (a): QWP at 45 deg on horizontal input -> circular
    print("\n--- Part (a): QWP at 45 deg on horizontal linear ---")
    qwp45 = qwp_matrix(np.pi/4)
    out_a = qwp45 @ H
    out_a_norm = out_a / np.linalg.norm(out_a)
    print(f"Input: H = [1, 0]")
    print(f"Output: [{out_a_norm[0]:.4f}, {out_a_norm[1]:.4f}]")
    print(f"State: {describe_polarization(out_a)}")

    # Part (b): HWP at 22.5 deg on horizontal -> +45 deg linear
    print("\n--- Part (b): HWP at 22.5 deg on horizontal linear ---")
    hwp22 = hwp_matrix(np.radians(22.5))
    out_b = hwp22 @ H
    out_b_norm = out_b / np.linalg.norm(out_b)
    print(f"Input: H = [1, 0]")
    print(f"Output: [{out_b_norm[0].real:.4f}, {out_b_norm[1].real:.4f}]")
    print(f"State: {describe_polarization(out_b)}")

    # Part (c): HWP rotates linear polarization by 2*theta
    print("\n--- Part (c): HWP rotation of linear polarization ---")
    print(f"{'HWP angle':>12} {'Input':>10} {'Output angle':>14}")
    print("-" * 38)
    for hwp_angle in [0, 15, 22.5, 30, 45]:
        hwp = hwp_matrix(np.radians(hwp_angle))
        out = hwp @ H
        # Angle of output linear polarization
        out_angle = np.degrees(np.arctan2(out[1].real, out[0].real))
        print(f"{hwp_angle:>12.1f} {'H (0 deg)':>10} {out_angle:>14.1f} deg")

    # Part (d): QWP on circular -> linear
    print("\n--- Part (d): QWP on right circular input ---")
    qwp0 = qwp_matrix(0)
    out_d = qwp0 @ R
    print(f"Input: RCP = [1, -i]/sqrt(2)")
    out_d_norm = out_d / np.linalg.norm(out_d)
    print(f"Output: [{out_d_norm[0]:.4f}, {out_d_norm[1]:.4f}]")
    print(f"State: {describe_polarization(out_d)}")


def exercise_4():
    """
    Exercise 4: Circular Polarizer Design
    Design a circular polarizer from a linear polarizer and QWP,
    and verify its operation on different input states.
    """
    print("Circular Polarizer = Linear Polarizer + QWP at 45 deg")

    def qwp_matrix(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.exp(-1j*np.pi/4) * np.array([
            [c**2 + 1j*s**2, (1 - 1j)*c*s],
            [(1 - 1j)*c*s, s**2 + 1j*c**2]
        ])

    # Linear polarizer (horizontal)
    LP_H = np.array([[1, 0], [0, 0]], dtype=complex)

    # QWP at 45 degrees
    QWP_45 = qwp_matrix(np.pi/4)

    # Circular polarizer: QWP @ LP
    CP = QWP_45 @ LP_H

    print(f"\nCircular polarizer Jones matrix:")
    print(f"  [[{CP[0,0]:.4f}, {CP[0,1]:.4f}],")
    print(f"   [{CP[1,0]:.4f}, {CP[1,1]:.4f}]]")

    # Test with various inputs
    inputs = [
        ("Horizontal", np.array([1, 0], dtype=complex)),
        ("Vertical", np.array([0, 1], dtype=complex)),
        ("+45 deg", np.array([1, 1], dtype=complex) / np.sqrt(2)),
        ("-45 deg", np.array([1, -1], dtype=complex) / np.sqrt(2)),
        ("RCP", np.array([1, -1j], dtype=complex) / np.sqrt(2)),
        ("LCP", np.array([1, 1j], dtype=complex) / np.sqrt(2)),
    ]

    print(f"\nOutput states:")
    print(f"{'Input':>12} {'|E_out|^2':>10} {'Output Jones vector':>30}")
    print("-" * 54)

    for name, E_in in inputs:
        E_out = CP @ E_in
        I_out = np.abs(E_out[0])**2 + np.abs(E_out[1])**2
        if I_out > 1e-10:
            E_norm = E_out / np.linalg.norm(E_out)
            print(f"{name:>12} {I_out:>10.4f} [{E_norm[0]:.3f}, {E_norm[1]:.3f}]")
        else:
            print(f"{name:>12} {I_out:>10.4f} [blocked]")

    # For use in 3D glasses: right eye gets RCP, left gets LCP
    print("\n3D Cinema Application:")
    print("  Right eye filter: LP(H) + QWP(45) -> produces RCP")
    print("  Left eye filter:  LP(V) + QWP(45) -> produces LCP")
    print("  Cross-talk is minimized because RCP and LCP are orthogonal")


def exercise_5():
    """
    Exercise 5: Optical Isolator
    Design an optical isolator using a Faraday rotator and polarizers
    to protect a laser from back-reflections.
    """
    print("Optical Isolator Design")
    print("="*40)
    print("Components: Polarizer 1 (0 deg) -> Faraday rotator (45 deg) -> Polarizer 2 (45 deg)")

    # Forward direction
    print("\n--- Forward propagation ---")
    I_0 = 1.0  # Input intensity (after initial polarizer)
    # After Faraday rotator: polarization rotated 45 deg
    # After analyzer at 45 deg: cos^2(0) = 1 (perfect transmission)
    theta_forward = 0  # Angle between rotated polarization and analyzer
    I_forward = I_0 * np.cos(np.radians(theta_forward))**2
    print(f"Input: Vertical polarization (0 deg)")
    print(f"After Faraday rotator: 45 deg polarization")
    print(f"After analyzer (45 deg): I = {I_forward:.4f} * I_0")
    print(f"Forward transmission: {I_forward*100:.1f}%")

    # Backward direction (reflected light)
    print("\n--- Backward propagation ---")
    print("Key: Faraday rotation is NON-RECIPROCAL")
    print("  (Unlike wave plates, it rotates in the same absolute direction)")
    # Light enters from analyzer side (polarized at 45 deg)
    # Faraday rotator adds another 45 deg (same direction) = 90 deg total
    # At input polarizer (0 deg): cos^2(90) = 0
    theta_backward = 90
    I_backward = I_0 * np.cos(np.radians(theta_backward))**2
    print(f"Reflected light enters at 45 deg polarization")
    print(f"After Faraday rotator: 90 deg polarization")
    print(f"At input polarizer (0 deg): I = {I_backward:.6f} * I_0")
    print(f"Backward transmission: {I_backward:.2e}")

    # Isolation ratio
    if I_backward > 0:
        isolation_dB = 10 * np.log10(I_forward / I_backward) if I_backward > 1e-15 else np.inf
    else:
        isolation_dB = np.inf
    print(f"\nIsolation ratio: {isolation_dB:.0f} dB (ideal)")

    # Real-world imperfections
    print("\n--- Real-world performance ---")
    # Typical Faraday rotation error: 0.5 deg
    rotation_errors = [0.1, 0.5, 1.0, 2.0]
    print(f"{'Rotation error (deg)':>22} {'Isolation (dB)':>16} {'Leakage':>12}")
    print("-" * 52)
    for err in rotation_errors:
        I_leak = np.cos(np.radians(90 - err))**2
        iso = -10 * np.log10(I_leak) if I_leak > 0 else np.inf
        print(f"{err:>22.1f} {iso:>16.1f} {I_leak:>12.2e}")

    # Faraday rotation: theta_F = V * B * L
    print("\n--- Faraday Rotator Parameters ---")
    # Terbium Gallium Garnet (TGG): V = -40 rad/(T*m) at 1064 nm
    V = 40  # Verdet constant (rad/(T*m))
    theta_target = np.radians(45)
    B = 0.5  # Magnetic field (T)
    L = theta_target / (V * B)
    print(f"Material: TGG (Verdet constant V = {V} rad/T/m)")
    print(f"Magnetic field: B = {B} T")
    print(f"Required crystal length: L = {L*1e3:.2f} mm")
    print(f"  (45 deg = {np.degrees(theta_target):.0f} deg rotation)")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Malus's Law Chain", exercise_1),
        ("Exercise 2: Brewster's Angle", exercise_2),
        ("Exercise 3: Wave Plate Analysis (Jones Calculus)", exercise_3),
        ("Exercise 4: Circular Polarizer Design", exercise_4),
        ("Exercise 5: Optical Isolator", exercise_5),
    ]
    for title, func in exercises:
        print(f"\n{'='*60}")
        print(f"=== {title} ===")
        print(f"{'='*60}")
        func()

    print(f"\n{'='*60}")
    print("All exercises completed!")
