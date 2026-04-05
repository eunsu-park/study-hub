"""
Exercises for Lesson 03: Transfer Functions and Block Diagrams
Topic: Control_Theory
Solutions to practice problems from the lesson.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def exercise_1():
    """
    Exercise 1: Poles and Zeros
    G(s) = 2(s+3) / [(s+1)(s^2 + 4s + 8)]
    """
    # Part 1: Find all poles and zeros
    print("G(s) = 2(s+3) / [(s+1)(s^2 + 4s + 8)]")

    # Zeros: roots of numerator
    zeros = np.array([-3.0])
    print(f"\nPart 1: Zeros = {zeros}")

    # Poles: roots of denominator
    # (s+1) gives s = -1
    # s^2 + 4s + 8 = 0 => s = (-4 +/- sqrt(16-32))/2 = -2 +/- j2
    pole1 = -1.0
    pole2 = -2.0 + 2.0j
    pole3 = -2.0 - 2.0j
    poles = np.array([pole1, pole2, pole3])
    print(f"  Poles = {poles}")

    # Part 2: Identify real vs complex
    print("\nPart 2:")
    print(f"  Pole at s = {pole1}: Real")
    print(f"  Poles at s = {pole2} and s = {pole3}: Complex conjugate pair")
    print(f"  The quadratic s^2 + 4s + 8 has discriminant = 16 - 32 = -16 < 0")

    # Part 3: Stability
    print("\nPart 3: Stability analysis")
    all_negative = all(p.real < 0 for p in poles)
    print(f"  All pole real parts: {[p.real for p in poles]}")
    print(f"  All poles in LHP: {all_negative}")
    print(f"  System is {'STABLE' if all_negative else 'UNSTABLE'}")

    # Part 4: Pole-zero plot
    print("\nPart 4: Pole-zero plot saved to 'ex03_pole_zero.png'")
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot poles
    ax.plot(np.real(poles), np.imag(poles), 'rx', markersize=12, markeredgewidth=2,
            label='Poles')
    # Plot zeros
    ax.plot(np.real(zeros), np.imag(zeros), 'bo', markersize=10, markeredgewidth=2,
            fillstyle='none', label='Zeros')

    # Axes
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlabel('Real Part')
    ax.set_ylabel('Imaginary Part')
    ax.set_title('Pole-Zero Plot: G(s) = 2(s+3)/[(s+1)(s^2+4s+8)]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Annotate
    ax.annotate(f's = {pole1}', (pole1, 0), textcoords="offset points",
                xytext=(10, 10))
    ax.annotate(f's = -2+j2', (-2, 2), textcoords="offset points",
                xytext=(10, 10))
    ax.annotate(f's = -2-j2', (-2, -2), textcoords="offset points",
                xytext=(10, -15))
    ax.annotate(f's = {zeros[0]}', (zeros[0], 0), textcoords="offset points",
                xytext=(10, 10))

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/exercises/Control_Theory/ex03_pole_zero.png',
                dpi=100)
    plt.close()


def exercise_2():
    """
    Exercise 2: Block Diagram Reduction
    R -> (+) -> [G1] -> (+) -> [G2] -> (+) -> [G3] -> Y
          ^              ^              ^              |
          |              +-- [H2] <-----+              |
          +------------ [H1] <-------------------------+
    """
    print("Block diagram:")
    print("R -> (+) -> [G1] -> (+) -> [G2] -> (+) -> [G3] -> Y")
    print("      ^              ^              ^              |")
    print("      |              +-- [H2] <-----+              |")
    print("      +------------ [H1] <-------------------------+")

    print("\nStep 1: Reduce inner feedback loop (G2 with H2)")
    print("  The inner loop has G2 and G3 in a feedback configuration with H2.")
    print("  Note: H2 feeds back from output of G3 to input of G2.")
    print("  Inner closed-loop: G2*G3 / (1 + G2*G3*H2)  -- wait, re-examine.")
    print()
    print("  Actually, looking more carefully:")
    print("  H2 feeds back from the output of G2 to the summing junction before G2.")
    print("  The summing junction after G3 doesn't have a feedback, it's just a pass-through.")
    print()
    print("  Correction based on diagram: H2 takes signal from output of G2 (before G3)")
    print("  and feeds back to the summing junction at the input of G2.")
    print()
    print("  Inner loop: G2 / (1 + G2*H2)")
    print()
    print("Step 2: Forward path is G1 * [G2/(1+G2*H2)] * G3")
    print("  = G1*G2*G3 / (1 + G2*H2)")
    print()
    print("Step 3: Close outer loop with H1")
    print("  Y/R = [G1*G2*G3/(1+G2*H2)] / [1 + G1*G2*G3*H1/(1+G2*H2)]")
    print("       = G1*G2*G3 / (1 + G2*H2 + G1*G2*G3*H1)")
    print()
    print("  Final answer:")
    print("  Y(s)/R(s) = G1*G2*G3 / (1 + G2*H2 + G1*G2*G3*H1)")


def exercise_3():
    """
    Exercise 3: Mason's Gain Formula
    Forward paths: P1 = ABCD, P2 = AEFD
    Loops: L1 = -BG, L2 = -CH, L3 = -EFHG
    L1 and L2 are non-touching; all other pairs are touching.
    """
    print("Given:")
    print("  Forward paths: P1 = ABCD, P2 = AEFD")
    print("  Loops: L1 = -BG, L2 = -CH, L3 = -EFHG")
    print("  L1 and L2 are non-touching; all other loop pairs touch.")

    print("\nStep 1: Compute graph determinant Delta")
    print("  Delta = 1 - (sum of individual loops) + (sum of products of non-touching loop pairs)")
    print("  Sum of loops: L1 + L2 + L3 = -BG - CH - EFHG")
    print("  Non-touching pairs: only L1*L2 = (-BG)(-CH) = BGCH")
    print("  (L1,L3 touch and L2,L3 touch)")
    print("  Delta = 1 - (-BG - CH - EFHG) + BGCH")
    print("        = 1 + BG + CH + EFHG + BGCH")

    print("\nStep 2: Compute cofactors")
    print("  Delta_1 (for P1 = ABCD):")
    print("    Remove all loops that touch path P1.")
    print("    Path P1 uses nodes A->B->C->D, which touches L1 (contains B)")
    print("    and L2 (contains C), and L3 (if it shares nodes).")
    print("    Since all loops touch path P1: Delta_1 = 1")
    print()
    print("  Delta_2 (for P2 = AEFD):")
    print("    Path P2 uses A->E->F->D.")
    print("    L1 = -BG: does it touch P2? If B and G are not on path P2, then no.")
    print("    L2 = -CH: does it touch P2? If C and H are not on path P2, then no.")
    print("    L3 = -EFHG: shares E, F with P2 -> touches P2.")
    print("    Non-touching loops for P2: L1 and L2 (both don't touch P2)")
    print("    Since L1 and L2 are also non-touching with each other:")
    print("    Delta_2 = 1 - (L1 + L2) + L1*L2")
    print("            = 1 + BG + CH + BGCH")

    print("\nStep 3: Apply Mason's formula")
    print("  T = (P1*Delta_1 + P2*Delta_2) / Delta")
    print("  T = [ABCD * 1 + AEFD * (1 + BG + CH + BGCH)] / (1 + BG + CH + EFHG + BGCH)")
    print()
    print("  T = [ABCD + AEFD(1 + BG + CH + BGCH)] / (1 + BG + CH + EFHG + BGCH)")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Poles and Zeros ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Block Diagram Reduction ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Mason's Gain Formula ===")
    print("=" * 60)
    exercise_3()

    print("\n\nAll exercises completed!")
