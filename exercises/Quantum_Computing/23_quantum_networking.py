"""
Exercises for Lesson 23: Quantum Networking
Topic: Quantum_Computing

Solutions covering BB84 security analysis, CHSH tests,
repeater optimization, and entanglement distillation.
"""

import numpy as np


def exercise_1():
    """BB84 with varying interception rates."""
    print("=" * 60)
    print("Exercise 1: BB84 Security Analysis")
    print("=" * 60)

    def bb84(n_bits, eve_frac=0.0, seed=42):
        rng = np.random.default_rng(seed)
        a_bits = rng.integers(0, 2, n_bits)
        a_bases = rng.integers(0, 2, n_bits)
        b_bases = rng.integers(0, 2, n_bits)
        b_bits = np.zeros(n_bits, dtype=int)

        for i in range(n_bits):
            intercepted = rng.random() < eve_frac
            if intercepted:
                e_base = rng.integers(0, 2)
                if e_base != a_bases[i]:
                    b_bits[i] = rng.integers(0, 2) if b_bases[i] == a_bases[i] else rng.integers(0, 2)
                else:
                    b_bits[i] = a_bits[i] if b_bases[i] == a_bases[i] else rng.integers(0, 2)
            else:
                b_bits[i] = a_bits[i] if b_bases[i] == a_bases[i] else rng.integers(0, 2)

        match = a_bases == b_bases
        sa, sb = a_bits[match], b_bits[match]
        nc = len(sa) // 2
        idx = rng.choice(len(sa), nc, replace=False)
        qber = np.mean(sa[idx] != sb[idx]) if nc > 0 else 0
        return qber

    print(f"\n{'Eve intercept %':>18} {'QBER':>10} {'Detected?':>12}")
    print("-" * 42)
    for frac in [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]:
        qber = bb84(20000, frac)
        detected = "Yes" if qber > 0.11 else "No"
        print(f"{frac*100:18.0f} {qber:10.4f} {detected:>12}")


def exercise_2():
    """CHSH test with Werner states."""
    print("\n" + "=" * 60)
    print("Exercise 2: CHSH Test")
    print("=" * 60)

    def chsh_value(visibility):
        # For |Phi+> with visibility v, optimal CHSH = 2*sqrt(2)*v
        return 2 * np.sqrt(2) * visibility

    print(f"\n{'Visibility':>12} {'CHSH S':>10} {'Violation?':>12}")
    print("-" * 38)
    for v in np.arange(0.0, 1.05, 0.1):
        S = chsh_value(v)
        violation = "Yes" if S > 2 else "No"
        print(f"{v:12.2f} {S:10.4f} {violation:>12}")

    # Critical visibility
    v_crit = 2 / (2 * np.sqrt(2))
    print(f"\nCritical visibility: v* = {v_crit:.4f} = 1/sqrt(2)")


def exercise_3():
    """Repeater chain optimization."""
    print("\n" + "=" * 60)
    print("Exercise 3: Repeater Optimization (500 km)")
    print("=" * 60)

    total_dist = 500
    loss_db_km = 0.2

    print(f"\n{'Segments':>10} {'Seg len (km)':>14} {'Transmission':>14} {'Rate (arb)':>14}")
    print("-" * 56)

    for n_seg in [1, 2, 4, 8, 16, 32]:
        seg_len = total_dist / n_seg
        trans_per_seg = 10 ** (-loss_db_km * seg_len / 10)
        # Rate proportional to trans_per_seg (need all segments to succeed)
        # With multiplexing: rate ~ trans_per_seg * (1/n_seg overhead)
        rate = trans_per_seg  # simplified per-segment rate
        print(f"{n_seg:10d} {seg_len:14.1f} {trans_per_seg:14.2e} {rate:14.2e}")

    # Direct transmission
    direct = 10 ** (-loss_db_km * total_dist / 10)
    print(f"\nDirect transmission: {direct:.2e}")
    print(f"Best repeater segment rate >> direct when n_segments ~ 16-32")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
