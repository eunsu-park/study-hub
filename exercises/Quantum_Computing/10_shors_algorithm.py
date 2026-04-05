"""
Exercises for Lesson 10: Shor's Algorithm
Topic: Quantum_Computing

Solutions to practice problems from the lesson.
All quantum operations simulated with numpy matrices (no qiskit).
"""

import numpy as np
from math import gcd, log2
from fractions import Fraction
from typing import List, Tuple, Optional


# ============================================================
# Shared utilities: modular arithmetic and quantum simulation
# ============================================================

def mod_exp(base: int, exp: int, mod: int) -> int:
    """Compute (base^exp) mod mod using fast exponentiation."""
    result = 1
    base = base % mod
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        exp >>= 1
        base = (base * base) % mod
    return result


def find_order(a: int, N: int) -> Optional[int]:
    """Find the multiplicative order of a modulo N (classical brute force)."""
    if gcd(a, N) != 1:
        return None  # a and N not coprime
    r = 1
    current = a % N
    while current != 1:
        current = (current * a) % N
        r += 1
        if r > N:
            return None
    return r


def continued_fraction_convergents(numerator: int, denominator: int,
                                    max_denominator: int) -> List[Tuple[int, int]]:
    """Compute convergents of a continued fraction expansion."""
    convergents = []
    n, d = numerator, denominator
    h_prev, h_curr = 0, 1
    k_prev, k_curr = 1, 0

    while d != 0:
        a = n // d
        n, d = d, n - a * d

        h_prev, h_curr = h_curr, a * h_curr + h_prev
        k_prev, k_curr = k_curr, a * k_curr + k_prev

        if k_curr > max_denominator:
            break
        convergents.append((h_curr, k_curr))

    return convergents


# === Exercise 1: Manual Shor's Algorithm ===
# Problem: Factor N=35 by hand using Shor's algorithm with a=2 and a=3.

def exercise_1():
    """Manual Shor's algorithm for N=35."""
    print("=" * 60)
    print("Exercise 1: Manual Shor's Algorithm (N=35)")
    print("=" * 60)

    N = 35

    # (a) Choose a=2, find order of 2 mod 35
    a = 2
    print(f"\n(a) Computing order of a={a} modulo N={N}:")
    powers = []
    current = 1
    for i in range(20):
        current = (a * current) % N if i > 0 else a % N
        if i == 0:
            current = a % N
        else:
            current = (powers[-1][1] * a) % N
        powers.append((i + 1, current))
        print(f"    {a}^{i+1} mod {N} = {current}")
        if current == 1:
            break

    r = find_order(a, N)
    print(f"    Order r = {r}")

    # (b) Check if r is even
    print(f"\n(b) Is r={r} even? {'Yes' if r % 2 == 0 else 'No'}")

    if r is not None and r % 2 == 0:
        a_half = mod_exp(a, r // 2, N)
        print(f"    a^(r/2) mod N = {a}^{r//2} mod {N} = {a_half}")

        # (c) Compute gcd(a^(r/2) +/- 1, N)
        factor1 = gcd(a_half - 1, N)
        factor2 = gcd(a_half + 1, N)
        print(f"\n(c) gcd({a_half} - 1, {N}) = gcd({a_half - 1}, {N}) = {factor1}")
        print(f"    gcd({a_half} + 1, {N}) = gcd({a_half + 1}, {N}) = {factor2}")

        if factor1 not in (1, N) or factor2 not in (1, N):
            print(f"    SUCCESS: {N} = {factor1} x {factor2}")
        else:
            print(f"    TRIVIAL factors found, need to try different a")
    else:
        print(f"    r is odd, Shor's algorithm fails for this a. Try another a.")

    # (d) Repeat with a=3
    a = 3
    print(f"\n(d) Trying a={a}:")
    r = find_order(a, N)
    print(f"    Order of {a} mod {N} = {r}")

    if r is not None and r % 2 == 0:
        a_half = mod_exp(a, r // 2, N)
        factor1 = gcd(a_half - 1, N)
        factor2 = gcd(a_half + 1, N)
        print(f"    a^(r/2) mod N = {a}^{r//2} mod {N} = {a_half}")
        print(f"    gcd({a_half - 1}, {N}) = {factor1}")
        print(f"    gcd({a_half + 1}, {N}) = {factor2}")
        if factor1 not in (1, N) and factor2 not in (1, N):
            print(f"    SUCCESS: {N} = {factor1} x {factor2}")
        elif factor1 not in (1, N):
            print(f"    Partial: found factor {factor1}, other = {N // factor1}")
        else:
            print(f"    TRIVIAL factors (1 and N), Shor's fails for a={a}")
    elif r is not None and r % 2 == 1:
        print(f"    r={r} is ODD -> Shor's algorithm fails for a={a}")
        print(f"    Why: We need r even so that a^(r/2) is an integer exponent")


# === Exercise 2: Measurement Distribution ===
# Problem: Simulate Shor's quantum order-finding for N=21, a=2 (period r=6).

def exercise_2():
    """Measurement distribution for Shor's algorithm (N=21, a=2)."""
    print("\n" + "=" * 60)
    print("Exercise 2: Measurement Distribution (N=21, a=2)")
    print("=" * 60)

    N = 21
    a = 2
    r = find_order(a, N)
    print(f"\n  N={N}, a={a}, actual period r={r}")

    # Choose Q = 2^n where Q >= N^2
    n_qubits = int(np.ceil(2 * np.log2(N)))
    Q = 2 ** n_qubits
    print(f"  Using {n_qubits} qubits, Q = {Q}")

    # (a) Simulate the QFT output state
    # After modular exponentiation + QFT, the output register has peaks
    # at multiples of Q/r (approximately)
    print(f"\n(a) Number of peaks: {r} (one per value s = 0, 1, ..., {r-1})")
    print(f"    Peaks at m = s * Q / r for s = 0, 1, ..., {r-1}")

    peaks = []
    for s in range(r):
        m_exact = s * Q / r
        m_nearest = round(m_exact)
        peaks.append((s, m_nearest, m_exact))
        print(f"    s={s}: m = {s}*{Q}/{r} = {m_exact:.2f} -> nearest integer {m_nearest}")

    # (b) Verify m/Q ~ s/r via continued fractions
    print(f"\n(b) Continued fraction verification:")
    success_count = 0
    for s, m_nearest, m_exact in peaks:
        if m_nearest == 0:
            print(f"    m={m_nearest}: s/r = 0/r -> trivial (skip)")
            continue

        frac = Fraction(m_nearest, Q).limit_denominator(N)
        recovered_r = frac.denominator
        print(
            f"    m={m_nearest}: {m_nearest}/{Q} = {m_nearest/Q:.6f} "
            f"-> convergent {frac.numerator}/{frac.denominator} "
            f"-> r_guess={recovered_r} "
            f"{'CORRECT' if recovered_r == r else 'WRONG'}"
        )
        if recovered_r == r:
            success_count += 1

    # (c) Success fraction
    total_non_trivial = r - 1  # Exclude s=0
    print(f"\n(c) Fraction yielding correct period: {success_count}/{total_non_trivial} "
          f"= {success_count/total_non_trivial:.2%}")


# === Exercise 3: Success Probability Analysis ===
# Problem: Analyze success probability for N=77=7*11 with random choices of a.

def exercise_3():
    """Success probability analysis for Shor's algorithm (N=77)."""
    print("\n" + "=" * 60)
    print("Exercise 3: Success Probability Analysis (N=77)")
    print("=" * 60)

    N = 77
    p, q = 7, 11

    # (a) phi(N) = (p-1)(q-1)
    phi_N = (p - 1) * (q - 1)
    coprime_count = phi_N
    print(f"\n(a) N = {N} = {p} x {q}")
    print(f"    phi(N) = (p-1)(q-1) = {p-1} x {q-1} = {phi_N}")
    print(f"    Number of values a coprime to N: {coprime_count}")

    # (b) Fraction giving even period
    np.random.seed(42)
    even_count = 0
    success_count = 0
    total_coprime = 0

    results = []
    for a in range(2, N):
        if gcd(a, N) != 1:
            # Non-coprime: already gives a factor directly
            if gcd(a, N) not in (1, N):
                results.append((a, "gcd_factor", gcd(a, N)))
            continue

        total_coprime += 1
        r = find_order(a, N)

        if r is not None and r % 2 == 0:
            even_count += 1
            a_half = mod_exp(a, r // 2, N)
            f1 = gcd(a_half - 1, N)
            f2 = gcd(a_half + 1, N)
            if f1 not in (1, N) or f2 not in (1, N):
                success_count += 1
                results.append((a, "success", r))
            else:
                results.append((a, "trivial", r))
        elif r is not None:
            results.append((a, "odd_period", r))

    print(f"\n(b) Among {total_coprime} coprime values:")
    print(f"    Even period: {even_count} ({even_count/total_coprime:.2%})")
    print(f"    Successful factorization: {success_count} ({success_count/total_coprime:.2%})")

    # (c) Simulate 100 random choices
    print(f"\n(c) Monte Carlo simulation (100 random choices of a):")
    n_trials = 100
    mc_successes = 0
    for _ in range(n_trials):
        a = np.random.randint(2, N)

        # Check if gcd gives factor directly
        g = gcd(a, N)
        if g not in (1, N):
            mc_successes += 1
            continue

        if g != 1:
            continue

        r = find_order(a, N)
        if r is None or r % 2 != 0:
            continue

        a_half = mod_exp(a, r // 2, N)
        f1 = gcd(a_half - 1, N)
        f2 = gcd(a_half + 1, N)
        if f1 not in (1, N) or f2 not in (1, N):
            mc_successes += 1

    print(f"    Successes: {mc_successes}/{n_trials} = {mc_successes/n_trials:.2%}")
    print(f"    (Theoretical: ~{success_count/total_coprime:.2%} for coprime a)")


# === Exercise 4: Post-Quantum Security ===
# Problem: Estimate time and qubit requirements for factoring RSA keys.

def exercise_4():
    """Post-quantum security resource estimation."""
    print("\n" + "=" * 60)
    print("Exercise 4: Post-Quantum Security")
    print("=" * 60)

    gates_per_second = 1e10  # 10^10 gates/sec

    for key_size in [2048, 4096]:
        n = key_size
        # (a,b) Shor's algorithm: O(n^3) gates
        # More precise estimate: ~72 * n^3 * log(n) gates (Beauregard/Gidney-Ekera)
        gate_count = 72 * n**3
        time_seconds = gate_count / gates_per_second
        time_hours = time_seconds / 3600

        # (c,d) Qubit requirements
        logical_qubits = 2 * n + n  # 2n + O(n) ancilla
        physical_per_logical = 1000  # Error correction overhead
        physical_qubits = logical_qubits * physical_per_logical

        print(f"\n  RSA-{key_size}:")
        print(f"    Gate count (O(n^3)):     {gate_count:.2e}")
        print(f"    Time @ {gates_per_second:.0e} gates/s: {time_seconds:.2e} s = {time_hours:.1f} hours")
        print(f"    Logical qubits (3n):     {logical_qubits:,}")
        print(f"    Physical qubits (x1000): {physical_qubits:,}")

    print("\n  Analysis:")
    print("    - RSA-2048: feasible in hours IF we had the qubits")
    print("    - RSA-4096: ~8x longer (n^3 scaling)")
    print("    - The bottleneck is qubits, not time")
    print("    - Current (2025): ~1000 physical qubits; need ~6M for RSA-2048")
    print("    - Post-quantum crypto (lattice-based, NIST PQC) is essential")


# === Exercise 5: Order Finding for Larger Numbers ===
# Problem: Extend factorization to N=143 and N=221.

def exercise_5():
    """Order finding for larger numbers (N=143, N=221)."""
    print("\n" + "=" * 60)
    print("Exercise 5: Order Finding for Larger Numbers")
    print("=" * 60)

    for N, expected_factors in [(143, (11, 13)), (221, (13, 17))]:
        print(f"\n  N = {N} = {expected_factors[0]} x {expected_factors[1]}")
        print("  " + "-" * 50)

        successful_a = []
        total_coprime = 0

        for a in range(2, N):
            if gcd(a, N) != 1:
                # Direct factor from gcd
                g = gcd(a, N)
                if g not in (1, N):
                    successful_a.append((a, "gcd", 0, (g, N // g)))
                continue

            total_coprime += 1
            r = find_order(a, N)

            if r is None or r % 2 != 0:
                continue

            a_half = mod_exp(a, r // 2, N)
            f1 = gcd(a_half - 1, N)
            f2 = gcd(a_half + 1, N)

            if f1 not in (1, N) and f2 not in (1, N):
                successful_a.append((a, "shor", r, (f1, f2)))
            elif f1 not in (1, N):
                successful_a.append((a, "shor", r, (f1, N // f1)))

        # (a) List successful values of a
        print(f"\n  (a) Successful values of a: {len(successful_a)} out of {total_coprime} coprime values")
        # Show first 10
        for a, method, r, factors in successful_a[:10]:
            if method == "gcd":
                print(f"      a={a}: gcd({a},{N})={factors[0]} -> {N}={factors[0]}x{factors[1]}")
            else:
                print(f"      a={a}: r={r}, factors={factors}")
        if len(successful_a) > 10:
            print(f"      ... and {len(successful_a) - 10} more")

        # (b) Success probability
        success_rate = len(successful_a) / max(total_coprime, 1)
        print(f"\n  (b) Success probability: {len(successful_a)}/{total_coprime} "
              f"= {success_rate:.2%}")

        # (c) Period distribution
        periods = [r for _, method, r, _ in successful_a if method == "shor"]
        if periods:
            unique_periods = sorted(set(periods))
            print(f"\n  (c) Unique periods found: {unique_periods}")
            print(f"      Period distribution:")
            from collections import Counter
            period_counts = Counter(periods)
            for period in sorted(period_counts.keys()):
                bar = "#" * min(period_counts[period], 40)
                print(f"        r={period:>3}: {bar} ({period_counts[period]})")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
