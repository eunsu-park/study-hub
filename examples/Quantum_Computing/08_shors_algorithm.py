"""
08_shors_algorithm.py — Shor's Algorithm for Integer Factorization

Demonstrates:
  - Classical number theory: GCD, modular exponentiation, continued fractions
  - Quantum order finding (period finding via QFT simulation)
  - Complete Shor's algorithm for factoring small numbers (N=15, N=21)
  - Why period finding leads to factoring
  - Success probability analysis

All computations use pure NumPy.
"""

import numpy as np
from math import gcd
from typing import Tuple, Optional, List

# ---------------------------------------------------------------------------
# Classical number theory helpers
# ---------------------------------------------------------------------------

def mod_exp(base: int, exponent: int, modulus: int) -> int:
    """Compute base^exponent mod modulus using fast exponentiation.

    Why: Naive computation of a^r mod N would overflow for large r.
    Square-and-multiply runs in O(log r) multiplications, each bounded by N².
    This is the same algorithm used inside the quantum circuit for Shor's.
    """
    result = 1
    base = base % modulus
    while exponent > 0:
        if exponent % 2 == 1:
            result = (result * base) % modulus
        exponent //= 2
        base = (base * base) % modulus
    return result


def find_order_classical(a: int, N: int) -> int:
    """Find the multiplicative order of a mod N classically.

    The order r is the smallest positive integer such that a^r ≡ 1 (mod N).

    Why: This is the brute-force classical approach — try r = 1, 2, 3, ...
    It takes O(N) time in the worst case.  Shor's quantum algorithm finds r
    in O((log N)³) time, an exponential speedup.
    """
    if gcd(a, N) != 1:
        return -1  # a and N share a factor

    r = 1
    current = a % N
    while current != 1:
        current = (current * a) % N
        r += 1
        if r > N:
            return -1
    return r


def continued_fraction_convergents(numerator: int, denominator: int,
                                    max_denom: int) -> List[Tuple[int, int]]:
    """Compute convergents of the continued fraction expansion of numerator/denominator.

    Why: After quantum measurement, we get an integer s that encodes the phase
    as s/2^n ≈ j/r for some j.  Continued fractions efficiently extract the
    denominator r (the period) from this rational approximation.
    This classical post-processing step is essential to Shor's algorithm.
    """
    convergents = []
    a, b = numerator, denominator

    p_prev, p_curr = 0, 1
    q_prev, q_curr = 1, 0

    while b != 0:
        quotient = a // b
        a, b = b, a % b

        p_new = quotient * p_curr + p_prev
        q_new = quotient * q_curr + q_prev

        if q_new > max_denom:
            break

        convergents.append((p_new, q_new))
        p_prev, p_curr = p_curr, p_new
        q_prev, q_curr = q_curr, q_new

    return convergents


# ---------------------------------------------------------------------------
# QFT (simplified for Shor's)
# ---------------------------------------------------------------------------

def qft_matrix(n: int) -> np.ndarray:
    """Construct n-qubit QFT matrix."""
    N = 2 ** n
    omega = np.exp(2j * np.pi / N)
    F = np.zeros((N, N), dtype=complex)
    for j in range(N):
        for k in range(N):
            F[j, k] = omega ** (j * k)
    return F / np.sqrt(N)


def inverse_qft_matrix(n: int) -> np.ndarray:
    return qft_matrix(n).conj().T


# ---------------------------------------------------------------------------
# Quantum Order Finding (simulated)
# ---------------------------------------------------------------------------

def quantum_order_finding(a: int, N: int, n_qubits: int = None,
                           verbose: bool = False) -> Optional[int]:
    """Simulate quantum order finding for Shor's algorithm.

    Why: This is the quantum heart of Shor's algorithm.  We simulate the
    quantum circuit classically (which takes O(N) time), but the actual
    quantum circuit uses O(log³N) gates.  The simulation demonstrates the
    mathematical structure of the algorithm even though we lose the speedup.

    The quantum circuit:
    1. Prepare superposition: |ψ⟩ = (1/√Q) Σ_{x=0}^{Q-1} |x⟩|0⟩
    2. Compute f(x) = a^x mod N into second register
    3. Apply QFT to first register
    4. Measure first register → get s such that s/Q ≈ j/r
    """
    if n_qubits is None:
        # Why: We need the precision register to have at least 2·log₂(N) qubits
        # to distinguish between different fractions j/r with denominator ≤ N.
        n_qubits = 2 * int(np.ceil(np.log2(N))) + 1

    Q = 2 ** n_qubits  # Size of precision register

    if verbose:
        print(f"    Quantum register: {n_qubits} qubits (Q = {Q})")

    # Step 1 & 2: Compute the state after applying f(x) = a^x mod N
    # Why: After measuring the second register (or equivalently, after the
    # function evaluation), the first register collapses to a superposition
    # of all x values that give the same f(x).  These x values are evenly
    # spaced with period r — exactly what the QFT can detect.

    # Build the state: Σ_x |x⟩|a^x mod N⟩
    # After tracing out the second register, the first register state is:
    state_first = np.zeros(Q, dtype=complex)
    for x in range(Q):
        ax_mod_N = mod_exp(a, x, N)
        state_first[x] = 1.0

    # Actually, we need to simulate the post-measurement state correctly.
    # Pick a random function value f₀ and keep only x where a^x mod N = f₀
    f_values = [mod_exp(a, x, N) for x in range(Q)]
    f0 = f_values[np.random.randint(0, Q)]

    state_first = np.zeros(Q, dtype=complex)
    for x in range(Q):
        if f_values[x] == f0:
            state_first[x] = 1.0
    state_first = state_first / np.linalg.norm(state_first)

    if verbose:
        nonzero = [x for x in range(Q) if np.abs(state_first[x]) > 1e-10]
        if len(nonzero) <= 10:
            print(f"    Surviving x values (a^x mod {N} = {f0}): {nonzero}")
        else:
            print(f"    {len(nonzero)} surviving x values (a^x mod {N} = {f0})")

    # Step 3: Apply QFT
    F = qft_matrix(n_qubits)
    state_freq = F @ state_first

    # Step 4: Measure
    probs = np.abs(state_freq) ** 2
    measured_s = np.random.choice(Q, p=probs)

    if verbose:
        # Show top peaks
        top_indices = np.argsort(probs)[::-1][:5]
        print(f"    Top QFT peaks: {[(int(i), f'{probs[i]:.4f}') for i in top_indices]}")
        print(f"    Measured: s = {measured_s}")

    # Step 5: Classical post-processing with continued fractions
    # Why: s/Q ≈ j/r for some integer j.  We use continued fractions to
    # find the best rational approximation with denominator ≤ N.
    if measured_s == 0:
        if verbose:
            print(f"    Measured s=0 → uninformative, retry needed")
        return None

    convergents = continued_fraction_convergents(measured_s, Q, N)

    if verbose:
        print(f"    Continued fraction convergents of {measured_s}/{Q}:")
        for p, q in convergents:
            print(f"      {p}/{q} = {p/q:.6f}")

    # Try each convergent's denominator as a candidate for r
    for p, q in convergents:
        if q > 0 and mod_exp(a, q, N) == 1:
            if verbose:
                print(f"    Found order: r = {q} (a^{q} mod {N} = 1)")
            return q

    # Also try multiples of convergent denominators
    for p, q in convergents:
        for mult in range(2, N // q + 1):
            candidate = q * mult
            if candidate <= N and mod_exp(a, candidate, N) == 1:
                if verbose:
                    print(f"    Found order: r = {candidate} (multiple of {q})")
                return candidate

    if verbose:
        print(f"    Failed to extract order from this measurement")
    return None


# ---------------------------------------------------------------------------
# Complete Shor's Algorithm
# ---------------------------------------------------------------------------

def shors_algorithm(N: int, max_attempts: int = 20,
                    verbose: bool = True) -> Optional[Tuple[int, int]]:
    """Run Shor's algorithm to factor N.

    Why: Shor's algorithm reduces factoring to order finding:
    1. Pick random a < N with gcd(a, N) = 1
    2. Find the order r of a mod N (quantum step)
    3. If r is even and a^{r/2} ≢ -1 (mod N), then:
       gcd(a^{r/2} ± 1, N) gives non-trivial factors

    The probability of success per attempt is ≥ 1/2, so a few attempts suffice.
    """
    if verbose:
        print(f"\n  Factoring N = {N}")

    # Why: Check trivial cases first — Shor's is for hard composites, not primes.
    if N % 2 == 0:
        return (2, N // 2)

    # Check if N is a prime power
    for k in range(2, int(np.log2(N)) + 1):
        root = round(N ** (1 / k))
        if root ** k == N:
            if verbose:
                print(f"  N = {root}^{k} is a prime power")
            return (root, N // root)

    for attempt in range(max_attempts):
        # Step 1: Choose random a
        a = np.random.randint(2, N)

        if verbose:
            print(f"\n  Attempt {attempt + 1}: a = {a}")

        # Step 2: Check gcd
        g = gcd(a, N)
        if g > 1:
            # Why: If gcd(a, N) > 1, we got lucky — a shares a factor with N.
            # No quantum computation needed!
            if verbose:
                print(f"    Lucky! gcd({a}, {N}) = {g}")
            return (g, N // g)

        # Step 3: Quantum order finding
        r = quantum_order_finding(a, N, verbose=verbose)

        if r is None:
            if verbose:
                print(f"    Order finding failed, retrying...")
            continue

        if verbose:
            print(f"    Order r = {r}")

        # Step 4: Check if r is useful
        # Why: We need r to be even so that a^{r/2} is an integer.
        # If r is odd, this attempt fails and we pick a new a.
        if r % 2 != 0:
            if verbose:
                print(f"    r = {r} is odd → useless, retrying")
            continue

        # Step 5: Compute factors
        x = mod_exp(a, r // 2, N)

        # Why: If a^{r/2} ≡ -1 (mod N), then a^{r/2}+1 ≡ 0 (mod N), so
        # gcd(a^{r/2}+1, N) = N — a trivial factor.  We need a^{r/2} ≢ ±1.
        if x == N - 1:  # x ≡ -1 (mod N)
            if verbose:
                print(f"    a^(r/2) ≡ -1 (mod N) → useless, retrying")
            continue

        factor1 = gcd(x - 1, N)
        factor2 = gcd(x + 1, N)

        if verbose:
            print(f"    a^(r/2) mod N = {x}")
            print(f"    gcd({x}-1, {N}) = {factor1}")
            print(f"    gcd({x}+1, {N}) = {factor2}")

        if 1 < factor1 < N:
            return (factor1, N // factor1)
        if 1 < factor2 < N:
            return (factor2, N // factor2)

        if verbose:
            print(f"    No non-trivial factor found, retrying")

    if verbose:
        print(f"\n  Failed to factor N = {N} after {max_attempts} attempts")
    return None


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_classical_order():
    """Show classical order finding."""
    print("=" * 60)
    print("DEMO 1: Classical Order Finding")
    print("=" * 60)

    N = 15
    print(f"\n  Orders of a mod {N}:")
    print(f"  {'a':<5} {'Order r':<10} {'a^r mod N':<12} {'Powers a^1, a^2, ..., a^r'}")
    print(f"  {'─' * 60}")

    for a in range(2, N):
        if gcd(a, N) != 1:
            print(f"  {a:<5} {'N/A':<10} {'–':<12} gcd({a},{N}) = {gcd(a,N)} > 1")
            continue
        r = find_order_classical(a, N)
        powers = [mod_exp(a, k, N) for k in range(1, r + 1)]
        print(f"  {a:<5} {r:<10} {mod_exp(a, r, N):<12} {powers}")


def demo_factoring_from_order():
    """Show how order finding leads to factoring."""
    print("\n" + "=" * 60)
    print("DEMO 2: From Order Finding to Factoring")
    print("=" * 60)

    N = 15
    print(f"\n  N = {N}")
    print(f"\n  {'a':<5} {'r':<5} {'r even?':<10} {'a^(r/2)':<10} {'gcd(a^(r/2)±1, N)':<25} {'Factors'}")
    print(f"  {'─' * 65}")

    for a in [2, 4, 7, 8, 11, 13, 14]:
        if gcd(a, N) > 1:
            continue
        r = find_order_classical(a, N)
        even = r % 2 == 0

        if even:
            x = mod_exp(a, r // 2, N)
            f1 = gcd(x - 1, N)
            f2 = gcd(x + 1, N)
            useful = 1 < f1 < N or 1 < f2 < N
            factors = f"{f1}, {N//f1}" if 1 < f1 < N else (f"{f2}, {N//f2}" if 1 < f2 < N else "trivial")
        else:
            x = '-'
            f1 = f2 = '-'
            useful = False
            factors = "r odd"

        print(f"  {a:<5} {r:<5} {'Yes' if even else 'No':<10} {str(x):<10} "
              f"{'gcd('+str(x)+'-1,'+str(N)+')='+str(f1)+', gcd('+str(x)+'+1,'+str(N)+')='+str(f2) if even else 'N/A':<25} {factors}")

    # Why: Not every (a, r) pair gives factors.  The success probability per
    # random a is at least 1 - 1/2^{k-1} where k is the number of distinct
    # prime factors of N.  For N = 15 = 3×5 (k=2), success probability ≥ 1/2.


def demo_shor_15():
    """Run Shor's algorithm on N=15."""
    print("\n" + "=" * 60)
    print("DEMO 3: Shor's Algorithm — Factoring N = 15")
    print("=" * 60)

    result = shors_algorithm(15, max_attempts=10, verbose=True)
    if result:
        print(f"\n  *** Result: 15 = {result[0]} × {result[1]} ***")


def demo_shor_21():
    """Run Shor's algorithm on N=21."""
    print("\n" + "=" * 60)
    print("DEMO 4: Shor's Algorithm — Factoring N = 21")
    print("=" * 60)

    result = shors_algorithm(21, max_attempts=10, verbose=True)
    if result:
        print(f"\n  *** Result: 21 = {result[0]} × {result[1]} ***")


def demo_continued_fractions():
    """Illustrate continued fraction expansion."""
    print("\n" + "=" * 60)
    print("DEMO 5: Continued Fractions in Shor's Algorithm")
    print("=" * 60)

    # Why: After measuring s from the QFT register, we need to find r from
    # s/Q ≈ j/r.  Continued fractions give the best rational approximation
    # with bounded denominator — exactly what we need.
    print(f"\n  Example: Factoring N=15 with a=7, order r=4")
    print(f"  QFT register has Q = 2^8 = 256 qubits (for illustration)")

    Q = 256
    r_true = 4

    # The QFT peaks at s = j·Q/r for j=0,1,...,r-1
    print(f"\n  Expected QFT peaks at s = j × Q/r = j × {Q}/{r_true}:")
    for j in range(r_true):
        s = j * Q // r_true
        print(f"    j={j}: s = {s}")
        if s > 0:
            convergents = continued_fraction_convergents(s, Q, 15)
            print(f"    Convergents of {s}/{Q}:")
            for p, q in convergents:
                match = " ← ORDER!" if q == r_true else ""
                print(f"      {p}/{q} = {p/q:.6f}{match}")


def demo_complexity_comparison():
    """Compare classical vs quantum factoring complexity."""
    print("\n" + "=" * 60)
    print("DEMO 6: Classical vs Quantum Factoring Complexity")
    print("=" * 60)

    print(f"\n  {'Bits (n)':<10} {'N ≈ 2^n':<15} {'Trial Division':<18} {'Shor (gates)':<15}")
    print(f"  {'─' * 58}")

    for n_bits in [10, 20, 50, 100, 256, 512, 1024, 2048]:
        N_approx = 2 ** n_bits
        # Classical trial division: O(√N) = O(2^{n/2})
        classical = f"2^{n_bits//2}"
        # Shor's: O(n³) = O(log³N)
        quantum = f"{n_bits**3}"
        print(f"  {n_bits:<10} {'2^'+str(n_bits):<15} {classical:<18} {quantum:<15}")

    # Why: The exponential gap between 2^{n/2} and n³ is why Shor's algorithm
    # threatens RSA encryption.  RSA-2048 (n=2048) requires 2^1024 operations
    # classically but only ~2048³ ≈ 10⁹⁺ quantum gates — feasible on a
    # future fault-tolerant quantum computer.
    print(f"\n  Classical: O(2^{{n/2}}) — exponential in key size")
    print(f"  Quantum:   O(n³) — polynomial in key size")
    print(f"  RSA-2048 would need ~10^9 quantum gates (feasible with error correction)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Quantum Computing — 08: Shor's Algorithm              ║")
    print("╚══════════════════════════════════════════════════════════╝")

    np.random.seed(2026)

    demo_classical_order()
    demo_factoring_from_order()
    demo_shor_15()
    demo_shor_21()
    demo_continued_fractions()
    demo_complexity_comparison()

    print("\n" + "=" * 60)
    print("All demonstrations complete.")
    print("=" * 60)
