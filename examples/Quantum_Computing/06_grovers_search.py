"""
06_grovers_search.py — Grover's Search Algorithm with Amplitude Amplification

Demonstrates:
  - Oracle construction (phase flip on marked items)
  - Diffusion operator (inversion about the mean)
  - Grover iteration: oracle + diffusion
  - Amplitude evolution across iterations
  - Optimal iteration count ≈ (π/4)√N
  - Multi-target Grover search
  - ASCII-art visualization of probability amplitudes

All computations use pure NumPy.
"""

import numpy as np
from typing import List, Set

# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def tensor_n(gate: np.ndarray, n: int) -> np.ndarray:
    """Compute gate^{⊗n} (n-fold tensor product)."""
    result = gate
    for _ in range(n - 1):
        result = np.kron(result, gate)
    return result


# ---------------------------------------------------------------------------
# Grover's Algorithm Components
# ---------------------------------------------------------------------------

def build_oracle(n_qubits: int, marked: Set[int]) -> np.ndarray:
    """Build the Grover oracle: flip the phase of marked states.

    O|x⟩ = -|x⟩  if x ∈ marked
    O|x⟩ =  |x⟩  otherwise

    Why: The oracle is a diagonal matrix with -1 for marked items and +1
    elsewhere.  This phase flip is invisible in measurement probabilities
    (|−1|² = 1) but is crucial for the subsequent diffusion step to amplify
    the marked state's amplitude.
    """
    N = 2 ** n_qubits
    diag = np.ones(N, dtype=complex)
    for m in marked:
        diag[m] = -1
    return np.diag(diag)


def build_diffusion(n_qubits: int) -> np.ndarray:
    """Build the Grover diffusion operator: D = 2|ψ⟩⟨ψ| - I.

    where |ψ⟩ = H^⊗n|0⟩ = (1/√N) Σ|x⟩ is the uniform superposition.

    Why: The diffusion operator performs "inversion about the mean."
    It reflects every amplitude about the average amplitude, effectively
    boosting amplitudes above the mean and suppressing those below.
    After the oracle flips the sign of the marked state (making it negative),
    the diffusion operator pushes it even further above the mean.
    """
    N = 2 ** n_qubits
    # |ψ⟩ = uniform superposition
    psi = np.ones(N, dtype=complex) / np.sqrt(N)
    # D = 2|ψ⟩⟨ψ| - I
    return 2 * np.outer(psi, psi.conj()) - np.eye(N, dtype=complex)


def grover_iteration(state: np.ndarray, oracle: np.ndarray,
                     diffusion: np.ndarray) -> np.ndarray:
    """Apply one Grover iteration: oracle then diffusion."""
    return diffusion @ (oracle @ state)


def optimal_iterations(N: int, M: int) -> int:
    """Compute the optimal number of Grover iterations.

    Why: The exact formula is ⌊(π/4)√(N/M)⌋, where N is the total number of
    states and M is the number of marked states.  Going beyond this count
    actually *decreases* the success probability — the amplitude oscillates
    like a sine wave, and we want to stop at the first peak.
    """
    return int(np.floor(np.pi / 4 * np.sqrt(N / M)))


# ---------------------------------------------------------------------------
# Full Grover's Algorithm
# ---------------------------------------------------------------------------

def grovers_search(n_qubits: int, marked: Set[int],
                   track_amplitudes: bool = False,
                   num_iterations: int = None) -> dict:
    """Run Grover's search algorithm.

    Args:
        n_qubits: Number of qubits (search space size = 2^n)
        marked: Set of marked item indices
        track_amplitudes: If True, record amplitudes at each step
        num_iterations: Override number of iterations (default: optimal)

    Returns:
        Dictionary with results and (optionally) amplitude history.
    """
    N = 2 ** n_qubits
    M = len(marked)

    if num_iterations is None:
        num_iterations = optimal_iterations(N, M)

    # Why: Start with uniform superposition — every item has equal amplitude
    # 1/√N.  This is the "blank slate" from which Grover amplifies the target.
    state = np.ones(N, dtype=complex) / np.sqrt(N)

    oracle = build_oracle(n_qubits, marked)
    diffusion = build_diffusion(n_qubits)

    history = []
    if track_amplitudes:
        history.append(np.abs(state) ** 2)

    for i in range(num_iterations):
        state = grover_iteration(state, oracle, diffusion)
        if track_amplitudes:
            history.append(np.abs(state) ** 2)

    probs = np.abs(state) ** 2
    success_prob = sum(probs[m] for m in marked)

    # Measure
    outcomes = np.random.choice(N, size=1000, p=probs)
    found = int(np.argmax(np.bincount(outcomes, minlength=N)))

    return {
        'state': state,
        'probs': probs,
        'success_prob': success_prob,
        'iterations': num_iterations,
        'found': found,
        'history': history,
        'N': N,
        'M': M,
    }


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def ascii_bar(value: float, max_width: int = 40) -> str:
    """Create an ASCII bar chart element."""
    width = int(value * max_width)
    return '#' * width + '·' * (max_width - width)


def print_amplitude_evolution(history: List[np.ndarray], marked: Set[int],
                               n_qubits: int) -> None:
    """Print amplitude evolution as ASCII art."""
    N = 2 ** n_qubits
    # Only show a few representative states
    unmarked_idx = next(i for i in range(N) if i not in marked)
    marked_idx = next(iter(marked))

    print(f"\n  Amplitude Evolution (probability):")
    print(f"  {'Iter':<6} {'Marked P':>10} {'Unmarked P':>12} {'Visual (marked)':>45}")
    print(f"  {'─' * 75}")

    for step, probs in enumerate(history):
        p_marked = probs[marked_idx]
        p_unmarked = probs[unmarked_idx]
        bar = ascii_bar(p_marked)
        print(f"  {step:<6} {p_marked:>10.6f} {p_unmarked:>12.6f}   |{bar}|")


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_basic_grover():
    """Basic Grover search on 3 qubits (8 items)."""
    print("=" * 60)
    print("DEMO 1: Basic Grover Search (3 qubits, 1 target)")
    print("=" * 60)

    n = 3
    N = 2 ** n
    target = 5  # Search for |101⟩

    result = grovers_search(n, {target}, track_amplitudes=True)

    print(f"\n  Search space: {N} items (|000⟩ to |111⟩)")
    print(f"  Target: |{format(target, f'0{n}b')}⟩ (index {target})")
    print(f"  Optimal iterations: {result['iterations']}")
    print(f"  Success probability: {result['success_prob']:.6f}")
    print(f"  Found: |{format(result['found'], f'0{n}b')}⟩ (index {result['found']})")

    print_amplitude_evolution(result['history'], {target}, n)

    # Why: With 8 items, Grover needs ≈ π/4·√8 ≈ 2.2 → 2 iterations.
    # Success probability won't be 100% because √8 doesn't give an integer.
    # The probability oscillates sinusoidally and we pick the nearest peak.


def demo_amplitude_amplification():
    """Show how Grover amplification works step by step."""
    print("\n" + "=" * 60)
    print("DEMO 2: Amplitude Amplification — Step by Step")
    print("=" * 60)

    n = 4
    N = 2 ** n
    target = 11

    print(f"\n  Search space: {N} items, target: |{format(target, f'0{n}b')}⟩")

    state = np.ones(N, dtype=complex) / np.sqrt(N)
    oracle = build_oracle(n, {target})
    diffusion = build_diffusion(n)

    opt = optimal_iterations(N, 1)
    print(f"  Optimal iterations: {opt}")
    print(f"  Initial amplitude: 1/√{N} = {1/np.sqrt(N):.6f}")

    # Why: Geometrically, each Grover iteration rotates the state vector by
    # angle 2·arcsin(√(M/N)) toward the target subspace.  The amplitude of
    # the target grows as sin((2k+1)θ) where θ = arcsin(√(M/N)).
    theta = np.arcsin(np.sqrt(1 / N))
    print(f"  Rotation angle per iteration: 2θ = {2*theta:.6f} rad ({np.degrees(2*theta):.2f}°)")

    for i in range(opt + 3):  # Go a bit past optimal
        probs = np.abs(state) ** 2
        p_target = probs[target]
        p_other = probs[0] if 0 != target else probs[1]
        bar = ascii_bar(p_target, 50)

        # Theoretical prediction
        p_theory = np.sin((2 * i + 1) * theta) ** 2

        print(f"\n  Iteration {i}:")
        print(f"    P(target)     = {p_target:.6f}  (theory: {p_theory:.6f})")
        print(f"    P(each other) = {p_other:.6f}")
        print(f"    |{bar}|")

        if i < opt + 2:
            state = grover_iteration(state, oracle, diffusion)

    print(f"\n  Note: probability DECREASES after optimal iteration {opt}!")
    print(f"  Grover's algorithm overshoots if you iterate too many times.")


def demo_multi_target():
    """Grover search with multiple targets."""
    print("\n" + "=" * 60)
    print("DEMO 3: Multi-Target Grover Search")
    print("=" * 60)

    n = 4
    N = 2 ** n

    for M in [1, 2, 4]:
        marked = set(range(M))  # First M items are marked
        result = grovers_search(n, marked, track_amplitudes=True)

        print(f"\n  N={N}, M={M} targets: iterations={result['iterations']}, "
              f"P(success)={result['success_prob']:.4f}")
        print(f"  Optimal = ⌊(π/4)√({N}/{M})⌋ = ⌊{np.pi/4 * np.sqrt(N/M):.2f}⌋ = {optimal_iterations(N, M)}")

    # Why: More targets → fewer iterations needed.  When M = N/4, only 1
    # iteration suffices.  When M = N (all marked), 0 iterations (the initial
    # uniform superposition already has 100% success).
    print(f"\n  Key insight: more targets → fewer iterations needed")
    print(f"  Formula: ≈ (π/4)√(N/M) iterations")


def demo_scaling():
    """Show how Grover's speedup scales with problem size."""
    print("\n" + "=" * 60)
    print("DEMO 4: Grover Scaling — Quadratic Speedup")
    print("=" * 60)

    print(f"\n  {'n qubits':<12} {'N = 2^n':<12} {'Classical':<12} {'Grover':<12} {'Speedup':<10}")
    print(f"  {'─' * 58}")

    for n in range(2, 21):
        N = 2 ** n
        classical = N  # Expected queries for 1 target: N/2 on average, N worst case
        grover = optimal_iterations(N, 1)
        speedup = classical / max(grover, 1)

        # Why: Classical search requires O(N) queries, Grover requires O(√N).
        # This is provably optimal — no quantum algorithm can do better than
        # O(√N) for unstructured search (BBBV theorem, 1997).
        print(f"  {n:<12} {N:<12} {classical:<12} {grover:<12} {speedup:<10.1f}x")

    print(f"\n  Speedup grows as √N — quadratic quantum advantage!")
    print(f"  For N = 10^6: classical ≈ 10^6, Grover ≈ 785 queries.")


def demo_inversion_about_mean():
    """Visualize the inversion-about-mean operation."""
    print("\n" + "=" * 60)
    print("DEMO 5: Inversion About the Mean")
    print("=" * 60)

    n = 3
    N = 2 ** n
    target = 3

    # Start with uniform superposition
    state = np.ones(N, dtype=complex) / np.sqrt(N)
    oracle = build_oracle(n, {target})
    diffusion = build_diffusion(n)

    print(f"\n  Step 1: Uniform superposition")
    amps = np.real(state)
    mean = np.mean(amps)
    print(f"    Amplitudes: [{', '.join(f'{a:.4f}' for a in amps)}]")
    print(f"    Mean: {mean:.4f}")

    # After oracle
    state_after_oracle = oracle @ state
    amps = np.real(state_after_oracle)
    mean = np.mean(amps)
    print(f"\n  Step 2: After oracle (phase flip on target |{format(target, f'0{n}b')}⟩)")
    print(f"    Amplitudes: [{', '.join(f'{a:.4f}' for a in amps)}]")
    print(f"    Mean: {mean:.4f}")

    # After diffusion
    state_after_diff = diffusion @ state_after_oracle
    amps = np.real(state_after_diff)
    mean_after = np.mean(amps)
    print(f"\n  Step 3: After diffusion (inversion about mean)")
    print(f"    Amplitudes: [{', '.join(f'{a:.4f}' for a in amps)}]")
    print(f"    Mean: {mean_after:.4f}")

    # Why: Inversion about the mean: a_new = 2·mean − a_old.
    # The oracle made the target amplitude negative, so it was far below the mean.
    # After inversion, it becomes far ABOVE the mean — amplified!
    print(f"\n  Verification: new_amp = 2·mean - old_amp")
    for i in range(N):
        old = np.real((oracle @ (np.ones(N, dtype=complex) / np.sqrt(N)))[i])
        new = np.real(state_after_diff[i])
        calc = 2 * mean - old
        label = f"|{format(i, f'0{n}b')}⟩"
        marker = " ← TARGET" if i == target else ""
        print(f"    {label}: 2×{mean:.4f} - ({old:+.4f}) = {calc:+.4f}  (actual: {new:+.4f}){marker}")


def demo_probability_oscillation():
    """Show probability oscillating with iteration count."""
    print("\n" + "=" * 60)
    print("DEMO 6: Probability Oscillation")
    print("=" * 60)

    n = 5
    N = 2 ** n
    target = 17
    opt = optimal_iterations(N, 1)

    # Why: The success probability oscillates as sin²((2k+1)θ).
    # Running too few OR too many iterations gives suboptimal results.
    # This is fundamentally different from classical algorithms, where more
    # work always means a better answer.
    print(f"\n  N = {N}, target = |{format(target, f'0{n}b')}⟩")
    print(f"  Optimal iterations: {opt}")
    print(f"\n  {'Iterations':<12} {'P(target)':>12} {'Bar':>45}")
    print(f"  {'─' * 70}")

    state = np.ones(N, dtype=complex) / np.sqrt(N)
    oracle = build_oracle(n, {target})
    diffusion = build_diffusion(n)

    for k in range(min(2 * opt + 4, 20)):
        probs = np.abs(state) ** 2
        p_target = probs[target]
        bar = ascii_bar(p_target, 40)
        marker = " ← OPTIMAL" if k == opt else ""
        print(f"  {k:<12} {p_target:>12.6f}   |{bar}|{marker}")
        state = grover_iteration(state, oracle, diffusion)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Quantum Computing — 06: Grover's Search Algorithm     ║")
    print("╚══════════════════════════════════════════════════════════╝")

    np.random.seed(2026)

    demo_basic_grover()
    demo_amplitude_amplification()
    demo_multi_target()
    demo_scaling()
    demo_inversion_about_mean()
    demo_probability_oscillation()

    print("\n" + "=" * 60)
    print("All demonstrations complete.")
    print("=" * 60)
