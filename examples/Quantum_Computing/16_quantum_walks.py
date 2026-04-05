"""
16_quantum_walks.py — Quantum Walks on Graphs

Demonstrates:
  - Discrete-time coined quantum walk on a line
  - Continuous-time quantum walk on a line graph
  - Classical random walk for comparison
  - Probability distribution comparison (quantum vs classical)
  - Quadratic speedup in spreading rate
  - Quantum walk on a cycle graph

All computations use pure NumPy.
"""

import numpy as np
from typing import Tuple

# ---------------------------------------------------------------------------
# Coin operators
# ---------------------------------------------------------------------------

HADAMARD_COIN = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

GROVER_COIN_2D = np.array([
    [-1,  1,  1,  1],
    [ 1, -1,  1,  1],
    [ 1,  1, -1,  1],
    [ 1,  1,  1, -1],
], dtype=complex) / 2


def y_coin(theta: float) -> np.ndarray:
    """Parametric coin operator (rotation around Y-axis).

    Why: Different coin operators produce different walk dynamics.
    The Hadamard coin is the standard choice, but a Y-rotation coin
    with θ = π/4 produces symmetric distributions, unlike the Hadamard
    coin which is asymmetric for the |0⟩ coin state.
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=complex)


# ---------------------------------------------------------------------------
# Discrete Coined Quantum Walk on a Line
# ---------------------------------------------------------------------------

def discrete_quantum_walk(n_steps: int, n_positions: int,
                           coin: np.ndarray = None,
                           initial_coin_state: np.ndarray = None
                           ) -> np.ndarray:
    """Simulate a discrete-time coined quantum walk on a line.

    The walker lives on positions {0, 1, ..., n_positions-1} and carries
    a 2-dimensional coin.  Each step: (1) apply coin, (2) shift position
    conditioned on coin state.

    Why: The discrete quantum walk is the quantum analog of a classical
    random walk, but with interference effects.  The walker spreads
    ballistically (∝ t) instead of diffusively (∝ √t), giving a quadratic
    speedup.  This is the basis for quantum walk search algorithms.
    """
    if coin is None:
        coin = HADAMARD_COIN
    if initial_coin_state is None:
        initial_coin_state = np.array([1, 0], dtype=complex)  # |0⟩ coin

    dim = 2 * n_positions  # coin ⊗ position
    state = np.zeros(dim, dtype=complex)

    # Start at the center position
    center = n_positions // 2
    # State encoding: index = coin_state * n_positions + position
    for c_idx in range(2):
        state[c_idx * n_positions + center] = initial_coin_state[c_idx]

    # Why: The walk operator is W = S · (C ⊗ I_pos), where S is the
    # conditional shift and C is the coin operator.  We apply this
    # operator n_steps times.
    for _ in range(n_steps):
        # Step 1: Apply coin to the coin register
        new_state = np.zeros(dim, dtype=complex)
        for pos in range(n_positions):
            amp_0 = state[0 * n_positions + pos]  # coin=|0⟩ at pos
            amp_1 = state[1 * n_positions + pos]  # coin=|1⟩ at pos
            coin_in = np.array([amp_0, amp_1])
            coin_out = coin @ coin_in
            new_state[0 * n_positions + pos] = coin_out[0]
            new_state[1 * n_positions + pos] = coin_out[1]
        state = new_state

        # Step 2: Conditional shift
        # |0⟩ → move left, |1⟩ → move right
        new_state = np.zeros(dim, dtype=complex)
        for pos in range(n_positions):
            # Coin |0⟩: shift left
            if pos > 0:
                new_state[0 * n_positions + (pos - 1)] += state[0 * n_positions + pos]
            # Coin |1⟩: shift right
            if pos < n_positions - 1:
                new_state[1 * n_positions + (pos + 1)] += state[1 * n_positions + pos]
        state = new_state

    # Compute position probabilities by tracing out coin
    probs = np.zeros(n_positions)
    for pos in range(n_positions):
        probs[pos] = (np.abs(state[0 * n_positions + pos]) ** 2 +
                      np.abs(state[1 * n_positions + pos]) ** 2)

    return probs


# ---------------------------------------------------------------------------
# Continuous-Time Quantum Walk
# ---------------------------------------------------------------------------

def continuous_quantum_walk(t: float, n_positions: int) -> np.ndarray:
    """Simulate a continuous-time quantum walk on a line graph.

    The evolution is U(t) = e^{-iAt} where A is the adjacency matrix.

    Why: Unlike the discrete walk, no coin is needed — the graph structure
    alone determines the dynamics.  The continuous-time walk on a line
    also spreads ballistically.  It is the natural framework for spatial
    search algorithms (Childs & Goldstone).
    """
    # Adjacency matrix for a line graph (path graph)
    A = np.zeros((n_positions, n_positions), dtype=complex)
    for i in range(n_positions - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0

    # Why: We compute e^{-iAt} exactly via eigendecomposition.
    # On a quantum computer, this would be implemented with Hamiltonian
    # simulation (e.g., Trotter decomposition).
    eigvals, eigvecs = np.linalg.eigh(A)
    U = eigvecs @ np.diag(np.exp(-1j * eigvals * t)) @ eigvecs.conj().T

    # Start at center
    center = n_positions // 2
    initial = np.zeros(n_positions, dtype=complex)
    initial[center] = 1.0

    state = U @ initial
    probs = np.abs(state) ** 2

    return probs


# ---------------------------------------------------------------------------
# Classical Random Walk
# ---------------------------------------------------------------------------

def classical_random_walk(n_steps: int, n_positions: int,
                           n_trials: int = 100000) -> np.ndarray:
    """Simulate a classical random walk on a line via sampling.

    Why: The classical walk spreads diffusively — the standard deviation
    grows as σ ∝ √t.  Comparing with the quantum walk (σ ∝ t) highlights
    the quadratic speedup from quantum coherence.
    """
    center = n_positions // 2
    counts = np.zeros(n_positions)

    for _ in range(n_trials):
        pos = center
        for _ in range(n_steps):
            step = np.random.choice([-1, 1])
            pos = max(0, min(n_positions - 1, pos + step))
        counts[pos] += 1

    return counts / n_trials


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def walk_statistics(probs: np.ndarray) -> Tuple[float, float]:
    """Compute mean and standard deviation of a probability distribution."""
    positions = np.arange(len(probs))
    mean = np.sum(positions * probs)
    var = np.sum((positions - mean) ** 2 * probs)
    return mean, np.sqrt(max(var, 0))


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_discrete_walk():
    """Show discrete coined quantum walk on a line."""
    print("=" * 60)
    print("DEMO 1: Discrete Coined Quantum Walk")
    print("=" * 60)

    n_positions = 101
    center = n_positions // 2

    # Why: The Hadamard coin with initial state |0⟩ produces an asymmetric
    # distribution — it moves more to the left.  This is because the Hadamard
    # gate treats |0⟩ and |1⟩ differently in its phase structure.
    for n_steps in [10, 25, 50]:
        probs = discrete_quantum_walk(n_steps, n_positions)
        _, std = walk_statistics(probs)

        print(f"\n  Hadamard walk, {n_steps} steps (σ = {std:.2f}):")
        # Show a condensed ASCII histogram
        _print_distribution(probs, center, n_steps, width=50)

    # Why: Using the symmetric initial coin state (|0⟩ + i|1⟩)/√2 produces
    # a symmetric distribution, which better illustrates the quantum speedup.
    print(f"\n  Symmetric coin state (|0⟩ + i|1⟩)/√2, 50 steps:")
    sym_coin = np.array([1, 1j], dtype=complex) / np.sqrt(2)
    probs_sym = discrete_quantum_walk(50, n_positions,
                                       initial_coin_state=sym_coin)
    _, std_sym = walk_statistics(probs_sym)
    print(f"  σ = {std_sym:.2f}")
    _print_distribution(probs_sym, center, 50, width=50)


def demo_continuous_walk():
    """Show continuous-time quantum walk."""
    print("\n" + "=" * 60)
    print("DEMO 2: Continuous-Time Quantum Walk")
    print("=" * 60)

    n_positions = 101
    center = n_positions // 2

    for t in [5.0, 15.0, 30.0]:
        probs = continuous_quantum_walk(t, n_positions)
        _, std = walk_statistics(probs)

        print(f"\n  t = {t:.1f} (σ = {std:.2f}):")
        _print_distribution(probs, center, int(t), width=50)


def demo_quantum_vs_classical():
    """Compare quantum and classical walk distributions."""
    print("\n" + "=" * 60)
    print("DEMO 3: Quantum vs Classical Random Walk")
    print("=" * 60)

    n_positions = 101
    center = n_positions // 2
    n_steps = 40

    # Why: The key difference is visible in the probability distribution:
    # classical → Gaussian (concentrated near center),
    # quantum → bimodal peaks at the edges of the spreading front.
    sym_coin = np.array([1, 1j], dtype=complex) / np.sqrt(2)
    q_probs = discrete_quantum_walk(n_steps, n_positions,
                                     initial_coin_state=sym_coin)
    c_probs = classical_random_walk(n_steps, n_positions, n_trials=200000)

    q_mean, q_std = walk_statistics(q_probs)
    c_mean, c_std = walk_statistics(c_probs)

    print(f"\n  {n_steps} steps:")
    print(f"    Quantum:   mean = {q_mean:.1f}, σ = {q_std:.2f}")
    print(f"    Classical: mean = {c_mean:.1f}, σ = {c_std:.2f}")
    print(f"    Ratio σ_q / σ_c = {q_std / c_std:.2f} "
          f"(expected ≈ √{n_steps} / const ≈ {np.sqrt(n_steps):.1f}x for large t)")

    # Side-by-side comparison
    print(f"\n  {'Position':>10} {'Quantum':>10} {'Classical':>10}")
    print(f"  {'─' * 32}")
    for pos in range(center - n_steps, center + n_steps + 1, 2):
        if 0 <= pos < n_positions:
            print(f"  {pos - center:10d} {q_probs[pos]:10.4f} {c_probs[pos]:10.4f}")


def demo_spreading_rate():
    """Quantify the spreading rate difference."""
    print("\n" + "=" * 60)
    print("DEMO 4: Spreading Rate: Linear vs Square Root")
    print("=" * 60)

    # Why: The quantum walk standard deviation grows linearly in time
    # (ballistic spreading), while the classical walk grows as √t
    # (diffusive spreading).  This is the source of quadratic quantum speedup.
    n_positions = 201
    sym_coin = np.array([1, 1j], dtype=complex) / np.sqrt(2)

    print(f"\n  {'Steps':>8} {'σ_quantum':>12} {'σ_classical':>14} "
          f"{'σ_q/t':>10} {'σ_c/√t':>10}")
    print(f"  {'─' * 58}")

    for n_steps in [5, 10, 20, 30, 40, 50, 60]:
        q_probs = discrete_quantum_walk(n_steps, n_positions,
                                         initial_coin_state=sym_coin)
        c_probs = classical_random_walk(n_steps, n_positions,
                                         n_trials=100000)

        _, q_std = walk_statistics(q_probs)
        _, c_std = walk_statistics(c_probs)

        # Why: σ_q / t should be roughly constant (linear growth),
        # while σ_c / √t should be roughly constant (diffusive growth).
        q_ratio = q_std / n_steps if n_steps > 0 else 0
        c_ratio = c_std / np.sqrt(n_steps) if n_steps > 0 else 0

        print(f"  {n_steps:8d} {q_std:12.4f} {c_std:14.4f} "
              f"{q_ratio:10.4f} {c_ratio:10.4f}")

    print(f"\n  σ_q/t ≈ const confirms ballistic (linear) spreading")
    print(f"  σ_c/√t ≈ const confirms diffusive (√t) spreading")


def demo_walk_on_cycle():
    """Quantum walk on a cycle graph (periodic boundary conditions)."""
    print("\n" + "=" * 60)
    print("DEMO 5: Quantum Walk on a Cycle Graph")
    print("=" * 60)

    # Why: On a cycle with N nodes, the continuous-time quantum walk
    # exhibits perfect state transfer and revival phenomena.  The walker
    # returns to the starting position after specific times determined
    # by the graph spectrum.
    n_nodes = 20

    # Adjacency matrix for cycle
    A = np.zeros((n_nodes, n_nodes), dtype=complex)
    for i in range(n_nodes):
        A[i, (i + 1) % n_nodes] = 1.0
        A[(i + 1) % n_nodes, i] = 1.0

    eigvals, eigvecs = np.linalg.eigh(A)

    # Start at node 0
    initial = np.zeros(n_nodes, dtype=complex)
    initial[0] = 1.0

    print(f"\n  Cycle graph with {n_nodes} nodes, starting at node 0:")
    print(f"\n  {'t':>8} {'P(start)':>10} {'P(opposite)':>14} {'Max P':>10} {'Max node':>10}")
    print(f"  {'─' * 55}")

    for t in np.linspace(0, 30, 25):
        U = eigvecs @ np.diag(np.exp(-1j * eigvals * t)) @ eigvecs.conj().T
        state = U @ initial
        probs = np.abs(state) ** 2

        p_start = probs[0]
        p_opposite = probs[n_nodes // 2]
        max_p = np.max(probs)
        max_node = np.argmax(probs)

        print(f"  {t:8.2f} {p_start:10.4f} {p_opposite:14.4f} "
              f"{max_p:10.4f} {max_node:10d}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_distribution(probs: np.ndarray, center: int, n_steps: int,
                        width: int = 50) -> None:
    """Print an ASCII histogram of the probability distribution."""
    start = max(0, center - n_steps - 2)
    end = min(len(probs), center + n_steps + 3)
    max_prob = max(probs[start:end])

    if max_prob < 1e-10:
        print("    (all probabilities near zero)")
        return

    # Sample every few positions for compact display
    step = max(1, (end - start) // 25)
    for pos in range(start, end, step):
        bar_len = int(probs[pos] / max_prob * width)
        label = pos - center
        print(f"    {label:+4d} |{'#' * bar_len}{' ' * (width - bar_len)}| "
              f"{probs[pos]:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Quantum Computing — 16: Quantum Walks                 ║")
    print("╚══════════════════════════════════════════════════════════╝")

    np.random.seed(2026)

    demo_discrete_walk()
    demo_continuous_walk()
    demo_quantum_vs_classical()
    demo_spreading_rate()
    demo_walk_on_cycle()

    print("\n" + "=" * 60)
    print("All demonstrations complete.")
    print("=" * 60)
