"""
12_qaoa.py — Quantum Approximate Optimization Algorithm (QAOA) for MaxCut

Demonstrates:
  - MaxCut problem formulation on small graphs
  - Cost Hamiltonian and mixer Hamiltonian construction
  - QAOA circuit for p=1 and p=2
  - Parameter optimization using scipy.optimize
  - Comparison with brute-force optimal cut
  - Approximation ratio analysis

Uses NumPy + scipy.optimize.minimize.
"""

import numpy as np
from scipy.optimize import minimize
from typing import List, Tuple, Dict, Set
from itertools import product as iter_product

# ---------------------------------------------------------------------------
# Pauli matrices and gates
# ---------------------------------------------------------------------------
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

KET_0 = np.array([1, 0], dtype=complex)


def tensor(*matrices: np.ndarray) -> np.ndarray:
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result


def Rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


# ---------------------------------------------------------------------------
# Graph representation
# ---------------------------------------------------------------------------

class Graph:
    """Simple undirected graph for MaxCut problems."""

    def __init__(self, n_vertices: int, edges: List[Tuple[int, int]]):
        self.n = n_vertices
        self.edges = edges

    def adjacency_matrix(self) -> np.ndarray:
        A = np.zeros((self.n, self.n), dtype=int)
        for u, v in self.edges:
            A[u][v] = A[v][u] = 1
        return A

    def __repr__(self) -> str:
        return f"Graph({self.n} vertices, {len(self.edges)} edges)"


# ---------------------------------------------------------------------------
# MaxCut problem
# ---------------------------------------------------------------------------

def maxcut_value(graph: Graph, partition: Tuple[int, ...]) -> int:
    """Compute the MaxCut value for a given partition (bitstring).

    Why: A "cut" counts edges between the two partitions (0-group and 1-group).
    For each edge (u,v), the edge is cut iff partition[u] ≠ partition[v].
    MaxCut is NP-hard in general, which is why QAOA is interesting —
    it may provide good approximate solutions on quantum hardware.
    """
    cut = 0
    for u, v in graph.edges:
        if partition[u] != partition[v]:
            cut += 1
    return cut


def brute_force_maxcut(graph: Graph) -> Tuple[int, List[Tuple[int, ...]]]:
    """Find optimal MaxCut by exhaustive search.

    Why: For small graphs, brute force gives the exact answer to compare
    against QAOA.  The complexity is O(2^n), making this infeasible for
    large graphs — precisely the regime where QAOA could help.
    """
    best_cut = 0
    best_partitions = []

    for bits in iter_product([0, 1], repeat=graph.n):
        cut = maxcut_value(graph, bits)
        if cut > best_cut:
            best_cut = cut
            best_partitions = [bits]
        elif cut == best_cut:
            best_partitions.append(bits)

    return best_cut, best_partitions


# ---------------------------------------------------------------------------
# QAOA Hamiltonians
# ---------------------------------------------------------------------------

def build_cost_hamiltonian(graph: Graph) -> np.ndarray:
    """Build the MaxCut cost Hamiltonian.

    C = Σ_{(u,v) ∈ E} (1/2)(I - Z_u Z_v)

    Why: The cost Hamiltonian encodes the MaxCut objective function.
    For edge (u,v): (I - Z_u Z_v)/2 has eigenvalue 1 when qubits u,v differ
    and 0 when they agree — exactly the cut contribution.  The ground state
    of -C (or equivalently, the maximum eigenvalue of C) corresponds to the
    optimal MaxCut partition.
    """
    n = graph.n
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)

    for u, v in graph.edges:
        # Build Z_u Z_v
        ops = [I] * n
        ops[u] = Z
        ops[v] = Z
        ZZ = ops[0]
        for op in ops[1:]:
            ZZ = np.kron(ZZ, op)

        H += 0.5 * (np.eye(dim, dtype=complex) - ZZ)

    return H


def build_mixer_hamiltonian(n: int) -> np.ndarray:
    """Build the standard QAOA mixer Hamiltonian: B = Σ_i X_i.

    Why: The mixer generates transitions between computational basis states
    (different partitions).  Starting from the uniform superposition (ground
    state of B), the mixer ensures that the algorithm can explore all possible
    partitions.  Without it, the cost unitary alone would only add phases
    and never change the measurement probabilities.
    """
    dim = 2 ** n
    B = np.zeros((dim, dim), dtype=complex)

    for i in range(n):
        ops = [I] * n
        ops[i] = X
        Xi = ops[0]
        for op in ops[1:]:
            Xi = np.kron(Xi, op)
        B += Xi

    return B


# ---------------------------------------------------------------------------
# QAOA Circuit
# ---------------------------------------------------------------------------

def qaoa_state(gamma: np.ndarray, beta: np.ndarray,
               graph: Graph) -> np.ndarray:
    """Construct the QAOA state for given parameters.

    |γ, β⟩ = U_B(β_p) U_C(γ_p) ... U_B(β_1) U_C(γ_1) |+⟩^n

    Why: QAOA alternates between cost unitary e^{-iγC} and mixer unitary
    e^{-iβB}.  The intuition: the cost unitary imprints the problem structure
    as phases, and the mixer unitary converts those phases into amplitude
    differences.  More layers (higher p) allows finer control.
    """
    n = graph.n
    p = len(gamma)  # Number of QAOA layers
    dim = 2 ** n

    C = build_cost_hamiltonian(graph)
    B = build_mixer_hamiltonian(n)

    # Start from |+⟩^n (uniform superposition)
    # Why: |+⟩^n is the ground state of the mixer B, which is the standard
    # QAOA initial state.  It gives equal probability to all partitions.
    state = np.ones(dim, dtype=complex) / np.sqrt(dim)

    for layer in range(p):
        # Cost unitary: e^{-iγ_k C}
        # Why: We use matrix exponentiation.  For diagonal C (which it is in
        # the computational basis), this is just element-wise e^{-iγ*c_j}.
        UC = np.diag(np.exp(-1j * gamma[layer] * np.diag(C)))
        state = UC @ state

        # Mixer unitary: e^{-iβ_k B}
        # Why: B is not diagonal, so we need full matrix exponential.
        # For small systems we compute it via eigendecomposition.
        eigenvalues, eigenvectors = np.linalg.eigh(B)
        UB = eigenvectors @ np.diag(np.exp(-1j * beta[layer] * eigenvalues)) @ eigenvectors.conj().T
        state = UB @ state

    return state


def qaoa_expectation(params: np.ndarray, graph: Graph, p: int) -> float:
    """Compute ⟨γ,β|C|γ,β⟩ (the expected cut value).

    Why: The negative of this is our objective function for optimization.
    We want to MAXIMIZE the cut, so we MINIMIZE -⟨C⟩.
    """
    gamma = params[:p]
    beta = params[p:]
    state = qaoa_state(gamma, beta, graph)
    C = build_cost_hamiltonian(graph)
    return np.real(state.conj() @ C @ state)


def run_qaoa(graph: Graph, p: int, n_restarts: int = 10,
             verbose: bool = True) -> Dict:
    """Run QAOA optimization.

    Why: Like VQE, QAOA uses a classical optimizer to find the best circuit
    parameters.  We use multiple restarts because the landscape can have
    local minima, especially for higher p.
    """
    best_energy = -np.inf
    best_params = None

    for restart in range(n_restarts):
        # Random initial parameters
        x0 = np.random.uniform(0, 2 * np.pi, 2 * p)

        # Minimize negative expectation (maximize cut)
        result = minimize(
            lambda params: -qaoa_expectation(params, graph, p),
            x0,
            method='COBYLA',
            options={'maxiter': 500, 'rhobeg': 0.5}
        )

        energy = -result.fun
        if energy > best_energy:
            best_energy = energy
            best_params = result.x

        if verbose and restart < 3:
            print(f"    Restart {restart + 1}: ⟨C⟩ = {energy:.6f}")

    gamma_opt = best_params[:p]
    beta_opt = best_params[p:]

    # Get the optimal state and measurement probabilities
    state = qaoa_state(gamma_opt, beta_opt, graph)
    probs = np.abs(state) ** 2

    return {
        'energy': best_energy,
        'gamma': gamma_opt,
        'beta': beta_opt,
        'state': state,
        'probs': probs,
    }


# ---------------------------------------------------------------------------
# Example graphs
# ---------------------------------------------------------------------------

def triangle_graph() -> Graph:
    """3-vertex triangle graph."""
    return Graph(3, [(0, 1), (1, 2), (0, 2)])


def square_graph() -> Graph:
    """4-vertex square (cycle) graph."""
    return Graph(4, [(0, 1), (1, 2), (2, 3), (3, 0)])


def pentagon_graph() -> Graph:
    """5-vertex pentagon (cycle) graph."""
    return Graph(5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])


def petersen_like_graph() -> Graph:
    """4-vertex graph with 5 edges (near-complete)."""
    return Graph(4, [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)])


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_maxcut_formulation():
    """Explain MaxCut and show brute force solution."""
    print("=" * 60)
    print("DEMO 1: MaxCut Problem Formulation")
    print("=" * 60)

    graph = square_graph()
    print(f"\n  Graph: {graph}")
    print(f"  Edges: {graph.edges}")
    print(f"  Adjacency matrix:")
    A = graph.adjacency_matrix()
    for row in A:
        print(f"    {list(row)}")

    print(f"\n  All partitions and their cut values:")
    print(f"  {'Partition':<15} {'Cut value':>10}")
    print(f"  {'─' * 27}")

    for bits in iter_product([0, 1], repeat=graph.n):
        cut = maxcut_value(graph, bits)
        bitstr = ''.join(map(str, bits))
        marker = " ← MAX" if cut == 4 else ""
        print(f"  |{bitstr}⟩         {cut:>10}{marker}")

    best_cut, best_parts = brute_force_maxcut(graph)
    print(f"\n  Optimal cut value: {best_cut}")
    print(f"  Optimal partitions: {[''.join(map(str, p)) for p in best_parts]}")


def demo_qaoa_triangle():
    """Run QAOA on a triangle graph."""
    print("\n" + "=" * 60)
    print("DEMO 2: QAOA on Triangle Graph (3 vertices)")
    print("=" * 60)

    graph = triangle_graph()
    best_cut, best_parts = brute_force_maxcut(graph)
    print(f"\n  {graph}, optimal cut = {best_cut}")

    # Why: The triangle is the simplest non-trivial MaxCut instance.
    # Any partition cuts exactly 2 of the 3 edges (you can't cut all 3
    # with a bipartition).
    for p in [1, 2]:
        print(f"\n  --- QAOA p = {p} ---")
        result = run_qaoa(graph, p, n_restarts=15, verbose=True)

        approx_ratio = result['energy'] / best_cut
        print(f"    ⟨C⟩ = {result['energy']:.6f}")
        print(f"    Approximation ratio: {approx_ratio:.4f} ({approx_ratio*100:.1f}%)")

        # Show top measurement outcomes
        probs = result['probs']
        top_idx = np.argsort(probs)[::-1][:4]
        print(f"    Top measurement outcomes:")
        for idx in top_idx:
            bitstr = format(idx, f'0{graph.n}b')
            cut = maxcut_value(graph, tuple(int(b) for b in bitstr))
            print(f"      |{bitstr}⟩: P = {probs[idx]:.4f}, cut = {cut}")


def demo_qaoa_square():
    """Run QAOA on a square graph."""
    print("\n" + "=" * 60)
    print("DEMO 3: QAOA on Square Graph (4 vertices)")
    print("=" * 60)

    graph = square_graph()
    best_cut, _ = brute_force_maxcut(graph)
    print(f"\n  {graph}, optimal cut = {best_cut}")

    for p in [1, 2, 3]:
        print(f"\n  --- QAOA p = {p} ---")
        result = run_qaoa(graph, p, n_restarts=20, verbose=False)

        approx_ratio = result['energy'] / best_cut
        print(f"    ⟨C⟩ = {result['energy']:.6f}, approx ratio = {approx_ratio:.4f}")
        print(f"    γ = [{', '.join(f'{g:.4f}' for g in result['gamma'])}]")
        print(f"    β = [{', '.join(f'{b:.4f}' for b in result['beta'])}]")

        # Probability of sampling optimal solution
        probs = result['probs']
        optimal_prob = 0
        for idx in range(2 ** graph.n):
            bitstr = format(idx, f'0{graph.n}b')
            cut = maxcut_value(graph, tuple(int(b) for b in bitstr))
            if cut == best_cut:
                optimal_prob += probs[idx]
        print(f"    P(optimal solution) = {optimal_prob:.4f}")


def demo_qaoa_depth_analysis():
    """Show how QAOA quality improves with depth p."""
    print("\n" + "=" * 60)
    print("DEMO 4: QAOA Quality vs Circuit Depth p")
    print("=" * 60)

    graph = petersen_like_graph()
    best_cut, _ = brute_force_maxcut(graph)
    print(f"\n  {graph}, optimal cut = {best_cut}")

    # Why: The QAOA approximation ratio monotonically improves with p.
    # At p → ∞, QAOA converges to the exact solution (it becomes equivalent
    # to quantum adiabatic computation).  At p=1, a worst-case guarantee
    # of 0.6924 exists for MaxCut on 3-regular graphs.
    print(f"\n  {'p':<5} {'⟨C⟩':>10} {'Ratio':>10} {'P(optimal)':>12} {'Params':>8}")
    print(f"  {'─' * 48}")

    for p in range(1, 6):
        result = run_qaoa(graph, p, n_restarts=25, verbose=False)
        approx_ratio = result['energy'] / best_cut

        probs = result['probs']
        optimal_prob = sum(probs[idx] for idx in range(2 ** graph.n)
                          if maxcut_value(graph, tuple(int(b) for b in format(idx, f'0{graph.n}b'))) == best_cut)

        print(f"  {p:<5} {result['energy']:>10.4f} {approx_ratio:>10.4f} "
              f"{optimal_prob:>12.4f} {2*p:>8}")

    print(f"\n  Higher p → better approximation (but more parameters & deeper circuit)")


def demo_cost_hamiltonian():
    """Inspect the cost Hamiltonian structure."""
    print("\n" + "=" * 60)
    print("DEMO 5: Cost Hamiltonian Structure")
    print("=" * 60)

    graph = triangle_graph()
    C = build_cost_hamiltonian(graph)

    print(f"\n  Triangle graph cost Hamiltonian C:")
    # Why: For MaxCut, C is diagonal in the computational basis.  Each diagonal
    # entry is the cut value of the corresponding bitstring.  This is why the
    # cost unitary e^{-iγC} simply applies phases proportional to cut values.
    diag = np.diag(C).real
    for idx in range(2 ** graph.n):
        bitstr = format(idx, f'0{graph.n}b')
        cut_check = maxcut_value(graph, tuple(int(b) for b in bitstr))
        print(f"    |{bitstr}⟩: C eigenvalue = {diag[idx]:.1f} (cut value = {cut_check})")

    print(f"\n  C is diagonal → eigenvalues = cut values")
    print(f"  Max eigenvalue = optimal cut = {max(diag):.0f}")

    # Show mixer
    B = build_mixer_hamiltonian(graph.n)
    print(f"\n  Mixer Hamiltonian B = X₀ + X₁ + X₂")
    eigenvalues_B = np.sort(np.linalg.eigvalsh(B))
    print(f"  B eigenvalues: {eigenvalues_B}")
    print(f"  Ground state of B is |+⟩^n (the QAOA initial state)")


def demo_sampling():
    """Show how to extract solutions from QAOA by sampling."""
    print("\n" + "=" * 60)
    print("DEMO 6: Sampling Solutions from QAOA")
    print("=" * 60)

    graph = pentagon_graph()
    best_cut, _ = brute_force_maxcut(graph)
    print(f"\n  {graph}, optimal cut = {best_cut}")

    # Run QAOA
    result = run_qaoa(graph, p=2, n_restarts=20, verbose=False)
    probs = result['probs']

    # Why: On real quantum hardware, we can't read the state vector directly.
    # Instead, we run the circuit many times and sample bitstrings.  The most
    # frequently observed bitstring is our candidate solution.
    n_samples = 1000
    samples = np.random.choice(2 ** graph.n, size=n_samples, p=probs)
    sample_counts = {}
    for s in samples:
        bs = format(s, f'0{graph.n}b')
        sample_counts[bs] = sample_counts.get(bs, 0) + 1

    # Sort by frequency
    sorted_samples = sorted(sample_counts.items(), key=lambda x: -x[1])

    print(f"\n  Top 8 sampled solutions ({n_samples} shots):")
    print(f"  {'Bitstring':<12} {'Frequency':>10} {'Cut Value':>10}")
    print(f"  {'─' * 34}")
    for bitstr, count in sorted_samples[:8]:
        cut = maxcut_value(graph, tuple(int(b) for b in bitstr))
        marker = " *" if cut == best_cut else ""
        print(f"  |{bitstr}⟩     {count:>10} {cut:>10}{marker}")

    # Best sampled solution
    best_sampled = sorted_samples[0]
    best_sampled_cut = maxcut_value(graph, tuple(int(b) for b in best_sampled[0]))
    print(f"\n  Most frequent: |{best_sampled[0]}⟩ (cut = {best_sampled_cut})")
    print(f"  Optimal cut: {best_cut}")
    print(f"  Found optimal? {'Yes' if best_sampled_cut == best_cut else 'No'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Quantum Computing — 12: QAOA for MaxCut               ║")
    print("╚══════════════════════════════════════════════════════════╝")

    np.random.seed(2026)

    demo_maxcut_formulation()
    demo_qaoa_triangle()
    demo_qaoa_square()
    demo_qaoa_depth_analysis()
    demo_cost_hamiltonian()
    demo_sampling()

    print("\n" + "=" * 60)
    print("All demonstrations complete.")
    print("=" * 60)
