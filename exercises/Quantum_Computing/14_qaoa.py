"""
Exercises for Lesson 14: QAOA and Combinatorial Optimization
Topic: Quantum_Computing

Solutions to practice problems from the lesson.
All quantum operations simulated with numpy matrices (no qiskit).
"""

import numpy as np
from scipy.optimize import minimize
from itertools import product as iter_product
from typing import List, Tuple, Dict

# ============================================================
# Shared utilities: QAOA simulation helpers
# ============================================================

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def tensor_list(ops):
    """Tensor product of a list of operators."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def maxcut_cost(bitstring: str, edges: List[Tuple[int, int]],
                weights: Dict[Tuple[int, int], float] = None) -> float:
    """Compute MaxCut cost for a given bitstring."""
    cost = 0
    for i, j in edges:
        w = weights.get((i, j), 1.0) if weights else 1.0
        if bitstring[i] != bitstring[j]:
            cost += w
    return cost


def build_cost_hamiltonian(n_qubits: int, edges: List[Tuple[int, int]],
                           weights: Dict = None) -> np.ndarray:
    """Build MaxCut cost Hamiltonian: C = sum_{(i,j)} w_ij * (1 - Z_i Z_j) / 2."""
    dim = 2 ** n_qubits
    H_C = np.zeros((dim, dim), dtype=complex)

    for i, j in edges:
        w = weights.get((i, j), 1.0) if weights else 1.0
        ops = [I2] * n_qubits
        ops[i] = Z
        ops[j] = Z
        ZiZj = tensor_list(ops)
        H_C += w * (np.eye(dim) - ZiZj) / 2

    return H_C


def build_mixer_hamiltonian(n_qubits: int) -> np.ndarray:
    """Build mixer Hamiltonian: B = sum_i X_i."""
    dim = 2 ** n_qubits
    H_B = np.zeros((dim, dim), dtype=complex)

    for i in range(n_qubits):
        ops = [I2] * n_qubits
        ops[i] = X
        H_B += tensor_list(ops)

    return H_B


def qaoa_state(gamma: np.ndarray, beta: np.ndarray, H_C: np.ndarray,
               H_B: np.ndarray, n_qubits: int) -> np.ndarray:
    """Compute QAOA state for given parameters."""
    from scipy.linalg import expm

    dim = 2 ** n_qubits
    # Initial state: |+>^n
    state = np.ones(dim, dtype=complex) / np.sqrt(dim)

    p = len(gamma)
    for layer in range(p):
        # Cost unitary
        U_C = expm(-1j * gamma[layer] * H_C)
        state = U_C @ state

        # Mixer unitary
        U_B = expm(-1j * beta[layer] * H_B)
        state = U_B @ state

    return state


def qaoa_expectation(gamma, beta, H_C, H_B, n_qubits):
    """Compute <C> for QAOA state."""
    state = qaoa_state(gamma, beta, H_C, H_B, n_qubits)
    return np.real(state.conj() @ H_C @ state)


def brute_force_maxcut(n_qubits, edges, weights=None):
    """Find optimal MaxCut by brute force."""
    best_cost = 0
    best_configs = []
    for bits in iter_product('01', repeat=n_qubits):
        bs = ''.join(bits)
        cost = maxcut_cost(bs, edges, weights)
        if cost > best_cost:
            best_cost = cost
            best_configs = [bs]
        elif cost == best_cost:
            best_configs.append(bs)
    return best_cost, best_configs


# === Exercise 1: QAOA for a Square Graph ===
# Problem: Apply QAOA to 4-vertex square graph.

def exercise_1():
    """QAOA for 4-vertex square graph."""
    print("=" * 60)
    print("Exercise 1: QAOA for a Square Graph")
    print("=" * 60)

    n = 4
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

    H_C = build_cost_hamiltonian(n, edges)
    H_B = build_mixer_hamiltonian(n)

    # (a) Optimal MaxCut by brute force
    opt_cost, opt_configs = brute_force_maxcut(n, edges)
    print(f"\n(a) Optimal MaxCut = {opt_cost}")
    print(f"    Optimal assignments: {opt_configs}")

    # (b) QAOA with p=1
    def cost_p1(params):
        gamma = [params[0]]
        beta = [params[1]]
        return -qaoa_expectation(gamma, beta, H_C, H_B, n)

    best_val = float('inf')
    best_params = None
    for _ in range(50):
        x0 = np.random.uniform(0, 2 * np.pi, 2)
        res = minimize(cost_p1, x0, method='COBYLA')
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x

    approx_ratio_p1 = -best_val / opt_cost
    print(f"\n(b) QAOA p=1:")
    print(f"    Best <C> = {-best_val:.4f}")
    print(f"    Approximation ratio = {approx_ratio_p1:.4f}")

    # (c) Energy landscape for p=1
    print(f"\n(c) Energy landscape scan (gamma, beta):")
    gamma_vals = np.linspace(0, np.pi, 20)
    beta_vals = np.linspace(0, np.pi, 20)
    landscape = np.zeros((len(gamma_vals), len(beta_vals)))

    for i, g in enumerate(gamma_vals):
        for j, b in enumerate(beta_vals):
            landscape[i, j] = qaoa_expectation([g], [b], H_C, H_B, n)

    max_idx = np.unravel_index(np.argmax(landscape), landscape.shape)
    print(f"    Maximum <C> = {landscape[max_idx]:.4f} at "
          f"gamma={gamma_vals[max_idx[0]]:.3f}, beta={beta_vals[max_idx[1]]:.3f}")

    # (d) QAOA with p=2,3
    for p_depth in [2, 3]:
        def cost_fn(params, _p=p_depth):
            gamma = params[:_p]
            beta = params[_p:]
            return -qaoa_expectation(gamma, beta, H_C, H_B, n)

        best_val_p = float('inf')
        for _ in range(100):
            x0 = np.random.uniform(0, np.pi, 2 * p_depth)
            res = minimize(cost_fn, x0, method='COBYLA', options={'maxiter': 500})
            if res.fun < best_val_p:
                best_val_p = res.fun

        ratio = -best_val_p / opt_cost
        print(f"\n    QAOA p={p_depth}: <C> = {-best_val_p:.4f}, ratio = {ratio:.4f}")


# === Exercise 2: Weighted MaxCut ===
# Problem: Extend QAOA to handle weighted graphs.

def exercise_2():
    """Weighted MaxCut with QAOA."""
    print("\n" + "=" * 60)
    print("Exercise 2: Weighted MaxCut")
    print("=" * 60)

    n = 4
    # Edges: (i, j, weight)
    weighted_edges_raw = [(0, 1, 3), (1, 2, 1), (2, 3, 2), (0, 3, 4)]
    edges = [(i, j) for i, j, _ in weighted_edges_raw]
    weights = {(i, j): w for i, j, w in weighted_edges_raw}

    # (a) Build weighted cost Hamiltonian
    H_C = build_cost_hamiltonian(n, edges, weights)
    H_B = build_mixer_hamiltonian(n)

    # (b) Brute force
    opt_cost, opt_configs = brute_force_maxcut(n, edges, weights)
    print(f"\n  Graph: {weighted_edges_raw}")
    print(f"  Optimal weighted MaxCut = {opt_cost}")
    print(f"  Optimal assignments: {opt_configs}")

    # (c) QAOA optimization
    for p_depth in [1, 2]:
        def cost_fn(params, _p=p_depth):
            gamma = params[:_p]
            beta = params[_p:]
            return -qaoa_expectation(gamma, beta, H_C, H_B, n)

        best_val = float('inf')
        for _ in range(80):
            x0 = np.random.uniform(0, np.pi, 2 * p_depth)
            res = minimize(cost_fn, x0, method='COBYLA', options={'maxiter': 300})
            if res.fun < best_val:
                best_val = res.fun

        ratio = -best_val / opt_cost
        print(f"\n  QAOA p={p_depth}: <C> = {-best_val:.4f}, ratio = {ratio:.4f}")


# === Exercise 3: QAOA for Max Independent Set ===
# Problem: Formulate and solve Max Independent Set.

def exercise_3():
    """Max Independent Set using QAOA with penalty terms."""
    print("\n" + "=" * 60)
    print("Exercise 3: QAOA for Max Independent Set")
    print("=" * 60)

    # Use a small graph (Petersen is 10 vertices — too large for exact sim)
    # Use a 6-vertex graph instead
    n = 6
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3), (1, 4), (2, 5)]

    # (a) Formulate as Ising Hamiltonian
    # Objective: maximize sum_i x_i (number of selected vertices)
    # Constraint: for each edge (i,j), x_i * x_j = 0 (cannot both be in set)
    # Map x_i = (1 - Z_i) / 2 (x_i=1 when qubit i in |1>)
    #
    # H = -sum_i (1-Z_i)/2 + penalty * sum_{(i,j)} (1-Z_i)(1-Z_j)/4
    # penalty >> 1 to enforce constraints

    penalty = 5.0
    dim = 2 ** n

    H_obj = np.zeros((dim, dim), dtype=complex)
    # Objective: maximize number of selected vertices
    for i in range(n):
        ops = [I2] * n
        ops[i] = Z
        H_obj -= (np.eye(dim) - tensor_list(ops)) / 2  # -sum (1-Z_i)/2

    H_penalty = np.zeros((dim, dim), dtype=complex)
    # Penalty: penalize adjacent selected vertices
    for i, j in edges:
        ops_i = [I2] * n
        ops_i[i] = Z
        Zi = tensor_list(ops_i)

        ops_j = [I2] * n
        ops_j[j] = Z
        Zj = tensor_list(ops_j)

        # (1-Z_i)(1-Z_j)/4
        H_penalty += (np.eye(dim) - Zi - Zj + Zi @ Zj) / 4

    H_total = H_obj + penalty * H_penalty

    # (b) Brute force solution
    best_size = 0
    best_sets = []
    for bits in iter_product('01', repeat=n):
        bs = ''.join(bits)
        selected = [i for i, b in enumerate(bs) if b == '1']

        # Check independence
        is_independent = True
        for i, j in edges:
            if bs[i] == '1' and bs[j] == '1':
                is_independent = False
                break

        if is_independent and len(selected) > best_size:
            best_size = len(selected)
            best_sets = [selected]
        elif is_independent and len(selected) == best_size:
            best_sets.append(selected)

    print(f"\n(a) Graph: {n} vertices, {len(edges)} edges")
    print(f"    Penalty coefficient: {penalty}")

    print(f"\n(b) Brute force Max Independent Set: size = {best_size}")
    for s in best_sets[:5]:
        print(f"    Vertices: {s}")

    # (c) QAOA
    H_B = build_mixer_hamiltonian(n)

    for p_depth in [1, 2]:
        def cost_fn(params, _p=p_depth):
            gamma = params[:_p]
            beta = params[_p:]
            state = qaoa_state(np.array(gamma), np.array(beta), H_total, H_B, n)
            return np.real(state.conj() @ H_total @ state)

        best_val = float('inf')
        best_state = None
        for _ in range(100):
            x0 = np.random.uniform(0, np.pi, 2 * p_depth)
            res = minimize(cost_fn, x0, method='COBYLA', options={'maxiter': 500})
            if res.fun < best_val:
                best_val = res.fun
                gamma_opt = res.x[:p_depth]
                beta_opt = res.x[p_depth:]
                best_state = qaoa_state(gamma_opt, beta_opt, H_total, H_B, n)

        # Sample from optimal state
        probs = np.abs(best_state) ** 2
        top_states = np.argsort(probs)[::-1][:5]

        print(f"\n    QAOA p={p_depth}: E = {best_val:.4f}")
        print(f"    Top probability states:")
        for idx in top_states:
            bs = format(idx, f'0{n}b')
            selected = [i for i, b in enumerate(bs) if b == '1']
            is_valid = all(not (bs[i] == '1' and bs[j] == '1') for i, j in edges)
            print(f"      {bs} (vertices {selected}): p={probs[idx]:.4f} "
                  f"{'VALID' if is_valid else 'INVALID'}")


# === Exercise 4: Noise Effects on QAOA ===
# Problem: Simulate depolarizing noise effects on QAOA.

def exercise_4():
    """Depolarizing noise effects on QAOA."""
    print("\n" + "=" * 60)
    print("Exercise 4: Noise Effects on QAOA")
    print("=" * 60)

    from scipy.linalg import expm

    # Triangle graph
    n = 3
    edges = [(0, 1), (1, 2), (0, 2)]
    H_C = build_cost_hamiltonian(n, edges)
    H_B = build_mixer_hamiltonian(n)

    opt_cost, _ = brute_force_maxcut(n, edges)

    def noisy_qaoa_energy(gamma, beta, H_C, H_B, n_qubits, p_noise):
        """QAOA with depolarizing noise (density matrix simulation)."""
        dim = 2 ** n_qubits

        # Initial state: |+>^n as density matrix
        psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
        rho = np.outer(psi, psi.conj())

        p_depth = len(gamma)
        for layer in range(p_depth):
            # Cost unitary
            U_C = expm(-1j * gamma[layer] * H_C)
            rho = U_C @ rho @ U_C.conj().T

            # Depolarizing noise after cost layer
            if p_noise > 0:
                for q in range(n_qubits):
                    rho = (1 - p_noise) * rho + p_noise * np.eye(dim) / dim

            # Mixer unitary
            U_B = expm(-1j * beta[layer] * H_B)
            rho = U_B @ rho @ U_B.conj().T

            # Depolarizing noise after mixer layer
            if p_noise > 0:
                for q in range(n_qubits):
                    rho = (1 - p_noise) * rho + p_noise * np.eye(dim) / dim

        return np.real(np.trace(H_C @ rho))

    noise_levels = [0.0, 0.001, 0.01, 0.05, 0.1]

    print(f"\n  Triangle graph: optimal MaxCut = {opt_cost}")
    print(f"\n  {'p_noise':<10}", end="")
    for p_depth in [1, 2, 3]:
        print(f"{'p=' + str(p_depth) + ' ratio':<15}", end="")
    print()
    print("  " + "-" * 55)

    for p_noise in noise_levels:
        print(f"  {p_noise:<10.3f}", end="")

        for p_depth in [1, 2, 3]:
            def cost_fn(params, _p=p_depth, _pn=p_noise):
                gamma = params[:_p]
                beta = params[_p:]
                return -noisy_qaoa_energy(gamma, beta, H_C, H_B, n, _pn)

            best_val = float('inf')
            for _ in range(30):
                x0 = np.random.uniform(0, np.pi, 2 * p_depth)
                res = minimize(cost_fn, x0, method='COBYLA', options={'maxiter': 200})
                if res.fun < best_val:
                    best_val = res.fun

            ratio = -best_val / opt_cost
            print(f"{ratio:<15.4f}", end="")
        print()

    print(f"\n  Key observation:")
    print(f"    - Higher noise -> lower approximation ratio")
    print(f"    - At high noise, deeper circuits (larger p) can perform WORSE")
    print(f"      because they accumulate more noise")
    print(f"    - Optimal depth decreases as noise increases")


# === Exercise 5: Parameter Concentration ===
# Problem: Test whether optimal QAOA parameters concentrate across graph instances.

def exercise_5():
    """Parameter concentration across random 3-regular graphs."""
    print("\n" + "=" * 60)
    print("Exercise 5: Parameter Concentration")
    print("=" * 60)

    def random_3regular_graph(n_vertices):
        """Generate a random 3-regular graph (approximate)."""
        edges = set()
        degree = [0] * n_vertices

        # Try to make each vertex degree 3
        max_attempts = 1000
        attempts = 0
        while attempts < max_attempts:
            # Find vertices with degree < 3
            candidates = [v for v in range(n_vertices) if degree[v] < 3]
            if len(candidates) < 2:
                break

            i, j = np.random.choice(candidates, 2, replace=False)
            edge = (min(i, j), max(i, j))
            if edge not in edges and degree[i] < 3 and degree[j] < 3:
                edges.add(edge)
                degree[i] += 1
                degree[j] += 1
            attempts += 1

        return list(edges)

    np.random.seed(42)
    n = 8
    n_graphs = 15  # Reduced from 20 for speed
    p_depth = 1

    # (a,b) Generate graphs and optimize QAOA
    optimal_params = []
    optimal_ratios = []

    print(f"\n(a,b) Optimizing p=1 QAOA on {n_graphs} random 3-regular graphs (n={n}):")
    print(f"    {'Graph':<8} {'gamma*':<10} {'beta*':<10} {'Ratio'}")
    print("    " + "-" * 35)

    for g_idx in range(n_graphs):
        edges = random_3regular_graph(n)
        if not edges:
            continue

        H_C = build_cost_hamiltonian(n, edges)
        H_B = build_mixer_hamiltonian(n)
        opt_cost, _ = brute_force_maxcut(n, edges)

        if opt_cost == 0:
            continue

        def cost_fn(params):
            return -qaoa_expectation([params[0]], [params[1]], H_C, H_B, n)

        best_val = float('inf')
        best_p = None
        for _ in range(30):
            x0 = np.random.uniform(0, np.pi, 2)
            res = minimize(cost_fn, x0, method='COBYLA')
            if res.fun < best_val:
                best_val = res.fun
                best_p = res.x

        ratio = -best_val / opt_cost
        optimal_params.append(best_p)
        optimal_ratios.append(ratio)

        if g_idx < 8:  # Show first 8
            print(f"    {g_idx:<8} {best_p[0]:<10.4f} {best_p[1]:<10.4f} {ratio:.4f}")

    if len(optimal_params) < 2:
        print("    Not enough valid graphs generated.")
        return

    params_array = np.array(optimal_params)

    # (c) Parameter concentration
    gamma_mean = np.mean(params_array[:, 0])
    gamma_std = np.std(params_array[:, 0])
    beta_mean = np.mean(params_array[:, 1])
    beta_std = np.std(params_array[:, 1])

    print(f"\n(c) Parameter statistics:")
    print(f"    gamma*: mean={gamma_mean:.4f}, std={gamma_std:.4f}")
    print(f"    beta*:  mean={beta_mean:.4f}, std={beta_std:.4f}")
    print(f"    Concentration: {'Strong' if gamma_std < 0.3 and beta_std < 0.3 else 'Weak'}")

    # (d) Transfer: use average parameters on new graphs
    avg_gamma = gamma_mean
    avg_beta = beta_mean

    print(f"\n(d) Transfer test (average params on 5 new graphs):")
    print(f"    {'Graph':<8} {'Ratio(optimized)':<18} {'Ratio(transferred)':<18} {'Loss'}")
    print("    " + "-" * 55)

    for g_idx in range(5):
        edges = random_3regular_graph(n)
        if not edges:
            continue

        H_C = build_cost_hamiltonian(n, edges)
        H_B = build_mixer_hamiltonian(n)
        opt_cost, _ = brute_force_maxcut(n, edges)
        if opt_cost == 0:
            continue

        # Optimized
        def cost_fn(params):
            return -qaoa_expectation([params[0]], [params[1]], H_C, H_B, n)

        best_val = float('inf')
        for _ in range(30):
            x0 = np.random.uniform(0, np.pi, 2)
            res = minimize(cost_fn, x0, method='COBYLA')
            if res.fun < best_val:
                best_val = res.fun

        ratio_opt = -best_val / opt_cost

        # Transferred
        ratio_transfer = qaoa_expectation([avg_gamma], [avg_beta], H_C, H_B, n) / opt_cost

        loss = ratio_opt - ratio_transfer
        print(f"    {g_idx:<8} {ratio_opt:<18.4f} {ratio_transfer:<18.4f} {loss:.4f}")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
