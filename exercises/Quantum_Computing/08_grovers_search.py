"""
Exercises for Lesson 08: Grover's Search Algorithm
Topic: Quantum_Computing

Solutions to practice problems from the lesson.
"""

import numpy as np


def grover_oracle(n, targets):
    """Build Grover oracle: flip phase of target states."""
    N = 2 ** n
    O = np.eye(N, dtype=complex)
    for t in targets:
        O[t, t] = -1
    return O


def diffusion_operator(n):
    """Build diffusion operator D = 2|s><s| - I."""
    N = 2 ** n
    s = np.ones(N, dtype=complex) / np.sqrt(N)
    D = 2 * np.outer(s, s.conj()) - np.eye(N, dtype=complex)
    return D


def run_grover(n, targets, k_iterations=None):
    """Run Grover's algorithm with optional iteration count."""
    N = 2 ** n
    M = len(targets)
    theta = np.arcsin(np.sqrt(M / N))

    if k_iterations is None:
        k_iterations = max(1, int(np.round(np.pi / (4 * theta) - 0.5)))

    O = grover_oracle(n, targets)
    D = diffusion_operator(n)
    G = D @ O

    state = np.ones(N, dtype=complex) / np.sqrt(N)
    for _ in range(k_iterations):
        state = G @ state

    probs = np.abs(state) ** 2
    p_success = sum(probs[t] for t in targets)
    return probs, p_success, k_iterations


# === Exercise 1: Basic Grover ===
# Problem: n=4 qubits, target w=7.

def exercise_1():
    """Basic Grover's algorithm for n=4."""
    n = 4
    N = 2 ** n
    target = 7
    theta = np.arcsin(1 / np.sqrt(N))
    k_opt = int(np.round(np.pi / (4 * theta) - 0.5))

    # (a) Iterations needed and success probability
    probs, p_success, k = run_grover(n, [target])
    print(f"(a) n={n}, N={N}, target w={target}")
    print(f"    Optimal iterations: k = {k_opt}")
    print(f"    P(success) = {p_success:.6f}")
    print(f"    Found index: {np.argmax(probs)} (correct? {np.argmax(probs) == target})")

    # (b) Success probability vs iteration count
    print(f"\n(b) P(target) vs iterations:")
    print(f"    {'k':>4} {'P(target)':>12} {'P(theory)':>12} {'Status':>15}")
    print(f"    {'-'*47}")
    for k_val in range(11):
        probs_k, p_k, _ = run_grover(n, [target], k_val)
        p_theory = np.sin((2 * k_val + 1) * theta) ** 2
        status = ""
        if k_val == k_opt:
            status = "<- optimal"
        elif k_val > k_opt and p_k < 0.5:
            status = "(overshooting!)"
        print(f"    {k_val:>4} {p_k:>12.6f} {p_theory:>12.6f} {status:>15}")

    # (c) Verify sinusoidal oscillation
    print(f"\n(c) Sinusoidal verification:")
    print(f"    theta = arcsin(1/sqrt({N})) = {theta:.6f} rad")
    print(f"    P(k) = sin^2((2k+1)*theta)")
    print(f"    All values match theoretical sin^2 formula above.")


# === Exercise 2: Oracle Construction ===
# Problem: Build oracles as circuits for n=3.

def exercise_2():
    """Oracle construction for different target conditions."""
    n = 3
    N = 8

    # (a) Single target w=5 (binary: 101)
    print("(a) Oracle for target w=5 (|101>):")
    O_a = grover_oracle(n, [5])
    print(f"    Oracle diagonal: {np.round(np.diag(O_a).real, 0).astype(int)}")
    probs_a, p_a, k_a = run_grover(n, [5])
    print(f"    Grover: k={k_a}, P(5)={p_a:.4f}")

    # As circuit: multi-controlled Z = flip phase when all qubits match target
    # For |101>: X(q1), then controlled-controlled-Z, then X(q1) to undo
    print(f"    Circuit: X(q1) -> CCZ -> X(q1)")
    print(f"    (Apply X to qubits where target bit is 0, then multi-controlled Z)")

    # (b) Two targets w in {3, 5}
    print(f"\n(b) Oracle for targets {{3, 5}} (|011>, |101>):")
    O_b = grover_oracle(n, [3, 5])
    print(f"    Oracle diagonal: {np.round(np.diag(O_b).real, 0).astype(int)}")
    probs_b, p_b, k_b = run_grover(n, [3, 5])
    print(f"    Grover: k={k_b}, P(correct)={p_b:.4f}")
    print(f"    Fewer iterations needed with M=2 solutions vs M=1")

    # (c) Target condition: exactly two 1-bits
    targets_c = [x for x in range(N) if bin(x).count('1') == 2]
    print(f"\n(c) Oracle for states with exactly two 1-bits:")
    print(f"    Targets: {[format(t, '03b') for t in targets_c]} = {targets_c}")
    O_c = grover_oracle(n, targets_c)
    probs_c, p_c, k_c = run_grover(n, targets_c)
    print(f"    M={len(targets_c)} solutions, k={k_c}, P(correct)={p_c:.4f}")

    # Multi-controlled Z for condition: popcount(x) == 2
    print(f"    Circuit: Use ancilla to compute popcount, then")
    print(f"    flip phase if ancilla encodes 2, then uncompute ancilla.")


# === Exercise 3: Amplitude Amplification ===
# Problem: Generalized Grover with non-uniform initial state.

def exercise_3():
    """Amplitude amplification with non-uniform initial state."""
    # (a) 1-qubit example
    print("(a) 1-qubit amplitude amplification:")
    psi = np.array([1 / np.sqrt(3), np.sqrt(2 / 3)], dtype=complex)
    target = 0  # Amplify |0>

    # Oracle: flip target
    O_1q = np.diag([(-1) if i == target else 1 for i in range(2)]).astype(complex)
    # Diffusion about |psi>: 2|psi><psi| - I
    D_psi = 2 * np.outer(psi, psi.conj()) - np.eye(2, dtype=complex)

    state = psi.copy()
    print(f"    Initial: {np.round(state, 4)}, P(|0>)={abs(state[0])**2:.4f}")

    for k in range(1, 5):
        state = D_psi @ O_1q @ state
        print(f"    After {k} iteration(s): P(|0>)={abs(state[0])**2:.4f}")

    # (b) 2-qubit with non-uniform initial state
    print(f"\n(b) 2-qubit amplitude amplification with non-uniform start:")
    n = 2
    N = 4
    target_idx = 2  # Looking for |10>

    # Non-uniform initial state: more amplitude on target
    psi_2q = np.array([0.3, 0.4, 0.7, 0.5], dtype=complex)
    psi_2q /= np.linalg.norm(psi_2q)

    print(f"    Initial amplitudes: {np.round(psi_2q, 4)}")
    print(f"    Initial P(target={target_idx}) = {abs(psi_2q[target_idx])**2:.4f}")

    O_2q = np.eye(N, dtype=complex)
    O_2q[target_idx, target_idx] = -1
    D_2q = 2 * np.outer(psi_2q, psi_2q.conj()) - np.eye(N, dtype=complex)

    state = psi_2q.copy()
    for k in range(1, 6):
        state = D_2q @ O_2q @ state
        p_target = abs(state[target_idx]) ** 2
        print(f"    Iteration {k}: P(target)={p_target:.4f}")

    # Compare: with higher initial amplitude, fewer iterations needed
    theta_nonuniform = np.arcsin(abs(psi_2q[target_idx]))
    theta_uniform = np.arcsin(1 / np.sqrt(N))
    k_nonuniform = max(1, int(np.round(np.pi / (4 * theta_nonuniform) - 0.5)))
    k_uniform = max(1, int(np.round(np.pi / (4 * theta_uniform) - 0.5)))

    print(f"\n    theta (non-uniform) = {theta_nonuniform:.4f}")
    print(f"    theta (uniform) = {theta_uniform:.4f}")
    print(f"    k_opt (non-uniform) = {k_nonuniform}")
    print(f"    k_opt (uniform) = {k_uniform}")
    print(f"    Higher initial amplitude -> fewer iterations needed!")


# === Exercise 4: Application to SAT ===
# Problem: (x0 OR x1) AND (NOT x1 OR x2) AND (x0 OR NOT x2)

def exercise_4():
    """Grover's algorithm for boolean satisfiability."""
    n = 3
    N = 8

    def sat_formula(x):
        """Evaluate (x0 OR x1) AND (NOT x1 OR x2) AND (x0 OR NOT x2)."""
        x0 = (x >> 0) & 1
        x1 = (x >> 1) & 1
        x2 = (x >> 2) & 1
        clause1 = x0 | x1
        clause2 = (1 - x1) | x2
        clause3 = x0 | (1 - x2)
        return clause1 & clause2 & clause3

    # (a) Enumerate all inputs
    print("(a) Truth table for (x0 OR x1) AND (NOT x1 OR x2) AND (x0 OR NOT x2):")
    print(f"    {'x2 x1 x0':>10} {'f(x)':>6}")
    print(f"    {'-'*20}")
    satisfying = []
    for x in range(N):
        result = sat_formula(x)
        x0, x1, x2 = (x >> 0) & 1, (x >> 1) & 1, (x >> 2) & 1
        print(f"    {x2:>2} {x1:>2} {x0:>2}     {result:>4}")
        if result:
            satisfying.append(x)

    M = len(satisfying)
    print(f"\n    Satisfying assignments: {satisfying}")
    print(f"    = {[format(s, '03b') for s in satisfying]}")
    print(f"    M = {M} solutions")

    # (b) Construct oracle
    oracle = grover_oracle(n, satisfying)
    print(f"\n(b) Oracle marks {M} satisfying states")

    # (c) Run Grover
    probs, p_success, k = run_grover(n, satisfying)
    print(f"\n(c) Grover's search:")
    print(f"    k = {k} iterations")
    print(f"    P(satisfying) = {p_success:.4f}")
    top = np.argsort(probs)[::-1][:M + 2]
    for idx in top:
        marker = " *" if idx in satisfying else ""
        print(f"    |{format(idx, '03b')}> P = {probs[idx]:.4f}{marker}")

    # (d) Optimal iterations
    theta = np.arcsin(np.sqrt(M / N))
    k_opt = max(1, int(np.round(np.pi / (4 * theta) - 0.5)))
    print(f"\n(d) With M={M} solutions:")
    print(f"    theta = arcsin(sqrt({M}/{N})) = {theta:.4f}")
    print(f"    k_opt = {k_opt}")
    print(f"    Fewer iterations than M=1 case because more solutions")


# === Exercise 5: Grover with Noise ===
# Problem: Add random phase errors and study degradation.

def exercise_5():
    """Grover's algorithm with simulated noise."""
    rng = np.random.default_rng(42)
    n = 4
    N = 2 ** n
    target = 7
    n_trials = 500

    theta = np.arcsin(1 / np.sqrt(N))
    k_opt = max(1, int(np.round(np.pi / (4 * theta) - 0.5)))

    O = grover_oracle(n, [target])
    D = diffusion_operator(n)
    G = D @ O

    print(f"Grover with noise: n={n}, target={target}, k_opt={k_opt}")
    print(f"{'sigma':>8} {'P(correct)':>12} {'Std':>8} {'vs Noiseless':>14} {'vs Random':>12}")
    print("-" * 58)

    p_random = 1 / N
    noiseless_probs, p_noiseless, _ = run_grover(n, [target], k_opt)

    for sigma in [0, 0.01, 0.05, 0.1, 0.5]:
        successes = []
        for _ in range(n_trials):
            state = np.ones(N, dtype=complex) / np.sqrt(N)
            for k in range(k_opt):
                state = G @ state
                if sigma > 0:
                    # Add random phase errors
                    phases = np.exp(1j * rng.normal(0, sigma, N))
                    state = phases * state
                    # Re-normalize (in practice noise also reduces norm)
                    state /= np.linalg.norm(state)

            p_target = abs(state[target]) ** 2
            successes.append(p_target)

        mean_p = np.mean(successes)
        std_p = np.std(successes)
        ratio = mean_p / p_noiseless if p_noiseless > 0 else 0
        advantage = mean_p / p_random

        print(f"{sigma:>8.3f} {mean_p:>12.4f} {std_p:>8.4f} "
              f"{ratio:>14.4f} {advantage:>12.2f}x")

    # (c) Find threshold where quantum advantage disappears
    print(f"\n    Noiseless P(correct) = {p_noiseless:.4f}")
    print(f"    Random guess P(correct) = {p_random:.4f}")
    print(f"    Quantum advantage disappears when P(correct) ~ 1/N = {p_random:.4f}")
    print(f"    From the table, this happens around sigma ~ 0.5")
    print(f"    At sigma = 0.1, significant degradation but still advantageous")
    print(f"    At sigma = 0.01, nearly noiseless performance")


if __name__ == "__main__":
    print("=== Exercise 1: Basic Grover ===")
    exercise_1()
    print("\n=== Exercise 2: Oracle Construction ===")
    exercise_2()
    print("\n=== Exercise 3: Amplitude Amplification ===")
    exercise_3()
    print("\n=== Exercise 4: Application to SAT ===")
    exercise_4()
    print("\n=== Exercise 5: Grover with Noise ===")
    exercise_5()
    print("\nAll exercises completed!")
