"""
05_deutsch_jozsa.py — Deutsch and Deutsch-Jozsa Algorithm

Demonstrates:
  - The Deutsch algorithm (1-qubit version): distinguish constant vs balanced f
  - The Deutsch-Jozsa algorithm (n-qubit generalization)
  - Oracle construction for constant and balanced functions
  - Quantum parallelism: query ALL inputs in a single oracle call
  - Comparing quantum (1 query) vs classical (up to 2^{n-1}+1 queries)

All computations use pure NumPy.
"""

import numpy as np
from typing import Callable, List, Tuple
from itertools import product as iter_product

# ---------------------------------------------------------------------------
# Gates and helpers
# ---------------------------------------------------------------------------
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

KET_0 = np.array([1, 0], dtype=complex)
KET_1 = np.array([0, 1], dtype=complex)


def tensor(*matrices: np.ndarray) -> np.ndarray:
    """Compute the tensor product of multiple matrices."""
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result


def format_state(state: np.ndarray, n_qubits: int) -> str:
    terms = []
    for idx in range(len(state)):
        amp = state[idx]
        if np.abs(amp) < 1e-10:
            continue
        label = format(idx, f'0{n_qubits}b')
        if np.abs(amp.imag) < 1e-10:
            amp_str = f"{amp.real:+.6f}"
        else:
            amp_str = f"({amp.real:+.4f}{amp.imag:+.4f}j)"
        terms.append(f"{amp_str}|{label}⟩")
    return " ".join(terms) if terms else "0"


# ---------------------------------------------------------------------------
# Oracle construction
# ---------------------------------------------------------------------------

def build_oracle(f: Callable[[Tuple[int, ...]], int], n_input: int) -> np.ndarray:
    """Build a quantum oracle U_f for function f: {0,1}^n → {0,1}.

    The oracle acts on n+1 qubits as: U_f|x⟩|y⟩ = |x⟩|y ⊕ f(x)⟩

    Why: Quantum oracles are always unitary and reversible.  The XOR trick
    (y ⊕ f(x)) ensures reversibility — we can always undo the computation
    by applying U_f again (since XOR is its own inverse).
    """
    dim = 2 ** (n_input + 1)
    oracle = np.zeros((dim, dim), dtype=complex)

    # Why: We construct the oracle column by column.  For each computational
    # basis state |x⟩|y⟩, the oracle maps it to |x⟩|y ⊕ f(x)⟩.
    for x_int in range(2 ** n_input):
        x_bits = tuple(int(b) for b in format(x_int, f'0{n_input}b'))
        fx = f(x_bits)

        for y in range(2):
            # Input basis state index: x_int * 2 + y
            in_idx = x_int * 2 + y
            # Output: y XOR f(x)
            out_y = y ^ fx
            out_idx = x_int * 2 + out_y
            oracle[out_idx, in_idx] = 1.0

    return oracle


def build_phase_oracle(f: Callable[[Tuple[int, ...]], int], n_input: int) -> np.ndarray:
    """Build a phase oracle: U_f|x⟩ = (-1)^{f(x)}|x⟩.

    Why: When the target qubit is in the |−⟩ state, the standard oracle U_f
    acts as a phase oracle (phase kickback).  We can compute this directly
    as a diagonal matrix, which is more efficient for analysis and avoids
    the extra ancilla qubit.
    """
    dim = 2 ** n_input
    diag = np.zeros(dim, dtype=complex)
    for x_int in range(dim):
        x_bits = tuple(int(b) for b in format(x_int, f'0{n_input}b'))
        diag[x_int] = (-1) ** f(x_bits)
    return np.diag(diag)


# ---------------------------------------------------------------------------
# Deutsch Algorithm (1-qubit input)
# ---------------------------------------------------------------------------

def deutsch_algorithm(f: Callable[[Tuple[int, ...]], int]) -> str:
    """Run the Deutsch algorithm to determine if f:{0,1}→{0,1} is constant or balanced.

    Why: Deutsch's algorithm is the simplest quantum algorithm that demonstrates
    quantum advantage.  Classically, we need 2 queries to distinguish constant
    from balanced.  Quantumly, 1 query suffices — thanks to quantum parallelism
    and interference.

    Circuit:
        |0⟩ ─ H ─ ┤     ├ ─ H ─ Measure
                   │ U_f │
        |1⟩ ─ H ─ ┤     ├ ────────────
    """
    n_input = 1
    n_total = n_input + 1  # 1 input + 1 ancilla

    # Step 1: Initialize |01⟩ (input=|0⟩, ancilla=|1⟩)
    state = np.kron(KET_0, KET_1)

    # Step 2: Apply H to both qubits → |+⟩|−⟩
    H_all = tensor(H, H)
    state = H_all @ state

    # Step 3: Apply oracle
    oracle = build_oracle(f, n_input)
    state = oracle @ state

    # Why: After the oracle, the input qubit encodes f(0)⊕f(1) in its phase.
    # If f is constant (f(0)⊕f(1)=0), the Hadamard maps it back to |0⟩.
    # If f is balanced (f(0)⊕f(1)=1), the Hadamard maps it to |1⟩.

    # Step 4: Apply H to input qubit only
    H_I = tensor(H, I)
    state = H_I @ state

    # Step 5: Measure input qubit
    # P(input=0) = |⟨0|⊗I · state|² summed over ancilla
    prob_0 = np.abs(state[0]) ** 2 + np.abs(state[1]) ** 2  # |00⟩ + |01⟩
    prob_1 = np.abs(state[2]) ** 2 + np.abs(state[3]) ** 2  # |10⟩ + |11⟩

    # Why: Measurement outcome 0 → f is constant, 1 → f is balanced.
    # This is deterministic, not probabilistic — quantum interference ensures
    # 100% certainty in a single query.
    if prob_0 > 0.99:
        return "constant"
    elif prob_1 > 0.99:
        return "balanced"
    else:
        return f"ambiguous (P(0)={prob_0:.4f}, P(1)={prob_1:.4f})"


# ---------------------------------------------------------------------------
# Deutsch-Jozsa Algorithm (n-qubit input)
# ---------------------------------------------------------------------------

def deutsch_jozsa(f: Callable[[Tuple[int, ...]], int], n: int,
                  verbose: bool = True) -> str:
    """Run the Deutsch-Jozsa algorithm on f:{0,1}^n → {0,1}.

    Why: The Deutsch-Jozsa algorithm generalizes Deutsch to n qubits.
    A classical algorithm needs up to 2^{n-1}+1 queries (worst case).
    The quantum algorithm needs exactly 1 query — an exponential speedup.

    The algorithm exploits the fact that quantum superposition allows querying
    f on all 2^n inputs simultaneously (quantum parallelism), and interference
    concentrates the amplitude on |00...0⟩ iff f is constant.

    Circuit:
        |0⟩^n ─ H^⊗n ─ ┤      ├ ─ H^⊗n ─ Measure
                        │ U_f  │
        |1⟩   ─ H     ─ ┤      ├ ──────────────
    """
    n_total = n + 1

    # Step 1: Initialize |0...0⟩|1⟩
    state = np.zeros(2 ** n_total, dtype=complex)
    # |0...01⟩ → index 1 (ancilla is the last qubit = |1⟩)
    state[1] = 1.0

    # Step 2: Apply H to all qubits
    H_all = H
    for _ in range(n_total - 1):
        H_all = np.kron(H_all, H)
    state = H_all @ state

    if verbose:
        print(f"    After H^⊗{n_total}: superposition of all {2**n} inputs")

    # Step 3: Apply oracle
    oracle = build_oracle(f, n)
    state = oracle @ state

    # Step 4: Apply H to input qubits only (not ancilla)
    H_input = H
    for _ in range(n - 1):
        H_input = np.kron(H_input, H)
    H_input_I = np.kron(H_input, I)  # H^⊗n ⊗ I
    state = H_input_I @ state

    # Step 5: Measure input qubits
    # P(all zeros) = sum of |amplitude|² for states |00...0⟩|0⟩ and |00...0⟩|1⟩
    # Why: If f is constant, all amplitude concentrates on |00...0⟩ (input qubits).
    # If f is balanced, the |00...0⟩ amplitude is exactly zero due to destructive
    # interference — every +1 term cancels a -1 term perfectly.
    prob_all_zero = np.abs(state[0]) ** 2 + np.abs(state[1]) ** 2

    if verbose:
        print(f"    P(all-zero input) = {prob_all_zero:.6f}")

    if prob_all_zero > 0.99:
        return "constant"
    else:
        return "balanced"


# ---------------------------------------------------------------------------
# Example oracle functions
# ---------------------------------------------------------------------------

def f_constant_0(x: Tuple[int, ...]) -> int:
    """f(x) = 0 for all x (constant)."""
    return 0

def f_constant_1(x: Tuple[int, ...]) -> int:
    """f(x) = 1 for all x (constant)."""
    return 1

def f_balanced_parity(x: Tuple[int, ...]) -> int:
    """f(x) = x₁ ⊕ x₂ ⊕ ... ⊕ xₙ (parity — balanced)."""
    return sum(x) % 2

def f_balanced_first_bit(x: Tuple[int, ...]) -> int:
    """f(x) = x₁ (first bit — balanced for n≥1)."""
    return x[0]

def make_random_balanced(n: int, seed: int = 42) -> Callable:
    """Create a random balanced function on n bits.

    Why: A balanced function outputs 0 for exactly half the inputs and 1 for
    the other half.  We randomly partition {0,1}^n into two equal halves.
    """
    rng = np.random.RandomState(seed)
    N = 2 ** n
    outputs = np.array([0] * (N // 2) + [1] * (N // 2))
    rng.shuffle(outputs)
    table = dict(enumerate(outputs))

    def f(x: Tuple[int, ...]) -> int:
        idx = int(''.join(str(b) for b in x), 2)
        return int(table[idx])

    return f


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_deutsch():
    """Run the original Deutsch algorithm (1-bit input)."""
    print("=" * 60)
    print("DEMO 1: Deutsch Algorithm (1-bit input)")
    print("=" * 60)

    # Why: There are exactly 4 possible functions f:{0,1}→{0,1}.
    # Two are constant (f=0, f=1), two are balanced (identity, NOT).
    functions = [
        ("f(x) = 0  (constant)", f_constant_0),
        ("f(x) = 1  (constant)", f_constant_1),
        ("f(x) = x  (balanced)", lambda x: x[0]),
        ("f(x) = NOT x (balanced)", lambda x: 1 - x[0]),
    ]

    print(f"\n  {'Function':<30} {'f(0)':>5} {'f(1)':>5} {'Result':>12}")
    print(f"  {'─' * 55}")

    for desc, f in functions:
        result = deutsch_algorithm(f)
        print(f"  {desc:<30} {f((0,)):>5} {f((1,)):>5} {result:>12}")

    print(f"\n  All results correct using only 1 quantum query!")
    print(f"  Classical approach needs 2 queries (evaluate f(0) AND f(1)).")


def demo_deutsch_jozsa_small():
    """Run Deutsch-Jozsa on 2 and 3-bit functions."""
    print("\n" + "=" * 60)
    print("DEMO 2: Deutsch-Jozsa Algorithm (2-bit and 3-bit)")
    print("=" * 60)

    for n in [2, 3]:
        print(f"\n  --- n = {n} ({2**n} possible inputs) ---")
        print(f"  Classical worst case: {2**(n-1)+1} queries")
        print(f"  Quantum: 1 query\n")

        functions = [
            (f"f = 0 (constant)", f_constant_0),
            (f"f = 1 (constant)", f_constant_1),
            (f"f = parity (balanced)", f_balanced_parity),
            (f"f = first bit (balanced)", f_balanced_first_bit),
            (f"f = random balanced", make_random_balanced(n)),
        ]

        for desc, f in functions:
            print(f"  {desc}:")
            result = deutsch_jozsa(f, n, verbose=True)
            print(f"    Result: {result}\n")


def demo_phase_kickback():
    """Illustrate the phase kickback mechanism."""
    print("\n" + "=" * 60)
    print("DEMO 3: Phase Kickback — The Engine of Deutsch-Jozsa")
    print("=" * 60)

    # Why: Phase kickback is the key quantum phenomenon that makes the algorithm
    # work.  When the oracle acts on |x⟩|−⟩, it transforms:
    #     U_f|x⟩|−⟩ = (-1)^{f(x)} |x⟩|−⟩
    # The function value f(x) is "kicked back" into the phase of the input
    # register, leaving the ancilla unchanged.

    print(f"\n  Phase kickback mechanism:")
    print(f"  U_f|x⟩|−⟩ = |x⟩|f(x)⊕1⟩ - |x⟩|f(x)⟩  (by linearity)")
    print(f"            = (-1)^{{f(x)}} |x⟩|−⟩        (by algebra)")
    print(f"\n  Demonstration with phase oracle (diagonal form):")

    n = 3
    for desc, f in [("f=0 (constant)", f_constant_0),
                     ("f=parity (balanced)", f_balanced_parity)]:
        phase_oracle = build_phase_oracle(f, n)
        diag = np.diag(phase_oracle)
        print(f"\n  {desc}:")
        print(f"    Phase oracle diagonal: {np.real(diag).astype(int)}")
        # Why: For constant f, all phases are the same (all +1 or all -1).
        # For balanced f, exactly half are +1 and half are -1.
        n_plus = int(np.sum(diag == 1))
        n_minus = int(np.sum(diag == -1))
        print(f"    +1 phases: {n_plus}, −1 phases: {n_minus}")


def demo_quantum_vs_classical():
    """Compare query complexity of quantum vs classical approaches."""
    print("\n" + "=" * 60)
    print("DEMO 4: Query Complexity — Quantum vs Classical")
    print("=" * 60)

    print(f"\n  {'n (bits)':<12} {'Inputs 2^n':<12} {'Classical (worst)':<20} {'Quantum':<10}")
    print(f"  {'─' * 55}")

    for n in range(1, 11):
        N = 2 ** n
        classical = N // 2 + 1
        quantum = 1
        print(f"  {n:<12} {N:<12} {classical:<20} {quantum:<10}")

    print(f"\n  Classical: O(2^{{n-1}}) queries needed.")
    print(f"  Quantum:   O(1) — always exactly 1 query.")
    print(f"  Speedup:   EXPONENTIAL!")

    # Why: This exponential speedup is provably optimal.  No classical algorithm
    # can do better than 2^{n-1}+1 queries, while the quantum algorithm uses
    # exactly 1.  However, this is a "promise problem" — we're promised f is
    # either constant or balanced, which limits practical applications.


def demo_oracle_verification():
    """Verify oracle properties: unitarity and correct action."""
    print("\n" + "=" * 60)
    print("DEMO 5: Oracle Verification")
    print("=" * 60)

    n = 2
    f = f_balanced_parity
    oracle = build_oracle(f, n)

    # Check unitarity
    product = oracle.conj().T @ oracle
    is_unitary = np.allclose(product, np.eye(len(oracle)))
    print(f"\n  Oracle for f=parity on {n} bits:")
    print(f"    Size: {oracle.shape[0]}×{oracle.shape[1]}")
    print(f"    Unitary? {is_unitary}")

    # Why: Verify the oracle acts correctly on specific inputs.
    # U_f|x⟩|y⟩ should give |x⟩|y⊕f(x)⟩
    print(f"\n  Verification U_f|x⟩|y⟩ = |x⟩|y⊕f(x)⟩:")
    for x_int in range(2 ** n):
        x_bits = tuple(int(b) for b in format(x_int, f'0{n}b'))
        fx = f(x_bits)
        for y in range(2):
            # Construct |x⟩|y⟩
            in_state = np.zeros(2 ** (n + 1), dtype=complex)
            in_state[x_int * 2 + y] = 1.0
            out_state = oracle @ in_state

            # Find which basis state has amplitude 1
            out_idx = np.argmax(np.abs(out_state))
            out_x = out_idx // 2
            out_y = out_idx % 2
            expected_y = y ^ fx

            ok = (out_x == x_int and out_y == expected_y)
            x_str = format(x_int, f'0{n}b')
            print(f"    |{x_str}⟩|{y}⟩ → |{format(out_x, f'0{n}b')}⟩|{out_y}⟩  "
                  f"(f({x_str})={fx}, expected y⊕f={expected_y})  {'OK' if ok else 'FAIL'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Quantum Computing — 05: Deutsch-Jozsa Algorithm       ║")
    print("╚══════════════════════════════════════════════════════════╝")

    np.random.seed(2026)

    demo_deutsch()
    demo_deutsch_jozsa_small()
    demo_phase_kickback()
    demo_quantum_vs_classical()
    demo_oracle_verification()

    print("\n" + "=" * 60)
    print("All demonstrations complete.")
    print("=" * 60)
