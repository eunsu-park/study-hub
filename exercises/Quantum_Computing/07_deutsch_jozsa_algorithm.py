"""
Exercises for Lesson 07: Deutsch-Jozsa Algorithm
Topic: Quantum_Computing

Solutions to practice problems from the lesson.
"""

import numpy as np
import time


H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)


def deutsch_jozsa(f, n):
    """Full Deutsch-Jozsa algorithm simulation (phase oracle version)."""
    dim = 2 ** n
    state = np.ones(dim, dtype=complex) / np.sqrt(dim)
    for x in range(dim):
        state[x] *= (-1) ** f(x)
    H_n = np.array([[1]], dtype=complex)
    for _ in range(n):
        H_n = np.kron(H_n, H)
    state = H_n @ state
    p_zero = abs(state[0]) ** 2
    return p_zero, 'constant' if p_zero > 0.5 else 'balanced', state


# === Exercise 1: Oracle Construction ===
# Problem: Build phase oracles for 3 functions on n=2 qubits.

def exercise_1():
    """Oracle construction and Deutsch-Jozsa verification."""
    n = 2
    dim = 4

    functions = {
        "(a) f(x) = 1 for all x (constant)": lambda x: 1,
        "(b) f(x) = x1 (MSB, balanced)": lambda x: (x >> 1) & 1,
        "(c) f(x) = x0 XOR x1 (balanced)": lambda x: ((x & 1) ^ ((x >> 1) & 1)),
    }

    for name, f in functions.items():
        print(f"{name}:")
        # Build phase oracle matrix
        oracle = np.diag([(-1) ** f(x) for x in range(dim)]).astype(complex)
        print(f"    Phase oracle diagonal: {[(-1)**f(x) for x in range(dim)]}")

        # Truth table
        print(f"    Truth table: {[f(x) for x in range(dim)]}")

        p_zero, result, _ = deutsch_jozsa(f, n)
        is_const = all(f(x) == f(0) for x in range(dim))
        is_bal = sum(f(x) for x in range(dim)) == dim // 2
        expected = "constant" if is_const else ("balanced" if is_bal else "neither")

        print(f"    P(00) = {p_zero:.6f}")
        print(f"    DJ result: {result} (expected: {expected})")
        print(f"    Correct? {result == expected}")
        print()


# === Exercise 2: Scaling Analysis ===
# Problem: Run DJ for n=1..15, verify correctness, time simulation.

def exercise_2():
    """Scaling analysis of Deutsch-Jozsa simulation."""
    print(f"{'n':>4} {'dim':>8} {'P(0^n) const':>14} {'P(0^n) bal':>14} "
          f"{'Time (ms)':>12} {'Correct':>8}")
    print("-" * 65)

    for n in range(1, 16):
        dim = 2 ** n
        f_const = lambda x: 0
        f_balanced = lambda x, n=n: (x >> (n - 1)) & 1  # MSB

        start = time.time()
        p_const, r_const, _ = deutsch_jozsa(f_const, n)
        p_bal, r_bal, _ = deutsch_jozsa(f_balanced, n)
        elapsed = (time.time() - start) * 1000

        correct = (r_const == 'constant' and r_bal == 'balanced')
        print(f"{n:>4} {dim:>8} {p_const:>14.6f} {p_bal:>14.6f} "
              f"{elapsed:>12.2f} {'YES' if correct else 'NO':>8}")

    print(f"\nSimulation becomes slow around n=15-20 due to exponential 2^n state size.")
    print(f"A real quantum computer would solve this in O(1) query regardless of n!")


# === Exercise 3: Non-Promise Functions ===
# Problem: What if f is neither constant nor balanced?

def exercise_3():
    """Deutsch-Jozsa on non-promise functions."""
    n = 4
    dim = 2 ** n

    # f(x) = 1 if x < dim/3, else 0
    threshold = dim // 3
    f_non_promise = lambda x: 1 if x < threshold else 0

    n_ones = sum(f_non_promise(x) for x in range(dim))
    fraction = n_ones / dim
    print(f"Non-promise function: f(x) = 1 if x < {threshold}, else 0")
    print(f"  n = {n}, dim = {dim}")
    print(f"  Number of 1s: {n_ones}/{dim} = {fraction:.4f}")
    print(f"  This is neither constant (need all same) nor balanced (need exactly half)")

    # (a) Analytical P(0^n)
    # P(0^n) = |1/2^n * sum_x (-1)^f(x)|^2
    phase_sum = sum((-1) ** f_non_promise(x) for x in range(dim))
    p_analytical = abs(phase_sum / dim) ** 2
    print(f"\n(a) Analytical P(0^n):")
    print(f"    Sum of phases: {phase_sum}")
    print(f"    P(0^n) = |{phase_sum}/{dim}|^2 = {p_analytical:.6f}")

    # (b) Verify with simulation
    p_sim, result, state = deutsch_jozsa(f_non_promise, n)
    print(f"\n(b) Simulation: P(0^n) = {p_sim:.6f}")
    print(f"    Match? {np.isclose(p_analytical, p_sim)}")
    print(f"    DJ output: {result}")

    # (c) Can we extract useful information?
    print(f"\n(c) Information from output:")
    print(f"    P(0^n) = {p_sim:.6f}")
    print(f"    If P(0^n) = 1: function is constant")
    print(f"    If P(0^n) = 0: function is balanced")
    print(f"    If 0 < P(0^n) < 1: function is NEITHER constant nor balanced")
    print(f"    The value of P(0^n) relates to the 'imbalance' of f:")
    print(f"    P(0^n) = |fraction(f=0) - fraction(f=1)|^2")
    imbalance = abs(1 - 2 * fraction)
    print(f"    Imbalance: {imbalance:.4f}, P(0^n) = {imbalance**2:.6f}")
    print(f"    So yes, P(0^n) measures how far f is from being balanced.")


# === Exercise 4: Full Circuit with Ancilla ===
# Problem: Implement Deutsch-Jozsa with ancilla for n=3.

def exercise_4():
    """Full Deutsch-Jozsa circuit with ancilla qubit for n=3."""
    n = 3
    total_qubits = n + 1
    dim = 2 ** total_qubits

    def build_oracle_with_ancilla(f, n):
        """Build U_f: |x>|y> -> |x>|y XOR f(x)>."""
        dim_n = 2 ** n
        dim_total = 2 ** (n + 1)
        U_f = np.zeros((dim_total, dim_total), dtype=complex)
        for x in range(dim_n):
            for y in range(2):
                in_idx = x * 2 + y
                out_idx = x * 2 + (y ^ f(x))
                U_f[out_idx, in_idx] = 1
        return U_f

    def dj_full_circuit(f, n):
        """Full circuit: |0^n>|1>, H^(n+1), U_f, H^n, measure input."""
        dim_n = 2 ** n
        dim_total = 2 ** (n + 1)

        # Initial state: |0...0>|0>
        state = np.zeros(dim_total, dtype=complex)
        state[0] = 1.0

        # X on ancilla: |0...0>|1>
        X_ancilla = np.eye(dim_total, dtype=complex)
        for i in range(dim_total):
            if i % 2 == 0:
                X_ancilla[i, i] = 0
                X_ancilla[i + 1, i] = 1
                X_ancilla[i, i + 1] = 1
                X_ancilla[i + 1, i + 1] = 0
        # Simpler: build X on ancilla
        X_anc = np.kron(np.eye(dim_n, dtype=complex), X)
        state = X_anc @ state

        # H on all qubits
        H_all = H
        for _ in range(n):
            H_all = np.kron(H, H_all)
        state = H_all @ state

        # Oracle
        U_f = build_oracle_with_ancilla(f, n)
        state = U_f @ state

        # H on input qubits only
        H_n = np.eye(2, dtype=complex)
        for _ in range(n - 1):
            H_n = np.kron(H, H_n)
        H_input = np.kron(H_n, I2)  # H^n on input, I on ancilla
        state = H_input @ state

        # Measure input register: P(0^n)
        p_zero = 0
        for x in range(dim_n):
            for y in range(2):
                idx = x * 2 + y
                if x == 0:
                    p_zero += abs(state[idx]) ** 2

        return p_zero, 'constant' if p_zero > 0.5 else 'balanced'

    # Test functions
    functions = {
        "f(x) = 0 (constant)": lambda x: 0,
        "f(x) = 1 (constant)": lambda x: 1,
        "f(x) = MSB (balanced)": lambda x: (x >> 2) & 1,
        "f(x) = parity (balanced)": lambda x: bin(x).count('1') % 2,
    }

    print(f"Full Deutsch-Jozsa circuit with ancilla (n={n}):")
    print(f"{'Function':>30} {'P(0^n)':>10} {'Result':>12} {'Correct':>8}")
    print("-" * 65)

    for name, f in functions.items():
        p, result = dj_full_circuit(f, n)
        is_const = all(f(x) == f(0) for x in range(2 ** n))
        expected = "constant" if is_const else "balanced"
        correct = result == expected
        print(f"{name:>30} {p:>10.6f} {result:>12} {'YES' if correct else 'NO':>8}")


# === Exercise 5: Bernstein-Vazirani Extension ===
# Problem: BV algorithm for n=8, handle f(x) = s.x XOR b.

def exercise_5():
    """Bernstein-Vazirani algorithm and extensions."""

    def bernstein_vazirani(f, n):
        dim = 2 ** n
        state = np.ones(dim, dtype=complex) / np.sqrt(dim)
        for x in range(dim):
            state[x] *= (-1) ** f(x)
        H_n = np.array([[1]], dtype=complex)
        for _ in range(n):
            H_n = np.kron(H_n, H)
        state = H_n @ state
        probs = np.abs(state) ** 2
        s = np.argmax(probs)
        return format(s, f'0{n}b'), probs[s]

    # (a) BV for n=8
    n = 8
    print(f"(a) Bernstein-Vazirani for n={n}:")
    for secret in [0b10110011, 0b11111111, 0b00000001, 0b10101010]:
        f = lambda x, s=secret: bin(x & s).count('1') % 2
        found, prob = bernstein_vazirani(f, n)
        expected = format(secret, f'0{n}b')
        print(f"    s={expected} -> found={found}, prob={prob:.6f}, "
              f"correct={found == expected}")

    # (b) Handle f(x) = s.x XOR b
    print(f"\n(b) BV with offset bit b:")
    secret = 0b10110
    n = 5
    for b in [0, 1]:
        f = lambda x, s=secret, b=b: (bin(x & s).count('1') + b) % 2

        found, prob = bernstein_vazirani(f, n)
        expected = format(secret, f'0{n}b')
        print(f"    s={expected}, b={b}: found={found}, prob={prob:.6f}")
        print(f"      The output is {'the same' if found == expected else 'different'}")
        print(f"      regardless of b (b only adds global phase -1 to all amplitudes)")

    print(f"\n    Explanation: f(x) = s.x XOR b means (-1)^f(x) = (-1)^b * (-1)^(s.x)")
    print(f"    The (-1)^b is a global phase that doesn't affect measurement.")
    print(f"    So the algorithm finds s correctly regardless of b.")
    print(f"    However, b itself is NOT recoverable from the measurement.")
    print(f"    To find b, evaluate f(0): f(0) = s.0 XOR b = b.")

    # (c) Query count comparison
    print(f"\n(c) Query count comparison:")
    print(f"    {'n':>4} {'Classical':>12} {'Quantum':>10} {'Speedup':>10}")
    print(f"    {'-'*40}")
    for n_val in [1, 5, 10, 50, 100, 1000]:
        print(f"    {n_val:>4} {n_val:>12} {1:>10} {n_val:>10}x")
    print(f"\n    The quantum advantage is exactly n-fold for BV.")
    print(f"    This matters LESS than DJ because:")
    print(f"    1. The speedup is polynomial (linear), not exponential")
    print(f"    2. Classical BV needs only n queries (already efficient)")
    print(f"    3. For practical n, the quantum overhead (error correction)")
    print(f"       likely exceeds the n-fold query savings")


if __name__ == "__main__":
    print("=== Exercise 1: Oracle Construction ===")
    exercise_1()
    print("\n=== Exercise 2: Scaling Analysis ===")
    exercise_2()
    print("\n=== Exercise 3: Non-Promise Functions ===")
    exercise_3()
    print("\n=== Exercise 4: Full Circuit with Ancilla ===")
    exercise_4()
    print("\n=== Exercise 5: Bernstein-Vazirani Extension ===")
    exercise_5()
    print("\nAll exercises completed!")
