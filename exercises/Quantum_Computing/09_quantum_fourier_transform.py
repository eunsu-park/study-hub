"""
Exercises for Lesson 09: Quantum Fourier Transform
Topic: Quantum_Computing

Solutions to practice problems from the lesson.
"""

import numpy as np


def qft_matrix(n):
    """Build the full QFT matrix for n qubits."""
    N = 2 ** n
    omega = np.exp(2j * np.pi / N)
    return np.array([[omega ** (j * k) for j in range(N)] for k in range(N)]) / np.sqrt(N)


def inverse_qft_matrix(n):
    """Build the inverse QFT matrix."""
    return qft_matrix(n).conj().T


# === Exercise 1: QFT by Hand (2-qubit) ===
# Problem: Compute QFT|3> for 2-qubit QFT.

def exercise_1():
    """2-qubit QFT by hand and product representation."""
    n = 2
    N = 4

    # (a) Direct matrix application
    F4 = qft_matrix(n)
    ket_3 = np.array([0, 0, 0, 1], dtype=complex)
    result_matrix = F4 @ ket_3

    print(f"(a) QFT|3> via matrix F4:")
    print(f"    F4 = ")
    for row in F4:
        print(f"      {np.round(row, 4)}")
    print(f"    QFT|3> = {np.round(result_matrix, 4)}")

    # (b) Product representation
    # |3> = |11>, so j1=1, j2=1
    # QFT|j1 j2> = (1/2) (|0> + e^{2pi i * 0.j2}|1>) tensor (|0> + e^{2pi i * 0.j1 j2}|1>)
    # j1=1, j2=1:
    # 0.j2 = 0.1 (binary) = 1/2
    # 0.j1 j2 = 0.11 (binary) = 3/4
    j1, j2 = 1, 1
    frac_j2 = j2 / 2  # 0.j2 in decimal
    frac_j1j2 = j1 / 2 + j2 / 4  # 0.j1 j2 in decimal

    qubit_0 = np.array([1, np.exp(2j * np.pi * frac_j2)], dtype=complex) / np.sqrt(2)
    qubit_1 = np.array([1, np.exp(2j * np.pi * frac_j1j2)], dtype=complex) / np.sqrt(2)

    # Note: QFT output has reversed qubit order
    result_product = np.kron(qubit_0, qubit_1)

    print(f"\n(b) Product representation:")
    print(f"    0.j2 = 0.{j2} = {frac_j2}")
    print(f"    0.j1j2 = 0.{j1}{j2} = {frac_j1j2}")
    print(f"    qubit 0: (|0> + e^{{2pi i * {frac_j2}}}|1>)/sqrt(2) = {np.round(qubit_0, 4)}")
    print(f"    qubit 1: (|0> + e^{{2pi i * {frac_j1j2}}}|1>)/sqrt(2) = {np.round(qubit_1, 4)}")
    print(f"    Product: {np.round(result_product, 4)}")

    # (c) Verify agreement
    print(f"\n(c) Agreement? {np.allclose(result_matrix, result_product)}")
    print(f"    Matrix result:  {np.round(result_matrix, 4)}")
    print(f"    Product result: {np.round(result_product, 4)}")


# === Exercise 2: Period Finding with QFT ===
# Problem: State with period r=3, N=16.

def exercise_2():
    """Period finding using QFT."""
    n = 4
    N = 16
    period = 3

    # (a) Build periodic state
    state = np.zeros(N, dtype=complex)
    for i in range(N):
        if i % period == 0:
            state[i] = 1.0
    state /= np.linalg.norm(state)

    print(f"(a) Periodic state with r={period}, N={N}:")
    nonzero = [i for i in range(N) if abs(state[i]) > 1e-10]
    print(f"    Nonzero at: {nonzero}")

    # Apply QFT
    F = qft_matrix(n)
    result = F @ state
    probs = np.abs(result) ** 2

    print(f"\n(b) QFT result probabilities:")
    print(f"    {'k':>4} {'P(k)':>10} {'k/N':>8} {'N/r':>6}")
    print(f"    {'-'*32}")
    for k in range(N):
        if probs[k] > 0.01:
            is_multiple = k % (N / period) < 0.01 or (N - k % (N / period)) < 0.01
            print(f"    {k:>4} {probs[k]:>10.4f} {k/N:>8.4f} {N/period:>6.2f}")

    # (c) Extract period
    print(f"\n(c) Extracting period r={period}:")
    peaks = [k for k in range(N) if probs[k] > 0.01]
    print(f"    Peaks at k = {peaks}")
    if len(peaks) > 1:
        spacings = [peaks[i + 1] - peaks[i] for i in range(len(peaks) - 1)]
        gcd = spacings[0]
        for s in spacings[1:]:
            gcd = np.gcd(gcd, s)
        estimated_period = N // gcd
        print(f"    Spacing between peaks: {spacings}")
        print(f"    GCD of spacings: {gcd}")
        print(f"    Estimated period: N/GCD = {N}/{gcd} = {estimated_period}")

    # (d) Why r=3 is subtle
    print(f"\n(d) Why r=3 is subtle:")
    print(f"    N/r = {N}/{period} = {N/period:.4f} is NOT an integer!")
    print(f"    Peaks are spread around the nearest integers to multiples of N/r.")
    print(f"    For r=2,4,8: N/r is integer -> peaks are exactly at multiples")
    print(f"    For r=3: N/r=5.333 -> peaks leak to neighbors (spectral leakage)")
    print(f"    Continued fractions or multiple measurements needed to extract r=3")


# === Exercise 3: QPE Precision ===
# Problem: QPE for Rz gate with various counting qubits.

def exercise_3():
    """QPE precision analysis."""

    def run_qpe(theta_true, n_counting):
        N = 2 ** n_counting
        counting_state = np.array([np.exp(2j * np.pi * k * theta_true)
                                   for k in range(N)]) / np.sqrt(N)
        F_inv = inverse_qft_matrix(n_counting)
        result = F_inv @ counting_state
        probs = np.abs(result) ** 2
        best_k = np.argmax(probs)
        return best_k / N, probs[best_k], probs

    # (a) theta = 13/16 = 0.8125 (exactly representable in 4 bits)
    theta_exact = 13 / 16
    print(f"(a) QPE for theta = 13/16 = {theta_exact}:")
    print(f"    Minimum counting qubits for exact: 4 (since 13/16 = 0.1101 in binary)")

    for n_count in [4, 5, 6]:
        est, prob, _ = run_qpe(theta_exact, n_count)
        print(f"    n={n_count}: estimated={est:.6f}, prob={prob:.4f}, "
              f"error={abs(est - theta_exact):.2e}")

    # (b) Simulate with 4, 5, 6 counting qubits
    print(f"\n(b) Probability distributions:")
    for n_count in [4, 5, 6]:
        N = 2 ** n_count
        _, _, probs = run_qpe(theta_exact, n_count)
        peaks = [(k, probs[k]) for k in range(N) if probs[k] > 0.01]
        print(f"    n={n_count}: peaks at {[(k, f'{p:.4f}') for k, p in peaks]}")

    # (c) theta = 0.3 (not exactly representable)
    theta_irrational = 0.3
    print(f"\n(c) QPE for theta = 0.3 (not exactly representable in binary):")
    print(f"    {'n':>4} {'Estimated':>12} {'Error':>12} {'P(best)':>10}")
    print(f"    {'-'*42}")
    for n_count in [4, 6, 8, 10]:
        est, prob, _ = run_qpe(theta_irrational, n_count)
        error = abs(est - theta_irrational)
        print(f"    {n_count:>4} {est:>12.6f} {error:>12.2e} {prob:>10.4f}")

    print(f"\n    Error decreases as O(2^{{-n}}), confirming exponential precision")


# === Exercise 4: Approximate QFT ===
# Problem: Truncate controlled rotations with cutoff m.

def exercise_4():
    """Approximate QFT with controlled rotation cutoff."""
    n = 6
    N = 2 ** n

    def approximate_qft_matrix(n, m):
        """Build QFT approximation omitting CR_k for k > m."""
        N_dim = 2 ** n
        H_gate = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        I_gate = np.eye(2, dtype=complex)

        def single_qubit_gate(gate, target, n_q):
            ops = [I_gate] * n_q
            ops[target] = gate
            result = ops[0]
            for op in ops[1:]:
                result = np.kron(result, op)
            return result

        def controlled_phase(control, target, phase, n_q):
            mat = np.eye(N_dim, dtype=complex)
            for s in range(N_dim):
                c_bit = (s >> (n_q - 1 - control)) & 1
                t_bit = (s >> (n_q - 1 - target)) & 1
                if c_bit == 1 and t_bit == 1:
                    mat[s, s] = phase
            return mat

        result = np.eye(N_dim, dtype=complex)

        for target in range(n):
            # Hadamard
            result = single_qubit_gate(H_gate, target, n) @ result

            # Controlled rotations (only up to distance m)
            for control in range(target + 1, min(target + m, n)):
                k = control - target + 1
                if k <= m:
                    phase = np.exp(2j * np.pi / 2 ** k)
                    result = controlled_phase(control, target, phase, n) @ result

        # Bit reversal
        swap_mat = np.eye(N_dim, dtype=complex)
        for i in range(n // 2):
            j = n - 1 - i
            new_swap = np.zeros((N_dim, N_dim), dtype=complex)
            for s in range(N_dim):
                bi = (s >> (n - 1 - i)) & 1
                bj = (s >> (n - 1 - j)) & 1
                if bi != bj:
                    ns = s ^ (1 << (n - 1 - i)) ^ (1 << (n - 1 - j))
                else:
                    ns = s
                new_swap[ns, s] = 1
            swap_mat = new_swap @ swap_mat
        result = swap_mat @ result

        return result

    exact_qft = qft_matrix(n)

    # (a) Error vs m
    print(f"(a) Approximate QFT error for n={n}:")
    print(f"    {'m':>4} {'||error||':>12} {'Gate count':>12}")
    print(f"    {'-'*32}")

    for m in [2, 3, 4, 5]:
        approx = approximate_qft_matrix(n, m)
        error = np.linalg.norm(exact_qft - approx, ord=2)
        # Gate count: n Hadamards + sum of min(k-1, m-1) for each qubit
        n_cr = sum(min(k, m - 1) for k in range(1, n))
        n_h = n
        total = n_h + n_cr
        print(f"    {m:>4} {error:>12.6f} {total:>12}")

    # (b) Gate count scaling
    print(f"\n(b) Gate count: O(n*m) instead of O(n^2)")
    print(f"    Exact QFT: {n} H + {n*(n-1)//2} CR = {n + n*(n-1)//2} gates")

    # (c) Error threshold
    print(f"\n(c) For error < 1e-4:")
    for m in range(2, n + 1):
        approx = approximate_qft_matrix(n, m)
        error = np.linalg.norm(exact_qft - approx, ord=2)
        if error < 1e-4:
            print(f"    m = {m} achieves error {error:.2e} < 1e-4")
            break
    else:
        print(f"    Need m = {n} (full QFT) for n = {n}")


# === Exercise 5: Inverse QFT Application ===
# Problem: Apply inverse QFT to a Fourier mode.

def exercise_5():
    """Inverse QFT on a Fourier mode state."""
    n = 3
    N = 8
    m = 5

    # (a) Build Fourier mode state with frequency m=5
    state = np.array([np.exp(2j * np.pi * m * k / N) for k in range(N)]) / np.sqrt(N)
    print(f"(a) Fourier mode state with frequency m={m}:")
    for k in range(N):
        print(f"    |{k}> = |{format(k, '03b')}>: {state[k]:.4f}")

    # Apply inverse QFT
    F_inv = inverse_qft_matrix(n)
    result = F_inv @ state
    print(f"\n    After QFT^-1:")
    for k in range(N):
        if abs(result[k]) > 1e-10:
            print(f"    |{k}> = |{format(k, '03b')}>: {result[k]:.4f} "
                  f"(P = {abs(result[k])**2:.4f})")
        else:
            print(f"    |{k}> = |{format(k, '03b')}>: ~0")

    print(f"\n    Result: QFT^-1 maps Fourier mode m={m} to |{m}> = |{format(m, '03b')}>")

    # (b) Verify with matrix and circuit (using full QFT matrix)
    F = qft_matrix(n)
    result_check = F.conj().T @ state
    print(f"\n(b) Verification:")
    print(f"    Matrix method matches? {np.allclose(result, result_check)}")
    print(f"    Peak at index: {np.argmax(np.abs(result)**2)} (expected: {m})")

    # (c) General result
    print(f"\n(c) General result:")
    print(f"    QFT^-1 (1/sqrt(N) sum_k e^{{2pi i m k/N}} |k>) = |m>")
    print(f"    This holds for any integer m (mod N).")
    print(f"    Proof: QFT|m> = (1/sqrt(N)) sum_k e^{{2pi i mk/N}} |k>")
    print(f"    Therefore QFT^-1 of the RHS is |m>.")

    # Verify for all m
    print(f"\n    Verification for all m (n={n}, N={N}):")
    all_correct = True
    for m_test in range(N):
        phi = np.array([np.exp(2j * np.pi * m_test * k / N)
                        for k in range(N)]) / np.sqrt(N)
        recovered = F_inv @ phi
        peak = np.argmax(np.abs(recovered) ** 2)
        if peak != m_test:
            all_correct = False
            print(f"    m={m_test}: FAILED (got {peak})")
    if all_correct:
        print(f"    All m=0..{N-1}: PASSED")


if __name__ == "__main__":
    print("=== Exercise 1: QFT by Hand ===")
    exercise_1()
    print("\n=== Exercise 2: Period Finding with QFT ===")
    exercise_2()
    print("\n=== Exercise 3: QPE Precision ===")
    exercise_3()
    print("\n=== Exercise 4: Approximate QFT ===")
    exercise_4()
    print("\n=== Exercise 5: Inverse QFT Application ===")
    exercise_5()
    print("\nAll exercises completed!")
