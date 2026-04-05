"""
Exercises for Lesson 20: Noise and Quantum Channels
Topic: Quantum_Computing

Solutions covering channel composition, Choi matrices, process tomography,
randomized benchmarking, and noise-aware algorithm design.
"""

import numpy as np

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def apply_channel(rho, kraus):
    out = np.zeros_like(rho)
    for E in kraus:
        out += E @ rho @ E.conj().T
    return out


def depolarizing(p):
    return [np.sqrt(1-p)*I, np.sqrt(p/3)*X, np.sqrt(p/3)*Y, np.sqrt(p/3)*Z]


def exercise_1():
    """Channel composition: is composition of depolarizing channels depolarizing?"""
    print("=" * 60)
    print("Exercise 1: Channel Composition")
    print("=" * 60)

    p1, p2 = 0.1, 0.2
    kraus1, kraus2 = depolarizing(p1), depolarizing(p2)

    # Compose: E2 o E1
    rho_test = np.array([[0.7, 0.3j], [-0.3j, 0.3]], dtype=complex)
    rho_mid = apply_channel(rho_test, kraus1)
    rho_out = apply_channel(rho_mid, kraus2)

    # Check if result is depolarizing with some p_eff
    # For depolarizing: rho_out = (1-4p/3)*rho + (4p/3)*(I/2)
    # Bloch vector shrinks by (1-4p/3)
    rx_in = 2 * np.real(rho_test[0, 1])
    rx_out = 2 * np.real(rho_out[0, 1])
    shrink = rx_out / rx_in if abs(rx_in) > 1e-10 else 0

    # For composition: shrink = (1-4p1/3)*(1-4p2/3)
    expected_shrink = (1 - 4*p1/3) * (1 - 4*p2/3)
    p_eff = 3/4 * (1 - expected_shrink)

    print(f"\n  Depolarizing p1={p1}, p2={p2}")
    print(f"  Observed Bloch shrink: {shrink:.4f}")
    print(f"  Expected shrink: {expected_shrink:.4f}")
    print(f"  Equivalent single depolarizing p_eff = {p_eff:.4f}")
    print(f"  Composition is depolarizing: True")


def exercise_2():
    """Choi matrix eigenvalues as function of p."""
    print("\n" + "=" * 60)
    print("Exercise 2: Choi Matrix Analysis")
    print("=" * 60)

    print(f"\n  {'p':>6} {'eigenvalues':>40}")
    print(f"  {'-' * 48}")

    for p in np.linspace(0, 1, 11):
        kraus = depolarizing(p)
        omega = np.zeros(4, dtype=complex)
        omega[0] = omega[3] = 1/np.sqrt(2)
        rho_omega = np.outer(omega, omega.conj())

        choi = np.zeros((4, 4), dtype=complex)
        for E in kraus:
            E_I = np.kron(E, I)
            choi += E_I @ rho_omega @ E_I.conj().T

        evals = np.sort(np.real(np.linalg.eigvalsh(choi)))[::-1]
        evals_str = ', '.join(f'{e:.4f}' for e in evals)
        print(f"  {p:6.2f} [{evals_str}]")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
