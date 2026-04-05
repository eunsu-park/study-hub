"""
Exercises for Lesson 21: Quantum Chemistry
Topic: Quantum_Computing

Solutions covering molecular integrals, Jordan-Wigner verification,
H2 dissociation curve, symmetry reduction, and VQE ansatz comparison.
"""

import numpy as np
from functools import reduce
from scipy.optimize import minimize

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def kron_list(ops):
    return reduce(np.kron, ops)

def jw_creation(p, n):
    raising = (X - 1j * Y) / 2
    ops = [Z if q < p else (raising if q == p else I2) for q in range(n)]
    return kron_list(ops)

def jw_annihilation(p, n):
    return jw_creation(p, n).conj().T


def exercise_1():
    """Verify anticommutation relations for JW operators."""
    print("=" * 60)
    print("Exercise 1 (partial) / Exercise 2: JW Anticommutation")
    print("=" * 60)

    n = 4
    N = 2 ** n
    print(f"\nVerifying {{a_p, a_q^dag}} = delta_pq for {n} qubits:")
    for p in range(n):
        for q in range(n):
            ap = jw_annihilation(p, n)
            aq_dag = jw_creation(q, n)
            anticomm = ap @ aq_dag + aq_dag @ ap
            expected = np.eye(N) if p == q else np.zeros((N, N))
            ok = np.allclose(anticomm, expected)
            if p == q or not ok:
                print(f"  {{a_{p}, a_{q}^dag}} = delta_{p}{q}: {ok}")


def exercise_3():
    """H2 potential energy curve."""
    print("\n" + "=" * 60)
    print("Exercise 3: H2 Dissociation Curve")
    print("=" * 60)

    def build_h2(R):
        g = {'I': -0.8105 + 0.1*(R-0.74), 'Z0': 0.1721, 'Z1': 0.1721,
             'Z2': -0.2232, 'Z3': -0.2232, 'Z0Z1': 0.1686,
             'Z0Z2': 0.1205, 'Z0Z3': 0.1659, 'Z1Z2': 0.1659,
             'Z1Z3': 0.1205, 'Z2Z3': 0.1743, 'XXYY': -0.0453 - 0.02*(R-0.74)}

        H = g['I'] * kron_list([I2]*4)
        H += g['Z0'] * kron_list([Z,I2,I2,I2]) + g['Z1'] * kron_list([I2,Z,I2,I2])
        H += g['Z2'] * kron_list([I2,I2,Z,I2]) + g['Z3'] * kron_list([I2,I2,I2,Z])
        H += g['Z0Z1'] * kron_list([Z,Z,I2,I2])
        H += g['Z0Z2'] * kron_list([Z,I2,Z,I2])
        H += g['Z0Z3'] * kron_list([Z,I2,I2,Z])
        H += g['Z1Z2'] * kron_list([I2,Z,Z,I2])
        H += g['Z1Z3'] * kron_list([I2,Z,I2,Z])
        H += g['Z2Z3'] * kron_list([I2,I2,Z,Z])
        c = g['XXYY']
        H += c * (kron_list([X,X,Y,Y]) - kron_list([X,Y,Y,X])
                + kron_list([Y,X,X,Y]) - kron_list([Y,Y,X,X]))
        H += 1.0/R * kron_list([I2]*4)  # nuclear repulsion
        return H

    print(f"\n{'R (A)':>8} {'E_exact (Ha)':>14} {'E_HF (Ha)':>14}")
    print("-" * 40)

    for R in [0.5, 0.74, 1.0, 1.5, 2.0, 2.5, 3.0]:
        H = build_h2(R)
        N_op = sum(kron_list([I2 if j != i else (I2-Z)/2 for j in range(4)])
                   for i in range(4))
        evals, evecs = np.linalg.eigh(H)

        e_exact = float('inf')
        for i in range(len(evals)):
            n_exp = np.real(evecs[:,i].conj() @ N_op @ evecs[:,i])
            if abs(n_exp - 2) < 0.1 and evals[i] < e_exact:
                e_exact = evals[i]

        hf = np.zeros(16, dtype=complex); hf[12] = 1.0
        e_hf = np.real(hf.conj() @ H @ hf)
        print(f"{R:8.2f} {e_exact:14.6f} {e_hf:14.6f}")


if __name__ == "__main__":
    exercise_1()
    exercise_3()
