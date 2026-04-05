"""
Exercises for Lesson 19: Quantum Walks
Topic: Quantum_Computing

Solutions covering discrete/continuous quantum walks, spreading analysis,
coin dependence, and graph isomorphism detection.
"""

import numpy as np
from scipy.linalg import expm


def exercise_1():
    """Quantum walk spreading rate analysis."""
    print("=" * 60)
    print("Exercise 1: Quantum Walk Spreading")
    print("=" * 60)

    for t in [10, 20, 50, 100, 200]:
        n_pos = 2 * t + 1
        center = n_pos // 2
        C = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        state = np.zeros(2 * n_pos, dtype=complex)
        state[center] = 1.0  # coin=0, pos=center

        for _ in range(t):
            new = np.zeros_like(state)
            for x in range(n_pos):
                c0, c1 = state[x], state[n_pos + x]
                new[x] = C[0,0]*c0 + C[0,1]*c1
                new[n_pos+x] = C[1,0]*c0 + C[1,1]*c1
            state = new
            new2 = np.zeros_like(state)
            for x in range(1, n_pos):
                new2[x-1] += state[x]
            for x in range(n_pos-1):
                new2[n_pos+x+1] += state[n_pos+x]
            state = new2

        prob = np.zeros(n_pos)
        for x in range(n_pos):
            prob[x] = abs(state[x])**2 + abs(state[n_pos+x])**2
        pos = np.arange(n_pos) - center
        std = np.sqrt(np.sum(pos**2 * prob))
        print(f"  t={t:4d}: std_quantum = {std:.2f}, std_classical = {np.sqrt(t):.2f}, ratio = {std/np.sqrt(t):.2f}")


def exercise_2():
    """CTQW perfect state transfer on paths."""
    print("\n" + "=" * 60)
    print("Exercise 2: CTQW Perfect State Transfer")
    print("=" * 60)

    for n in [2, 3, 4, 5, 6]:
        A = np.zeros((n, n))
        for i in range(n-1):
            A[i, i+1] = A[i+1, i] = 1.0

        psi0 = np.zeros(n, dtype=complex)
        psi0[0] = 1.0

        best_prob = 0
        best_t = 0
        for t in np.linspace(0.01, 20, 2000):
            psi = expm(-1j * A * t) @ psi0
            p = abs(psi[n-1])**2
            if p > best_prob:
                best_prob = p
                best_t = t

        print(f"  n={n}: max P(0->{n-1}) = {best_prob:.4f} at t = {best_t:.2f}"
              f"  {'PERFECT' if best_prob > 0.99 else ''}")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
