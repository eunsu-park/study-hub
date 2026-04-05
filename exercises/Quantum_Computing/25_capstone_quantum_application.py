"""
Exercises for Lesson 25: Capstone Quantum Application
Topic: Quantum_Computing

End-to-end VQE and QAOA capstone projects with noise simulation
and error mitigation.
"""

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from functools import reduce

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def kron_list(ops):
    return reduce(np.kron, ops)


def exercise_vqe():
    """Complete VQE capstone for H2."""
    print("=" * 60)
    print("Capstone Exercise: VQE for H2")
    print("=" * 60)

    g = {'I': -0.8105, 'Z0': 0.1721, 'Z1': 0.1721, 'Z2': -0.2232,
         'Z3': -0.2232, 'Z0Z1': 0.1686, 'Z0Z2': 0.1205, 'Z0Z3': 0.1659,
         'Z1Z2': 0.1659, 'Z1Z3': 0.1205, 'Z2Z3': 0.1743, 'XXYY': -0.0453}

    H = g['I'] * kron_list([I2]*4)
    H += g['Z0'] * kron_list([Z,I2,I2,I2]) + g['Z1'] * kron_list([I2,Z,I2,I2])
    H += g['Z2'] * kron_list([I2,I2,Z,I2]) + g['Z3'] * kron_list([I2,I2,I2,Z])
    H += g['Z0Z1']*kron_list([Z,Z,I2,I2]) + g['Z0Z2']*kron_list([Z,I2,Z,I2])
    H += g['Z0Z3']*kron_list([Z,I2,I2,Z]) + g['Z1Z2']*kron_list([I2,Z,Z,I2])
    H += g['Z1Z3']*kron_list([I2,Z,I2,Z]) + g['Z2Z3']*kron_list([I2,I2,Z,Z])
    c = g['XXYY']
    H += c*(kron_list([X,X,Y,Y]) - kron_list([X,Y,Y,X])
          + kron_list([Y,X,X,Y]) - kron_list([Y,Y,X,X]))

    exact = np.min(np.linalg.eigvalsh(H))

    def vqe(theta, noise=0):
        state = np.zeros(16, dtype=complex); state[12] = 1.0
        a, b = state[12], state[3]
        state[12] = np.cos(theta)*a - np.sin(theta)*b
        state[3] = np.sin(theta)*a + np.cos(theta)*b
        if noise > 0:
            rho = np.outer(state, state.conj())
            fid = (1-noise)**20
            rho = fid*rho + (1-fid)*np.eye(16)/16
            return np.real(np.trace(H @ rho))
        return np.real(state.conj() @ H @ state)

    # Ideal
    r = minimize(lambda t: vqe(t[0]), [0.0], method='COBYLA')
    print(f"\n  Exact energy: {exact:.6f} Ha")
    print(f"  VQE (ideal):  {r.fun:.6f} Ha, error: {abs(r.fun-exact)*1000:.2f} mHa")

    # Noisy
    r_n = minimize(lambda t: vqe(t[0], 0.005), [0.0], method='COBYLA')
    print(f"  VQE (noisy):  {r_n.fun:.6f} Ha, error: {abs(r_n.fun-exact)*1000:.2f} mHa")

    # ZNE
    theta_opt = r.x[0]
    vals = [vqe(theta_opt, 0.005*f) for f in [1, 1.5, 2, 2.5]]
    coeffs = np.polyfit([1, 1.5, 2, 2.5], vals, 1)
    zne = np.polyval(coeffs, 0)
    print(f"  VQE + ZNE:    {zne:.6f} Ha, error: {abs(zne-exact)*1000:.2f} mHa")


def exercise_qaoa():
    """Complete QAOA capstone for Max-Cut."""
    print("\n" + "=" * 60)
    print("Capstone Exercise: QAOA for Max-Cut")
    print("=" * 60)

    np.random.seed(42)
    n = 6; N = 2**n
    adj = np.zeros((n,n), dtype=int)
    for i in range(n):
        for j in range(i+1,n):
            if np.random.random() < 0.5:
                adj[i,j] = adj[j,i] = 1

    H_C = np.zeros((N,N), dtype=complex)
    for i in range(n):
        for j in range(i+1,n):
            if adj[i,j]:
                ops = [I2]*n; ops[i] = Z; ops[j] = Z
                H_C += (np.eye(N) - kron_list(ops))/2

    H_M = sum(kron_list([X if k==i else I2 for k in range(n)]) for i in range(n))

    # Brute force optimal
    best_cut = 0
    for z in range(N):
        cut = sum((1 - (1-2*((z>>(n-1-i))&1))*(1-2*((z>>(n-1-j))&1)))/2
                  for i in range(n) for j in range(i+1,n) if adj[i,j])
        best_cut = max(best_cut, int(cut))

    print(f"\n  Graph: {n} vertices, optimal cut = {best_cut}")

    for p in [1, 2, 3]:
        best = 0
        for _ in range(10):
            def cost(params):
                g, b = params[:p], params[p:]
                s = np.ones(N, dtype=complex)/np.sqrt(N)
                for l in range(p):
                    s = expm(-1j*g[l]*H_C) @ s
                    s = expm(-1j*b[l]*H_M) @ s
                return -np.real(s.conj() @ H_C @ s)
            r = minimize(cost, np.random.uniform(0, np.pi, 2*p), method='COBYLA')
            best = max(best, -r.fun)
        print(f"  QAOA p={p}: <C> = {best:.4f}, ratio = {best/best_cut:.4f}")


if __name__ == "__main__":
    exercise_vqe()
    exercise_qaoa()
