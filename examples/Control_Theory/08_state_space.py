"""
Control Theory — Lessons 11-12: State-Space, Controllability, Observability

Demonstrates:
1. State-space representation and simulation
2. Transfer function ↔ state-space conversion
3. Controllability and observability tests
4. State transition matrix computation
5. PBH test
"""
import numpy as np
from numpy.linalg import matrix_rank, eigvals, inv, det


# ── 1. State-Space System ────────────────────────────────────────────────

class StateSpace:
    """Continuous-time LTI state-space system: ẋ = Ax + Bu, y = Cx + Du."""

    def __init__(self, A, B, C, D=None):
        self.A = np.array(A, dtype=float)
        self.B = np.array(B, dtype=float)
        self.C = np.array(C, dtype=float)
        self.n = self.A.shape[0]
        if D is None:
            self.D = np.zeros((self.C.shape[0], self.B.shape[1]))
        else:
            self.D = np.array(D, dtype=float)

    @property
    def poles(self) -> np.ndarray:
        return eigvals(self.A)

    @property
    def is_stable(self) -> bool:
        return all(p.real < 0 for p in self.poles)

    def controllability_matrix(self) -> np.ndarray:
        """C = [B  AB  A²B  ...  A^(n-1)B]"""
        C_mat = self.B.copy()
        Ak_B = self.B.copy()
        for _ in range(1, self.n):
            Ak_B = self.A @ Ak_B
            C_mat = np.hstack([C_mat, Ak_B])
        return C_mat

    def observability_matrix(self) -> np.ndarray:
        """O = [C; CA; CA²; ...; CA^(n-1)]"""
        O_mat = self.C.copy()
        CA_k = self.C.copy()
        for _ in range(1, self.n):
            CA_k = CA_k @ self.A
            O_mat = np.vstack([O_mat, CA_k])
        return O_mat

    def is_controllable(self) -> bool:
        return matrix_rank(self.controllability_matrix()) == self.n

    def is_observable(self) -> bool:
        return matrix_rank(self.observability_matrix()) == self.n

    def transfer_function(self) -> tuple:
        """Compute G(s) = C(sI-A)^{-1}B + D  for SISO."""
        # Characteristic polynomial
        char_poly = np.real(np.poly(self.A))  # [1, a_{n-1}, ..., a_0]

        # Numerator via adjugate
        # For SISO: G(s) = C * adj(sI-A) * B / det(sI-A)
        # We compute numerically at several points and fit
        n_points = 2 * self.n + 2
        s_vals = np.linspace(-10, 10, n_points) + 1j * np.linspace(0.1, 5, n_points)
        G_vals = []
        for s in s_vals:
            sI_A = s * np.eye(self.n) - self.A
            G_s = self.C @ inv(sI_A) @ self.B + self.D
            G_vals.append(G_s[0, 0])

        # Fit numerator polynomial
        den_vals = np.array([np.polyval(char_poly, s) for s in s_vals])
        num_vals = np.array(G_vals) * den_vals
        num_coeffs = np.polyfit(s_vals, num_vals, self.n - 1)
        num_coeffs = np.real(num_coeffs)

        return num_coeffs, char_poly

    def simulate(self, u_func, x0, t) -> tuple:
        """Euler integration."""
        dt = t[1] - t[0]
        x = np.zeros((len(t), self.n))
        y = np.zeros((len(t), self.C.shape[0]))
        x[0] = x0

        for i in range(len(t) - 1):
            u = np.array(u_func(t[i])).reshape(-1, 1)
            x_col = x[i].reshape(-1, 1)
            x_next = x_col + dt * (self.A @ x_col + self.B @ u)
            x[i + 1] = x_next.flatten()
            y[i] = (self.C @ x_col + self.D @ u).flatten()

        y[-1] = (self.C @ x[-1].reshape(-1, 1)
                 + self.D @ np.array(u_func(t[-1])).reshape(-1, 1)).flatten()
        return x, y


# ── 2. PBH Test ──────────────────────────────────────────────────────────

def pbh_controllability_test(A, B) -> list[dict]:
    """PBH test: rank([λI-A, B]) = n for all eigenvalues λ."""
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    n = A.shape[0]
    eigs = eigvals(A)
    results = []
    for lam in eigs:
        M = np.hstack([lam * np.eye(n) - A, B])
        r = matrix_rank(M)
        results.append({
            "eigenvalue": lam,
            "rank": r,
            "controllable": r == n
        })
    return results


def pbh_observability_test(A, C) -> list[dict]:
    """PBH test: rank([λI-A; C]) = n for all eigenvalues λ."""
    A = np.array(A, dtype=float)
    C = np.array(C, dtype=float)
    n = A.shape[0]
    eigs = eigvals(A)
    results = []
    for lam in eigs:
        M = np.vstack([lam * np.eye(n) - A, C])
        r = matrix_rank(M)
        results.append({
            "eigenvalue": lam,
            "rank": r,
            "observable": r == n
        })
    return results


# ── 3. Canonical Forms ───────────────────────────────────────────────────

def controllable_canonical_form(num: list[float],
                                den: list[float]) -> tuple:
    """Convert TF to controllable canonical form (CCF)."""
    # Normalize so leading den coefficient = 1
    den = np.array(den, dtype=float)
    num = np.array(num, dtype=float)
    den = den / den[0]
    num = num / den[0] if len(num) > 0 else num

    n = len(den) - 1  # system order

    # A matrix (companion form)
    A = np.zeros((n, n))
    A[:n - 1, 1:] = np.eye(n - 1)
    A[-1, :] = -den[-1:0:-1]  # -[a0, a1, ..., a_{n-1}]

    # B matrix
    B = np.zeros((n, 1))
    B[-1, 0] = 1.0

    # C matrix
    C = np.zeros((1, n))
    # Pad num to length n
    num_padded = np.zeros(n)
    num_padded[n - len(num):] = num
    C[0, :] = num_padded[::-1]

    D = np.zeros((1, 1))

    return A, B, C, D


# ── Demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Mass-spring-damper
    print("=== Mass-Spring-Damper State Space ===")
    m, b, k = 1.0, 0.5, 4.0
    sys = StateSpace(
        A=[[0, 1], [-k / m, -b / m]],
        B=[[0], [1 / m]],
        C=[[1, 0]]
    )
    print(f"  Poles: {sys.poles}")
    print(f"  Stable: {sys.is_stable}")
    print(f"  Controllable: {sys.is_controllable()}")
    print(f"  Observable: {sys.is_observable()}")

    # Controllability matrix
    C_mat = sys.controllability_matrix()
    print(f"  Controllability matrix:\n{C_mat}")
    print(f"  Rank: {matrix_rank(C_mat)}")

    # Simulate step response
    t = np.linspace(0, 10, 1000)
    x, y = sys.simulate(lambda t: [1.0], [0, 0], t)
    print(f"  Step response final value: {y[-1, 0]:.4f} (expect {1 / k:.4f})")

    # Uncontrollable example
    print("\n=== Uncontrollable System ===")
    sys_uc = StateSpace(
        A=[[-1, 0], [0, -2]],
        B=[[1], [0]],
        C=[[1, 1]]
    )
    print(f"  Controllable: {sys_uc.is_controllable()}")
    print(f"  Observable: {sys_uc.is_observable()}")

    # PBH test
    pbh = pbh_controllability_test(sys_uc.A, sys_uc.B)
    for r in pbh:
        print(f"  PBH at λ={r['eigenvalue']:.1f}: "
              f"rank={r['rank']}, controllable={r['controllable']}")

    # Unobservable example (from Lesson 12)
    print("\n=== Unobservable System (Hidden Mode) ===")
    sys_uo = StateSpace(
        A=[[-1, 0], [0, -3]],
        B=[[1], [1]],
        C=[[1, 0]]
    )
    print(f"  Controllable: {sys_uo.is_controllable()}")
    print(f"  Observable: {sys_uo.is_observable()}")
    # Transfer function should show only s=-1 pole
    num_tf, den_tf = sys_uo.transfer_function()
    print(f"  TF numerator ≈ {np.round(num_tf, 2)}")
    print(f"  TF denominator ≈ {np.round(den_tf, 2)}")
    print("  → Mode at s=-3 is hidden (unobservable)")

    # Controllable canonical form
    print("\n=== Controllable Canonical Form ===")
    # G(s) = (2s+3) / (s³+4s²+5s+6)
    A_ccf, B_ccf, C_ccf, D_ccf = controllable_canonical_form(
        [2, 3], [1, 4, 5, 6])
    print(f"  A =\n{A_ccf}")
    print(f"  B = {B_ccf.flatten()}")
    print(f"  C = {C_ccf.flatten()}")
    sys_ccf = StateSpace(A_ccf, B_ccf, C_ccf, D_ccf)
    print(f"  Poles: {np.round(sys_ccf.poles, 3)}")
