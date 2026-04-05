"""
Control Theory — Lesson 15: Digital Control Systems

Demonstrates:
1. Zero-order hold (ZOH) discretization
2. Tustin's (bilinear) method
3. Digital PID (positional and velocity forms)
4. Jury stability test
5. s-to-z mapping verification
"""
import numpy as np
from numpy.linalg import eigvals


# ── 1. ZOH Discretization ───────────────────────────────────────────────

def zoh_first_order(a: float, T: float) -> tuple:
    """
    ZOH-equivalent discretization of G(s) = 1/(s+a).
    Returns (num_z, den_z) for G(z).
    """
    # G(z) = (1-e^{-aT}) / (z - e^{-aT})
    e_aT = np.exp(-a * T)
    num = [1 - e_aT]                    # numerator coefficients
    den = [1, -e_aT]                     # denominator: z - e^{-aT}
    return num, den


def zoh_state_space(A_c, B_c, T):
    """
    ZOH discretization of continuous state-space (A_c, B_c).
    Returns (A_d, B_d) where x[k+1] = A_d x[k] + B_d u[k].
    """
    A_c = np.array(A_c, dtype=float)
    B_c = np.array(B_c, dtype=float)
    n = A_c.shape[0]

    # A_d = e^{A_c T}  (matrix exponential via Padé or Taylor)
    # Simple Taylor series for small A_c T
    A_d = np.eye(n)
    term = np.eye(n)
    for k in range(1, 20):
        term = term @ (A_c * T) / k
        A_d += term

    # B_d = (A_d - I) A_c^{-1} B_c  (if A_c is invertible)
    try:
        B_d = np.linalg.solve(A_c, (A_d - np.eye(n))) @ B_c
    except np.linalg.LinAlgError:
        # Fallback: numerical integration
        B_d = np.zeros_like(B_c)
        steps = 100
        dt = T / steps
        eAt = np.eye(n)
        for _ in range(steps):
            B_d += eAt @ B_c * dt
            eAt = eAt + eAt @ A_c * dt

    return A_d, B_d


# ── 2. Tustin's Method ──────────────────────────────────────────────────

def tustin_first_order(tau: float, K: float, T: float) -> tuple:
    """
    Discretize G(s) = K/(τs+1) using Tustin's bilinear transform.
    s = (2/T)(z-1)/(z+1)
    Returns (num_z, den_z).
    """
    # G(z) = K / (τ·(2/T)(z-1)/(z+1) + 1)
    #       = K(z+1) / (τ(2/T)(z-1) + (z+1))
    #       = K(z+1) / ((2τ/T + 1)z + (1 - 2τ/T))
    a = 2 * tau / T + 1
    b = 1 - 2 * tau / T
    num = [K / a, K / a]      # K/a · (z + 1)
    den = [1, b / a]           # z + b/a
    return num, den


# ── 3. Digital PID ──────────────────────────────────────────────────────

class DigitalPID:
    """Velocity (incremental) form PID — avoids windup naturally."""

    def __init__(self, Kp: float, Ki: float, Kd: float, T: float,
                 u_min: float = -float('inf'),
                 u_max: float = float('inf')):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.T = T
        self.u_min = u_min
        self.u_max = u_max
        self._e_prev = 0.0
        self._e_prev2 = 0.0
        self._u = 0.0

    def compute(self, error: float) -> float:
        """Velocity form: Δu[k] and accumulate."""
        delta_u = (self.Kp * (error - self._e_prev)
                   + self.Ki * self.T * error
                   + self.Kd / self.T * (error - 2 * self._e_prev + self._e_prev2))

        self._u = np.clip(self._u + delta_u, self.u_min, self.u_max)
        self._e_prev2 = self._e_prev
        self._e_prev = error
        return self._u

    def reset(self):
        self._e_prev = 0.0
        self._e_prev2 = 0.0
        self._u = 0.0


# ── 4. Jury Stability Test ──────────────────────────────────────────────

def jury_test(coeffs: list[float]) -> dict:
    """
    Jury stability test for discrete-time polynomial P(z).

    Args:
        coeffs: [a_n, a_{n-1}, ..., a_1, a_0]  (a_n z^n + ... + a_0)

    Returns dict with necessary conditions and stability result.
    """
    a = np.array(coeffs, dtype=float)
    n = len(a) - 1  # degree

    P_1 = np.polyval(a, 1)
    P_neg1 = np.polyval(a, -1)
    neg_n_P_neg1 = ((-1)**n) * P_neg1

    # Necessary conditions
    cond1 = P_1 > 0
    cond2 = neg_n_P_neg1 > 0
    cond3 = abs(a[-1]) < abs(a[0])

    # For full sufficiency, build the Jury array
    # (simplified: just check roots directly for verification)
    roots = np.roots(a)
    all_inside = all(abs(r) < 1 for r in roots)

    return {
        "P(1)": P_1,
        "(-1)^n P(-1)": neg_n_P_neg1,
        "|a_0| < |a_n|": (abs(a[-1]), abs(a[0])),
        "cond1_P1_pos": cond1,
        "cond2_neg_n_P_neg1_pos": cond2,
        "cond3_a0_lt_an": cond3,
        "necessary_conditions_met": cond1 and cond2 and cond3,
        "roots": roots,
        "all_inside_unit_circle": all_inside,
        "stable": all_inside,
    }


# ── 5. s-to-z Mapping Verification ──────────────────────────────────────

def s_to_z_map(s_poles: list[complex], T: float) -> list[complex]:
    """Map continuous poles to discrete: z = e^{sT}."""
    return [np.exp(s * T) for s in s_poles]


# ── Demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    T = 0.1  # sampling period

    # ZOH discretization of G(s) = 1/(s+2)
    print("=== ZOH Discretization: G(s) = 1/(s+2), T=0.1 ===")
    num_z, den_z = zoh_first_order(a=2, T=T)
    print(f"  G(z) numerator:   {num_z}")
    print(f"  G(z) denominator: {den_z}")
    z_pole = -den_z[1]
    expected = np.exp(-2 * T)
    print(f"  Discrete pole: z = {z_pole:.6f}")
    print(f"  Expected e^{{-aT}} = {expected:.6f}")

    # ZOH for state-space: double integrator
    print("\n=== ZOH State-Space: Double Integrator ===")
    Ac = np.array([[0, 1], [0, 0]])
    Bc = np.array([[0], [1]])
    Ad, Bd = zoh_state_space(Ac, Bc, T)
    print(f"  A_d =\n{Ad}")
    print(f"  B_d = {Bd.flatten()}")
    print(f"  Expected A_d = [[1, T], [0, 1]] = [[1, {T}], [0, 1]]")

    # Tustin's method
    print(f"\n=== Tustin Discretization: G(s) = 5/(0.5s+1), T={T} ===")
    num_t, den_t = tustin_first_order(tau=0.5, K=5, T=T)
    print(f"  G(z) numerator:   {[f'{x:.4f}' for x in num_t]}")
    print(f"  G(z) denominator: {[f'{x:.4f}' for x in den_t]}")

    # Digital PID simulation
    print("\n=== Digital PID (Velocity Form) ===")
    pid = DigitalPID(Kp=2.0, Ki=1.0, Kd=0.1, T=T, u_min=-5, u_max=5)

    # Simple first-order plant simulation
    plant_a, plant_K = 2.0, 5.0
    y = 0.0
    setpoint = 1.0
    print(f"  Setpoint = {setpoint}, Plant: G(s) = {plant_K}/(s+{plant_a})")
    for step in range(100):
        e = setpoint - y
        u = pid.compute(e)
        # Euler step for plant
        y = y + T * (plant_K * u - plant_a * y)
        if step in [0, 5, 10, 20, 50, 99]:
            print(f"  k={step:3d}: e={e:+.4f}, u={u:+.4f}, y={y:.4f}")

    # Jury stability test
    print("\n=== Jury Stability Test ===")
    # P(z) = z³ - 1.2z² + 0.5z - 0.1
    result = jury_test([1, -1.2, 0.5, -0.1])
    print(f"  P(z) = z³ - 1.2z² + 0.5z - 0.1")
    print(f"  P(1) = {result['P(1)']:.4f} > 0? {result['cond1_P1_pos']}")
    print(f"  (-1)³P(-1) = {result['(-1)^n P(-1)']:.4f} > 0? "
          f"{result['cond2_neg_n_P_neg1_pos']}")
    a0, an = result['|a_0| < |a_n|']
    print(f"  |a_0|={a0:.1f} < |a_n|={an:.1f}? {result['cond3_a0_lt_an']}")
    print(f"  Roots: {result['roots']}")
    print(f"  All inside unit circle: {result['all_inside_unit_circle']}")
    print(f"  Stable: {result['stable']}")

    # s-to-z mapping
    print("\n=== s-to-z Mapping Verification ===")
    s_poles = [-1, -5, -2 + 3j, -2 - 3j]
    z_poles = s_to_z_map(s_poles, T)
    print(f"  T = {T}")
    for sp, zp in zip(s_poles, z_poles):
        inside = "|z|<1" if abs(zp) < 1 else "|z|≥1"
        print(f"  s = {sp:>10s} → z = {zp:.4f}  ({inside})"
              .replace("s = ", f"s = {str(sp):>10s} → z = "))
    # Simpler print
    for sp, zp in zip(s_poles, z_poles):
        print(f"  s={sp!s:>12} → z={zp:.4f} (|z|={abs(zp):.4f})")
