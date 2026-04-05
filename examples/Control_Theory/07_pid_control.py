"""
Control Theory — Lesson 9: PID Control

Demonstrates:
1. P, PI, PD, PID controller simulation
2. Ziegler-Nichols tuning (open-loop and closed-loop methods)
3. Anti-windup (clamping and back-calculation)
4. Derivative filtering
"""
import numpy as np
from dataclasses import dataclass, field


@dataclass
class PIDController:
    """Discrete-time PID controller with practical features."""
    Kp: float = 1.0
    Ki: float = 0.0
    Kd: float = 0.0
    dt: float = 0.01
    u_min: float = -float('inf')
    u_max: float = float('inf')
    derivative_filter_N: float = 20.0  # derivative filter coefficient
    anti_windup: bool = True

    # Internal state
    _integral: float = field(default=0.0, init=False)
    _prev_error: float = field(default=0.0, init=False)
    _prev_derivative: float = field(default=0.0, init=False)

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_derivative = 0.0

    def compute(self, error: float) -> float:
        """Compute PID output with anti-windup and derivative filtering."""
        # Proportional
        P = self.Kp * error

        # Integral (trapezoidal)
        self._integral += self.Ki * self.dt * (error + self._prev_error) / 2

        # Filtered derivative (first-order filter)
        alpha = self.derivative_filter_N * self.dt / (1 + self.derivative_filter_N * self.dt)
        raw_deriv = self.Kd * (error - self._prev_error) / self.dt
        D = alpha * raw_deriv + (1 - alpha) * self._prev_derivative
        self._prev_derivative = D

        # Total
        u_raw = P + self._integral + D

        # Saturation
        u = np.clip(u_raw, self.u_min, self.u_max)

        # Anti-windup (back-calculation)
        if self.anti_windup and u != u_raw:
            self._integral += (u - u_raw) * 0.5  # back-calculation

        self._prev_error = error
        return u


# ── Plant: First-Order Plus Dead Time (FOPDT) ───────────────────────────

def simulate_fopdt(K: float, tau: float, L: float, u: np.ndarray,
                   dt: float) -> np.ndarray:
    """Simulate G(s) = K*e^(-Ls) / (τs + 1) using Euler + delay buffer."""
    n = len(u)
    y = np.zeros(n)
    delay_steps = max(1, int(L / dt))

    # Delayed input buffer
    u_delayed = np.zeros(n)
    for i in range(n):
        idx = i - delay_steps
        u_delayed[i] = u[idx] if idx >= 0 else 0

    for i in range(1, n):
        y[i] = y[i - 1] + dt / tau * (K * u_delayed[i] - y[i - 1])

    return y


# ── Ziegler-Nichols Tuning ───────────────────────────────────────────────

def zn_open_loop(K0: float, L: float, T: float,
                 controller_type: str = "PID") -> dict:
    """
    Ziegler-Nichols open-loop (process reaction curve) tuning.

    Args:
        K0: plant DC gain
        L: apparent dead time
        T: apparent time constant
        controller_type: "P", "PI", or "PID"
    """
    rules = {
        "P":   {"Kp": T / (K0 * L)},
        "PI":  {"Kp": 0.9 * T / (K0 * L), "Ti": L / 0.3},
        "PID": {"Kp": 1.2 * T / (K0 * L), "Ti": 2 * L, "Td": 0.5 * L},
    }
    params = rules[controller_type]
    result = {"Kp": params["Kp"], "Ki": 0, "Kd": 0}
    if "Ti" in params:
        result["Ki"] = params["Kp"] / params["Ti"]
    if "Td" in params:
        result["Kd"] = params["Kp"] * params["Td"]
    return result


def zn_closed_loop(Ku: float, Tu: float,
                   controller_type: str = "PID") -> dict:
    """
    Ziegler-Nichols closed-loop (ultimate gain) tuning.

    Args:
        Ku: ultimate gain (gain at sustained oscillation)
        Tu: ultimate period
    """
    rules = {
        "P":   {"Kp": 0.5 * Ku},
        "PI":  {"Kp": 0.45 * Ku, "Ti": Tu / 1.2},
        "PID": {"Kp": 0.6 * Ku, "Ti": Tu / 2, "Td": Tu / 8},
    }
    params = rules[controller_type]
    result = {"Kp": params["Kp"], "Ki": 0, "Kd": 0}
    if "Ti" in params:
        result["Ki"] = params["Kp"] / params["Ti"]
    if "Td" in params:
        result["Kd"] = params["Kp"] * params["Td"]
    return result


# ── Closed-Loop Simulation ───────────────────────────────────────────────

def simulate_closed_loop(pid: PIDController, plant_K: float,
                         plant_tau: float, plant_L: float,
                         r: np.ndarray, dt: float) -> dict:
    """Simulate PID + FOPDT plant in closed loop."""
    n = len(r)
    y = np.zeros(n)
    u = np.zeros(n)
    e = np.zeros(n)

    delay_steps = max(1, int(plant_L / dt))
    pid.reset()

    for i in range(1, n):
        e[i] = r[i] - y[i - 1]
        u[i] = pid.compute(e[i])

        # Plant dynamics with delay
        idx = i - delay_steps
        u_delayed = u[idx] if idx >= 0 else 0
        y[i] = y[i - 1] + dt / plant_tau * (plant_K * u_delayed - y[i - 1])

    return {"y": y, "u": u, "e": e}


# ── Demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dt = 0.01
    t = np.arange(0, 20, dt)
    r = np.ones_like(t)  # unit step reference

    # Plant parameters (FOPDT)
    K0, tau, L = 2.0, 3.0, 0.5
    print("=== Plant: K=2, τ=3, L=0.5 ===\n")

    # ZN open-loop tuning
    print("=== Ziegler-Nichols Open-Loop Tuning ===")
    for ctype in ["P", "PI", "PID"]:
        params = zn_open_loop(K0, L, tau, ctype)
        print(f"  {ctype:3s}: Kp={params['Kp']:.3f}, "
              f"Ki={params['Ki']:.3f}, Kd={params['Kd']:.3f}")

    # Simulate with PID (ZN tuning)
    print("\n=== Closed-Loop Simulation ===")
    zn_params = zn_open_loop(K0, L, tau, "PID")
    pid = PIDController(Kp=zn_params["Kp"], Ki=zn_params["Ki"],
                        Kd=zn_params["Kd"], dt=dt,
                        u_min=-10, u_max=10)
    result = simulate_closed_loop(pid, K0, tau, L, r, dt)

    # Performance metrics
    y = result["y"]
    overshoot = (np.max(y) - 1.0) * 100
    # Settling time (2%)
    settled = np.where(np.abs(y - 1.0) > 0.02)[0]
    ts = t[settled[-1]] if len(settled) > 0 else 0
    ess = abs(1.0 - y[-1])

    print(f"  ZN-PID: Kp={pid.Kp:.3f}, Ki={pid.Ki:.3f}, Kd={pid.Kd:.3f}")
    print(f"  Overshoot: {overshoot:.1f}%")
    print(f"  Settling time (2%): {ts:.2f} s")
    print(f"  Steady-state error: {ess:.4f}")
    print(f"  Max control effort: {np.max(np.abs(result['u'])):.2f}")

    # Compare P, PI, PID
    print("\n=== Controller Comparison ===")
    for name, kp, ki, kd in [("P-only", 1.0, 0, 0),
                              ("PI", 1.0, 0.5, 0),
                              ("PID", 1.0, 0.5, 0.3)]:
        pid = PIDController(Kp=kp, Ki=ki, Kd=kd, dt=dt)
        res = simulate_closed_loop(pid, K0, tau, L, r, dt)
        y = res["y"]
        mp = (np.max(y) - 1.0) * 100
        ess = abs(1.0 - y[-1])
        print(f"  {name:8s}: Kp={kp}, Ki={ki}, Kd={kd} → "
              f"Mp={mp:5.1f}%, e_ss={ess:.4f}")

    # Anti-windup demo
    print("\n=== Anti-Windup Demo ===")
    for aw in [True, False]:
        pid = PIDController(Kp=2.0, Ki=1.0, Kd=0.0, dt=dt,
                            u_min=-1, u_max=1, anti_windup=aw)
        r_large = 5.0 * np.ones_like(t)  # large step to cause saturation
        res = simulate_closed_loop(pid, K0, tau, L, r_large, dt)
        y = res["y"]
        mp = (np.max(y) / 5.0 - 1.0) * 100
        print(f"  Anti-windup={'On' if aw else 'Off':3s}: "
              f"overshoot={mp:5.1f}%, max|u|={np.max(np.abs(res['u'])):.2f}")
