"""
Control Theory — Lesson 1: Introduction to Control Systems

Demonstrates:
1. Open-loop vs closed-loop behavior
2. Feedback effect on disturbance rejection
3. Sensitivity reduction with feedback
4. First-order plant step response comparison
"""
import numpy as np


# ── 1. First-Order Plant Simulation ────────────────────────────────────

def simulate_first_order(K: float, tau: float, dt: float,
                         t_end: float, u_func=None) -> dict:
    """
    Simulate dy/dt = (-y + K*u) / tau using Euler integration.

    Args:
        K: DC gain
        tau: time constant [s]
        dt: time step [s]
        t_end: simulation end time [s]
        u_func: input function u(t), defaults to unit step

    Returns:
        dict with 't', 'y', 'u' arrays
    """
    if u_func is None:
        u_func = lambda t: 1.0  # unit step

    steps = int(t_end / dt)
    t = np.linspace(0, t_end, steps)
    y = np.zeros(steps)
    u = np.zeros(steps)

    for i in range(1, steps):
        u[i - 1] = u_func(t[i - 1])
        y[i] = y[i - 1] + dt * (-y[i - 1] + K * u[i - 1]) / tau

    u[-1] = u_func(t[-1])
    return {"t": t, "y": y, "u": u}


# ── 2. Open-Loop Controller ───────────────────────────────────────────

def open_loop_response(plant_K: float, plant_tau: float,
                       reference: float, disturbance: float = 0.0,
                       dt: float = 0.01, t_end: float = 10.0) -> dict:
    """
    Open-loop system: controller output = reference / plant_K (feedforward).
    Disturbance enters at the plant output additively.
    """
    u_ff = reference / plant_K  # feedforward command

    def u_func(t):
        return u_ff

    result = simulate_first_order(plant_K, plant_tau, dt, t_end, u_func)
    # Add disturbance after t = t_end/2
    for i, ti in enumerate(result["t"]):
        if ti >= t_end / 2:
            result["y"][i] += disturbance
    return result


# ── 3. Closed-Loop (Proportional Feedback) ────────────────────────────

def closed_loop_response(plant_K: float, plant_tau: float,
                         Kp: float, reference: float,
                         disturbance: float = 0.0,
                         dt: float = 0.01, t_end: float = 10.0) -> dict:
    """
    Closed-loop with proportional controller: u = Kp * (r - y).
    Disturbance enters at the plant output.
    """
    steps = int(t_end / dt)
    t = np.linspace(0, t_end, steps)
    y = np.zeros(steps)
    u = np.zeros(steps)
    e = np.zeros(steps)

    for i in range(1, steps):
        # Apply disturbance after midpoint
        d = disturbance if t[i - 1] >= t_end / 2 else 0.0
        e[i - 1] = reference - y[i - 1]
        u[i - 1] = Kp * e[i - 1]
        dydt = (-y[i - 1] + plant_K * u[i - 1] + d) / plant_tau
        y[i] = y[i - 1] + dt * dydt

    e[-1] = reference - y[-1]
    u[-1] = Kp * e[-1]
    return {"t": t, "y": y, "u": u, "e": e}


# ── 4. Sensitivity Analysis ───────────────────────────────────────────

def sensitivity_function(plant_K: float, Kp: float) -> dict:
    """
    For a proportional controller with first-order plant:
      T(s) = Kp*K / (tau*s + 1 + Kp*K)     (closed-loop)
      S(s) = (tau*s + 1) / (tau*s + 1 + Kp*K)  (sensitivity)

    At DC (s=0):
      T(0) = Kp*K / (1 + Kp*K)
      S(0) = 1 / (1 + Kp*K)

    Returns DC values and loop gain.
    """
    L = Kp * plant_K  # loop gain at DC
    T_dc = L / (1 + L)
    S_dc = 1 / (1 + L)
    return {
        "loop_gain": L,
        "T_dc": T_dc,
        "S_dc": S_dc,
        "steady_state_error": 1 - T_dc,
    }


# ── 5. System Classification ──────────────────────────────────────────

def classify_system(poles: list[complex]) -> dict:
    """
    Classify a system based on its pole locations.

    Returns:
        dict with 'order', 'type', 'stable', 'dominant_pole'
    """
    order = len(poles)
    stable = all(p.real < 0 for p in poles)
    marginally_stable = (not stable and
                         all(p.real <= 0 for p in poles) and
                         any(abs(p.real) < 1e-10 for p in poles))

    # Dominant pole: the one closest to the imaginary axis (largest real part)
    dominant = max(poles, key=lambda p: p.real)

    if marginally_stable:
        stability = "marginally stable"
    elif stable:
        stability = "stable"
    else:
        stability = "unstable"

    return {
        "order": order,
        "stable": stable,
        "stability": stability,
        "dominant_pole": dominant,
        "natural_frequencies": [abs(p) for p in poles],
    }


# ── Demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    plant_K = 2.0
    plant_tau = 1.0
    ref = 1.0
    dist = 0.3

    # Open-loop with disturbance
    ol = open_loop_response(plant_K, plant_tau, ref, dist)
    ol_final = ol["y"][-1]
    print("=== Open-Loop vs Closed-Loop ===")
    print(f"  Reference: {ref}")
    print(f"  Disturbance: {dist} (applied at t={5.0:.1f}s)")
    print(f"  Open-loop final output: {ol_final:.3f}")

    # Closed-loop with different gains
    for Kp in [1.0, 5.0, 20.0]:
        cl = closed_loop_response(plant_K, plant_tau, Kp, ref, dist)
        sens = sensitivity_function(plant_K, Kp)
        print(f"\n  Kp = {Kp:.1f}:")
        print(f"    Closed-loop final output: {cl['y'][-1]:.4f}")
        print(f"    Steady-state error (no dist): {sens['steady_state_error']:.4f}")
        print(f"    Sensitivity S(0): {sens['S_dc']:.4f}")
        print(f"    Disturbance effect: {abs(cl['y'][-1] - ref):.4f}")

    # System classification
    print("\n=== System Classification ===")
    test_poles = [
        ([-1, -2, -3], "3 real negative poles"),
        ([-1 + 2j, -1 - 2j], "complex conjugate pair"),
        ([0, -1], "integrator + stable pole"),
        ([1, -2], "one unstable pole"),
    ]
    for poles, desc in test_poles:
        poles_c = [complex(p) for p in poles]
        info = classify_system(poles_c)
        print(f"\n  {desc}: {poles}")
        print(f"    Order: {info['order']}")
        print(f"    Stability: {info['stability']}")
        print(f"    Dominant pole: {info['dominant_pole']}")
