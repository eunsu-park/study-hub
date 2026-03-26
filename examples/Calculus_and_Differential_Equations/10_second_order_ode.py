"""
Second-Order ODEs — Spring-Mass-Damper Systems

Demonstrates:
  - Spring-mass-damper simulation (underdamped, overdamped, critically damped)
  - Forced oscillation and resonance curve
  - Beat phenomenon (two close frequencies)
  - Phase plane analysis

Dependencies: numpy, matplotlib, scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# 1. Free Vibration: Damping Regimes
# ---------------------------------------------------------------------------
def spring_mass_damper(m, c, k, x0, v0, t_span, t_eval=None):
    """Solve m*x'' + c*x' + k*x = 0 as a first-order system.

    Rewrite as:
      x' = v
      v' = -(c/m)*v - (k/m)*x

    The behavior depends on the discriminant Delta = c^2 - 4mk:
      Delta > 0: overdamped (two real roots, no oscillation)
      Delta = 0: critically damped (fastest decay without oscillation)
      Delta < 0: underdamped (decaying oscillation)
    """
    if t_eval is None:
        t_eval = np.linspace(*t_span, 1000)

    def rhs(t, state):
        x, v = state
        return [v, -(c / m) * v - (k / m) * x]

    sol = solve_ivp(rhs, t_span, [x0, v0], t_eval=t_eval,
                    method="RK45", rtol=1e-10)
    return sol.t, sol.y[0], sol.y[1]


def plot_damping_regimes():
    """Compare the three damping regimes for a spring-mass system.

    We fix m = 1 and k = 4 (natural frequency omega_n = 2 rad/s),
    then vary the damping coefficient c to illustrate each regime.
    Critical damping occurs at c_crit = 2*sqrt(m*k) = 4.
    """
    m, k = 1.0, 4.0
    c_crit = 2 * np.sqrt(m * k)
    x0, v0 = 1.0, 0.0
    t_span = (0, 8)

    cases = [
        (0.5, "Underdamped (c=0.5)"),
        (c_crit, f"Critically damped (c={c_crit:.1f})"),
        (8.0, "Overdamped (c=8)"),
        (0.0, "Undamped (c=0)"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["C0", "C1", "C2", "C3"]

    for (c, label), color in zip(cases, colors):
        t, x, v = spring_mass_damper(m, c, k, x0, v0, t_span)
        ax.plot(t, x, "-", color=color, lw=2, label=label)

    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement x(t)")
    ax.set_title("Spring-Mass-Damper: Damping Regimes (m=1, k=4)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("10_damping_regimes.png", dpi=100)
    plt.close()
    print("[Saved] 10_damping_regimes.png")


# ---------------------------------------------------------------------------
# 2. Forced Oscillation
# ---------------------------------------------------------------------------
def forced_oscillation(m, c, k, F0, omega, x0, v0, t_span):
    """Solve m*x'' + c*x' + k*x = F0*cos(omega*t).

    The steady-state amplitude is:
      A(omega) = F0 / sqrt((k - m*omega^2)^2 + (c*omega)^2)

    When omega = omega_n = sqrt(k/m) and damping is small, the amplitude
    is maximized — this is resonance.
    """
    t_eval = np.linspace(*t_span, 2000)

    def rhs(t, state):
        x, v = state
        return [v, (F0 * np.cos(omega * t) - c * v - k * x) / m]

    sol = solve_ivp(rhs, t_span, [x0, v0], t_eval=t_eval,
                    method="RK45", rtol=1e-10)
    return sol.t, sol.y[0]


def plot_resonance_curve(m, c, k, F0):
    """Plot the steady-state amplitude as a function of driving frequency.

    The resonance peak occurs near omega_n = sqrt(k/m) and becomes
    sharper (and taller) as damping decreases.
    """
    omega_n = np.sqrt(k / m)
    omega = np.linspace(0.01, 3 * omega_n, 500)

    fig, ax = plt.subplots(figsize=(8, 5))

    for c_val in [0.2, 0.5, 1.0, 2.0, 4.0]:
        # Steady-state amplitude formula
        A = F0 / np.sqrt((k - m * omega ** 2) ** 2 + (c_val * omega) ** 2)
        ax.plot(omega / omega_n, A, lw=2, label=f"c = {c_val}")

    ax.axvline(1.0, color="gray", ls="--", alpha=0.5, label="omega/omega_n = 1")
    ax.set_xlabel("omega / omega_n")
    ax.set_ylabel("Steady-state amplitude A")
    ax.set_title(f"Resonance Curve (m={m}, k={k}, F0={F0})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("10_resonance_curve.png", dpi=100)
    plt.close()
    print("[Saved] 10_resonance_curve.png")


# ---------------------------------------------------------------------------
# 3. Beat Phenomenon
# ---------------------------------------------------------------------------
def plot_beats():
    """Demonstrate beats: superposition of two close frequencies.

    When an undamped oscillator (omega_n) is driven at omega close to
    omega_n, the solution exhibits "beats" — the amplitude modulates
    at the difference frequency (omega_n - omega) / 2.

    x(t) ~ [2F0/(m(omega_n^2 - omega^2))] * sin(delta_t/2) * sin(avg_t)
    where delta = omega_n - omega, avg = (omega_n + omega)/2.
    """
    m, k = 1.0, 100.0
    omega_n = np.sqrt(k / m)  # = 10 rad/s
    omega = 9.5                # close to omega_n
    F0 = 1.0
    c = 0.0  # undamped for clean beats

    t_span = (0, 40)
    t, x = forced_oscillation(m, c, k, F0, omega, 0, 0, t_span)

    # Envelope: modulation at beat frequency
    delta_omega = abs(omega_n - omega)
    envelope = 2 * F0 / (m * abs(omega_n ** 2 - omega ** 2))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, x, "b-", lw=0.8, label="x(t)")
    ax.plot(t, envelope * np.abs(np.sin(delta_omega * t / 2)),
            "r--", lw=1.5, label="Envelope", alpha=0.7)
    ax.plot(t, -envelope * np.abs(np.sin(delta_omega * t / 2)),
            "r--", lw=1.5, alpha=0.7)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement")
    ax.set_title(f"Beat Phenomenon: omega_n={omega_n:.1f}, omega_drive={omega}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("10_beats.png", dpi=100)
    plt.close()
    print("[Saved] 10_beats.png")


# ---------------------------------------------------------------------------
# 4. Phase Plane
# ---------------------------------------------------------------------------
def plot_phase_plane():
    """Plot the phase portrait (x vs v) for different initial conditions.

    For an undamped oscillator, phase trajectories are ellipses (energy
    conservation).  Damping causes spiral trajectories toward the origin.
    """
    m, k = 1.0, 4.0
    t_span = (0, 15)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, c, title in zip(axes, [0.0, 0.5],
                             ["Undamped (c=0): Ellipses",
                              "Underdamped (c=0.5): Spirals"]):
        for x0 in [0.5, 1.0, 1.5, 2.0]:
            for v0 in [0]:
                t, x, v = spring_mass_damper(m, c, k, x0, v0, t_span)
                ax.plot(x, v, lw=1.5, alpha=0.8)

        ax.set_xlabel("x (displacement)")
        ax.set_ylabel("v (velocity)")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("10_phase_plane.png", dpi=100)
    plt.close()
    print("[Saved] 10_phase_plane.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Second-Order ODE: Spring-Mass-Damper Systems")
    print("=" * 60)

    # --- Demo 1: Damping regimes ---
    print("\nDemo 1: Free vibration — damping regimes")
    plot_damping_regimes()

    # --- Demo 2: Forced oscillation ---
    print("\nDemo 2: Forced oscillation")
    m, k, F0 = 1.0, 4.0, 1.0
    omega_n = np.sqrt(k / m)
    print(f"  Natural frequency: omega_n = {omega_n:.2f} rad/s")

    # Near resonance
    t, x = forced_oscillation(m, 0.3, k, F0, omega_n, 0, 0, (0, 30))
    print(f"  Max amplitude at resonance: {np.max(np.abs(x)):.4f}")

    # --- Demo 3: Resonance curve ---
    print("\nDemo 3: Resonance curve")
    plot_resonance_curve(m, 0.5, k, F0)

    # --- Demo 4: Beats ---
    print("\nDemo 4: Beat phenomenon")
    plot_beats()

    # --- Demo 5: Phase plane ---
    print("\nDemo 5: Phase plane portraits")
    plot_phase_plane()
