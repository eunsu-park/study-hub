"""
Faraday's Law: Electromagnetic Induction, Lenz's Law, and RL Circuits
=====================================================================

Topics covered:
  1. Moving conductor in magnetic field (motional EMF)
  2. Changing magnetic flux computation
  3. Lenz's law demonstration
  4. RL circuit transient response

Why Faraday's law is a turning point in electrodynamics:
  Electrostatics and magnetostatics are separate worlds: charges make E,
  currents make B, and neither changes. Faraday's law *couples* them:
  a CHANGING B field creates an E field (and thus EMF). This is the
  first step toward Maxwell's equations and electromagnetic waves.

Physics background:
  - Faraday's law:  EMF = -d(Phi_B)/dt  where Phi_B = integral(B . dA)
  - Motional EMF:   EMF = integral((v x B) . dl)  for moving conductor
  - Lenz's law:     induced current opposes the change in flux
  - RL circuit:     L * dI/dt + R * I = EMF  (exponential decay/growth)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# Physical constants
MU0 = 4 * np.pi * 1e-7


# ===========================
# 1. Motional EMF: Sliding Rail
# ===========================

def motional_emf_demo():
    """
    Simulate a conducting bar sliding on parallel rails in a uniform B field.

    Geometry:
      - Uniform B = B0 * z_hat (out of page)
      - Rails along x-axis, separated by distance L
      - Bar slides along y-axis with velocity v(t)
      - Area A(t) = L * y(t) increases with time

    Why this is the simplest Faraday's law example:
      The flux Phi = B * L * y changes because the area changes
      (the bar moves), not because B changes. The induced EMF is:
        EMF = -d(Phi)/dt = -B * L * dy/dt = -B * L * v
      This is the "motional EMF" -- the Lorentz force v x B on charges
      in the moving bar creates a potential difference.
    """
    B0 = 0.5      # T (magnetic field strength)
    L = 0.3       # m (rail separation)
    v0 = 2.0      # m/s (constant bar velocity)
    R_load = 1.0  # Ohm (resistance in circuit)

    t = np.linspace(0, 5, 500)
    y = v0 * t                    # bar position
    Phi = B0 * L * y              # magnetic flux
    EMF = -B0 * L * v0 * np.ones_like(t)  # constant for constant v
    I_induced = EMF / R_load      # Ohm's law

    # Why is the current negative?
    #   Lenz's law: the induced current creates a magnetic field that
    #   opposes the increase in flux. Since flux is increasing (bar
    #   moves out), the induced B must point into the page, requiring
    #   a clockwise current (negative in our convention).

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(t, y, 'b-', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Bar position y (m)')
    ax.set_title('Bar Position')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, Phi, 'g-', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$\Phi_B$ (Wb)')
    ax.set_title('Magnetic Flux')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t, EMF, 'r-', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('EMF (V)')
    ax.set_title('Induced EMF (Faraday)')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t, I_induced * 1000, 'm-', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('I (mA)')
    ax.set_title('Induced Current')
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Motional EMF: B={B0} T, L={L} m, v={v0} m/s, R={R_load} Ohm',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig('05_motional_emf.png', dpi=150)
    plt.close()
    print("[Saved] 05_motional_emf.png")


# ===========================
# 2. Changing Flux: Time-Varying B Field
# ===========================

def changing_flux_demo():
    """
    Compute EMF from a time-varying B field through a fixed loop.

    B(t) = B0 * sin(omega * t)  through a circular loop of radius a.

    Why both motional and changing-B EMFs?
      Faraday's law is general: EMF = -dPhi/dt. The flux can change
      because the area changes (motional), or because B changes
      (transformer EMF), or both. Here we demonstrate the transformer
      case, which is the operating principle of every transformer and
      the basis for electromagnetic wave generation.
    """
    B0 = 0.1         # T
    omega = 2 * np.pi * 50  # 50 Hz
    a = 0.05          # loop radius (m)
    A_loop = np.pi * a**2

    t = np.linspace(0, 0.06, 1000)  # 3 cycles at 50 Hz
    B = B0 * np.sin(omega * t)
    Phi = B * A_loop

    # EMF = -dPhi/dt = -B0 * A * omega * cos(omega*t)
    EMF_analytic = -B0 * A_loop * omega * np.cos(omega * t)

    # Also compute EMF numerically from finite differences
    # Why compare analytic and numerical?
    #   This validates our understanding and demonstrates that
    #   dPhi/dt computed via finite differences agrees with the
    #   exact derivative.
    EMF_numerical = -np.gradient(Phi, t)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    ax = axes[0]
    ax.plot(t * 1000, B * 1000, 'b-', linewidth=2)
    ax.set_ylabel('B (mT)')
    ax.set_title('Time-Varying Magnetic Field')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t * 1000, Phi * 1e6, 'g-', linewidth=2)
    ax.set_ylabel(r'$\Phi$ ($\mu$Wb)')
    ax.set_title('Magnetic Flux Through Loop')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(t * 1000, EMF_analytic * 1000, 'r-', linewidth=2, label='Analytic')
    ax.plot(t * 1000, EMF_numerical * 1000, 'k--', linewidth=1, label='Numerical')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('EMF (mV)')
    ax.set_title(r'Induced EMF: $\mathcal{E} = -d\Phi/dt$')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Why annotate phase relationship?
    #   The EMF is 90 degrees out of phase with B (cosine vs sine).
    #   When B is at maximum, dB/dt = 0 so EMF = 0.
    #   When B passes through zero, dB/dt is maximum so |EMF| is maximum.
    #   This phase relationship is crucial for AC circuit analysis.
    ax.annotate('EMF peaks when B = 0\n(maximum dB/dt)',
                xy=(5, EMF_analytic[int(0.005 / 0.06 * 1000)] * 1000),
                xytext=(15, max(EMF_analytic) * 1000 * 0.7),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))

    plt.tight_layout()
    plt.savefig('05_changing_flux.png', dpi=150)
    plt.close()
    print("[Saved] 05_changing_flux.png")


# ===========================
# 3. Lenz's Law Demonstration
# ===========================

def lenz_law_demo():
    """
    Demonstrate Lenz's law: induced currents oppose flux changes.

    Scenario: A circular loop with resistance R in a B field that
    increases linearly, then is held constant, then decreases.

    Why is Lenz's law important?
      Without Lenz's law (the minus sign in Faraday's law), energy
      would not be conserved. If the induced current *reinforced* the
      flux change, it would create a runaway positive feedback: more
      flux -> more current -> more flux -> ... This would be a free
      energy machine, violating thermodynamics. The minus sign ensures
      the system is self-regulating.
    """
    R_circuit = 2.0     # Ohms
    A_loop = 0.01       # m^2

    # B(t): ramp up, hold, ramp down
    t = np.linspace(0, 3, 1000)
    B = np.piecewise(t,
                     [t < 1, (t >= 1) & (t < 2), t >= 2],
                     [lambda t: 0.5 * t,        # ramp up
                      lambda t: 0.5,             # hold
                      lambda t: 0.5 * (3 - t)])  # ramp down

    Phi = B * A_loop
    EMF = -np.gradient(Phi, t)
    I = EMF / R_circuit

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    ax = axes[0]
    ax.plot(t, B * 1000, 'b-', linewidth=2)
    ax.set_ylabel('B (mT)')
    ax.set_title('External Magnetic Field')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t, Phi * 1e6, 'g-', linewidth=2)
    ax.set_ylabel(r'$\Phi$ ($\mu$Wb)')
    ax.set_title('Magnetic Flux')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(t, EMF * 1000, 'r-', linewidth=2)
    ax.set_ylabel('EMF (mV)')
    ax.set_title('Induced EMF')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    ax.fill_between(t, I * 1000, alpha=0.3, color='purple')
    ax.plot(t, I * 1000, 'm-', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('I (mA)')
    ax.set_title("Induced Current (Lenz's Law: opposes flux change)")
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Why annotate each phase?
    #   Lenz's law is best understood by seeing WHEN and WHY current flows.
    ax.annotate('B increasing\nI opposes (negative)', xy=(0.5, I[250] * 1000),
                fontsize=9, ha='center', color='blue')
    ax.annotate('B constant\nno EMF', xy=(1.5, 0), fontsize=9, ha='center',
                color='green')
    ax.annotate('B decreasing\nI supports (positive)', xy=(2.5, I[750] * 1000),
                fontsize=9, ha='center', color='red')

    plt.tight_layout()
    plt.savefig('05_lenz_law.png', dpi=150)
    plt.close()
    print("[Saved] 05_lenz_law.png")


# ===========================
# 4. RL Circuit Transient Response
# ===========================

def rl_circuit():
    """
    Solve the RL circuit differential equation:
      L * dI/dt + R * I = V(t)

    Case 1: Step voltage (switch closes at t=0)
    Case 2: Sinusoidal driving voltage

    Why RL circuits in an electrodynamics context?
      The inductor is where Faraday's law meets circuit theory.
      The self-induced EMF: EMF_L = -L * dI/dt opposes current changes.
      The time constant tau = L/R governs how fast currents build up
      or decay. This is the circuit-level manifestation of Lenz's law.
    """
    R = 10.0    # Ohms
    L = 0.1     # Henries
    tau = L / R  # Time constant

    print(f"\n  RL circuit: R = {R} Ohm, L = {L} H, tau = {tau*1000:.1f} ms")

    # --- Case 1: Step response ---
    V0 = 5.0  # Step voltage
    t = np.linspace(0, 5 * tau, 500)

    # Analytic solution: I(t) = (V0/R) * (1 - exp(-t/tau))
    # Why exponential approach?
    #   At t=0, the inductor acts like an open circuit (opposes sudden change).
    #   As t -> inf, the inductor acts like a short circuit (steady state).
    #   The transition time scale is tau = L/R.
    I_step = (V0 / R) * (1 - np.exp(-t / tau))

    # --- Case 2: Sinusoidal response ---
    omega = 2 * np.pi * 100  # 100 Hz

    def rl_ode(I, t, R, L, V0, omega):
        """RL circuit ODE: L*dI/dt + R*I = V0*sin(omega*t)"""
        V = V0 * np.sin(omega * t)
        dIdt = (V - R * I) / L
        return dIdt

    t_ac = np.linspace(0, 0.05, 1000)  # 5 cycles at 100 Hz
    I_ac = odeint(rl_ode, 0, t_ac, args=(R, L, V0, omega)).flatten()

    # Steady-state analytic solution for comparison
    # I_ss = (V0/Z) * sin(omega*t - phi)
    # where Z = sqrt(R^2 + (omega*L)^2), phi = arctan(omega*L/R)
    Z = np.sqrt(R**2 + (omega * L)**2)
    phi = np.arctan2(omega * L, R)
    I_ss = (V0 / Z) * np.sin(omega * t_ac - phi)

    # Why does the AC current lag the voltage?
    #   The inductor opposes changes in current. When voltage is rising,
    #   the inductor "resists" the increase, so current peaks later.
    #   The phase lag phi = arctan(omega*L/R) increases with frequency
    #   (more lag at higher frequencies, where changes are faster).

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Step response
    ax = axes[0]
    ax.plot(t * 1000, I_step * 1000, 'b-', linewidth=2, label='I(t)')
    ax.axhline(y=V0 / R * 1000, color='red', linestyle='--',
               label=f'Steady state = V/R = {V0/R*1000:.0f} mA')
    ax.axvline(x=tau * 1000, color='gray', linestyle=':', alpha=0.7,
               label=f'tau = L/R = {tau*1000:.1f} ms')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('I (mA)')
    ax.set_title('RL Circuit: Step Response')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # AC response
    ax = axes[1]
    ax.plot(t_ac * 1000, V0 * np.sin(omega * t_ac), 'gray', linewidth=1,
            alpha=0.5, label='V(t) / arbitrary scale')
    ax.plot(t_ac * 1000, I_ac * 1000, 'b-', linewidth=2, label='I(t) numerical')
    ax.plot(t_ac * 1000, I_ss * 1000, 'r--', linewidth=1, label='I(t) steady-state')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('I (mA)')
    ax.set_title(f'RL Circuit: AC Response (f=100 Hz, phase lag = {np.degrees(phi):.1f} deg)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('05_rl_circuit.png', dpi=150)
    plt.close()
    print("[Saved] 05_rl_circuit.png")


# ===========================
# Main
# ===========================

if __name__ == '__main__':
    print("=== Motional EMF ===")
    motional_emf_demo()

    print("\n=== Changing Flux ===")
    changing_flux_demo()

    print("\n=== Lenz's Law ===")
    lenz_law_demo()

    print("\n=== RL Circuit ===")
    rl_circuit()
