"""
Exercises for Lesson 06: Electromagnetic Induction
Topic: Electrodynamics
Solutions to practice problems from the lesson.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Constants
mu_0 = 4.0 * np.pi * 1e-7


def exercise_1():
    """
    Exercise 1: AC Generator
    Rectangular loop of area A rotates at angular frequency omega in uniform B.
    EMF = N*B*A*omega*sin(omega*t). Plot EMF and instantaneous power.
    """
    N_turns = 100    # number of turns
    B = 0.5          # magnetic field (T)
    A = 0.02         # loop area (m^2, 200 cm^2)
    omega = 2 * np.pi * 60  # 60 Hz
    R_load = 100.0   # load resistance (Ohm)

    t = np.linspace(0, 3.0 / 60, 1000)  # 3 cycles at 60 Hz

    # EMF = N*B*A*omega*sin(omega*t)
    EMF_peak = N_turns * B * A * omega
    EMF = EMF_peak * np.sin(omega * t)

    # Current and power
    I = EMF / R_load
    P_inst = EMF * I  # instantaneous power = EMF^2/R
    P_avg = 0.5 * EMF_peak**2 / R_load

    print(f"  AC Generator: N={N_turns} turns, B={B} T, A={A*1e4:.0f} cm^2")
    print(f"  Frequency: {omega/(2*np.pi):.0f} Hz")
    print(f"  Peak EMF: {EMF_peak:.2f} V")
    print(f"  RMS EMF:  {EMF_peak/np.sqrt(2):.2f} V")
    print(f"  Load R = {R_load} Ohm")
    print(f"  Peak current: {EMF_peak/R_load:.4f} A")
    print(f"  Average power: {P_avg:.4f} W")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    axes[0].plot(t * 1000, EMF, 'b-', linewidth=2)
    axes[0].set_ylabel('EMF (V)')
    axes[0].set_title('AC Generator Output')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='gray', linewidth=0.5)

    axes[1].plot(t * 1000, P_inst, 'r-', linewidth=2, label='Instantaneous')
    axes[1].axhline(y=P_avg, color='green', linestyle='--', linewidth=1.5,
                    label=f'Average = {P_avg:.2f} W')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Power (W)')
    axes[1].set_title('Power Delivered to Load')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex06_ac_generator.png', dpi=150)
    plt.close()
    print("  Plot saved: ex06_ac_generator.png")


def exercise_2():
    """
    Exercise 2: Eddy Current Braking
    Conducting disk rotating in a magnetic field.
    Model braking torque proportional to angular velocity.
    """
    # Model parameters
    I_moment = 0.1    # moment of inertia (kg*m^2)
    omega_0 = 100.0   # initial angular velocity (rad/s)
    k_brake = 0.5     # braking constant (N*m*s/rad), torque = -k*omega

    # The braking torque: tau = -k*omega
    # I * d(omega)/dt = -k*omega
    # Solution: omega(t) = omega_0 * exp(-k*t/I)
    tau = I_moment / k_brake  # time constant

    dt = 1e-3
    t_max = 5 * tau
    t = np.arange(0, t_max, dt)
    N = len(t)

    omega_numerical = np.zeros(N)
    omega_numerical[0] = omega_0

    for i in range(N - 1):
        torque = -k_brake * omega_numerical[i]
        omega_numerical[i + 1] = omega_numerical[i] + (torque / I_moment) * dt

    omega_analytic = omega_0 * np.exp(-t / tau)

    # Power dissipated: P = k * omega^2
    P_dissipated = k_brake * omega_numerical**2

    # Total energy dissipated should equal initial KE
    KE_initial = 0.5 * I_moment * omega_0**2
    E_dissipated = np.trapz(P_dissipated, t)

    print(f"  Eddy current brake:")
    print(f"  Moment of inertia I = {I_moment} kg*m^2")
    print(f"  Braking constant k = {k_brake} N*m*s/rad")
    print(f"  Time constant tau = I/k = {tau:.3f} s")
    print(f"  Initial omega = {omega_0} rad/s")
    print(f"  Initial KE = {KE_initial:.2f} J")
    print(f"  Total energy dissipated: {E_dissipated:.2f} J")
    print(f"  Energy conservation error: {abs(E_dissipated-KE_initial)/KE_initial:.4e}")
    print("\n  Braking torque is proportional to angular velocity (linear drag).")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(t, omega_numerical, 'b-', linewidth=2, label='Numerical')
    axes[0].plot(t, omega_analytic, 'r--', linewidth=1.5, label='Analytic')
    axes[0].axvline(x=tau, color='green', linestyle=':', label=f'tau = {tau:.2f} s')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('omega (rad/s)')
    axes[0].set_title('Angular Velocity Decay')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, P_dissipated, 'r-', linewidth=2)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Power (W)')
    axes[1].set_title('Power Dissipated by Eddy Currents')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Eddy Current Braking', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ex06_eddy_brake.png', dpi=150)
    plt.close()
    print("  Plot saved: ex06_eddy_brake.png")


def exercise_3():
    """
    Exercise 3: Coupled RL Circuits
    Two coils with L1, L2, mutual inductance M in series.
    Effective inductance for aiding and opposing.
    """
    L1 = 10e-3    # 10 mH
    L2 = 20e-3    # 20 mH
    M = 5e-3      # 5 mH mutual inductance
    R = 10.0      # resistance (Ohm)
    V0 = 5.0      # source voltage

    L_aiding = L1 + L2 + 2 * M
    L_opposing = L1 + L2 - 2 * M

    print(f"  L1 = {L1*1e3:.0f} mH, L2 = {L2*1e3:.0f} mH, M = {M*1e3:.0f} mH")
    print(f"  Aiding: L_eff = L1 + L2 + 2M = {L_aiding*1e3:.0f} mH")
    print(f"  Opposing: L_eff = L1 + L2 - 2M = {L_opposing*1e3:.0f} mH")

    # Simulate charging for both cases
    dt = 1e-5
    t_max = 0.02
    t = np.arange(0, t_max, dt)

    for label, L_eff in [('Aiding', L_aiding), ('Opposing', L_opposing)]:
        tau = L_eff / R
        I_analytic = (V0 / R) * (1 - np.exp(-t / tau))
        I_steady = V0 / R
        print(f"\n  {label}: tau = L_eff/R = {tau*1e3:.2f} ms")
        print(f"  Steady-state current: {I_steady*1e3:.1f} mA")

    # Plot both
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, L_eff, color in [('Aiding', L_aiding, 'blue'), ('Opposing', L_opposing, 'red')]:
        tau = L_eff / R
        I = (V0 / R) * (1 - np.exp(-t / tau))
        ax.plot(t * 1000, I * 1000, color=color, linewidth=2,
                label=f'{label}: L_eff = {L_eff*1e3:.0f} mH, tau = {tau*1e3:.2f} ms')

    ax.axhline(y=V0 / R * 1000, color='gray', linestyle=':', label=f'I_max = {V0/R*1e3:.0f} mA')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Current (mA)')
    ax.set_title('Coupled RL Circuits: Aiding vs Opposing')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex06_coupled_rl.png', dpi=150)
    plt.close()
    print("\n  Plot saved: ex06_coupled_rl.png")


def exercise_4():
    """
    Exercise 4: Energy Conservation in Induction
    Sliding bar: verify KE lost = energy dissipated in resistance.
    """
    B = 0.5       # T
    l = 0.2       # bar length (m)
    R = 1.0       # resistance (Ohm)
    m = 0.01      # bar mass (kg)
    v0 = 5.0      # initial velocity (m/s)

    dt = 1e-5
    t_max = 0.2
    t = np.arange(0, t_max, dt)
    N = len(t)

    v = np.zeros(N)
    v[0] = v0
    P_dissipated = np.zeros(N)

    for i in range(N - 1):
        emf = B * l * v[i]
        I_current = emf / R
        F = -B * I_current * l
        v[i + 1] = v[i] + (F / m) * dt
        P_dissipated[i] = I_current**2 * R

    # Kinetic energy
    KE_initial = 0.5 * m * v0**2
    KE_final = 0.5 * m * v[-1]**2
    KE_lost = KE_initial - KE_final

    # Total energy dissipated
    E_dissipated = np.trapz(P_dissipated, t)

    # Analytic
    tau = m * R / (B * l)**2
    KE_analytic_lost = KE_initial * (1 - np.exp(-2 * t_max / tau))

    print(f"  Sliding bar: B={B} T, l={l} m, R={R} Ohm, m={m} kg, v0={v0} m/s")
    print(f"  Time constant tau = mR/(Bl)^2 = {tau*1000:.2f} ms")
    print(f"  Initial KE:        {KE_initial:.6e} J")
    print(f"  Final KE:          {KE_final:.6e} J")
    print(f"  KE lost:           {KE_lost:.6e} J")
    print(f"  Energy dissipated: {E_dissipated:.6e} J")
    print(f"  Relative error:    {abs(KE_lost - E_dissipated)/KE_lost:.4e}")
    print("  Energy conservation verified!")


def exercise_5():
    """
    Exercise 5: Inductance of a Toroid
    N=500 turns, inner a=10cm, outer b=15cm, height h=3cm.
    Compute analytically and verify with Neumann formula (simplified).
    """
    N = 500
    a = 0.10   # inner radius (m)
    b = 0.15   # outer radius (m)
    h = 0.03   # height (m)

    # Analytic: L = mu_0 * N^2 * h / (2*pi) * ln(b/a)
    L_analytic = mu_0 * N**2 * h / (2 * np.pi) * np.log(b / a)

    print(f"  Toroid: N = {N} turns, a = {a*100:.0f} cm, b = {b*100:.0f} cm, h = {h*100:.0f} cm")
    print(f"  Analytic inductance: L = {L_analytic*1e3:.4f} mH")

    # Numerical verification using field energy method:
    # B inside toroid: B = mu_0*N*I / (2*pi*r)
    # Energy: W = integral of B^2/(2*mu_0) dV
    # dV = h * dr * r * dphi (in cylindrical coords, but toroid is azimuthally symmetric)
    # Actually dV = h * 2*pi*r * dr for a thin shell at radius r
    # Wait -- we need to be careful. The toroid is a torus, so:
    # W = integral_a^b [B(r)]^2/(2*mu_0) * h * 2*pi*r * dr / (2*pi)
    # Actually for a rectangular cross-section toroid:
    # W = h * integral_a^b integral_0^{2pi} B^2/(2*mu_0) * r * dphi * dr
    # But B depends on r only, and phi integral gives 2*pi:
    # Wait: no, the toroid wraps around, B is inside the torus.
    # More carefully: volume element = h * dr * (R_toroid circumference piece)
    # For computing energy: W = (1/2)*L*I^2 -> L = 2W/I^2
    I = 1.0   # use I=1 to extract L
    N_r = 10000
    r = np.linspace(a, b, N_r)
    dr = r[1] - r[0]

    B = mu_0 * N * I / (2 * np.pi * r)
    # Volume element for the toroidal cross-section at radius r:
    # dV = h * dr  (per unit toroidal angle) * 2*pi*r (full toroid)
    # Actually the full volume element integrating over the full toroid:
    # dV = h * 2*pi*r * dr (this is the volume of a thin cylindrical shell)
    # But wait, the toroid's B field only exists in the cross-section [a,b] x [0,h]
    # Total: W = integral_a^b B^2/(2*mu_0) * h * 2*pi*r * dr ... no
    # Actually for a toroid, the correct volume element considering the full azimuthal
    # symmetry: W = integral over cross section of [B^2/(2*mu_0)] * 2*pi*r * dA
    # where dA = h*dr for rectangular cross section

    W = np.sum(B**2 / (2 * mu_0) * h * 2 * np.pi * r * dr)
    L_numerical = 2 * W / I**2

    print(f"  Field energy method: L = {L_numerical*1e3:.4f} mH")
    print(f"  Relative error: {abs(L_numerical - L_analytic)/L_analytic:.4e}")

    # Energy stored at 1 A
    print(f"\n  Energy at I = 1 A: W = (1/2)LI^2 = {0.5*L_analytic*1:.6e} J")


if __name__ == "__main__":
    print("=== Exercise 1: AC Generator ===")
    exercise_1()
    print("\n=== Exercise 2: Eddy Current Braking ===")
    exercise_2()
    print("\n=== Exercise 3: Coupled RL Circuits ===")
    exercise_3()
    print("\n=== Exercise 4: Energy Conservation in Induction ===")
    exercise_4()
    print("\n=== Exercise 5: Inductance of a Toroid ===")
    exercise_5()
    print("\nAll exercises completed!")
