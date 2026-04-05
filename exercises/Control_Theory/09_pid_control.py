"""
Exercises for Lesson 09: PID Control
Topic: Control_Theory
Solutions to practice problems from the lesson.
"""

import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def exercise_1():
    """
    Exercise 1: PID Effect Analysis
    Unity-feedback, plant Gp(s) = 1/(s+1)
    """
    print("Plant: Gp(s) = 1/(s+1)")

    # Part 1: P control with Kp = 10
    print("\nPart 1: P control (Kp = 10)")
    Kp = 10
    # T(s) = Kp*Gp / (1 + Kp*Gp) = Kp/(s+1+Kp) = 10/(s+11)
    num_p = [Kp]
    den_p = [1, 1 + Kp]
    sys_p = signal.TransferFunction(num_p, den_p)

    print(f"  T(s) = {Kp}/(s + {1+Kp})")

    # Steady-state step error
    dc_gain = Kp / (1 + Kp)
    ess = 1 - dc_gain
    print(f"  DC gain = {Kp}/{1+Kp} = {dc_gain:.4f}")
    print(f"  Steady-state step error = 1 - {dc_gain:.4f} = {ess:.4f} ({ess*100:.1f}%)")

    # Overshoot: first-order system -> no overshoot
    print(f"  System is first-order => M_p = 0% (no overshoot)")
    print(f"  Time constant = 1/{1+Kp} = {1/(1+Kp):.4f} s")

    # Part 2: PI control (Kp = 10, Ti = 2)
    print(f"\nPart 2: PI control (Kp = {Kp}, Ti = 2)")
    Ti = 2.0
    Ki = Kp / Ti  # = 5
    print(f"  Ki = Kp/Ti = {Ki}")

    # Gc(s) = Kp(1 + 1/(Ti*s)) = Kp*(Ti*s + 1)/(Ti*s) = 10(2s+1)/(2s)
    # Open-loop: Gc*Gp = 10(2s+1) / [2s(s+1)] = 5(2s+1)/[s(s+1)]
    # Closed-loop: T(s) = 10(2s+1) / [2s(s+1) + 10(2s+1)]
    #                    = 10(2s+1) / [2s^2 + 2s + 20s + 10]
    #                    = 10(2s+1) / [2s^2 + 22s + 10]
    #                    = 5(2s+1) / [s^2 + 11s + 5]
    num_pi = [Kp * Ti, Kp]       # 10*2s + 10 = 20s + 10
    den_ol_pi = np.polymul([Ti, 0], [1, 1])  # 2s * (s+1) = 2s^2 + 2s
    den_cl_pi = np.polyadd(den_ol_pi, num_pi)  # 2s^2 + 2s + 20s + 10 = 2s^2 + 22s + 10

    # Normalize: divide by leading coeff
    num_cl_pi = num_pi / den_cl_pi[0]
    den_cl_pi_norm = den_cl_pi / den_cl_pi[0]

    sys_pi = signal.TransferFunction(num_pi, den_cl_pi)

    # DC gain
    dc_pi = np.polyval(num_pi, 0) / np.polyval(den_cl_pi, 0)
    print(f"  T(s) = (20s + 10) / (2s^2 + 22s + 10)")
    print(f"       = 5(2s+1) / (s^2 + 11s + 5)")
    print(f"  DC gain = 10/10 = {dc_pi:.4f}")
    print(f"  Steady-state step error = {1 - dc_pi:.4f} (integrator eliminates error)")

    # Find poles for overshoot
    poles_pi = np.roots(den_cl_pi)
    print(f"  Closed-loop poles: {np.round(poles_pi, 4)}")

    # Step response
    t = np.linspace(0, 3, 1000)
    t_pi, y_pi = signal.step(sys_pi, T=t)
    Mp_pi = (np.max(y_pi) - dc_pi) / dc_pi * 100
    print(f"  Peak overshoot: {Mp_pi:.1f}%")

    # Part 3: PID control (add Td = 0.1)
    print(f"\nPart 3: PID control (Kp = {Kp}, Ti = {Ti}, Td = 0.1)")
    Td = 0.1
    Kd = Kp * Td  # = 1

    # Gc(s) = Kp(1 + 1/(Ti*s) + Td*s) = Kp*(Td*Ti*s^2 + Ti*s + 1)/(Ti*s)
    # = 10*(0.2*s^2 + 2s + 1)/(2s)
    # Open-loop: Gc*Gp = 10*(0.2s^2+2s+1) / [2s(s+1)]
    # Closed-loop: T = num / (den_ol + num)
    num_pid = [Kp*Td*Ti, Kp*Ti, Kp]  # 10*0.1*2*s^2 + 10*2*s + 10 = 2s^2 + 20s + 10
    den_ol_pid = np.polymul([Ti, 0], [1, 1])  # 2s^2 + 2s

    # Need to pad for addition
    den_ol_padded = np.zeros(len(num_pid))
    den_ol_padded[-len(den_ol_pid):] = den_ol_pid

    den_cl_pid = den_ol_padded + num_pid  # 4s^2 + 22s + 10

    sys_pid = signal.TransferFunction(num_pid, den_cl_pid)

    t_pid, y_pid = signal.step(sys_pid, T=t)
    dc_pid = np.polyval(num_pid, 0) / np.polyval(den_cl_pid, 0)
    Mp_pid = (np.max(y_pid) - dc_pid) / dc_pid * 100 if np.max(y_pid) > dc_pid else 0

    poles_pid = np.roots(den_cl_pid)
    print(f"  T(s) numerator: {num_pid}")
    print(f"  T(s) denominator: {den_cl_pid}")
    print(f"  DC gain = {dc_pid:.4f}")
    print(f"  Closed-loop poles: {np.round(poles_pid, 4)}")
    print(f"  Peak overshoot: {Mp_pid:.1f}%")
    print(f"  Derivative action reduces overshoot from {Mp_pi:.1f}% to {Mp_pid:.1f}%")

    # Plot comparison
    t_p, y_p = signal.step(sys_p, T=t)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_p, y_p, 'b-', linewidth=2, label=f'P (Kp={Kp})')
    ax.plot(t_pi, y_pi, 'r-', linewidth=2, label=f'PI (Kp={Kp}, Ti={Ti})')
    ax.plot(t_pid, y_pid, 'g-', linewidth=2, label=f'PID (Kp={Kp}, Ti={Ti}, Td={Td})')
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Setpoint')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Response')
    ax.set_title('Comparison: P vs PI vs PID Control')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/exercises/Control_Theory/ex09_pid_comparison.png',
                dpi=100)
    plt.close()
    print("  Comparison plot saved to 'ex09_pid_comparison.png'")


def exercise_2():
    """
    Exercise 2: Ziegler-Nichols Tuning
    Step response parameters: L = 0.5 s, T = 3 s, K0 = 2
    """
    L = 0.5   # delay time
    T = 3.0   # time constant
    K0 = 2.0  # DC gain

    print(f"Plant step response: L = {L} s, T = {T} s, K0 = {K0}")

    # Part 1: ZN Open-loop PID
    print(f"\nPart 1: Ziegler-Nichols open-loop PID tuning")
    Kp_pid = 1.2 * T / (K0 * L)
    Ti_pid = 2 * L
    Td_pid = 0.5 * L
    Ki_pid = Kp_pid / Ti_pid
    Kd_pid = Kp_pid * Td_pid

    print(f"  Kp = 1.2 * T / (K0 * L) = 1.2 * {T} / ({K0} * {L}) = {Kp_pid:.2f}")
    print(f"  Ti = 2L = 2 * {L} = {Ti_pid:.2f} s")
    print(f"  Td = 0.5L = 0.5 * {L} = {Td_pid:.2f} s")
    print(f"  Ki = Kp/Ti = {Ki_pid:.2f}")
    print(f"  Kd = Kp*Td = {Kd_pid:.2f}")

    # Part 2: ZN Open-loop PI
    print(f"\nPart 2: Ziegler-Nichols open-loop PI tuning")
    Kp_pi = 0.9 * T / (K0 * L)
    Ti_pi = L / 0.3
    Ki_pi = Kp_pi / Ti_pi

    print(f"  Kp = 0.9 * T / (K0 * L) = 0.9 * {T} / ({K0} * {L}) = {Kp_pi:.2f}")
    print(f"  Ti = L / 0.3 = {L} / 0.3 = {Ti_pi:.4f} s")
    print(f"  Ki = Kp/Ti = {Ki_pi:.4f}")

    # Part 3: IMC tuning with lambda = 1
    print(f"\nPart 3: IMC PI tuning (lambda = 1 s)")
    lam = 1.0
    tau = T  # plant time constant

    Kp_imc = tau / (K0 * (lam + L))
    Ti_imc = tau  # = T
    Ki_imc = Kp_imc / Ti_imc

    print(f"  Kp = tau / (K0 * (lambda + L)) = {tau} / ({K0} * ({lam} + {L})) = {Kp_imc:.4f}")
    print(f"  Ti = tau = {Ti_imc:.2f} s")
    print(f"  Ki = Kp/Ti = {Ki_imc:.4f}")

    # Comparison
    print(f"\nComparison:")
    print(f"  {'Method':<20s} {'Kp':>8s} {'Ti':>8s} {'Ki':>8s}")
    print(f"  {'ZN Open-Loop PID':<20s} {Kp_pid:8.3f} {Ti_pid:8.3f} {Ki_pid:8.3f}")
    print(f"  {'ZN Open-Loop PI':<20s} {Kp_pi:8.3f} {Ti_pi:8.3f} {Ki_pi:8.3f}")
    print(f"  {'IMC (lambda=1)':<20s} {Kp_imc:8.3f} {Ti_imc:8.3f} {Ki_imc:8.3f}")
    print()
    print("  IMC gives much more conservative (lower gain) tuning.")
    print("  ZN is aggressive (expects ~25% overshoot).")
    print("  IMC provides a single tuning knob (lambda) for performance/robustness tradeoff.")


def exercise_3():
    """
    Exercise 3: Anti-Windup
    PI controller: Kp = 5, Ki = 10
    Actuator saturation at +/- 1
    Step reference of magnitude 2
    """
    Kp = 5
    Ki = 10
    u_max = 1.0
    r = 2.0

    print(f"PI controller: Kp = {Kp}, Ki = {Ki}")
    print(f"Actuator saturation: +/- {u_max}")
    print(f"Step reference magnitude: {r}")

    print(f"\nWhy windup occurs:")
    print(f"  1. At t=0+, the step reference creates error e = {r}")
    print(f"  2. P action: u_P = Kp * e = {Kp} * {r} = {Kp*r}")
    print(f"  3. Even without integral, u_P = {Kp*r} >> u_max = {u_max}")
    print(f"     So the actuator immediately saturates at {u_max}")
    print(f"  4. The integral term accumulates: integral += Ki * e * dt")
    print(f"     Since the actual plant input is clamped at {u_max}, the plant")
    print(f"     responds slowly, and the error remains large")
    print(f"  5. The integral accumulates a large value over time")
    print(f"  6. When the output finally approaches the setpoint (or overshoots),")
    print(f"     e changes sign, but the accumulated integral is so large that")
    print(f"     the controller output remains saturated for a long time")
    print(f"  7. This causes excessive overshoot and long settling time")

    print(f"\nBack-calculation anti-windup:")
    print(f"  Idea: Feed back the difference between the desired controller")
    print(f"  output and the actual (saturated) actuator output to 'unwind'")
    print(f"  the integrator.")
    print(f"  Modified integral update:")
    print(f"    u_unsat = Kp*e + Ki*integral  (unsaturated controller output)")
    print(f"    u_sat = clip(u_unsat, -{u_max}, +{u_max})  (actual actuator input)")
    print(f"    integral += [Ki*e + (1/Tt)*(u_sat - u_unsat)] * dt")
    print(f"  where Tt is the tracking time constant (often = Ti = Kp/Ki = {Kp/Ki})")
    print(f"  When u_sat != u_unsat (saturation active), the extra term")
    print(f"  drives the integral back so that u_unsat approaches u_sat,")
    print(f"  preventing excessive accumulation.")

    # Simulate to demonstrate
    dt = 0.001
    T_sim = 10.0
    t = np.arange(0, T_sim, dt)
    N = len(t)
    Tt = Kp / Ki  # tracking time constant

    # Plant: simple integrator 1/s for illustration
    # dy/dt = u_actual
    y_no_aw = np.zeros(N)
    y_aw = np.zeros(N)
    u_no_aw = np.zeros(N)
    u_aw = np.zeros(N)
    integral_no_aw = 0.0
    integral_aw = 0.0

    for k in range(1, N):
        e = r - y_no_aw[k-1]
        integral_no_aw += Ki * e * dt
        u_unsat = Kp * e + integral_no_aw
        u_sat = np.clip(u_unsat, -u_max, u_max)
        u_no_aw[k] = u_sat
        y_no_aw[k] = y_no_aw[k-1] + u_sat * dt

    for k in range(1, N):
        e = r - y_aw[k-1]
        u_unsat = Kp * e + integral_aw
        u_sat = np.clip(u_unsat, -u_max, u_max)
        # Back-calculation: modify integral
        integral_aw += (Ki * e + (1/Tt) * (u_sat - u_unsat)) * dt
        u_aw[k] = u_sat
        y_aw[k] = y_aw[k-1] + u_sat * dt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(t, y_no_aw, 'r-', linewidth=2, label='Without anti-windup')
    ax1.plot(t, y_aw, 'b-', linewidth=2, label='With back-calculation')
    ax1.axhline(y=r, color='k', linestyle='--', alpha=0.5, label=f'Setpoint = {r}')
    ax1.set_ylabel('Output y(t)')
    ax1.set_title('Integrator Windup: PI Control with Saturation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, u_no_aw, 'r-', linewidth=2, label='Without anti-windup')
    ax2.plot(t, u_aw, 'b-', linewidth=2, label='With back-calculation')
    ax2.axhline(y=u_max, color='k', linestyle='--', alpha=0.3)
    ax2.axhline(y=-u_max, color='k', linestyle='--', alpha=0.3)
    ax2.set_ylabel('Control signal u(t)')
    ax2.set_xlabel('Time (s)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/exercises/Control_Theory/ex09_antiwindup.png',
                dpi=100)
    plt.close()
    print("\n  Anti-windup simulation saved to 'ex09_antiwindup.png'")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: PID Effect Analysis ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Ziegler-Nichols Tuning ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Anti-Windup ===")
    print("=" * 60)
    exercise_3()

    print("\n\nAll exercises completed!")
