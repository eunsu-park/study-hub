"""
Exercises for Lesson 20: Power Management
Topic: Computer_Architecture

Solutions to practice problems covering CMOS power calculation,
DVFS energy analysis, dark silicon budget, C-state selection,
and data center PUE cost modeling.
"""


def exercise_1():
    """
    Power calculation: Given processor parameters, compute dynamic, static,
    and total power. Then recalculate after voltage/frequency reduction.
    """
    print("=== Exercise 1: Power Calculation ===\n")

    # Given parameters
    alpha = 0.2      # activity factor
    C = 20e-9        # 20 nF in farads
    V1 = 1.0         # supply voltage (V)
    f1 = 3.0e9       # 3.0 GHz in Hz
    I_leak = 10      # leakage current (A)

    # (a) Dynamic power: P_dyn = α × C × V² × f
    P_dyn1 = alpha * C * V1**2 * f1
    print(f"(a) Dynamic power at V={V1}V, f={f1/1e9}GHz:")
    print(f"    P_dyn = α·C·V²·f = {alpha}×{C*1e9}nF×{V1}²×{f1/1e9}GHz")
    print(f"    P_dyn = {P_dyn1:.1f} W")

    # (b) Static power: P_static = V_dd × I_leak
    P_static1 = V1 * I_leak
    print(f"\n(b) Static power:")
    print(f"    P_static = V×I_leak = {V1}×{I_leak} = {P_static1:.1f} W")

    # (c) Total power
    P_total1 = P_dyn1 + P_static1
    print(f"\n(c) Total power:")
    print(f"    P_total = {P_dyn1:.1f} + {P_static1:.1f} = {P_total1:.1f} W")

    # Reduced voltage/frequency
    V2 = 0.7
    f2 = 2.0e9  # 2.0 GHz
    I_leak2 = 5  # Leakage drops with voltage (approximately)

    P_dyn2 = alpha * C * V2**2 * f2
    P_static2 = V2 * I_leak2
    P_total2 = P_dyn2 + P_static2

    print(f"\nReduced: V={V2}V, f={f2/1e9}GHz, I_leak={I_leak2}A")
    print(f"    P_dyn    = {P_dyn2:.2f} W  (was {P_dyn1:.1f} W, "
          f"{(1-P_dyn2/P_dyn1)*100:.0f}% reduction)")
    print(f"    P_static = {P_static2:.2f} W  (was {P_static1:.1f} W)")
    print(f"    P_total  = {P_total2:.2f} W  (was {P_total1:.1f} W, "
          f"{(1-P_total2/P_total1)*100:.0f}% reduction)")
    print()


def exercise_2():
    """
    DVFS energy analysis: Compare two operating points for a fixed task.
    """
    print("=== Exercise 2: DVFS Energy Analysis ===\n")

    task_cycles = 10e9  # 10 billion cycles

    # Option A: 4 GHz, 1.2V, TDP = 100W dynamic
    fA = 4e9     # Hz
    VA = 1.2     # V
    P_dynA = 100 # W (given)

    timeA = task_cycles / fA
    energyA = P_dynA * timeA

    print("Option A: 4 GHz, 1.2V, P_dynamic = 100W")
    print(f"  Execution time = {task_cycles:.0e} / {fA:.0e} = {timeA:.2f} s")
    print(f"  Dynamic power  = {P_dynA} W")
    print(f"  Energy         = {P_dynA}W × {timeA:.2f}s = {energyA:.1f} J")

    # Option B: 2 GHz, 0.8V
    fB = 2e9
    VB = 0.8
    # Power scales as V²×f relative to Option A
    P_dynB = P_dynA * (VB/VA)**2 * (fB/fA)
    timeB = task_cycles / fB
    energyB = P_dynB * timeB

    print(f"\nOption B: 2 GHz, 0.8V")
    print(f"  Power scaling: P_B = P_A × (V_B/V_A)² × (f_B/f_A)")
    print(f"               = {P_dynA} × ({VB}/{VA})² × ({fB/1e9}/{fA/1e9})")
    print(f"               = {P_dynA} × {(VB/VA)**2:.3f} × {fB/fA:.2f}")
    print(f"               = {P_dynB:.1f} W")
    print(f"  Execution time = {timeB:.2f} s")
    print(f"  Energy         = {P_dynB:.1f}W × {timeB:.2f}s = {energyB:.1f} J")

    print(f"\nComparison:")
    print(f"  {'Metric':<20} {'Option A':<15} {'Option B':<15}")
    print(f"  {'Time':<20} {timeA:<15.2f} {timeB:<15.2f} s")
    print(f"  {'Power':<20} {P_dynA:<15.1f} {P_dynB:<15.1f} W")
    print(f"  {'Energy':<20} {energyA:<15.1f} {energyB:<15.1f} J")

    winner = "B" if energyB < energyA else "A"
    savings = abs(energyA - energyB) / max(energyA, energyB) * 100
    print(f"\n  Option {winner} is {savings:.0f}% more energy-efficient!")
    print(f"  Even though Option B is 2x slower, it uses {savings:.0f}% less energy")
    print(f"  because power drops with V² while time only doubles.")
    print()


def exercise_3():
    """
    Dark silicon budget: Calculate active transistor fraction at 5nm and 3nm.
    """
    print("=== Exercise 3: Dark Silicon Budget ===\n")

    # 5nm: 100 mm², 20B transistors, TDP=15W (mobile), 0.5 pW/transistor
    T1 = 20e9
    TDP1 = 15
    ppt1 = 0.5e-12  # picowatts to watts

    max_power1 = T1 * ppt1
    active1 = TDP1 / max_power1

    print("5nm Mobile SoC:")
    print(f"  Transistors: {T1/1e9:.0f} billion")
    print(f"  TDP: {TDP1} W")
    print(f"  Power if all active: {T1}×{ppt1*1e12}pW = {max_power1:.0f} W")
    print(f"  Active fraction: {TDP1}W / {max_power1:.0f}W = {active1*100:.1f}%")
    print(f"  Dark silicon: {(1-active1)*100:.1f}%")

    # 3nm: 1.5x density = 30B transistors, same TDP
    T2 = T1 * 1.5
    ppt2 = 0.35e-12  # Slightly lower per-transistor power at 3nm
    max_power2 = T2 * ppt2
    active2 = TDP1 / max_power2  # Same TDP

    print(f"\n3nm Mobile SoC (1.5x density):")
    print(f"  Transistors: {T2/1e9:.0f} billion")
    print(f"  TDP: {TDP1} W (same thermal budget)")
    print(f"  Power if all active: {max_power2:.0f} W")
    print(f"  Active fraction: {active2*100:.1f}%")
    print(f"  Dark silicon: {(1-active2)*100:.1f}%")

    print(f"\n  Dark silicon worsened from {(1-active1)*100:.0f}% to "
          f"{(1-active2)*100:.0f}%")

    print(f"\n3) Heterogeneous design proposal:")
    print(f"   Budget: {T2/1e9:.0f}B transistors, only {active2*100:.0f}% active")
    print(f"   Active budget: {T2*active2/1e9:.1f}B transistors at any time")
    print()
    print(f"   Proposed allocation:")
    print(f"   ┌────────────────────────────────────────────────┐")
    print(f"   │  High-perf cores (2): 2B transistors (active)  │")
    print(f"   │  Efficiency cores (4): 1B transistors (active) │")
    print(f"   │  GPU (16 cores): 4B transistors (on-demand)    │")
    print(f"   │  NPU (Neural): 3B transistors (on-demand)      │")
    print(f"   │  Media engine: 1B transistors (on-demand)      │")
    print(f"   │  Always-dark reserve: ~19B transistors          │")
    print(f"   └────────────────────────────────────────────────┘")
    print(f"   Only the needed accelerator is powered for each workload.")
    print()


def exercise_4():
    """
    C-state selection with break-even analysis.
    """
    print("=== Exercise 4: C-State Analysis ===\n")

    idle_ms = 8.0  # 8 ms idle period
    idle_us = idle_ms * 1000

    states = [
        {"name": "C1", "wake_us": 1, "save_pct": 30},
        {"name": "C3", "wake_us": 100, "save_pct": 70},
        {"name": "C6", "wake_us": 500, "save_pct": 95},
    ]

    print(f"Idle duration: {idle_ms} ms = {idle_us:.0f} μs\n")
    print(f"{'State':<6} {'Wake(μs)':<12} {'Saving%':<10} "
          f"{'Break-even':<15} {'Eligible?':<12} {'Net Savings'}")
    print("-" * 70)

    active_power = 100  # reference: 100W active
    best_net = -1
    best_state = None

    for s in states:
        # Break-even: idle must be > 2 × wake_latency to justify entry/exit
        break_even_us = s["wake_us"] * 2
        eligible = idle_us >= break_even_us

        if eligible:
            # Effective idle time minus transition overhead
            effective_idle = idle_us - s["wake_us"]
            # Power saved = save_pct of active power during effective idle
            saved_fraction = s["save_pct"] / 100
            # Net energy saved (normalized to active power × microseconds)
            net_saved = effective_idle * saved_fraction
        else:
            net_saved = 0

        be_str = f"{break_even_us:.0f} μs" if break_even_us < 1000 else f"{break_even_us/1000:.1f} ms"
        net_str = f"{net_saved:.0f} μs·%" if eligible else "N/A"

        print(f"{s['name']:<6} {s['wake_us']:<12} {s['save_pct']:<10}% "
              f"{be_str:<15} {'YES' if eligible else 'NO':<12} {net_str}")

        if eligible and net_saved > best_net:
            best_net = net_saved
            best_state = s

    print(f"\nOptimal choice: {best_state['name']}")
    print(f"  8ms idle >> {best_state['wake_us']}μs wake latency")
    print(f"  {best_state['save_pct']}% power savings during idle")
    print(f"  Wake overhead is only {best_state['wake_us']/idle_us*100:.1f}% of idle time")
    print()


def exercise_5():
    """
    PUE and total cost of ownership analysis.
    """
    print("=== Exercise 5: PUE and Total Cost ===\n")

    servers = 10000
    avg_power_w = 300
    cost_kwh = 0.10  # $/kWh
    hours_year = 8760

    # 1) Annual electricity cost at PUE=1.5
    pue1 = 1.5
    it_power_kw = servers * avg_power_w / 1000
    total_power_kw1 = it_power_kw * pue1
    annual_kwh1 = total_power_kw1 * hours_year
    annual_cost1 = annual_kwh1 * cost_kwh

    print(f"1) PUE = {pue1}")
    print(f"   IT power: {servers}×{avg_power_w}W = {it_power_kw:.0f} kW = {it_power_kw/1000:.1f} MW")
    print(f"   Total power: {it_power_kw:.0f}×{pue1} = {total_power_kw1:.0f} kW = {total_power_kw1/1000:.1f} MW")
    print(f"   Annual energy: {total_power_kw1:.0f}×{hours_year} = {annual_kwh1:,.0f} kWh")
    print(f"   Annual cost: {annual_kwh1:,.0f}×${cost_kwh} = ${annual_cost1:,.0f}")

    # 2) Improved PUE=1.2
    pue2 = 1.2
    total_power_kw2 = it_power_kw * pue2
    annual_kwh2 = total_power_kw2 * hours_year
    annual_cost2 = annual_kwh2 * cost_kwh
    savings = annual_cost1 - annual_cost2

    print(f"\n2) PUE improved to {pue2}")
    print(f"   Total power: {total_power_kw2:.0f} kW")
    print(f"   Annual cost: ${annual_cost2:,.0f}")
    print(f"   Annual savings: ${annual_cost1:,.0f} - ${annual_cost2:,.0f} = ${savings:,.0f}")

    # 3) Break-even capital investment
    print(f"\n3) Break-even analysis (3-year payback)")
    max_investment = savings * 3
    print(f"   Annual savings: ${savings:,.0f}")
    print(f"   3-year savings: ${max_investment:,.0f}")
    print(f"   Maximum justified investment: ${max_investment:,.0f}")
    print(f"   This could fund: liquid cooling, hot/cold aisle containment,")
    print(f"   efficient UPS, free cooling, or building design improvements.")
    print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
