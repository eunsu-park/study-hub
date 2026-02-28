"""
Power Management Simulator

Simulates key power management concepts:
1. CMOS power calculation (dynamic + static + short-circuit)
2. DVFS analysis (voltage/frequency scaling trade-offs)
3. C-state selection (idle power state optimization)
4. Dark silicon budget estimation
5. Data center PUE and cost modeling

All calculations use standard power equations from computer architecture.
"""

import math


# ========================================================================
# 1. CMOS Power Calculation
# ========================================================================

def calculate_cmos_power(alpha: float, capacitance_nf: float,
                         voltage_v: float, freq_ghz: float,
                         leakage_a: float,
                         short_circuit_fraction: float = 0.1) -> dict:
    """Calculate total CMOS power from its three components.

    P_total = P_dynamic + P_static + P_short_circuit

    Args:
        alpha: Activity factor (fraction of gates switching per cycle)
        capacitance_nf: Total load capacitance in nanofarads
        voltage_v: Supply voltage in volts
        freq_ghz: Clock frequency in GHz
        leakage_a: Leakage current in amperes
        short_circuit_fraction: Short-circuit power as fraction of dynamic

    Returns:
        Dict with dynamic, static, short-circuit, and total power in watts.
    """
    cap_f = capacitance_nf * 1e-9  # Convert nF to F
    freq_hz = freq_ghz * 1e9       # Convert GHz to Hz

    # P_dynamic = alpha * C * V^2 * f
    p_dynamic = alpha * cap_f * (voltage_v ** 2) * freq_hz

    # P_static = V_dd * I_leak
    p_static = voltage_v * leakage_a

    # P_short_circuit ≈ fraction of P_dynamic
    p_short_circuit = p_dynamic * short_circuit_fraction

    p_total = p_dynamic + p_static + p_short_circuit

    return {
        "dynamic_w": p_dynamic,
        "static_w": p_static,
        "short_circuit_w": p_short_circuit,
        "total_w": p_total,
    }


def demo_cmos_power():
    """Compare power at different voltage/frequency operating points."""
    print("=" * 65)
    print("1. CMOS Power Calculation")
    print("=" * 65)

    configs = [
        {"label": "High Perf", "alpha": 0.2, "cap": 20, "v": 1.0, "f": 3.0, "leak": 10},
        {"label": "Reduced V", "alpha": 0.2, "cap": 20, "v": 0.7, "f": 2.0, "leak": 5},
        {"label": "Low Power", "alpha": 0.15, "cap": 20, "v": 0.6, "f": 1.0, "leak": 3},
    ]

    print(f"\n{'Config':<12} {'V(V)':<6} {'f(GHz)':<8} {'P_dyn(W)':<10} "
          f"{'P_stat(W)':<10} {'P_sc(W)':<9} {'P_total(W)':<11}")
    print("-" * 65)

    for cfg in configs:
        result = calculate_cmos_power(cfg["alpha"], cfg["cap"],
                                      cfg["v"], cfg["f"], cfg["leak"])
        print(f"{cfg['label']:<12} {cfg['v']:<6.1f} {cfg['f']:<8.1f} "
              f"{result['dynamic_w']:<10.2f} {result['static_w']:<10.2f} "
              f"{result['short_circuit_w']:<9.2f} {result['total_w']:<11.2f}")

    # Show voltage scaling effect
    print("\nVoltage scaling insight:")
    base = calculate_cmos_power(0.2, 20, 1.0, 3.0, 10)
    reduced = calculate_cmos_power(0.2, 20, 0.7, 2.0, 5)
    savings = (1 - reduced["total_w"] / base["total_w"]) * 100
    print(f"  Reducing V from 1.0V to 0.7V and f from 3.0 to 2.0 GHz")
    print(f"  saves {savings:.1f}% total power ({base['total_w']:.1f}W → "
          f"{reduced['total_w']:.1f}W)")


# ========================================================================
# 2. DVFS Analysis
# ========================================================================

def dvfs_analysis(task_cycles: int, configs: list[dict]) -> list[dict]:
    """Analyze DVFS trade-offs for a fixed workload.

    For each configuration (voltage, frequency), calculate:
    - Execution time
    - Dynamic power
    - Energy consumed (power × time)

    Args:
        task_cycles: Number of cycles the task requires.
        configs: List of dicts with 'label', 'freq_ghz', 'voltage_v'.

    Returns:
        List of result dicts with added time_s, power_w, energy_j fields.
    """
    results = []
    for cfg in configs:
        freq_hz = cfg["freq_ghz"] * 1e9
        time_s = task_cycles / freq_hz

        # Dynamic power ∝ V^2 * f (using normalized capacitance)
        # We use the first config as the reference for power scaling
        power_w = cfg.get("tdp_w", (cfg["voltage_v"] ** 2) * cfg["freq_ghz"] * 100)

        energy_j = power_w * time_s

        results.append({
            **cfg,
            "time_s": time_s,
            "power_w": power_w,
            "energy_j": energy_j,
        })
    return results


def demo_dvfs():
    """Compare energy efficiency at different DVFS operating points."""
    print("\n" + "=" * 65)
    print("2. DVFS Energy Analysis")
    print("=" * 65)

    task_cycles = 10_000_000_000  # 10 billion cycles

    configs = [
        {"label": "Turbo",  "freq_ghz": 4.5, "voltage_v": 1.20, "tdp_w": 150},
        {"label": "High",   "freq_ghz": 3.5, "voltage_v": 1.00, "tdp_w": 85},
        {"label": "Normal", "freq_ghz": 2.5, "voltage_v": 0.85, "tdp_w": 45},
        {"label": "Low",    "freq_ghz": 1.5, "voltage_v": 0.70, "tdp_w": 20},
        {"label": "Idle",   "freq_ghz": 0.8, "voltage_v": 0.60, "tdp_w": 8},
    ]

    results = dvfs_analysis(task_cycles, configs)

    print(f"\nTask: {task_cycles/1e9:.0f} billion cycles\n")
    print(f"{'Mode':<8} {'f(GHz)':<8} {'V(V)':<6} {'Power(W)':<10} "
          f"{'Time(s)':<10} {'Energy(J)':<10} {'Eff':<6}")
    print("-" * 58)

    # Find best energy for efficiency comparison
    min_energy = min(r["energy_j"] for r in results)

    for r in results:
        eff = min_energy / r["energy_j"] * 100
        print(f"{r['label']:<8} {r['freq_ghz']:<8.1f} {r['voltage_v']:<6.2f} "
              f"{r['power_w']:<10.1f} {r['time_s']:<10.2f} "
              f"{r['energy_j']:<10.1f} {eff:<5.0f}%")

    best = min(results, key=lambda r: r["energy_j"])
    worst = max(results, key=lambda r: r["energy_j"])
    print(f"\nMost energy-efficient: {best['label']} "
          f"({best['energy_j']:.1f}J, {best['time_s']:.2f}s)")
    print(f"Least efficient:      {worst['label']} "
          f"({worst['energy_j']:.1f}J, {worst['time_s']:.2f}s)")
    print(f"Energy savings: {(1 - best['energy_j']/worst['energy_j'])*100:.1f}%")


# ========================================================================
# 3. C-State Selection
# ========================================================================

def select_c_state(idle_duration_us: float,
                   states: list[dict]) -> dict:
    """Select the optimal C-state for a given idle duration.

    A deeper C-state saves more power but has a higher wake-up cost.
    The break-even time is: t_break = t_wake * (P_active - P_state) / (P_active - P_deeper)
    Simplified: use a state only if idle_duration > 2 * wake_latency (rule of thumb).

    Args:
        idle_duration_us: Expected idle duration in microseconds.
        states: List of C-state dicts with name, wake_us, power_fraction.

    Returns:
        The selected C-state dict.
    """
    best_state = states[0]  # Default: shallowest

    for state in states:
        # Only use this state if idle duration is at least 2x the wake latency
        # (accounting for both entry and exit overhead)
        min_duration = state["wake_us"] * 2
        if idle_duration_us >= min_duration:
            # Deeper state is better (lower power fraction)
            if state["power_fraction"] < best_state["power_fraction"]:
                best_state = state

    return best_state


def demo_c_states():
    """Simulate C-state selection for various idle durations."""
    print("\n" + "=" * 65)
    print("3. C-State Selection Simulator")
    print("=" * 65)

    c_states = [
        {"name": "C0 (Active)", "wake_us": 0,    "power_fraction": 1.00},
        {"name": "C1 (Halt)",   "wake_us": 1,    "power_fraction": 0.70},
        {"name": "C1E (Enh.)",  "wake_us": 10,   "power_fraction": 0.50},
        {"name": "C3 (Sleep)",  "wake_us": 100,  "power_fraction": 0.30},
        {"name": "C6 (Deep)",   "wake_us": 500,  "power_fraction": 0.05},
        {"name": "C7 (Deeper)", "wake_us": 1000, "power_fraction": 0.02},
    ]

    print("\nAvailable C-States:")
    print(f"  {'State':<16} {'Wake Latency':<15} {'Power (% active)':<18}")
    print("  " + "-" * 48)
    for cs in c_states:
        print(f"  {cs['name']:<16} {cs['wake_us']:>8} μs     {cs['power_fraction']*100:>5.0f}%")

    idle_durations = [0.5, 5, 50, 500, 2000, 8000, 50000]

    print(f"\n  {'Idle Duration':<15} → {'Selected State':<16} {'Power Savings':<15}")
    print("  " + "-" * 48)
    for dur in idle_durations:
        selected = select_c_state(dur, c_states)
        savings = (1 - selected["power_fraction"]) * 100
        if dur >= 1000:
            dur_str = f"{dur/1000:.1f} ms"
        else:
            dur_str = f"{dur:.0f} μs"
        print(f"  {dur_str:<15} → {selected['name']:<16} {savings:>5.0f}% saved")


# ========================================================================
# 4. Dark Silicon Budget
# ========================================================================

def dark_silicon_analysis(chip_area_mm2: float, transistor_count: float,
                          tdp_w: float, power_per_transistor_pw: float) -> dict:
    """Estimate dark silicon fraction.

    Args:
        chip_area_mm2: Total chip area in mm².
        transistor_count: Total transistor count.
        tdp_w: Thermal design power in watts.
        power_per_transistor_pw: Average power per active transistor in picowatts.

    Returns:
        Dict with active fraction, dark fraction, and active transistor count.
    """
    # Maximum power if all transistors active
    max_power_w = transistor_count * power_per_transistor_pw * 1e-12

    # How many transistors can we power within TDP?
    active_fraction = min(tdp_w / max_power_w, 1.0)
    active_count = transistor_count * active_fraction
    dark_fraction = 1.0 - active_fraction

    return {
        "total_transistors": transistor_count,
        "max_power_w": max_power_w,
        "active_fraction": active_fraction,
        "dark_fraction": dark_fraction,
        "active_count": active_count,
        "tdp_w": tdp_w,
    }


def demo_dark_silicon():
    """Estimate dark silicon at different technology nodes."""
    print("\n" + "=" * 65)
    print("4. Dark Silicon Analysis")
    print("=" * 65)

    nodes = [
        {"name": "7nm (Mobile)", "area": 100, "transistors": 10e9,
         "tdp": 15, "ppt": 0.5},
        {"name": "7nm (Desktop)", "area": 200, "transistors": 20e9,
         "tdp": 125, "ppt": 0.5},
        {"name": "5nm (Mobile)", "area": 120, "transistors": 20e9,
         "tdp": 15, "ppt": 0.4},
        {"name": "5nm (Desktop)", "area": 250, "transistors": 50e9,
         "tdp": 170, "ppt": 0.4},
        {"name": "3nm (Mobile)", "area": 120, "transistors": 30e9,
         "tdp": 20, "ppt": 0.35},
    ]

    print(f"\n{'Node':<18} {'Transistors':<14} {'TDP(W)':<8} "
          f"{'Max P(W)':<10} {'Active%':<9} {'Dark%':<8}")
    print("-" * 65)

    for node in nodes:
        result = dark_silicon_analysis(node["area"], node["transistors"],
                                       node["tdp"], node["ppt"])
        t_str = f"{node['transistors']/1e9:.0f}B"
        print(f"{node['name']:<18} {t_str:<14} {node['tdp']:<8.0f} "
              f"{result['max_power_w']:<10.0f} "
              f"{result['active_fraction']*100:<9.1f} "
              f"{result['dark_fraction']*100:<8.1f}")

    print("\nKey insight: At 3nm mobile, ~75% of transistors must remain dark.")
    print("This drives the shift to heterogeneous SoCs with specialized accelerators.")


# ========================================================================
# 5. Data Center PUE and Cost
# ========================================================================

def datacenter_cost(num_servers: int, avg_power_w: float,
                    pue: float, electricity_cost_per_kwh: float,
                    hours_per_year: int = 8760) -> dict:
    """Calculate annual data center electricity cost.

    Args:
        num_servers: Number of servers.
        avg_power_w: Average power per server in watts.
        pue: Power Usage Effectiveness (>= 1.0).
        electricity_cost_per_kwh: Cost in $/kWh.
        hours_per_year: Typically 8760.

    Returns:
        Dict with IT power, total power, annual energy, and annual cost.
    """
    it_power_kw = num_servers * avg_power_w / 1000
    total_power_kw = it_power_kw * pue
    annual_kwh = total_power_kw * hours_per_year
    annual_cost = annual_kwh * electricity_cost_per_kwh

    return {
        "it_power_kw": it_power_kw,
        "total_power_kw": total_power_kw,
        "annual_kwh": annual_kwh,
        "annual_cost": annual_cost,
    }


def demo_pue():
    """Compare data center costs at different PUE values."""
    print("\n" + "=" * 65)
    print("5. Data Center PUE and Cost Analysis")
    print("=" * 65)

    servers = 10000
    avg_power = 300  # watts per server
    cost_kwh = 0.10

    pue_values = [2.0, 1.5, 1.4, 1.2, 1.1]

    print(f"\nScenario: {servers:,} servers × {avg_power}W avg, "
          f"${cost_kwh}/kWh\n")
    print(f"{'PUE':<6} {'IT Power(MW)':<14} {'Total(MW)':<12} "
          f"{'Annual(GWh)':<13} {'Annual Cost($)':<15}")
    print("-" * 60)

    results = {}
    for pue in pue_values:
        r = datacenter_cost(servers, avg_power, pue, cost_kwh)
        results[pue] = r
        print(f"{pue:<6.1f} {r['it_power_kw']/1000:<14.1f} "
              f"{r['total_power_kw']/1000:<12.1f} "
              f"{r['annual_kwh']/1e6:<13.1f} "
              f"${r['annual_cost']:>12,.0f}")

    # Savings analysis
    worst_cost = results[2.0]["annual_cost"]
    best_cost = results[1.1]["annual_cost"]
    savings = worst_cost - best_cost
    print(f"\nImproving PUE from 2.0 to 1.1 saves ${savings:,.0f}/year "
          f"({savings/worst_cost*100:.0f}%)")

    # Break-even analysis for PUE improvement
    print(f"\nBreak-even analysis (PUE 1.5 → 1.2):")
    annual_save = results[1.5]["annual_cost"] - results[1.2]["annual_cost"]
    investments = [1_000_000, 5_000_000, 10_000_000]
    for inv in investments:
        years = inv / annual_save
        print(f"  ${inv:>12,} investment → break-even in {years:.1f} years")


# ========================================================================
# Main
# ========================================================================

if __name__ == "__main__":
    demo_cmos_power()
    demo_dvfs()
    demo_c_states()
    demo_dark_silicon()
    demo_pue()
