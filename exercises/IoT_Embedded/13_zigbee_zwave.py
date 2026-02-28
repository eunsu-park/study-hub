"""
Exercises for Lesson 13: Zigbee and Z-Wave
Topic: IoT_Embedded

Solutions to practice problems covering protocol selection,
mesh network design, power budget analysis, range calculations,
and Thread network simulation.
"""

import math
import random
from collections import deque


def exercise_1():
    """
    Protocol selection for different IoT scenarios.
    Choose the most appropriate protocol and justify.
    """
    print("=== Exercise 1: Protocol Selection ===\n")

    scenarios = [
        {
            "desc": "200-room hotel with wireless door locks, 5-year battery life",
            "choice": "Z-Wave",
            "reasons": [
                "Sub-GHz frequency penetrates hotel walls/doors better",
                "Mandatory certification ensures lock interoperability",
                "232 device limit is sufficient (200 rooms + some extras)",
                "5-year battery life achievable with Z-Wave's low duty cycle",
                "Source routing reduces memory needs in battery-powered locks",
            ],
        },
        {
            "desc": "Greenhouse with 500 soil moisture sensors over 10 acres",
            "choice": "Zigbee",
            "reasons": [
                "65,000 device limit handles 500 sensors easily",
                "Mesh topology covers large area through router relaying",
                "Low data rate (moisture readings) fits 250 kbps well",
                "ZCL temperature/humidity clusters provide standard profiles",
                "Multiple routers with solar power extend range across field",
            ],
        },
        {
            "desc": "Homeowner: 15 smart bulbs, 3 thermostats, 2 door locks",
            "choice": "Matter (over Thread/WiFi)",
            "reasons": [
                "20 devices is within Thread's capacity",
                "Matter ensures Apple/Google/Alexa compatibility",
                "Future-proof as industry converges on Matter",
                "Thread mesh for low-power sensors, WiFi for cameras",
                "No single-vendor lock-in",
            ],
        },
        {
            "desc": "Factory: real-time vibration monitoring on 50 machines, <100ms latency",
            "choice": "Thread or Industrial Zigbee",
            "reasons": [
                "Thread's IP-based routing integrates with factory IT systems",
                "Mesh self-healing handles harsh RF (metal structures)",
                "250 kbps is sufficient for vibration FFT data at 100 Hz",
                "<100ms latency achievable with 2-3 hop mesh",
                "Multiple border routers eliminate single points of failure",
            ],
        },
    ]

    for i, s in enumerate(scenarios, 1):
        print(f"  Scenario {i}: {s['desc']}")
        print(f"  Best protocol: {s['choice']}")
        for r in s['reasons']:
            print(f"    • {r}")
        print()


def exercise_2():
    """
    Mesh network design for a two-story house.
    """
    print("=== Exercise 2: Mesh Network Design ===\n")

    print("  House: 150 m² per floor (≈ 12m × 12.5m), 2 floors")
    print("  Zigbee indoor range: ~15-20m (conservative for walls)")
    print()

    print("  1) Minimum routers needed:")
    print("     Each floor: 12m × 12.5m")
    print("     With 15m range and walls reducing it to ~10m effective:")
    print("     Ground floor: 2 routers (one each side)")
    print("     Second floor: 2 routers (one each side)")
    print("     + 1 coordinator (ground floor center)")
    print("     Total: 1 coordinator + 4 routers = 5 mains-powered devices")
    print()

    print("  2) Network topology:")
    print()
    print("     Second Floor:")
    print("     ┌─────────────────────────────┐")
    print("     │  R3              R4          │")
    print("     │  (bedroom)      (study)      │")
    print("     │       \\         /            │")
    print("     │  E3    \\  E4  /    E5        │")
    print("     │  (temp) \\ (hum) /  (motion)  │")
    print("     └──────────┼──────┼────────────┘")
    print("                │stairs│")
    print("     ┌──────────┼──────┼────────────┐")
    print("     │  R1      │ COORD│     R2     │")
    print("     │  (kitchen)│(living)│  (hallway)│")
    print("     │       \\   │  /   │  /        │")
    print("     │  E1    \\  │ /    │ /    E2   │")
    print("     │  (temp) \\─┼/─────┘ (door)    │")
    print("     └─────────────────────────────┘")
    print()

    print("  3) Routing path: Basement sensor → Second floor display")
    print("     E1 → R1 → COORD → R3 → E4 (display)")
    print("     Total: 4 hops")
    print()

    print("  4) Self-healing when ground-floor stairway router fails:")
    print("     If R1 fails:")
    print("     - E1 loses parent, scans for new parent")
    print("     - E1 associates with COORD (if in range) or R2")
    print("     - Route: E1 → COORD → R3 → E4")
    print("     - Or:    E1 → R2 → COORD → R3 → E4")
    print("     Network remains functional with one more hop")
    print()


def exercise_3():
    """
    Power budget analysis for a Zigbee temperature sensor.
    """
    print("=== Exercise 3: Power Budget Analysis ===\n")

    # Given
    battery_mah = 3000     # 2× AA = 3000 mAh total
    voltage = 3.0          # V
    sleep_current_ua = 5   # μA
    tx_current_ma = 20     # mA
    tx_time_ms = 15        # ms per reading
    reading_interval_s = 300  # 5 minutes

    # Calculate duty cycle
    period_ms = reading_interval_s * 1000
    duty_cycle = tx_time_ms / period_ms

    print(f"  Battery: {battery_mah} mAh at {voltage}V")
    print(f"  Sleep current: {sleep_current_ua} μA")
    print(f"  Transmit current: {tx_current_ma} mA")
    print(f"  Transmit time: {tx_time_ms} ms per reading")
    print(f"  Reading interval: {reading_interval_s}s ({reading_interval_s/60:.0f} min)")
    print(f"  Duty cycle: {duty_cycle*100:.4f}%")
    print()

    # Average current
    sleep_time_fraction = 1 - duty_cycle
    avg_current_ma = (sleep_current_ua / 1000 * sleep_time_fraction +
                      tx_current_ma * duty_cycle)

    print(f"  Average current calculation:")
    print(f"    Sleep contribution: {sleep_current_ua} μA × {sleep_time_fraction:.6f} "
          f"= {sleep_current_ua/1000 * sleep_time_fraction:.5f} mA")
    print(f"    TX contribution:    {tx_current_ma} mA × {duty_cycle:.6f} "
          f"= {tx_current_ma * duty_cycle:.5f} mA")
    print(f"    Average current:    {avg_current_ma:.5f} mA = {avg_current_ma*1000:.2f} μA")
    print()

    # Battery life
    battery_life_hours = battery_mah / avg_current_ma
    battery_life_years = battery_life_hours / (24 * 365.25)

    print(f"  Battery life:")
    print(f"    {battery_mah} mAh / {avg_current_ma:.5f} mA = {battery_life_hours:,.0f} hours")
    print(f"    = {battery_life_hours/24:,.0f} days")
    print(f"    = {battery_life_years:.1f} years")
    print()

    # Power breakdown
    sleep_energy_pct = (sleep_current_ua / 1000 * sleep_time_fraction) / avg_current_ma * 100
    tx_energy_pct = (tx_current_ma * duty_cycle) / avg_current_ma * 100

    print(f"  Power breakdown:")
    print(f"    Sleep: {sleep_energy_pct:.1f}% of total energy")
    print(f"    Transmit: {tx_energy_pct:.1f}% of total energy")
    print(f"    Dominant consumer: {'Sleep' if sleep_energy_pct > tx_energy_pct else 'Transmit'}")
    print()


def exercise_4():
    """
    Z-Wave vs Zigbee range through concrete walls.
    """
    print("=== Exercise 4: Z-Wave vs Zigbee Range ===\n")

    wall_attenuation_db = 12  # dB per concrete wall
    min_rssi_zigbee = -95     # dBm
    min_rssi_zwave = -100     # dBm (better sensitivity)
    tx_power_zigbee = 0       # dBm
    tx_power_zwave = -2       # dBm
    free_space_loss_10m = -60 # dBm at 10m

    # Zigbee: how many walls?
    available_budget_zigbee = tx_power_zigbee + abs(free_space_loss_10m) - abs(min_rssi_zigbee)
    # Signal at receiver without walls: tx + path_loss = 0 + (-60) = -60 dBm
    # Margin above sensitivity: -60 - (-95) = 35 dB
    margin_zigbee = (tx_power_zigbee + free_space_loss_10m) - min_rssi_zigbee
    walls_zigbee = int(margin_zigbee / wall_attenuation_db)

    print(f"  1) Zigbee (2.4 GHz):")
    print(f"     TX power: {tx_power_zigbee} dBm")
    print(f"     Path loss at 10m: {free_space_loss_10m} dBm")
    print(f"     Signal at 10m: {tx_power_zigbee + free_space_loss_10m} dBm")
    print(f"     Rx sensitivity: {min_rssi_zigbee} dBm")
    print(f"     Link margin: {margin_zigbee} dB")
    print(f"     Walls penetrated: {margin_zigbee} / {wall_attenuation_db} = {walls_zigbee}")
    print()

    # Z-Wave: better sensitivity compensates for lower TX power
    # Sub-GHz also has ~6dB better wall penetration (use 6dB less attenuation)
    wall_attenuation_zwave = wall_attenuation_db - 6  # Sub-GHz advantage
    margin_zwave = (tx_power_zwave + free_space_loss_10m) - min_rssi_zwave
    walls_zwave = int(margin_zwave / wall_attenuation_zwave)

    print(f"  2) Z-Wave (908 MHz):")
    print(f"     TX power: {tx_power_zwave} dBm")
    print(f"     Path loss at 10m: {free_space_loss_10m} dBm (approx)")
    print(f"     Signal at 10m: {tx_power_zwave + free_space_loss_10m} dBm")
    print(f"     Rx sensitivity: {min_rssi_zwave} dBm (5 dB better)")
    print(f"     Link margin: {margin_zwave} dB")
    print(f"     Wall attenuation (sub-GHz): {wall_attenuation_zwave} dB/wall")
    print(f"     Walls penetrated: {margin_zwave} / {wall_attenuation_zwave} = {walls_zwave}")
    print()

    print(f"  3) How mesh routing changes the analysis:")
    print(f"     With mesh: walls between SOURCE and DESTINATION don't all matter")
    print(f"     Each hop only needs to penetrate walls to the NEXT router")
    print(f"     Example: 4 walls between endpoints, but with a router after every 2 walls:")
    print(f"       Without mesh: need {4 * wall_attenuation_db} dB budget (4 walls)")
    print(f"       With mesh: 2 hops × {2 * wall_attenuation_db} dB each = feasible")
    print(f"     Mesh effectively removes the multi-wall penetration problem")
    print()


def exercise_5():
    """
    Thread-like mesh network simulator with self-healing.
    """
    print("=== Exercise 5: Thread Network Simulation ===\n")

    random.seed(42)

    # Create 20 nodes at random positions in 100×100m area
    n_nodes = 20
    radio_range = 30  # meters

    nodes = {}
    for i in range(n_nodes):
        nodes[i] = {"x": random.uniform(0, 100), "y": random.uniform(0, 100)}

    # Designate 2 border routers (nodes 0 and 1, placed strategically)
    nodes[0] = {"x": 25, "y": 50}   # Border Router 1
    nodes[1] = {"x": 75, "y": 50}   # Border Router 2
    border_routers = {0, 1}

    print(f"  Network: {n_nodes} nodes in 100×100m area")
    print(f"  Radio range: {radio_range}m")
    print(f"  Border Routers: nodes {sorted(border_routers)}")

    def distance(a, b):
        return math.sqrt((nodes[a]["x"] - nodes[b]["x"])**2 +
                         (nodes[a]["y"] - nodes[b]["y"])**2)

    def get_neighbors(node_id, active_nodes):
        return [n for n in active_nodes
                if n != node_id and distance(node_id, n) <= radio_range]

    def shortest_path_to_br(node_id, active_nodes):
        """BFS to find shortest path to any border router."""
        if node_id in border_routers:
            return [node_id]

        visited = {node_id}
        queue = deque([(node_id, [node_id])])

        while queue:
            current, path = queue.popleft()
            for neighbor in get_neighbors(current, active_nodes):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = path + [neighbor]
                    if neighbor in border_routers:
                        return new_path
                    queue.append((neighbor, new_path))

        return None  # Unreachable

    # Find paths for all nodes
    all_nodes = set(range(n_nodes))
    print(f"\n  --- Initial Network ---")
    reachable = 0
    total_hops = 0

    for i in range(2, n_nodes):  # Skip BRs themselves
        path = shortest_path_to_br(i, all_nodes)
        if path:
            reachable += 1
            hops = len(path) - 1
            total_hops += hops
            if i < 7:  # Print first few
                path_str = " → ".join(str(p) for p in path)
                br_id = path[-1]
                print(f"    Node {i:2d} → BR{br_id}: {hops} hops [{path_str}]")
        else:
            print(f"    Node {i:2d} → UNREACHABLE")

    print(f"    ... ({n_nodes - 2} end devices total)")
    print(f"    Reachable: {reachable}/{n_nodes-2} "
          f"({reachable/(n_nodes-2)*100:.0f}%)")
    if reachable > 0:
        print(f"    Average hops: {total_hops/reachable:.1f}")

    # Simulate random node failure
    failed_node = random.choice([n for n in range(2, n_nodes)])
    active_nodes = all_nodes - {failed_node}

    print(f"\n  --- After Node {failed_node} Fails ---")
    print(f"    Node {failed_node} at ({nodes[failed_node]['x']:.0f}, "
          f"{nodes[failed_node]['y']:.0f}) removed")

    reachable_after = 0
    total_hops_after = 0
    rerouted = 0

    for i in range(2, n_nodes):
        if i == failed_node:
            continue
        path_before = shortest_path_to_br(i, all_nodes)
        path_after = shortest_path_to_br(i, active_nodes)

        if path_after:
            reachable_after += 1
            total_hops_after += len(path_after) - 1
            if path_before and failed_node in path_before:
                rerouted += 1
                before_str = " → ".join(str(p) for p in path_before)
                after_str = " → ".join(str(p) for p in path_after)
                print(f"    Node {i:2d} REROUTED:")
                print(f"      Before: [{before_str}]")
                print(f"      After:  [{after_str}]")
        else:
            if path_before:
                print(f"    Node {i:2d} DISCONNECTED (was reachable)")

    remaining = n_nodes - 3  # Exclude BRs and failed node
    print(f"\n    Reachable: {reachable_after}/{remaining}")
    print(f"    Rerouted: {rerouted} nodes found alternative paths")
    if reachable_after > 0:
        print(f"    Average hops: {total_hops_after/reachable_after:.1f}")
    print(f"    Self-healing: {'SUCCESS' if reachable_after >= remaining - 1 else 'PARTIAL'}")
    print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
