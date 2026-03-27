"""
Exercises for Lesson 04: Physical Layer
Topic: Networking
Solutions to practice problems from the lesson.
"""
import math


def exercise_1():
    """
    Problem: Explain three major functions of the physical layer.
    """
    functions = [
        ("Bit Synchronization", "Synchronize TX/RX clocks so receiver samples bits at correct time"),
        ("Bit Rate Control", "Determine transmission speed (bps) based on medium capabilities"),
        ("Transmission Mode", "Define simplex, half-duplex, or full-duplex communication"),
        ("Physical Interface", "Specify connectors, voltage levels, pin assignments, signaling"),
    ]

    print("Physical Layer (L1) Major Functions:")
    for i, (func, description) in enumerate(functions, 1):
        print(f"  {i}. {func}: {description}")


def exercise_2():
    """
    Problem: Arrange the following media in order of transmission distance (longest first):
    UTP, Single-mode fiber, Coaxial cable, Multi-mode fiber
    """
    media = {
        "Single-mode Fiber": {"max_distance_m": 100000, "typical_speed": "10-400 Gbps"},
        "Multi-mode Fiber":  {"max_distance_m": 2000, "typical_speed": "1-100 Gbps"},
        "Coaxial Cable":     {"max_distance_m": 500, "typical_speed": "10 Mbps-10 Gbps"},
        "UTP":               {"max_distance_m": 100, "typical_speed": "100 Mbps-10 Gbps"},
    }

    sorted_media = sorted(media.items(), key=lambda x: x[1]["max_distance_m"], reverse=True)

    print("Transmission media ordered by distance (longest first):")
    for name, info in sorted_media:
        print(f"  {name:25s} ~{info['max_distance_m']:>6,}m  ({info['typical_speed']})")

    print("\nAnswer: Single-mode fiber > Multi-mode fiber > Coaxial cable > UTP")


def exercise_3():
    """
    Problem: Explain the differences between Cat5e and Cat6 cables.
    """
    cables = {
        "Cat5e": {"max_speed": "1 Gbps", "bandwidth": "100 MHz", "max_length": "100m",
                  "shielding": "Usually unshielded", "notes": "Most common in older installations"},
        "Cat6":  {"max_speed": "10 Gbps (55m)", "bandwidth": "250 MHz", "max_length": "100m (1Gbps) / 55m (10Gbps)",
                  "shielding": "Better crosstalk reduction", "notes": "Stricter specifications, thicker"},
    }

    print("Cat5e vs Cat6 Cable Comparison:")
    for cable, specs in cables.items():
        print(f"\n  {cable}:")
        for spec, value in specs.items():
            print(f"    {spec:15s}: {value}")


def exercise_4():
    """
    Problem: Explain the usage of straight-through and crossover cables.

    Reasoning: Auto-MDIX in modern switches has made crossover cables largely
    unnecessary, but understanding when each type is needed is still important
    for legacy equipment and certification exams.
    """
    cable_usage = {
        "Straight-through": {
            "wiring": "Pin 1->1, Pin 2->2, ... (same on both ends)",
            "use_cases": [
                "PC to Switch (different device types)",
                "Switch to Router (different device types)",
                "Router to Modem",
            ],
            "rule": "Connect DIFFERENT types of devices",
        },
        "Crossover": {
            "wiring": "TX pins connected to RX pins (swapped)",
            "use_cases": [
                "PC to PC (same device type)",
                "Switch to Switch (same device type)",
                "Router to Router (same device type)",
            ],
            "rule": "Connect SAME types of devices",
        },
    }

    print("Straight-through vs Crossover Cables:")
    for cable_type, info in cable_usage.items():
        print(f"\n  {cable_type}:")
        print(f"    Wiring: {info['wiring']}")
        print(f"    Rule: {info['rule']}")
        print(f"    Use cases:")
        for uc in info["use_cases"]:
            print(f"      - {uc}")

    print("\n  Note: Modern devices with Auto-MDIX detect and auto-adjust,")
    print("  making crossover cables unnecessary in most cases.")


def exercise_5():
    """
    Problem: Choose appropriate transmission media for each situation:
    (a) Connecting 100 PCs in an office
    (b) Connecting two buildings 1km apart
    (c) 40Gbps connection between servers in data center
    """
    scenarios = [
        {
            "scenario": "Connecting 100 PCs in an office",
            "medium": "Cat5e or Cat6 UTP",
            "reason": "Cost-effective for short distances (<100m), sufficient bandwidth for office use, "
                      "easy to install and maintain",
        },
        {
            "scenario": "Connecting two buildings 1km apart",
            "medium": "Single-mode Fiber",
            "reason": "Supports long distance (>100m UTP limit), immune to EMI between buildings, "
                      "weather-resistant, future-proof bandwidth",
        },
        {
            "scenario": "40Gbps connection between servers in data center",
            "medium": "Multi-mode Fiber (OM4) or Single-mode Fiber",
            "reason": "OM4 supports 40/100G up to 150m (sufficient for data center), "
                      "low latency, high density cabling possible",
        },
    ]

    print("Media selection for each scenario:")
    for s in scenarios:
        print(f"\n  Scenario: {s['scenario']}")
        print(f"  Recommended: {s['medium']}")
        print(f"  Reason: {s['reason']}")


def exercise_6():
    """
    Problem: Compare the advantages and disadvantages of 2.4GHz and 5GHz Wi-Fi.
    """
    wifi_bands = {
        "2.4 GHz": {
            "advantages": [
                "Wider coverage range (longer wavelength penetrates walls better)",
                "Better wall/obstacle penetration",
                "Universal device compatibility",
            ],
            "disadvantages": [
                "More congested (shared with Bluetooth, microwaves, etc.)",
                "Lower maximum speed",
                "Only 3 non-overlapping channels (1, 6, 11)",
            ],
        },
        "5 GHz": {
            "advantages": [
                "Higher speeds (wider channels available)",
                "Less interference (more channels, fewer devices)",
                "23+ non-overlapping channels",
            ],
            "disadvantages": [
                "Shorter range",
                "Poor wall penetration (higher frequency absorbed more)",
                "Not all devices support it (though most modern ones do)",
            ],
        },
    }

    print("2.4 GHz vs 5 GHz Wi-Fi Comparison:")
    for band, info in wifi_bands.items():
        print(f"\n  {band}:")
        print("    Advantages:")
        for adv in info["advantages"]:
            print(f"      + {adv}")
        print("    Disadvantages:")
        for dis in info["disadvantages"]:
            print(f"      - {dis}")


def exercise_7():
    """
    Problem: Calculate transmission delay and propagation delay when transmitting
    1Gbps data through a 100m Ethernet cable.
    (Assume speed of light in cable is 2x10^8 m/s)

    Reasoning: Understanding the difference between transmission delay (time to
    push bits onto the wire) and propagation delay (time for signal to travel)
    is key to calculating total latency.
    """
    # Parameters
    data_size_bits = 1500 * 8  # 1500 bytes (typical Ethernet frame) in bits
    bandwidth_bps = 1e9        # 1 Gbps
    distance_m = 100           # 100 meters
    signal_speed = 2e8         # 2 x 10^8 m/s

    # Calculations
    transmission_delay = data_size_bits / bandwidth_bps  # seconds
    propagation_delay = distance_m / signal_speed        # seconds

    print("Delay Calculation for 1500-byte frame on 1Gbps, 100m Ethernet:")
    print(f"\n  Transmission delay = Data size / Bandwidth")
    print(f"    = {data_size_bits} bits / {bandwidth_bps:.0e} bps")
    print(f"    = {transmission_delay * 1e6:.2f} microseconds")
    print(f"\n  Propagation delay = Distance / Signal speed")
    print(f"    = {distance_m}m / {signal_speed:.0e} m/s")
    print(f"    = {propagation_delay * 1e6:.2f} microseconds")
    print(f"\n  Total delay = {(transmission_delay + propagation_delay) * 1e6:.2f} microseconds")
    print(f"\n  Note: At 1Gbps over 100m, propagation delay ({propagation_delay * 1e6:.2f} us)")
    print(f"  is comparable to transmission delay ({transmission_delay * 1e6:.2f} us).")


def exercise_8():
    """
    Problem: Using Shannon's formula, calculate the maximum transmission speed
    of a channel with 10MHz bandwidth and SNR of 1000.

    Shannon's formula: C = B * log2(1 + S/N)
    where C = channel capacity (bps), B = bandwidth (Hz), S/N = signal-to-noise ratio
    """
    bandwidth_hz = 10e6  # 10 MHz
    snr = 1000           # Signal-to-noise ratio (linear, not dB)

    # Shannon's formula
    capacity = bandwidth_hz * math.log2(1 + snr)

    print("Shannon's Channel Capacity Theorem:")
    print(f"  C = B * log2(1 + S/N)")
    print(f"  C = {bandwidth_hz / 1e6:.0f} MHz * log2(1 + {snr})")
    print(f"  C = {bandwidth_hz / 1e6:.0f} * 10^6 * log2({1 + snr})")
    print(f"  C = {bandwidth_hz / 1e6:.0f} * 10^6 * {math.log2(1 + snr):.2f}")
    print(f"  C = {capacity / 1e6:.1f} Mbps")
    print(f"\n  SNR in dB: {10 * math.log10(snr):.1f} dB")
    print(f"\n  Interpretation: Even with 10 MHz bandwidth, an SNR of 1000")
    print(f"  limits the theoretical maximum to ~{capacity / 1e6:.1f} Mbps.")


def exercise_9():
    """
    Problem: Explain at least 5 advantages of fiber optic over copper cable.
    """
    advantages = [
        ("EMI Immunity", "Uses light, not electricity -- completely immune to electromagnetic interference"),
        ("High Bandwidth", "Terabit-class capacity, far exceeding any copper technology"),
        ("Long Distance", "Single-mode fiber reaches ~100km without repeaters vs 100m for UTP"),
        ("Security", "Very difficult to tap without detection (no electromagnetic emanation)"),
        ("Lightweight", "Much lighter than copper cables of equivalent capacity"),
        ("No Corrosion", "Glass/plastic doesn't corrode like copper in harsh environments"),
        ("Low Attenuation", "Signal loses less strength per km compared to copper"),
    ]

    print("Advantages of Fiber Optic over Copper:")
    for i, (advantage, detail) in enumerate(advantages, 1):
        print(f"\n  {i}. {advantage}")
        print(f"     {detail}")


def exercise_10():
    """
    Problem: Explain why fiber optic is chosen over Cat6a in data centers.

    Reasoning: Data centers have specific requirements (density, speed, distance
    between racks/rows, future scalability) that favor fiber optic solutions.
    """
    reasons = [
        ("Speed Requirements", "40G/100G/400G speeds needed between spine-leaf switches; "
         "Cat6a maxes out at 10G"),
        ("Distance", "Data center rows can span >100m; Cat6a limited to 100m, "
         "fiber supports much longer runs"),
        ("Cable Density", "Fiber cables are thinner, allowing better airflow in cable trays "
         "and higher port density"),
        ("EMI Environment", "Data centers have heavy electrical equipment; "
         "fiber is immune to interference"),
        ("Future-proofing", "Fiber infrastructure supports next-gen speeds without re-cabling"),
        ("Heat Generation", "Fiber produces less heat than copper at high speeds, "
         "reducing cooling costs"),
    ]

    print("Why data centers prefer fiber over Cat6a:")
    for i, (reason, detail) in enumerate(reasons, 1):
        print(f"\n  {i}. {reason}")
        print(f"     {detail}")


if __name__ == "__main__":
    exercises = [
        exercise_1, exercise_2, exercise_3, exercise_4, exercise_5,
        exercise_6, exercise_7, exercise_8, exercise_9, exercise_10,
    ]
    for i, ex in enumerate(exercises, 1):
        print(f"\n{'=' * 60}")
        print(f"=== Exercise {i} ===")
        print(f"{'=' * 60}")
        ex()

    print(f"\n{'=' * 60}")
    print("All exercises completed!")
