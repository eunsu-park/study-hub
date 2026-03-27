"""
Exercises for Lesson 05: Data Link Layer
Topic: Networking
Solutions to practice problems from the lesson.
"""


def exercise_1():
    """
    Problem: Explain the structure of a MAC address and what OUI means.

    Reasoning: MAC addresses are the foundation of L2 communication.
    Understanding their structure helps with troubleshooting, security
    analysis, and vendor identification.
    """
    mac_example = "00:1A:2B:3C:4D:5E"
    oui = mac_example[:8]     # First 3 bytes
    nic_id = mac_example[9:]  # Last 3 bytes

    print("MAC Address Structure:")
    print(f"  Example: {mac_example}")
    print(f"  Total: 48 bits (6 bytes), written in hexadecimal")
    print(f"\n  OUI (Organizationally Unique Identifier): {oui}")
    print(f"    - First 3 bytes (24 bits)")
    print(f"    - Assigned by IEEE to each manufacturer")
    print(f"    - Identifies the vendor (e.g., Cisco, Intel, Apple)")
    print(f"\n  NIC ID (Device Identifier): {nic_id}")
    print(f"    - Last 3 bytes (24 bits)")
    print(f"    - Assigned by manufacturer to each NIC")
    print(f"    - Unique within that vendor's range")

    # Show some known OUIs
    known_ouis = {
        "00:00:0C": "Cisco",
        "00:0C:29": "VMware",
        "00:50:56": "VMware",
        "00:25:00": "Apple",
        "AC:DE:48": "Intel",
    }
    print("\n  Common OUI examples:")
    for oui_prefix, vendor in known_ouis.items():
        print(f"    {oui_prefix} -> {vendor}")


def exercise_2():
    """
    Problem: Which of the following is a broadcast MAC address?
    (a) 00:00:00:00:00:00
    (b) FF:FF:FF:FF:FF:FF
    (c) 01:00:5E:00:00:01
    (d) AA:BB:CC:DD:EE:FF
    """
    addresses = {
        "00:00:00:00:00:00": "Invalid/Unspecified (all zeros)",
        "FF:FF:FF:FF:FF:FF": "BROADCAST - sent to ALL devices on the LAN",
        "01:00:5E:00:00:01": "Multicast address (IGMP all-hosts group)",
        "AA:BB:CC:DD:EE:FF": "Regular unicast address",
    }

    print("MAC Address Analysis:")
    for mac, description in addresses.items():
        is_broadcast = "<<< ANSWER" if mac == "FF:FF:FF:FF:FF:FF" else ""
        print(f"  {mac} : {description} {is_broadcast}")

    print("\nAnswer: (b) FF:FF:FF:FF:FF:FF is the broadcast MAC address.")
    print("All 48 bits set to 1 = broadcast to every device on the local network.")


def exercise_3():
    """
    Problem: How does a switch handle a frame with an unknown destination MAC?

    Reasoning: The switch's MAC address table learning process is fundamental
    to understanding how L2 forwarding works. Flooding unknown destinations
    ensures delivery while the switch learns the network topology.
    """
    class Switch:
        def __init__(self, ports=4):
            self.mac_table = {}
            self.num_ports = ports

        def process_frame(self, src_mac, dst_mac, ingress_port):
            """Simulate switch frame processing."""
            # Learn source MAC
            self.mac_table[src_mac] = ingress_port
            print(f"    Learned: {src_mac} on port {ingress_port}")

            # Forward based on destination
            if dst_mac == "FF:FF:FF:FF:FF:FF":
                flood_ports = [p for p in range(self.num_ports) if p != ingress_port]
                print(f"    Broadcast: flood to ports {flood_ports}")
                return flood_ports
            elif dst_mac in self.mac_table:
                out_port = self.mac_table[dst_mac]
                print(f"    Known destination: forward to port {out_port}")
                return [out_port]
            else:
                flood_ports = [p for p in range(self.num_ports) if p != ingress_port]
                print(f"    Unknown destination: FLOOD to ports {flood_ports}")
                return flood_ports

    switch = Switch(ports=4)
    print("Switch MAC table learning and forwarding:")

    print("\n  Frame 1: PC-A (port 0) -> PC-C (unknown)")
    switch.process_frame("AA:AA:AA:AA:AA:AA", "CC:CC:CC:CC:CC:CC", 0)

    print("\n  Frame 2: PC-C (port 2) replies to PC-A")
    switch.process_frame("CC:CC:CC:CC:CC:CC", "AA:AA:AA:AA:AA:AA", 2)

    print("\n  Frame 3: PC-A -> PC-C again (now known)")
    switch.process_frame("AA:AA:AA:AA:AA:AA", "CC:CC:CC:CC:CC:CC", 0)

    print(f"\n  Final MAC table: {switch.mac_table}")


def exercise_4():
    """
    Problem: Explain why the minimum Ethernet frame size is 64 bytes
    in relation to CSMA/CD.

    Reasoning: CSMA/CD requires that a frame be long enough for the sender
    to detect a collision before finishing transmission. The round-trip time
    at 10Mbps over max distance determines the minimum frame size.
    """
    # At 10 Mbps, 2.5 km maximum network diameter
    speed_bps = 10e6        # 10 Mbps
    max_distance_m = 2500   # Maximum network diameter
    signal_speed = 2e8      # Signal propagation speed in cable

    # Round-trip time
    rtt = (2 * max_distance_m) / signal_speed

    # Minimum bits needed = speed * RTT
    min_bits = speed_bps * rtt
    min_bytes = min_bits / 8

    print("Why minimum Ethernet frame size is 64 bytes:")
    print(f"\n  Network parameters (original 10 Mbps Ethernet):")
    print(f"    Speed: {speed_bps / 1e6:.0f} Mbps")
    print(f"    Max distance: {max_distance_m} m")
    print(f"    Signal speed: {signal_speed:.0e} m/s")
    print(f"\n  Round-trip time = 2 * {max_distance_m} / {signal_speed:.0e}")
    print(f"    = {rtt * 1e6:.1f} microseconds")
    print(f"\n  Minimum bits = {speed_bps / 1e6:.0f} Mbps * {rtt * 1e6:.1f} us")
    print(f"    = {min_bits:.0f} bits = {min_bytes:.0f} bytes")
    print(f"\n  Rounded up to 512 bits = 64 bytes for practical implementation.")
    print(f"\n  If a frame is shorter than 64 bytes, the sender might finish")
    print(f"  transmitting before a collision signal returns from the far end,")
    print(f"  making collision detection impossible.")


def exercise_5():
    """
    Problem: For the given network, determine collision domains and broadcast domains.

    [PC1]--+--[HUB]--+--[PC2]
           |         |
           +--[SWITCH]--+--[PC3]
                        |
                   [ROUTER]
                        |
                  [SWITCH]--+--[PC4]
                             +--[PC5]
    """
    print("Network domain analysis:")
    print()
    print("  Network topology:")
    print("  [PC1]--+--[HUB]--+--[PC2]")
    print("         |         |")
    print("         +--[SWITCH]--+--[PC3]")
    print("                      |")
    print("                 [ROUTER]")
    print("                      |")
    print("                [SWITCH]--+--[PC4]")
    print("                           +--[PC5]")

    print("\n  Collision Domains: 5")
    print("    - HUB segment: PC1, PC2, and hub port to switch = 1 collision domain")
    print("      (Hub repeats to all ports, so all share one collision domain)")
    print("    - Switch port to PC3 = 1 collision domain")
    print("    - Switch port to Router = 1 collision domain")
    print("    - Switch port to PC4 = 1 collision domain")
    print("    - Switch port to PC5 = 1 collision domain")

    print("\n  Broadcast Domains: 2")
    print("    - Upper segment (PC1, PC2, PC3, Hub, Switch) = 1 broadcast domain")
    print("    - Lower segment (PC4, PC5, Switch) = 1 broadcast domain")
    print("    - Router separates broadcast domains")


def exercise_6():
    """
    Problem: Explain how ARP spoofing attacks work and suggest defense methods.

    Reasoning: ARP has no authentication mechanism, making it vulnerable to
    spoofing attacks where an attacker sends false ARP replies to redirect traffic.
    """
    print("ARP Spoofing Attack:")
    print("=" * 50)

    attack_steps = [
        "Attacker sends forged ARP Reply to victim",
        "Reply maps gateway's IP to attacker's MAC address",
        "Victim updates ARP cache with false entry",
        "Victim sends traffic destined for gateway to attacker instead",
        "Attacker forwards traffic to real gateway (man-in-the-middle)",
        "Attacker can intercept, modify, or drop packets",
    ]

    print("\n  Attack steps:")
    for i, step in enumerate(attack_steps, 1):
        print(f"    {i}. {step}")

    defenses = [
        ("Static ARP entries", "Manually configure ARP for critical hosts (doesn't scale)"),
        ("Dynamic ARP Inspection (DAI)", "Switch validates ARP packets against DHCP snooping table"),
        ("802.1X Authentication", "Port-based access control prevents unauthorized devices"),
        ("ARP monitoring tools", "arpwatch or similar tools detect ARP table changes"),
        ("VPN/Encryption", "Even if traffic is intercepted, content is encrypted"),
    ]

    print("\n  Defense methods:")
    for defense, detail in defenses:
        print(f"    - {defense}: {detail}")


def exercise_7():
    """
    Problem: When PC A (192.168.1.10) transmits data to PC B (192.168.1.20) for
    the first time, explain step-by-step what happens at L2 and L3.
    """
    steps = [
        ("L3", "PC A checks: is 192.168.1.20 in the same subnet as 192.168.1.10? Yes."),
        ("L3", "PC A checks ARP cache for 192.168.1.20's MAC address. Not found."),
        ("L2", "PC A broadcasts ARP Request: 'Who has 192.168.1.20? Tell 192.168.1.10'"),
        ("L2", "ARP Request frame: Src MAC=AA:AA, Dst MAC=FF:FF:FF:FF:FF:FF"),
        ("L2", "Switch floods ARP Request to all ports (broadcast)"),
        ("L2", "PC B receives ARP Request, recognizes its own IP"),
        ("L2", "PC B sends ARP Reply: 'I am 192.168.1.20, my MAC is BB:BB'"),
        ("L3", "PC A updates ARP cache: 192.168.1.20 -> BB:BB"),
        ("L2", "PC A creates Ethernet frame: Src MAC=AA:AA, Dst MAC=BB:BB"),
        ("L3", "IP packet encapsulated: Src IP=192.168.1.10, Dst IP=192.168.1.20"),
        ("L2", "Switch forwards frame to PC B's port via MAC table lookup"),
    ]

    print("First-time communication: PC A (192.168.1.10) -> PC B (192.168.1.20)")
    for i, (layer, description) in enumerate(steps, 1):
        print(f"  {i:2d}. [{layer}] {description}")


def exercise_8():
    """
    Problem: Explain Store-and-Forward vs Cut-Through switching methods.
    """
    methods = {
        "Store-and-Forward": {
            "process": "Receive entire frame -> Check CRC -> Forward if valid",
            "advantages": [
                "Error detection: drops corrupted frames",
                "Supports speed conversion between ports",
                "Most reliable method",
            ],
            "disadvantages": [
                "Higher latency (must buffer entire frame)",
                "Latency proportional to frame size",
            ],
            "used_by": "Most modern enterprise switches (default)",
        },
        "Cut-Through": {
            "process": "Read destination MAC (first 6 bytes) -> Forward immediately",
            "advantages": [
                "Very low latency (forwards as soon as dst MAC read)",
                "Fixed latency regardless of frame size",
            ],
            "disadvantages": [
                "May forward corrupted frames",
                "Cannot convert between port speeds",
            ],
            "used_by": "Low-latency environments (HPC, trading systems)",
        },
    }

    print("Switching Methods Comparison:")
    for method, info in methods.items():
        print(f"\n  {method}:")
        print(f"    Process: {info['process']}")
        print("    Advantages:")
        for adv in info["advantages"]:
            print(f"      + {adv}")
        print("    Disadvantages:")
        for dis in info["disadvantages"]:
            print(f"      - {dis}")
        print(f"    Used by: {info['used_by']}")


def exercise_9():
    """
    Problem: Explain why CSMA/CD is not used in full-duplex mode.

    Reasoning: Full-duplex separates TX and RX onto different wire pairs,
    eliminating the physical possibility of collisions.
    """
    print("Why CSMA/CD is unnecessary in full-duplex:")
    print()
    print("  Half-duplex (with CSMA/CD):")
    print("    - TX and RX share the same wire")
    print("    - Only one device can transmit at a time")
    print("    - Collision possible -> CSMA/CD detects and handles")
    print()
    print("  Full-duplex (no CSMA/CD needed):")
    print("    - Separate wire pairs for TX and RX")
    print("    - Both devices can transmit simultaneously")
    print("    - No collisions are physically possible")
    print("    - Each switch port is a separate segment")
    print()
    print("  Modern switched networks:")
    print("    - Each port is a separate collision domain")
    print("    - Point-to-point links (switch <-> device)")
    print("    - Full-duplex is the default")
    print("    - CSMA/CD is effectively obsolete")


def exercise_10():
    """
    Problem: Explain how VLANs separate broadcast domains.

    Reasoning: VLANs allow a single physical switch to be logically divided
    into multiple broadcast domains, improving security and performance
    without requiring additional hardware.
    """
    class VLANSwitch:
        def __init__(self, ports=8):
            self.port_vlans = {}  # port -> VLAN ID
            self.mac_table = {}   # (mac, vlan) -> port

        def assign_port(self, port, vlan_id):
            self.port_vlans[port] = vlan_id

        def send_broadcast(self, src_port, src_mac):
            """Broadcast only reaches ports in the same VLAN."""
            src_vlan = self.port_vlans.get(src_port)
            flood_ports = [
                p for p, v in self.port_vlans.items()
                if v == src_vlan and p != src_port
            ]
            return flood_ports

    switch = VLANSwitch()
    # Engineering department: VLAN 10
    switch.assign_port(0, 10)  # PC1
    switch.assign_port(1, 10)  # PC2
    switch.assign_port(2, 10)  # PC3

    # Sales department: VLAN 20
    switch.assign_port(3, 20)  # PC4
    switch.assign_port(4, 20)  # PC5

    print("VLAN Broadcast Domain Separation:")
    print(f"\n  Port assignments:")
    print(f"    VLAN 10 (Engineering): Ports 0, 1, 2")
    print(f"    VLAN 20 (Sales):       Ports 3, 4")

    # Test broadcasts
    print(f"\n  Broadcast from PC1 (VLAN 10, Port 0):")
    print(f"    Reaches ports: {switch.send_broadcast(0, 'AA:AA')}")
    print(f"    Only VLAN 10 ports receive the broadcast")

    print(f"\n  Broadcast from PC4 (VLAN 20, Port 3):")
    print(f"    Reaches ports: {switch.send_broadcast(3, 'DD:DD')}")
    print(f"    Only VLAN 20 ports receive the broadcast")

    print(f"\n  Key points:")
    print(f"    - Broadcasts are confined to the same VLAN")
    print(f"    - Inter-VLAN communication requires a router (L3)")
    print(f"    - Reduces broadcast domain size -> improved performance")
    print(f"    - Logical separation -> improved security")


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
