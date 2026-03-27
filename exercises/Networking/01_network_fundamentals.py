"""
Exercises for Lesson 01: Network Fundamentals
Topic: Networking
Solutions to practice problems from the lesson.
"""


def exercise_1():
    """
    Problem: Arrange network types from smallest to largest range.
    WAN, LAN, PAN, MAN

    Reasoning: Network classification by geographic coverage helps
    choose the right technology stack for each scale of connectivity.
    """
    network_types = {
        "PAN": {"range_m": 10, "description": "Personal Area Network (Bluetooth, USB)"},
        "LAN": {"range_m": 1000, "description": "Local Area Network (Office, Home)"},
        "MAN": {"range_m": 50000, "description": "Metropolitan Area Network (City-wide)"},
        "WAN": {"range_m": float("inf"), "description": "Wide Area Network (Internet)"},
    }

    # Sort by range
    sorted_types = sorted(network_types.items(), key=lambda x: x[1]["range_m"])

    print("Network types ordered by range (smallest to largest):")
    for name, info in sorted_types:
        range_str = f"{info['range_m']}m" if info["range_m"] != float("inf") else "Unlimited"
        print(f"  {name} ({range_str}) - {info['description']}")

    # Verify answer
    answer = [name for name, _ in sorted_types]
    assert answer == ["PAN", "LAN", "MAN", "WAN"]
    print("\nAnswer: PAN < LAN < MAN < WAN")


def exercise_2():
    """
    Problem: What happens when the central device in a star topology fails?

    Reasoning: Star topology's single point of failure at the central device
    (hub/switch) is its main weakness despite being the most common topology
    for modern LANs due to easy management and fault isolation per link.
    """
    class StarTopology:
        def __init__(self):
            self.central_device = {"name": "Switch", "active": True}
            self.nodes = ["PC1", "PC2", "PC3", "PC4", "PC5"]

        def check_connectivity(self, src, dst):
            """All traffic routes through central device in star topology."""
            if not self.central_device["active"]:
                return False
            return src in self.nodes and dst in self.nodes

        def fail_central(self):
            self.central_device["active"] = False

    topology = StarTopology()

    print("Star topology with central switch and 5 PCs:")
    print(f"  PC1 -> PC3 connectivity: {topology.check_connectivity('PC1', 'PC3')}")
    print(f"  PC2 -> PC5 connectivity: {topology.check_connectivity('PC2', 'PC5')}")

    print("\nCentral switch fails...")
    topology.fail_central()
    print(f"  PC1 -> PC3 connectivity: {topology.check_connectivity('PC1', 'PC3')}")
    print(f"  PC2 -> PC5 connectivity: {topology.check_connectivity('PC2', 'PC5')}")

    print("\nAnswer: The entire network goes down (single point of failure).")
    print("All nodes lose connectivity because every path goes through the central device.")


def exercise_3():
    """
    Problem: Which of the following is an L2 device?
    (a) Hub, (b) Switch, (c) Router, (d) Repeater

    Reasoning: OSI layer classification determines what information a device
    uses to make forwarding decisions. L2 devices use MAC addresses.
    """
    devices = {
        "Hub":      {"layer": 1, "uses": "Electrical signals", "forwarding": "Broadcast to all ports"},
        "Switch":   {"layer": 2, "uses": "MAC addresses", "forwarding": "Unicast via MAC table"},
        "Router":   {"layer": 3, "uses": "IP addresses", "forwarding": "Route via routing table"},
        "Repeater": {"layer": 1, "uses": "Electrical signals", "forwarding": "Signal amplification"},
    }

    print("Network devices by OSI layer:")
    for name, info in devices.items():
        marker = " <-- L2 device" if info["layer"] == 2 else ""
        print(f"  {name}: Layer {info['layer']} - Uses {info['uses']}, {info['forwarding']}{marker}")

    print("\nAnswer: (b) Switch")
    print("A switch operates at Layer 2 (Data Link), using MAC addresses for forwarding decisions.")


def exercise_4():
    """
    Problem: Explain the difference between packet switching and circuit switching.

    Reasoning: Understanding switching methods is fundamental to networking.
    The Internet uses packet switching for its efficiency and resilience,
    while traditional telephony used circuit switching for guaranteed QoS.
    """
    comparison = {
        "Aspect": [
            "Connection Setup",
            "Bandwidth",
            "Resource Efficiency",
            "Latency",
            "Reliability",
            "Use Cases",
        ],
        "Circuit Switching": [
            "Required (dedicated path established)",
            "Fixed allocation during connection",
            "Low (reserved even when idle)",
            "Constant and predictable",
            "High (guaranteed path)",
            "Traditional telephony, ISDN",
        ],
        "Packet Switching": [
            "Not required",
            "Dynamic allocation (shared)",
            "High (statistical multiplexing)",
            "Variable (queuing delays)",
            "Protocol-dependent (TCP adds reliability)",
            "Internet, VoIP, modern data networks",
        ],
    }

    print("Circuit Switching vs Packet Switching:")
    print("-" * 80)
    for i, aspect in enumerate(comparison["Aspect"]):
        print(f"\n  {aspect}:")
        print(f"    Circuit: {comparison['Circuit Switching'][i]}")
        print(f"    Packet:  {comparison['Packet Switching'][i]}")

    # Simulate packet switching
    print("\n\nPacket switching simulation:")
    message = "Hello, World! This is a message."
    packet_size = 8
    packets = [message[i:i + packet_size] for i in range(0, len(message), packet_size)]

    for i, pkt in enumerate(packets):
        # Simulate different paths
        path = "A -> R1 -> B" if i % 2 == 0 else "A -> R1 -> R2 -> B"
        print(f"  Packet {i + 1}: '{pkt}' via {path}")


def exercise_5():
    """
    Problem: Choose appropriate topology for each scenario:
    - A small office with 10 people
    - A bank's ATM network
    - Intercontinental connection via submarine cable

    Reasoning: Topology selection depends on reliability requirements,
    cost constraints, and scalability needs of the specific use case.
    """
    scenarios = [
        {
            "scenario": "Small office with 10 people",
            "best": "Star",
            "reason": "Easy management, fault isolation per link, cost-effective for small scale",
        },
        {
            "scenario": "Bank's ATM network",
            "best": "Mesh (or Star with redundancy)",
            "reason": "High reliability critical for financial transactions, redundant paths for failover",
        },
        {
            "scenario": "Intercontinental connection via submarine cable",
            "best": "Mesh",
            "reason": "Multiple paths for resilience, high bandwidth distribution, no single point of failure",
        },
    ]

    # Topology properties for reference
    topologies = {
        "Bus":    {"reliability": "Low", "cost": "Low", "scalability": "Low"},
        "Star":   {"reliability": "Medium", "cost": "Medium", "scalability": "High"},
        "Ring":   {"reliability": "Medium", "cost": "Medium", "scalability": "Medium"},
        "Mesh":   {"reliability": "High", "cost": "High", "scalability": "High"},
        "Hybrid": {"reliability": "High", "cost": "Medium", "scalability": "High"},
    }

    print("Topology selection for each scenario:")
    for s in scenarios:
        print(f"\n  Scenario: {s['scenario']}")
        print(f"  Recommended: {s['best']}")
        print(f"  Reason: {s['reason']}")


def exercise_6():
    """
    Problem: Describe two situations each where client-server model is appropriate
    and where P2P model is appropriate.

    Reasoning: The choice between client-server and P2P depends on requirements
    for centralized control, security, scalability, and data consistency.
    """
    models = {
        "Client-Server": {
            "situations": [
                ("Online Banking", "Requires centralized control, transaction integrity, audit trail"),
                ("Corporate Email", "Centralized management, security policies, compliance needs"),
            ],
            "properties": {
                "control": "Centralized",
                "scalability": "Limited (server bottleneck)",
                "security": "Easier to manage",
                "consistency": "Strong (single source of truth)",
            },
        },
        "P2P": {
            "situations": [
                ("File Sharing (BitTorrent)", "Distributed load, more peers = more bandwidth"),
                ("Cryptocurrency (Bitcoin)", "Distributed ledger, no central authority needed"),
            ],
            "properties": {
                "control": "Distributed",
                "scalability": "Excellent (grows with users)",
                "security": "Harder to manage",
                "consistency": "Eventual (consensus-based)",
            },
        },
    }

    for model_name, model_info in models.items():
        print(f"\n{model_name} Model:")
        for situation, reason in model_info["situations"]:
            print(f"  - {situation}: {reason}")
        print("  Properties:")
        for prop, value in model_info["properties"].items():
            print(f"    {prop}: {value}")


def exercise_7():
    """
    Problem: Given network diagram, determine device paths.
    [PC1] --+
            |
    [PC2] --+--[SwitchA]--[Router]--[SwitchB]--+--[PC5]
            |                                   |
    [PC3] --+                                   +--[PC6]

    (a) PC1 to PC3: through which devices?
    (b) PC1 to PC5: through which devices?
    (c) If SwitchA fails, which PCs are affected?
    """
    # Model the network as a graph of connections
    network = {
        "PC1": ["SwitchA"],
        "PC2": ["SwitchA"],
        "PC3": ["SwitchA"],
        "SwitchA": ["PC1", "PC2", "PC3", "Router"],
        "Router": ["SwitchA", "SwitchB"],
        "SwitchB": ["Router", "PC5", "PC6"],
        "PC5": ["SwitchB"],
        "PC6": ["SwitchB"],
    }

    def find_path(graph, start, end, visited=None):
        """BFS to find path between two nodes."""
        if visited is None:
            visited = set()
        if start == end:
            return [start]
        visited.add(start)
        for neighbor in graph.get(start, []):
            if neighbor not in visited:
                path = find_path(graph, neighbor, end, visited)
                if path:
                    return [start] + path
        return None

    print("Network path analysis:")

    # (a) PC1 to PC3
    path_a = find_path(network, "PC1", "PC3")
    devices_a = [d for d in path_a if d.startswith("Switch") or d.startswith("Router")]
    print(f"\n  (a) PC1 -> PC3 path: {' -> '.join(path_a)}")
    print(f"      Devices traversed: {', '.join(devices_a)}")
    print(f"      Answer: SwitchA only (same LAN segment)")

    # (b) PC1 to PC5
    path_b = find_path(network, "PC1", "PC5")
    devices_b = [d for d in path_b if d.startswith("Switch") or d.startswith("Router")]
    print(f"\n  (b) PC1 -> PC5 path: {' -> '.join(path_b)}")
    print(f"      Devices traversed: {', '.join(devices_b)}")
    print(f"      Answer: SwitchA -> Router -> SwitchB")

    # (c) SwitchA failure impact
    print(f"\n  (c) If SwitchA fails:")
    affected = [node for node in network["SwitchA"] if node.startswith("PC")]
    print(f"      Affected PCs: {', '.join(affected)}")
    print(f"      Answer: PC1, PC2, PC3 lose all connectivity")


def exercise_8():
    """
    Problem: Explain the role of TCP/IP in the evolution from ARPANET to the Internet.

    Reasoning: TCP/IP's adoption as the standard protocol was the pivotal moment
    that transformed ARPANET from a research network into the global Internet.
    """
    timeline = [
        ("1969", "ARPANET", "First packet-switched network, used NCP protocol"),
        ("1974", "TCP/IP Proposed", "Vint Cerf and Bob Kahn publish TCP/IP design"),
        ("1983", "Flag Day", "ARPANET switches from NCP to TCP/IP"),
        ("1990s", "WWW", "Tim Berners-Lee creates HTTP on top of TCP/IP"),
        ("Today", "Internet", "TCP/IP is the universal protocol connecting billions of devices"),
    ]

    print("ARPANET to Internet evolution:")
    print("=" * 70)
    for year, event, description in timeline:
        print(f"  [{year}] {event}")
        print(f"          {description}")
    print("\nKey role of TCP/IP:")
    print("  - Provided standardized communication between heterogeneous networks")
    print("  - Open protocol allowed any network to join")
    print("  - Layered architecture enabled independent evolution of components")
    print("  - Made the Internet possible by enabling inter-network communication")


def exercise_9():
    """
    Problem: Explain at least 3 reasons why mesh topology is suitable
    for the Internet backbone.

    Reasoning: The Internet backbone requires the highest levels of reliability,
    bandwidth, and fault tolerance, which mesh topology provides through redundancy.
    """
    reasons = [
        ("Redundant Paths", "Multiple alternative routes when one link fails"),
        ("High Bandwidth", "Traffic distributed across many links, avoiding bottlenecks"),
        ("Fault Tolerance", "No single point of failure can bring down the network"),
        ("Load Distribution", "Traffic engineering spreads load across available paths"),
        ("Scalability", "New nodes can be added without disrupting existing connections"),
    ]

    # Demonstrate mesh connectivity
    nodes = ["NYC", "London", "Tokyo", "Sydney", "Seoul"]
    # Full mesh: n(n-1)/2 links
    n = len(nodes)
    total_links = n * (n - 1) // 2

    print("Why mesh topology suits the Internet backbone:")
    for i, (reason, explanation) in enumerate(reasons, 1):
        print(f"\n  {i}. {reason}")
        print(f"     {explanation}")

    print(f"\n\nFull mesh example with {n} backbone nodes:")
    print(f"  Nodes: {', '.join(nodes)}")
    print(f"  Total links: {total_links}")
    print(f"  Formula: n(n-1)/2 = {n}({n-1})/2 = {total_links}")


def exercise_10():
    """
    Problem: Explain why bus topology is rarely used in modern networks.

    Reasoning: Bus topology's limitations become critical as network speeds
    and device counts have grown dramatically since its heyday in early Ethernet.
    """
    limitations = [
        ("Single Point of Failure", "Cable break affects entire network - unacceptable for modern uptime requirements"),
        ("Collision Domain", "All devices share one collision domain - performance degrades with more devices"),
        ("Troubleshooting", "Difficult to isolate faults on a shared bus cable"),
        ("Bandwidth Sharing", "All devices compete for the same bandwidth - incompatible with modern high-speed needs"),
        ("Security", "All traffic visible to all devices on the bus (promiscuous mode sniffing)"),
        ("Scalability", "Adding devices increases collisions and reduces per-device bandwidth"),
    ]

    print("Why bus topology is obsolete in modern networks:")
    for i, (issue, detail) in enumerate(limitations, 1):
        print(f"\n  {i}. {issue}")
        print(f"     {detail}")

    print("\n\nModern alternative: Star topology with switches")
    print("  - Each port is a separate collision domain")
    print("  - Dedicated bandwidth per device")
    print("  - Easy to add/remove devices")
    print("  - Fault isolation per link")


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
