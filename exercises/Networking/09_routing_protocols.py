"""
Exercises for Lesson 09: Routing Protocols
Topic: Networking
Solutions to practice problems from the lesson.
"""


def exercise_1():
    """
    Problem 1: Protocol Feature Matching
    Match features with RIP, OSPF, or BGP.
    a) Maximum hop count 15
    b) Uses Dijkstra algorithm
    c) Used for inter-AS routing
    d) Sends complete routing table every 30 seconds
    e) Area-based hierarchical structure
    f) Uses TCP port 179

    Reasoning: Each routing protocol has distinct characteristics that make it
    suitable for different network sizes and architectures.
    """
    features = {
        "Maximum hop count 15": ("RIP", "Distance-vector, simple metric limits scalability"),
        "Uses Dijkstra algorithm": ("OSPF", "Link-state protocol builds shortest path tree"),
        "Used for inter-AS routing": ("BGP", "Only EGP in use today, connects autonomous systems"),
        "Sends complete routing table every 30 sec": ("RIP", "Periodic full updates, bandwidth-intensive"),
        "Area-based hierarchical structure": ("OSPF", "Areas reduce LSA flooding, Area 0 is backbone"),
        "Uses TCP port 179": ("BGP", "Reliable transport for routing updates between peers"),
    }

    print("Routing Protocol Feature Matching:")
    for feature, (protocol, explanation) in features.items():
        print(f"\n  {feature}")
        print(f"    -> {protocol}: {explanation}")

    # Summary comparison
    print("\n\nProtocol Summary:")
    protocols = {
        "RIP": {"type": "Distance Vector", "metric": "Hop count (max 15)", "scope": "Small networks",
                "convergence": "Slow (30s updates)", "algorithm": "Bellman-Ford"},
        "OSPF": {"type": "Link State", "metric": "Cost (bandwidth-based)", "scope": "Large enterprise",
                 "convergence": "Fast (triggered updates)", "algorithm": "Dijkstra (SPF)"},
        "BGP": {"type": "Path Vector", "metric": "AS_PATH + policies", "scope": "Internet (inter-AS)",
                "convergence": "Slow (policy-based)", "algorithm": "Best path selection"},
    }
    for name, info in protocols.items():
        print(f"\n  {name}:")
        for key, value in info.items():
            print(f"    {key:15s}: {value}")


def exercise_2():
    """
    Problem 2: OSPF Cost Calculation
    Reference bandwidth: 100 Mbps
    Path: R1 --(FastEthernet 100Mbps)-- R2 --(Serial T1 1.544Mbps)-- R3 --(GigE 1000Mbps)-- R4

    OSPF Cost = Reference Bandwidth / Interface Bandwidth
    Minimum cost is 1 (even for interfaces faster than reference).

    Reasoning: OSPF uses cost inversely proportional to bandwidth, automatically
    preferring higher-bandwidth paths. The reference bandwidth should be set
    high enough to differentiate between fast interfaces.
    """
    reference_bw = 100  # Mbps

    links = [
        ("R1-R2", "FastEthernet", 100),
        ("R2-R3", "Serial T1", 1.544),
        ("R3-R4", "GigabitEthernet", 1000),
    ]

    print(f"OSPF Cost Calculation (Reference BW: {reference_bw} Mbps):")
    print(f"  Formula: Cost = Reference BW / Interface BW (minimum 1)")
    print()

    total_cost = 0
    for link_name, iface_type, bw in links:
        cost = max(1, int(reference_bw / bw))
        total_cost += cost
        print(f"  {link_name} ({iface_type}, {bw} Mbps):")
        print(f"    Cost = {reference_bw} / {bw} = {reference_bw / bw:.1f} -> {cost}")

    print(f"\n  Total path cost R1 -> R4: {total_cost}")
    print(f"\n  Note: With default reference BW of 100 Mbps,")
    print(f"  FastEthernet and GigabitEthernet have the same cost (1).")
    print(f"  To differentiate, set reference BW higher:")
    print(f"    auto-cost reference-bandwidth 10000  (10 Gbps)")
    print(f"    Then: GigE=10, FastEth=100, T1=6477")


def exercise_3():
    """
    Problem 3: BGP Path Selection
    Route A: AS_PATH: 100 200 300, LOCAL_PREF: 150, MED: 100
    Route B: AS_PATH: 400 500, LOCAL_PREF: 150, MED: 50

    Reasoning: BGP path selection follows a strict priority order.
    LOCAL_PREF is checked first, then AS_PATH length.
    MED is only compared between routes from the SAME neighbor AS.
    """
    routes = {
        "Route A": {"as_path": [100, 200, 300], "local_pref": 150, "med": 100},
        "Route B": {"as_path": [400, 500], "local_pref": 150, "med": 50},
    }

    print("BGP Path Selection:")
    for name, attrs in routes.items():
        print(f"\n  {name}:")
        print(f"    AS_PATH: {attrs['as_path']} (length: {len(attrs['as_path'])})")
        print(f"    LOCAL_PREF: {attrs['local_pref']}")
        print(f"    MED: {attrs['med']}")

    # BGP selection process (simplified)
    selection_steps = [
        ("1. Highest LOCAL_PREF", "A=150, B=150", "TIE"),
        ("2. Shortest AS_PATH", "A=3 hops, B=2 hops", "Route B WINS"),
    ]

    print("\n  Selection process:")
    for step, comparison, result in selection_steps:
        print(f"    {step}: {comparison} -> {result}")

    print(f"\n  Answer: Route B is selected (shorter AS_PATH: 2 vs 3)")
    print(f"  Note: MED (100 vs 50) is NOT compared here because")
    print(f"  the routes come from different neighbor ASes (100 vs 400).")
    print(f"  MED is only used to compare routes from the SAME neighbor AS.")


def exercise_4():
    """
    Problem 4: Routing Protocol Selection
    a) Small office with 10 routers
    b) Large enterprise with 500 routers
    c) Connection between two ISPs
    d) Branch office with single path

    Reasoning: Protocol selection depends on network size, complexity,
    convergence requirements, and administrative boundaries.
    """
    scenarios = [
        {
            "scenario": "Small office with 10 routers",
            "recommended": "RIP or Static Routing",
            "reasoning": [
                "Simple network, few routers within hop count limit",
                "Easy to configure and manage",
                "OSPF would be overkill for this scale",
            ],
        },
        {
            "scenario": "Large enterprise with 500 routers",
            "recommended": "OSPF",
            "reasoning": [
                "Fast convergence with triggered updates",
                "Scalable via area hierarchy (divide into areas)",
                "VLSM/CIDR support for efficient addressing",
                "RIP cannot handle >15 hops; BGP is for inter-AS",
            ],
        },
        {
            "scenario": "Connection between two ISPs",
            "recommended": "BGP (eBGP)",
            "reasoning": [
                "Standard for inter-AS routing (the only EGP in use)",
                "Policy-based routing for traffic engineering",
                "Controls which routes are advertised to peers",
            ],
        },
        {
            "scenario": "Branch office with single path",
            "recommended": "Static Routing + Default Route",
            "reasoning": [
                "Only one path exists, no routing decisions to make",
                "Dynamic protocol would be wasted overhead",
                "Default route sends everything to headquarters",
                "Simplest and most resource-efficient option",
            ],
        },
    ]

    print("Routing Protocol Selection for Each Scenario:")
    for s in scenarios:
        print(f"\n  Scenario: {s['scenario']}")
        print(f"  Recommended: {s['recommended']}")
        print(f"  Reasoning:")
        for r in s["reasoning"]:
            print(f"    - {r}")


if __name__ == "__main__":
    exercises = [exercise_1, exercise_2, exercise_3, exercise_4]
    for i, ex in enumerate(exercises, 1):
        print(f"\n{'=' * 60}")
        print(f"=== Exercise {i} ===")
        print(f"{'=' * 60}")
        ex()

    print(f"\n{'=' * 60}")
    print("All exercises completed!")
