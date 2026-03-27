"""
Exercises for Lesson 08: Routing Basics
Topic: Networking
Solutions to practice problems from the lesson.
"""


def ip_to_int(ip):
    """Convert dotted-decimal IP to 32-bit integer."""
    octets = [int(o) for o in ip.split(".")]
    return (octets[0] << 24) | (octets[1] << 16) | (octets[2] << 8) | octets[3]


def int_to_ip(n):
    """Convert 32-bit integer to dotted-decimal IP."""
    return f"{(n >> 24) & 0xFF}.{(n >> 16) & 0xFF}.{(n >> 8) & 0xFF}.{n & 0xFF}"


def cidr_to_mask(prefix_len):
    """Convert CIDR prefix length to subnet mask integer."""
    return (0xFFFFFFFF << (32 - prefix_len)) & 0xFFFFFFFF


def exercise_1():
    """
    Problem 1: Routing Table Analysis
    Analyze the routing table and determine next hops.

    Router# show ip route
    C    192.168.1.0/24 is directly connected, Eth0
    C    192.168.2.0/24 is directly connected, Eth1
    S    10.0.0.0/8 [1/0] via 192.168.1.254
    S    172.16.0.0/16 [1/0] via 192.168.2.254
    O    172.16.10.0/24 [110/20] via 192.168.2.253
    S*   0.0.0.0/0 [1/0] via 192.168.1.1

    Reasoning: Routers use longest prefix match to select the most specific
    route. Route type codes indicate how the route was learned.
    """
    routing_table = [
        {"type": "C", "network": "192.168.1.0", "prefix": 24, "next_hop": "directly connected", "iface": "Eth0", "ad": 0},
        {"type": "C", "network": "192.168.2.0", "prefix": 24, "next_hop": "directly connected", "iface": "Eth1", "ad": 0},
        {"type": "S", "network": "10.0.0.0", "prefix": 8, "next_hop": "192.168.1.254", "iface": None, "ad": 1},
        {"type": "S", "network": "172.16.0.0", "prefix": 16, "next_hop": "192.168.2.254", "iface": None, "ad": 1},
        {"type": "O", "network": "172.16.10.0", "prefix": 24, "next_hop": "192.168.2.253", "iface": None, "ad": 110},
        {"type": "S*", "network": "0.0.0.0", "prefix": 0, "next_hop": "192.168.1.1", "iface": None, "ad": 1},
    ]

    def lookup(dest_ip, table):
        """Longest prefix match lookup."""
        dest_int = ip_to_int(dest_ip)
        best_match = None
        best_prefix = -1
        for route in table:
            net_int = ip_to_int(route["network"])
            mask_int = cidr_to_mask(route["prefix"])
            if (dest_int & mask_int) == net_int and route["prefix"] > best_prefix:
                best_match = route
                best_prefix = route["prefix"]
        return best_match

    queries = [
        ("10.10.10.10", "192.168.1.254", "matches 10.0.0.0/8 (static route)"),
        ("172.16.10.50", "192.168.2.253", "matches 172.16.10.0/24 (OSPF, more specific than /16)"),
        ("8.8.8.8", "192.168.1.1", "matches default route 0.0.0.0/0"),
    ]

    print("Routing Table Analysis:")
    for route in routing_table:
        print(f"  {route['type']:3s} {route['network']}/{route['prefix']:<3d} "
              f"-> {route['next_hop']}")

    print("\nRoute lookups:")
    for dest, expected_nh, explanation in queries:
        match = lookup(dest, routing_table)
        print(f"\n  Destination: {dest}")
        print(f"    Matched: {match['network']}/{match['prefix']} ({match['type']})")
        print(f"    Next hop: {match['next_hop']}")
        print(f"    Reason: {explanation}")

    print("\nRoute type codes:")
    codes = {"C": "Connected (directly connected)", "S": "Static (manually configured)",
             "O": "OSPF (dynamic routing protocol)", "S*": "Static default route"}
    for code, meaning in codes.items():
        print(f"  {code}: {meaning}")


def exercise_2():
    """
    Problem 2: Static Routing Configuration
    Write static routes for R1 and R2.

    Network A          R1           R2          Network B
    192.168.1.0/24 --[.1]--[.2]--[.1]--[.2]-- 10.0.0.0/24
                       192.168.100.0/30

    Reasoning: Each router needs a route to reach the network on the other
    side. The next hop is the peer router's interface on the connecting link.
    """
    print("Static Routing Configuration:")
    print()
    print("  Network topology:")
    print("  Net A (192.168.1.0/24) -- R1 -- [192.168.100.0/30] -- R2 -- Net B (10.0.0.0/24)")
    print("                           .1  .2                    .1  .2")

    print("\n  R1 configuration:")
    print("    # R1 needs a route to Network B (10.0.0.0/24)")
    print("    # Next hop is R2's interface on the connecting link")
    print("    ip route 10.0.0.0 255.255.255.0 192.168.100.2")

    print("\n  R2 configuration:")
    print("    # R2 needs a route to Network A (192.168.1.0/24)")
    print("    # Next hop is R1's interface on the connecting link")
    print("    ip route 192.168.1.0 255.255.255.0 192.168.100.1")

    # Verify with simulation
    print("\n  Verification simulation:")
    r1_routes = [
        {"network": "192.168.1.0", "prefix": 24, "next_hop": "directly connected"},
        {"network": "192.168.100.0", "prefix": 30, "next_hop": "directly connected"},
        {"network": "10.0.0.0", "prefix": 24, "next_hop": "192.168.100.2"},
    ]
    r2_routes = [
        {"network": "10.0.0.0", "prefix": 24, "next_hop": "directly connected"},
        {"network": "192.168.100.0", "prefix": 30, "next_hop": "directly connected"},
        {"network": "192.168.1.0", "prefix": 24, "next_hop": "192.168.100.1"},
    ]
    print(f"    R1 can reach 10.0.0.0/24 via {r1_routes[2]['next_hop']}")
    print(f"    R2 can reach 192.168.1.0/24 via {r2_routes[2]['next_hop']}")


def exercise_3():
    """
    Problem 3: Longest Prefix Match
    Determine next hop for each destination using the routing table.

    Routing Table:
    - 0.0.0.0/0       -> 10.0.0.1
    - 10.0.0.0/8      -> 10.0.0.2
    - 10.10.0.0/16    -> 10.0.0.3
    - 10.10.10.0/24   -> 10.0.0.4
    - 10.10.10.128/25 -> 10.0.0.5

    Reasoning: The longest prefix match algorithm is the core of IP routing.
    The most specific (longest prefix) matching route wins.
    """
    routes = [
        ("0.0.0.0", 0, "10.0.0.1"),
        ("10.0.0.0", 8, "10.0.0.2"),
        ("10.10.0.0", 16, "10.0.0.3"),
        ("10.10.10.0", 24, "10.0.0.4"),
        ("10.10.10.128", 25, "10.0.0.5"),
    ]

    def longest_prefix_match(dest_ip, routing_table):
        dest_int = ip_to_int(dest_ip)
        best_match = None
        best_prefix = -1
        for net, prefix, nh in routing_table:
            net_int = ip_to_int(net)
            mask_int = cidr_to_mask(prefix)
            if (dest_int & mask_int) == (net_int & mask_int) and prefix > best_prefix:
                best_match = (net, prefix, nh)
                best_prefix = prefix
        return best_match

    destinations = [
        ("10.10.10.200", "10.0.0.5", "matches /25 (200 is in 128-255 range)"),
        ("10.10.10.50", "10.0.0.4", "matches /24 (50 is in 0-127 range)"),
        ("10.10.20.100", "10.0.0.3", "matches /16 (10.10.x.x)"),
        ("10.20.30.40", "10.0.0.2", "matches /8 (10.x.x.x)"),
        ("8.8.8.8", "10.0.0.1", "matches default route /0"),
    ]

    print("Longest Prefix Match:")
    print(f"\n  Routing Table:")
    for net, prefix, nh in routes:
        print(f"    {net + '/' + str(prefix):20s} -> {nh}")

    print(f"\n  Lookups:")
    for dest, expected_nh, explanation in destinations:
        match = longest_prefix_match(dest, routes)
        assert match[2] == expected_nh, f"Expected {expected_nh}, got {match[2]}"
        print(f"    {dest:18s} -> {match[2]} (matched {match[0]}/{match[1]}, {explanation})")


def exercise_4():
    """
    Problem 4: Network Design
    Design routing for company with 3 branch offices.

            HQ (192.168.0.0/24)
                   |
              [Core Router]
             /      |      \\
        Branch A  Branch B  Branch C
        10.1.0/24 10.2.0/24 10.3.0/24

    Reasoning: Branch routers can use a default route pointing to the core
    router, simplifying configuration. The core router needs specific routes
    to each branch network.
    """
    print("Network Design - 3 Branch Offices:")
    print()
    print("  Topology:")
    print("       HQ (192.168.0.0/24)")
    print("              |")
    print("         [Core Router]")
    print("        /      |       \\")
    print("    Branch A  Branch B  Branch C")
    print("    10.1.0/24 10.2.0/24 10.3.0/24")

    print("\n  Core Router Configuration:")
    print("    # Directly connected: HQ, links to each branch")
    print("    # Default route to ISP")
    print("    ip route 0.0.0.0 0.0.0.0 [ISP-Gateway-IP]")
    print("    # Branch routes added as directly connected or static")

    for branch, network in [("A", "10.1.0.0"), ("B", "10.2.0.0"), ("C", "10.3.0.0")]:
        print(f"\n  Branch {branch} Router Configuration:")
        print(f"    # Default route -> everything goes to Core Router")
        print(f"    ip route 0.0.0.0 0.0.0.0 [CoreRouter-IP]")
        print(f"    # This covers: HQ, other branches, and Internet")
        print(f"    # Local network {network}/24 is directly connected")

    print("\n  Design rationale:")
    print("    - Branch routers: single default route (simple, minimal config)")
    print("    - Core router: knows all branch networks (specific routes)")
    print("    - Scalable: adding new branch requires only one route on core")
    print("    - Consider OSPF if the network grows beyond 5-10 branches")


if __name__ == "__main__":
    exercises = [exercise_1, exercise_2, exercise_3, exercise_4]
    for i, ex in enumerate(exercises, 1):
        print(f"\n{'=' * 60}")
        print(f"=== Exercise {i} ===")
        print(f"{'=' * 60}")
        ex()

    print(f"\n{'=' * 60}")
    print("All exercises completed!")
