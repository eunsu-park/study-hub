"""
Routing Table Simulator

Demonstrates:
- Routing table with CIDR entries
- Longest prefix match algorithm
- Default route fallback
- Route aggregation

Theory:
- Routers forward packets based on destination IP.
- The routing table maps network prefixes to next hops.
- Longest prefix match: the most specific matching route wins.
  E.g., 10.1.1.5 matches 10.1.1.0/24 over 10.1.0.0/16.
- Default route (0.0.0.0/0) is the fallback when no match exists.

Adapted from Networking Lesson 08.
"""

import struct
import socket
from dataclasses import dataclass


def ip_to_int(ip: str) -> int:
    return struct.unpack("!I", socket.inet_aton(ip))[0]


def int_to_ip(n: int) -> str:
    return socket.inet_ntoa(struct.pack("!I", n))


def prefix_to_mask(prefix: int) -> int:
    return (0xFFFFFFFF << (32 - prefix)) & 0xFFFFFFFF


@dataclass
class Route:
    network: str
    prefix: int
    next_hop: str
    interface: str
    metric: int = 0

    @property
    def network_int(self) -> int:
        return ip_to_int(self.network)

    @property
    def mask(self) -> int:
        return prefix_to_mask(self.prefix)

    def matches(self, ip: str) -> bool:
        return (ip_to_int(ip) & self.mask) == (self.network_int & self.mask)

    def __str__(self) -> str:
        return (f"{self.network}/{self.prefix:<3} via {self.next_hop:<16} "
                f"dev {self.interface:<6} metric {self.metric}")


# Why: Using a sorted list instead of a trie keeps the code simple for
# educational purposes. Real routers use Patricia/radix tries for O(W)
# lookup (W = address width), but a sorted list with linear scan is
# sufficient to demonstrate the longest-prefix-match principle clearly.
class RoutingTable:
    """IP routing table with longest prefix match."""

    def __init__(self):
        self.routes: list[Route] = []

    def add_route(self, network: str, prefix: int, next_hop: str,
                  interface: str, metric: int = 0) -> None:
        route = Route(network, prefix, next_hop, interface, metric)
        self.routes.append(route)
        # Why: Pre-sorting by descending prefix length means the first match
        # found during linear scan is automatically the longest (most specific)
        # match. Tie-breaking by metric selects the lowest-cost path among
        # equally specific routes.
        self.routes.sort(key=lambda r: (-r.prefix, r.metric))

    def remove_route(self, network: str, prefix: int) -> bool:
        for i, r in enumerate(self.routes):
            if r.network == network and r.prefix == prefix:
                self.routes.pop(i)
                return True
        return False

    def lookup(self, ip: str) -> Route | None:
        """Longest prefix match lookup."""
        # Why: Because routes are pre-sorted by descending prefix length,
        # we can return the first match — it is guaranteed to be the
        # longest (most specific) prefix. This mirrors TCAM behavior in
        # hardware routers where entries are priority-ordered.
        for route in self.routes:
            if route.matches(ip):
                return route
        return None

    def lookup_all(self, ip: str) -> list[Route]:
        """Return all matching routes (for debugging)."""
        return [r for r in self.routes if r.matches(ip)]

    def print_table(self) -> None:
        print(f"    {'Destination':<20} {'Next Hop':<16} {'Iface':<8} {'Metric':>7}")
        print(f"    {'-'*20} {'-'*16} {'-'*8} {'-'*7}")
        for r in self.routes:
            print(f"    {r.network + '/' + str(r.prefix):<20} "
                  f"{r.next_hop:<16} {r.interface:<8} {r.metric:>7}")


# ── Demos ──────────────────────────────────────────────────────────────

def demo_basic_routing():
    print("=" * 60)
    print("ROUTING TABLE LOOKUP")
    print("=" * 60)

    rt = RoutingTable()
    rt.add_route("192.168.1.0", 24, "10.0.0.1", "eth0")
    rt.add_route("192.168.0.0", 16, "10.0.0.2", "eth1")
    rt.add_route("10.0.0.0", 8, "10.0.0.3", "eth2")
    # Why: The default route (0.0.0.0/0) has prefix length 0, so it matches
    # every IP address but is always the least specific match. It acts as
    # the "gateway of last resort" — if no more specific route exists,
    # send the packet here (typically an upstream ISP router).
    rt.add_route("0.0.0.0", 0, "10.0.0.254", "eth0")  # Default

    print(f"\n  Routing table:")
    rt.print_table()

    # Test lookups
    test_ips = [
        "192.168.1.100",  # Matches /24 and /16
        "192.168.2.50",   # Matches /16 only
        "10.5.5.5",       # Matches /8
        "8.8.8.8",        # Default route only
    ]

    print(f"\n  Lookups:")
    print(f"    {'Destination':<20} {'Match':<25} {'Next Hop'}")
    print(f"    {'-'*20} {'-'*25} {'-'*16}")
    for ip in test_ips:
        route = rt.lookup(ip)
        if route:
            print(f"    {ip:<20} {route.network}/{route.prefix:<22} "
                  f"{route.next_hop}")
        else:
            print(f"    {ip:<20} {'No match':<25}")


def demo_longest_prefix():
    print("\n" + "=" * 60)
    print("LONGEST PREFIX MATCH")
    print("=" * 60)

    rt = RoutingTable()
    rt.add_route("10.0.0.0", 8, "gw-1", "eth0")
    rt.add_route("10.1.0.0", 16, "gw-2", "eth1")
    rt.add_route("10.1.1.0", 24, "gw-3", "eth2")
    rt.add_route("10.1.1.128", 25, "gw-4", "eth3")

    print(f"\n  Routing table (multiple specificity levels):")
    rt.print_table()

    test_ip = "10.1.1.200"
    print(f"\n  Looking up: {test_ip}")
    all_matches = rt.lookup_all(test_ip)
    print(f"\n  All matching routes (most specific first):")
    for r in all_matches:
        print(f"    /{r.prefix:<3} {r.network}/{r.prefix} → {r.next_hop}")

    best = rt.lookup(test_ip)
    print(f"\n  Best match (longest prefix): "
          f"{best.network}/{best.prefix} → {best.next_hop}")


def demo_ecmp():
    print("\n" + "=" * 60)
    print("EQUAL-COST MULTI-PATH (ECMP)")
    print("=" * 60)

    rt = RoutingTable()
    rt.add_route("10.0.0.0", 8, "gw-1", "eth0", metric=10)
    rt.add_route("10.0.0.0", 8, "gw-2", "eth1", metric=10)
    rt.add_route("10.0.0.0", 8, "gw-3", "eth2", metric=20)

    print(f"\n  Routes to 10.0.0.0/8 (multiple paths):")
    rt.print_table()

    test_ip = "10.5.5.5"
    all_routes = rt.lookup_all(test_ip)
    equal_cost = [r for r in all_routes
                  if r.metric == all_routes[0].metric]

    print(f"\n  Lookup {test_ip}:")
    print(f"    Best metric: {all_routes[0].metric}")
    print(f"    Equal-cost paths: {len(equal_cost)}")
    for r in equal_cost:
        print(f"      → {r.next_hop} via {r.interface}")

    # Why: Hash-based ECMP distributes flows across equal-cost paths
    # deterministically — the same destination always uses the same path,
    # preserving packet ordering within a flow. Real routers hash on a
    # 5-tuple (src/dst IP, src/dst port, protocol) for finer distribution.
    hash_val = hash(test_ip) % len(equal_cost)
    chosen = equal_cost[hash_val]
    print(f"\n  Hash-based selection: {chosen.next_hop}")


def demo_route_changes():
    print("\n" + "=" * 60)
    print("ROUTE ADDITION AND REMOVAL")
    print("=" * 60)

    rt = RoutingTable()
    rt.add_route("192.168.0.0", 16, "gw-1", "eth0")
    rt.add_route("0.0.0.0", 0, "gw-default", "eth0")

    test_ip = "192.168.1.100"

    print(f"\n  Initial lookup for {test_ip}:")
    route = rt.lookup(test_ip)
    print(f"    → {route.next_hop} via {route.network}/{route.prefix}")

    # Add more specific route
    print(f"\n  Adding 192.168.1.0/24 → gw-2:")
    rt.add_route("192.168.1.0", 24, "gw-2", "eth1")
    route = rt.lookup(test_ip)
    print(f"    Now → {route.next_hop} via {route.network}/{route.prefix}")

    # Remove it
    print(f"\n  Removing 192.168.1.0/24:")
    rt.remove_route("192.168.1.0", 24)
    route = rt.lookup(test_ip)
    print(f"    Falls back → {route.next_hop} via {route.network}/{route.prefix}")


if __name__ == "__main__":
    demo_basic_routing()
    demo_longest_prefix()
    demo_ecmp()
    demo_route_changes()
