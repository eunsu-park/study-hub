"""
Example: IP Multicast Simulator
Topic: Networking – Lesson 22

Demonstrates core multicast concepts:
  1. IGMP (Internet Group Management Protocol) state machine
  2. Multicast distribution tree construction
  3. PIM-SM (Sparse Mode) shared tree and shortest-path tree
  4. Multicast address allocation and scoping

Run: python 22_multicast_sim.py
"""

import random
from collections import defaultdict, deque


# ============================================================
# Core: Multicast Group and Address Management
# ============================================================
class MulticastAddress:
    """Multicast address utilities (IPv4 Class D: 224.0.0.0 – 239.255.255.255)."""

    # Well-known multicast addresses
    WELL_KNOWN = {
        "224.0.0.1": "All Hosts",
        "224.0.0.2": "All Routers",
        "224.0.0.5": "OSPF Routers",
        "224.0.0.6": "OSPF DRs",
        "224.0.0.13": "PIM Routers",
        "224.0.0.22": "IGMPv3",
    }

    # Scoping ranges
    SCOPES = {
        "link_local": ("224.0.0.0", "224.0.0.255"),
        "admin_local": ("239.0.0.0", "239.255.255.255"),
        "global":      ("224.0.1.0", "238.255.255.255"),
    }

    @staticmethod
    def is_multicast(addr: str) -> bool:
        first_octet = int(addr.split(".")[0])
        return 224 <= first_octet <= 239

    @staticmethod
    def get_scope(addr: str) -> str:
        octets = list(map(int, addr.split(".")))
        if octets[0] == 224 and octets[1] == 0 and octets[2] == 0:
            return "link_local"
        if octets[0] == 239:
            return "admin_local"
        return "global"


# ============================================================
# IGMP State Machine (per interface per group)
# ============================================================
class IGMPState:
    """Simplified IGMP v2 host state machine."""

    # States
    NON_MEMBER = "NON_MEMBER"
    DELAYING_MEMBER = "DELAYING_MEMBER"
    IDLE_MEMBER = "IDLE_MEMBER"

    def __init__(self, group: str, interface: str):
        self.group = group
        self.interface = interface
        self.state = self.NON_MEMBER
        self.timer = 0  # report delay timer
        self.log = []

    def join(self):
        """Host wants to join this group."""
        if self.state == self.NON_MEMBER:
            self.state = self.DELAYING_MEMBER
            self.timer = random.randint(1, 10)  # random delay
            self._log(f"JOIN → send unsolicited report, timer={self.timer}")

    def leave(self):
        """Host wants to leave this group."""
        prev = self.state
        self.state = self.NON_MEMBER
        self.timer = 0
        self._log(f"LEAVE from {prev} → send leave message")

    def query_received(self):
        """Router sent a membership query."""
        if self.state == self.IDLE_MEMBER:
            self.state = self.DELAYING_MEMBER
            self.timer = random.randint(1, 10)
            self._log(f"QUERY → start timer={self.timer}")
        elif self.state == self.DELAYING_MEMBER:
            new_timer = random.randint(1, 10)
            if new_timer < self.timer:
                self.timer = new_timer
                self._log(f"QUERY → reduce timer to {self.timer}")

    def timer_expired(self):
        """Report delay timer expires."""
        if self.state == self.DELAYING_MEMBER:
            self.state = self.IDLE_MEMBER
            self._log("TIMER EXPIRED → send report, become IDLE_MEMBER")

    def report_heard(self):
        """Another host on same link sent a report for this group."""
        if self.state == self.DELAYING_MEMBER:
            self.state = self.IDLE_MEMBER
            self.timer = 0
            self._log("REPORT HEARD → suppress own report, become IDLE_MEMBER")

    def _log(self, msg: str):
        self.log.append(f"[{self.interface}:{self.group}] {msg}")


# ============================================================
# Demo 1: IGMP Membership Protocol
# ============================================================
def demo_igmp():
    """Demonstrate IGMP v2 join/query/leave lifecycle."""
    print("=" * 60)
    print("Demo 1: IGMP Membership Protocol (v2)")
    print("=" * 60)

    random.seed(42)

    # Two hosts on the same LAN segment
    host_a = IGMPState(group="239.1.1.1", interface="HostA-eth0")
    host_b = IGMPState(group="239.1.1.1", interface="HostB-eth0")

    print(f"\n  Scenario: Two hosts joining group 239.1.1.1\n")

    # Step 1: Host A joins
    host_a.join()

    # Step 2: Host B joins shortly after
    host_b.join()

    # Step 3: Host B hears Host A's report (suppression)
    host_b.report_heard()

    # Step 4: Router sends periodic query
    host_a.query_received()
    host_b.query_received()

    # Step 5: Host A's timer expires (sends report)
    host_a.timer_expired()

    # Step 6: Host B hears it (suppresses again)
    host_b.report_heard()

    # Step 7: Host A leaves
    host_a.leave()

    # Print event log
    all_events = host_a.log + host_b.log
    for event in host_a.log:
        print(f"    {event}")
    print()
    for event in host_b.log:
        print(f"    {event}")

    print(f"\n  IGMP report suppression reduces redundant traffic on the LAN.")
    print(f"  Only one host per group needs to respond to a query.")
    print()


# ============================================================
# Multicast Tree Builder
# ============================================================
class Router:
    """A router participating in PIM multicast routing."""

    def __init__(self, name: str):
        self.name = name
        self.neighbors: dict[str, str] = {}  # neighbor_name -> interface
        self.group_state: dict[str, set] = defaultdict(set)  # group -> set of interfaces
        self.is_rp = False  # Rendezvous Point flag

    def add_neighbor(self, neighbor_name: str, interface: str):
        self.neighbors[neighbor_name] = interface

    def __repr__(self):
        return f"Router({self.name})"


class MulticastNetwork:
    """Network topology for multicast tree construction."""

    def __init__(self):
        self.routers: dict[str, Router] = {}
        self.links: list[tuple] = []
        self.rp: str | None = None  # Rendezvous Point

    def add_router(self, name: str) -> Router:
        r = Router(name)
        self.routers[name] = r
        return r

    def add_link(self, r1: str, iface1: str, r2: str, iface2: str):
        self.routers[r1].add_neighbor(r2, iface1)
        self.routers[r2].add_neighbor(r1, iface2)
        self.links.append((r1, r2))

    def set_rp(self, name: str):
        """Set the Rendezvous Point for PIM-SM."""
        self.rp = name
        self.routers[name].is_rp = True

    def shortest_path(self, src: str, dst: str) -> list:
        """BFS shortest path between two routers."""
        if src == dst:
            return [src]
        visited = {src}
        queue = deque([(src, [src])])
        while queue:
            node, path = queue.popleft()
            for neighbor in self.routers[node].neighbors:
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    if neighbor == dst:
                        return new_path
                    visited.add(neighbor)
                    queue.append((neighbor, new_path))
        return []

    def build_shared_tree(self, group: str, receivers: list) -> dict:
        """Build PIM-SM shared tree (RPT) rooted at the RP.

        All receivers join via the RP, creating a shared tree (*,G).
        """
        tree = defaultdict(set)  # parent -> set of children

        for receiver in receivers:
            path = self.shortest_path(receiver, self.rp)
            for i in range(len(path) - 1):
                child, parent = path[i], path[i + 1]
                tree[parent].add(child)

        return dict(tree)

    def build_spt(self, group: str, source: str, receivers: list) -> dict:
        """Build shortest-path tree (SPT) rooted at the source.

        After data flows, receivers may switch to SPT for lower latency.
        This creates (S,G) state in routers.
        """
        tree = defaultdict(set)

        for receiver in receivers:
            path = self.shortest_path(source, receiver)
            for i in range(len(path) - 1):
                parent, child = path[i], path[i + 1]
                tree[parent].add(child)

        return dict(tree)


def print_tree(tree: dict, root: str, indent: int = 4):
    """Pretty-print a multicast distribution tree."""
    def _print(node, depth):
        prefix = " " * indent + "  " * depth
        children = tree.get(node, set())
        suffix = " (leaf)" if not children else ""
        print(f"{prefix}{node}{suffix}")
        for child in sorted(children):
            _print(child, depth + 1)
    _print(root, 0)


# ============================================================
# Demo 2: PIM-SM Shared Tree vs Shortest-Path Tree
# ============================================================
def demo_pim_trees():
    """Compare PIM-SM shared tree (RPT) and shortest-path tree (SPT)."""
    print("=" * 60)
    print("Demo 2: PIM-SM Shared Tree vs Shortest-Path Tree")
    print("=" * 60)

    # Build topology:
    #        R1 (RP)
    #       / \
    #      R2   R3
    #     / \     \
    #    R4  R5    R6
    #    |         |
    #   [Src]    [Rcv1, Rcv2 on R4, Rcv3 on R6]

    net = MulticastNetwork()
    for name in ["R1", "R2", "R3", "R4", "R5", "R6"]:
        net.add_router(name)

    net.add_link("R1", "eth0", "R2", "eth0")
    net.add_link("R1", "eth1", "R3", "eth0")
    net.add_link("R2", "eth1", "R4", "eth0")
    net.add_link("R2", "eth2", "R5", "eth0")
    net.add_link("R3", "eth1", "R6", "eth0")

    net.set_rp("R1")
    group = "239.1.1.1"
    source = "R4"
    receivers = ["R4", "R5", "R6"]

    print(f"\n  Topology:")
    print(f"            R1 (RP)")
    print(f"           / \\")
    print(f"          R2   R3")
    print(f"         / \\     \\")
    print(f"        R4  R5    R6")
    print(f"       [Src]    [Rcv]")
    print(f"\n  Source: {source}, Receivers: {receivers}, Group: {group}")

    # Shared tree (*,G) — all traffic goes through RP
    print(f"\n  Shared Tree (*,G) via RP={net.rp}:")
    shared = net.build_shared_tree(group, receivers)
    print_tree(shared, net.rp)

    # Count hops
    shared_hops = sum(len(net.shortest_path(source, r)) - 1
                      for r in receivers)

    # Shortest-path tree (S,G) — rooted at source
    print(f"\n  Shortest-Path Tree (S,G) from {source}:")
    spt = net.build_spt(group, source, receivers)
    print_tree(spt, source)

    spt_hops = sum(len(net.shortest_path(source, r)) - 1
                   for r in receivers)

    print(f"\n  Total hops — RPT: source→RP→receivers, SPT: source→receivers")
    print(f"  SPT typically has lower latency for high-bandwidth sources.")
    print(f"  PIM-SM switchover: receivers initially join RPT, then switch to")
    print(f"  SPT once data rate exceeds a threshold.")
    print()


# ============================================================
# Demo 3: Multicast Addressing and Scoping
# ============================================================
def demo_addressing():
    """Demonstrate multicast address classification and scoping."""
    print("=" * 60)
    print("Demo 3: Multicast Addressing and Scoping")
    print("=" * 60)

    test_addrs = [
        "224.0.0.1",
        "224.0.0.5",
        "224.0.0.22",
        "224.0.1.1",
        "232.1.1.1",
        "239.192.0.1",
        "239.255.255.250",
        "10.0.0.1",       # unicast — should fail
    ]

    print(f"\n  {'Address':>20} | {'Multicast':>9} | {'Scope':>12} | {'Note'}")
    print(f"  {'-'*65}")

    for addr in test_addrs:
        is_mc = MulticastAddress.is_multicast(addr)
        scope = MulticastAddress.get_scope(addr) if is_mc else "N/A"
        note = MulticastAddress.WELL_KNOWN.get(addr, "")

        # Special ranges
        first_octet = int(addr.split(".")[0])
        if first_octet == 232:
            note = "SSM range (Source-Specific Multicast)"
        elif first_octet == 239 and not note:
            note = "Administratively scoped"

        print(f"  {addr:>20} | {str(is_mc):>9} | {scope:>12} | {note}")

    print(f"\n  Key ranges:")
    print(f"    224.0.0.0/24   — Link-local (never forwarded by routers)")
    print(f"    232.0.0.0/8    — Source-Specific Multicast (SSM)")
    print(f"    239.0.0.0/8    — Administratively scoped (private)")
    print(f"    224.0.1.0-238.x — Globally scoped (internet-wide)")
    print()


# ============================================================
# Demo 4: Multicast Application — Live Streaming
# ============================================================
def demo_streaming():
    """Simulate bandwidth savings of multicast vs unicast streaming."""
    print("=" * 60)
    print("Demo 4: Multicast vs Unicast Bandwidth Comparison")
    print("=" * 60)

    # Scenario: live video streaming to N viewers
    bitrate = 5_000_000   # 5 Mbps per stream
    viewers_list = [10, 100, 1000, 10000]

    # Simple tree topology: source → core → N edge routers → viewers
    core_links = 4    # links between core routers
    edge_routers = 8  # edge routers

    print(f"\n  Stream bitrate: {bitrate/1e6:.0f} Mbps")
    print(f"  Topology: source → {core_links} core links → {edge_routers} "
          f"edge routers → viewers")
    print(f"\n  {'Viewers':>10} | {'Unicast BW':>12} | {'Multicast BW':>14} | "
          f"{'Savings':>8}")
    print(f"  {'-'*55}")

    for n_viewers in viewers_list:
        # Unicast: one copy per viewer traverses the entire path
        unicast_bw = n_viewers * bitrate

        # Multicast: one copy per link in the distribution tree
        # Core: 1 copy × core_links, Edge: 1 copy per edge router
        multicast_bw = (core_links + edge_routers) * bitrate

        savings = 1 - multicast_bw / unicast_bw

        print(f"  {n_viewers:>10,} | {unicast_bw/1e9:>10.1f} Gb | "
              f"{multicast_bw/1e6:>12.0f} Mb | {savings:>7.1%}")

    print(f"\n  Multicast bandwidth is constant regardless of viewer count.")
    print(f"  This is why IPTV, financial data feeds, and live events")
    print(f"  use multicast in enterprise/ISP networks.")
    print()


if __name__ == "__main__":
    demo_igmp()
    demo_pim_trees()
    demo_addressing()
    demo_streaming()
