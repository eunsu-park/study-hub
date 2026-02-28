"""
Exercises for Lesson 22: Multicast
Topic: Networking

Solutions to practice problems covering multicast addressing,
IGMP state machines, PIM tree construction, multicast vs unicast
efficiency, and SSM (Source-Specific Multicast).
"""

import numpy as np
from collections import defaultdict, deque


# ============================================================
# Exercise 1: Multicast Address Analysis
# ============================================================
def exercise_1():
    """
    Classify multicast addresses by scope and purpose.
    Validate multicast address ranges.
    """
    print("=== Exercise 1: Multicast Address Analysis ===\n")

    def classify_multicast(addr):
        octets = list(map(int, addr.split(".")))
        first = octets[0]

        if not (224 <= first <= 239):
            return {"valid": False, "reason": "Not multicast (not 224-239)"}

        result = {"valid": True, "addr": addr}

        if first == 224 and octets[1] == 0 and octets[2] == 0:
            result["scope"] = "link-local"
            result["forwarded"] = False
            result["ttl"] = 1
        elif first == 232:
            result["scope"] = "SSM (Source-Specific)"
            result["forwarded"] = True
            result["ttl"] = 255
        elif first == 239:
            result["scope"] = "admin-scoped (private)"
            result["forwarded"] = True
            result["ttl"] = "site-dependent"
        else:
            result["scope"] = "global"
            result["forwarded"] = True
            result["ttl"] = 255

        # Well-known addresses
        well_known = {
            "224.0.0.1": "All Hosts",
            "224.0.0.2": "All Routers",
            "224.0.0.5": "OSPF All Routers",
            "224.0.0.6": "OSPF DR Routers",
            "224.0.0.9": "RIPv2 Routers",
            "224.0.0.13": "PIM Routers",
            "224.0.0.22": "IGMPv3",
            "224.0.0.251": "mDNS",
            "224.0.1.1": "NTP",
            "239.255.255.250": "SSDP/UPnP",
        }
        result["name"] = well_known.get(addr, "-")
        return result

    test_addrs = [
        "224.0.0.1", "224.0.0.5", "224.0.0.22", "224.0.0.251",
        "224.0.1.1", "232.1.2.3", "239.192.0.1", "239.255.255.250",
        "225.1.1.1", "10.0.0.1", "240.0.0.1",
    ]

    print(f"  {'Address':>20} | {'Valid':>5} | {'Scope':>18} | "
          f"{'Fwd':>3} | {'Name'}")
    print(f"  {'-'*70}")

    for addr in test_addrs:
        info = classify_multicast(addr)
        if info["valid"]:
            print(f"  {addr:>20} | {'Yes':>5} | {info['scope']:>18} | "
                  f"{'Y' if info['forwarded'] else 'N':>3} | {info['name']}")
        else:
            print(f"  {addr:>20} | {'No':>5} | {info['reason']}")

    print(f"\n  Multicast-to-MAC mapping (IEEE):")
    print(f"    IP 224.0.0.1 → MAC 01:00:5e:00:00:01")
    print(f"    Lower 23 bits of IP map to lower 23 bits of MAC")
    print(f"    This means 32 IP addresses share 1 MAC (overlap possible)")
    print()


# ============================================================
# Exercise 2: IGMP Protocol Simulation
# ============================================================
def exercise_2():
    """
    Simulate IGMP v2 protocol including join, leave, query,
    and report suppression.
    """
    print("=== Exercise 2: IGMP v2 Simulation ===\n")

    class IGMPRouter:
        """Router-side IGMP state machine."""

        def __init__(self, name):
            self.name = name
            # group -> {"members": set of ports, "timer": int}
            self.group_table = defaultdict(
                lambda: {"members": set(), "timer": 0})
            self.query_interval = 125  # seconds

        def receive_report(self, group, port):
            self.group_table[group]["members"].add(port)
            self.group_table[group]["timer"] = self.query_interval * 2

        def receive_leave(self, group, port):
            if group in self.group_table:
                self.group_table[group]["members"].discard(port)
                if not self.group_table[group]["members"]:
                    del self.group_table[group]

        def send_query(self, group=None):
            """Send general or group-specific query."""
            if group:
                return f"Query for {group}"
            return "General Query for all groups"

    class IGMPHost:
        """Host-side IGMP v2."""

        def __init__(self, name, port):
            self.name = name
            self.port = port
            self.groups = set()

        def join(self, group, router):
            self.groups.add(group)
            router.receive_report(group, self.port)
            return f"{self.name} joined {group}"

        def leave(self, group, router):
            self.groups.discard(group)
            router.receive_leave(group, self.port)
            return f"{self.name} left {group}"

    router = IGMPRouter("R1")
    hosts = [
        IGMPHost("PC1", port=1),
        IGMPHost("PC2", port=2),
        IGMPHost("PC3", port=3),
    ]

    events = []

    # Scenario: hosts join and leave groups
    events.append(hosts[0].join("239.1.1.1", router))
    events.append(hosts[1].join("239.1.1.1", router))
    events.append(hosts[2].join("239.2.2.2", router))
    events.append(hosts[0].join("239.2.2.2", router))

    print(f"  Events:")
    for e in events:
        print(f"    {e}")

    print(f"\n  Router group table:")
    for group, info in sorted(router.group_table.items()):
        ports = sorted(info["members"])
        print(f"    {group}: ports {ports}")

    # Host leaves
    events2 = []
    events2.append(hosts[0].leave("239.1.1.1", router))
    events2.append(router.send_query("239.1.1.1"))

    print(f"\n  Leave events:")
    for e in events2:
        print(f"    {e}")

    print(f"\n  Router group table after leave:")
    for group, info in sorted(router.group_table.items()):
        ports = sorted(info["members"])
        print(f"    {group}: ports {ports}")

    print(f"\n  IGMPv2 report suppression: only one host per group responds")
    print(f"  to queries, reducing multicast control traffic on the LAN.")
    print()


# ============================================================
# Exercise 3: Multicast Tree Construction (PIM-SM)
# ============================================================
def exercise_3():
    """
    Build both shared (RPT) and shortest-path (SPT) multicast trees.
    Compare hop counts and latency.
    """
    print("=== Exercise 3: PIM-SM Tree Construction ===\n")

    class MulticastTopology:
        def __init__(self):
            self.adj = defaultdict(list)

        def add_link(self, a, b, cost=1):
            self.adj[a].append((b, cost))
            self.adj[b].append((a, cost))

        def shortest_path(self, src, dst):
            import heapq
            dist = {src: 0}
            prev = {}
            pq = [(0, src)]
            while pq:
                d, u = heapq.heappop(pq)
                if u == dst:
                    break
                if d > dist.get(u, float('inf')):
                    continue
                for v, w in self.adj[u]:
                    nd = d + w
                    if nd < dist.get(v, float('inf')):
                        dist[v] = nd
                        prev[v] = u
                        heapq.heappush(pq, (nd, v))
            path = []
            node = dst
            while node in prev:
                path.append(node)
                node = prev[node]
            path.append(src)
            return list(reversed(path)), dist.get(dst, float('inf'))

        def build_shared_tree(self, rp, receivers):
            tree_edges = set()
            total_cost = 0
            for rcv in receivers:
                path, cost = self.shortest_path(rcv, rp)
                total_cost += cost
                for i in range(len(path) - 1):
                    tree_edges.add((path[i + 1], path[i]))
            return tree_edges, total_cost

        def build_spt(self, source, receivers):
            tree_edges = set()
            total_cost = 0
            for rcv in receivers:
                path, cost = self.shortest_path(source, rcv)
                total_cost += cost
                for i in range(len(path) - 1):
                    tree_edges.add((path[i], path[i + 1]))
            return tree_edges, total_cost

    topo = MulticastTopology()
    # Build topology (ISP-like)
    #       R1(RP)
    #      /  |  \
    #    R2  R3   R4
    #   /  \   \    \
    #  R5   R6  R7   R8(Source)
    for a, b, c in [
        ("R1", "R2", 1), ("R1", "R3", 2), ("R1", "R4", 1),
        ("R2", "R5", 1), ("R2", "R6", 1), ("R3", "R7", 1),
        ("R4", "R8", 1), ("R5", "R6", 2),
    ]:
        topo.add_link(a, b, c)

    rp = "R1"
    source = "R8"
    receivers = ["R5", "R6", "R7"]

    print(f"  RP: {rp}, Source: {source}, Receivers: {receivers}\n")

    # Shared tree
    rpt_edges, rpt_cost = topo.build_shared_tree(rp, receivers)
    print(f"  Shared Tree (*,G) via RP={rp}:")
    print(f"    Edges: {sorted(rpt_edges)}")
    print(f"    Total cost: {rpt_cost}")

    # SPT
    spt_edges, spt_cost = topo.build_spt(source, receivers)
    print(f"\n  Shortest-Path Tree (S,G) from {source}:")
    print(f"    Edges: {sorted(spt_edges)}")
    print(f"    Total cost: {spt_cost}")

    # Data path comparison
    for rcv in receivers:
        rpt_path, rpt_c = topo.shortest_path(source, rp)
        rpt_path2, rpt_c2 = topo.shortest_path(rp, rcv)
        spt_path, spt_c = topo.shortest_path(source, rcv)
        print(f"\n    {source}→{rcv}: RPT cost={rpt_c + rpt_c2} "
              f"(via RP), SPT cost={spt_c}")

    print(f"\n  SPT has lower latency but more state in routers.")
    print(f"  PIM-SM switchover: start with RPT, switch to SPT when")
    print(f"  data rate exceeds threshold (SPT-Threshold).")
    print()


# ============================================================
# Exercise 4: Multicast Efficiency Analysis
# ============================================================
def exercise_4():
    """
    Compare bandwidth usage: unicast vs multicast vs application-layer
    multicast for various receiver counts.
    """
    print("=== Exercise 4: Multicast Efficiency Analysis ===\n")

    stream_rate = 10  # Mbps
    tree_links = 20   # links in the network
    avg_path_len = 5  # average unicast path length in hops

    # Application-layer multicast: overlay tree
    # Each receiver re-sends to 2 children (binary overlay tree)
    def app_layer_bw(n_receivers, stream_rate, avg_hops):
        """Each overlay link uses avg_hops physical links."""
        import math
        if n_receivers <= 1:
            return stream_rate * avg_hops
        # Binary tree: n-1 overlay links, each traverses avg_hops physical
        overlay_links = n_receivers - 1
        return overlay_links * stream_rate * avg_hops

    receivers_list = [1, 5, 10, 50, 100, 500, 1000]

    print(f"  Stream: {stream_rate} Mbps, Network: {tree_links} links, "
          f"Avg path: {avg_path_len} hops\n")
    print(f"  {'Receivers':>10} | {'Unicast':>10} | {'IP Multicast':>13} | "
          f"{'App-Layer MC':>13} | {'MC Savings':>11}")
    print(f"  {'-'*62}")

    for n in receivers_list:
        unicast = n * stream_rate * avg_path_len
        multicast = tree_links * stream_rate  # one copy per tree link
        app_mc = app_layer_bw(n, stream_rate, avg_path_len)

        savings = (1 - multicast / unicast) * 100 if unicast > 0 else 0
        print(f"  {n:>10} | {unicast:>8} Mb | {multicast:>11} Mb | "
              f"{app_mc:>11} Mb | {savings:>10.1f}%")

    print(f"\n  IP multicast: bandwidth independent of receiver count")
    print(f"  Application-layer MC: O(N) but better than unicast")
    print(f"  Unicast: O(N × path_length) — worst scaling")
    print()


# ============================================================
# Exercise 5: Source-Specific Multicast (SSM)
# ============================================================
def exercise_5():
    """
    Implement SSM (Source-Specific Multicast) using IGMPv3-style
    channel model: (S, G) instead of just (*,G).
    """
    print("=== Exercise 5: Source-Specific Multicast (SSM) ===\n")

    class SSMRouter:
        """Router supporting SSM (S,G) channels."""

        def __init__(self, name):
            self.name = name
            # (source, group) -> set of downstream interfaces
            self.channels = defaultdict(set)

        def subscribe(self, source, group, interface):
            """IGMPv3 INCLUDE mode: join specific (S,G)."""
            self.channels[(source, group)].add(interface)

        def unsubscribe(self, source, group, interface):
            key = (source, group)
            self.channels[key].discard(interface)
            if not self.channels[key]:
                del self.channels[key]

        def forward(self, source, group, data):
            key = (source, group)
            if key in self.channels:
                interfaces = self.channels[key]
                return list(interfaces)
            return []

    router = SSMRouter("Edge-R1")

    # Viewers subscribe to specific channels
    subscriptions = [
        ("10.1.1.1", "232.1.1.1", "eth1", "ESPN stream"),
        ("10.1.1.1", "232.1.1.1", "eth2", "ESPN stream"),
        ("10.2.2.2", "232.1.1.1", "eth1", "BBC stream (same group, diff source)"),
        ("10.1.1.1", "232.2.2.2", "eth3", "CNN stream"),
    ]

    print(f"  SSM uses 232.0.0.0/8 range")
    print(f"  Channel = (Source, Group) pair — no RP needed\n")

    for src, grp, iface, desc in subscriptions:
        router.subscribe(src, grp, iface)
        print(f"  Subscribe: ({src}, {grp}) on {iface}  [{desc}]")

    print(f"\n  Router channel table:")
    for (src, grp), ifaces in sorted(router.channels.items()):
        print(f"    ({src}, {grp}) → {sorted(ifaces)}")

    # Forward test
    print(f"\n  Forwarding decisions:")
    tests = [
        ("10.1.1.1", "232.1.1.1", "ESPN data"),
        ("10.2.2.2", "232.1.1.1", "BBC data"),
        ("10.3.3.3", "232.1.1.1", "Unknown source"),
    ]

    for src, grp, desc in tests:
        out = router.forward(src, grp, "payload")
        print(f"    From {src} to {grp}: forward to {out or 'DROP'} [{desc}]")

    print(f"\n  SSM advantages over ASM (Any-Source Multicast):")
    print(f"    1. No RP needed (eliminates single point of failure)")
    print(f"    2. Source validation built-in (prevents spoofing)")
    print(f"    3. Simpler state: (S,G) only, no (*,G) RPT")
    print(f"    4. Better for one-to-many streaming (IPTV, live events)")
    print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
