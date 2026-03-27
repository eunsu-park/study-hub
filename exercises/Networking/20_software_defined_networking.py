"""
Exercises for Lesson 20: Software-Defined Networking
Topic: Networking

Solutions to practice problems covering OpenFlow flow tables,
SDN controller logic, shortest-path routing, network slicing,
and SDN security considerations.
"""

import numpy as np
from collections import defaultdict, deque


# ============================================================
# Exercise 1: OpenFlow Flow Table Matching
# ============================================================
def exercise_1():
    """
    Implement an OpenFlow flow table with priority-based matching.
    Test with overlapping rules and verify correct match order.
    """
    print("=== Exercise 1: OpenFlow Flow Table ===\n")

    class FlowEntry:
        def __init__(self, match, actions, priority=0):
            self.match = match
            self.actions = actions
            self.priority = priority
            self.packet_count = 0

        def matches(self, packet):
            for field, value in self.match.items():
                if field not in packet or packet[field] != value:
                    return False
            return True

    class FlowTable:
        def __init__(self):
            self.entries = []

        def add(self, entry):
            self.entries.append(entry)
            self.entries.sort(key=lambda e: e.priority, reverse=True)

        def lookup(self, packet):
            for entry in self.entries:
                if entry.matches(packet):
                    entry.packet_count += 1
                    return entry.actions
            return [("drop",)]

    # Build flow table with overlapping rules
    table = FlowTable()
    table.add(FlowEntry({"dst": "10.0.0.1"}, [("output", 1)], priority=100))
    table.add(FlowEntry({"dst": "10.0.0.1", "proto": "tcp", "port": 80},
                         [("output", 2)], priority=200))
    table.add(FlowEntry({}, [("output", 3)], priority=0))  # default

    # Test packets
    tests = [
        ({"src": "A", "dst": "10.0.0.1", "proto": "tcp", "port": 80},
         "HTTP to 10.0.0.1"),
        ({"src": "A", "dst": "10.0.0.1", "proto": "udp"},
         "UDP to 10.0.0.1"),
        ({"src": "A", "dst": "10.0.0.2"},
         "Any to 10.0.0.2"),
    ]

    for pkt, desc in tests:
        actions = table.lookup(pkt)
        print(f"  {desc:30} → {actions}")

    print(f"\n  HTTP traffic matched higher-priority rule (port 2)")
    print(f"  UDP matched generic dst rule (port 1)")
    print(f"  Unknown dst matched default rule (port 3)")
    print()


# ============================================================
# Exercise 2: SDN Controller — Shortest-Path Routing
# ============================================================
def exercise_2():
    """
    Build an SDN controller that computes shortest paths
    and installs flow entries along the path.
    """
    print("=== Exercise 2: Shortest-Path SDN Routing ===\n")

    class Network:
        def __init__(self):
            self.adj = defaultdict(dict)  # node -> {neighbor: cost}
            self.flows = defaultdict(list)  # switch -> [(match, action)]

        def add_link(self, a, b, cost=1):
            self.adj[a][b] = cost
            self.adj[b][a] = cost

        def dijkstra(self, src, dst):
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
                for v, w in self.adj[u].items():
                    nd = d + w
                    if nd < dist.get(v, float('inf')):
                        dist[v] = nd
                        prev[v] = u
                        heapq.heappush(pq, (nd, v))
            # Reconstruct
            path = []
            node = dst
            while node != src:
                if node not in prev:
                    return []
                path.append(node)
                node = prev[node]
            path.append(src)
            return list(reversed(path))

        def install_path(self, src_host, dst_host, path):
            """Install flow entries on each switch along the path."""
            for i in range(len(path) - 1):
                sw = path[i]
                next_hop = path[i + 1]
                self.flows[sw].append(({"dst": dst_host}, next_hop))

    net = Network()
    # Diamond topology: S1 --1-- S2 --1-- S4
    #                    \--2-- S3 --1--/
    net.add_link("S1", "S2", 1)
    net.add_link("S1", "S3", 2)
    net.add_link("S2", "S4", 1)
    net.add_link("S3", "S4", 1)

    path = net.dijkstra("S1", "S4")
    print(f"  Topology: S1--(1)--S2--(1)--S4")
    print(f"             \\--(2)--S3--(1)--/")
    print(f"  Shortest path S1→S4: {' → '.join(path)}")
    print(f"  Path cost: {sum(net.adj[path[i]][path[i+1]] for i in range(len(path)-1))}")

    net.install_path("HostA", "HostD", path)
    print(f"\n  Installed flows:")
    for sw, flows in net.flows.items():
        for match, action in flows:
            print(f"    {sw}: match={match} → forward to {action}")

    print(f"\n  Controller computed shortest path and pushed flows to switches.")
    print()


# ============================================================
# Exercise 3: Network Slicing with Isolation
# ============================================================
def exercise_3():
    """
    Implement network slicing: multiple virtual networks over
    shared physical infrastructure using flow table isolation.
    """
    print("=== Exercise 3: Network Slicing ===\n")

    class NetworkSlice:
        def __init__(self, name, vlan_id, bandwidth_mbps):
            self.name = name
            self.vlan_id = vlan_id
            self.bandwidth = bandwidth_mbps
            self.flows = []

        def add_flow(self, match, actions):
            self.flows.append({"match": match, "actions": actions,
                              "slice": self.name})

    # Define slices on shared infrastructure
    slices = [
        NetworkSlice("Enterprise", vlan_id=100, bandwidth_mbps=500),
        NetworkSlice("IoT", vlan_id=200, bandwidth_mbps=50),
        NetworkSlice("Video", vlan_id=300, bandwidth_mbps=1000),
    ]

    # Each slice gets isolated flow entries
    slices[0].add_flow(
        {"vlan": 100, "dst": "10.1.0.0/16"},
        [("set_queue", "gold"), ("output", 1)]
    )
    slices[1].add_flow(
        {"vlan": 200, "dst": "10.2.0.0/16"},
        [("set_queue", "bronze"), ("output", 2)]
    )
    slices[2].add_flow(
        {"vlan": 300, "dst": "10.3.0.0/16"},
        [("set_queue", "silver"), ("output", 3)]
    )

    total_bw = sum(s.bandwidth for s in slices)

    print(f"  {'Slice':>12} | {'VLAN':>6} | {'BW (Mbps)':>10} | {'Share':>6}")
    print(f"  {'-'*42}")
    for s in slices:
        share = s.bandwidth / total_bw
        print(f"  {s.name:>12} | {s.vlan_id:>6} | {s.bandwidth:>10} | "
              f"{share:>5.0%}")

    print(f"\n  Total allocated: {total_bw} Mbps")
    print(f"  Each slice sees its own virtual network with guaranteed BW.")
    print(f"  SDN controller enforces isolation via VLAN tags and queues.")
    print()


# ============================================================
# Exercise 4: Flow Statistics and Monitoring
# ============================================================
def exercise_4():
    """
    Implement flow statistics collection and traffic anomaly detection
    using SDN controller's global view.
    """
    print("=== Exercise 4: SDN Traffic Monitoring ===\n")

    rng = np.random.RandomState(42)

    # Simulate flow statistics from 5 switches over 10 time steps
    n_switches = 5
    n_steps = 10

    # Normal traffic: ~1000 pkts/step with some variation
    traffic = rng.poisson(1000, (n_steps, n_switches))

    # Inject anomaly: switch 3 gets a traffic spike at step 7
    traffic[7, 3] = 8000
    traffic[8, 3] = 6000

    # Detect anomalies using Z-score on per-switch statistics
    print(f"  Traffic matrix (packets per time step per switch):")
    print(f"  {'Step':>6}", end="")
    for s in range(n_switches):
        print(f" | {'S'+str(s+1):>6}", end="")
    print(f" | {'Anomaly?'}")
    print(f"  {'-'*(6 + 9*n_switches + 12)}")

    # Compute running statistics
    window = 5
    for t in range(n_steps):
        print(f"  {t:>6}", end="")
        anomalies = []
        for s in range(n_switches):
            print(f" | {traffic[t, s]:>6}", end="")

            # Use recent history for anomaly detection
            if t >= window:
                history = traffic[t - window:t, s]
                mean = history.mean()
                std = history.std() + 1e-6
                z_score = (traffic[t, s] - mean) / std
                if abs(z_score) > 3.0:
                    anomalies.append(f"S{s+1}(z={z_score:.1f})")

        if anomalies:
            print(f" | {', '.join(anomalies)}")
        else:
            print(f" | -")

    print(f"\n  SDN's centralized view enables network-wide anomaly detection.")
    print(f"  Traditional networks require per-device monitoring (SNMP polling).")
    print()


# ============================================================
# Exercise 5: SDN Failover — Link Failure Recovery
# ============================================================
def exercise_5():
    """
    Implement fast failover in SDN: when a link fails, the controller
    recomputes paths and updates flow entries.
    """
    print("=== Exercise 5: SDN Link Failure Recovery ===\n")

    class SDNNetwork:
        def __init__(self):
            self.adj = defaultdict(dict)
            self.flow_tables = defaultdict(list)

        def add_link(self, a, b):
            self.adj[a][b] = 1
            self.adj[b][a] = 1

        def remove_link(self, a, b):
            self.adj[a].pop(b, None)
            self.adj[b].pop(a, None)

        def bfs_path(self, src, dst):
            if src == dst:
                return [src]
            visited = {src}
            queue = deque([(src, [src])])
            while queue:
                node, path = queue.popleft()
                for neighbor in sorted(self.adj[node]):
                    if neighbor not in visited:
                        new_path = path + [neighbor]
                        if neighbor == dst:
                            return new_path
                        visited.add(neighbor)
                        queue.append((neighbor, new_path))
            return []

        def install_routes(self, src, dst):
            """Compute and install routes for src→dst traffic."""
            path = self.bfs_path(src, dst)
            self.flow_tables.clear()
            if path:
                for i in range(len(path) - 1):
                    self.flow_tables[path[i]].append(
                        (dst, path[i + 1]))
            return path

    # Build redundant topology
    net = SDNNetwork()
    # Ring: S1 -- S2 -- S3 -- S4 -- S1
    #       with cross-link S1 -- S3
    for a, b in [("S1", "S2"), ("S2", "S3"), ("S3", "S4"),
                 ("S4", "S1"), ("S1", "S3")]:
        net.add_link(a, b)

    print(f"  Topology: S1--S2--S3--S4--S1, cross-link S1--S3")

    # Initial path
    path1 = net.install_routes("S1", "S4")
    print(f"\n  Initial path S1→S4: {' → '.join(path1)}")
    print(f"  Flow entries:")
    for sw, entries in sorted(net.flow_tables.items()):
        for dst, next_hop in entries:
            print(f"    {sw}: dst={dst} → forward to {next_hop}")

    # Simulate link failure: S1--S4 goes down
    print(f"\n  [!] Link S1--S4 failed!")
    net.remove_link("S1", "S4")

    # Controller recomputes
    path2 = net.install_routes("S1", "S4")
    print(f"  New path S1→S4: {' → '.join(path2)}")
    print(f"  Updated flow entries:")
    for sw, entries in sorted(net.flow_tables.items()):
        for dst, next_hop in entries:
            print(f"    {sw}: dst={dst} → forward to {next_hop}")

    # Another failure
    print(f"\n  [!] Link S1--S3 also failed!")
    net.remove_link("S1", "S3")
    path3 = net.install_routes("S1", "S4")
    print(f"  New path S1→S4: {' → '.join(path3)}")

    print(f"\n  SDN enables sub-second failover by centralized recomputation.")
    print(f"  Traditional routing convergence (OSPF/BGP) takes seconds to minutes.")
    print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
