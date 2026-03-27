"""
Example: Software-Defined Networking (SDN) Simulator
Topic: Networking – Lesson 20

Demonstrates core SDN concepts:
  1. Separation of control plane and data plane
  2. OpenFlow-style flow table matching
  3. Centralized controller logic (shortest-path routing)
  4. Reactive vs proactive flow installation

Run: python 20_sdn_sim.py
"""

import heapq
from collections import defaultdict


# ============================================================
# Data Plane: OpenFlow-style Switch
# ============================================================
class FlowEntry:
    """A single flow table entry (simplified OpenFlow 1.0 style)."""

    def __init__(self, match: dict, actions: list, priority: int = 0,
                 idle_timeout: int = 0):
        self.match = match          # e.g., {"dst": "10.0.0.3"}
        self.actions = actions      # e.g., [("output", 2)]
        self.priority = priority
        self.idle_timeout = idle_timeout
        self.packet_count = 0
        self.byte_count = 0

    def matches(self, packet: dict) -> bool:
        """Check if a packet matches this flow entry."""
        for field, value in self.match.items():
            if field not in packet or packet[field] != value:
                return False
        return True

    def __repr__(self):
        return (f"FlowEntry(match={self.match}, actions={self.actions}, "
                f"priority={self.priority}, pkts={self.packet_count})")


class OFSwitch:
    """An OpenFlow-capable switch with a flow table."""

    def __init__(self, dpid: int, num_ports: int = 4):
        self.dpid = dpid
        self.num_ports = num_ports
        self.flow_table: list[FlowEntry] = []
        self.controller = None  # set by controller

    def install_flow(self, entry: FlowEntry):
        """Install a flow entry (ordered by priority, highest first)."""
        self.flow_table.append(entry)
        self.flow_table.sort(key=lambda e: e.priority, reverse=True)

    def process_packet(self, packet: dict, in_port: int) -> list:
        """Process a packet: match against flow table or send to controller."""
        # Try each flow entry in priority order
        for entry in self.flow_table:
            if entry.matches(packet):
                entry.packet_count += 1
                entry.byte_count += packet.get("size", 64)
                return entry.actions

        # Table miss → send to controller (reactive mode)
        if self.controller:
            return self.controller.packet_in(self.dpid, packet, in_port)
        return [("drop",)]

    def __repr__(self):
        return f"OFSwitch(dpid={self.dpid}, flows={len(self.flow_table)})"


# ============================================================
# Control Plane: SDN Controller
# ============================================================
class SDNController:
    """Centralized SDN controller with topology discovery and routing."""

    def __init__(self):
        self.switches: dict[int, OFSwitch] = {}
        # Topology: adjacency list  dpid -> [(neighbor_dpid, local_port, remote_port)]
        self.topology: dict[int, list] = defaultdict(list)
        # Host location: host_addr -> (dpid, port)
        self.host_table: dict[str, tuple] = {}
        self.stats = {"packet_in": 0, "flow_mod": 0}

    def add_switch(self, switch: OFSwitch):
        """Register a switch with the controller."""
        switch.controller = self
        self.switches[switch.dpid] = switch
        print(f"  [Controller] Switch {switch.dpid} connected")

    def add_link(self, dpid1: int, port1: int, dpid2: int, port2: int):
        """Register a link between two switches."""
        self.topology[dpid1].append((dpid2, port1, port2))
        self.topology[dpid2].append((dpid1, port2, port1))

    def register_host(self, host_addr: str, dpid: int, port: int):
        """Record where a host is attached."""
        self.host_table[host_addr] = (dpid, port)

    def shortest_path(self, src_dpid: int, dst_dpid: int) -> list:
        """Dijkstra shortest path between two switches.

        Returns list of (dpid, out_port) hops.
        """
        if src_dpid == dst_dpid:
            return []

        dist = {src_dpid: 0}
        prev = {}
        pq = [(0, src_dpid)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, float("inf")):
                continue
            for (v, u_port, v_port) in self.topology[u]:
                nd = d + 1
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    prev[v] = (u, u_port, v_port)
                    heapq.heappush(pq, (nd, v))

        # Reconstruct path
        if dst_dpid not in prev:
            return []

        path = []
        node = dst_dpid
        while node in prev:
            parent, out_port, in_port = prev[node]
            path.append((parent, out_port))
            node = parent
        path.reverse()
        return path

    def packet_in(self, dpid: int, packet: dict, in_port: int) -> list:
        """Handle a packet-in event (table miss).

        This is the reactive flow installation path:
        1. Learn source host location
        2. Compute shortest path to destination
        3. Install flow entries along the path
        """
        self.stats["packet_in"] += 1
        src = packet.get("src", "")
        dst = packet.get("dst", "")

        # Learn source location
        if src and src not in self.host_table:
            self.host_table[src] = (dpid, in_port)

        # If destination unknown, flood
        if dst not in self.host_table:
            return [("flood",)]

        dst_dpid, dst_port = self.host_table[dst]

        # Compute path and install flows
        if dpid == dst_dpid:
            # Destination is directly connected
            self._install_flow(dpid, dst, dst_port)
            return [("output", dst_port)]

        path = self.shortest_path(dpid, dst_dpid)
        if not path:
            return [("drop",)]

        # Install flow on each switch along the path
        for (sw_dpid, out_port) in path:
            self._install_flow(sw_dpid, dst, out_port)

        # Also install on the final switch
        self._install_flow(dst_dpid, dst, dst_port)

        # Return action for the first switch
        return [("output", path[0][1])]

    def _install_flow(self, dpid: int, dst: str, out_port: int):
        """Install a flow entry on a switch (flow_mod)."""
        entry = FlowEntry(
            match={"dst": dst},
            actions=[("output", out_port)],
            priority=100,
            idle_timeout=300,
        )
        self.switches[dpid].install_flow(entry)
        self.stats["flow_mod"] += 1

    def proactive_install(self):
        """Proactive mode: pre-install flows for all known host pairs.

        Trade-off: uses more flow table space but avoids packet-in latency.
        """
        hosts = list(self.host_table.items())
        for i, (src_addr, (src_dpid, src_port)) in enumerate(hosts):
            for dst_addr, (dst_dpid, dst_port) in hosts[i + 1:]:
                # Forward direction
                path = self.shortest_path(src_dpid, dst_dpid)
                for (sw_dpid, out_port) in path:
                    self._install_flow(sw_dpid, dst_addr, out_port)
                self._install_flow(dst_dpid, dst_addr, dst_port)

                # Reverse direction
                rpath = self.shortest_path(dst_dpid, src_dpid)
                for (sw_dpid, out_port) in rpath:
                    self._install_flow(sw_dpid, src_addr, out_port)
                self._install_flow(src_dpid, src_addr, src_port)

        print(f"  [Controller] Proactive install: {self.stats['flow_mod']} "
              f"flow entries across {len(self.switches)} switches")


# ============================================================
# Demo 1: Reactive Forwarding
# ============================================================
def demo_reactive():
    """Demonstrate reactive flow installation via packet-in events."""
    print("=" * 60)
    print("Demo 1: Reactive SDN Forwarding")
    print("=" * 60)

    # Build a 4-switch linear topology: S1 -- S2 -- S3 -- S4
    controller = SDNController()

    switches = []
    for i in range(1, 5):
        sw = OFSwitch(dpid=i)
        controller.add_switch(sw)
        switches.append(sw)

    # Links between adjacent switches (port 1 = left, port 2 = right)
    controller.add_link(1, 2, 2, 1)  # S1:p2 <-> S2:p1
    controller.add_link(2, 2, 3, 1)  # S2:p2 <-> S3:p1
    controller.add_link(3, 2, 4, 1)  # S3:p2 <-> S4:p1

    # Hosts (port 3 = host-facing port)
    hosts = {
        "10.0.0.1": (1, 3),  # Host A on S1, port 3
        "10.0.0.2": (2, 3),  # Host B on S2, port 3
        "10.0.0.3": (4, 3),  # Host C on S4, port 3
    }
    for addr, (dpid, port) in hosts.items():
        controller.register_host(addr, dpid, port)

    print(f"\n  Topology: H1-[S1]--[S2]--[S3]--[S4]-H3")
    print(f"                     |")
    print(f"                     H2\n")

    # Send packets (reactive mode — first packet triggers packet-in)
    packets = [
        {"src": "10.0.0.1", "dst": "10.0.0.3", "size": 1500},
        {"src": "10.0.0.1", "dst": "10.0.0.3", "size": 1500},  # cached
        {"src": "10.0.0.3", "dst": "10.0.0.1", "size": 64},
        {"src": "10.0.0.1", "dst": "10.0.0.2", "size": 512},
    ]

    for pkt in packets:
        src_dpid = hosts[pkt["src"]][0]
        sw = switches[src_dpid - 1]
        actions = sw.process_packet(pkt, in_port=3)
        print(f"  {pkt['src']} → {pkt['dst']}: actions={actions}")

    print(f"\n  Controller stats: {controller.stats}")
    print(f"  S1 flow table: {switches[0].flow_table}")
    print()


# ============================================================
# Demo 2: Proactive vs Reactive Comparison
# ============================================================
def demo_proactive_vs_reactive():
    """Compare proactive and reactive flow installation approaches."""
    print("=" * 60)
    print("Demo 2: Proactive vs Reactive Comparison")
    print("=" * 60)

    # Build a fat-tree-like topology (simplified)
    #
    #     S1 --- S2
    #    / \    / \
    #  H1  H2 H3  H4

    controller = SDNController()
    s1 = OFSwitch(dpid=1)
    s2 = OFSwitch(dpid=2)
    controller.add_switch(s1)
    controller.add_switch(s2)
    controller.add_link(1, 3, 2, 3)  # S1:p3 <-> S2:p3

    hosts = {
        "10.0.0.1": (1, 1),
        "10.0.0.2": (1, 2),
        "10.0.0.3": (2, 1),
        "10.0.0.4": (2, 2),
    }
    for addr, (dpid, port) in hosts.items():
        controller.register_host(addr, dpid, port)

    # Proactive install
    print("\n  [Proactive mode]")
    controller.proactive_install()
    print(f"  S1 flows: {len(s1.flow_table)}, S2 flows: {len(s2.flow_table)}")

    # Now all packets are handled locally — no packet-in
    pin_before = controller.stats["packet_in"]
    s1.process_packet({"src": "10.0.0.1", "dst": "10.0.0.3"}, in_port=1)
    s1.process_packet({"src": "10.0.0.1", "dst": "10.0.0.4"}, in_port=1)
    s2.process_packet({"src": "10.0.0.3", "dst": "10.0.0.2"}, in_port=1)
    pin_after = controller.stats["packet_in"]
    print(f"  Packet-in events after 3 packets: {pin_after - pin_before} "
          f"(should be 0)")

    print(f"\n  Trade-offs:")
    print(f"    Proactive: zero first-packet latency, more TCAM usage")
    print(f"    Reactive:  on-demand, less TCAM, first-packet delay")
    print()


# ============================================================
# Demo 3: Network Slicing with Flow Priorities
# ============================================================
def demo_network_slicing():
    """Demonstrate network slicing via priority-based flow entries."""
    print("=" * 60)
    print("Demo 3: Network Slicing with Flow Priorities")
    print("=" * 60)

    sw = OFSwitch(dpid=1)
    controller = SDNController()
    controller.add_switch(sw)

    # Slice 1: VoIP traffic (high priority) → fast path
    sw.install_flow(FlowEntry(
        match={"dst": "10.0.1.0/24", "proto": "udp", "dst_port": 5060},
        actions=[("set_queue", "high"), ("output", 1)],
        priority=200,
    ))

    # Slice 2: Video streaming (medium priority) → medium path
    sw.install_flow(FlowEntry(
        match={"dst": "10.0.2.0/24", "proto": "tcp", "dst_port": 8080},
        actions=[("set_queue", "medium"), ("output", 2)],
        priority=150,
    ))

    # Default: best-effort → low priority queue
    sw.install_flow(FlowEntry(
        match={},  # match all
        actions=[("set_queue", "low"), ("output", 3)],
        priority=0,
    ))

    print(f"\n  Flow table ({len(sw.flow_table)} entries):")
    for entry in sw.flow_table:
        print(f"    {entry}")

    # Test packets
    test_packets = [
        {"src": "10.0.0.1", "dst": "10.0.1.0/24", "proto": "udp",
         "dst_port": 5060, "label": "VoIP"},
        {"src": "10.0.0.1", "dst": "10.0.2.0/24", "proto": "tcp",
         "dst_port": 8080, "label": "Video"},
        {"src": "10.0.0.1", "dst": "10.0.3.5", "proto": "tcp",
         "dst_port": 443, "label": "HTTPS"},
    ]

    print(f"\n  Packet classification:")
    for pkt in test_packets:
        label = pkt.pop("label")
        actions = sw.process_packet(pkt, in_port=0)
        print(f"    {label:>8} → {actions}")

    print(f"\n  SDN enables dynamic slicing: each tenant gets isolated")
    print(f"  forwarding rules without physical separation.")
    print()


if __name__ == "__main__":
    demo_reactive()
    demo_proactive_vs_reactive()
    demo_network_slicing()
