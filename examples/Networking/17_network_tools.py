"""
Network Tools Simulation

Demonstrates:
- Ping (ICMP echo) simulation
- Traceroute (TTL-based path discovery)
- Port scanner simulation
- Network latency measurement

Theory:
- Ping: sends ICMP Echo Request, measures round-trip time.
  Used to check reachability and latency.
- Traceroute: sends packets with increasing TTL values.
  Each hop returns ICMP Time Exceeded, revealing the path.
- Port scanner: attempts TCP connections to discover open ports.
  SYN scan, connect scan, etc.

Adapted from Networking Lesson 17.
"""

import random
import time
from dataclasses import dataclass, field


# ── Simulated Network ─────────────────────────────────────────────────

@dataclass
class NetworkNode:
    name: str
    ip: str
    latency_ms: float  # One-way latency to this node
    open_ports: list[int] = field(default_factory=list)
    alive: bool = True
    packet_loss: float = 0.0  # 0.0 to 1.0


# Why: Simulating the network in-memory rather than using real sockets lets
# us demonstrate tool behavior (ping, traceroute, port scan) without needing
# root privileges, network access, or risking accidental scanning of real hosts.
# The RNG with a fixed seed ensures reproducible results for demonstration.
class SimNetwork:
    """Simulated network for tool demos."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.nodes: dict[str, NetworkNode] = {}
        self.paths: dict[str, list[str]] = {}

    def add_node(self, node: NetworkNode) -> None:
        self.nodes[node.ip] = node

    def add_path(self, dst_ip: str, hops: list[str]) -> None:
        """Define the route from source to destination."""
        self.paths[dst_ip] = hops

    def ping(self, dst_ip: str) -> tuple[bool, float]:
        """Simulate ping. Returns (success, rtt_ms)."""
        node = self.nodes.get(dst_ip)
        if not node or not node.alive:
            return False, 0.0
        if self.rng.random() < node.packet_loss:
            return False, 0.0
        # Why: RTT is 2x one-way latency (there and back) plus Gaussian jitter.
        # Real networks exhibit similar jitter from queuing delays, route
        # variations, and processing time at intermediate hops. The 10% stddev
        # produces realistic-looking variation without extreme outliers.
        rtt = 2 * node.latency_ms + self.rng.gauss(0, node.latency_ms * 0.1)
        return True, max(0.1, rtt)

    # Why: Real traceroute works by sending packets with incrementally increasing
    # TTL values (1, 2, 3...). Each router that decrements TTL to 0 responds
    # with ICMP Time Exceeded, revealing its IP. We simulate this by iterating
    # through the pre-defined path, with cumulative latency growing at each hop.
    def traceroute(self, dst_ip: str) -> list[tuple[str, str, float]]:
        """Simulate traceroute. Returns [(hop#, ip, rtt)]."""
        hops = self.paths.get(dst_ip, [])
        results = []
        cumulative_latency = 0.0

        for i, hop_ip in enumerate(hops, 1):
            node = self.nodes.get(hop_ip)
            if node:
                cumulative_latency += node.latency_ms
                rtt = 2 * cumulative_latency + self.rng.gauss(0, 2)
                if self.rng.random() < node.packet_loss:
                    results.append((str(i), "*", 0.0))
                else:
                    results.append((str(i), hop_ip, max(0.1, rtt)))
            else:
                results.append((str(i), "*", 0.0))

        return results

    # Why: Port scanning reveals a host's attack surface. The three states
    # (open/closed/filtered) map to real TCP responses: open = SYN+ACK,
    # closed = RST, filtered = no response (firewall dropped the probe).
    # Nmap uses exactly these categories in its scan output.
    def port_scan(self, dst_ip: str, ports: list[int]) -> dict[int, str]:
        """Simulate port scan. Returns {port: status}."""
        node = self.nodes.get(dst_ip)
        if not node or not node.alive:
            return {p: "filtered" for p in ports}

        results = {}
        for port in ports:
            if port in node.open_ports:
                results[port] = "open"
            elif self.rng.random() < 0.1:
                results[port] = "filtered"  # Firewall drops silently
            else:
                results[port] = "closed"
        return results


def build_network() -> SimNetwork:
    net = SimNetwork()

    # Local gateway
    net.add_node(NetworkNode("gateway", "192.168.1.1", 1.0))
    # ISP router
    net.add_node(NetworkNode("isp-router", "10.0.0.1", 5.0))
    # Internet routers
    net.add_node(NetworkNode("ix-east", "198.32.132.1", 15.0))
    net.add_node(NetworkNode("ix-west", "198.32.132.2", 25.0))
    # Servers
    net.add_node(NetworkNode("web-server", "93.184.216.34", 30.0,
                             open_ports=[80, 443]))
    net.add_node(NetworkNode("dns-server", "8.8.8.8", 20.0,
                             open_ports=[53, 443]))
    net.add_node(NetworkNode("ssh-server", "10.0.1.100", 2.0,
                             open_ports=[22, 80, 443, 3306]))
    net.add_node(NetworkNode("unreachable", "10.99.99.99", 0, alive=False))
    net.add_node(NetworkNode("lossy-server", "203.0.113.50", 50.0,
                             open_ports=[80], packet_loss=0.3))

    # Paths
    net.add_path("93.184.216.34",
                 ["192.168.1.1", "10.0.0.1", "198.32.132.1", "93.184.216.34"])
    net.add_path("8.8.8.8",
                 ["192.168.1.1", "10.0.0.1", "8.8.8.8"])
    net.add_path("203.0.113.50",
                 ["192.168.1.1", "10.0.0.1", "198.32.132.2", "203.0.113.50"])

    return net


# ── Demos ──────────────────────────────────────────────────────────────

def demo_ping():
    print("=" * 60)
    print("PING SIMULATION")
    print("=" * 60)

    net = build_network()

    targets = ["93.184.216.34", "8.8.8.8", "10.99.99.99", "203.0.113.50"]

    for target in targets:
        node = net.nodes.get(target)
        name = node.name if node else "unknown"
        print(f"\n  Pinging {target} ({name}):")

        rtts = []
        lost = 0
        for i in range(5):
            success, rtt = net.ping(target)
            if success:
                rtts.append(rtt)
                print(f"    Reply: rtt={rtt:.1f}ms")
            else:
                lost += 1
                print(f"    Request timed out")

        if rtts:
            print(f"    --- Statistics ---")
            print(f"    Sent=5, Received={5-lost}, Lost={lost} "
                  f"({lost/5*100:.0f}% loss)")
            print(f"    min={min(rtts):.1f}ms, avg={sum(rtts)/len(rtts):.1f}ms, "
                  f"max={max(rtts):.1f}ms")
        else:
            print(f"    --- Host unreachable ---")


def demo_traceroute():
    print("\n" + "=" * 60)
    print("TRACEROUTE SIMULATION")
    print("=" * 60)

    net = build_network()

    targets = ["93.184.216.34", "203.0.113.50"]
    for target in targets:
        node = net.nodes.get(target)
        print(f"\n  Traceroute to {target} ({node.name if node else '?'}):")
        print(f"    {'Hop':>4}  {'IP':<20}  {'RTT':>10}")
        print(f"    {'-'*4}  {'-'*20}  {'-'*10}")

        hops = net.traceroute(target)
        for hop_num, ip, rtt in hops:
            if ip == "*":
                print(f"    {hop_num:>4}  {'*':<20}  {'*':>10}")
            else:
                name = net.nodes[ip].name if ip in net.nodes else ""
                display = f"{ip} ({name})" if name else ip
                print(f"    {hop_num:>4}  {display:<20}  {rtt:>8.1f}ms")


def demo_port_scan():
    print("\n" + "=" * 60)
    print("PORT SCAN SIMULATION")
    print("=" * 60)

    net = build_network()
    target = "10.0.1.100"
    node = net.nodes[target]
    print(f"\n  Scanning {target} ({node.name}):")

    # Well-known ports
    common_ports = {
        21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP",
        53: "DNS", 80: "HTTP", 110: "POP3", 143: "IMAP",
        443: "HTTPS", 993: "IMAPS", 3306: "MySQL", 5432: "PostgreSQL",
        6379: "Redis", 8080: "HTTP-Alt",
    }

    results = net.port_scan(target, list(common_ports.keys()))

    print(f"\n    {'Port':>6}  {'Service':<12}  {'State':<10}")
    print(f"    {'-'*6}  {'-'*12}  {'-'*10}")

    open_count = 0
    for port in sorted(results):
        state = results[port]
        service = common_ports.get(port, "unknown")
        if state == "open":
            print(f"    {port:>6}  {service:<12}  {state}")
            open_count += 1

    print(f"\n    {open_count} open port(s) found out of "
          f"{len(common_ports)} scanned")

    # Show all states
    closed = sum(1 for s in results.values() if s == "closed")
    filtered = sum(1 for s in results.values() if s == "filtered")
    print(f"    Open: {open_count}, Closed: {closed}, Filtered: {filtered}")


def demo_latency_analysis():
    print("\n" + "=" * 60)
    print("LATENCY ANALYSIS")
    print("=" * 60)

    net = build_network()

    targets = ["192.168.1.1", "10.0.0.1", "8.8.8.8", "93.184.216.34"]
    print(f"\n  {'Target':<20} {'Min':>8} {'Avg':>8} {'Max':>8} {'Jitter':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for target in targets:
        rtts = []
        for _ in range(20):
            ok, rtt = net.ping(target)
            if ok:
                rtts.append(rtt)

        if rtts:
            avg = sum(rtts) / len(rtts)
            # Why: Jitter (RTT standard deviation) is a key metric for
            # real-time applications like VoIP and video conferencing.
            # High jitter causes audio gaps and frame drops even when
            # average latency is acceptable.
            jitter = (sum((r - avg) ** 2 for r in rtts) / len(rtts)) ** 0.5
            name = net.nodes[target].name
            print(f"  {name:<20} {min(rtts):>7.1f} {avg:>7.1f} "
                  f"{max(rtts):>7.1f} {jitter:>7.1f}")


if __name__ == "__main__":
    demo_ping()
    demo_traceroute()
    demo_port_scan()
    demo_latency_analysis()
