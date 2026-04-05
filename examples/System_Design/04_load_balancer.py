"""
Load Balancer Algorithms

Demonstrates:
- Round-robin
- Weighted round-robin
- Least connections
- IP hash (consistent routing)

Theory:
- Load balancers distribute requests across servers to improve
  throughput, reduce latency, and provide fault tolerance.
- Round-robin: simple rotation, assumes equal server capacity.
- Weighted: accounts for heterogeneous server capacity.
- Least connections: routes to the server with fewest active connections.
- IP hash: deterministic routing for session affinity.

Adapted from System Design Lesson 04.
"""

import random
import hashlib
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class Server:
    name: str
    weight: int = 1
    active_connections: int = 0
    total_requests: int = 0
    healthy: bool = True


# Why: Round-robin is the simplest LB strategy — it assumes all servers have
# equal capacity. Good baseline, but in practice servers are heterogeneous.
class RoundRobinLB:
    """Simple round-robin load balancer."""

    def __init__(self, servers: list[Server]):
        self.servers = servers
        self.index = 0

    def next_server(self) -> Server:
        # Why: Filtering unhealthy servers on every call ensures instant failover
        # without needing a separate health-check loop or reconfiguration step.
        healthy = [s for s in self.servers if s.healthy]
        if not healthy:
            raise RuntimeError("No healthy servers")
        server = healthy[self.index % len(healthy)]
        self.index += 1
        server.total_requests += 1
        return server


# Why: Nginx's smooth weighted round-robin avoids bursts to high-weight servers.
# A naive approach would send N consecutive requests to the heaviest server;
# this algorithm interleaves selections so no server gets a long burst.
class WeightedRoundRobinLB:
    """Weighted round-robin using smooth weighted round-robin (Nginx-style)."""

    def __init__(self, servers: list[Server]):
        self.servers = [s for s in servers if s.healthy]
        self.current_weights = [0] * len(self.servers)
        self.total_weight = sum(s.weight for s in self.servers)

    def next_server(self) -> Server:
        # Increase each server's current weight by its configured weight
        for i, s in enumerate(self.servers):
            self.current_weights[i] += s.weight

        # Select server with highest current weight
        best_idx = max(range(len(self.servers)),
                       key=lambda i: self.current_weights[i])

        # Decrease selected server's weight by total weight
        self.current_weights[best_idx] -= self.total_weight

        server = self.servers[best_idx]
        server.total_requests += 1
        return server


# Why: Least-connections is ideal when request durations vary significantly.
# Round-robin would overload a server stuck processing slow requests, while
# this strategy naturally adapts by routing to the least-busy server.
class LeastConnectionsLB:
    """Least connections load balancer."""

    def __init__(self, servers: list[Server]):
        self.servers = servers

    def next_server(self) -> Server:
        healthy = [s for s in self.servers if s.healthy]
        server = min(healthy, key=lambda s: s.active_connections)
        server.active_connections += 1
        server.total_requests += 1
        return server

    def release(self, server: Server) -> None:
        server.active_connections = max(0, server.active_connections - 1)


# Why: IP hashing gives deterministic routing — the same client always hits the
# same server. Essential for stateful applications (e.g., shopping carts stored
# in server memory) where session affinity avoids expensive session migration.
class IPHashLB:
    """IP hash load balancer for session affinity."""

    def __init__(self, servers: list[Server]):
        self.servers = [s for s in servers if s.healthy]

    def next_server(self, client_ip: str) -> Server:
        hash_val = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
        idx = hash_val % len(self.servers)
        server = self.servers[idx]
        server.total_requests += 1
        return server


# ── Demos ───────────────────────────────────────────────────────────────

def print_distribution(servers: list[Server], label: str) -> None:
    total = sum(s.total_requests for s in servers)
    print(f"\n  {label}:")
    print(f"    {'Server':<12} {'Weight':>7} {'Requests':>9} {'Share':>7}")
    print(f"    {'-'*12} {'-'*7} {'-'*9} {'-'*7}")
    for s in servers:
        pct = s.total_requests / total * 100 if total > 0 else 0
        print(f"    {s.name:<12} {s.weight:>7} {s.total_requests:>9} {pct:>6.1f}%")


def demo_round_robin():
    print("=" * 60)
    print("ROUND-ROBIN LOAD BALANCER")
    print("=" * 60)

    servers = [Server(f"srv-{i}") for i in range(3)]
    lb = RoundRobinLB(servers)

    for _ in range(12):
        s = lb.next_server()
    print_distribution(servers, "Round-Robin (12 requests, 3 servers)")


def demo_weighted():
    print("\n" + "=" * 60)
    print("WEIGHTED ROUND-ROBIN")
    print("=" * 60)

    servers = [
        Server("large", weight=5),
        Server("medium", weight=3),
        Server("small", weight=1),
    ]
    lb = WeightedRoundRobinLB(servers)

    # Show first 9 selections
    print("\n  First 9 selections:")
    for i in range(9):
        s = lb.next_server()
        print(f"    Request {i+1}: → {s.name}")

    # Reset and run more
    for s in servers:
        s.total_requests = 0
    lb = WeightedRoundRobinLB(servers)
    for _ in range(90):
        lb.next_server()
    print_distribution(servers, "Weighted RR (90 requests)")


def demo_least_connections():
    print("\n" + "=" * 60)
    print("LEAST CONNECTIONS")
    print("=" * 60)

    servers = [Server(f"srv-{i}") for i in range(3)]
    lb = LeastConnectionsLB(servers)

    # Simulate varying connection durations
    random.seed(42)
    active: list[tuple[Server, int]] = []  # (server, release_time)

    for t in range(50):
        # Release completed connections
        remaining = []
        for srv, end_time in active:
            if t >= end_time:
                lb.release(srv)
            else:
                remaining.append((srv, end_time))
        active = remaining

        # New request
        s = lb.next_server()
        duration = random.randint(1, 10)
        active.append((s, t + duration))

    print_distribution(servers, "Least Connections (50 requests, varying duration)")
    print(f"\n    Active connections: {[s.active_connections for s in servers]}")


def demo_ip_hash():
    print("\n" + "=" * 60)
    print("IP HASH (Session Affinity)")
    print("=" * 60)

    servers = [Server(f"srv-{i}") for i in range(4)]
    lb = IPHashLB(servers)

    # Same client always goes to same server
    clients = [f"192.168.1.{i}" for i in range(10)]
    print("\n  Client routing:")
    for ip in clients:
        s = lb.next_server(ip)
        print(f"    {ip} → {s.name}")

    # Verify consistency
    print("\n  Consistency check (same clients again):")
    for s in servers:
        s.total_requests = 0
    routing = {}
    for ip in clients:
        s = lb.next_server(ip)
        routing[ip] = s.name

    for ip in clients:
        s = lb.next_server(ip)
        consistent = s.name == routing[ip]
        print(f"    {ip} → {s.name} {'✓' if consistent else '✗'}")


if __name__ == "__main__":
    demo_round_robin()
    demo_weighted()
    demo_least_connections()
    demo_ip_hash()
