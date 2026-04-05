"""
Exercises for Lesson 04: Load Balancing
Topic: System_Design

Solutions to practice problems and hands-on exercises.
Covers load balancer selection, distribution algorithms, health checks,
and weighted least-connections implementation.
"""

import random
import time
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# === Exercise 1: Choose Load Balancer (L4/L7) ===
# Problem: Choose appropriate load balancer type for scenarios.

def exercise_1():
    """Load balancer type selection."""
    scenarios = [
        {
            "scenario": "gRPC communication between microservices",
            "choice": "L4 (NLB)",
            "reason": "gRPC is HTTP/2-based but L4 is sufficient for internal service-to-service. "
                      "Low latency, high throughput. L7 adds unnecessary overhead.",
        },
        {
            "scenario": "Multi-language website (URL: /ko/, /en/, /jp/)",
            "choice": "L7 (ALB)",
            "reason": "URL-based routing needed. /ko -> Korean server pool, "
                      "/en -> English server pool. Must inspect HTTP path.",
        },
        {
            "scenario": "Real-time game server",
            "choice": "L4 (NLB)",
            "reason": "UDP support needed for game protocols. Minimal latency is critical. "
                      "No need to inspect application layer.",
        },
        {
            "scenario": "API gateway",
            "choice": "L7 (ALB)",
            "reason": "Needs URL/header-based routing, authentication, rate limiting, "
                      "SSL termination. All require L7 inspection.",
        },
    ]

    print("Load Balancer Type Selection:")
    print("=" * 60)
    for i, s in enumerate(scenarios, 1):
        label = chr(96 + i)
        print(f"\n{label}) {s['scenario']}")
        print(f"   Choice: {s['choice']}")
        print(f"   Reason: {s['reason']}")


# === Exercise 2: Choose Distribution Algorithm ===
# Problem: Choose most suitable distribution algorithm for situations.

def exercise_2():
    """Distribution algorithm selection."""
    scenarios = [
        {
            "scenario": "All servers same specs, similar request processing times",
            "choice": "Round Robin",
            "reason": "Same servers + similar requests = simple rotation is most efficient. "
                      "No need for complex tracking.",
        },
        {
            "scenario": "WebSocket-based chat service",
            "choice": "Least Connections",
            "reason": "WebSocket connections are long-lived. Connection count-based "
                      "distribution ensures even load across servers.",
        },
        {
            "scenario": "Gradually introducing new server (canary)",
            "choice": "Weighted Round Robin",
            "reason": "New server starts with low weight (e.g., 10%). "
                      "Gradually increase weight as confidence grows.",
        },
        {
            "scenario": "Service utilizing server-side caching",
            "choice": "IP Hash / Consistent Hashing",
            "reason": "Same user always hits same server, maximizing cache hit rate. "
                      "Consistent hashing minimizes disruption when servers change.",
        },
    ]

    print("Distribution Algorithm Selection:")
    print("=" * 60)
    for i, s in enumerate(scenarios, 1):
        label = chr(96 + i)
        print(f"\n{label}) {s['scenario']}")
        print(f"   Choice: {s['choice']}")
        print(f"   Reason: {s['reason']}")


# === Exercise 3: Health Check Simulator ===
# Problem: Design health check endpoint and simulate failure detection.

@dataclass
class Server:
    server_id: int
    healthy: bool = True
    active_connections: int = 0
    weight: int = 1

    # Health check tracking
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    marked_healthy: bool = True

    # Simulation: server starts failing at this time
    fail_at: Optional[float] = None
    recover_at: Optional[float] = None

    def is_actually_healthy(self, t):
        """Simulate real server health based on time."""
        if self.fail_at and self.recover_at:
            return not (self.fail_at <= t < self.recover_at)
        return self.healthy


class HealthChecker:
    """Health checker with configurable thresholds."""

    def __init__(self, failure_threshold=3, recovery_threshold=2, interval=1.0):
        self.failure_threshold = failure_threshold
        self.recovery_threshold = recovery_threshold
        self.interval = interval  # seconds between checks

    def check(self, server, current_time):
        """Perform a health check on a server."""
        is_healthy = server.is_actually_healthy(current_time)

        if is_healthy:
            server.consecutive_failures = 0
            server.consecutive_successes += 1
            if (not server.marked_healthy and
                    server.consecutive_successes >= self.recovery_threshold):
                server.marked_healthy = True
                return "RECOVERED"
        else:
            server.consecutive_successes = 0
            server.consecutive_failures += 1
            if (server.marked_healthy and
                    server.consecutive_failures >= self.failure_threshold):
                server.marked_healthy = False
                return "FAILED"

        return "HEALTHY" if server.marked_healthy else "UNHEALTHY"


def exercise_3():
    """Health check simulation with failure detection."""
    print("Health Check Simulator:")
    print("=" * 60)

    # Server 2 fails at t=10, recovers at t=30
    servers = [
        Server(0),
        Server(1),
        Server(2, fail_at=10, recover_at=30),
    ]

    checker = HealthChecker(failure_threshold=3, recovery_threshold=2, interval=1.0)

    # Track metrics
    requests_per_server = defaultdict(int)
    dropped_requests = 0
    events = []

    print("\nHealth check endpoint design:")
    print("""
    GET /health/ready (Readiness):
    {
      "status": "healthy",
      "checks": {
        "database": {"status": "healthy", "latency_ms": 5},
        "redis": {"status": "healthy", "latency_ms": 1},
        "payment_api": {"status": "healthy", "latency_ms": 50}
      }
    }

    GET /health/live (Liveness):
    { "status": "healthy" }
    """)

    print("Simulation: Server 2 fails at t=10, recovers at t=30")
    print("-" * 60)

    for t in range(40):
        # Health checks
        for server in servers:
            status = checker.check(server, t)
            if status in ("FAILED", "RECOVERED"):
                events.append((t, server.server_id, status))
                print(f"  t={t:2d}: Server {server.server_id} -> {status}")

        # Route request to a healthy server
        healthy_servers = [s for s in servers if s.marked_healthy]
        if healthy_servers:
            chosen = random.choice(healthy_servers)
            requests_per_server[chosen.server_id] += 1
        else:
            dropped_requests += 1

    print(f"\nMetrics:")
    for sid in sorted(requests_per_server):
        print(f"  Server {sid}: {requests_per_server[sid]} requests served")
    print(f"  Dropped requests: {dropped_requests}")
    print(f"  Detection delay: ~{checker.failure_threshold} seconds after actual failure")

    # Downtime for Server 2
    actual_down = 30 - 10  # 20 seconds
    detected_down = 0
    for t in range(40):
        s2 = servers[2]
        checker.check(s2, t)
        if not s2.marked_healthy:
            detected_down += 1
    print(f"  Server 2 actual downtime: {actual_down}s")
    print(f"  Server 2 detected downtime: ~{detected_down}s")


# === Exercise 4: Weighted Least Connections ===
# Problem: Implement weighted least connections algorithm.

class WeightedLeastConnectionsLB:
    """Load balancer using weighted least connections algorithm.

    Score = active_connections / weight
    Lower score = more available relative to its capacity.
    """

    def __init__(self, servers):
        self.servers = servers

    def select(self):
        """Select server with lowest connections/weight ratio."""
        healthy = [s for s in self.servers if s.marked_healthy]
        if not healthy:
            return None

        # Score = active_connections / weight (lower is better)
        best = min(healthy, key=lambda s: s.active_connections / s.weight)
        best.active_connections += 1
        return best

    def release(self, server):
        """Release a connection from a server."""
        if server.active_connections > 0:
            server.active_connections -= 1


def exercise_4():
    """Weighted Least Connections load balancer."""
    print("Weighted Least Connections Algorithm:")
    print("=" * 60)

    # 5 servers where one has 3x the weight
    servers = [
        Server(0, weight=1),
        Server(1, weight=1),
        Server(2, weight=3),  # 3x capacity
        Server(3, weight=1),
        Server(4, weight=1),
    ]

    lb = WeightedLeastConnectionsLB(servers)

    # Simulate 1000 requests with random connection durations
    request_log = []
    active_requests = []

    random.seed(42)
    for t in range(1000):
        # Release completed requests
        active_requests = [(srv, end_t) for srv, end_t in active_requests if end_t > t]
        for srv in servers:
            srv.active_connections = sum(
                1 for s, _ in active_requests if s.server_id == srv.server_id
            )

        # New request
        server = lb.select()
        if server:
            duration = random.randint(1, 20)
            active_requests.append((server, t + duration))
            request_log.append(server.server_id)

    # Count distribution
    distribution = defaultdict(int)
    for sid in request_log:
        distribution[sid] += 1

    print("\nServer weights: [1, 1, 3, 1, 1] (total weight = 7)")
    print(f"Total requests: {len(request_log)}")
    print("\nDistribution:")
    for sid in sorted(distribution):
        count = distribution[sid]
        pct = count / len(request_log) * 100
        expected_pct = servers[sid].weight / sum(s.weight for s in servers) * 100
        bar = "#" * int(pct / 2)
        print(f"  Server {sid} (w={servers[sid].weight}): {count:>4} "
              f"({pct:5.1f}%) expected ~{expected_pct:.1f}%  [{bar}]")


# === Exercise 5: Session Affinity with Failover ===
# Problem: Implement IP-hash with consistent hashing failover.

class ConsistentHashLB:
    """Load balancer using consistent hashing for session affinity."""

    def __init__(self, servers, virtual_nodes=100):
        self.servers = {s.server_id: s for s in servers}
        self.virtual_nodes = virtual_nodes
        self.ring = {}
        self._build_ring()

    def _hash(self, key):
        """Simple hash function."""
        h = 0
        for c in str(key):
            h = (h * 31 + ord(c)) & 0xFFFFFFFF
        return h

    def _build_ring(self):
        """Build consistent hash ring."""
        self.ring = {}
        for sid, server in self.servers.items():
            if server.marked_healthy:
                for i in range(self.virtual_nodes):
                    vnode_key = f"{sid}:{i}"
                    h = self._hash(vnode_key)
                    self.ring[h] = sid

    def get_server(self, client_ip):
        """Get server for a client IP using consistent hashing."""
        if not self.ring:
            return None

        h = self._hash(client_ip)
        sorted_hashes = sorted(self.ring.keys())

        # Find first hash >= client hash (clockwise on ring)
        for ring_hash in sorted_hashes:
            if ring_hash >= h:
                return self.servers[self.ring[ring_hash]]

        # Wrap around
        return self.servers[self.ring[sorted_hashes[0]]]

    def remove_server(self, server_id):
        """Remove a server (simulate failure)."""
        if server_id in self.servers:
            self.servers[server_id].marked_healthy = False
            self._build_ring()

    def add_server(self, server_id):
        """Add server back (simulate recovery)."""
        if server_id in self.servers:
            self.servers[server_id].marked_healthy = True
            self._build_ring()


def exercise_5():
    """Session affinity with consistent hashing failover."""
    print("Session Affinity with Failover:")
    print("=" * 60)

    servers = [Server(i) for i in range(5)]
    lb = ConsistentHashLB(servers)

    # Map 100 clients to servers
    clients = [f"192.168.1.{i}" for i in range(100)]
    initial_mapping = {}
    for client in clients:
        server = lb.get_server(client)
        initial_mapping[client] = server.server_id

    print("Initial client distribution:")
    dist = defaultdict(int)
    for sid in initial_mapping.values():
        dist[sid] += 1
    for sid in sorted(dist):
        print(f"  Server {sid}: {dist[sid]} clients")

    # Fail server 2
    print("\n--- Server 2 goes down ---")
    lb.remove_server(2)

    moved_clients = 0
    new_mapping = {}
    for client in clients:
        server = lb.get_server(client)
        new_mapping[client] = server.server_id
        if new_mapping[client] != initial_mapping[client]:
            moved_clients += 1

    print(f"Clients that changed servers: {moved_clients}/{len(clients)} "
          f"({moved_clients/len(clients):.1%})")

    dist2 = defaultdict(int)
    for sid in new_mapping.values():
        dist2[sid] += 1
    print("New distribution:")
    for sid in sorted(dist2):
        print(f"  Server {sid}: {dist2[sid]} clients")

    # Compare with modular hash approach
    print("\n--- Comparison: Modular Hash Failover ---")
    # With modular hash, removing server 2 means hash % 4 instead of hash % 5
    moved_modular = 0
    for client in clients:
        h = hash(client) & 0xFFFFFFFF
        old_server = h % 5
        new_server = h % 4
        # Map new server IDs to skip server 2
        if new_server >= 2:
            new_server += 1
        if old_server != new_server:
            moved_modular += 1

    print(f"Modular hash: {moved_modular}/{len(clients)} clients moved "
          f"({moved_modular/len(clients):.1%})")
    print(f"Consistent hash: {moved_clients}/{len(clients)} clients moved "
          f"({moved_clients/len(clients):.1%})")
    print(f"Consistent hashing moves {moved_modular - moved_clients} fewer clients!")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Choose Load Balancer Type ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Choose Distribution Algorithm ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Health Check Simulator ===")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("=== Exercise 4: Weighted Least Connections ===")
    print("=" * 60)
    exercise_4()

    print("\n" + "=" * 60)
    print("=== Exercise 5: Session Affinity with Failover ===")
    print("=" * 60)
    exercise_5()

    print("\nAll exercises completed!")
