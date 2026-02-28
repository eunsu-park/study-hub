"""
Load Balancer Simulation

Demonstrates load balancing concepts used in cloud environments:
- Round-robin, weighted, and least-connections algorithms
- Health check mechanisms with configurable thresholds
- Session affinity (sticky sessions) for stateful applications
- Connection draining during instance removal
- Traffic distribution visualization

No cloud account required -- all behavior is simulated locally.
"""

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class HealthStatus(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"  # Finishing existing connections before removal


class Algorithm(Enum):
    ROUND_ROBIN = "round_robin"          # Equal distribution, ignores server capacity
    WEIGHTED = "weighted"                 # Proportional to assigned weights
    LEAST_CONNECTIONS = "least_connections"  # Route to least-busy server


@dataclass
class BackendServer:
    """A server behind the load balancer (e.g., an EC2 instance).
    Each server has a weight (capacity indicator) and tracks active connections."""
    server_id: str
    weight: int = 1                   # Higher weight = more traffic
    health: HealthStatus = HealthStatus.HEALTHY
    active_connections: int = 0
    max_connections: int = 1000
    total_requests_served: int = 0
    # Health check state -- servers must pass N consecutive checks to be healthy
    consecutive_healthy: int = 0
    consecutive_unhealthy: int = 0

    @property
    def is_available(self) -> bool:
        return self.health == HealthStatus.HEALTHY

    def handle_request(self) -> float:
        """Simulate handling a request. Returns response time in ms.
        Response time degrades as the server approaches max connections
        -- this is why least-connections routing can improve latency."""
        self.active_connections += 1
        self.total_requests_served += 1
        # Response time increases with load (simulating resource contention)
        load_factor = self.active_connections / self.max_connections
        base_time = random.uniform(10, 50)
        response_time = base_time * (1 + load_factor * 3)
        # Release connection after handling
        self.active_connections = max(0, self.active_connections - 1)
        return round(response_time, 1)


@dataclass
class HealthCheckConfig:
    """Health check configuration. These parameters determine how quickly
    the load balancer detects and reacts to server failures.
    - interval: how often to check (lower = faster detection, more overhead)
    - healthy_threshold: consecutive passes needed to mark healthy
    - unhealthy_threshold: consecutive failures needed to mark unhealthy
    """
    path: str = "/health"
    interval_sec: int = 10
    timeout_sec: int = 5
    healthy_threshold: int = 3
    unhealthy_threshold: int = 2


class LoadBalancer:
    """Simulates an Application Load Balancer (ALB) with multiple routing algorithms."""

    def __init__(self, name: str, algorithm: Algorithm = Algorithm.ROUND_ROBIN):
        self.name = name
        self.algorithm = algorithm
        self.servers: List[BackendServer] = []
        self.health_config = HealthCheckConfig()
        self._rr_index = 0  # Current index for round-robin
        # Session affinity maps: client_id -> server_id
        self.sticky_sessions: Dict[str, str] = {}
        self.session_affinity_enabled = False

    def add_server(self, server: BackendServer) -> None:
        self.servers.append(server)

    def _get_healthy_servers(self) -> List[BackendServer]:
        return [s for s in self.servers if s.is_available]

    def _select_round_robin(self, healthy: List[BackendServer]) -> BackendServer:
        """Simple round-robin: rotate through servers sequentially.
        Fair but blind to server capacity or current load."""
        server = healthy[self._rr_index % len(healthy)]
        self._rr_index += 1
        return server

    def _select_weighted(self, healthy: List[BackendServer]) -> BackendServer:
        """Weighted selection: probability proportional to weight.
        Use case: mix of different instance sizes (e.g., m5.large weight=2,
        m5.xlarge weight=4) to distribute traffic proportionally to capacity."""
        total_weight = sum(s.weight for s in healthy)
        r = random.uniform(0, total_weight)
        cumulative = 0
        for server in healthy:
            cumulative += server.weight
            if r <= cumulative:
                return server
        return healthy[-1]

    def _select_least_connections(self, healthy: List[BackendServer]) -> BackendServer:
        """Route to the server with the fewest active connections.
        Best for workloads with varying request processing times -- prevents
        slow requests from overloading a single server."""
        return min(healthy, key=lambda s: s.active_connections)

    def route_request(self, client_id: Optional[str] = None) -> Optional[dict]:
        """Route a single request to a backend server."""
        healthy = self._get_healthy_servers()
        if not healthy:
            return {"error": "503 Service Unavailable - No healthy backends"}

        # Check session affinity first
        if self.session_affinity_enabled and client_id:
            sticky_server_id = self.sticky_sessions.get(client_id)
            if sticky_server_id:
                sticky = next((s for s in healthy if s.server_id == sticky_server_id), None)
                if sticky:
                    response_time = sticky.handle_request()
                    return {"server": sticky.server_id, "response_ms": response_time,
                            "sticky": True}

        # Select server based on algorithm
        selectors = {
            Algorithm.ROUND_ROBIN: self._select_round_robin,
            Algorithm.WEIGHTED: self._select_weighted,
            Algorithm.LEAST_CONNECTIONS: self._select_least_connections,
        }
        server = selectors[self.algorithm](healthy)
        response_time = server.handle_request()

        # Record sticky session
        if self.session_affinity_enabled and client_id:
            self.sticky_sessions[client_id] = server.server_id

        return {"server": server.server_id, "response_ms": response_time, "sticky": False}

    def run_health_checks(self) -> List[str]:
        """Simulate health check round. Servers randomly fail to demonstrate
        how the LB handles backend failures gracefully."""
        events = []
        for server in self.servers:
            # Simulate: 90% chance of passing health check
            passed = random.random() > 0.1

            if passed:
                server.consecutive_healthy += 1
                server.consecutive_unhealthy = 0
                if (server.health == HealthStatus.UNHEALTHY
                        and server.consecutive_healthy >= self.health_config.healthy_threshold):
                    server.health = HealthStatus.HEALTHY
                    events.append(f"  [HC] {server.server_id}: RECOVERED (passed "
                                  f"{self.health_config.healthy_threshold} consecutive checks)")
            else:
                server.consecutive_unhealthy += 1
                server.consecutive_healthy = 0
                if (server.health == HealthStatus.HEALTHY
                        and server.consecutive_unhealthy >= self.health_config.unhealthy_threshold):
                    server.health = HealthStatus.UNHEALTHY
                    events.append(f"  [HC] {server.server_id}: MARKED UNHEALTHY (failed "
                                  f"{self.health_config.unhealthy_threshold} consecutive checks)")
        return events


def compare_algorithms():
    """Compare traffic distribution across all three algorithms."""
    print("=" * 70)
    print("Algorithm Comparison (1000 requests, 3 servers)")
    print("=" * 70)

    for algo in Algorithm:
        lb = LoadBalancer(f"lb-{algo.value}", algorithm=algo)
        # Servers with different capacities (weights reflect instance sizes)
        lb.add_server(BackendServer("srv-1", weight=1, max_connections=500))
        lb.add_server(BackendServer("srv-2", weight=2, max_connections=1000))
        lb.add_server(BackendServer("srv-3", weight=3, max_connections=1500))

        distribution: Counter = Counter()
        total_latency = 0.0

        for _ in range(1000):
            result = lb.route_request()
            if "error" not in result:
                distribution[result["server"]] += 1
                total_latency += result["response_ms"]

        avg_latency = total_latency / 1000
        print(f"\n  {algo.value}:")
        print(f"    Distribution: {dict(sorted(distribution.items()))}")
        print(f"    Avg latency:  {avg_latency:.1f} ms")
    print()


def demonstrate_health_checks():
    """Show health check detection and recovery."""
    print("=" * 70)
    print("Health Check Simulation (10 rounds)")
    print("=" * 70)

    lb = LoadBalancer("lb-hc", Algorithm.ROUND_ROBIN)
    for i in range(4):
        lb.add_server(BackendServer(f"srv-{i+1}"))

    for round_num in range(1, 11):
        events = lb.run_health_checks()
        healthy_count = len(lb._get_healthy_servers())
        print(f"  Round {round_num:>2}: {healthy_count}/{len(lb.servers)} servers healthy")
        for event in events:
            print(event)
    print()


def demonstrate_sticky_sessions():
    """Show session affinity routing behavior."""
    print("=" * 70)
    print("Session Affinity (Sticky Sessions)")
    print("=" * 70)

    lb = LoadBalancer("lb-sticky", Algorithm.ROUND_ROBIN)
    lb.session_affinity_enabled = True
    for i in range(3):
        lb.add_server(BackendServer(f"srv-{i+1}"))

    # Simulate 3 clients making multiple requests each
    print(f"\n  With sticky sessions enabled:")
    for client in ["user-alice", "user-bob", "user-charlie"]:
        servers_hit = []
        for _ in range(5):
            result = lb.route_request(client_id=client)
            servers_hit.append(result["server"])
        # All requests from same client should go to same server
        unique_servers = set(servers_hit)
        print(f"    {client}: routed to {servers_hit} "
              f"(unique servers: {len(unique_servers)})")

    # Compare without sticky sessions
    lb2 = LoadBalancer("lb-no-sticky", Algorithm.ROUND_ROBIN)
    lb2.session_affinity_enabled = False
    for i in range(3):
        lb2.add_server(BackendServer(f"srv-{i+1}"))

    print(f"\n  Without sticky sessions:")
    for client in ["user-alice", "user-bob", "user-charlie"]:
        servers_hit = []
        for _ in range(5):
            result = lb2.route_request(client_id=client)
            servers_hit.append(result["server"])
        unique_servers = set(servers_hit)
        print(f"    {client}: routed to {servers_hit} "
              f"(unique servers: {len(unique_servers)})")

    print(f"\n  Insight: Sticky sessions ensure stateful apps work correctly,")
    print(f"  but they can cause uneven load distribution. Prefer stateless")
    print(f"  apps with external session stores (Redis/DynamoDB) when possible.")
    print()


if __name__ == "__main__":
    random.seed(42)
    compare_algorithms()
    demonstrate_health_checks()
    demonstrate_sticky_sessions()
