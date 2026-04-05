"""
Scalability Basics — Horizontal vs Vertical Scaling

Demonstrates:
- Vertical scaling: single server with increasing resources
- Horizontal scaling: multiple servers behind a load balancer
- Throughput and latency simulation under load
- Amdahl's law for parallel speedup limits

Theory:
- Vertical scaling (scale up): add more CPU, RAM, disk to one machine.
  Simple but has a hard ceiling and single point of failure.
- Horizontal scaling (scale out): add more machines.
  Near-linear scalability, fault-tolerant, but adds complexity
  (load balancing, data consistency, session management).
- Amdahl's law: speedup = 1 / (S + (1-S)/N), where S is the serial
  fraction and N is the number of processors/servers.

Adapted from System Design Lesson 02.
"""

import random
from dataclasses import dataclass, field


# ── Server Model ──────────────────────────────────────────────────────

@dataclass
class Server:
    """Simulated server with capacity and processing time."""
    name: str
    max_qps: int              # max queries per second
    base_latency_ms: float    # base latency at low load
    current_load: int = 0

    def latency_at_load(self, qps: int) -> float:
        """Latency increases as load approaches capacity (M/M/1 model)."""
        # Why: The M/M/1 queuing model shows that latency grows hyperbolically
        # as utilization approaches 1. This captures the real-world phenomenon
        # where a server at 90% utilization has 10x the latency of one at 50%.
        utilization = min(qps / self.max_qps, 0.99)
        return self.base_latency_ms / (1 - utilization)

    def can_handle(self, qps: int) -> bool:
        return qps < self.max_qps


# ── Vertical Scaling Simulation ──────────────────────────────────────

# Why: Vertical scaling is modeled by doubling server resources. In practice,
# doubling CPU cores does NOT double throughput due to Amdahl's law — there is
# always a serial fraction (locks, I/O, single-threaded code) that limits gains.
def simulate_vertical_scaling(base_qps: int, base_latency: float,
                               load_levels: list[int]) -> list[dict]:
    """Simulate scaling up a single server."""
    results = []
    tiers = [
        ("Small (4 CPU, 16GB)",   base_qps,     base_latency),
        ("Medium (8 CPU, 32GB)",  base_qps * 2, base_latency * 0.8),
        ("Large (16 CPU, 64GB)",  base_qps * 4, base_latency * 0.6),
        ("XLarge (32 CPU, 128GB)", base_qps * 7, base_latency * 0.5),
    ]

    for tier_name, max_qps, lat in tiers:
        server = Server(tier_name, max_qps, lat)
        tier_results = {"tier": tier_name, "max_qps": max_qps, "latencies": {}}
        for load in load_levels:
            if server.can_handle(load):
                tier_results["latencies"][load] = server.latency_at_load(load)
            else:
                tier_results["latencies"][load] = float("inf")
        results.append(tier_results)
    return results


# ── Horizontal Scaling Simulation ─────────────────────────────────────

@dataclass
class HorizontalCluster:
    """Cluster of identical servers with round-robin load balancing."""
    server_qps: int
    server_latency_ms: float
    num_servers: int
    servers: list[Server] = field(default_factory=list)

    def __post_init__(self):
        self.servers = [
            Server(f"srv-{i}", self.server_qps, self.server_latency_ms)
            for i in range(self.num_servers)
        ]

    @property
    def total_capacity(self) -> int:
        return self.server_qps * self.num_servers

    def avg_latency(self, total_qps: int) -> float:
        """Average latency with load evenly distributed."""
        # Why: With perfect load balancing, each server sees total_qps/N load.
        # This is the best case — in practice, hash-based routing can cause
        # imbalance, and sticky sessions prevent even distribution.
        per_server = total_qps / self.num_servers
        if per_server >= self.server_qps:
            return float("inf")
        return self.servers[0].latency_at_load(int(per_server))

    def can_handle(self, total_qps: int) -> bool:
        return total_qps < self.total_capacity


# ── Amdahl's Law ─────────────────────────────────────────────────────

# Why: Amdahl's law reveals the fundamental limit of horizontal scaling.
# If 5% of your workload is serial (e.g., a global lock, sequential DB write),
# then no matter how many servers you add, max speedup is 1/0.05 = 20x.
def amdahl_speedup(serial_fraction: float, num_processors: int) -> float:
    """Calculate theoretical speedup using Amdahl's law."""
    return 1 / (serial_fraction + (1 - serial_fraction) / num_processors)


def gustafson_speedup(serial_fraction: float, num_processors: int) -> float:
    """Gustafson's law: scaled speedup assuming problem size grows with N."""
    return num_processors - serial_fraction * (num_processors - 1)


# ── Throughput Simulation ─────────────────────────────────────────────

def simulate_requests(cluster: HorizontalCluster, total_qps: int,
                      duration_seconds: int = 10) -> dict:
    """Simulate request processing over time."""
    total_requests = total_qps * duration_seconds
    successful = 0
    latencies = []

    for _ in range(total_requests):
        # Round-robin assignment
        server_idx = random.randint(0, cluster.num_servers - 1)
        server = cluster.servers[server_idx]
        per_server_qps = total_qps // cluster.num_servers

        if server.can_handle(per_server_qps + random.randint(-10, 10)):
            lat = server.latency_at_load(per_server_qps + random.randint(-5, 5))
            latencies.append(lat)
            successful += 1

    latencies.sort()
    return {
        "total": total_requests,
        "successful": successful,
        "success_rate": successful / total_requests * 100,
        "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
        "p50": latencies[len(latencies) // 2] if latencies else 0,
        "p99": latencies[int(len(latencies) * 0.99)] if latencies else 0,
    }


# ── Demos ─────────────────────────────────────────────────────────────

def demo_vertical():
    print("=" * 60)
    print("VERTICAL SCALING (Scale Up)")
    print("=" * 60)

    loads = [100, 500, 1000, 2000, 5000]
    results = simulate_vertical_scaling(1000, 10.0, loads)

    print(f"\n  {'Tier':<28}", end="")
    for load in loads:
        print(f" {load:>7} QPS", end="")
    print()
    print(f"  {'-'*28}", end="")
    for _ in loads:
        print(f" {'-'*11}", end="")
    print()

    for tier in results:
        print(f"  {tier['tier']:<28}", end="")
        for load in loads:
            lat = tier["latencies"][load]
            if lat == float("inf"):
                print(f" {'OVERLOAD':>11}", end="")
            else:
                print(f" {lat:>8.1f} ms", end="")
        print()

    print(f"\n  Observation: Vertical scaling has diminishing returns.")
    print(f"  XLarge (8x CPU) gives ~7x throughput, not 8x (Amdahl's law).")


def demo_horizontal():
    print("\n" + "=" * 60)
    print("HORIZONTAL SCALING (Scale Out)")
    print("=" * 60)

    target_qps = 5000
    server_qps = 1000
    server_lat = 10.0

    print(f"\n  Target: {target_qps} QPS, each server handles {server_qps} QPS")
    print(f"\n  {'Servers':>8} {'Capacity':>10} {'Load %':>8} "
          f"{'Avg Latency':>12} {'Status':>10}")
    print(f"  {'-'*8} {'-'*10} {'-'*8} {'-'*12} {'-'*10}")

    for n in [1, 2, 3, 5, 8, 10]:
        cluster = HorizontalCluster(server_qps, server_lat, n)
        capacity = cluster.total_capacity
        can = cluster.can_handle(target_qps)
        if can:
            lat = cluster.avg_latency(target_qps)
            load_pct = target_qps / capacity * 100
            print(f"  {n:>8} {capacity:>10} {load_pct:>7.1f}% "
                  f"{lat:>9.1f} ms  {'OK':>10}")
        else:
            print(f"  {n:>8} {capacity:>10} {'100+%':>8} "
                  f"{'∞':>12} {'OVERLOAD':>10}")


def demo_amdahl():
    print("\n" + "=" * 60)
    print("AMDAHL'S LAW — SCALING LIMITS")
    print("=" * 60)

    serial_fractions = [0.01, 0.05, 0.10, 0.25, 0.50]
    processors = [1, 2, 4, 8, 16, 64, 256, 1024]

    print(f"\n  Speedup with N processors (serial fraction S):")
    print(f"\n  {'S':>6}", end="")
    for n in processors:
        print(f" {f'N={n}':>7}", end="")
    print(f" {'Max':>7}")
    print(f"  {'-'*6}", end="")
    for _ in processors:
        print(f" {'-'*7}", end="")
    print(f" {'-'*7}")

    for s in serial_fractions:
        print(f"  {s:>5.0%}", end="")
        for n in processors:
            sp = amdahl_speedup(s, n)
            print(f" {sp:>7.1f}", end="")
        max_sp = 1 / s
        print(f" {max_sp:>7.1f}")

    print(f"\n  Key insight: With 5% serial work, max speedup is 20x")
    print(f"  regardless of how many servers you add.")


def demo_comparison():
    print("\n" + "=" * 60)
    print("VERTICAL vs HORIZONTAL COMPARISON")
    print("=" * 60)

    print(f"\n  {'Aspect':<25} {'Vertical':>20} {'Horizontal':>20}")
    print(f"  {'-'*25} {'-'*20} {'-'*20}")
    comparisons = [
        ("Cost curve",        "Exponential",     "Linear"),
        ("Complexity",        "Low",             "High"),
        ("Max capacity",      "Hardware limit",  "Near-infinite"),
        ("Fault tolerance",   "SPOF",            "Redundant"),
        ("Downtime to scale", "Yes (restart)",   "No (add nodes)"),
        ("Data consistency",  "Simple",          "Complex"),
        ("Session handling",  "In-memory",       "Distributed"),
    ]
    for aspect, vert, horiz in comparisons:
        print(f"  {aspect:<25} {vert:>20} {horiz:>20}")


if __name__ == "__main__":
    demo_vertical()
    demo_horizontal()
    demo_amdahl()
    demo_comparison()
