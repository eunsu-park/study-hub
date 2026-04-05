"""
Service Discovery Simulator

Simulates a service discovery system with service registration,
health checking, DNS-based resolution, and load balancing. Models
the core patterns used by Consul, etcd, and DNS-SD.

Key concepts:
- Service registration and deregistration
- Health check mechanisms (HTTP, TCP, TTL)
- DNS-based service discovery (SRV records)
- Client-side vs server-side load balancing
- Consistent health monitoring and TTL expiry

Usage:
    python 23_service_discovery.py
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum


class HealthStatus(Enum):
    PASSING = "passing"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class CheckType(Enum):
    HTTP = "http"
    TCP = "tcp"
    TTL = "ttl"


@dataclass
class HealthCheck:
    """A health check configuration."""
    check_type: CheckType
    interval_s: float = 10.0
    timeout_s: float = 5.0
    endpoint: str = ""
    last_check_time: float = 0.0
    last_status: HealthStatus = HealthStatus.UNKNOWN
    consecutive_failures: int = 0


@dataclass
class ServiceInstance:
    """A registered service instance."""
    service_name: str
    instance_id: str
    address: str
    port: int
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    health: HealthCheck = field(default_factory=lambda: HealthCheck(CheckType.HTTP))
    registered_at: float = 0.0
    weight: int = 1


class ServiceRegistry:
    """
    A service discovery registry with health checking.
    """

    def __init__(self):
        self._services: dict[str, dict[str, ServiceInstance]] = defaultdict(dict)
        self.log: list[str] = []

    def register(self, instance: ServiceInstance, current_time: float) -> None:
        """Register a service instance."""
        instance.registered_at = current_time
        instance.health.last_check_time = current_time
        instance.health.last_status = HealthStatus.PASSING
        self._services[instance.service_name][instance.instance_id] = instance
        self.log.append(
            f"REGISTER {instance.service_name}/{instance.instance_id} "
            f"at {instance.address}:{instance.port}")

    def deregister(self, service_name: str, instance_id: str) -> bool:
        """Deregister a service instance."""
        instances = self._services.get(service_name, {})
        if instance_id in instances:
            del instances[instance_id]
            self.log.append(f"DEREGISTER {service_name}/{instance_id}")
            return True
        return False

    def health_check(self, current_time: float,
                     simulate_failures: dict[str, bool] | None = None) -> None:
        """Run health checks on all instances."""
        failures = simulate_failures or {}
        for svc_name, instances in self._services.items():
            for inst_id, inst in list(instances.items()):
                check = inst.health

                if current_time - check.last_check_time < check.interval_s:
                    continue

                check.last_check_time = current_time
                healthy = failures.get(inst_id, True)

                if healthy:
                    check.last_status = HealthStatus.PASSING
                    check.consecutive_failures = 0
                else:
                    check.consecutive_failures += 1
                    if check.consecutive_failures >= 3:
                        check.last_status = HealthStatus.CRITICAL
                        self.log.append(
                            f"CRITICAL {svc_name}/{inst_id} "
                            f"({check.consecutive_failures} failures)")
                    else:
                        check.last_status = HealthStatus.WARNING
                        self.log.append(
                            f"WARNING {svc_name}/{inst_id} "
                            f"(failure {check.consecutive_failures}/3)")

    def discover(self, service_name: str,
                 only_healthy: bool = True,
                 tags: list[str] | None = None) -> list[ServiceInstance]:
        """Discover service instances."""
        instances = list(self._services.get(service_name, {}).values())

        if only_healthy:
            instances = [i for i in instances
                         if i.health.last_status == HealthStatus.PASSING]

        if tags:
            instances = [i for i in instances
                         if all(t in i.tags for t in tags)]

        return instances

    def dns_lookup(self, query: str) -> list[dict]:
        """
        Simulate DNS-SD lookup.
        Query format: "service_name.service.consul"
        Returns SRV-like records.
        """
        parts = query.split(".")
        if len(parts) < 3 or parts[1] != "service":
            return []

        svc_name = parts[0]
        instances = self.discover(svc_name, only_healthy=True)

        return [
            {
                "type": "SRV",
                "target": inst.address,
                "port": inst.port,
                "weight": inst.weight,
                "priority": 0,
            }
            for inst in instances
        ]


# ---------------------------------------------------------------------------
# Load Balancing
# ---------------------------------------------------------------------------

class LoadBalancer:
    """Client-side load balancer with multiple strategies."""

    def __init__(self, strategy: str = "round_robin", seed: int = 42):
        self.strategy = strategy
        self._rng = random.Random(seed)
        self._rr_index = 0

    def select(self, instances: list[ServiceInstance]) -> ServiceInstance | None:
        if not instances:
            return None

        if self.strategy == "round_robin":
            inst = instances[self._rr_index % len(instances)]
            self._rr_index += 1
            return inst
        elif self.strategy == "random":
            return self._rng.choice(instances)
        elif self.strategy == "weighted":
            total = sum(i.weight for i in instances)
            r = self._rng.uniform(0, total)
            cumulative = 0
            for inst in instances:
                cumulative += inst.weight
                if r <= cumulative:
                    return inst
            return instances[-1]

        return instances[0]


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_registration() -> None:
    """Demonstrate service registration and discovery."""
    print("=" * 70)
    print("Service Discovery: Registration and Lookup")
    print("=" * 70)

    registry = ServiceRegistry()

    # Register web service instances
    for i in range(3):
        inst = ServiceInstance(
            service_name="web",
            instance_id=f"web-{i}",
            address=f"10.0.1.{10+i}",
            port=8080,
            tags=["production", "v2"],
            metadata={"version": "2.1.0"},
        )
        registry.register(inst, current_time=0.0)

    # Register API instances
    for i in range(2):
        inst = ServiceInstance(
            service_name="api",
            instance_id=f"api-{i}",
            address=f"10.0.2.{10+i}",
            port=9090,
            tags=["production", "v3"],
        )
        registry.register(inst, current_time=0.0)

    print(f"\n  Registered services:")
    for line in registry.log:
        print(f"    {line}")

    # Discover
    web_instances = registry.discover("web")
    api_instances = registry.discover("api")
    print(f"\n  Discover 'web': {len(web_instances)} healthy instances")
    for inst in web_instances:
        print(f"    {inst.instance_id} => {inst.address}:{inst.port}")

    print(f"\n  Discover 'api': {len(api_instances)} healthy instances")

    # DNS lookup
    records = registry.dns_lookup("web.service.consul")
    print(f"\n  DNS lookup 'web.service.consul':")
    for rec in records:
        print(f"    SRV {rec['target']}:{rec['port']} weight={rec['weight']}")


def demo_health_checking() -> None:
    """Demonstrate health check mechanisms."""
    print("\n" + "=" * 70)
    print("Health Checking: Detecting Unhealthy Instances")
    print("=" * 70)

    registry = ServiceRegistry()

    for i in range(4):
        inst = ServiceInstance(
            service_name="backend",
            instance_id=f"backend-{i}",
            address=f"10.0.3.{10+i}",
            port=3000,
            health=HealthCheck(CheckType.HTTP, interval_s=10.0,
                               endpoint="/health"),
        )
        registry.register(inst, current_time=0.0)

    # backend-2 starts failing
    print(f"\n  4 instances registered, backend-2 starts failing at t=10:\n")

    for t in range(0, 60, 10):
        failures = {"backend-2": False} if t >= 10 else {}
        registry.health_check(float(t), simulate_failures=failures)

        healthy = registry.discover("backend", only_healthy=True)
        healthy_ids = [i.instance_id for i in healthy]
        print(f"    t={t:>3}s: healthy={healthy_ids}")

    print(f"\n  Health check events:")
    for line in registry.log:
        if "WARNING" in line or "CRITICAL" in line:
            print(f"    {line}")


def demo_load_balancing() -> None:
    """Compare load balancing strategies."""
    print("\n" + "=" * 70)
    print("Client-Side Load Balancing Strategies")
    print("=" * 70)

    instances = [
        ServiceInstance("svc", f"inst-{i}", f"10.0.0.{i}", 8080,
                        weight=w)
        for i, w in enumerate([1, 2, 3])
    ]

    strategies = ["round_robin", "random", "weighted"]
    n_requests = 30

    for strategy in strategies:
        lb = LoadBalancer(strategy, seed=42)
        counts: dict[str, int] = defaultdict(int)
        for _ in range(n_requests):
            selected = lb.select(instances)
            if selected:
                counts[selected.instance_id] += 1

        print(f"\n  {strategy} ({n_requests} requests):")
        for inst in instances:
            count = counts.get(inst.instance_id, 0)
            pct = 100.0 * count / n_requests
            bar = "#" * int(pct)
            print(f"    {inst.instance_id} (w={inst.weight}): "
                  f"{count:>3} requests ({pct:5.1f}%) {bar}")


def demo_comparison() -> None:
    """Compare service discovery approaches."""
    print("\n" + "=" * 70)
    print("Service Discovery Approaches Comparison")
    print("=" * 70)

    print("""
  ┌────────────────┬──────────────┬──────────────┬──────────────────┐
  │ Approach       │ Consul       │ etcd + proxy │ DNS-SD           │
  ├────────────────┼──────────────┼──────────────┼──────────────────┤
  │ Protocol       │ Gossip + Raft│ Raft         │ DNS (mDNS/uDNS)  │
  │ Health check   │ Built-in     │ Lease/TTL    │ SRV TTL          │
  │ Load balancing │ Client-side  │ Proxy (envoy)│ DNS round-robin  │
  │ KV store       │ Yes          │ Yes          │ No               │
  │ ACL            │ Yes          │ Yes (RBAC)   │ Limited          │
  │ Multi-DC       │ Built-in     │ Manual       │ DNS views        │
  │ Best for       │ Service mesh │ K8s native   │ Simple setups    │
  └────────────────┴──────────────┴──────────────┴──────────────────┘
""")


if __name__ == "__main__":
    demo_registration()
    demo_health_checking()
    demo_load_balancing()
    demo_comparison()
    print("Done.")
