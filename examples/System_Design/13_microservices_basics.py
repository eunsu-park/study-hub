"""
Microservices Basics — Service Registry, Discovery, API Composition

Demonstrates:
- Service registry with health checking
- Client-side and server-side service discovery
- API gateway and composition pattern
- Comparison with monolithic architecture

Theory:
- Microservices decompose an application into small, independently deployable
  services, each owning its own data and communicating via APIs.
- Service discovery: services register themselves; consumers look up
  addresses dynamically (no hardcoded URLs).
- API gateway: single entry point that routes requests, handles
  cross-cutting concerns (auth, rate limiting, aggregation).
- Trade-offs: operational complexity, network latency, distributed
  transactions, eventual consistency.

Adapted from System Design Lesson 13.
"""

import random
import time
from dataclasses import dataclass, field
from enum import Enum


# ── Service Registry ──────────────────────────────────────────────────

class HealthStatus(Enum):
    HEALTHY = "HEALTHY"
    UNHEALTHY = "UNHEALTHY"
    UNKNOWN = "UNKNOWN"


@dataclass
class ServiceInstance:
    """A single instance of a microservice."""
    service_name: str
    instance_id: str
    host: str
    port: int
    status: HealthStatus = HealthStatus.HEALTHY
    metadata: dict = field(default_factory=dict)
    last_heartbeat: float = 0.0


# Why: A service registry is the backbone of microservice communication.
# Without it, services would need hardcoded addresses, making scaling and
# failover impossible. The registry acts as a phone book — services register
# on startup and deregister on shutdown.
class ServiceRegistry:
    """Service registry with health checking (Eureka/Consul-like)."""

    def __init__(self, heartbeat_timeout: float = 30.0):
        self.services: dict[str, dict[str, ServiceInstance]] = {}
        self.heartbeat_timeout = heartbeat_timeout
        self.event_log: list[str] = []

    def register(self, instance: ServiceInstance) -> None:
        """Register a service instance."""
        if instance.service_name not in self.services:
            self.services[instance.service_name] = {}
        instance.last_heartbeat = time.monotonic()
        self.services[instance.service_name][instance.instance_id] = instance
        self.event_log.append(
            f"  REGISTER: {instance.service_name}/{instance.instance_id} "
            f"at {instance.host}:{instance.port}"
        )

    def deregister(self, service_name: str, instance_id: str) -> None:
        if service_name in self.services:
            self.services[service_name].pop(instance_id, None)
            self.event_log.append(
                f"  DEREGISTER: {service_name}/{instance_id}"
            )

    def heartbeat(self, service_name: str, instance_id: str) -> None:
        if (service_name in self.services and
                instance_id in self.services[service_name]):
            inst = self.services[service_name][instance_id]
            inst.last_heartbeat = time.monotonic()
            inst.status = HealthStatus.HEALTHY

    def get_instances(self, service_name: str) -> list[ServiceInstance]:
        """Get healthy instances of a service."""
        instances = self.services.get(service_name, {})
        return [i for i in instances.values() if i.status == HealthStatus.HEALTHY]

    def check_health(self, current_time: float) -> list[str]:
        """Mark instances as unhealthy if heartbeat expired."""
        evicted = []
        for svc_name, instances in self.services.items():
            for inst_id, inst in list(instances.items()):
                if current_time - inst.last_heartbeat > self.heartbeat_timeout:
                    inst.status = HealthStatus.UNHEALTHY
                    evicted.append(f"{svc_name}/{inst_id}")
                    self.event_log.append(
                        f"  UNHEALTHY: {svc_name}/{inst_id} "
                        f"(no heartbeat for {self.heartbeat_timeout}s)"
                    )
        return evicted


# ── Service Discovery ─────────────────────────────────────────────────

# Why: Client-side discovery (the client queries the registry and picks an
# instance) gives clients control over load balancing strategy. Server-side
# discovery (a load balancer sits between client and registry) is simpler
# for clients but adds a network hop and potential SPOF.
class ClientSideDiscovery:
    """Client queries registry directly, then calls the instance."""

    def __init__(self, registry: ServiceRegistry):
        self.registry = registry

    def call(self, service_name: str, request: str) -> dict:
        instances = self.registry.get_instances(service_name)
        if not instances:
            return {"error": f"No healthy instances for {service_name}"}

        # Round-robin (simplified: random)
        chosen = random.choice(instances)
        latency = random.uniform(1, 20)
        return {
            "instance": chosen.instance_id,
            "host": f"{chosen.host}:{chosen.port}",
            "request": request,
            "latency_ms": round(latency, 1),
            "status": "OK",
        }


class ServerSideDiscovery:
    """Load balancer queries registry on behalf of client."""

    def __init__(self, registry: ServiceRegistry):
        self.registry = registry
        self.request_counts: dict[str, int] = {}

    def call(self, service_name: str, request: str) -> dict:
        instances = self.registry.get_instances(service_name)
        if not instances:
            return {"error": f"No healthy instances for {service_name}"}

        # Round-robin via counter
        count = self.request_counts.get(service_name, 0)
        chosen = instances[count % len(instances)]
        self.request_counts[service_name] = count + 1

        lb_latency = random.uniform(0.5, 2)  # LB overhead
        svc_latency = random.uniform(1, 20)
        return {
            "instance": chosen.instance_id,
            "host": f"{chosen.host}:{chosen.port}",
            "request": request,
            "lb_latency_ms": round(lb_latency, 1),
            "svc_latency_ms": round(svc_latency, 1),
            "total_ms": round(lb_latency + svc_latency, 1),
            "status": "OK",
        }


# ── API Gateway / Composition ────────────────────────────────────────

# Why: The API composition pattern aggregates data from multiple microservices
# into a single response. Without it, clients would make N separate calls,
# increasing latency and coupling the frontend to internal service boundaries.
class APIGateway:
    """API Gateway that composes responses from multiple services."""

    def __init__(self, registry: ServiceRegistry):
        self.discovery = ClientSideDiscovery(registry)
        self.routes: dict[str, list[str]] = {}

    def register_route(self, path: str, services: list[str]) -> None:
        """Register an API route that aggregates from multiple services."""
        self.routes[path] = services

    def handle_request(self, path: str, request: str) -> dict:
        """Handle a request by calling required services and composing."""
        services = self.routes.get(path, [])
        if not services:
            return {"error": f"No route for {path}"}

        results = {}
        total_latency = 0.0

        for svc in services:
            response = self.discovery.call(svc, request)
            results[svc] = response
            total_latency += response.get("latency_ms", 0)

        return {
            "path": path,
            "composed_from": services,
            "results": results,
            "total_latency_ms": round(total_latency, 1),
        }


# ── Demos ─────────────────────────────────────────────────────────────

def demo_service_registry():
    print("=" * 60)
    print("SERVICE REGISTRY")
    print("=" * 60)

    registry = ServiceRegistry(heartbeat_timeout=30.0)

    # Register services
    services = [
        ("user-service", "user-1", "10.0.1.1", 8080),
        ("user-service", "user-2", "10.0.1.2", 8080),
        ("order-service", "order-1", "10.0.2.1", 8081),
        ("order-service", "order-2", "10.0.2.2", 8081),
        ("order-service", "order-3", "10.0.2.3", 8081),
        ("payment-service", "pay-1", "10.0.3.1", 8082),
    ]

    print(f"\n  Registering {len(services)} instances:")
    for svc, iid, host, port in services:
        inst = ServiceInstance(svc, iid, host, port)
        registry.register(inst)

    for msg in registry.event_log:
        print(msg)

    # Show registry state
    print(f"\n  Registry state:")
    for svc_name in sorted(registry.services):
        instances = registry.get_instances(svc_name)
        print(f"    {svc_name}: {len(instances)} healthy instances")
        for inst in instances:
            print(f"      - {inst.instance_id} @ {inst.host}:{inst.port}")


def demo_service_discovery():
    print("\n" + "=" * 60)
    print("SERVICE DISCOVERY PATTERNS")
    print("=" * 60)

    registry = ServiceRegistry()
    for svc, iid, host, port in [
        ("user-service", "user-1", "10.0.1.1", 8080),
        ("user-service", "user-2", "10.0.1.2", 8080),
        ("user-service", "user-3", "10.0.1.3", 8080),
    ]:
        registry.register(ServiceInstance(svc, iid, host, port))

    # Client-side discovery
    print(f"\n  --- Client-Side Discovery ---")
    client_disc = ClientSideDiscovery(registry)
    for i in range(5):
        result = client_disc.call("user-service", f"GET /users/{i}")
        print(f"    Request {i}: → {result['instance']} "
              f"({result['latency_ms']}ms)")

    # Server-side discovery
    print(f"\n  --- Server-Side Discovery ---")
    server_disc = ServerSideDiscovery(registry)
    for i in range(5):
        result = server_disc.call("user-service", f"GET /users/{i}")
        print(f"    Request {i}: → {result['instance']} "
              f"(LB:{result['lb_latency_ms']}ms + "
              f"Svc:{result['svc_latency_ms']}ms = "
              f"{result['total_ms']}ms)")

    print(f"\n  Client-side: client picks instance (Netflix Ribbon)")
    print(f"  Server-side: LB picks instance (AWS ALB, Nginx)")


def demo_api_composition():
    print("\n" + "=" * 60)
    print("API GATEWAY — COMPOSITION PATTERN")
    print("=" * 60)

    registry = ServiceRegistry()
    for svc, iid, host, port in [
        ("user-service", "user-1", "10.0.1.1", 8080),
        ("order-service", "order-1", "10.0.2.1", 8081),
        ("payment-service", "pay-1", "10.0.3.1", 8082),
        ("inventory-service", "inv-1", "10.0.4.1", 8083),
    ]:
        registry.register(ServiceInstance(svc, iid, host, port))

    gateway = APIGateway(registry)
    gateway.register_route("/api/user-dashboard",
                           ["user-service", "order-service"])
    gateway.register_route("/api/checkout",
                           ["order-service", "payment-service",
                            "inventory-service"])

    print(f"\n  Routes:")
    for path, svcs in gateway.routes.items():
        print(f"    {path} → {svcs}")

    print(f"\n  --- Request: /api/user-dashboard ---")
    result = gateway.handle_request("/api/user-dashboard", "user_id=42")
    print(f"    Composed from: {result['composed_from']}")
    for svc, resp in result["results"].items():
        print(f"    {svc}: {resp['instance']} ({resp['latency_ms']}ms)")
    print(f"    Total latency: {result['total_latency_ms']}ms (sequential)")

    print(f"\n  --- Request: /api/checkout ---")
    result = gateway.handle_request("/api/checkout", "order_id=99")
    print(f"    Composed from: {result['composed_from']}")
    for svc, resp in result["results"].items():
        print(f"    {svc}: {resp['instance']} ({resp['latency_ms']}ms)")
    print(f"    Total latency: {result['total_latency_ms']}ms (sequential)")


def demo_comparison():
    print("\n" + "=" * 60)
    print("MONOLITH vs MICROSERVICES")
    print("=" * 60)

    print(f"\n  {'Aspect':<28} {'Monolith':>16} {'Microservices':>16}")
    print(f"  {'-'*28} {'-'*16} {'-'*16}")
    rows = [
        ("Deployment",         "All-or-nothing", "Per-service"),
        ("Scaling",            "Entire app",     "Per-service"),
        ("Technology choice",  "Single stack",   "Polyglot"),
        ("Data management",    "Shared DB",      "DB per service"),
        ("Communication",      "In-process",     "Network (RPC)"),
        ("Fault isolation",    "One bug = down", "Contained"),
        ("Development speed",  "Fast (small)",   "Fast (large)"),
        ("Operational cost",   "Low",            "High"),
        ("Testing",            "Simple",         "Complex"),
        ("Transactions",       "ACID",           "Saga/eventual"),
    ]
    for aspect, mono, micro in rows:
        print(f"  {aspect:<28} {mono:>16} {micro:>16}")

    print(f"\n  Rule of thumb: start monolith, extract services when needed.")


if __name__ == "__main__":
    demo_service_registry()
    demo_service_discovery()
    demo_api_composition()
    demo_comparison()
