# Lesson 22: Service Discovery

[Overview](./00_Overview.md) | [Previous: Gossip Protocols](./21_Gossip_Protocols.md) | [Next: Distributed Rate Limiting](./23_Distributed_Rate_Limiting.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Design service discovery systems using DNS-based, registry-based, and gossip-based approaches
2. Implement health checking with liveness probes, readiness probes, and circuit breakers
3. Build client-side and server-side load balancing strategies for distributed services
4. Implement service registration with TTL, heartbeats, and deregistration
5. Compare Consul, etcd, and ZooKeeper for service discovery and configuration management

---

## Table of Contents

1. [Service Discovery Fundamentals](#1-service-discovery-fundamentals)
2. [DNS-Based Discovery](#2-dns-based-discovery)
3. [Registry-Based Discovery](#3-registry-based-discovery)
4. [Health Checking](#4-health-checking)
5. [Load Balancing Strategies](#5-load-balancing-strategies)
6. [Client-Side vs Server-Side Discovery](#6-client-side-vs-server-side-discovery)
7. [Consul Deep Dive](#7-consul-deep-dive)
8. [Service Mesh Integration](#8-service-mesh-integration)
9. [Real-World Patterns](#9-real-world-patterns)
10. [Summary and Key Takeaways](#10-summary-and-key-takeaways)
11. [Practice Problems](#11-practice-problems)
12. [References](#12-references)

---

## 1. Service Discovery Fundamentals

### 1.1 The Problem

In a distributed system with dynamically scaling services, how does Service A find the current network addresses of Service B's instances?

```
Static configuration:                  Dynamic discovery:
  Service A → 10.0.1.5:8080           Service A → Registry → 10.0.1.5:8080
                                                           → 10.0.2.3:8080
  Problem: What if B moves?                                → 10.0.3.7:8080
  What if B scales to 3 instances?       Registry tracks live instances
```

### 1.2 Discovery Patterns

```python
import time
import random
import hashlib
import threading
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum


class DiscoveryPattern(Enum):
    DNS_BASED = "dns"          # DNS SRV/A records
    REGISTRY = "registry"      # Dedicated service registry
    GOSSIP = "gossip"          # Peer-to-peer gossip
    PLATFORM = "platform"      # Platform-native (K8s, ECS)


@dataclass
class ServiceInstance:
    """A single instance of a service."""
    service_name: str
    instance_id: str
    host: str
    port: int
    metadata: dict = field(default_factory=dict)
    health: str = "healthy"  # healthy, unhealthy, draining
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    ttl: float = 30.0  # Seconds before auto-deregister
    weight: int = 100   # Load balancing weight
    zone: str = ""       # Availability zone
```

---

## 2. DNS-Based Discovery

### 2.1 DNS SRV Records

```python
class DNSServiceDiscovery:
    """
    DNS-based service discovery using SRV records.

    DNS SRV records provide:
    - Service name → (host, port, priority, weight)
    - TTL-based caching for performance
    - Universal client support

    Limitations:
    - TTL caching delays propagation of changes
    - No health checking (requires external mechanism)
    - Limited metadata support
    """

    def __init__(self):
        self.records: Dict[str, list[dict]] = defaultdict(list)  # service → SRV records
        self.cache: Dict[str, dict] = {}  # Cached lookups
        self.cache_ttl: float = 5.0  # Default TTL seconds

    def register(self, service_name: str, host: str, port: int,
                 priority: int = 10, weight: int = 100, ttl: float = 30.0):
        """Register a service instance via DNS SRV record."""
        record = {
            "host": host,
            "port": port,
            "priority": priority,
            "weight": weight,
            "ttl": ttl,
        }
        self.records[service_name].append(record)

    def resolve(self, service_name: str) -> list[dict]:
        """
        Resolve a service name to instances.

        Mimics DNS resolution with caching.
        """
        # Check cache
        cached = self.cache.get(service_name)
        if cached and time.time() - cached["timestamp"] < self.cache_ttl:
            return cached["records"]

        # "DNS lookup"
        records = self.records.get(service_name, [])

        # Sort by priority (lower = higher priority), then by weight
        records = sorted(records, key=lambda r: (r["priority"], -r["weight"]))

        # Cache the result
        self.cache[service_name] = {
            "records": records,
            "timestamp": time.time(),
        }

        return records

    def weighted_select(self, service_name: str) -> Optional[dict]:
        """Select an instance using weighted random selection."""
        records = self.resolve(service_name)
        if not records:
            return None

        # Group by priority — only use highest priority group
        best_priority = records[0]["priority"]
        candidates = [r for r in records if r["priority"] == best_priority]

        # Weighted random selection within priority group
        total_weight = sum(r["weight"] for r in candidates)
        if total_weight == 0:
            return random.choice(candidates)

        r = random.uniform(0, total_weight)
        cumulative = 0
        for record in candidates:
            cumulative += record["weight"]
            if r <= cumulative:
                return record

        return candidates[-1]


def demonstrate_dns_discovery():
    """Demonstrate DNS-based service discovery."""
    print("=== DNS-Based Service Discovery ===\n")

    dns = DNSServiceDiscovery()

    # Register instances
    dns.register("api.example.com", "10.0.1.5", 8080, priority=10, weight=70)
    dns.register("api.example.com", "10.0.2.3", 8080, priority=10, weight=30)
    dns.register("api.example.com", "10.0.3.7", 8080, priority=20, weight=100)  # Backup

    # Resolve
    records = dns.resolve("api.example.com")
    print("SRV records for api.example.com:")
    for r in records:
        print(f"  {r['host']}:{r['port']} priority={r['priority']} weight={r['weight']}")

    # Weighted selection simulation
    selections = defaultdict(int)
    for _ in range(1000):
        selected = dns.weighted_select("api.example.com")
        if selected:
            selections[f"{selected['host']}:{selected['port']}"] += 1

    print(f"\nWeighted selection (1000 requests):")
    for addr, count in sorted(selections.items()):
        print(f"  {addr}: {count} ({count/10:.1f}%)")


demonstrate_dns_discovery()
```

---

## 3. Registry-Based Discovery

### 3.1 Service Registry

```python
class ServiceRegistry:
    """
    Centralized service registry with health checking.

    Services register themselves and send periodic heartbeats.
    The registry deregisters instances that miss heartbeats.
    Clients query the registry to discover healthy instances.
    """

    def __init__(self):
        self.services: Dict[str, Dict[str, ServiceInstance]] = defaultdict(dict)
        self.watchers: Dict[str, list] = defaultdict(list)  # service → callbacks
        self.lock = threading.Lock()

    def register(self, instance: ServiceInstance) -> bool:
        """Register a service instance."""
        with self.lock:
            self.services[instance.service_name][instance.instance_id] = instance
            self._notify_watchers(instance.service_name, "register", instance)
            return True

    def deregister(self, service_name: str, instance_id: str) -> bool:
        """Gracefully deregister a service instance."""
        with self.lock:
            instances = self.services.get(service_name, {})
            if instance_id in instances:
                instance = instances.pop(instance_id)
                self._notify_watchers(service_name, "deregister", instance)
                return True
            return False

    def heartbeat(self, service_name: str, instance_id: str) -> bool:
        """Process a heartbeat from a service instance."""
        with self.lock:
            instances = self.services.get(service_name, {})
            if instance_id in instances:
                instances[instance_id].last_heartbeat = time.time()
                return True
            return False

    def discover(self, service_name: str, healthy_only: bool = True) -> list[ServiceInstance]:
        """Discover instances of a service."""
        with self.lock:
            instances = list(self.services.get(service_name, {}).values())
            if healthy_only:
                instances = [i for i in instances if i.health == "healthy"]
            return instances

    def check_health(self):
        """
        Check for expired instances (missed heartbeats).

        Called periodically by the registry.
        """
        now = time.time()
        with self.lock:
            for service_name in list(self.services.keys()):
                for iid in list(self.services[service_name].keys()):
                    instance = self.services[service_name][iid]
                    if now - instance.last_heartbeat > instance.ttl:
                        instance.health = "unhealthy"
                        if now - instance.last_heartbeat > instance.ttl * 3:
                            del self.services[service_name][iid]
                            self._notify_watchers(service_name, "expired", instance)

    def watch(self, service_name: str, callback):
        """Register a watcher for service changes."""
        self.watchers[service_name].append(callback)

    def _notify_watchers(self, service_name: str, event: str, instance: ServiceInstance):
        """Notify watchers of a change."""
        for callback in self.watchers.get(service_name, []):
            try:
                callback(event, instance)
            except Exception:
                pass

    def stats(self) -> dict:
        """Return registry statistics."""
        total = sum(len(instances) for instances in self.services.values())
        healthy = sum(
            sum(1 for i in instances.values() if i.health == "healthy")
            for instances in self.services.values()
        )
        return {
            "total_services": len(self.services),
            "total_instances": total,
            "healthy_instances": healthy,
        }


def demonstrate_service_registry():
    """Demonstrate a service registry with registration and discovery."""
    print("=== Service Registry ===\n")

    registry = ServiceRegistry()

    # Register instances
    for i in range(3):
        instance = ServiceInstance(
            service_name="user-service",
            instance_id=f"user-{i}",
            host=f"10.0.1.{i+1}",
            port=8080,
            metadata={"version": "2.1.0", "region": "us-east-1"},
            zone=f"us-east-1{'abc'[i]}",
        )
        registry.register(instance)

    for i in range(2):
        instance = ServiceInstance(
            service_name="order-service",
            instance_id=f"order-{i}",
            host=f"10.0.2.{i+1}",
            port=9090,
        )
        registry.register(instance)

    # Discover
    print("Discovering user-service:")
    for inst in registry.discover("user-service"):
        print(f"  {inst.instance_id}: {inst.host}:{inst.port} [{inst.zone}]")

    print(f"\nRegistry stats: {registry.stats()}")

    # Simulate heartbeat failure
    print("\nSimulating heartbeat timeout for user-1...")
    registry.services["user-service"]["user-1"].last_heartbeat = time.time() - 100
    registry.check_health()

    print("After health check:")
    for inst in registry.discover("user-service", healthy_only=True):
        print(f"  {inst.instance_id}: {inst.health}")


demonstrate_service_registry()
```

---

## 4. Health Checking

### 4.1 Health Check Types

```python
class HealthCheckType(Enum):
    HTTP = "http"           # HTTP GET to health endpoint
    TCP = "tcp"             # TCP connection check
    GRPC = "grpc"           # gRPC health check protocol
    SCRIPT = "script"       # Run a script/command
    TTL = "ttl"             # Passive TTL-based (service reports in)


@dataclass
class HealthCheck:
    """Configuration for a health check."""
    check_type: HealthCheckType
    interval: float = 10.0       # Seconds between checks
    timeout: float = 5.0         # Seconds before check timeout
    deregister_after: float = 60.0  # Deregister if critical this long
    healthy_threshold: int = 3    # Consecutive successes to mark healthy
    unhealthy_threshold: int = 2  # Consecutive failures to mark unhealthy


class HealthChecker:
    """
    Health checking system for service instances.

    Supports liveness probes (is the process alive?),
    readiness probes (is it ready to serve traffic?),
    and startup probes (has it finished initializing?).
    """

    def __init__(self):
        self.checks: Dict[str, HealthCheck] = {}
        self.results: Dict[str, list[bool]] = defaultdict(list)
        self.status: Dict[str, str] = {}  # instance_id → healthy/unhealthy/unknown
        self.consecutive_successes: Dict[str, int] = defaultdict(int)
        self.consecutive_failures: Dict[str, int] = defaultdict(int)

    def register_check(self, instance_id: str, check: HealthCheck):
        """Register a health check for an instance."""
        self.checks[instance_id] = check
        self.status[instance_id] = "unknown"

    def record_result(self, instance_id: str, success: bool):
        """Record a health check result."""
        self.results[instance_id].append(success)
        check = self.checks.get(instance_id)
        if not check:
            return

        if success:
            self.consecutive_successes[instance_id] += 1
            self.consecutive_failures[instance_id] = 0

            if self.consecutive_successes[instance_id] >= check.healthy_threshold:
                self.status[instance_id] = "healthy"
        else:
            self.consecutive_failures[instance_id] += 1
            self.consecutive_successes[instance_id] = 0

            if self.consecutive_failures[instance_id] >= check.unhealthy_threshold:
                self.status[instance_id] = "unhealthy"

    def get_healthy(self) -> list[str]:
        """Return list of healthy instances."""
        return [iid for iid, s in self.status.items() if s == "healthy"]

    def simulate_checks(self, instance_id: str, results: list[bool]):
        """Simulate a sequence of health check results."""
        for result in results:
            self.record_result(instance_id, result)


class CircuitBreaker:
    """
    Circuit breaker for service calls.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service is failing, requests are rejected
    - HALF_OPEN: Testing if service has recovered
    """

    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 30.0,
                 half_open_max: int = 3):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_max = half_open_max

        self.state: str = "closed"
        self.failure_count: int = 0
        self.success_count: int = 0
        self.last_failure_time: float = 0
        self.half_open_attempts: int = 0

    def can_execute(self) -> bool:
        """Check if a request should be allowed."""
        if self.state == "closed":
            return True
        elif self.state == "open":
            # Check if reset timeout has elapsed
            if time.time() - self.last_failure_time >= self.reset_timeout:
                self.state = "half_open"
                self.half_open_attempts = 0
                return True
            return False
        elif self.state == "half_open":
            return self.half_open_attempts < self.half_open_max
        return False

    def record_success(self):
        """Record a successful request."""
        if self.state == "half_open":
            self.success_count += 1
            if self.success_count >= self.half_open_max:
                self.state = "closed"
                self.failure_count = 0
                self.success_count = 0
        elif self.state == "closed":
            self.failure_count = max(0, self.failure_count - 1)

    def record_failure(self):
        """Record a failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == "half_open":
            self.state = "open"
        elif self.state == "closed" and self.failure_count >= self.failure_threshold:
            self.state = "open"

    def get_state(self) -> dict:
        return {
            "state": self.state,
            "failures": self.failure_count,
            "successes": self.success_count,
        }


def demonstrate_health_checking():
    """Demonstrate health checking with circuit breaker."""
    print("=== Health Checking ===\n")

    checker = HealthChecker()
    checker.register_check("api-1", HealthCheck(
        check_type=HealthCheckType.HTTP,
        healthy_threshold=3,
        unhealthy_threshold=2,
    ))

    # Simulate: healthy, then fails, then recovers
    sequence = [True, True, True, True, False, False, True, True, True, True]
    print("Health check sequence for api-1:")
    for i, result in enumerate(sequence):
        checker.record_result("api-1", result)
        print(f"  Check {i+1}: {'PASS' if result else 'FAIL'} → "
              f"status={checker.status['api-1']}")

    # Circuit breaker demo
    print("\n=== Circuit Breaker ===\n")
    cb = CircuitBreaker(failure_threshold=3, reset_timeout=0.5)

    operations = [
        ("success", True), ("success", True), ("failure", False),
        ("failure", False), ("failure", False),  # trips open
        ("blocked", None), ("blocked", None),    # requests blocked
    ]

    for desc, success in operations:
        allowed = cb.can_execute()
        if allowed and success is not None:
            if success:
                cb.record_success()
            else:
                cb.record_failure()
        print(f"  {desc:10s}: allowed={allowed}, state={cb.get_state()}")

    # Wait for reset
    time.sleep(0.6)
    print(f"\n  After timeout: allowed={cb.can_execute()}, state={cb.get_state()}")


demonstrate_health_checking()
```

---

## 5. Load Balancing Strategies

### 5.1 Algorithm Implementations

```python
class LoadBalancer:
    """
    Load balancer with multiple strategies.

    Supports: round-robin, weighted round-robin, least-connections,
    random, consistent hashing, and zone-aware routing.
    """

    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.instances: list[ServiceInstance] = []
        self.rr_index: int = 0
        self.connections: Dict[str, int] = defaultdict(int)
        self.wrr_state: Dict[str, int] = {}

    def update_instances(self, instances: list[ServiceInstance]):
        """Update the list of available instances."""
        self.instances = [i for i in instances if i.health == "healthy"]

    def select(self, key: str = "") -> Optional[ServiceInstance]:
        """Select an instance using the configured strategy."""
        if not self.instances:
            return None

        if self.strategy == "round_robin":
            return self._round_robin()
        elif self.strategy == "weighted_round_robin":
            return self._weighted_round_robin()
        elif self.strategy == "least_connections":
            return self._least_connections()
        elif self.strategy == "random":
            return self._random_select()
        elif self.strategy == "consistent_hash":
            return self._consistent_hash(key)
        elif self.strategy == "power_of_two":
            return self._power_of_two()
        else:
            return self._round_robin()

    def _round_robin(self) -> ServiceInstance:
        """Simple round-robin selection."""
        instance = self.instances[self.rr_index % len(self.instances)]
        self.rr_index += 1
        return instance

    def _weighted_round_robin(self) -> ServiceInstance:
        """Weighted round-robin using smooth weighted algorithm."""
        if not self.wrr_state:
            self.wrr_state = {i.instance_id: 0 for i in self.instances}

        total_weight = sum(i.weight for i in self.instances)

        # Increase each by its weight
        for inst in self.instances:
            self.wrr_state[inst.instance_id] = (
                self.wrr_state.get(inst.instance_id, 0) + inst.weight
            )

        # Select highest current weight
        best = max(self.instances, key=lambda i: self.wrr_state.get(i.instance_id, 0))
        self.wrr_state[best.instance_id] -= total_weight
        return best

    def _least_connections(self) -> ServiceInstance:
        """Select instance with fewest active connections."""
        return min(self.instances,
                   key=lambda i: self.connections.get(i.instance_id, 0))

    def _random_select(self) -> ServiceInstance:
        """Random selection."""
        return random.choice(self.instances)

    def _consistent_hash(self, key: str) -> ServiceInstance:
        """Consistent hash-based selection for session affinity."""
        if not key:
            return self._random_select()
        h = int(hashlib.md5(key.encode()).hexdigest(), 16)
        idx = h % len(self.instances)
        return self.instances[idx]

    def _power_of_two(self) -> ServiceInstance:
        """
        Power-of-two-choices: pick 2 random instances,
        choose the one with fewer connections.

        This simple strategy achieves near-optimal load distribution
        with minimal overhead (just 2 random picks).
        """
        if len(self.instances) < 2:
            return self.instances[0]

        a, b = random.sample(self.instances, 2)
        conn_a = self.connections.get(a.instance_id, 0)
        conn_b = self.connections.get(b.instance_id, 0)
        return a if conn_a <= conn_b else b

    def connect(self, instance_id: str):
        """Record a new connection."""
        self.connections[instance_id] += 1

    def disconnect(self, instance_id: str):
        """Record a disconnection."""
        self.connections[instance_id] = max(0, self.connections.get(instance_id, 0) - 1)


def compare_load_balancing():
    """Compare load balancing strategies."""
    print("=== Load Balancing Strategies ===\n")

    instances = []
    for i in range(4):
        instances.append(ServiceInstance(
            service_name="api",
            instance_id=f"api-{i}",
            host=f"10.0.1.{i+1}",
            port=8080,
            weight=[100, 200, 50, 150][i],
        ))

    num_requests = 10000

    for strategy in ["round_robin", "weighted_round_robin", "random",
                     "least_connections", "power_of_two"]:
        lb = LoadBalancer(strategy=strategy)
        lb.update_instances(instances)

        counts = defaultdict(int)
        for _ in range(num_requests):
            selected = lb.select()
            if selected:
                counts[selected.instance_id] += 1
                lb.connect(selected.instance_id)
                # Simulate varying request duration
                if random.random() < 0.3:
                    lb.disconnect(selected.instance_id)

        print(f"{strategy}:")
        for iid in sorted(counts.keys()):
            pct = counts[iid] / num_requests * 100
            print(f"  {iid}: {counts[iid]:5d} ({pct:5.1f}%)")
        print()


compare_load_balancing()
```

---

## 6. Client-Side vs Server-Side Discovery

### 6.1 Comparison

```python
def compare_discovery_patterns():
    """Compare client-side and server-side service discovery."""
    print("=== Discovery Pattern Comparison ===\n")

    patterns = {
        "Client-Side Discovery": {
            "description": "Client queries registry directly, performs load balancing",
            "examples": "Netflix Eureka + Ribbon, gRPC client LB",
            "pros": ["No extra hop", "Client can make smart choices", "No LB bottleneck"],
            "cons": ["Client complexity", "Per-language implementation", "Tight coupling"],
            "diagram": """
  Client → Registry → [list of instances]
    ↓
  Client → Instance (direct)
""",
        },
        "Server-Side Discovery": {
            "description": "Load balancer/proxy handles discovery and routing",
            "examples": "AWS ALB, Nginx, Envoy, Kubernetes Service",
            "pros": ["Simple clients", "Centralized policy", "Language-agnostic"],
            "cons": ["Extra network hop", "LB can be bottleneck", "More infrastructure"],
            "diagram": """
  Client → Load Balancer → Instance
               ↕
            Registry
""",
        },
        "Service Mesh (Sidecar)": {
            "description": "Sidecar proxy handles discovery transparently",
            "examples": "Istio/Envoy, Linkerd, Consul Connect",
            "pros": ["Transparent to app", "Rich features (mTLS, retry)", "Uniform"],
            "cons": ["Resource overhead", "Operational complexity", "Latency"],
            "diagram": """
  App → Sidecar Proxy → Sidecar Proxy → App
            ↕                 ↕
         Control Plane (registry, config)
""",
        },
    }

    for name, info in patterns.items():
        print(f"── {name} ──")
        print(f"  {info['description']}")
        print(f"  Examples: {info['examples']}")
        print(f"  Pros: {', '.join(info['pros'])}")
        print(f"  Cons: {', '.join(info['cons'])}")
        print(f"  {info['diagram']}")


compare_discovery_patterns()
```

---

## 7. Consul Deep Dive

### 7.1 Consul Architecture Simulation

```python
class ConsulAgent:
    """
    Simulated Consul agent for service discovery.

    Consul uses:
    - Raft consensus for the catalog (server nodes)
    - Gossip (Serf) for membership and failure detection
    - DNS and HTTP API for service discovery
    - Health checks for automatic service management
    """

    def __init__(self, node_id: str, datacenter: str = "dc1", is_server: bool = False):
        self.node_id = node_id
        self.datacenter = datacenter
        self.is_server = is_server

        # Service catalog (Raft-replicated on servers)
        self.catalog: Dict[str, Dict[str, ServiceInstance]] = defaultdict(dict)
        # Local services (registered on this agent)
        self.local_services: Dict[str, ServiceInstance] = {}
        # Health check results
        self.health_checks: Dict[str, bool] = {}
        # KV store
        self.kv_store: Dict[str, dict] = {}

    def register_service(self, instance: ServiceInstance, check: Optional[HealthCheck] = None):
        """Register a local service with optional health check."""
        self.local_services[instance.instance_id] = instance
        self.catalog[instance.service_name][instance.instance_id] = instance
        if check:
            self.health_checks[instance.instance_id] = True  # Initially healthy

    def discover(self, service_name: str, dc: str = "", tag: str = "",
                 healthy_only: bool = True) -> list[ServiceInstance]:
        """
        Discover service instances.

        Supports:
        - Cross-datacenter queries
        - Tag-based filtering
        - Health-based filtering
        """
        instances = list(self.catalog.get(service_name, {}).values())

        if dc and dc != self.datacenter:
            return []  # Would query remote DC in real Consul

        if tag:
            instances = [
                i for i in instances
                if tag in i.metadata.get("tags", [])
            ]

        if healthy_only:
            instances = [
                i for i in instances
                if self.health_checks.get(i.instance_id, True)
            ]

        return instances

    def dns_query(self, name: str) -> list[dict]:
        """
        Simulate DNS query to Consul DNS interface.

        Format: <service>.service[.datacenter].consul
        """
        parts = name.split(".")
        if len(parts) >= 3 and parts[-1] == "consul" and parts[-2] == "service":
            service_name = parts[0]
            instances = self.discover(service_name)
            return [{"host": i.host, "port": i.port} for i in instances]
        return []

    def kv_put(self, key: str, value: str, flags: int = 0) -> bool:
        """Put a value in the KV store."""
        self.kv_store[key] = {
            "value": value,
            "flags": flags,
            "modify_index": time.time(),
        }
        return True

    def kv_get(self, key: str) -> Optional[dict]:
        """Get a value from the KV store."""
        return self.kv_store.get(key)

    def kv_list(self, prefix: str) -> list[str]:
        """List keys with a given prefix."""
        return [k for k in self.kv_store if k.startswith(prefix)]


def demonstrate_consul():
    """Demonstrate Consul service discovery features."""
    print("=== Consul Service Discovery ===\n")

    agent = ConsulAgent("agent-1", datacenter="us-east-1", is_server=True)

    # Register services
    services = [
        ServiceInstance("web", "web-1", "10.0.1.1", 8080,
                       metadata={"tags": ["v2", "primary"]}, zone="us-east-1a"),
        ServiceInstance("web", "web-2", "10.0.1.2", 8080,
                       metadata={"tags": ["v2"]}, zone="us-east-1b"),
        ServiceInstance("api", "api-1", "10.0.2.1", 9090,
                       metadata={"tags": ["v1"]}, zone="us-east-1a"),
        ServiceInstance("api", "api-2", "10.0.2.2", 9090,
                       metadata={"tags": ["v2", "canary"]}, zone="us-east-1b"),
    ]

    for svc in services:
        agent.register_service(svc, HealthCheck(check_type=HealthCheckType.HTTP))

    # DNS-style discovery
    print("DNS query: web.service.consul")
    results = agent.dns_query("web.service.consul")
    for r in results:
        print(f"  {r['host']}:{r['port']}")

    # API-style discovery
    print("\nAPI query: api service (healthy only)")
    instances = agent.discover("api")
    for i in instances:
        print(f"  {i.instance_id}: {i.host}:{i.port} tags={i.metadata.get('tags')}")

    # Mark one unhealthy
    agent.health_checks["api-1"] = False
    print("\nAfter api-1 fails health check:")
    instances = agent.discover("api", healthy_only=True)
    for i in instances:
        print(f"  {i.instance_id}: {i.host}:{i.port}")

    # KV store for configuration
    agent.kv_put("config/api/rate_limit", "1000")
    agent.kv_put("config/api/timeout_ms", "5000")
    agent.kv_put("config/web/cache_ttl", "300")

    print(f"\nKV store (config/ prefix):")
    for key in agent.kv_list("config/"):
        val = agent.kv_get(key)
        print(f"  {key} = {val['value']}")


demonstrate_consul()
```

---

## 8. Service Mesh Integration

### 8.1 Sidecar-Based Discovery

```python
class SidecarProxy:
    """
    Simulated service mesh sidecar proxy.

    The sidecar intercepts all inbound and outbound traffic,
    handling service discovery, load balancing, mTLS, retries,
    and observability transparently to the application.
    """

    def __init__(self, service_name: str, instance_id: str, registry: ServiceRegistry):
        self.service_name = service_name
        self.instance_id = instance_id
        self.registry = registry
        self.outbound_lb: Dict[str, LoadBalancer] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.request_count: int = 0
        self.retry_count: int = 0

    def resolve(self, target_service: str) -> Optional[ServiceInstance]:
        """Resolve a target service to an instance."""
        if target_service not in self.outbound_lb:
            self.outbound_lb[target_service] = LoadBalancer("power_of_two")

        instances = self.registry.discover(target_service)
        self.outbound_lb[target_service].update_instances(instances)
        return self.outbound_lb[target_service].select()

    def call(self, target_service: str, request: dict,
             max_retries: int = 3) -> dict:
        """
        Make a service call with automatic retry and circuit breaking.
        """
        self.request_count += 1

        # Check circuit breaker
        if target_service not in self.circuit_breakers:
            self.circuit_breakers[target_service] = CircuitBreaker()

        cb = self.circuit_breakers[target_service]
        if not cb.can_execute():
            return {"error": "circuit_breaker_open", "service": target_service}

        # Resolve and call with retries
        for attempt in range(max_retries + 1):
            instance = self.resolve(target_service)
            if not instance:
                return {"error": "no_instances", "service": target_service}

            # Simulate call (80% success rate)
            success = random.random() < 0.8
            if success:
                cb.record_success()
                return {
                    "ok": True,
                    "instance": instance.instance_id,
                    "attempts": attempt + 1,
                }
            else:
                cb.record_failure()
                if attempt < max_retries:
                    self.retry_count += 1

        return {"error": "all_retries_failed", "attempts": max_retries + 1}


def demonstrate_service_mesh():
    """Demonstrate service mesh sidecar discovery."""
    print("=== Service Mesh Sidecar ===\n")

    registry = ServiceRegistry()

    # Register backend services
    for i in range(3):
        registry.register(ServiceInstance(
            service_name="payment-service",
            instance_id=f"payment-{i}",
            host=f"10.0.3.{i+1}",
            port=8080,
        ))

    # Create sidecar for order-service
    sidecar = SidecarProxy("order-service", "order-0", registry)

    # Make 20 calls to payment-service
    results = {"ok": 0, "retry": 0, "fail": 0}
    for i in range(20):
        result = sidecar.call("payment-service", {"order_id": i})
        if result.get("ok"):
            results["ok"] += 1
        else:
            results["fail"] += 1

    print(f"Results: {results}")
    print(f"Total requests: {sidecar.request_count}")
    print(f"Total retries: {sidecar.retry_count}")
    print(f"Circuit breaker state: {sidecar.circuit_breakers['payment-service'].get_state()}")


demonstrate_service_mesh()
```

---

## 9. Real-World Patterns

### 9.1 System Comparison

```python
def compare_discovery_systems():
    """Compare real-world service discovery systems."""
    print("=== Service Discovery Systems ===\n")

    systems = [
        {"name": "Consul", "consensus": "Raft", "health": "Agent-based",
         "dns": "Yes", "kv": "Yes", "mesh": "Connect"},
        {"name": "etcd", "consensus": "Raft", "health": "TTL leases",
         "dns": "No (CoreDNS)", "kv": "Yes", "mesh": "No"},
        {"name": "ZooKeeper", "consensus": "ZAB", "health": "Ephemeral nodes",
         "dns": "No", "kv": "Yes (znodes)", "mesh": "No"},
        {"name": "Kubernetes", "consensus": "etcd (Raft)", "health": "Probes",
         "dns": "CoreDNS", "kv": "ConfigMap", "mesh": "Istio/Linkerd"},
        {"name": "Eureka", "consensus": "AP (peer replication)", "health": "Heartbeats",
         "dns": "No", "kv": "No", "mesh": "No"},
    ]

    header = f"{'System':<12} {'Consensus':<12} {'Health':<18} {'DNS':<14} {'KV':<14} {'Mesh'}"
    print(header)
    print("-" * len(header))
    for s in systems:
        print(f"{s['name']:<12} {s['consensus']:<12} {s['health']:<18} "
              f"{s['dns']:<14} {s['kv']:<14} {s['mesh']}")


compare_discovery_systems()
```

---

## 10. Summary and Key Takeaways

### Service Discovery Checklist

> **SERVICE DISCOVERY REQUIREMENTS**
>
> ☐ Registration with TTL and heartbeat
> ☐ Health checking (liveness + readiness)
> ☐ Load balancing (at least round-robin + power-of-two)
> ☐ Circuit breaker for cascading failure prevention
> ☐ DNS interface for simple clients
> ☐ Watch/notification for reactive updates
> ☐ Multi-datacenter support
> ☐ Graceful deregistration on shutdown

### Key Principles

1. **Health checks are non-negotiable**: Without them, clients route to dead instances.
2. **Client-side LB is faster but harder**: Server-side is simpler, sidecar is the best of both.
3. **Power-of-two-choices is underrated**: Near-optimal balance with minimal overhead.
4. **Circuit breakers prevent cascading failures**: A failed service should not take down its callers.
5. **DNS is universal but slow to update**: Use for bootstrapping; prefer registry for dynamic updates.

---

## 11. Practice Problems

### Problem 1: TTL Tuning

A service registry uses TTL=30s with heartbeat interval=10s. If network latency spikes to 15s for 1 minute, what happens? Design a TTL strategy that avoids false deregistrations while detecting real failures within 60 seconds.

### Problem 2: Load Balancer Comparison

Simulate 100,000 requests to 10 instances where 2 instances are 3x slower. Compare round-robin, least-connections, and power-of-two. Which achieves the best p99 latency?

### Problem 3: Multi-DC Discovery

Design a service discovery system that works across 3 datacenters with:
- Local-first routing (prefer same DC)
- Failover to other DCs if local instances are unhealthy
- Configuration replication across DCs

### Problem 4: Implementation Challenge

Build a complete service discovery system with:
- HTTP-based registration and discovery API
- TTL-based health checking
- Weighted round-robin load balancing
- Watch API for change notifications

### Problem 5: Failure Scenarios

Analyze what happens in each scenario and propose mitigations:
1. Registry leader election during high traffic
2. Split-brain between two DCs
3. All instances of a service crash simultaneously
4. Registry is unreachable but services are healthy

---

## 12. References

1. HashiCorp (2024). Consul documentation: https://developer.hashicorp.com/consul
2. etcd documentation: https://etcd.io/docs/
3. Hunt, P. et al. (2010). "ZooKeeper: Wait-free Coordination for Internet-scale Systems." *USENIX ATC*.
4. Netflix (2012). "Eureka! Why You Shouldn't Use ZooKeeper for Service Discovery."
5. Burns, B. (2018). "Designing Distributed Systems." O'Reilly Media.
6. Mielikainen, T. (2019). "The Power of Two Random Choices." (Survey paper)
7. Kleppmann, M. (2017). *Designing Data-Intensive Applications*, Ch. 8. O'Reilly Media.

---

[Next: Lesson 23 — Distributed Rate Limiting](./23_Distributed_Rate_Limiting.md)
