#!/bin/bash
# Exercises for Lesson 13: Networking and Service Mesh
# Topic: DevOps
# Solutions to practice problems from the lesson.

# === Exercise 1: Load Balancing Algorithms ===
# Problem: Implement and compare common load balancing algorithms.
exercise_1() {
    echo "=== Exercise 1: Load Balancing Algorithms ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import hashlib
import random
from dataclasses import dataclass, field

@dataclass
class Backend:
    name: str
    weight: int = 1
    healthy: bool = True
    connections: int = 0

    def handle(self) -> None:
        self.connections += 1

    def release(self) -> None:
        self.connections = max(0, self.connections - 1)

# 1. Round Robin — simplest, equal distribution
class RoundRobinLB:
    def __init__(self, backends: list[Backend]):
        self.backends = backends
        self._index = 0

    def next(self) -> Backend:
        healthy = [b for b in self.backends if b.healthy]
        if not healthy:
            raise RuntimeError("No healthy backends")
        backend = healthy[self._index % len(healthy)]
        self._index += 1
        return backend

# 2. Weighted Round Robin — distribute by capacity
class WeightedRoundRobinLB:
    def __init__(self, backends: list[Backend]):
        self.backends = backends
        self._pool: list[Backend] = []
        for b in backends:
            if b.healthy:
                self._pool.extend([b] * b.weight)
        self._index = 0

    def next(self) -> Backend:
        if not self._pool:
            raise RuntimeError("No healthy backends")
        backend = self._pool[self._index % len(self._pool)]
        self._index += 1
        return backend

# 3. Least Connections — route to least busy server
class LeastConnectionsLB:
    def __init__(self, backends: list[Backend]):
        self.backends = backends

    def next(self) -> Backend:
        healthy = [b for b in self.backends if b.healthy]
        return min(healthy, key=lambda b: b.connections)

# 4. IP Hash — sticky sessions (same client -> same backend)
class IPHashLB:
    def __init__(self, backends: list[Backend]):
        self.backends = backends

    def next(self, client_ip: str) -> Backend:
        healthy = [b for b in self.backends if b.healthy]
        hash_val = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
        return healthy[hash_val % len(healthy)]

# Demo
backends = [Backend("srv-1", weight=3), Backend("srv-2", weight=1), Backend("srv-3", weight=2)]

rr = RoundRobinLB(backends)
print("Round Robin:")
for i in range(6):
    b = rr.next()
    print(f"  Request {i+1} -> {b.name}")

print("\nAlgorithm comparison:")
print(f"  {'Algorithm':<25} {'Best For':<40}")
print(f"  {'Round Robin':<25} {'Uniform servers, stateless requests':<40}")
print(f"  {'Weighted Round Robin':<25} {'Servers with different capacities':<40}")
print(f"  {'Least Connections':<25} {'Variable request durations':<40}")
print(f"  {'IP Hash':<25} {'Session affinity requirements':<40}")
SOLUTION
}

# === Exercise 2: Service Mesh Concepts ===
# Problem: Explain how a service mesh (Istio/Envoy) provides
# traffic management, security, and observability.
exercise_2() {
    echo "=== Exercise 2: Service Mesh Concepts ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
service_mesh_features = {
    "Traffic Management": {
        "capabilities": [
            "Request routing (path/header-based)",
            "Traffic splitting (canary, A/B)",
            "Circuit breaking (fail fast on degraded services)",
            "Retry and timeout policies",
            "Rate limiting per service or user",
        ],
        "how": "Envoy sidecar proxy intercepts all network traffic",
    },
    "Security": {
        "capabilities": [
            "Mutual TLS (mTLS) between all services automatically",
            "Service-to-service authorization policies",
            "Certificate rotation (no manual cert management)",
            "JWT validation at the mesh level",
        ],
        "how": "Each pod gets a sidecar with a SPIFFE identity certificate",
    },
    "Observability": {
        "capabilities": [
            "Distributed tracing (auto-inject trace headers)",
            "Metrics for every request (latency, status, retries)",
            "Service dependency graph (who calls whom)",
            "Access logging for all service-to-service calls",
        ],
        "how": "Sidecar reports telemetry to control plane (no app changes)",
    },
}

for feature, details in service_mesh_features.items():
    print(f"\n{feature}:")
    print(f"  How: {details['how']}")
    for cap in details["capabilities"]:
        print(f"    - {cap}")

# Service mesh architecture:
# +-------+    +-------+    +-------+
# | App A |    | App B |    | App C |
# +---+---+    +---+---+    +---+---+
#     |            |            |
# +---+---+    +---+---+    +---+---+
# | Envoy |<-->| Envoy |<-->| Envoy |    <-- Data plane (sidecar proxies)
# +-------+    +-------+    +-------+
#     |            |            |
# +------------------------------------------+
# |           Istiod (Control Plane)          |    <-- Config, certs, discovery
# +------------------------------------------+
SOLUTION
}

# === Exercise 3: DNS and Service Discovery ===
# Problem: Explain Kubernetes DNS resolution and how services discover
# each other in a cluster.
exercise_3() {
    echo "=== Exercise 3: DNS and Service Discovery ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Kubernetes DNS (CoreDNS) auto-creates records for Services and Pods.

dns_patterns = {
    "Service (same namespace)": {
        "dns_name": "order-api",
        "resolves_to": "ClusterIP (e.g., 10.96.45.12)",
        "example": "curl http://order-api:8080/health",
    },
    "Service (cross-namespace)": {
        "dns_name": "order-api.production.svc.cluster.local",
        "resolves_to": "ClusterIP in 'production' namespace",
        "example": "curl http://order-api.production.svc.cluster.local:8080",
    },
    "Headless Service": {
        "dns_name": "order-api-headless",
        "resolves_to": "Individual pod IPs (A records per pod)",
        "example": "Used for StatefulSets (order-api-0.order-api-headless)",
    },
    "ExternalName Service": {
        "dns_name": "external-db",
        "resolves_to": "CNAME to external-db.rds.amazonaws.com",
        "example": "Abstracts external services behind a K8s Service name",
    },
}

print("Kubernetes DNS Resolution Patterns:")
for pattern, details in dns_patterns.items():
    print(f"\n  {pattern}:")
    print(f"    DNS:      {details['dns_name']}")
    print(f"    Resolves: {details['resolves_to']}")
    print(f"    Example:  {details['example']}")

# Full FQDN format:
# <service>.<namespace>.svc.<cluster-domain>
# order-api.production.svc.cluster.local

# Debugging DNS:
# kubectl run -it dns-test --image=busybox --rm -- nslookup order-api
# kubectl exec -it <pod> -- cat /etc/resolv.conf
SOLUTION
}

# === Exercise 4: Network Policies ===
# Problem: Write Kubernetes NetworkPolicies to implement micro-segmentation.
exercise_4() {
    echo "=== Exercise 4: Network Policies ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# NetworkPolicy: firewall rules at the pod level.
# Default: all pods can talk to all pods (no isolation).
# With NetworkPolicy: deny by default, allow explicitly.

# Policy 1: Default deny all ingress in production namespace
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-ingress
  namespace: production
spec:
  podSelector: {}           # Applies to ALL pods in namespace
  policyTypes:
    - Ingress               # Deny all incoming traffic by default

---
# Policy 2: Allow order-api to receive traffic from ingress controller only
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-ingress-to-order-api
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: order-api
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-system
        - podSelector:
            matchLabels:
              app: ingress-nginx
      ports:
        - port: 8080
          protocol: TCP

---
# Policy 3: Allow order-api to connect to database only
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: order-api-egress
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: order-api
  policyTypes:
    - Egress
  egress:
    - to:
        - podSelector:
            matchLabels:
              app: postgres
      ports:
        - port: 5432
    - to:                    # Allow DNS resolution
        - namespaceSelector: {}
      ports:
        - port: 53
          protocol: UDP
        - port: 53
          protocol: TCP

# Principle: Zero-trust networking
# 1. Default deny all ingress AND egress
# 2. Allow only the specific flows your app needs
# 3. Enforce at the network layer (independent of app code)
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 13: Networking and Service Mesh"
echo "=============================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
