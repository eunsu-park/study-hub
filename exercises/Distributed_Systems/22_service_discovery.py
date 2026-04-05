"""
Exercises for Lesson 22: Service Discovery
Topic: Distributed_Systems

Solutions to practice problems from the lesson.
"""

import time
import random
from typing import Dict, List, Optional
from collections import defaultdict
from dataclasses import dataclass, field


# === Exercise 1: TTL Tuning ===
def exercise_1():
    """Design TTL strategy that handles latency spikes."""
    print("=== Exercise 1: TTL Tuning ===\n")
    print("  Current: TTL=30s, heartbeat=10s")
    print("  Problem: 15s latency for 1 minute → heartbeats arrive late")
    print()
    print("  Scenario: heartbeat at t=0, next at t=10+15=25s, next at t=20+15=35s")
    print("  With TTL=30s, instance deregistered at t=30 (missed!)")
    print()
    print("  Solution: Adaptive TTL with grace period")
    print("    1. Set TTL = 3 × heartbeat_interval + max_expected_latency")
    print("       TTL = 3 × 10 + 15 = 45s")
    print("    2. Use phi-accrual detector instead of fixed timeout")
    print("    3. Add 'suspect' state before deregistration (like SWIM)")
    print("    4. Recommended: TTL=60s, heartbeat=15s, suspect_timeout=30s")
    print("    5. Real failure detected within: TTL = 60s")


exercise_1()


# === Exercise 2: Load Balancer Comparison ===
def exercise_2():
    """Compare LB strategies with slow instances."""
    print("\n=== Exercise 2: Load Balancer with Slow Instances ===\n")

    num_requests = 100000
    num_instances = 10
    slow_instances = {2, 7}  # 3x slower

    for strategy in ["round_robin", "least_connections", "power_of_two"]:
        latencies = []
        connections = defaultdict(int)
        rr_idx = 0

        for _ in range(num_requests):
            if strategy == "round_robin":
                idx = rr_idx % num_instances
                rr_idx += 1
            elif strategy == "least_connections":
                idx = min(range(num_instances), key=lambda i: connections[i])
            elif strategy == "power_of_two":
                a, b = random.sample(range(num_instances), 2)
                idx = a if connections[a] <= connections[b] else b

            connections[idx] += 1
            base_latency = 30 if idx in slow_instances else 10
            latency = random.expovariate(1.0 / base_latency)
            latencies.append(latency)

            # Simulate request completion
            if random.random() < (10.0 / base_latency):
                connections[idx] = max(0, connections[idx] - 1)

        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p99 = latencies[int(len(latencies) * 0.99)]
        print(f"  {strategy:20s}: p50={p50:.1f}ms, p99={p99:.1f}ms")


exercise_2()


# === Exercise 3: Multi-DC Discovery Design ===
def exercise_3():
    """Design multi-DC service discovery."""
    print("\n=== Exercise 3: Multi-DC Discovery ===\n")

    class MultiDCDiscovery:
        def __init__(self, local_dc):
            self.local_dc = local_dc
            self.registries = defaultdict(lambda: defaultdict(list))

        def register(self, dc, service, instance):
            self.registries[dc][service].append(instance)

        def discover(self, service, prefer_local=True):
            # 1. Try local DC first
            local = self.registries[self.local_dc].get(service, [])
            healthy_local = [i for i in local if i.get("healthy", True)]

            if healthy_local and prefer_local:
                return healthy_local, self.local_dc

            # 2. Failover to other DCs
            for dc, services in self.registries.items():
                if dc != self.local_dc:
                    remote = services.get(service, [])
                    healthy_remote = [i for i in remote if i.get("healthy", True)]
                    if healthy_remote:
                        return healthy_remote, dc

            return [], None

    disco = MultiDCDiscovery("us-east-1")
    disco.register("us-east-1", "api", {"id": "api-1", "healthy": True})
    disco.register("us-east-1", "api", {"id": "api-2", "healthy": False})
    disco.register("us-west-2", "api", {"id": "api-3", "healthy": True})
    disco.register("eu-west-1", "api", {"id": "api-4", "healthy": True})

    instances, dc = disco.discover("api")
    print(f"  Local-first: {len(instances)} instances from {dc}")

    # All local unhealthy
    for i in disco.registries["us-east-1"]["api"]:
        i["healthy"] = False
    instances, dc = disco.discover("api")
    print(f"  Failover: {len(instances)} instances from {dc}")


exercise_3()


# === Exercise 4: Complete Discovery System ===
def exercise_4():
    """Build HTTP-based service discovery system."""
    print("\n=== Exercise 4: Service Discovery System ===\n")

    class DiscoverySystem:
        def __init__(self):
            self.services = defaultdict(dict)
            self.watchers = defaultdict(list)

        def register(self, service, instance_id, host, port, ttl=30):
            self.services[service][instance_id] = {
                "host": host, "port": port,
                "registered": time.time(), "ttl": ttl,
                "last_heartbeat": time.time(), "healthy": True,
            }
            self._notify(service, "registered", instance_id)

        def heartbeat(self, service, instance_id):
            if instance_id in self.services.get(service, {}):
                self.services[service][instance_id]["last_heartbeat"] = time.time()

        def discover(self, service, healthy_only=True):
            instances = self.services.get(service, {})
            if healthy_only:
                return {k: v for k, v in instances.items() if v["healthy"]}
            return dict(instances)

        def watch(self, service, callback):
            self.watchers[service].append(callback)

        def _notify(self, service, event, instance_id):
            for cb in self.watchers[service]:
                cb(event, instance_id)

        def health_check(self):
            now = time.time()
            for service in self.services:
                for iid, info in self.services[service].items():
                    if now - info["last_heartbeat"] > info["ttl"]:
                        info["healthy"] = False

    system = DiscoverySystem()
    events = []
    system.watch("api", lambda e, i: events.append((e, i)))

    system.register("api", "api-1", "10.0.1.1", 8080)
    system.register("api", "api-2", "10.0.1.2", 8080)

    print(f"  Registered: {list(system.discover('api').keys())}")
    print(f"  Events: {events}")


exercise_4()


# === Exercise 5: Failure Scenarios ===
def exercise_5():
    """Analyze service discovery failure scenarios."""
    print("\n=== Exercise 5: Failure Scenarios ===\n")

    scenarios = [
        ("Registry leader election during high traffic",
         "Mitigation: client-side caching with stale-while-revalidate"),
        ("Split-brain between two DCs",
         "Mitigation: each DC operates independently; eventual reconciliation"),
        ("All instances crash simultaneously",
         "Mitigation: circuit breaker + queue + retry with backoff"),
        ("Registry unreachable but services healthy",
         "Mitigation: cached discovery results + direct health probes"),
    ]

    for i, (scenario, mitigation) in enumerate(scenarios, 1):
        print(f"  {i}. {scenario}")
        print(f"     {mitigation}\n")


exercise_5()


if __name__ == "__main__":
    print("\nAll exercises completed.")
