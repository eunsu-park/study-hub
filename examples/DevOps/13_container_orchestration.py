#!/usr/bin/env python3
"""Example: Container Orchestration Operations — Scheduling & Health Checks

Demonstrates core container orchestration concepts: pod scheduling with
resource constraints, health-check probes, rolling update simulation,
and replica set management (modeled after Kubernetes semantics).
Related lesson: 08_Container_Orchestration_Operations.md
"""

# =============================================================================
# WHY CONTAINER ORCHESTRATION?
# Running containers on a single host is simple. Running hundreds across a
# cluster requires automated scheduling, health monitoring, self-healing,
# and traffic management. This example models the key primitives that
# orchestrators like Kubernetes implement under the hood.
# =============================================================================

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# =============================================================================
# 1. RESOURCE MODEL
# =============================================================================

@dataclass
class Resources:
    """CPU (millicores) and memory (MiB) resource quantities."""
    cpu_milli: int = 0
    memory_mib: int = 0

    def fits(self, capacity: "Resources") -> bool:
        return self.cpu_milli <= capacity.cpu_milli and self.memory_mib <= capacity.memory_mib

    def subtract(self, other: "Resources") -> "Resources":
        return Resources(self.cpu_milli - other.cpu_milli, self.memory_mib - other.memory_mib)

    def __str__(self) -> str:
        return f"cpu={self.cpu_milli}m, mem={self.memory_mib}Mi"


# =============================================================================
# 2. NODE AND POD MODELS
# =============================================================================

class PodPhase(Enum):
    PENDING = "Pending"
    RUNNING = "Running"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    TERMINATING = "Terminating"


class ProbeResult(Enum):
    SUCCESS = "Success"
    FAILURE = "Failure"


@dataclass
class HealthProbe:
    """Simulates liveness/readiness probes."""
    path: str = "/healthz"
    period_seconds: int = 10
    failure_threshold: int = 3
    consecutive_failures: int = 0

    def check(self, healthy: bool = True) -> ProbeResult:
        if healthy:
            self.consecutive_failures = 0
            return ProbeResult.SUCCESS
        self.consecutive_failures += 1
        return ProbeResult.FAILURE

    def is_failed(self) -> bool:
        return self.consecutive_failures >= self.failure_threshold


@dataclass
class Pod:
    """Represents a scheduled container unit."""
    name: str
    image: str
    requests: Resources = field(default_factory=lambda: Resources(100, 128))
    node: str = ""
    phase: PodPhase = PodPhase.PENDING
    restart_count: int = 0
    liveness_probe: HealthProbe = field(default_factory=HealthProbe)
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class Node:
    """A worker node in the cluster."""
    name: str
    capacity: Resources
    allocatable: Resources = field(init=False)
    pods: list[Pod] = field(default_factory=list)
    taints: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.allocatable = Resources(self.capacity.cpu_milli, self.capacity.memory_mib)

    def can_schedule(self, pod: Pod) -> bool:
        return pod.requests.fits(self.allocatable)

    def schedule_pod(self, pod: Pod) -> bool:
        if not self.can_schedule(pod):
            return False
        self.allocatable = self.allocatable.subtract(pod.requests)
        pod.node = self.name
        pod.phase = PodPhase.RUNNING
        self.pods.append(pod)
        return True


# =============================================================================
# 3. SCHEDULER
# =============================================================================

@dataclass
class Scheduler:
    """Bin-packing scheduler that assigns pods to nodes."""
    nodes: list[Node] = field(default_factory=list)

    def schedule(self, pod: Pod) -> Optional[str]:
        """Schedule a pod using least-allocated-first strategy."""
        # Sort nodes by most available resources (spread workloads)
        candidates = sorted(
            self.nodes,
            key=lambda n: n.allocatable.cpu_milli + n.allocatable.memory_mib,
            reverse=True,
        )
        for node in candidates:
            if node.can_schedule(pod):
                node.schedule_pod(pod)
                return node.name
        return None  # No node can fit the pod


# =============================================================================
# 4. REPLICA SET CONTROLLER
# =============================================================================

@dataclass
class ReplicaSet:
    """Ensures a desired number of pod replicas are running."""
    name: str
    image: str
    replicas: int
    pod_requests: Resources = field(default_factory=lambda: Resources(200, 256))
    selector: dict[str, str] = field(default_factory=dict)
    pods: list[Pod] = field(default_factory=list)

    def reconcile(self, scheduler: Scheduler) -> list[str]:
        """Reconcile actual vs desired state. Returns action log."""
        actions: list[str] = []
        running = [p for p in self.pods if p.phase == PodPhase.RUNNING]

        # Scale up
        while len(running) < self.replicas:
            pod = Pod(
                name=f"{self.name}-{len(self.pods):03d}",
                image=self.image,
                requests=Resources(self.pod_requests.cpu_milli, self.pod_requests.memory_mib),
                labels=self.selector.copy(),
            )
            node = scheduler.schedule(pod)
            if node:
                self.pods.append(pod)
                running.append(pod)
                actions.append(f"Scheduled {pod.name} on {node}")
            else:
                actions.append(f"WARN: Cannot schedule {pod.name} — insufficient resources")
                break

        # Scale down
        while len(running) > self.replicas:
            victim = running.pop()
            victim.phase = PodPhase.TERMINATING
            actions.append(f"Terminating {victim.name}")

        return actions


# =============================================================================
# 5. ROLLING UPDATE
# =============================================================================

def rolling_update(rs: ReplicaSet, new_image: str, scheduler: Scheduler,
                   max_surge: int = 1, max_unavailable: int = 1) -> list[str]:
    """Perform a rolling update of the ReplicaSet to a new image."""
    log: list[str] = []
    old_pods = [p for p in rs.pods if p.phase == PodPhase.RUNNING]
    total = len(old_pods)
    updated = 0

    log.append(f"Rolling update: {rs.image} -> {new_image} ({total} pods)")

    while updated < total:
        # Terminate old pods (up to max_unavailable)
        batch = min(max_unavailable, total - updated)
        for i in range(batch):
            idx = updated + i
            if idx < len(old_pods):
                old_pods[idx].phase = PodPhase.TERMINATING
                log.append(f"  Terminating {old_pods[idx].name}")

        # Create new pods (up to max_surge)
        for i in range(batch):
            new_pod = Pod(
                name=f"{rs.name}-new-{updated + i:03d}",
                image=new_image,
                requests=Resources(rs.pod_requests.cpu_milli, rs.pod_requests.memory_mib),
                labels=rs.selector.copy(),
            )
            node = scheduler.schedule(new_pod)
            status = f"on {node}" if node else "PENDING (no capacity)"
            log.append(f"  Created {new_pod.name} {status}")

        updated += batch

    rs.image = new_image
    log.append(f"Rolling update complete. Image: {new_image}")
    return log


# =============================================================================
# 6. DEMO
# =============================================================================

if __name__ == "__main__":
    # Create a small cluster
    nodes = [
        Node(name="node-1", capacity=Resources(4000, 8192)),
        Node(name="node-2", capacity=Resources(4000, 8192)),
        Node(name="node-3", capacity=Resources(2000, 4096)),
    ]
    scheduler = Scheduler(nodes=nodes)

    # --- ReplicaSet reconciliation ---
    print("=" * 60)
    print("ReplicaSet Reconciliation")
    print("=" * 60)
    rs = ReplicaSet(
        name="web-app",
        image="myapp:1.0",
        replicas=5,
        selector={"app": "web"},
    )
    for action in rs.reconcile(scheduler):
        print(f"  {action}")

    # --- Health check simulation ---
    print(f"\n{'=' * 60}")
    print("Health Probe Simulation")
    print("=" * 60)
    pod = rs.pods[0]
    for i in range(5):
        healthy = i < 2  # Fails after 2 checks
        result = pod.liveness_probe.check(healthy)
        print(f"  Check {i+1}: {result.value} "
              f"(failures={pod.liveness_probe.consecutive_failures})")
    if pod.liveness_probe.is_failed():
        pod.restart_count += 1
        print(f"  -> Pod {pod.name} restarted (count={pod.restart_count})")

    # --- Rolling update ---
    print(f"\n{'=' * 60}")
    print("Rolling Update")
    print("=" * 60)
    for entry in rolling_update(rs, "myapp:2.0", scheduler):
        print(entry)

    # --- Cluster state ---
    print(f"\n{'=' * 60}")
    print("Cluster Resource Usage")
    print("=" * 60)
    for node in nodes:
        used_cpu = node.capacity.cpu_milli - node.allocatable.cpu_milli
        used_mem = node.capacity.memory_mib - node.allocatable.memory_mib
        print(f"  {node.name}: {used_cpu}m/{node.capacity.cpu_milli}m CPU, "
              f"{used_mem}Mi/{node.capacity.memory_mib}Mi RAM, "
              f"{len(node.pods)} pods")
