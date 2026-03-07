#!/usr/bin/env python3
"""Example: GitOps Reconciliation Loop Simulation

Simulates a GitOps controller (like ArgoCD or Flux) that continuously
reconciles the desired state from a Git repository with the actual state
in a cluster. Covers drift detection, sync strategies, and health checks.
Related lesson: 14_GitOps.md
"""

# =============================================================================
# GITOPS PRINCIPLES
#   1. Declarative — desired state is described, not scripted
#   2. Versioned   — Git is the single source of truth
#   3. Automated   — approved changes are automatically applied
#   4. Self-healing — agents reconcile drift continuously
# =============================================================================

import time
import random
import json
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from copy import deepcopy


# =============================================================================
# 1. RESOURCE MODEL
# =============================================================================

class SyncStatus(Enum):
    SYNCED = "Synced"
    OUT_OF_SYNC = "OutOfSync"
    UNKNOWN = "Unknown"


class HealthStatus(Enum):
    HEALTHY = "Healthy"
    DEGRADED = "Degraded"
    PROGRESSING = "Progressing"
    MISSING = "Missing"
    UNKNOWN = "Unknown"


@dataclass
class Resource:
    """A Kubernetes-like resource with name, kind, and spec."""
    kind: str
    name: str
    namespace: str = "default"
    spec: dict[str, Any] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.namespace}/{self.kind}/{self.name}"

    def spec_hash(self) -> str:
        """Content hash for drift detection."""
        content = json.dumps(self.spec, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "namespace": self.namespace,
            "spec": self.spec,
            "labels": self.labels,
        }


# =============================================================================
# 2. GIT REPOSITORY (source of truth)
# =============================================================================

@dataclass
class GitCommit:
    """A commit in the Git repository."""
    sha: str
    message: str
    author: str
    timestamp: float
    resources: list[Resource]


class GitRepository:
    """Simulated Git repository containing desired state manifests."""

    def __init__(self, repo_url: str):
        self.repo_url = repo_url
        self._commits: list[GitCommit] = []
        self._current_idx: int = -1

    def add_commit(self, message: str, resources: list[Resource],
                   author: str = "deploy-bot") -> GitCommit:
        sha = hashlib.sha256(
            f"{message}{time.time()}{random.random()}".encode()
        ).hexdigest()[:7]
        commit = GitCommit(
            sha=sha,
            message=message,
            author=author,
            timestamp=time.time(),
            resources=resources,
        )
        self._commits.append(commit)
        self._current_idx = len(self._commits) - 1
        return commit

    @property
    def head(self) -> GitCommit | None:
        if self._current_idx < 0:
            return None
        return self._commits[self._current_idx]

    @property
    def commit_count(self) -> int:
        return len(self._commits)


# =============================================================================
# 3. CLUSTER STATE (actual state)
# =============================================================================

class ClusterState:
    """Simulated Kubernetes cluster state."""

    def __init__(self):
        self._resources: dict[str, Resource] = {}

    def apply(self, resource: Resource) -> str:
        """Apply a resource to the cluster. Returns action taken."""
        key = resource.key
        if key in self._resources:
            existing = self._resources[key]
            if existing.spec_hash() == resource.spec_hash():
                return "unchanged"
            self._resources[key] = deepcopy(resource)
            return "updated"
        self._resources[key] = deepcopy(resource)
        return "created"

    def delete(self, key: str) -> bool:
        """Delete a resource from the cluster."""
        if key in self._resources:
            del self._resources[key]
            return True
        return False

    def get(self, key: str) -> Resource | None:
        return self._resources.get(key)

    def list_resources(self) -> list[Resource]:
        return list(self._resources.values())

    def simulate_drift(self) -> list[str]:
        """Simulate random drift in the cluster (manual changes, failures)."""
        drifted = []
        for key, resource in list(self._resources.items()):
            if random.random() < 0.15:  # 15% chance of drift per resource
                # Mutate a spec field to simulate drift
                if "replicas" in resource.spec:
                    old = resource.spec["replicas"]
                    resource.spec["replicas"] = max(1, old + random.choice([-1, 1]))
                    drifted.append(f"{key}: replicas {old} -> {resource.spec['replicas']}")
                elif "image" in resource.spec:
                    resource.spec["image"] += "-drifted"
                    drifted.append(f"{key}: image tag drifted")
        return drifted


# =============================================================================
# 4. RECONCILIATION ENGINE
# =============================================================================

@dataclass
class SyncResult:
    """Result of a single reconciliation cycle."""
    cycle: int
    commit_sha: str
    sync_status: SyncStatus
    health_status: HealthStatus
    created: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    drift_detected: list[str] = field(default_factory=list)


class GitOpsReconciler:
    """GitOps reconciliation controller.

    Continuously compares desired state (Git) with actual state (cluster)
    and applies changes to converge them.
    """

    def __init__(self, repo: GitRepository, cluster: ClusterState,
                 app_name: str = "my-app"):
        self.repo = repo
        self.cluster = cluster
        self.app_name = app_name
        self._last_synced_sha: str | None = None
        self._cycle_count: int = 0
        self._history: list[SyncResult] = []
        self.prune_enabled: bool = True  # Delete resources not in Git

    def diff(self) -> tuple[list[Resource], list[Resource], list[str]]:
        """Compare desired (Git) vs actual (cluster) state.

        Returns:
            (to_apply, to_update, to_delete_keys)
        """
        head = self.repo.head
        if head is None:
            return [], [], []

        desired_map = {r.key: r for r in head.resources}
        actual_map = {r.key: r for r in self.cluster.list_resources()}

        to_apply: list[Resource] = []
        to_update: list[Resource] = []
        to_delete: list[str] = []

        # Resources in desired but not in actual (create)
        # Resources in both but different spec (update)
        for key, desired in desired_map.items():
            actual = actual_map.get(key)
            if actual is None:
                to_apply.append(desired)
            elif actual.spec_hash() != desired.spec_hash():
                to_update.append(desired)

        # Resources in actual but not in desired (prune)
        if self.prune_enabled:
            for key in actual_map:
                if key not in desired_map:
                    to_delete.append(key)

        return to_apply, to_update, to_delete

    def reconcile(self) -> SyncResult:
        """Run one reconciliation cycle."""
        self._cycle_count += 1
        head = self.repo.head

        if head is None:
            return SyncResult(
                cycle=self._cycle_count,
                commit_sha="(none)",
                sync_status=SyncStatus.UNKNOWN,
                health_status=HealthStatus.UNKNOWN,
            )

        to_create, to_update, to_delete = self.diff()

        result = SyncResult(
            cycle=self._cycle_count,
            commit_sha=head.sha,
            sync_status=SyncStatus.SYNCED,
            health_status=HealthStatus.HEALTHY,
        )

        # Apply creates
        for resource in to_create:
            action = self.cluster.apply(resource)
            result.created.append(resource.key)
            result.sync_status = SyncStatus.OUT_OF_SYNC

        # Apply updates (drift correction)
        for resource in to_update:
            action = self.cluster.apply(resource)
            result.updated.append(resource.key)
            result.drift_detected.append(resource.key)
            result.sync_status = SyncStatus.OUT_OF_SYNC

        # Prune deletes
        for key in to_delete:
            self.cluster.delete(key)
            result.deleted.append(key)
            result.sync_status = SyncStatus.OUT_OF_SYNC

        # If nothing changed, we're synced
        if not to_create and not to_update and not to_delete:
            result.sync_status = SyncStatus.SYNCED

        self._last_synced_sha = head.sha
        self._history.append(result)
        return result

    def get_sync_status(self) -> SyncStatus:
        """Check if cluster matches the latest Git commit."""
        head = self.repo.head
        if head is None:
            return SyncStatus.UNKNOWN
        to_create, to_update, to_delete = self.diff()
        if to_create or to_update or to_delete:
            return SyncStatus.OUT_OF_SYNC
        return SyncStatus.SYNCED


# =============================================================================
# 5. DEMO: SIMULATE GITOPS LIFECYCLE
# =============================================================================

def print_sync_result(result: SyncResult) -> None:
    """Pretty-print a sync result."""
    print(f"  Cycle {result.cycle} | Commit: {result.commit_sha} | "
          f"Sync: {result.sync_status.value} | "
          f"Health: {result.health_status.value}")
    if result.created:
        for r in result.created:
            print(f"    + CREATED: {r}")
    if result.updated:
        for r in result.updated:
            print(f"    ~ UPDATED: {r}")
    if result.deleted:
        for r in result.deleted:
            print(f"    - DELETED: {r}")
    if result.drift_detected:
        print(f"    ! Drift corrected: {len(result.drift_detected)} resources")
    if not result.created and not result.updated and not result.deleted:
        print(f"    (no changes needed)")


if __name__ == "__main__":
    print("=" * 70)
    print("GitOps Reconciliation Loop Simulation")
    print("=" * 70)

    repo = GitRepository("git@github.com:myorg/infra-manifests.git")
    cluster = ClusterState()
    reconciler = GitOpsReconciler(repo, cluster, app_name="order-api")

    # --- Commit 1: Initial deployment ---
    print("\n[Step 1] Initial commit — deploy 3 resources")
    v1_resources = [
        Resource("Deployment", "order-api", spec={
            "replicas": 3, "image": "order-api:1.0.0",
            "resources": {"cpu": "500m", "memory": "512Mi"},
        }, labels={"app": "order-api", "version": "1.0.0"}),
        Resource("Service", "order-api", spec={
            "type": "ClusterIP", "port": 8080, "targetPort": 8080,
        }),
        Resource("ConfigMap", "order-api-config", spec={
            "LOG_LEVEL": "info", "DB_HOST": "postgres.default.svc",
        }),
    ]
    commit1 = repo.add_commit("feat: initial deployment of order-api v1.0.0",
                               v1_resources)
    result = reconciler.reconcile()
    print_sync_result(result)

    # --- Reconcile with no changes ---
    print("\n[Step 2] Reconcile (no changes)")
    result = reconciler.reconcile()
    print_sync_result(result)

    # --- Simulate drift ---
    print("\n[Step 3] Simulate cluster drift...")
    drift = cluster.simulate_drift()
    if drift:
        for d in drift:
            print(f"    ! Drift: {d}")
    else:
        # Force a drift for demo purposes
        svc = cluster.get("default/Deployment/order-api")
        if svc:
            svc.spec["replicas"] = 1
            print("    ! Drift: default/Deployment/order-api: replicas 3 -> 1")

    print("\n[Step 4] Reconcile (correct drift)")
    result = reconciler.reconcile()
    print_sync_result(result)

    # --- Commit 2: Version update ---
    print("\n[Step 5] New commit — update to v1.1.0 with 5 replicas")
    v2_resources = [
        Resource("Deployment", "order-api", spec={
            "replicas": 5, "image": "order-api:1.1.0",
            "resources": {"cpu": "500m", "memory": "512Mi"},
        }, labels={"app": "order-api", "version": "1.1.0"}),
        Resource("Service", "order-api", spec={
            "type": "ClusterIP", "port": 8080, "targetPort": 8080,
        }),
        Resource("ConfigMap", "order-api-config", spec={
            "LOG_LEVEL": "debug", "DB_HOST": "postgres.default.svc",
            "FEATURE_FLAGS": "new-checkout=true",
        }),
        Resource("HorizontalPodAutoscaler", "order-api-hpa", spec={
            "minReplicas": 5, "maxReplicas": 20,
            "targetCPU": 70,
        }),
    ]
    commit2 = repo.add_commit("feat: scale to 5 replicas, add HPA, enable new-checkout",
                               v2_resources)
    result = reconciler.reconcile()
    print_sync_result(result)

    # --- Commit 3: Remove HPA (prune) ---
    print("\n[Step 6] New commit — remove HPA, scale down to 3")
    v3_resources = [
        Resource("Deployment", "order-api", spec={
            "replicas": 3, "image": "order-api:1.1.0",
            "resources": {"cpu": "500m", "memory": "512Mi"},
        }),
        Resource("Service", "order-api", spec={
            "type": "ClusterIP", "port": 8080, "targetPort": 8080,
        }),
        Resource("ConfigMap", "order-api-config", spec={
            "LOG_LEVEL": "info", "DB_HOST": "postgres.default.svc",
        }),
    ]
    commit3 = repo.add_commit("chore: remove HPA, scale down, disable feature flag",
                               v3_resources)
    result = reconciler.reconcile()
    print_sync_result(result)

    # --- Final status ---
    print(f"\n{'=' * 70}")
    print("Final State")
    print("=" * 70)
    print(f"  Git commits:       {repo.commit_count}")
    print(f"  Cluster resources: {len(cluster.list_resources())}")
    print(f"  Sync status:       {reconciler.get_sync_status().value}")
    print(f"  Reconcile cycles:  {reconciler._cycle_count}")
    print(f"\n  Cluster resources:")
    for r in cluster.list_resources():
        print(f"    {r.key}: spec_hash={r.spec_hash()}")
