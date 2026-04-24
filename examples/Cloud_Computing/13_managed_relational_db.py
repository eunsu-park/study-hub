"""
Managed Relational Database — RDS / Cloud SQL Concept Simulation

No cloud account required. Models the three levers every managed SQL
offering exposes:

1. Instance sizing (CPU + memory + IOPS)
2. Replication topology (primary, standby, read replicas)
3. Backup and retention (automated snapshots + point-in-time recovery)

The script simulates failover, read-replica routing, and computes a
realistic monthly bill. Provides a concrete basis for discussing when
managed SQL makes sense and when it does not.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional


# =============================================================================
# 1. Instance class
# =============================================================================

@dataclass
class InstanceClass:
    """Teaching approximation of an AWS RDS / GCP Cloud SQL instance tier."""
    name: str
    vcpu: int
    memory_gb: int
    hourly_usd: float         # compute-hour price (on-demand)


INSTANCE_CATALOG = [
    InstanceClass("db.t3.medium",  2,   4, 0.068),
    InstanceClass("db.r6g.large",  2,  16, 0.24),
    InstanceClass("db.r6g.xlarge", 4,  32, 0.48),
    InstanceClass("db.r6g.2xlarge", 8, 64, 0.96),
]


# =============================================================================
# 2. Cluster topology
# =============================================================================

@dataclass
class Replica:
    region: str
    lag_ms: float = 0.0          # populated by simulate_replication

    def __str__(self) -> str:
        return f"replica@{self.region} (lag {self.lag_ms:.1f} ms)"


@dataclass
class DatabaseCluster:
    instance_class: InstanceClass
    storage_gb: int
    multi_az_standby: bool
    read_replicas: List[Replica] = field(default_factory=list)
    backup_retention_days: int = 7

    # --- cost ---
    def monthly_compute_usd(self) -> float:
        hours = 24 * 30
        factor = 2 if self.multi_az_standby else 1  # hot standby doubles the bill
        replica_factor = len(self.read_replicas)
        return hours * self.instance_class.hourly_usd * (factor + replica_factor)

    def monthly_storage_usd(self, gb_month_usd: float = 0.115) -> float:
        # Storage is billed per GB. Backup storage beyond retention bills extra.
        base = self.storage_gb * gb_month_usd
        backup_ratio = 1.2   # small allowance for delta-based backups
        return base + self.storage_gb * gb_month_usd * backup_ratio * self.backup_retention_days / 30

    def monthly_cost(self) -> float:
        return self.monthly_compute_usd() + self.monthly_storage_usd()


# =============================================================================
# 3. Simulations
# =============================================================================

def simulate_failover(cluster: DatabaseCluster) -> float:
    """Return estimated downtime seconds for a primary failure."""
    if cluster.multi_az_standby:
        # Synchronous standby → ~30–90 seconds of downtime (DNS + rehydrate caches)
        return random.uniform(30, 90)
    else:
        # No standby → restore from snapshot or promote a read replica
        # Restore from snapshot is slow (tens of minutes for non-trivial data)
        return random.uniform(900, 1800)


def simulate_replication(cluster: DatabaseCluster) -> None:
    """Fill in plausible async-replication lag for each read replica."""
    for r in cluster.read_replicas:
        # Cross-region replicas see higher lag
        base = 5.0 if r.region == cluster_primary_region(cluster) else 80.0
        r.lag_ms = base + random.uniform(0, base)


def cluster_primary_region(cluster: DatabaseCluster) -> str:
    # Stub: in a real system, the primary region is a property of the cluster.
    return "us-east-1"


def route_read_query(cluster: DatabaseCluster, staleness_tolerance_ms: float) -> str:
    """Decide where to send a read based on acceptable staleness."""
    if staleness_tolerance_ms == 0:
        return "primary (zero-staleness requirement)"
    eligible = [r for r in cluster.read_replicas if r.lag_ms <= staleness_tolerance_ms]
    if not eligible:
        return f"primary (no replica met {staleness_tolerance_ms} ms tolerance)"
    # Prefer the least-lagged eligible replica
    pick = min(eligible, key=lambda r: r.lag_ms)
    return f"replica@{pick.region} (lag {pick.lag_ms:.1f} ms)"


# =============================================================================
# 4. Demo
# =============================================================================

def demo() -> None:
    random.seed(7)
    cluster = DatabaseCluster(
        instance_class=INSTANCE_CATALOG[2],      # r6g.xlarge
        storage_gb=500,
        multi_az_standby=True,
        read_replicas=[Replica("us-east-1"), Replica("us-east-1"), Replica("eu-west-1")],
    )

    print("=== Cluster Configuration ===")
    print(f"  primary: {cluster.instance_class.name}  ({cluster.instance_class.vcpu} vCPU, "
          f"{cluster.instance_class.memory_gb} GiB)")
    print(f"  storage: {cluster.storage_gb} GiB, backup retention {cluster.backup_retention_days} days")
    print(f"  multi-AZ standby: {cluster.multi_az_standby}")
    print(f"  read replicas:")
    for r in cluster.read_replicas:
        print(f"    - {r}")

    print("\n=== Monthly Cost ===")
    print(f"  compute: ${cluster.monthly_compute_usd():>8.2f}")
    print(f"  storage: ${cluster.monthly_storage_usd():>8.2f}")
    print(f"  total:   ${cluster.monthly_cost():>8.2f}")

    simulate_replication(cluster)
    print("\n=== Replication Lag (simulated) ===")
    for r in cluster.read_replicas:
        print(f"  {r}")

    print("\n=== Read Routing Decisions ===")
    for tolerance in [0, 20, 150, 1000]:
        print(f"  staleness tolerance {tolerance:>4} ms → {route_read_query(cluster, tolerance)}")

    print("\n=== Failover Simulation ===")
    downtime = simulate_failover(cluster)
    print(f"  estimated downtime on primary failure: {downtime:.1f} s")
    print("  multi_az_standby=True keeps it under ~90 s; without standby it jumps to 15–30 min.")


if __name__ == "__main__":
    demo()
