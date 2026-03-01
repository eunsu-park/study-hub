"""
Multi-Region Deployment and Availability Simulation

Demonstrates cloud region and availability zone concepts:
- Latency calculation between regions using geographic distance
- Availability zone failover with health monitoring
- Data replication strategies (synchronous vs asynchronous)
- Regional availability and composite SLA computation

No cloud account required -- all behavior is simulated locally.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class AvailabilityZone:
    """An AZ is an isolated data center within a region.
    Cloud providers use multiple AZs per region so that a single
    facility failure does not take down the entire region."""
    name: str
    healthy: bool = True
    # Each AZ has independent power, cooling, and networking
    failure_probability: float = 0.001  # 0.1% chance of failure per check


@dataclass
class Region:
    """A geographic region containing multiple availability zones.
    Regions are fully independent -- a failure in us-east-1 does not
    affect eu-west-1. This is the basis of disaster recovery planning."""
    name: str
    latitude: float
    longitude: float
    zones: List[AvailabilityZone] = field(default_factory=list)

    @property
    def available_zones(self) -> List[AvailabilityZone]:
        return [z for z in self.zones if z.healthy]

    @property
    def is_available(self) -> bool:
        """Region is available if at least one AZ is healthy."""
        return len(self.available_zones) > 0


# Simulated cloud regions with real-world approximate coordinates
REGIONS = {
    "us-east-1":      Region("us-east-1 (Virginia)", 39.0, -77.5),
    "us-west-2":      Region("us-west-2 (Oregon)", 45.6, -122.7),
    "eu-west-1":      Region("eu-west-1 (Ireland)", 53.3, -6.3),
    "eu-central-1":   Region("eu-central-1 (Frankfurt)", 50.1, 8.7),
    "ap-northeast-1": Region("ap-northeast-1 (Tokyo)", 35.7, 139.7),
    "ap-southeast-1": Region("ap-southeast-1 (Singapore)", 1.3, 103.8),
}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two points.
    Used to estimate network latency -- light in fiber travels at roughly
    2/3 the speed of light, plus routing overhead adds ~30-50%."""
    R = 6371  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def estimate_latency_ms(region_a: Region, region_b: Region) -> float:
    """Estimate round-trip latency between two regions.
    Rule of thumb: ~5ms per 1000km in fiber, plus ~10ms for routing/processing.
    This is why region selection matters for user experience."""
    distance_km = _haversine_km(
        region_a.latitude, region_a.longitude,
        region_b.latitude, region_b.longitude,
    )
    # Light in fiber: ~200,000 km/s, round-trip doubles it, routing adds overhead
    fiber_latency = (distance_km / 200_000) * 1000 * 2  # ms, round-trip
    routing_overhead = 10 + random.uniform(0, 5)  # ms
    return round(fiber_latency + routing_overhead, 1)


def initialize_zones():
    """Create 3 AZs per region -- matching AWS's typical setup.
    The 3-AZ design allows majority quorum for distributed systems
    even if one AZ fails (2 out of 3 remain)."""
    for key, region in REGIONS.items():
        region.zones = [
            AvailabilityZone(f"{key}-{suffix}")
            for suffix in ["a", "b", "c"]
        ]


def show_latency_matrix():
    """Display inter-region latency estimates."""
    print("=" * 75)
    print("Inter-Region Latency Matrix (estimated round-trip ms)")
    print("=" * 75)
    keys = list(REGIONS.keys())
    header = f"{'':>18}" + "".join(f"{k:>18}" for k in keys)
    print(header)
    for k1 in keys:
        row = f"{k1:>18}"
        for k2 in keys:
            if k1 == k2:
                row += f"{'< 1':>18}"
            else:
                latency = estimate_latency_ms(REGIONS[k1], REGIONS[k2])
                row += f"{latency:>17.1f}"
        print(row)
    print()


class ReplicationStrategy:
    """Simulate data replication across regions.

    Synchronous replication: write completes only after ALL replicas confirm.
      - Guarantees strong consistency but adds latency (write = max replica RTT).
    Asynchronous replication: write completes after PRIMARY confirms; replicas
      catch up eventually.
      - Lower latency but risks data loss if primary fails before replication.
    """

    def __init__(self, primary: str, replicas: List[str], sync: bool = False):
        self.primary = primary
        self.replicas = replicas
        self.sync = sync

    def simulate_write(self, data_size_kb: float = 1.0) -> dict:
        primary_region = REGIONS[self.primary]
        # Primary write: fast local operation (~2-5ms)
        primary_latency = random.uniform(2, 5)

        replica_latencies = []
        for r in self.replicas:
            rtt = estimate_latency_ms(primary_region, REGIONS[r])
            # Replication adds overhead on top of network latency
            replica_latencies.append(rtt + random.uniform(1, 3))

        if self.sync:
            # Synchronous: total latency = primary + slowest replica
            total_latency = primary_latency + max(replica_latencies)
            consistency = "Strong"
        else:
            # Asynchronous: total latency = primary only; replicas catch up later
            total_latency = primary_latency
            consistency = "Eventual"

        return {
            "strategy": "Synchronous" if self.sync else "Asynchronous",
            "write_latency_ms": round(total_latency, 1),
            "replica_lag_ms": [round(l, 1) for l in replica_latencies],
            "consistency": consistency,
            "data_loss_risk": "None" if self.sync else "Possible (async window)",
        }


def simulate_failover():
    """Simulate AZ failures and demonstrate failover behavior."""
    print("=" * 75)
    print("Availability Zone Failover Simulation")
    print("=" * 75)

    region = REGIONS["us-east-1"]
    print(f"\n  Region: {region.name}")
    print(f"  Zones: {[z.name for z in region.zones]}")

    # Simulate random AZ failures over 5 rounds
    for round_num in range(1, 6):
        for zone in region.zones:
            if random.random() < zone.failure_probability * 100:  # amplified for demo
                zone.healthy = False
                print(f"  [Round {round_num}] FAILURE: {zone.name} went down!")
            elif not zone.healthy and random.random() > 0.5:
                zone.healthy = True
                print(f"  [Round {round_num}] RECOVERY: {zone.name} restored")

        healthy = region.available_zones
        status = "AVAILABLE" if region.is_available else "DOWN"
        print(f"  [Round {round_num}] Region status: {status} "
              f"({len(healthy)}/{len(region.zones)} AZs healthy)")

    # Composite SLA calculation
    # Individual AZ SLA ~99.99% => multi-AZ SLA = 1 - (1-0.9999)^3
    az_sla = 0.9999
    multi_az_sla = 1 - (1 - az_sla) ** 3
    print(f"\n  Single AZ SLA: {az_sla * 100:.2f}%")
    print(f"  Multi-AZ SLA (3 AZs): {multi_az_sla * 100:.6f}%")
    print(f"  Annual downtime (single AZ): {(1 - az_sla) * 525960:.1f} minutes")
    print(f"  Annual downtime (multi-AZ):  {(1 - multi_az_sla) * 525960:.4f} minutes")
    print()


def compare_replication():
    """Compare sync vs async replication strategies."""
    print("=" * 75)
    print("Replication Strategy Comparison (us-east-1 -> eu-west-1, ap-northeast-1)")
    print("=" * 75)

    replicas = ["eu-west-1", "ap-northeast-1"]
    for sync in [True, False]:
        strategy = ReplicationStrategy("us-east-1", replicas, sync=sync)
        result = strategy.simulate_write()
        print(f"\n  {result['strategy']} Replication:")
        print(f"    Write latency:   {result['write_latency_ms']} ms")
        print(f"    Replica lag:     {result['replica_lag_ms']} ms")
        print(f"    Consistency:     {result['consistency']}")
        print(f"    Data loss risk:  {result['data_loss_risk']}")
    print()


if __name__ == "__main__":
    random.seed(42)
    initialize_zones()
    show_latency_matrix()
    compare_replication()
    simulate_failover()
