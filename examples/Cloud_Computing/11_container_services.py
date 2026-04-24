"""
Container Services — Managed Kubernetes vs. Task-based Services

Simulates the decision surface between the three main categories of
managed container services that AWS, GCP, and Azure all offer in some form:

1. Fully-managed Kubernetes (EKS / GKE / AKS): you get raw K8s with the
   control plane run for you. You still manage node pools and pods.
2. Task/Service orchestrators (ECS / Cloud Run for Anthos / Container
   Apps): provider-specific, simpler API than K8s, good for teams that do
   not need K8s flexibility.
3. Serverless containers (Fargate / Cloud Run / Container Apps):
   scale-to-zero, no node management, pay per request or per second.

The script is offline — no cloud calls. It compares the options on cost,
operational burden, and cold-start latency for a realistic workload.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


# =============================================================================
# 1. Workload profile — the input every decision depends on
# =============================================================================

@dataclass
class Workload:
    """A representative workload for comparing container platforms."""
    name: str
    requests_per_day: int
    avg_duration_ms: int          # per request
    memory_mb: int                # per container instance
    steady_traffic_pct: float     # fraction of requests during peak hours
    burst_ratio: float            # ratio of peak RPS to average RPS


# =============================================================================
# 2. Option — a specific service choice with operational characteristics
# =============================================================================

@dataclass
class ContainerOption:
    name: str
    managed_k8s: bool             # True = you see K8s API
    scale_to_zero: bool
    cold_start_ms: int
    monthly_control_plane_usd: float   # $72 EKS, $72 GKE standard, $0 ECS, etc.
    per_vcpu_hour_usd: float
    per_gb_hour_usd: float
    ops_complexity: int           # 1 (simplest) .. 5 (most)

    def monthly_cost(self, workload: Workload) -> float:
        """
        Very rough cost model:
          - Scale-to-zero services are charged only for actual request-seconds.
          - Always-on services pay for provisioned capacity 24/7.
        """
        req_per_month = workload.requests_per_day * 30
        compute_sec = req_per_month * workload.avg_duration_ms / 1000.0
        compute_hr = compute_sec / 3600.0

        # Assume 0.5 vCPU per instance for this workload
        vcpu_hr = compute_hr * 0.5
        mem_hr = compute_hr * (workload.memory_mb / 1024.0)

        if self.scale_to_zero:
            usage = vcpu_hr * self.per_vcpu_hour_usd + mem_hr * self.per_gb_hour_usd
        else:
            # Always-on: assume you provision to handle peak, 24h/day
            peak_rps = (workload.requests_per_day / 86400) * workload.burst_ratio
            instances = max(1, int(peak_rps * workload.avg_duration_ms / 1000.0 + 0.99))
            # 0.5 vCPU + memory per always-on instance, 24h x 30 days
            hours = 24 * 30
            usage = instances * hours * (
                0.5 * self.per_vcpu_hour_usd + (workload.memory_mb / 1024.0) * self.per_gb_hour_usd
            )

        return self.monthly_control_plane_usd + usage


# =============================================================================
# 3. Comparison table
# =============================================================================

def build_options() -> List[ContainerOption]:
    # Numbers are representative, not exact — intended for teaching.
    return [
        ContainerOption(
            name="EKS + EC2 node pool",
            managed_k8s=True, scale_to_zero=False,
            cold_start_ms=0,
            monthly_control_plane_usd=72.0,
            per_vcpu_hour_usd=0.0416,
            per_gb_hour_usd=0.00456,
            ops_complexity=5,
        ),
        ContainerOption(
            name="ECS on Fargate",
            managed_k8s=False, scale_to_zero=False,
            cold_start_ms=0,
            monthly_control_plane_usd=0.0,
            per_vcpu_hour_usd=0.04048,
            per_gb_hour_usd=0.004445,
            ops_complexity=3,
        ),
        ContainerOption(
            name="Cloud Run",
            managed_k8s=False, scale_to_zero=True,
            cold_start_ms=400,
            monthly_control_plane_usd=0.0,
            per_vcpu_hour_usd=0.0864,            # higher per-hour — offset by only paying for actual requests
            per_gb_hour_usd=0.009,
            ops_complexity=1,
        ),
    ]


def compare(workload: Workload) -> None:
    options = build_options()

    print("=" * 78)
    print(f"Workload: {workload.name}  ({workload.requests_per_day:,} req/day, "
          f"{workload.avg_duration_ms} ms avg)")
    print("=" * 78)
    print(f"{'Service':<28} {'Monthly':<15} {'Cold start':<14} {'Ops':<6} {'K8s API':<8}")
    print(f"{'-'*27:<28} {'-'*14:<15} {'-'*13:<14} {'-'*5:<6} {'-'*7:<8}")
    for opt in options:
        print(
            f"{opt.name:<28} "
            f"${opt.monthly_cost(workload):>10.2f}   "
            f"{opt.cold_start_ms} ms{'':<9}  "
            f"{opt.ops_complexity}/5   "
            f"{'yes' if opt.managed_k8s else 'no'}"
        )


def discuss() -> None:
    print()
    print("Decision heuristics:")
    print("  * Low, spiky traffic (dev/test, internal tools) → Cloud Run / Container Apps.")
    print("    Scale-to-zero dominates cost at low utilization.")
    print("  * Steady production traffic, simple requirements → ECS / Fargate.")
    print("    Avoids the EKS control-plane fee and K8s operational overhead.")
    print("  * Complex multi-tenant or custom K8s CRDs needed → EKS / GKE / AKS.")
    print("    Pay the control-plane fee and the ops cost in exchange for API flexibility.")
    print()
    print("Cold start note: a 400 ms cold start matters for user-facing APIs but is")
    print("invisible for async workers processing a queue.")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    workloads = [
        Workload("Internal tool (low traffic)", 10_000, 200, 512, 0.8, 3.0),
        Workload("Public API (steady)",       2_000_000, 120, 1024, 0.6, 1.8),
        Workload("Batch worker",                500_000, 800,  512, 0.5, 1.2),
    ]
    for w in workloads:
        compare(w)
        print()
    discuss()


if __name__ == "__main__":
    main()
