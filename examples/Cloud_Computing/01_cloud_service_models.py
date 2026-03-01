"""
Cloud Service Models Simulation: IaaS vs PaaS vs SaaS

Demonstrates the fundamental cloud service models by simulating:
- Resource allocation and management responsibilities
- Cost modeling (CapEx vs OpEx) for each service tier
- Responsibility matrix (what you manage vs what the provider manages)
- Horizontal and vertical scaling behavior differences

No cloud account required -- all behavior is simulated locally.
"""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class ServiceModel(Enum):
    IAAS = "IaaS"  # Infrastructure as a Service (e.g., EC2, GCE)
    PAAS = "PaaS"  # Platform as a Service (e.g., Elastic Beanstalk, App Engine)
    SAAS = "SaaS"  # Software as a Service (e.g., Gmail, Salesforce)


# Responsibility matrix: True means the CUSTOMER manages it, False means PROVIDER manages it.
# This is a core concept -- understanding the "shared responsibility model" is essential
# for cloud architecture decisions and security planning.
RESPONSIBILITY_MATRIX: Dict[str, Dict[ServiceModel, bool]] = {
    "Physical Hardware":  {ServiceModel.IAAS: False, ServiceModel.PAAS: False, ServiceModel.SAAS: False},
    "Networking":         {ServiceModel.IAAS: False, ServiceModel.PAAS: False, ServiceModel.SAAS: False},
    "Virtualization":     {ServiceModel.IAAS: False, ServiceModel.PAAS: False, ServiceModel.SAAS: False},
    "Operating System":   {ServiceModel.IAAS: True,  ServiceModel.PAAS: False, ServiceModel.SAAS: False},
    "Middleware/Runtime":  {ServiceModel.IAAS: True,  ServiceModel.PAAS: False, ServiceModel.SAAS: False},
    "Application Code":   {ServiceModel.IAAS: True,  ServiceModel.PAAS: True,  ServiceModel.SAAS: False},
    "Data":               {ServiceModel.IAAS: True,  ServiceModel.PAAS: True,  ServiceModel.SAAS: True},
}


@dataclass
class ResourceAllocation:
    """Simulates how resources are allocated differently per service model."""
    vcpus: int
    memory_gb: float
    storage_gb: float
    # IaaS gives raw resources; PaaS abstracts them; SaaS hides them entirely
    model: ServiceModel

    @property
    def hourly_cost(self) -> float:
        """Cost varies by model: IaaS is cheapest per-resource but requires more ops work.
        PaaS adds a platform premium. SaaS charges per-user, not per-resource."""
        base = self.vcpus * 0.05 + self.memory_gb * 0.01 + self.storage_gb * 0.0002
        multipliers = {ServiceModel.IAAS: 1.0, ServiceModel.PAAS: 1.4, ServiceModel.SAAS: 2.0}
        return round(base * multipliers[self.model], 4)


@dataclass
class ScalingSimulation:
    """Simulates scaling behavior under increasing load for each model."""
    model: ServiceModel
    base_capacity: int = 100  # requests per second
    instances: int = 1
    max_instances: int = 10
    scale_history: List[int] = field(default_factory=list)

    def handle_load(self, rps: int) -> dict:
        """Simulate how each model responds to a given request-per-second load.

        Key insight: IaaS scaling is manual/slow (you configure auto-scaling rules),
        PaaS scaling is semi-automatic (platform handles it with some config),
        SaaS scaling is fully transparent (provider handles everything).
        """
        capacity = self.base_capacity * self.instances
        # Scaling reaction time differs by model -- IaaS is slowest because
        # you must provision actual VMs; SaaS is instant (provider pre-provisions)
        scale_delay = {ServiceModel.IAAS: 3, ServiceModel.PAAS: 1, ServiceModel.SAAS: 0}

        if rps > capacity and self.instances < self.max_instances:
            needed = min((rps // self.base_capacity) + 1, self.max_instances)
            # Simulate gradual scaling with delay
            step = max(1, (needed - self.instances) // max(1, scale_delay[self.model]))
            self.instances = min(self.instances + step, self.max_instances)

        capacity = self.base_capacity * self.instances
        dropped = max(0, rps - capacity)
        self.scale_history.append(self.instances)

        return {
            "model": self.model.value,
            "rps": rps,
            "instances": self.instances,
            "capacity": capacity,
            "dropped_requests": dropped,
            "success_rate": round((rps - dropped) / rps * 100, 1) if rps > 0 else 100.0,
        }


def print_responsibility_matrix():
    """Display the shared responsibility model as a formatted table."""
    print("=" * 68)
    print("Shared Responsibility Matrix (Customer=YOU, Provider=CLOUD)")
    print("=" * 68)
    header = f"{'Layer':<22} {'IaaS':<10} {'PaaS':<10} {'SaaS':<10}"
    print(header)
    print("-" * 68)
    for layer, models in RESPONSIBILITY_MATRIX.items():
        row = f"{layer:<22}"
        for model in ServiceModel:
            owner = "YOU" if models[model] else "Provider"
            row += f" {owner:<10}"
        print(row)
    print()


def compare_costs():
    """Compare monthly costs across service models for equivalent workloads."""
    print("=" * 68)
    print("Monthly Cost Comparison (equivalent workload)")
    print("=" * 68)

    # Same logical workload, different resource profiles per model
    configs = {
        ServiceModel.IAAS: ResourceAllocation(4, 16.0, 100.0, ServiceModel.IAAS),
        ServiceModel.PAAS: ResourceAllocation(4, 16.0, 100.0, ServiceModel.PAAS),
        ServiceModel.SAAS: ResourceAllocation(4, 16.0, 100.0, ServiceModel.SAAS),
    }

    for model, alloc in configs.items():
        monthly = alloc.hourly_cost * 730  # ~730 hours per month
        # IaaS has additional hidden costs: ops team, patching, monitoring setup
        ops_overhead = {ServiceModel.IAAS: 200, ServiceModel.PAAS: 50, ServiceModel.SAAS: 0}
        total = monthly + ops_overhead[model]
        print(f"  {model.value}: ${monthly:>8.2f} compute + ${ops_overhead[model]:>5} ops = ${total:>8.2f}/mo")
    print()


def simulate_scaling():
    """Run a scaling simulation showing how each model handles a traffic spike."""
    print("=" * 68)
    print("Scaling Simulation: Traffic Spike (100 -> 800 -> 200 RPS)")
    print("=" * 68)

    # Simulate a traffic pattern: ramp up, spike, cool down
    traffic_pattern = [100, 200, 400, 600, 800, 800, 600, 400, 200, 100]

    simulators = {m: ScalingSimulation(model=m) for m in ServiceModel}

    for rps in traffic_pattern:
        print(f"\n  Load: {rps} RPS")
        for model, sim in simulators.items():
            result = sim.handle_load(rps)
            print(f"    {result['model']}: {result['instances']} instances, "
                  f"capacity={result['capacity']} RPS, "
                  f"success={result['success_rate']}%")


if __name__ == "__main__":
    random.seed(42)
    print_responsibility_matrix()
    compare_costs()
    simulate_scaling()
