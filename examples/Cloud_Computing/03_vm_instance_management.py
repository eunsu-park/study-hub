"""
Virtual Machine Instance Management Simulation

Demonstrates core VM/compute concepts:
- VM lifecycle states (pending, running, stopping, stopped, terminated)
- Instance type comparison (CPU, memory, cost trade-offs)
- Auto-scaling based on CPU utilization metrics
- Spot vs on-demand vs reserved pricing models

No cloud account required -- all behavior is simulated locally.
"""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class VMState(Enum):
    PENDING = "pending"        # Provisioning hardware
    RUNNING = "running"        # Active and serving traffic
    STOPPING = "stopping"      # Graceful shutdown in progress
    STOPPED = "stopped"        # Halted but preserving disk/state
    TERMINATED = "terminated"  # Fully destroyed, no recovery


@dataclass
class InstanceType:
    """Models a cloud instance type (like AWS t3.medium or GCP e2-standard-4).
    Instance families are designed for different workloads:
      - General purpose (t/m): balanced CPU/memory ratio
      - Compute optimized (c): high CPU, less memory -- batch processing, HPC
      - Memory optimized (r/x): high memory -- databases, in-memory caching
      - GPU (p/g): accelerated computing -- ML training, rendering
    """
    name: str
    family: str
    vcpus: int
    memory_gb: float
    hourly_on_demand: float   # On-demand: pay by the hour, no commitment
    hourly_reserved: float    # Reserved: 1-3 year commitment, ~40-60% discount
    hourly_spot: float        # Spot: use spare capacity, ~60-90% discount, can be interrupted


# Simulated instance catalog -- prices approximate real AWS pricing
INSTANCE_CATALOG: Dict[str, InstanceType] = {
    "t3.micro":     InstanceType("t3.micro",     "General",  2,   1.0,  0.0104, 0.0065, 0.0031),
    "t3.medium":    InstanceType("t3.medium",    "General",  2,   4.0,  0.0416, 0.0260, 0.0125),
    "m5.large":     InstanceType("m5.large",     "General",  2,   8.0,  0.0960, 0.0590, 0.0288),
    "c5.xlarge":    InstanceType("c5.xlarge",    "Compute",  4,   8.0,  0.1700, 0.1060, 0.0510),
    "r5.large":     InstanceType("r5.large",     "Memory",   2,  16.0,  0.1260, 0.0790, 0.0378),
    "p3.2xlarge":   InstanceType("p3.2xlarge",   "GPU",      8,  61.0,  3.0600, 1.8400, 0.9180),
}


@dataclass
class VMInstance:
    """A single virtual machine instance with lifecycle management."""
    instance_id: str
    instance_type: InstanceType
    state: VMState = VMState.PENDING
    cpu_utilization: float = 0.0  # 0-100%

    def launch(self) -> str:
        """Transition from PENDING to RUNNING. In real clouds, this takes
        30-90 seconds as the hypervisor allocates resources and boots the OS."""
        if self.state == VMState.PENDING:
            self.state = VMState.RUNNING
            return f"  [{self.instance_id}] Launched ({self.instance_type.name})"
        return f"  [{self.instance_id}] Cannot launch from state {self.state.value}"

    def stop(self) -> str:
        """Stop preserves the EBS volume (disk) but releases the compute.
        You stop paying for compute but still pay for storage."""
        if self.state == VMState.RUNNING:
            self.state = VMState.STOPPED
            self.cpu_utilization = 0.0
            return f"  [{self.instance_id}] Stopped (disk preserved, compute released)"
        return f"  [{self.instance_id}] Cannot stop from state {self.state.value}"

    def terminate(self) -> str:
        """Terminate destroys everything -- this is irreversible.
        Always use termination protection in production to prevent accidents."""
        if self.state in (VMState.RUNNING, VMState.STOPPED):
            self.state = VMState.TERMINATED
            self.cpu_utilization = 0.0
            return f"  [{self.instance_id}] TERMINATED (all resources released)"
        return f"  [{self.instance_id}] Cannot terminate from state {self.state.value}"

    def simulate_load(self) -> None:
        """Simulate fluctuating CPU utilization for auto-scaling demos."""
        if self.state == VMState.RUNNING:
            self.cpu_utilization = min(100.0, max(0.0,
                self.cpu_utilization + random.uniform(-15, 20)))


@dataclass
class AutoScalingGroup:
    """Simulates an auto-scaling group that adjusts instance count based on
    CPU utilization. The key parameters are:
      - target_cpu: desired average CPU (e.g., 60%)
      - min/max_instances: guardrails to prevent under/over-provisioning
      - cooldown: seconds between scaling actions to prevent thrashing
    """
    name: str
    instance_type_name: str
    min_instances: int = 1
    max_instances: int = 10
    desired: int = 2
    target_cpu: float = 60.0    # Scale out above this, scale in below half of this
    cooldown_ticks: int = 2     # Wait N ticks between scaling actions
    instances: List[VMInstance] = field(default_factory=list)
    _cooldown_remaining: int = 0
    _next_id: int = 1

    def _create_instance(self) -> VMInstance:
        inst_type = INSTANCE_CATALOG[self.instance_type_name]
        vm = VMInstance(f"i-{self._next_id:04d}", inst_type, VMState.PENDING)
        self._next_id += 1
        vm.launch()
        vm.cpu_utilization = random.uniform(20, 40)
        return vm

    def initialize(self) -> None:
        """Bring the group to its desired instance count."""
        for _ in range(self.desired):
            self.instances.append(self._create_instance())

    def evaluate_scaling(self) -> Optional[str]:
        """Core auto-scaling logic: compare average CPU to target.
        Scale-out is aggressive (react quickly to demand spikes).
        Scale-in is conservative (avoid killing instances too early)."""
        running = [i for i in self.instances if i.state == VMState.RUNNING]
        if not running:
            return None

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return "  [ASG] In cooldown, no scaling action"

        avg_cpu = sum(i.cpu_utilization for i in running) / len(running)

        # Scale OUT: average CPU exceeds target
        if avg_cpu > self.target_cpu and len(running) < self.max_instances:
            new_vm = self._create_instance()
            self.instances.append(new_vm)
            self._cooldown_remaining = self.cooldown_ticks
            return (f"  [ASG] SCALE OUT: avg CPU={avg_cpu:.1f}% > {self.target_cpu}% "
                    f"-> {len(running)+1} instances")

        # Scale IN: average CPU well below target (use half of target as threshold)
        if avg_cpu < self.target_cpu * 0.4 and len(running) > self.min_instances:
            victim = running[-1]
            victim.terminate()
            self._cooldown_remaining = self.cooldown_ticks
            return (f"  [ASG] SCALE IN: avg CPU={avg_cpu:.1f}% < {self.target_cpu * 0.4:.0f}% "
                    f"-> {len(running)-1} instances")

        return f"  [ASG] Steady: avg CPU={avg_cpu:.1f}%, {len(running)} instances"


def compare_instance_types():
    """Show instance type catalog with cost comparison across pricing models."""
    print("=" * 90)
    print("Instance Type Catalog and Pricing Comparison")
    print("=" * 90)
    print(f"{'Type':<14} {'Family':<10} {'vCPUs':>5} {'RAM(GB)':>8} "
          f"{'On-Demand':>10} {'Reserved':>10} {'Spot':>10} {'Savings':>10}")
    print("-" * 90)

    for itype in INSTANCE_CATALOG.values():
        # Show how much you save with reserved vs on-demand
        savings_pct = (1 - itype.hourly_reserved / itype.hourly_on_demand) * 100
        print(f"{itype.name:<14} {itype.family:<10} {itype.vcpus:>5} {itype.memory_gb:>8.1f} "
              f"${itype.hourly_on_demand:>9.4f} ${itype.hourly_reserved:>9.4f} "
              f"${itype.hourly_spot:>9.4f} {savings_pct:>9.0f}%")

    # Monthly cost for a common workload
    print(f"\n  Monthly cost for m5.large (730 hours):")
    m5 = INSTANCE_CATALOG["m5.large"]
    print(f"    On-Demand: ${m5.hourly_on_demand * 730:>8.2f}")
    print(f"    Reserved:  ${m5.hourly_reserved * 730:>8.2f}")
    print(f"    Spot:      ${m5.hourly_spot * 730:>8.2f}")
    print()


def demonstrate_lifecycle():
    """Walk through the VM lifecycle states."""
    print("=" * 90)
    print("VM Lifecycle Demonstration")
    print("=" * 90)
    vm = VMInstance("i-demo", INSTANCE_CATALOG["t3.medium"])
    print(f"  Initial state: {vm.state.value}")
    for action in [vm.launch, vm.stop, vm.launch, vm.stop, vm.terminate]:
        # Stopped VMs must go through PENDING again to restart -- in real clouds
        # this is implicit, but it reflects re-provisioning on potentially different hardware
        if action == vm.launch and vm.state == VMState.STOPPED:
            vm.state = VMState.PENDING
        print(action())
    print()


def simulate_autoscaling():
    """Run an auto-scaling simulation with synthetic load."""
    print("=" * 90)
    print("Auto-Scaling Group Simulation (target CPU: 60%)")
    print("=" * 90)

    asg = AutoScalingGroup("web-asg", "t3.medium", min_instances=2, max_instances=6)
    asg.initialize()

    # Simulate a load pattern: gradual increase, spike, then decrease
    load_multipliers = [1.0, 1.5, 2.5, 3.5, 4.0, 3.0, 2.0, 1.0, 0.5, 0.3]

    for tick, mult in enumerate(load_multipliers):
        print(f"\n  --- Tick {tick + 1} (load multiplier: {mult}x) ---")
        running = [i for i in asg.instances if i.state == VMState.RUNNING]
        for inst in running:
            # Inject load proportional to multiplier
            inst.cpu_utilization = min(100, 20 * mult + random.uniform(-10, 10))
        result = asg.evaluate_scaling()
        if result:
            print(result)


if __name__ == "__main__":
    random.seed(42)
    compare_instance_types()
    demonstrate_lifecycle()
    simulate_autoscaling()
