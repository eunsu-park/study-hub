#!/usr/bin/env python3
"""Example: Deployment Strategies — Blue-Green & Canary Simulation

Simulates blue-green and canary deployment logic including traffic
shifting, health checking, rollback decisions, and promotion criteria.
Related lesson: 12_Deployment_Strategies.md
"""

# =============================================================================
# DEPLOYMENT STRATEGIES OVERVIEW
#   Rolling      — replace instances one at a time (default K8s)
#   Blue-Green   — two identical environments; swap traffic atomically
#   Canary       — route a small % of traffic to the new version first
#   A/B Testing  — route based on user attributes (feature flags)
# =============================================================================

import time
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


# =============================================================================
# 1. COMMON TYPES
# =============================================================================

class DeploymentState(Enum):
    IDLE = "idle"
    IN_PROGRESS = "in_progress"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    ROLLED_BACK = "rolled_back"


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ServiceInstance:
    """A running instance of a service version."""
    instance_id: str
    version: str
    healthy: bool = True
    request_count: int = 0
    error_count: int = 0

    @property
    def error_rate(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.error_count / self.request_count

    def handle_request(self, fail_rate: float = 0.01) -> bool:
        """Simulate handling a request. Returns True if successful."""
        self.request_count += 1
        if random.random() < fail_rate:
            self.error_count += 1
            return False
        return True


@dataclass
class HealthCheck:
    """Health check configuration and execution."""
    endpoint: str = "/health"
    interval_seconds: float = 5.0
    timeout_seconds: float = 2.0
    healthy_threshold: int = 3
    unhealthy_threshold: int = 2

    def check(self, instance: ServiceInstance) -> HealthStatus:
        """Simulate a health check against an instance."""
        if not instance.healthy:
            return HealthStatus.UNHEALTHY
        if instance.error_rate > 0.05:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY


# =============================================================================
# 2. BLUE-GREEN DEPLOYMENT
# =============================================================================

@dataclass
class BlueGreenDeployment:
    """Blue-green deployment controller.

    Maintains two environments (blue and green). Traffic is served by
    exactly one environment at a time. Deployment creates the new version
    in the idle environment, verifies it, then swaps the router.
    """
    service_name: str
    blue: list[ServiceInstance] = field(default_factory=list)
    green: list[ServiceInstance] = field(default_factory=list)
    active_color: str = "blue"
    state: DeploymentState = DeploymentState.IDLE
    health_check: HealthCheck = field(default_factory=HealthCheck)

    @property
    def active(self) -> list[ServiceInstance]:
        return self.blue if self.active_color == "blue" else self.green

    @property
    def inactive(self) -> list[ServiceInstance]:
        return self.green if self.active_color == "blue" else self.blue

    @property
    def inactive_color(self) -> str:
        return "green" if self.active_color == "blue" else "blue"

    def deploy(self, new_version: str, instance_count: int = 3,
               fail_rate: float = 0.01) -> bool:
        """Deploy a new version using blue-green strategy.

        Steps:
        1. Provision new instances in the INACTIVE environment
        2. Run health checks on the new instances
        3. Swap traffic to the new environment
        4. Keep the old environment as rollback target

        Returns True if deployment succeeded, False if rolled back.
        """
        self.state = DeploymentState.IN_PROGRESS
        inactive_color = self.inactive_color
        print(f"\n[Blue-Green] Deploying v{new_version} to {inactive_color}")

        # Step 1: Provision new instances
        new_instances = []
        for i in range(instance_count):
            inst = ServiceInstance(
                instance_id=f"{inactive_color}-{i}",
                version=new_version,
            )
            new_instances.append(inst)
            print(f"  Provisioned {inst.instance_id} (v{new_version})")

        # Replace inactive pool
        if self.active_color == "blue":
            self.green = new_instances
        else:
            self.blue = new_instances

        # Step 2: Health check new instances
        self.state = DeploymentState.VERIFYING
        print(f"  Running health checks on {inactive_color}...")

        # Simulate some traffic to new instances for health verification
        all_healthy = True
        for inst in new_instances:
            for _ in range(20):
                inst.handle_request(fail_rate)
            status = self.health_check.check(inst)
            print(f"    {inst.instance_id}: {status.value} "
                  f"(error_rate={inst.error_rate:.1%})")
            if status == HealthStatus.UNHEALTHY:
                all_healthy = False

        if not all_healthy:
            print(f"  Health checks FAILED -> rolling back")
            self.state = DeploymentState.ROLLED_BACK
            return False

        # Step 3: Swap traffic
        old_color = self.active_color
        self.active_color = inactive_color
        print(f"  Traffic swapped: {old_color} -> {inactive_color}")
        print(f"  Active: {inactive_color} (v{new_version})")
        print(f"  Standby: {old_color} (rollback target)")

        self.state = DeploymentState.COMPLETED
        return True

    def rollback(self) -> None:
        """Instant rollback by swapping back to the previous environment."""
        old_color = self.active_color
        self.active_color = self.inactive_color
        self.state = DeploymentState.ROLLED_BACK
        print(f"[Blue-Green] ROLLBACK: {old_color} -> {self.active_color}")


# =============================================================================
# 3. CANARY DEPLOYMENT
# =============================================================================

@dataclass
class CanaryStage:
    """A stage in a canary rollout."""
    traffic_percent: float
    duration_seconds: float
    max_error_rate: float = 0.05


@dataclass
class CanaryDeployment:
    """Canary deployment controller.

    Gradually shifts traffic from the stable version to the canary.
    Each stage has a traffic percentage, duration, and success criteria.
    If any stage fails the error rate threshold, the deployment rolls back.
    """
    service_name: str
    stable_version: str
    stable_instances: list[ServiceInstance] = field(default_factory=list)
    canary_instances: list[ServiceInstance] = field(default_factory=list)
    stages: list[CanaryStage] = field(default_factory=lambda: [
        CanaryStage(traffic_percent=5.0, duration_seconds=0.1, max_error_rate=0.05),
        CanaryStage(traffic_percent=25.0, duration_seconds=0.1, max_error_rate=0.05),
        CanaryStage(traffic_percent=50.0, duration_seconds=0.1, max_error_rate=0.05),
        CanaryStage(traffic_percent=100.0, duration_seconds=0.1, max_error_rate=0.05),
    ])
    state: DeploymentState = DeploymentState.IDLE
    current_stage: int = 0
    health_check: HealthCheck = field(default_factory=HealthCheck)

    def deploy(self, new_version: str, canary_count: int = 2,
               canary_fail_rate: float = 0.01) -> bool:
        """Deploy using canary strategy with progressive traffic shifting.

        Returns True if fully promoted, False if rolled back.
        """
        self.state = DeploymentState.IN_PROGRESS
        print(f"\n[Canary] Starting canary for v{new_version}")

        # Provision canary instances
        self.canary_instances = [
            ServiceInstance(
                instance_id=f"canary-{i}",
                version=new_version,
            )
            for i in range(canary_count)
        ]
        print(f"  Provisioned {canary_count} canary instances")

        # Progressive rollout through stages
        for i, stage in enumerate(self.stages):
            self.current_stage = i
            self.state = DeploymentState.VERIFYING
            print(f"\n  Stage {i+1}/{len(self.stages)}: "
                  f"{stage.traffic_percent}% traffic to canary")

            # Simulate traffic split
            total_requests = 100
            canary_requests = int(total_requests * stage.traffic_percent / 100)
            stable_requests = total_requests - canary_requests

            # Send traffic to canary
            for inst in self.canary_instances:
                per_instance = canary_requests // len(self.canary_instances)
                for _ in range(per_instance):
                    inst.handle_request(canary_fail_rate)

            # Send traffic to stable
            for inst in self.stable_instances:
                if not inst.healthy:
                    continue
                per_instance = stable_requests // max(len(self.stable_instances), 1)
                for _ in range(per_instance):
                    inst.handle_request(0.005)  # Stable has lower error rate

            # Check canary error rate
            canary_errors = sum(c.error_count for c in self.canary_instances)
            canary_reqs = sum(c.request_count for c in self.canary_instances)
            canary_error_rate = canary_errors / max(canary_reqs, 1)

            stable_errors = sum(s.error_count for s in self.stable_instances)
            stable_reqs = sum(s.request_count for s in self.stable_instances)
            stable_error_rate = stable_errors / max(stable_reqs, 1)

            print(f"    Canary error rate:  {canary_error_rate:.2%} "
                  f"(threshold: {stage.max_error_rate:.2%})")
            print(f"    Stable error rate:  {stable_error_rate:.2%}")

            if canary_error_rate > stage.max_error_rate:
                print(f"    FAILED: canary error rate exceeds threshold")
                self._rollback(new_version)
                return False

            print(f"    PASSED: proceeding to next stage")

        # Full promotion
        self._promote(new_version)
        return True

    def _promote(self, new_version: str) -> None:
        """Promote canary to stable (replace all instances)."""
        print(f"\n  [Canary] PROMOTED: v{new_version} is now stable")
        self.stable_version = new_version
        self.stable_instances = self.canary_instances
        self.canary_instances = []
        self.state = DeploymentState.COMPLETED

    def _rollback(self, failed_version: str) -> None:
        """Remove canary instances and restore full traffic to stable."""
        print(f"\n  [Canary] ROLLBACK: removing v{failed_version} canary")
        self.canary_instances = []
        self.state = DeploymentState.ROLLED_BACK


# =============================================================================
# 4. DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Deployment Strategies Simulation")
    print("=" * 70)

    # --- Blue-Green: successful deployment ---
    print("\n" + "=" * 70)
    print("Scenario 1: Blue-Green Deployment (Success)")
    print("=" * 70)

    bg = BlueGreenDeployment(
        service_name="order-api",
        blue=[
            ServiceInstance("blue-0", "1.0.0"),
            ServiceInstance("blue-1", "1.0.0"),
            ServiceInstance("blue-2", "1.0.0"),
        ],
    )
    success = bg.deploy("1.1.0", instance_count=3, fail_rate=0.01)
    print(f"  Result: {'SUCCESS' if success else 'ROLLED BACK'}")
    print(f"  State: {bg.state.value}")

    # --- Blue-Green: rollback scenario ---
    print("\n" + "=" * 70)
    print("Scenario 2: Blue-Green Deployment (Rollback)")
    print("=" * 70)

    bg2 = BlueGreenDeployment(
        service_name="payment-api",
        blue=[
            ServiceInstance("blue-0", "2.0.0"),
            ServiceInstance("blue-1", "2.0.0"),
        ],
    )
    # High fail rate triggers rollback
    success2 = bg2.deploy("2.1.0-buggy", instance_count=2, fail_rate=0.50)
    print(f"  Result: {'SUCCESS' if success2 else 'ROLLED BACK'}")

    # --- Canary: successful progressive rollout ---
    print("\n" + "=" * 70)
    print("Scenario 3: Canary Deployment (Success)")
    print("=" * 70)

    canary = CanaryDeployment(
        service_name="user-api",
        stable_version="3.0.0",
        stable_instances=[
            ServiceInstance("stable-0", "3.0.0"),
            ServiceInstance("stable-1", "3.0.0"),
            ServiceInstance("stable-2", "3.0.0"),
        ],
    )
    success3 = canary.deploy("3.1.0", canary_count=2, canary_fail_rate=0.01)
    print(f"\n  Result: {'PROMOTED' if success3 else 'ROLLED BACK'}")
    print(f"  State: {canary.state.value}")

    # --- Canary: rollback at 25% ---
    print("\n" + "=" * 70)
    print("Scenario 4: Canary Deployment (Rollback at 25%)")
    print("=" * 70)

    canary2 = CanaryDeployment(
        service_name="search-api",
        stable_version="4.0.0",
        stable_instances=[
            ServiceInstance("stable-0", "4.0.0"),
            ServiceInstance("stable-1", "4.0.0"),
        ],
    )
    success4 = canary2.deploy("4.1.0-buggy", canary_count=2, canary_fail_rate=0.30)
    print(f"\n  Result: {'PROMOTED' if success4 else 'ROLLED BACK'}")
