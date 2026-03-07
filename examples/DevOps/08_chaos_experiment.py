#!/usr/bin/env python3
"""Example: Chaos Experiment Framework

A simple chaos engineering framework demonstrating steady-state
hypothesis definition, fault injection, validation, and rollback.
Related lesson: 16_Chaos_Engineering.md
"""

# =============================================================================
# CHAOS ENGINEERING PRINCIPLES (from Principles of Chaos)
#   1. Start by defining 'steady state' as measurable output
#   2. Hypothesize that steady state will continue in both control and experiment
#   3. Introduce real-world events (server failure, network partition, etc.)
#   4. Try to disprove the hypothesis
#   5. Minimize blast radius
# =============================================================================

import time
import random
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
from contextlib import contextmanager


# =============================================================================
# 1. STEADY STATE HYPOTHESIS
# =============================================================================

class ComparisonOp(Enum):
    LT = "less_than"
    LTE = "less_than_or_equal"
    GT = "greater_than"
    GTE = "greater_than_or_equal"
    EQ = "equal"
    BETWEEN = "between"


@dataclass
class SteadyStateProbe:
    """A single check that defines normal system behavior."""
    name: str
    probe_fn: Callable[[], float]
    operator: ComparisonOp
    threshold: float | tuple[float, float]

    def evaluate(self) -> tuple[bool, float]:
        """Run the probe and check against threshold. Returns (passed, value)."""
        value = self.probe_fn()
        if self.operator == ComparisonOp.LT:
            return value < self.threshold, value
        elif self.operator == ComparisonOp.LTE:
            return value <= self.threshold, value
        elif self.operator == ComparisonOp.GT:
            return value > self.threshold, value
        elif self.operator == ComparisonOp.GTE:
            return value >= self.threshold, value
        elif self.operator == ComparisonOp.EQ:
            return value == self.threshold, value
        elif self.operator == ComparisonOp.BETWEEN:
            low, high = self.threshold
            return low <= value <= high, value
        return False, value


@dataclass
class SteadyStateHypothesis:
    """Collection of probes defining system steady state."""
    description: str
    probes: list[SteadyStateProbe] = field(default_factory=list)

    def validate(self) -> tuple[bool, list[dict]]:
        """Validate all probes. Returns (all_passed, results)."""
        results = []
        all_passed = True
        for probe in self.probes:
            passed, value = probe.evaluate()
            results.append({
                "probe": probe.name,
                "passed": passed,
                "value": value,
                "threshold": probe.threshold,
                "operator": probe.operator.value,
            })
            if not passed:
                all_passed = False
        return all_passed, results


# =============================================================================
# 2. FAULT INJECTION
# =============================================================================

class FaultType(Enum):
    LATENCY = "latency_injection"
    ERROR = "error_injection"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_PARTITION = "network_partition"
    DEPENDENCY_FAILURE = "dependency_failure"


@dataclass
class FaultAction:
    """A fault to inject into the system."""
    name: str
    fault_type: FaultType
    inject_fn: Callable[[], None]
    rollback_fn: Callable[[], None]
    duration_seconds: float = 10.0
    blast_radius: str = "single-instance"  # single-instance, service, zone


# =============================================================================
# 3. SIMULATED SYSTEM
# =============================================================================
# A simplified service with controllable failure modes.

class SimulatedService:
    """A service with tunable latency, error rate, and capacity."""

    def __init__(self, name: str, capacity: int = 100):
        self.name = name
        self.capacity = capacity
        self._latency_ms: float = 10.0
        self._error_rate: float = 0.01
        self._is_partitioned: bool = False
        self._dependency_available: bool = True
        self._request_count: int = 0
        self._error_count: int = 0
        self._latencies: list[float] = []
        self._lock = threading.Lock()

    def handle_request(self) -> tuple[bool, float]:
        """Handle a request. Returns (success, latency_ms)."""
        with self._lock:
            self._request_count += 1

            if self._is_partitioned:
                self._error_count += 1
                return False, 0.0

            if not self._dependency_available:
                self._error_count += 1
                return False, self._latency_ms * 5  # Timeout waiting for dep

            latency = random.gauss(self._latency_ms, self._latency_ms * 0.2)
            latency = max(1.0, latency)
            self._latencies.append(latency)

            if random.random() < self._error_rate:
                self._error_count += 1
                return False, latency

            return True, latency

    @property
    def error_rate(self) -> float:
        if self._request_count == 0:
            return 0.0
        return self._error_count / self._request_count

    @property
    def p99_latency(self) -> float:
        if not self._latencies:
            return 0.0
        sorted_lat = sorted(self._latencies)
        idx = int(len(sorted_lat) * 0.99)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    @property
    def availability(self) -> float:
        if self._request_count == 0:
            return 1.0
        return 1.0 - self.error_rate

    def reset_metrics(self) -> None:
        with self._lock:
            self._request_count = 0
            self._error_count = 0
            self._latencies = []

    # Fault injection points
    def inject_latency(self, additional_ms: float) -> None:
        self._latency_ms += additional_ms

    def restore_latency(self, additional_ms: float) -> None:
        self._latency_ms -= additional_ms

    def inject_errors(self, rate: float) -> None:
        self._error_rate = rate

    def restore_errors(self, rate: float) -> None:
        self._error_rate = rate

    def partition(self) -> None:
        self._is_partitioned = True

    def unpartition(self) -> None:
        self._is_partitioned = False

    def disable_dependency(self) -> None:
        self._dependency_available = False

    def enable_dependency(self) -> None:
        self._dependency_available = True


# =============================================================================
# 4. EXPERIMENT ENGINE
# =============================================================================

class ExperimentResult(Enum):
    PASSED = "passed"       # Hypothesis held under fault
    FAILED = "failed"       # Hypothesis disproved — system is not resilient
    ABORTED = "aborted"     # Pre-check failed — system not in steady state


@dataclass
class ExperimentReport:
    """Result of a chaos experiment run."""
    experiment_name: str
    result: ExperimentResult
    pre_check: dict
    fault_applied: str
    post_check: dict
    duration_seconds: float
    recommendation: str


def run_experiment(
    name: str,
    hypothesis: SteadyStateHypothesis,
    fault: FaultAction,
    traffic_fn: Callable[[], None],
    traffic_duration: float = 2.0,
) -> ExperimentReport:
    """Execute a chaos experiment following the scientific method.

    Steps:
    1. Validate steady state (pre-check)
    2. Inject fault
    3. Generate traffic under fault conditions
    4. Validate steady state again (post-check)
    5. Roll back the fault
    6. Report findings
    """
    start = time.monotonic()

    # Step 1: Pre-check
    print(f"\n[Experiment: {name}]")
    print(f"  Hypothesis: {hypothesis.description}")
    print(f"  Fault: {fault.name} ({fault.fault_type.value})")
    print(f"  Blast radius: {fault.blast_radius}")

    pre_passed, pre_results = hypothesis.validate()
    print(f"\n  Pre-check:")
    for r in pre_results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"    [{status}] {r['probe']}: {r['value']:.4f} "
              f"({r['operator']} {r['threshold']})")

    if not pre_passed:
        print(f"  ABORTED: System not in steady state before experiment")
        return ExperimentReport(
            experiment_name=name,
            result=ExperimentResult.ABORTED,
            pre_check={"passed": False, "results": pre_results},
            fault_applied=fault.name,
            post_check={},
            duration_seconds=time.monotonic() - start,
            recommendation="Fix existing issues before running chaos experiments",
        )

    # Step 2: Inject fault
    print(f"\n  Injecting fault: {fault.name}...")
    fault.inject_fn()

    # Step 3: Generate traffic under fault
    print(f"  Generating traffic for {traffic_duration}s...")
    traffic_fn()

    # Step 4: Post-check
    post_passed, post_results = hypothesis.validate()
    print(f"\n  Post-check (under fault):")
    for r in post_results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"    [{status}] {r['probe']}: {r['value']:.4f} "
              f"({r['operator']} {r['threshold']})")

    # Step 5: Rollback
    print(f"\n  Rolling back fault...")
    fault.rollback_fn()

    # Step 6: Determine result
    duration = time.monotonic() - start
    if post_passed:
        result = ExperimentResult.PASSED
        recommendation = (
            "System maintained steady state under fault. "
            "Consider increasing blast radius in next experiment."
        )
        print(f"\n  RESULT: PASSED (system is resilient to {fault.name})")
    else:
        result = ExperimentResult.FAILED
        failed_probes = [r["probe"] for r in post_results if not r["passed"]]
        recommendation = (
            f"System failed to maintain steady state. "
            f"Failed probes: {', '.join(failed_probes)}. "
            f"Add retry logic, circuit breakers, or fallback mechanisms."
        )
        print(f"\n  RESULT: FAILED (system is NOT resilient to {fault.name})")

    return ExperimentReport(
        experiment_name=name,
        result=result,
        pre_check={"passed": True, "results": pre_results},
        fault_applied=fault.name,
        post_check={"passed": post_passed, "results": post_results},
        duration_seconds=duration,
        recommendation=recommendation,
    )


# =============================================================================
# 5. DEMO: RUN EXPERIMENTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Chaos Experiment Framework Demo")
    print("=" * 70)

    # Create a simulated service
    svc = SimulatedService("order-api", capacity=100)

    # Helper: generate traffic
    def generate_traffic(n: int = 200) -> None:
        svc.reset_metrics()
        for _ in range(n):
            svc.handle_request()

    # --- Experiment 1: Latency injection ---
    svc.reset_metrics()
    generate_traffic(500)  # Warm up for baseline

    svc.reset_metrics()
    hypothesis1 = SteadyStateHypothesis(
        description="Service maintains <50ms p99 latency and >99% availability",
        probes=[
            SteadyStateProbe(
                "p99_latency",
                lambda: svc.p99_latency,
                ComparisonOp.LT,
                50.0,
            ),
            SteadyStateProbe(
                "availability",
                lambda: svc.availability,
                ComparisonOp.GTE,
                0.99,
            ),
        ],
    )

    additional_latency = 30.0
    fault1 = FaultAction(
        name="30ms additional latency",
        fault_type=FaultType.LATENCY,
        inject_fn=lambda: svc.inject_latency(additional_latency),
        rollback_fn=lambda: svc.restore_latency(additional_latency),
        blast_radius="single-instance",
    )

    generate_traffic(200)  # Establish baseline
    report1 = run_experiment(
        "latency-injection-30ms",
        hypothesis1,
        fault1,
        lambda: generate_traffic(300),
    )

    # --- Experiment 2: Dependency failure ---
    svc.reset_metrics()
    generate_traffic(200)  # Fresh baseline

    hypothesis2 = SteadyStateHypothesis(
        description="Service degrades gracefully when dependency is unavailable",
        probes=[
            SteadyStateProbe(
                "availability",
                lambda: svc.availability,
                ComparisonOp.GTE,
                0.90,  # Lower threshold — graceful degradation
            ),
        ],
    )

    fault2 = FaultAction(
        name="database dependency unavailable",
        fault_type=FaultType.DEPENDENCY_FAILURE,
        inject_fn=svc.disable_dependency,
        rollback_fn=svc.enable_dependency,
        blast_radius="service",
    )

    report2 = run_experiment(
        "dependency-failure",
        hypothesis2,
        fault2,
        lambda: generate_traffic(200),
    )

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("Experiment Summary")
    print("=" * 70)
    for report in [report1, report2]:
        print(f"\n  {report.experiment_name}")
        print(f"    Result:         {report.result.value}")
        print(f"    Duration:       {report.duration_seconds:.2f}s")
        print(f"    Recommendation: {report.recommendation}")
