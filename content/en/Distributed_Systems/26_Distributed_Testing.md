# Lesson 26: Distributed Testing

[Overview](./00_Overview.md) | [Previous: Vector Clocks](./25_Vector_Clocks.md) | [Next: Distributed Observability](./27_Distributed_Observability.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Design Jepsen-style consistency tests for distributed databases and consensus systems
2. Implement fault injection frameworks for network partitions, crashes, and clock skew
3. Build deterministic simulation testing for reproducible distributed system verification
4. Apply chaos engineering principles to production-grade distributed systems
5. Analyze test coverage strategies specific to distributed systems (linearizability checking, trace analysis)

---

## Table of Contents

1. [Why Distributed Testing Is Hard](#1-why-distributed-testing-is-hard)
2. [Fault Injection Framework](#2-fault-injection-framework)
3. [Jepsen-Style Testing](#3-jepsen-style-testing)
4. [Linearizability Checking](#4-linearizability-checking)
5. [Deterministic Simulation Testing](#5-deterministic-simulation-testing)
6. [Chaos Engineering](#6-chaos-engineering)
7. [Property-Based Testing](#7-property-based-testing)
8. [Trace Analysis](#8-trace-analysis)
9. [Real-World Testing Frameworks](#9-real-world-testing-frameworks)
10. [Summary and Key Takeaways](#10-summary-and-key-takeaways)
11. [Practice Problems](#11-practice-problems)
12. [References](#12-references)

---

## 1. Why Distributed Testing Is Hard

### 1.1 The Challenge

```python
import random
import time
import json
import threading
from typing import Dict, List, Optional, Tuple, Set, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum


def explain_testing_challenges():
    """Explain why distributed systems are uniquely difficult to test."""
    print("=== Why Distributed Testing Is Hard ===\n")

    challenges = {
        "Non-determinism": (
            "Same inputs can produce different outputs due to message ordering, "
            "timing, and thread scheduling."
        ),
        "Partial failures": (
            "Any component can fail independently: nodes, network links, disks. "
            "A 5-node system has 2^5 - 1 = 31 failure combinations."
        ),
        "Heisenbugs": (
            "Adding logging or debugging changes timing, which changes behavior. "
            "The bug disappears when you look for it."
        ),
        "State space explosion": (
            "With N nodes, M messages, and F possible failures, the state space "
            "is O(M! × 2^F × N!) — astronomical even for small systems."
        ),
        "Emergent failures": (
            "Individual components work correctly in isolation, but the system "
            "fails when they interact under specific conditions."
        ),
    }

    for name, desc in challenges.items():
        print(f"  {name}:")
        print(f"    {desc}\n")


explain_testing_challenges()
```

---

## 2. Fault Injection Framework

### 2.1 Network Fault Injection

```python
class FaultType(Enum):
    PARTITION = "partition"        # Network partition between groups
    DELAY = "delay"                # Message delay
    DROP = "drop"                  # Message drop
    DUPLICATE = "duplicate"        # Message duplication
    REORDER = "reorder"            # Message reordering
    CORRUPT = "corrupt"            # Message corruption
    CRASH = "crash"                # Node crash
    SLOW = "slow"                  # Slow node (GC pause)
    CLOCK_SKEW = "clock_skew"     # Clock drift


@dataclass
class Fault:
    """A fault to inject into the system."""
    fault_type: FaultType
    target_nodes: list[str] = field(default_factory=list)
    duration_seconds: float = 5.0
    parameters: dict = field(default_factory=dict)
    start_time: float = 0.0


class FaultInjector:
    """
    Fault injection framework for distributed system testing.

    Injects faults (partitions, delays, crashes) into a simulated
    distributed system and records the results.
    """

    def __init__(self):
        self.active_faults: list[Fault] = []
        self.fault_history: list[dict] = []
        self.message_interceptors: list[Callable] = []

    def inject(self, fault: Fault):
        """Inject a fault into the system."""
        fault.start_time = time.time()
        self.active_faults.append(fault)
        self.fault_history.append({
            "type": fault.fault_type.value,
            "targets": fault.target_nodes,
            "duration": fault.duration_seconds,
            "start": fault.start_time,
        })

    def clear_expired(self):
        """Remove expired faults."""
        now = time.time()
        self.active_faults = [
            f for f in self.active_faults
            if now - f.start_time < f.duration_seconds
        ]

    def should_drop_message(self, src: str, dst: str) -> bool:
        """Check if a message should be dropped due to active faults."""
        self.clear_expired()
        for fault in self.active_faults:
            if fault.fault_type == FaultType.PARTITION:
                groups = fault.parameters.get("groups", [])
                for group in groups:
                    if src in group and dst not in group:
                        return True

            elif fault.fault_type == FaultType.DROP:
                rate = fault.parameters.get("rate", 0.5)
                if (src in fault.target_nodes or dst in fault.target_nodes):
                    if random.random() < rate:
                        return True

        return False

    def message_delay(self, src: str, dst: str) -> float:
        """Calculate additional delay for a message."""
        delay = 0.0
        for fault in self.active_faults:
            if fault.fault_type == FaultType.DELAY:
                if src in fault.target_nodes or dst in fault.target_nodes:
                    delay += fault.parameters.get("delay_ms", 100) / 1000.0
        return delay

    def is_node_crashed(self, node_id: str) -> bool:
        """Check if a node is currently crashed."""
        for fault in self.active_faults:
            if fault.fault_type == FaultType.CRASH and node_id in fault.target_nodes:
                return True
        return False

    def generate_random_scenario(self, nodes: list[str],
                                  duration: float = 30.0) -> list[Fault]:
        """Generate a random fault injection scenario."""
        faults = []
        num_faults = random.randint(1, 5)

        for _ in range(num_faults):
            fault_type = random.choice([
                FaultType.PARTITION, FaultType.CRASH, FaultType.DELAY, FaultType.DROP
            ])
            num_targets = random.randint(1, len(nodes) // 2)
            targets = random.sample(nodes, num_targets)
            fault_duration = random.uniform(1.0, duration / 2)

            params = {}
            if fault_type == FaultType.PARTITION:
                # Split into two groups
                mid = len(nodes) // 2
                params["groups"] = [nodes[:mid], nodes[mid:]]
            elif fault_type == FaultType.DELAY:
                params["delay_ms"] = random.randint(50, 500)
            elif fault_type == FaultType.DROP:
                params["rate"] = random.uniform(0.1, 0.5)

            faults.append(Fault(
                fault_type=fault_type,
                target_nodes=targets,
                duration_seconds=fault_duration,
                parameters=params,
            ))

        return faults


def demonstrate_fault_injection():
    """Demonstrate fault injection for testing."""
    print("=== Fault Injection Framework ===\n")

    injector = FaultInjector()
    nodes = ["n1", "n2", "n3", "n4", "n5"]

    # Inject a network partition
    injector.inject(Fault(
        fault_type=FaultType.PARTITION,
        target_nodes=nodes,
        duration_seconds=5.0,
        parameters={"groups": [["n1", "n2"], ["n3", "n4", "n5"]]},
    ))

    print("Network partition: {n1,n2} | {n3,n4,n5}")
    test_pairs = [("n1", "n2"), ("n1", "n3"), ("n3", "n4"), ("n2", "n5")]
    for src, dst in test_pairs:
        dropped = injector.should_drop_message(src, dst)
        print(f"  {src} → {dst}: {'DROPPED' if dropped else 'delivered'}")

    # Generate random scenario
    print(f"\nRandom fault scenario:")
    scenario = injector.generate_random_scenario(nodes, duration=10.0)
    for fault in scenario:
        print(f"  {fault.fault_type.value}: targets={fault.target_nodes}, "
              f"duration={fault.duration_seconds:.1f}s")


demonstrate_fault_injection()
```

---

## 3. Jepsen-Style Testing

### 3.1 Test Structure

```python
class JepsenTest:
    """
    Jepsen-style distributed system test.

    Structure:
    1. Setup: Start cluster, configure
    2. Nemesis: Inject faults (partitions, crashes)
    3. Workload: Run client operations (reads, writes)
    4. Check: Verify consistency properties (linearizability)
    """

    def __init__(self, name: str, nodes: list[str]):
        self.name = name
        self.nodes = nodes
        self.history: list[dict] = []
        self.errors: list[str] = []
        self.injector = FaultInjector()

    def setup(self):
        """Phase 1: Setup the cluster."""
        print(f"  Setup: Initializing {len(self.nodes)} nodes")

    def nemesis(self, faults: list[Fault]):
        """Phase 2: Inject faults."""
        for fault in faults:
            self.injector.inject(fault)
            print(f"  Nemesis: {fault.fault_type.value} on {fault.target_nodes}")

    def workload(self, operations: list[dict]):
        """
        Phase 3: Execute client operations and record history.

        Each operation records:
        - type: "invoke" (start) or "ok"/"fail"/"info" (result)
        - f: the function (read, write, cas)
        - value: the value involved
        - process: which client process
        """
        for op in operations:
            invoke = {
                "type": "invoke",
                "f": op["f"],
                "value": op.get("value"),
                "process": op.get("process", 0),
                "time": time.time(),
            }
            self.history.append(invoke)

            # Simulate execution
            success = not self.injector.is_node_crashed(
                random.choice(self.nodes)
            )

            result = {
                "type": "ok" if success else "fail",
                "f": op["f"],
                "value": op.get("value"),
                "process": op.get("process", 0),
                "time": time.time(),
            }
            self.history.append(result)

    def check(self, checker: 'ConsistencyChecker') -> dict:
        """Phase 4: Check consistency properties."""
        return checker.check(self.history)

    def run(self, faults: list[Fault], operations: list[dict],
            checker: 'ConsistencyChecker') -> dict:
        """Run the complete test."""
        print(f"\n=== Jepsen Test: {self.name} ===")
        self.setup()
        self.nemesis(faults)
        self.workload(operations)
        result = self.check(checker)
        print(f"  Result: {'PASS' if result['valid'] else 'FAIL'}")
        return result


class ConsistencyChecker:
    """Base class for consistency checkers."""

    def check(self, history: list[dict]) -> dict:
        """Check a history for consistency violations."""
        raise NotImplementedError


class RegisterChecker(ConsistencyChecker):
    """
    Check linearizability of a single register.

    A history is linearizable if there exists a total order of
    operations (consistent with real-time ordering) where every
    read returns the value of the most recent preceding write.
    """

    def check(self, history: list[dict]) -> dict:
        # Pair invocations with responses
        ops = []
        pending = {}

        for entry in history:
            if entry["type"] == "invoke":
                pending[entry["process"]] = entry
            elif entry["type"] in ("ok", "fail"):
                invoke = pending.pop(entry["process"], None)
                if invoke and entry["type"] == "ok":
                    ops.append({
                        "f": entry["f"],
                        "value": entry["value"],
                        "invoke_time": invoke["time"],
                        "complete_time": entry["time"],
                        "process": entry["process"],
                    })

        # Simple linearizability check for a register
        # (Full check is NP-complete; this is a simplified version)
        writes = [op for op in ops if op["f"] == "write"]
        reads = [op for op in ops if op["f"] == "read"]

        violations = []
        for read in reads:
            # Find the most recent write that completed before this read started
            preceding_writes = [
                w for w in writes
                if w["complete_time"] <= read["invoke_time"]
            ]
            if preceding_writes:
                last_write = max(preceding_writes, key=lambda w: w["complete_time"])
                if read["value"] != last_write["value"]:
                    violations.append({
                        "read": read,
                        "expected": last_write["value"],
                        "got": read["value"],
                    })

        return {
            "valid": len(violations) == 0,
            "operations": len(ops),
            "violations": violations,
        }


def demonstrate_jepsen_test():
    """Demonstrate a Jepsen-style test."""
    print("=== Jepsen-Style Testing ===\n")

    # Test 1: No faults — should pass
    test = JepsenTest("register-no-faults", ["n1", "n2", "n3"])
    operations = [
        {"f": "write", "value": 1, "process": 0},
        {"f": "read", "value": 1, "process": 1},
        {"f": "write", "value": 2, "process": 0},
        {"f": "read", "value": 2, "process": 1},
    ]
    result = test.run([], operations, RegisterChecker())

    # Test 2: With partition — may fail
    test2 = JepsenTest("register-with-partition", ["n1", "n2", "n3"])
    faults = [Fault(
        fault_type=FaultType.PARTITION,
        target_nodes=["n1", "n2", "n3"],
        duration_seconds=5.0,
        parameters={"groups": [["n1"], ["n2", "n3"]]},
    )]
    result2 = test2.run(faults, operations, RegisterChecker())

    print(f"\n  Test results:")
    print(f"    No faults: {'PASS' if result['valid'] else 'FAIL'}")
    print(f"    With partition: {'PASS' if result2['valid'] else 'FAIL'}")


demonstrate_jepsen_test()
```

---

## 4. Linearizability Checking

### 4.1 WGL Algorithm (Simplified)

```python
class LinearizabilityChecker:
    """
    Linearizability checker using brute-force enumeration.

    For small histories, enumerate all possible linearizations
    and check if any is valid. This is NP-complete in general,
    but feasible for small test cases.

    For production, use Wing & Gong's algorithm or Knossos.
    """

    def __init__(self):
        self.checked: int = 0

    def check(self, operations: list[dict], model: dict) -> bool:
        """
        Check if a history of operations is linearizable.

        Args:
            operations: List of {f, args, ret, start, end}
            model: Initial state of the sequential specification

        Returns:
            True if linearizable
        """
        self.checked = 0
        return self._search(operations, dict(model), set())

    def _search(self, remaining: list[dict], state: dict,
                linearized: set) -> bool:
        """Recursive search for a valid linearization."""
        if not remaining:
            return True

        self.checked += 1

        # Try linearizing each operation that could go next
        for i, op in enumerate(remaining):
            # Operation can be linearized if its interval overlaps with "now"
            # Simplified: just try each remaining op
            new_state = dict(state)
            valid = self._apply_op(new_state, op)

            if valid:
                rest = remaining[:i] + remaining[i+1:]
                if self._search(rest, new_state, linearized | {i}):
                    return True

        return False

    def _apply_op(self, state: dict, op: dict) -> bool:
        """Apply an operation to the model state and check if return matches."""
        f = op["f"]
        if f == "write":
            state["register"] = op["args"]
            return True  # Writes always succeed in the model
        elif f == "read":
            expected = state.get("register")
            return op["ret"] == expected
        elif f == "cas":
            old, new = op["args"]
            if state.get("register") == old:
                state["register"] = new
                return op["ret"] == True
            else:
                return op["ret"] == False
        return False


def demonstrate_linearizability_check():
    """Demonstrate linearizability checking."""
    print("=== Linearizability Checking ===\n")

    checker = LinearizabilityChecker()

    # Linearizable history
    history1 = [
        {"f": "write", "args": 1, "ret": None, "start": 0, "end": 1},
        {"f": "read", "args": None, "ret": 1, "start": 2, "end": 3},
        {"f": "write", "args": 2, "ret": None, "start": 4, "end": 5},
        {"f": "read", "args": None, "ret": 2, "start": 6, "end": 7},
    ]
    result1 = checker.check(history1, {"register": None})
    print(f"History 1 (sequential): linearizable={result1} "
          f"(checked {checker.checked} orderings)")

    # Non-linearizable history
    history2 = [
        {"f": "write", "args": 1, "ret": None, "start": 0, "end": 2},
        {"f": "write", "args": 2, "ret": None, "start": 1, "end": 3},
        {"f": "read", "args": None, "ret": 1, "start": 4, "end": 5},
        # Read returns 1 after write(2) completed → not linearizable
    ]
    result2 = checker.check(history2, {"register": None})
    print(f"History 2 (stale read): linearizable={result2} "
          f"(checked {checker.checked} orderings)")


demonstrate_linearizability_check()
```

---

## 5. Deterministic Simulation Testing

### 5.1 Deterministic Simulation

```python
class DeterministicSimulator:
    """
    Deterministic simulation testing framework.

    All sources of non-determinism (time, network, randomness)
    are controlled by the simulator. This makes tests:
    - Reproducible: same seed → same execution
    - Fast: no real I/O, no sleeps
    - Exhaustive: can explore many schedules

    Used by FoundationDB, TigerBeetle, and others.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.virtual_time: float = 0.0
        self.event_queue: list[Tuple[float, str, dict]] = []
        self.nodes: Dict[str, Any] = {}
        self.message_log: list[dict] = []
        self.delivered: int = 0
        self.dropped: int = 0

    def register_node(self, node_id: str, handler: Callable):
        """Register a node with its message handler."""
        self.nodes[node_id] = handler

    def send(self, src: str, dst: str, msg: dict, delay: Optional[float] = None):
        """Schedule a message delivery."""
        if delay is None:
            delay = self.rng.uniform(0.001, 0.050)  # 1-50ms

        deliver_time = self.virtual_time + delay
        self.event_queue.append((deliver_time, dst, {
            "from": src,
            "to": dst,
            **msg,
        }))
        # Keep queue sorted by time
        self.event_queue.sort(key=lambda x: x[0])

    def schedule_timer(self, node_id: str, delay: float, msg: dict):
        """Schedule a timer event for a node."""
        deliver_time = self.virtual_time + delay
        self.event_queue.append((deliver_time, node_id, {
            "type": "timer",
            "node": node_id,
            **msg,
        }))
        self.event_queue.sort(key=lambda x: x[0])

    def step(self) -> bool:
        """Process the next event. Returns False if no events remain."""
        if not self.event_queue:
            return False

        deliver_time, node_id, msg = self.event_queue.pop(0)
        self.virtual_time = deliver_time

        # Optionally drop messages (controlled by RNG for reproducibility)
        if msg.get("type") != "timer" and self.rng.random() < 0.0:  # 0% drop rate
            self.dropped += 1
            return True

        handler = self.nodes.get(node_id)
        if handler:
            handler(msg)
            self.delivered += 1

        self.message_log.append({
            "time": self.virtual_time,
            "node": node_id,
            "msg": msg,
        })

        return True

    def run(self, max_steps: int = 10000) -> int:
        """Run simulation until completion or max steps."""
        steps = 0
        while steps < max_steps and self.step():
            steps += 1
        return steps

    def stats(self) -> dict:
        return {
            "seed": self.seed,
            "virtual_time": round(self.virtual_time, 6),
            "delivered": self.delivered,
            "dropped": self.dropped,
            "remaining_events": len(self.event_queue),
        }


def demonstrate_deterministic_sim():
    """Demonstrate deterministic simulation testing."""
    print("=== Deterministic Simulation Testing ===\n")

    # Simple leader election simulation
    elected_leader = {"leader": None}

    def make_handler(node_id, sim):
        def handler(msg):
            if msg.get("type") == "timer":
                # Election timeout — start election
                for peer in ["n1", "n2", "n3"]:
                    if peer != node_id:
                        sim.send(node_id, peer, {
                            "type": "vote_request",
                            "candidate": node_id,
                        })
            elif msg.get("type") == "vote_request":
                sim.send(node_id, msg["from"], {
                    "type": "vote_response",
                    "voter": node_id,
                    "granted": True,
                })
            elif msg.get("type") == "vote_response":
                if msg.get("granted") and elected_leader["leader"] is None:
                    elected_leader["leader"] = node_id
        return handler

    # Run with two different seeds
    for seed in [42, 123]:
        sim = DeterministicSimulator(seed=seed)
        elected_leader["leader"] = None

        for nid in ["n1", "n2", "n3"]:
            sim.register_node(nid, make_handler(nid, sim))
            # Random election timeout
            timeout = sim.rng.uniform(0.150, 0.300)
            sim.schedule_timer(nid, timeout, {"election": True})

        steps = sim.run(max_steps=100)
        print(f"  Seed {seed}: leader={elected_leader['leader']}, "
              f"steps={steps}, {sim.stats()}")

    print(f"\n  Key insight: Same seed → same leader every time")
    print(f"  Different seeds explore different schedules")


demonstrate_deterministic_sim()
```

---

## 6. Chaos Engineering

### 6.1 Chaos Experiment Design

```python
@dataclass
class ChaosExperiment:
    """A chaos engineering experiment definition."""
    name: str
    hypothesis: str
    method: str
    abort_conditions: list[str]
    metrics: list[str]
    blast_radius: str = "single-service"


class ChaosRunner:
    """
    Chaos engineering experiment runner.

    Follows the chaos engineering principles:
    1. Define steady state (normal behavior)
    2. Hypothesize that steady state continues during fault
    3. Inject real-world failures
    4. Observe the difference
    """

    def __init__(self):
        self.experiments: list[ChaosExperiment] = []
        self.results: list[dict] = []

    def define(self, experiment: ChaosExperiment):
        self.experiments.append(experiment)

    def run(self, experiment: ChaosExperiment,
            steady_state_check: Callable,
            inject_fault: Callable,
            observe: Callable) -> dict:
        """Run a single chaos experiment."""
        # 1. Verify steady state
        baseline = steady_state_check()
        if not baseline["healthy"]:
            return {"status": "aborted", "reason": "System not in steady state"}

        # 2. Inject fault
        inject_fault()

        # 3. Observe
        observation = observe()

        # 4. Compare
        deviation = abs(observation.get("metric", 0) - baseline.get("metric", 0))
        passed = deviation < observation.get("tolerance", float("inf"))

        result = {
            "experiment": experiment.name,
            "hypothesis": experiment.hypothesis,
            "baseline": baseline,
            "observation": observation,
            "deviation": deviation,
            "passed": passed,
        }
        self.results.append(result)
        return result


def demonstrate_chaos_engineering():
    """Demonstrate chaos engineering experiment design."""
    print("=== Chaos Engineering ===\n")

    experiments = [
        ChaosExperiment(
            name="leader-crash",
            hypothesis="System elects new leader within 5s and maintains availability",
            method="Kill the Raft leader process",
            abort_conditions=["Error rate > 50%", "Latency p99 > 10s"],
            metrics=["election_time", "error_rate", "latency_p99"],
        ),
        ChaosExperiment(
            name="network-partition",
            hypothesis="Minority partition stops accepting writes; majority continues",
            method="iptables partition isolating 2 of 5 nodes",
            abort_conditions=["Data loss detected", "Split brain detected"],
            metrics=["write_availability", "read_consistency", "partition_duration"],
        ),
        ChaosExperiment(
            name="clock-skew",
            hypothesis="System maintains consistency with 500ms clock skew",
            method="Inject 500ms clock offset on one node via ntpd",
            abort_conditions=["Consistency violation", "Transaction rollback > 5%"],
            metrics=["consistency_violations", "transaction_success_rate"],
        ),
    ]

    for exp in experiments:
        print(f"  Experiment: {exp.name}")
        print(f"    Hypothesis: {exp.hypothesis}")
        print(f"    Method: {exp.method}")
        print(f"    Abort if: {', '.join(exp.abort_conditions)}")
        print(f"    Metrics: {', '.join(exp.metrics)}")
        print()

    # Simulate running one experiment
    runner = ChaosRunner()
    result = runner.run(
        experiments[0],
        steady_state_check=lambda: {"healthy": True, "metric": 0.01},
        inject_fault=lambda: None,
        observe=lambda: {"metric": 0.03, "tolerance": 0.05},
    )
    print(f"  Result: {'PASS' if result['passed'] else 'FAIL'}")
    print(f"  Deviation: {result['deviation']}")


demonstrate_chaos_engineering()
```

---

## 7. Property-Based Testing

### 7.1 Generating Random Operations

```python
class DistributedPropertyTest:
    """
    Property-based testing for distributed systems.

    Generates random sequences of operations and faults,
    then verifies invariants after each sequence.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.invariant_checks: list[Callable] = []

    def add_invariant(self, name: str, check: Callable):
        """Add an invariant that must hold after every operation sequence."""
        self.invariant_checks.append((name, check))

    def generate_operations(self, num_ops: int,
                            op_types: list[str]) -> list[dict]:
        """Generate a random sequence of operations."""
        ops = []
        for _ in range(num_ops):
            op_type = self.rng.choice(op_types)
            key = f"key-{self.rng.randint(0, 9)}"
            value = f"val-{self.rng.randint(0, 99)}"

            if op_type == "write":
                ops.append({"f": "write", "key": key, "value": value})
            elif op_type == "read":
                ops.append({"f": "read", "key": key})
            elif op_type == "delete":
                ops.append({"f": "delete", "key": key})
            elif op_type == "cas":
                old = f"val-{self.rng.randint(0, 99)}"
                ops.append({"f": "cas", "key": key, "old": old, "new": value})
        return ops

    def run(self, system_under_test, num_trials: int = 100,
            ops_per_trial: int = 50) -> dict:
        """Run property-based tests."""
        failures = []

        for trial in range(num_trials):
            ops = self.generate_operations(
                ops_per_trial, ["write", "read", "delete", "cas"]
            )

            # Execute operations
            for op in ops:
                system_under_test.execute(op)

            # Check invariants
            for name, check in self.invariant_checks:
                try:
                    result = check(system_under_test)
                    if not result:
                        failures.append({
                            "trial": trial,
                            "invariant": name,
                            "ops": ops,
                        })
                except Exception as e:
                    failures.append({
                        "trial": trial,
                        "invariant": name,
                        "error": str(e),
                    })

        return {
            "trials": num_trials,
            "failures": len(failures),
            "first_failure": failures[0] if failures else None,
        }


def demonstrate_property_testing():
    """Demonstrate property-based testing for distributed systems."""
    print("=== Property-Based Testing ===\n")

    print("Properties to verify:")
    properties = [
        ("Monotonic reads", "Once a value is read, subsequent reads return same or newer"),
        ("Read-your-writes", "A write followed by a read returns the written value"),
        ("No phantom reads", "A read that returns 'not found' cannot later return 'found' without a write"),
        ("Consistent prefix", "If write A happened before write B, no observer sees B without A"),
        ("Convergence", "All replicas eventually have the same state after quiescence"),
    ]

    for name, desc in properties:
        print(f"  {name}: {desc}")


demonstrate_property_testing()
```

---

## 8. Trace Analysis

### 8.1 Distributed Trace Verification

```python
class TraceAnalyzer:
    """
    Analyzes execution traces from distributed systems.

    Checks for:
    - Causal consistency violations
    - Message ordering violations
    - State machine invariant violations
    - Performance anomalies
    """

    def __init__(self):
        self.events: list[dict] = []

    def add_event(self, event: dict):
        self.events.append(event)

    def check_causal_consistency(self) -> list[dict]:
        """Find causal consistency violations in the trace."""
        violations = []
        writes_by_key: Dict[str, list] = defaultdict(list)
        reads_by_key: Dict[str, list] = defaultdict(list)

        for event in self.events:
            if event.get("op") == "write":
                writes_by_key[event["key"]].append(event)
            elif event.get("op") == "read":
                reads_by_key[event["key"]].append(event)

        for key, reads in reads_by_key.items():
            writes = writes_by_key.get(key, [])
            for read in reads:
                # Check if the read value corresponds to a valid write
                valid_values = {w["value"] for w in writes if w["time"] <= read["time"]}
                if read["value"] not in valid_values and read["value"] is not None:
                    violations.append({
                        "type": "stale_read",
                        "key": key,
                        "read_value": read["value"],
                        "valid_values": valid_values,
                    })

        return violations

    def check_state_machine_invariants(self,
                                        invariant: Callable) -> list[dict]:
        """Check state machine invariants at each point in the trace."""
        violations = []
        state: Dict[str, Any] = {}

        for event in self.events:
            if event.get("op") == "write":
                state[event["key"]] = event["value"]
            elif event.get("op") == "delete":
                state.pop(event.get("key"), None)

            if not invariant(state):
                violations.append({
                    "type": "invariant_violation",
                    "event": event,
                    "state": dict(state),
                })

        return violations


def demonstrate_trace_analysis():
    """Demonstrate trace analysis for distributed system verification."""
    print("=== Trace Analysis ===\n")

    analyzer = TraceAnalyzer()

    # Add trace events
    trace = [
        {"op": "write", "key": "x", "value": 1, "node": "n1", "time": 1.0},
        {"op": "write", "key": "x", "value": 2, "node": "n2", "time": 2.0},
        {"op": "read", "key": "x", "value": 1, "node": "n3", "time": 3.0},  # Stale!
        {"op": "read", "key": "x", "value": 2, "node": "n1", "time": 4.0},
    ]

    for event in trace:
        analyzer.add_event(event)

    violations = analyzer.check_causal_consistency()
    print(f"Causal consistency violations: {len(violations)}")
    for v in violations:
        print(f"  {v}")

    # Invariant check: balance should never be negative
    bank_trace = [
        {"op": "write", "key": "balance", "value": 100, "time": 1.0},
        {"op": "write", "key": "balance", "value": 50, "time": 2.0},
        {"op": "write", "key": "balance", "value": -10, "time": 3.0},
    ]

    analyzer2 = TraceAnalyzer()
    for event in bank_trace:
        analyzer2.add_event(event)

    inv_violations = analyzer2.check_state_machine_invariants(
        lambda state: state.get("balance", 0) >= 0
    )
    print(f"\nBalance invariant violations: {len(inv_violations)}")
    for v in inv_violations:
        print(f"  balance={v['state'].get('balance')} at event {v['event']}")


demonstrate_trace_analysis()
```

---

## 9. Real-World Testing Frameworks

### 9.1 Framework Comparison

```python
def compare_testing_frameworks():
    """Compare distributed system testing frameworks."""
    print("=== Testing Framework Comparison ===\n")

    frameworks = [
        {"name": "Jepsen", "language": "Clojure",
         "approach": "Black-box, fault injection, linearizability checking",
         "used_by": "CockroachDB, etcd, MongoDB, Redis, Kafka"},
        {"name": "FoundationDB Simulation", "language": "C++",
         "approach": "Deterministic simulation, single-threaded, virtual time",
         "used_by": "FoundationDB (100M+ random test hours)"},
        {"name": "TLA+/TLC", "language": "TLA+",
         "approach": "Model checking, exhaustive state space exploration",
         "used_by": "AWS (S3, DynamoDB, EBS), Azure (Cosmos DB)"},
        {"name": "Chaos Monkey", "language": "Go",
         "approach": "Random instance termination in production",
         "used_by": "Netflix"},
        {"name": "Litmus", "language": "Go",
         "approach": "Kubernetes-native chaos engineering",
         "used_by": "CNCF ecosystem"},
    ]

    for fw in frameworks:
        print(f"  {fw['name']} ({fw['language']}):")
        print(f"    Approach: {fw['approach']}")
        print(f"    Used by: {fw['used_by']}")
        print()


compare_testing_frameworks()
```

---

## 10. Summary and Key Takeaways

### Testing Strategy Matrix

> **DISTRIBUTED TESTING STRATEGY**
>
> Unit tests → Correctness of individual components
> Integration tests → Component interactions
> Deterministic simulation → Exhaustive schedule exploration
> Property-based tests → Invariant verification under random ops
> Jepsen tests → Consistency under real faults
> Chaos engineering → Resilience in production

### Key Principles

1. **Deterministic simulation is the gold standard**: Reproducible, fast, exhaustive. Used by FoundationDB.
2. **Jepsen catches real bugs**: Every major database has had bugs found by Jepsen.
3. **Property-based > example-based**: Generate random operations; check invariants hold.
4. **Chaos in production discovers what tests miss**: Failure modes you didn't think of.
5. **Linearizability checking is NP-complete**: Use approximations for large histories.

---

## 11. Practice Problems

### Problem 1: Fault Scenario Design

Design 5 fault injection scenarios for a 5-node Raft cluster. Each should target a specific safety or liveness property. Describe the expected behavior.

### Problem 2: Linearizability Check

Given this concurrent history, determine if it is linearizable:
- Client A: write(1) at t=0, ok at t=2
- Client B: write(2) at t=1, ok at t=3
- Client C: read() at t=4, returns 1

### Problem 3: Simulation Design

Design a deterministic simulator for a gossip protocol. The simulator should control message ordering, timing, and failures. Verify convergence within O(log N) rounds.

### Problem 4: Implementation Challenge

Build a Jepsen-like testing framework that: starts a 3-node KV store, runs concurrent write/read workloads, injects network partitions, and checks linearizability of the resulting history.

### Problem 5: Chaos Experiment

Design a chaos experiment for a microservice system with 3 services (A→B→C). Define: steady state hypothesis, fault injection method, rollback criteria, and expected blast radius.

---

## 12. References

1. Kingsbury, K. (2013-2024). "Jepsen: Distributed Systems Safety Research." https://jepsen.io
2. FoundationDB (2021). "Testing Distributed Systems w/ Deterministic Simulation." (FoundationDB paper, SIGMOD 2021)
3. Alvaro, P. et al. (2015). "Lineage-driven Fault Injection." *SIGMOD*.
4. Netflix (2012). "Chaos Monkey." (Principles of Chaos Engineering)
5. Holzmann, G. (2003). *The SPIN Model Checker*. Addison-Wesley.
6. Wing, J. & Gong, C. (1993). "Testing and Verifying Concurrent Objects." *JPSM*.
7. Lamport, L. (2002). "Specifying Systems: The TLA+ Language and Tools for Hardware and Software Engineers."

---

[Next: Lesson 27 — Distributed Observability](./27_Distributed_Observability.md)
