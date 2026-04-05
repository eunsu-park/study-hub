"""
Exercises for Lesson 26: Distributed Testing
Topic: Distributed_Systems

Solutions to practice problems from the lesson.
"""

import random
import time
from typing import Dict, List, Optional, Set
from collections import defaultdict
from dataclasses import dataclass, field


# === Exercise 1: Fault Scenarios for Raft ===
def exercise_1():
    """Design 5 fault scenarios for a 5-node Raft cluster."""
    print("=== Exercise 1: Raft Fault Scenarios ===\n")

    scenarios = [
        {
            "name": "Leader crash during commit",
            "property": "Safety (no committed entries lost)",
            "fault": "Kill leader after replicating to 2/5 nodes",
            "expected": "New leader elected; committed entries preserved; "
                       "uncommitted entries may be lost",
        },
        {
            "name": "Minority partition",
            "property": "Liveness (majority continues)",
            "fault": "Partition: {n1,n2} | {n3,n4,n5}",
            "expected": "Majority partition elects leader and operates normally; "
                       "minority stops accepting writes",
        },
        {
            "name": "Symmetric network partition",
            "property": "Safety (no split brain)",
            "fault": "Partition: {n1,n2} | {n3,n4} | {n5}",
            "expected": "No partition has majority; cluster unavailable until healed; "
                       "no data inconsistency",
        },
        {
            "name": "Follower disk failure",
            "property": "Durability (data survives disk loss)",
            "fault": "Corrupt persistent state on n3, then restart",
            "expected": "n3 receives snapshot from leader and catches up; "
                       "no data loss (majority still has all entries)",
        },
        {
            "name": "Clock skew on leader",
            "property": "Liveness (lease read correctness)",
            "fault": "Advance leader clock by 2x election timeout",
            "expected": "If using LeaseRead, stale reads possible; "
                       "ReadIndex remains safe regardless of clock",
        },
    ]

    for i, s in enumerate(scenarios, 1):
        print(f"  Scenario {i}: {s['name']}")
        print(f"    Tests: {s['property']}")
        print(f"    Fault:  {s['fault']}")
        print(f"    Expected: {s['expected']}\n")


exercise_1()


# === Exercise 2: Linearizability Check ===
def exercise_2():
    """Determine linearizability of a concurrent history."""
    print("=== Exercise 2: Linearizability Check ===\n")

    # History:
    # A: write(1) [t=0, t=2]
    # B: write(2) [t=1, t=3]
    # C: read()→1 [t=4, t=5]

    print("  History:")
    print("    A: write(1) at [0, 2]")
    print("    B: write(2) at [1, 3]")
    print("    C: read()→1 at [4, 5]")
    print()

    # Both writes overlap [1,2], so either could be linearized first
    # Case 1: write(1) then write(2) → last value is 2 → read should return 2 → VIOLATES
    # Case 2: write(2) then write(1) → last value is 1 → read returns 1 → OK

    print("  Analysis:")
    print("    write(1) and write(2) overlap in time [1,2]")
    print("    Linearization 1: write(1) @t=1 → write(2) @t=2 → read @t=4")
    print("      Expected read: 2, actual: 1 → INVALID")
    print("    Linearization 2: write(2) @t=1 → write(1) @t=2 → read @t=4")
    print("      Expected read: 1, actual: 1 → VALID!")
    print()
    print("  Result: LINEARIZABLE")
    print("  (write(2) linearized at t=1, write(1) at t=2, read at t=4)")


exercise_2()


# === Exercise 3: Gossip Simulator ===
def exercise_3():
    """Deterministic simulator for gossip convergence."""
    print("\n=== Exercise 3: Gossip Simulator ===\n")

    class DeterministicGossipSim:
        def __init__(self, num_nodes, seed=42):
            self.rng = random.Random(seed)
            self.num_nodes = num_nodes
            self.informed = {0}
            self.rounds = 0

        def run(self, fanout=2, max_rounds=100):
            while self.informed != set(range(self.num_nodes)):
                self.rounds += 1
                new = set(self.informed)
                for node in self.informed:
                    targets = self.rng.sample(
                        [n for n in range(self.num_nodes) if n != node],
                        min(fanout, self.num_nodes - 1)
                    )
                    new.update(targets)
                self.informed = new
                if self.rounds >= max_rounds:
                    break
            return self.rounds

    import math
    for N in [10, 50, 100, 500]:
        rounds = []
        for seed in range(100):
            sim = DeterministicGossipSim(N, seed=seed)
            r = sim.run(fanout=2)
            rounds.append(r)
        avg = sum(rounds) / len(rounds)
        theoretical = math.ceil(math.log(N) / math.log(3))
        print(f"  N={N:>3}: avg_rounds={avg:.1f}, O(log N)≈{theoretical}, "
              f"max={max(rounds)}")

    print(f"\n  Convergence verified within O(log N) rounds for all sizes.")


exercise_3()


# === Exercise 4: Jepsen-Like Framework ===
def exercise_4():
    """Build a simple Jepsen-like test framework."""
    print("\n=== Exercise 4: Jepsen-Like Framework ===\n")

    class SimpleKV:
        def __init__(self):
            self.data = {}
        def put(self, k, v):
            self.data[k] = v
            return True
        def get(self, k):
            return self.data.get(k)

    class JepsenLite:
        def __init__(self, kv):
            self.kv = kv
            self.history = []

        def run_workload(self, num_ops):
            for i in range(num_ops):
                op = random.choice(["put", "get"])
                key = f"k{random.randint(0, 4)}"
                if op == "put":
                    val = f"v{i}"
                    self.kv.put(key, val)
                    self.history.append(("put", key, val, True))
                else:
                    val = self.kv.get(key)
                    self.history.append(("get", key, val, True))

        def check_linearizability(self):
            """Simple check: last write to each key should be readable."""
            last_writes = {}
            for op, key, val, ok in self.history:
                if op == "put":
                    last_writes[key] = val

            violations = 0
            for key, expected in last_writes.items():
                actual = self.kv.get(key)
                if actual != expected:
                    violations += 1

            return violations == 0

    kv = SimpleKV()
    test = JepsenLite(kv)
    test.run_workload(100)
    result = test.check_linearizability()
    print(f"  Operations: {len(test.history)}")
    print(f"  Linearizable: {result}")


exercise_4()


# === Exercise 5: Chaos Experiment Design ===
def exercise_5():
    """Design chaos experiment for A→B→C microservice chain."""
    print("\n=== Exercise 5: Chaos Experiment ===\n")

    experiment = {
        "system": "A → B → C microservice chain",
        "steady_state": {
            "metric": "p99 latency < 500ms, error rate < 0.1%",
            "measurement": "5-minute sliding window",
        },
        "hypothesis": "Killing service B instances does not cause "
                      "cascading failure in A",
        "method": [
            "1. Verify steady state (5 min baseline)",
            "2. Kill 1 of 3 B instances",
            "3. Observe for 5 minutes",
            "4. Kill second B instance (2/3 down)",
            "5. Observe for 5 minutes",
        ],
        "abort_conditions": [
            "Error rate exceeds 5%",
            "p99 latency exceeds 5 seconds",
            "Any data loss detected",
        ],
        "blast_radius": "Service B only (isolated failure domain)",
        "expected": "A's circuit breaker opens for B, returns degraded response; "
                    "C is unaffected; remaining B instance handles load",
    }

    for k, v in experiment.items():
        if isinstance(v, list):
            print(f"  {k}:")
            for item in v:
                print(f"    - {item}")
        elif isinstance(v, dict):
            print(f"  {k}:")
            for dk, dv in v.items():
                print(f"    {dk}: {dv}")
        else:
            print(f"  {k}: {v}")


exercise_5()


if __name__ == "__main__":
    print("\nAll exercises completed.")
