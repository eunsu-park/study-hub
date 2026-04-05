"""
Exercises for Lesson 09: Database Replication
Topic: System_Design

Solutions to practice problems from the lesson.
Covers replication strategy selection, quorum calculation, failover scenarios,
conflict resolution, and read-your-writes consistency.
"""

import time
import random
from collections import defaultdict
from typing import Dict, List, Optional, Any


# === Exercise 1: Choosing Replication Strategy ===
# Problem: Choose replication strategy for different services.

def exercise_1():
    """Replication strategy selection."""
    scenarios = [
        {
            "service": "Social media service with global users",
            "choice": "C - Multi-leader replication",
            "reason": "Geographically distributed users need low latency in each region. "
                      "Leaders in each region handle local writes. "
                      "Slight consistency delay is acceptable for social media.",
        },
        {
            "service": "Bank system processing financial transactions",
            "choice": "A - Single-leader + Synchronous replication",
            "reason": "Strong consistency required for financial data. "
                      "Data loss is unacceptable. Write latency is tolerable.",
        },
        {
            "service": "Real-time chat application",
            "choice": "B - Single-leader + Asynchronous replication",
            "reason": "Fast response needed for user experience. "
                      "Message order matters (single leader preserves ordering). "
                      "Slight data loss on failover is acceptable.",
        },
        {
            "service": "Log collection and analysis system",
            "choice": "D - Leaderless replication",
            "reason": "High write throughput needed. "
                      "Eventual consistency is sufficient for logs. "
                      "High availability is more important than consistency.",
        },
    ]

    print("Replication Strategy Selection:")
    print("=" * 60)
    for i, s in enumerate(scenarios, 1):
        print(f"\n{i}. {s['service']}")
        print(f"   Choice: {s['choice']}")
        print(f"   Reason: {s['reason']}")


# === Exercise 2: Quorum Calculation ===
# Problem: Analyze quorum configurations for N=5 cluster.

def exercise_2():
    """Quorum calculation and analysis."""
    print("Quorum Calculation (N=5 replicas):")
    print("=" * 60)

    N = 5

    print("\nProblem 1: W, R values for strong consistency with max availability")
    print(f"  Constraint: R + W > N = {N}")
    configs = []
    for W in range(1, N + 1):
        for R in range(1, N + 1):
            if R + W > N:
                write_tolerance = N - W
                read_tolerance = N - R
                min_tolerance = min(write_tolerance, read_tolerance)
                configs.append((W, R, write_tolerance, read_tolerance, min_tolerance))

    # Best balanced config
    configs.sort(key=lambda x: (-x[4], abs(x[0] - x[1])))
    print(f"\n  {'W':>3} {'R':>3} | {'W+R':>5} | {'W fail tol':>10} | {'R fail tol':>10} | {'Min tol':>8}")
    print("  " + "-" * 55)
    for W, R, wt, rt, mt in configs[:5]:
        print(f"  {W:>3} {R:>3} | {W+R:>5} | {wt:>10} | {rt:>10} | {mt:>8}")

    print(f"\n  Best balanced: W=3, R=3 (tolerates 2 failures for both reads and writes)")

    print(f"\nProblem 2: W=3, R=2 analysis")
    W, R = 3, 2
    print(f"  R + W = {R + W} = N = {N}")
    print(f"  R + W > N? {R + W > N} -> NOT strong consistency (R + W must be > N)")
    print(f"  Write availability: {N - W} node failures tolerated")
    print(f"  Read availability: {N - R} node failures tolerated")

    print(f"\nProblem 3: Maximize write availability with strong consistency")
    print(f"  W=1, R=5: Write tolerates {N-1} failures, but read requires ALL nodes")
    print(f"  This is extreme: great write availability, terrible read availability")
    print(f"  Better: W=2, R=4 -> Write tolerates 3, Read tolerates 1")


# === Exercise 3: Failover Scenario ===
# Problem: MySQL master-slave failover with async replication.

class ReplicaNode:
    """Simulates a replica node with binlog position."""
    def __init__(self, name, binlog_position):
        self.name = name
        self.binlog_position = binlog_position
        self.is_master = False


def exercise_3():
    """Failover scenario analysis."""
    print("Failover Scenario (MySQL Master-Slave):")
    print("=" * 60)

    master_last_pos = 1020
    slaves = [
        ReplicaNode("Slave A", 1000),
        ReplicaNode("Slave B", 950),
    ]

    print(f"\nMaster failed! Last binlog position: {master_last_pos}")
    for slave in slaves:
        print(f"  {slave.name}: binlog position {slave.binlog_position}")

    # 1. Which slave to promote?
    best_slave = max(slaves, key=lambda s: s.binlog_position)
    print(f"\n1. Promote: {best_slave.name}")
    print(f"   Reason: Position {best_slave.binlog_position} > "
          f"{min(s.binlog_position for s in slaves)} (most up-to-date)")

    # 2. Maximum lost transactions
    lost = master_last_pos - best_slave.binlog_position
    print(f"\n2. Maximum lost transactions: {lost}")
    print(f"   Master was at {master_last_pos}, best slave at {best_slave.binlog_position}")

    # 3. How to handle old master when recovered
    print(f"\n3. Old master recovery procedure:")
    print(f"   a) Reconfigure old master as slave of new master ({best_slave.name})")
    print(f"   b) Discard positions {best_slave.binlog_position}-{master_last_pos} "
          f"(conflict prevention)")
    print(f"   c) Start replication from new master's current position")
    print(f"   d) Monitor replication lag until fully caught up")

    # Simulation of the failover
    print("\n--- Failover Simulation ---")
    best_slave.is_master = True
    other_slave = [s for s in slaves if s != best_slave][0]

    print(f"  Step 1: {best_slave.name} promoted to master (pos={best_slave.binlog_position})")
    print(f"  Step 2: {other_slave.name} replicating from {best_slave.name}")
    print(f"  Step 3: {other_slave.name} catches up "
          f"({best_slave.binlog_position - other_slave.binlog_position} transactions behind)")
    print(f"  Step 4: Old master joins as slave after recovery")
    print(f"  Step 5: DNS/VIP updated to point to {best_slave.name}")


# === Exercise 4: Conflict Resolution Design ===
# Problem: Multi-leader e-commerce inventory conflict resolution.

def exercise_4():
    """Conflict resolution for multi-leader inventory updates."""
    print("Conflict Resolution: Multi-Leader Inventory:")
    print("=" * 60)

    initial_inventory = 100

    print(f"\nInitial inventory: {initial_inventory}")
    print(f"Leader A (Seoul):  100 - 5 = 95  (5 units sold)")
    print(f"Leader B (Tokyo):  100 - 3 = 97  (3 units sold)")
    print(f"\nConflict: Which value is correct?")

    methods = [
        {
            "name": "LWW (Last Write Wins)",
            "result": "95 or 97 (depends on timestamp)",
            "pros": "Simple implementation",
            "cons": "Data loss: either 5 or 3 sales are missing",
            "recommended": False,
        },
        {
            "name": "CRDT (Counter)",
            "result": f"{initial_inventory} - 5 - 3 = {initial_inventory - 5 - 3}",
            "pros": "No data loss, mathematically correct",
            "cons": "Inventory could go negative temporarily",
            "recommended": True,
        },
        {
            "name": "Distributed Lock",
            "result": "Sequential: 100 -> 95 -> 92",
            "pros": "Strict correctness",
            "cons": "High latency across regions, complexity",
            "recommended": False,
        },
        {
            "name": "Route to Single Leader",
            "result": "Only one leader handles inventory writes",
            "pros": "Prevents conflicts at source",
            "cons": "Single point of failure for writes",
            "recommended": False,
        },
    ]

    for m in methods:
        marker = " [RECOMMENDED]" if m["recommended"] else ""
        print(f"\n  {m['name']}{marker}")
        print(f"    Result: {m['result']}")
        print(f"    Pros: {m['pros']}")
        print(f"    Cons: {m['cons']}")

    # CRDT counter simulation
    print("\n--- CRDT Counter Simulation ---")

    class CRDTCounter:
        """Grow-only counter CRDT for inventory decrements."""
        def __init__(self, replicas):
            self.decrements = {r: 0 for r in replicas}

        def decrement(self, replica, amount):
            self.decrements[replica] += amount

        def merge(self, other):
            for r in self.decrements:
                self.decrements[r] = max(self.decrements[r],
                                          other.decrements.get(r, 0))

        def total_decrements(self):
            return sum(self.decrements.values())

        def value(self, initial):
            return initial - self.total_decrements()

    # Two replicas making independent decrements
    counter_a = CRDTCounter(["Seoul", "Tokyo"])
    counter_b = CRDTCounter(["Seoul", "Tokyo"])

    counter_a.decrement("Seoul", 5)
    counter_b.decrement("Tokyo", 3)

    print(f"  Before merge:")
    print(f"    Seoul counter: decrements={dict(counter_a.decrements)}, "
          f"inventory={counter_a.value(100)}")
    print(f"    Tokyo counter: decrements={dict(counter_b.decrements)}, "
          f"inventory={counter_b.value(100)}")

    counter_a.merge(counter_b)
    counter_b.merge(counter_a)

    print(f"  After merge:")
    print(f"    Seoul counter: decrements={dict(counter_a.decrements)}, "
          f"inventory={counter_a.value(100)}")
    print(f"    Tokyo counter: decrements={dict(counter_b.decrements)}, "
          f"inventory={counter_b.value(100)}")


# === Exercise 5: Read-Your-Writes Consistency ===
# Problem: Implement Read-Your-Writes in async replication environment.

class ReplicatedDB:
    """Database with async replication and read-your-writes support."""

    def __init__(self, num_replicas=3):
        self.leader = {"data": {}, "lsn": 0}  # Log Sequence Number
        self.replicas = [
            {"data": {}, "lsn": 0} for _ in range(num_replicas)
        ]
        self.replication_lag = [0] * num_replicas  # Simulated lag in LSN

    def write(self, key, value):
        """Write to leader, async replicate."""
        self.leader["lsn"] += 1
        self.leader["data"][key] = (value, self.leader["lsn"])

        # Async replication (with lag)
        for i, replica in enumerate(self.replicas):
            lag = random.randint(0, 3)  # 0-3 LSN behind
            self.replication_lag[i] = lag
            catchup_lsn = max(0, self.leader["lsn"] - lag)
            # Copy data up to catchup_lsn
            for k, (v, lsn) in self.leader["data"].items():
                if lsn <= catchup_lsn:
                    replica["data"][k] = (v, lsn)
            replica["lsn"] = catchup_lsn

        return self.leader["lsn"]

    def read_from_leader(self, key):
        """Always consistent but adds load to leader."""
        entry = self.leader["data"].get(key)
        return entry[0] if entry else None

    def read_from_replica(self, key, min_lsn=0):
        """Read from a replica that meets min_lsn requirement."""
        for replica in self.replicas:
            if replica["lsn"] >= min_lsn:
                entry = replica["data"].get(key)
                if entry and entry[1] <= replica["lsn"]:
                    return entry[0]
        # No replica meets requirement, fall back to leader
        return self.read_from_leader(key)


class ReadYourWritesClient:
    """Client that implements Read-Your-Writes guarantee."""

    def __init__(self, db):
        self.db = db
        self.last_write_lsn = 0

    def write(self, key, value):
        lsn = self.db.write(key, value)
        self.last_write_lsn = lsn
        return lsn

    def read(self, key):
        """Read with RYW guarantee using LSN tracking."""
        return self.db.read_from_replica(key, min_lsn=self.last_write_lsn)


def exercise_5():
    """Read-Your-Writes consistency implementation."""
    print("Read-Your-Writes Consistency:")
    print("=" * 60)

    random.seed(42)
    db = ReplicatedDB(num_replicas=3)
    client = ReadYourWritesClient(db)

    # Write some data
    print("\n--- Writes ---")
    lsn1 = client.write("balance", "1000")
    print(f"  Write balance=1000 (LSN={lsn1})")
    lsn2 = client.write("balance", "900")
    print(f"  Write balance=900 (LSN={lsn2})")

    # Read with RYW guarantee
    print("\n--- Reads with RYW guarantee ---")
    value = client.read("balance")
    print(f"  Read balance = {value} (client tracks LSN={client.last_write_lsn})")
    print(f"  Replica LSNs: {[r['lsn'] for r in db.replicas]}")

    # Demonstrate without RYW
    print("\n--- Reads WITHOUT RYW (from random replica) ---")
    for i in range(5):
        replica = random.choice(db.replicas)
        entry = replica["data"].get("balance")
        val = entry[0] if entry else "NOT FOUND"
        print(f"  Read from replica (LSN={replica['lsn']}): balance = {val}")

    print("\nMethods to implement RYW:")
    print("  1. Read from leader for T seconds after write")
    print("  2. Client tracks last write timestamp, waits for replica to catch up")
    print("  3. Client tracks LSN, reads from replica with sufficient LSN (used above)")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Replication Strategy Selection ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Quorum Calculation ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Failover Scenario ===")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("=== Exercise 4: Conflict Resolution ===")
    print("=" * 60)
    exercise_4()

    print("\n" + "=" * 60)
    print("=== Exercise 5: Read-Your-Writes Consistency ===")
    print("=" * 60)
    exercise_5()

    print("\nAll exercises completed!")
