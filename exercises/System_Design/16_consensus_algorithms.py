"""
Exercises for Lesson 16: Consensus Algorithms
Topic: System_Design

Solutions to practice problems from the lesson.
Covers Paxos scenario analysis, Raft log recovery, and distributed
configuration management system design.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import random


# === Exercise 1: Paxos Scenario ===
# Problem: Analyze Paxos with 5 Acceptors, 2 Proposers with reordered messages.

@dataclass
class PaxosAcceptor:
    """Simplified Paxos Acceptor."""
    node_id: int
    promised_n: int = 0       # Highest proposal number promised
    accepted_n: int = 0       # Highest proposal number accepted
    accepted_value: Optional[str] = None

    def prepare(self, n):
        """Phase 1: Prepare request."""
        if n > self.promised_n:
            self.promised_n = n
            return True, self.accepted_n, self.accepted_value  # Promise
        return False, None, None  # Reject

    def accept(self, n, value):
        """Phase 2: Accept request."""
        if n >= self.promised_n:
            self.promised_n = n
            self.accepted_n = n
            self.accepted_value = value
            return True
        return False


class PaxosProposer:
    """Simplified Paxos Proposer."""
    def __init__(self, proposal_n, value, acceptors):
        self.n = proposal_n
        self.value = value
        self.acceptors = acceptors
        self.majority = len(acceptors) // 2 + 1

    def run_phase1(self):
        """Send Prepare to all acceptors, collect promises."""
        promises = []
        for acc in self.acceptors:
            ok, acc_n, acc_val = acc.prepare(self.n)
            if ok:
                promises.append((acc.node_id, acc_n, acc_val))
        return promises

    def run_phase2(self, promises):
        """Send Accept to all acceptors."""
        # If any acceptor already accepted a value, use it
        accepted_values = [(n, v) for _, n, v in promises if v is not None]
        if accepted_values:
            # Use the value from highest accepted proposal
            _, self.value = max(accepted_values, key=lambda x: x[0])

        accepts = 0
        for acc in self.acceptors:
            if acc.accept(self.n, self.value):
                accepts += 1
        return accepts


def exercise_1():
    """Paxos scenario analysis."""
    print("Paxos Scenario Analysis:")
    print("=" * 60)

    print("\nSetup: 5 Acceptors, 2 Proposers")
    print("  Proposer 1: proposes 'A' with n=1")
    print("  Proposer 2: proposes 'B' with n=2")
    print("  Messages reordered due to network delay")

    # Scenario: P2's prepare arrives first due to reordering
    acceptors = [PaxosAcceptor(i) for i in range(5)]

    print("\n--- Scenario: P2's Prepare arrives first ---")

    # P2 sends Prepare(n=2) first
    p2 = PaxosProposer(2, "B", acceptors)
    promises_p2 = p2.run_phase1()
    print(f"  P2 Prepare(n=2): {len(promises_p2)} promises (need {p2.majority})")

    # P1 sends Prepare(n=1) after
    p1 = PaxosProposer(1, "A", acceptors)
    promises_p1 = p1.run_phase1()
    print(f"  P1 Prepare(n=1): {len(promises_p1)} promises (need {p1.majority})")
    print(f"  -> P1's prepare REJECTED by all (already promised n=2)")

    # P2 sends Accept(n=2, v="B")
    accepts_p2 = p2.run_phase2(promises_p2)
    print(f"  P2 Accept(n=2, v='B'): {accepts_p2} accepts")

    # P1 tries Accept (will fail)
    accepts_p1 = p1.run_phase2(promises_p1)
    print(f"  P1 Accept(n=1, v='A'): {accepts_p1} accepts")

    print(f"\n  Result: Value 'B' is selected (P2 wins with higher proposal number)")
    print(f"  Acceptor states:")
    for acc in acceptors:
        print(f"    Acceptor {acc.node_id}: accepted_value={acc.accepted_value}, "
              f"accepted_n={acc.accepted_n}")

    # Alternative: P1's prepare arrives first
    print("\n--- Alternative: P1's Prepare arrives first ---")
    acceptors2 = [PaxosAcceptor(i) for i in range(5)]

    p1b = PaxosProposer(1, "A", acceptors2)
    promises = p1b.run_phase1()
    print(f"  P1 Prepare(n=1): {len(promises)} promises")

    # P1 gets accepted
    accepts = p1b.run_phase2(promises)
    print(f"  P1 Accept(n=1, v='A'): {accepts} accepts")

    # P2 then sends Prepare(n=2)
    p2b = PaxosProposer(2, "B", acceptors2)
    promises2 = p2b.run_phase1()
    print(f"  P2 Prepare(n=2): {len(promises2)} promises")
    print(f"  P2 sees accepted value 'A' from P1")

    # P2 must adopt value 'A' (Paxos safety)
    accepts2 = p2b.run_phase2(promises2)
    print(f"  P2 Accept(n=2, v='{p2b.value}'): {accepts2} accepts")
    print(f"  -> P2 adopts 'A' instead of 'B' (Paxos guarantees safety)")


# === Exercise 2: Raft Log Recovery ===
# Problem: Explain log consistency recovery in Raft.

@dataclass
class LogEntry:
    term: int
    index: int
    command: str = ""


def exercise_2():
    """Raft log recovery process."""
    print("Raft Log Recovery:")
    print("=" * 60)

    leader_log = [
        LogEntry(1, 1, "x=1"),
        LogEntry(1, 2, "y=2"),
        LogEntry(2, 3, "z=3"),
        LogEntry(2, 4, "x=4"),   # Follower has wrong term here
        LogEntry(3, 5, "y=5"),
    ]

    follower_log = [
        LogEntry(1, 1, "x=1"),
        LogEntry(1, 2, "y=2"),
        LogEntry(2, 3, "z=3"),
        LogEntry(2, 4, "a=9"),   # Conflict: different term at index 4
    ]

    print("\n  Leader log:   ", end="")
    for e in leader_log:
        print(f"[t{e.term},i{e.index}]", end=" ")
    print()

    print("  Follower log: ", end="")
    for e in follower_log:
        print(f"[t{e.term},i{e.index}]", end=" ")
    print()

    # Recovery process
    print("\n  Recovery Process (AppendEntries RPC):")
    print("  " + "-" * 50)

    # Leader starts from its last log entry and works backwards
    next_index = len(leader_log)  # Start at leader's last index

    print(f"  Step 1: Leader sends AppendEntries with prevLogIndex={next_index-1}, "
          f"prevLogTerm={leader_log[next_index-2].term}")
    print(f"          Follower: no entry at index {next_index-1} -> REJECT")

    next_index -= 1
    print(f"  Step 2: Leader decrements nextIndex to {next_index}")
    print(f"          Sends with prevLogIndex={next_index-1}, "
          f"prevLogTerm={leader_log[next_index-2].term}")

    # Check consistency at index 3
    leader_entry_3 = leader_log[2]
    follower_entry_3 = follower_log[2]
    match = leader_entry_3.term == follower_entry_3.term
    print(f"  Step 3: Follower checks index 3: "
          f"leader term={leader_entry_3.term}, follower term={follower_entry_3.term} "
          f"-> {'MATCH' if match else 'MISMATCH'}")

    print(f"  Step 4: Follower deletes conflicting entries from index 4 onwards")
    print(f"          Follower accepts leader's entries at index 4 and 5")

    # Show final state
    print(f"\n  Final follower log: ", end="")
    for e in leader_log:
        print(f"[t{e.term},i{e.index}]", end=" ")
    print("  (matches leader)")

    print("\n  Key Raft log properties:")
    print("    - If two entries have same index and term, they are identical")
    print("    - If two entries have same index and term, all preceding entries match")
    print("    - Leader never overwrites its own log")
    print("    - Followers overwrite conflicting entries to match leader")


# === Exercise 3: Distributed Configuration Management System ===
# Problem: Design a system across 3 DCs with read optimization.

def exercise_3():
    """Distributed configuration management system design."""
    print("Distributed Configuration Management System:")
    print("=" * 60)

    print("\nRequirements:")
    print("  - Deployed across 3 datacenters")
    print("  - Continue service even with 1 DC failure")
    print("  - Config changes propagate within milliseconds")
    print("  - Optimize read performance")

    print("\nArchitecture Design:")
    print("=" * 40)

    design = {
        "Consensus": {
            "algorithm": "Raft (5 nodes: 2 in DC1, 2 in DC2, 1 in DC3)",
            "reason": "5 nodes tolerate 2 failures (entire DC). "
                      "Raft is simpler to implement than Paxos.",
        },
        "Data Model": {
            "structure": "Hierarchical key-value (/app/config/db_host)",
            "versioning": "Each key has a version (monotonic revision number)",
            "watches": "Clients subscribe to key changes (push notifications)",
        },
        "Write Path": {
            "flow": "Client -> Leader -> Quorum commit (3/5 nodes) -> ACK",
            "latency": "~10-50ms (cross-DC RTT for quorum)",
            "consistency": "Linearizable (all writes go through leader)",
        },
        "Read Path": {
            "default": "Linearizable reads: Leader confirms it's still leader",
            "optimized": "Serializable reads from any node (stale OK for some configs)",
            "local_cache": "Client-side cache with watch-based invalidation",
            "reason": "Most config reads tolerate slight staleness. "
                      "Use linearizable only for critical configs.",
        },
        "DC Failure": {
            "scenario": "DC3 goes down (1 node lost)",
            "impact": "4/5 nodes still available -> quorum maintained",
            "recovery": "Automatic: remaining nodes continue serving",
        },
    }

    for component, details in design.items():
        print(f"\n  {component}:")
        for key, value in details.items():
            print(f"    {key}: {value}")

    # Simulate read optimization
    print("\n--- Read Performance Optimization ---")

    class ConfigStore:
        """Simulated config store with read caching."""
        def __init__(self):
            self.data = {}
            self.revision = 0
            self.watchers = defaultdict(list)  # key -> [callbacks]
            self.read_count = {"leader": 0, "local": 0, "cache": 0}

        def put(self, key, value):
            self.revision += 1
            self.data[key] = (value, self.revision)
            # Notify watchers
            for callback in self.watchers.get(key, []):
                callback(key, value, self.revision)

        def get_linearizable(self, key):
            """Read from leader (strongest consistency)."""
            self.read_count["leader"] += 1
            return self.data.get(key, (None, 0))

        def get_serializable(self, key, local_revision=None):
            """Read from any node (may be stale)."""
            self.read_count["local"] += 1
            return self.data.get(key, (None, 0))

    class ConfigClient:
        """Client with local cache and watch-based invalidation."""
        def __init__(self, store):
            self.store = store
            self.cache = {}
            self.store_reads = 0

        def watch(self, key):
            def on_change(k, v, rev):
                self.cache[k] = (v, rev)
            self.store.watchers[key].append(on_change)

        def get(self, key, linearizable=False):
            # Check cache first
            if key in self.cache:
                self.store.read_count["cache"] += 1
                return self.cache[key]

            if linearizable:
                value, rev = self.store.get_linearizable(key)
            else:
                value, rev = self.store.get_serializable(key)

            self.cache[key] = (value, rev)
            return value, rev

    store = ConfigStore()
    client = ConfigClient(store)

    # Set up watches
    client.watch("db_host")
    client.watch("cache_ttl")

    # Write configs
    store.put("db_host", "db.prod.internal")
    store.put("cache_ttl", "300")
    store.put("log_level", "INFO")

    # Read configs (should hit cache for watched keys)
    for _ in range(100):
        client.get("db_host")
        client.get("cache_ttl")
        client.get("log_level")  # Not watched, hits store

    print(f"  Read sources after 300 reads:")
    print(f"    Cache hits: {store.read_count['cache']}")
    print(f"    Local reads: {store.read_count['local']}")
    print(f"    Leader reads: {store.read_count['leader']}")

    # Update a config
    store.put("db_host", "db2.prod.internal")
    val, rev = client.get("db_host")
    print(f"\n  After db_host update: value='{val}', revision={rev}")
    print(f"  Client cache was invalidated by watch -> got new value immediately")

    print("\n  Technology recommendations: etcd or Consul")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Paxos Scenario ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Raft Log Recovery ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Config Management System ===")
    print("=" * 60)
    exercise_3()

    print("\nAll exercises completed!")
