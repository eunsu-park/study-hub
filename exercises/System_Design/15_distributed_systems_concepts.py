"""
Exercises for Lesson 15: Distributed Systems Concepts
Topic: System_Design

Solutions to practice problems from the lesson.
Covers Lamport clocks, vector clocks, and leader election.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import random
import time


# === Exercise 1: Lamport Clock Calculation ===
# Problem: Calculate Lamport timestamp for each event.
# P1: a -> b -> c
#      |        ^
#      v        |
# P2:  d -> e -> f

def exercise_1():
    """Lamport clock calculation for distributed events."""
    print("Lamport Clock Calculation:")
    print("=" * 60)

    print("""
    Event diagram:
    P1: a ──────────> b ──────────────────> c
             |                       ^
             v                       |
    P2: ─────d ─────────> e ─────────f ────>
    """)

    class LamportClock:
        def __init__(self):
            self.time = 0

        def local_event(self):
            self.time += 1
            return self.time

        def send_event(self):
            self.time += 1
            return self.time

        def receive_event(self, sender_time):
            self.time = max(self.time, sender_time) + 1
            return self.time

    p1 = LamportClock()
    p2 = LamportClock()

    # Event a: local event on P1
    a = p1.local_event()
    print(f"  Event a (P1 local):          L(a) = {a}")

    # Event a sends message to P2 -> event d receives
    send_time_a = p1.send_event()
    # Actually, 'a' IS the send event. Let's redo:

    # Reset
    p1 = LamportClock()
    p2 = LamportClock()

    # a: P1 local + send to P2
    a = p1.send_event()
    print(f"  Event a (P1, sends to P2):   L(a) = {a}")

    # d: P2 receives from P1
    d = p2.receive_event(a)
    print(f"  Event d (P2, receives a):    L(d) = {d}")

    # b: P1 local event
    b = p1.local_event()
    print(f"  Event b (P1, local):         L(b) = {b}")

    # e: P2 local event
    e = p2.local_event()
    print(f"  Event e (P2, local):         L(e) = {e}")

    # f: P2 sends to P1
    f = p2.send_event()
    print(f"  Event f (P2, sends to P1):   L(f) = {f}")

    # c: P1 receives from P2
    c = p1.receive_event(f)
    print(f"  Event c (P1, receives f):    L(c) = {c}")

    print(f"\n  Ordering: a({a}) < d({d}) < b({b}) = e({e}) < f({f}) < c({c})")
    print(f"  Note: b and e have same timestamp but are concurrent")
    print(f"  Lamport clocks cannot detect concurrency (use Vector Clocks)")


# === Exercise 2: Vector Clock Analysis ===
# Problem: Determine relationships for given vector clock values.

@dataclass
class VectorClock:
    values: Tuple[int, ...]

    def __le__(self, other):
        """VC1 <= VC2 iff all components of VC1 <= corresponding component of VC2."""
        return all(a <= b for a, b in zip(self.values, other.values))

    def __lt__(self, other):
        """VC1 < VC2 (happens-before) iff VC1 <= VC2 and VC1 != VC2."""
        return self <= other and self.values != other.values

    def __eq__(self, other):
        return self.values == other.values

    def concurrent_with(self, other):
        """Two events are concurrent if neither happens-before the other."""
        return not (self < other) and not (other < self) and not (self == other)

    def relationship(self, other):
        if self < other:
            return "happens-before"
        elif other < self:
            return "happens-after"
        elif self == other:
            return "same"
        else:
            return "concurrent"


def exercise_2():
    """Vector clock relationship analysis."""
    print("Vector Clock Analysis:")
    print("=" * 60)

    clocks = {
        "V1": VectorClock((2, 1, 0)),
        "V2": VectorClock((1, 2, 0)),
        "V3": VectorClock((2, 2, 1)),
        "V4": VectorClock((3, 1, 0)),
    }

    print("\n  Given vector clocks:")
    for name, vc in clocks.items():
        print(f"    {name} = {list(vc.values)}")

    print("\n  Pairwise relationships:")
    names = list(clocks.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            vc1, vc2 = clocks[n1], clocks[n2]
            rel = vc1.relationship(vc2)
            print(f"    {n1} vs {n2}: {rel}")

            # Explain
            if rel == "happens-before":
                print(f"      {n1} < {n2}: all components of {n1} <= {n2} "
                      f"and at least one is strictly less")
            elif rel == "happens-after":
                print(f"      {n1} > {n2}: {n2} happens-before {n1}")
            elif rel == "concurrent":
                print(f"      Neither dominates: some components of {n1} > {n2} "
                      f"and some < {n2}")

    print("\n  Summary:")
    print("    V1 || V2 (concurrent): V1[0]=2>1 but V1[1]=1<2")
    print("    V1 < V3 (happens-before): [2,1,0] <= [2,2,1]")
    print("    V1 < V4 (happens-before): [2,1,0] <= [3,1,0]")
    print("    V2 < V3 (happens-before): [1,2,0] <= [2,2,1]")
    print("    V2 || V4 (concurrent): V2[0]=1<3 but V2[1]=2>1")
    print("    V3 || V4 (concurrent): V3[1]=2>1 but V3[0]=2<3")


# === Exercise 3: Leader Election Design ===
# Problem: Design leader election for 5-node distributed database.

class BullyElection:
    """Bully algorithm for leader election.

    Rules:
    - Highest ID wins
    - On leader failure, any node can start election
    - Node sends ELECTION to all higher-ID nodes
    - If no higher node responds, it becomes leader
    """

    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.nodes = {i: True for i in range(num_nodes)}  # id -> alive
        self.leader = num_nodes - 1  # Highest ID starts as leader
        self.election_messages = 0

    def fail_node(self, node_id):
        self.nodes[node_id] = False
        if node_id == self.leader:
            print(f"    Leader (node {node_id}) has failed!")

    def recover_node(self, node_id):
        self.nodes[node_id] = True
        print(f"    Node {node_id} recovered")
        # Recovered node starts election if it has higher ID
        if node_id > self.leader or not self.nodes.get(self.leader, False):
            self.start_election(node_id)

    def start_election(self, initiator):
        """Start election from initiator node."""
        print(f"    Node {initiator} starts election")
        self.election_messages = 0

        # Send ELECTION to all higher-ID nodes
        higher_alive = False
        for node_id in range(initiator + 1, self.num_nodes):
            if self.nodes.get(node_id, False):
                self.election_messages += 1
                print(f"      Node {initiator} -> ELECTION -> Node {node_id}")
                higher_alive = True
                # Higher node takes over the election
                self.start_election(node_id)
                return

        if not higher_alive:
            # No higher node alive, become leader
            self.leader = initiator
            self.election_messages += self.num_nodes  # Broadcast COORDINATOR
            print(f"    Node {initiator} is the new LEADER! "
                  f"(Messages: {self.election_messages})")

    def detect_leader_failure(self, detector_node):
        """A node detects the leader has failed."""
        if not self.nodes.get(self.leader, False):
            print(f"    Node {detector_node} detects leader failure")
            self.start_election(detector_node)


class RaftElection:
    """Simplified Raft leader election.

    - Nodes have terms (epochs)
    - Candidate requests votes from all nodes
    - Majority vote wins
    - Split-brain prevention through term numbers
    """

    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.nodes = {i: {"alive": True, "term": 0, "voted_for": None}
                      for i in range(num_nodes)}
        self.leader = None
        self.current_term = 0

    def fail_node(self, node_id):
        self.nodes[node_id]["alive"] = False
        if node_id == self.leader:
            self.leader = None
            print(f"    Leader (node {node_id}) has failed!")

    def start_election(self, candidate_id):
        """Candidate requests votes."""
        self.current_term += 1
        term = self.current_term
        print(f"    Node {candidate_id} starts election (term {term})")

        self.nodes[candidate_id]["term"] = term
        self.nodes[candidate_id]["voted_for"] = candidate_id
        votes = 1  # Vote for self

        for node_id, node in self.nodes.items():
            if node_id == candidate_id:
                continue
            if not node["alive"]:
                continue

            # Grant vote if haven't voted in this term
            if node["term"] < term:
                node["term"] = term
                node["voted_for"] = candidate_id
                votes += 1
                print(f"      Node {node_id} votes for {candidate_id}")

        majority = self.num_nodes // 2 + 1
        if votes >= majority:
            self.leader = candidate_id
            print(f"    Node {candidate_id} elected LEADER "
                  f"(votes: {votes}/{self.num_nodes}, majority: {majority})")
        else:
            print(f"    Election failed (votes: {votes}/{self.num_nodes})")

    def check_partition(self, partition_a, partition_b):
        """Check if split-brain is possible."""
        majority = self.num_nodes // 2 + 1
        a_alive = sum(1 for n in partition_a if self.nodes[n]["alive"])
        b_alive = sum(1 for n in partition_b if self.nodes[n]["alive"])

        print(f"    Partition A ({partition_a}): {a_alive} nodes alive")
        print(f"    Partition B ({partition_b}): {b_alive} nodes alive")
        print(f"    Majority needed: {majority}")

        if a_alive >= majority and b_alive >= majority:
            print(f"    DANGER: Both partitions have majority!")
        elif a_alive >= majority:
            print(f"    Partition A can elect leader")
        elif b_alive >= majority:
            print(f"    Partition B can elect leader")
        else:
            print(f"    Neither partition has majority - no leader possible")


def exercise_3():
    """Leader election design for distributed database."""
    print("Leader Election Design:")
    print("=" * 60)

    # Bully Algorithm Demo
    print("\n--- Bully Algorithm ---")
    bully = BullyElection(5)
    print(f"  Initial leader: node {bully.leader}")

    bully.fail_node(4)  # Leader fails
    bully.detect_leader_failure(1)

    print()
    bully.recover_node(4)  # Old leader recovers

    # Raft Algorithm Demo
    print("\n--- Raft Algorithm ---")
    raft = RaftElection(5)

    # Initial election
    raft.start_election(2)

    # Leader failure
    print()
    raft.fail_node(2)
    raft.start_election(3)

    # Network partition scenario
    print("\n--- Network Partition Scenario ---")
    raft2 = RaftElection(5)
    raft2.start_election(4)

    print("\n  Partition: {0,1} vs {2,3,4}")
    raft2.check_partition([0, 1], [2, 3, 4])

    print("\n  Partition: {0,1,2} vs {3,4}")
    raft2.check_partition([0, 1, 2], [3, 4])

    print("\n  Split-brain prevention:")
    print("    - Raft: Requires majority vote -> at most one partition can elect")
    print("    - Odd number of nodes (3, 5, 7) ensures no tie")
    print("    - Fencing tokens prevent stale leaders from making changes")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Lamport Clock Calculation ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Vector Clock Analysis ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Leader Election Design ===")
    print("=" * 60)
    exercise_3()

    print("\nAll exercises completed!")
