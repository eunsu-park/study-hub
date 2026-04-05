"""
Exercises for Lesson 20: Distributed Hash Tables
Topic: Distributed_Systems

Solutions to practice problems from the lesson.
"""

import hashlib
import random
import bisect
import math
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from itertools import combinations


# === Exercise 1: Consistent Hashing Balance Analysis ===
def exercise_1():
    """
    Analyze standard deviation of key distribution with
    different virtual node counts.
    """
    print("=== Exercise 1: Consistent Hashing Balance ===\n")

    def simulate_balance(num_nodes, vnodes_per_node, num_keys=100000):
        ring = {}
        sorted_positions = []
        node_map = defaultdict(list)

        for n in range(num_nodes):
            for v in range(vnodes_per_node):
                key = f"node-{n}#vnode-{v}"
                pos = int(hashlib.sha1(key.encode()).hexdigest(), 16) % (2**32)
                ring[pos] = f"node-{n}"
                bisect.insort(sorted_positions, pos)
                node_map[f"node-{n}"].append(pos)

        counts = defaultdict(int)
        for i in range(num_keys):
            h = int(hashlib.sha1(f"key-{i}".encode()).hexdigest(), 16) % (2**32)
            idx = bisect.bisect_right(sorted_positions, h) % len(sorted_positions)
            node = ring[sorted_positions[idx]]
            counts[node] += 1

        values = list(counts.values())
        mean = sum(values) / len(values)
        std_dev = (sum((v - mean)**2 for v in values) / len(values)) ** 0.5
        return std_dev, mean

    for vnodes in [1, 10, 50, 100, 200]:
        std_dev, mean = simulate_balance(10, vnodes)
        cv = std_dev / mean * 100  # Coefficient of variation
        print(f"  vnodes={vnodes:3d}: std_dev={std_dev:.0f}, "
              f"mean={mean:.0f}, CV={cv:.1f}%")


exercise_1()


# === Exercise 2: Chord Finger Table Construction ===
def exercise_2():
    """
    Construct complete finger table for node 14 in a Chord ring
    with m=6 and nodes at {1, 8, 14, 21, 32, 38, 42, 51}.
    """
    print("\n=== Exercise 2: Chord Finger Table ===\n")

    m = 6
    ring_size = 2 ** m  # 64
    nodes = sorted([1, 8, 14, 21, 32, 38, 42, 51])
    target_node = 14

    print(f"  Ring size: {ring_size}")
    print(f"  Nodes: {nodes}")
    print(f"  Target: node {target_node}\n")

    print(f"  {'i':>3} | {'start=(n+2^i)%64':>18} | {'successor':>10}")
    print(f"  {'---':>3} | {'------------------':>18} | {'----------':>10}")

    for i in range(m):
        start = (target_node + 2**i) % ring_size
        # Find successor of start
        successor = None
        for n in nodes:
            if n >= start:
                successor = n
                break
        if successor is None:
            successor = nodes[0]  # Wrap around

        print(f"  {i:>3} | {start:>18} | {successor:>10}")


exercise_2()


# === Exercise 3: Kademlia Routing Analysis ===
def exercise_3():
    """
    Analyze Kademlia routing with B=8, ALPHA=3, K=4.
    """
    print("\n=== Exercise 3: Kademlia Routing ===\n")

    B = 8
    ALPHA = 3
    K = 4

    # Max messages per lookup: ALPHA * ceil(log_2(N)) iterations
    # Each iteration queries ALPHA nodes
    # For B=8, max N=256, log_2(256)=8
    max_iterations = B
    max_messages = ALPHA * max_iterations
    print(f"  B={B}, ALPHA={ALPHA}, K={K}")
    print(f"  Max messages per lookup: {max_messages} ({ALPHA} * {max_iterations})")
    print(f"  Number of k-buckets: {B} (one per bit)")

    # Eviction strategy
    print(f"\n  Eviction strategy when bucket is full:")
    print(f"    1. Ping the least-recently-seen (LRS) contact")
    print(f"    2. If LRS responds: keep LRS, discard new contact")
    print(f"    3. If LRS does not respond: evict LRS, add new contact")
    print(f"    Rationale: Long-lived nodes are more likely to stay alive")


exercise_3()


# === Exercise 4: Chord Stabilize ===
def exercise_4():
    """Implement Chord stabilize() protocol."""
    print("\n=== Exercise 4: Chord Stabilize ===\n")

    class SimpleChordNode:
        def __init__(self, node_id, ring_size=64):
            self.node_id = node_id
            self.ring_size = ring_size
            self.successor = node_id
            self.predecessor = None

        def in_range(self, x, start, end):
            """Check if x is in (start, end] on circular ring."""
            if start < end:
                return start < x <= end
            else:
                return x > start or x <= end

        def stabilize(self, network):
            """
            Stabilize protocol:
            1. Ask successor for its predecessor
            2. If predecessor is between us and successor, adopt it
            3. Notify successor
            """
            succ = network.get(self.successor)
            if succ and succ.predecessor is not None:
                x = succ.predecessor
                if self.in_range(x, self.node_id, self.successor):
                    self.successor = x
                    print(f"    Node {self.node_id}: updated successor to {x}")

            # Notify successor
            succ = network.get(self.successor)
            if succ:
                succ.notify(self.node_id)

        def notify(self, candidate):
            """Handle notification from a potential predecessor."""
            if (self.predecessor is None or
                    self.in_range(candidate, self.predecessor, self.node_id)):
                self.predecessor = candidate

    # Build small network
    network = {}
    for nid in [10, 30, 50]:
        network[nid] = SimpleChordNode(nid)

    # Initial: each node thinks its successor is itself
    network[10].successor = 30
    network[30].successor = 50
    network[50].successor = 10

    # Add node 40 between 30 and 50
    network[40] = SimpleChordNode(40)
    network[40].successor = 50  # Knows about 50

    print("  Before stabilization:")
    for nid, node in sorted(network.items()):
        print(f"    Node {nid}: succ={node.successor}, pred={node.predecessor}")

    # Run stabilize rounds
    for round_num in range(3):
        print(f"\n  Stabilize round {round_num + 1}:")
        for nid in sorted(network.keys()):
            network[nid].stabilize(network)

    print(f"\n  After stabilization:")
    for nid, node in sorted(network.items()):
        print(f"    Node {nid}: succ={node.successor}, pred={node.predecessor}")


exercise_4()


# === Exercise 5: Bounded Load Analysis ===
def exercise_5():
    """
    Analyze bounded load consistent hashing.
    """
    print("\n=== Exercise 5: Bounded Load ===\n")

    epsilon = 0.25
    num_nodes = 8
    num_keys = 10000
    avg_load = num_keys / num_nodes
    max_allowed = avg_load * (1 + epsilon)

    print(f"  epsilon={epsilon}, nodes={num_nodes}, keys={num_keys}")
    print(f"  Average load: {avg_load:.0f}")
    print(f"  Max allowed: {max_allowed:.0f}")
    print(f"  Guarantee: no node gets more than {1+epsilon}x average")
    print(f"\n  When keys are redirected due to load bounds:")
    print(f"    - Lookup efficiency decreases slightly")
    print(f"    - Instead of O(1) on the ring, may need to check O(1/epsilon) nodes")
    print(f"    - Trade-off: better balance vs slightly longer lookup chain")


exercise_5()


if __name__ == "__main__":
    print("\nAll exercises completed.")
