# Lesson 20: Distributed Hash Tables

[Overview](./00_Overview.md) | [Previous: Raft Implementation Part 2](./19_Raft_Implementation_Part2.md) | [Next: Gossip Protocols](./21_Gossip_Protocols.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement consistent hashing with virtual nodes and bounded load balancing
2. Build a Chord distributed hash table with finger tables and stabilization
3. Implement the Kademlia DHT protocol with XOR-based distance routing
4. Design replication strategies on top of DHTs for fault tolerance
5. Analyze the trade-offs between different DHT designs in terms of lookup latency, churn handling, and load balance

---

## Table of Contents

1. [Introduction to DHTs](#1-introduction-to-dhts)
2. [Consistent Hashing Fundamentals](#2-consistent-hashing-fundamentals)
3. [Virtual Nodes](#3-virtual-nodes)
4. [Chord Protocol](#4-chord-protocol)
5. [Kademlia Protocol](#5-kademlia-protocol)
6. [Replication on DHTs](#6-replication-on-dhts)
7. [Churn Handling](#7-churn-handling)
8. [Load Balancing](#8-load-balancing)
9. [Real-World DHT Systems](#9-real-world-dht-systems)
10. [Summary and Key Takeaways](#10-summary-and-key-takeaways)
11. [Practice Problems](#11-practice-problems)
12. [References](#12-references)

---

## 1. Introduction to DHTs

### 1.1 What is a DHT?

A Distributed Hash Table (DHT) is a decentralized system that provides a hash-table-like lookup service. Each node in the network is responsible for a portion of the key space, and any node can efficiently route a lookup to the node responsible for a given key.

```
Traditional Hash Table:          Distributed Hash Table:
┌─────────────────────┐         ┌───┐ ┌───┐ ┌───┐ ┌───┐
│ key → bucket → value│         │ N1│ │ N2│ │ N3│ │ N4│
└─────────────────────┘         └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘
  Single machine                  │     │     │     │
  O(1) lookup                     │  ┌──┴─────┴──┐  │
  Single point of failure         │  │ Key Space  │  │
                                  │  │ 0 ──── 2^m │  │
                                  │  └────────────┘  │
                                  └──── O(log N) ────┘
```

### 1.2 Key Properties

| Property | Description |
|----------|-------------|
| **Decentralization** | No central coordinator; all nodes are equal |
| **Scalability** | O(log N) routing with O(log N) state per node |
| **Fault tolerance** | Nodes can join/leave without global reorganization |
| **Load balance** | Keys are evenly distributed across nodes |

---

## 2. Consistent Hashing Fundamentals

### 2.1 The Ring

Consistent hashing maps both keys and nodes onto a circular identifier space [0, 2^m):

```python
import hashlib
import bisect
import random
from typing import Optional, Dict, List, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass, field


class ConsistentHashRing:
    """
    Consistent hashing ring with configurable hash function.

    Keys are assigned to the first node encountered clockwise
    on the ring from the key's hash position.
    """

    def __init__(self, hash_bits: int = 160):
        self.hash_bits = hash_bits
        self.ring_size = 2 ** hash_bits
        self.nodes: dict[int, str] = {}  # position → node_id
        self.sorted_positions: list[int] = []
        self.node_positions: dict[str, list[int]] = defaultdict(list)

    def _hash(self, key: str) -> int:
        """Hash a key to a position on the ring."""
        h = hashlib.sha1(key.encode()).hexdigest()
        return int(h, 16) % self.ring_size

    def add_node(self, node_id: str) -> int:
        """Add a physical node to the ring. Returns its position."""
        pos = self._hash(node_id)
        self.nodes[pos] = node_id
        bisect.insort(self.sorted_positions, pos)
        self.node_positions[node_id].append(pos)
        return pos

    def remove_node(self, node_id: str):
        """Remove a node from the ring."""
        for pos in self.node_positions.get(node_id, []):
            if pos in self.nodes:
                del self.nodes[pos]
                self.sorted_positions.remove(pos)
        del self.node_positions[node_id]

    def get_node(self, key: str) -> Optional[str]:
        """Find the node responsible for a key."""
        if not self.sorted_positions:
            return None

        pos = self._hash(key)
        # Find first node clockwise from pos
        idx = bisect.bisect_right(self.sorted_positions, pos)
        if idx >= len(self.sorted_positions):
            idx = 0  # Wrap around
        return self.nodes[self.sorted_positions[idx]]

    def get_node_and_replicas(self, key: str, num_replicas: int = 3) -> list[str]:
        """Find the primary and replica nodes for a key."""
        if not self.sorted_positions:
            return []

        pos = self._hash(key)
        idx = bisect.bisect_right(self.sorted_positions, pos)
        result = []
        seen = set()

        for i in range(len(self.sorted_positions)):
            actual_idx = (idx + i) % len(self.sorted_positions)
            node_id = self.nodes[self.sorted_positions[actual_idx]]
            if node_id not in seen:
                result.append(node_id)
                seen.add(node_id)
                if len(result) >= num_replicas:
                    break

        return result

    def key_distribution(self, keys: list[str]) -> dict[str, int]:
        """Analyze how keys are distributed across nodes."""
        distribution: dict[str, int] = defaultdict(int)
        for key in keys:
            node = self.get_node(key)
            if node:
                distribution[node] += 1
        return dict(distribution)


def demonstrate_consistent_hashing():
    """Demonstrate consistent hashing basics."""
    print("=== Consistent Hashing ===\n")

    ring = ConsistentHashRing(hash_bits=16)  # Smaller ring for demo

    # Add nodes
    nodes = ["server-A", "server-B", "server-C"]
    for node in nodes:
        pos = ring.add_node(node)
        print(f"  Added {node} at position {pos}")

    # Distribute keys
    keys = [f"key-{i}" for i in range(1000)]
    dist = ring.key_distribution(keys)
    print(f"\nKey distribution ({len(keys)} keys, {len(nodes)} nodes):")
    for node, count in sorted(dist.items()):
        pct = count / len(keys) * 100
        bar = "█" * int(pct / 2)
        print(f"  {node}: {count:4d} ({pct:5.1f}%) {bar}")

    # Show impact of adding/removing a node
    print(f"\nAdding server-D...")
    ring.add_node("server-D")
    new_dist = ring.key_distribution(keys)

    moved = 0
    for key in keys:
        old_node = None
        for node, count in dist.items():
            pass  # Simplified — in practice, track per-key
        new_node = ring.get_node(key)
        # Count movement by comparing distributions
    print(f"New distribution:")
    for node, count in sorted(new_dist.items()):
        pct = count / len(keys) * 100
        print(f"  {node}: {count:4d} ({pct:5.1f}%)")


demonstrate_consistent_hashing()
```

### 2.2 The Imbalance Problem

With just N physical nodes, the key distribution can be highly skewed. With 3 nodes, the ideal distribution is 33.3% each, but in practice it can range from 10% to 60% due to hash collisions.

---

## 3. Virtual Nodes

### 3.1 Solution: Multiple Tokens per Node

Each physical node maps to multiple positions (virtual nodes) on the ring:

```python
class VirtualNodeRing:
    """
    Consistent hashing with virtual nodes for improved balance.

    Each physical node is mapped to `vnodes_per_node` positions
    on the ring. This dramatically improves load distribution.
    """

    def __init__(self, vnodes_per_node: int = 150, hash_bits: int = 160):
        self.vnodes_per_node = vnodes_per_node
        self.hash_bits = hash_bits
        self.ring_size = 2 ** hash_bits
        self.ring: dict[int, str] = {}  # position → physical_node
        self.sorted_positions: list[int] = []
        self.physical_nodes: dict[str, list[int]] = defaultdict(list)

    def _hash(self, key: str) -> int:
        h = hashlib.sha1(key.encode()).hexdigest()
        return int(h, 16) % self.ring_size

    def add_node(self, node_id: str, weight: float = 1.0):
        """
        Add a physical node with virtual nodes.

        Weight allows heterogeneous hardware: a node with weight=2
        gets twice as many virtual nodes, thus twice the load.
        """
        num_vnodes = int(self.vnodes_per_node * weight)
        for i in range(num_vnodes):
            vnode_key = f"{node_id}#vnode{i}"
            pos = self._hash(vnode_key)
            self.ring[pos] = node_id
            bisect.insort(self.sorted_positions, pos)
            self.physical_nodes[node_id].append(pos)

    def remove_node(self, node_id: str) -> int:
        """Remove a physical node and all its virtual nodes."""
        positions = self.physical_nodes.pop(node_id, [])
        for pos in positions:
            if pos in self.ring:
                del self.ring[pos]
                self.sorted_positions.remove(pos)
        return len(positions)

    def get_node(self, key: str) -> Optional[str]:
        """Find the physical node responsible for a key."""
        if not self.sorted_positions:
            return None
        pos = self._hash(key)
        idx = bisect.bisect_right(self.sorted_positions, pos)
        if idx >= len(self.sorted_positions):
            idx = 0
        return self.ring[self.sorted_positions[idx]]

    def analyze_balance(self, num_keys: int = 10000) -> dict:
        """Analyze load balance across physical nodes."""
        counts: dict[str, int] = defaultdict(int)
        for i in range(num_keys):
            key = f"test-key-{i}"
            node = self.get_node(key)
            if node:
                counts[node] += 1

        values = list(counts.values())
        if not values:
            return {}

        mean = sum(values) / len(values)
        max_val = max(values)
        min_val = min(values)
        std_dev = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5

        return {
            "node_counts": dict(counts),
            "mean": mean,
            "max": max_val,
            "min": min_val,
            "std_dev": round(std_dev, 1),
            "imbalance_ratio": round(max_val / max(min_val, 1), 2),
        }


def compare_vnode_counts():
    """Compare load balance with different virtual node counts."""
    print("=== Virtual Nodes: Balance Analysis ===\n")

    nodes = [f"node-{i}" for i in range(5)]
    num_keys = 10000

    for vnodes in [1, 10, 50, 150, 500]:
        ring = VirtualNodeRing(vnodes_per_node=vnodes)
        for node in nodes:
            ring.add_node(node)

        stats = ring.analyze_balance(num_keys)
        print(f"Virtual nodes per physical node: {vnodes}")
        print(f"  Ideal: {num_keys // len(nodes)} keys per node")
        print(f"  Actual: min={stats['min']}, max={stats['max']}, "
              f"std_dev={stats['std_dev']}")
        print(f"  Imbalance ratio: {stats['imbalance_ratio']}x")
        print()


compare_vnode_counts()
```

### 3.2 Weighted Virtual Nodes

For heterogeneous clusters where nodes have different capacities:

```python
def demonstrate_weighted_vnodes():
    """Demonstrate weighted virtual nodes for heterogeneous hardware."""
    print("=== Weighted Virtual Nodes ===\n")

    ring = VirtualNodeRing(vnodes_per_node=100)

    # Different weights for different hardware
    ring.add_node("small-1", weight=1.0)   # 100 vnodes
    ring.add_node("small-2", weight=1.0)   # 100 vnodes
    ring.add_node("large-1", weight=3.0)   # 300 vnodes
    ring.add_node("large-2", weight=2.0)   # 200 vnodes

    stats = ring.analyze_balance(10000)

    print("Node capacities and actual load:")
    for node, count in sorted(stats["node_counts"].items()):
        weight = {"small-1": 1, "small-2": 1, "large-1": 3, "large-2": 2}[node]
        expected_pct = weight / 7 * 100  # Total weight = 7
        actual_pct = count / 10000 * 100
        print(f"  {node} (weight={weight}): "
              f"expected={expected_pct:.1f}%, actual={actual_pct:.1f}%, "
              f"keys={count}")


demonstrate_weighted_vnodes()
```

---

## 4. Chord Protocol

### 4.1 Chord Overview

Chord provides O(log N) lookup with O(log N) state per node using a finger table:

```python
class ChordNode:
    """
    Implementation of the Chord distributed hash table protocol.

    Each node maintains a finger table for O(log N) routing,
    a successor list for fault tolerance, and a predecessor
    pointer for ring maintenance.
    """

    M = 8  # Key space: 2^M = 256 identifiers (small for demo)

    def __init__(self, node_id: int):
        self.node_id = node_id
        self.finger_table: list[Optional[int]] = [None] * self.M
        self.predecessor: Optional[int] = None
        self.successor_list: list[int] = []  # For fault tolerance
        self.data: dict[int, str] = {}  # key → value storage
        self.lookup_hops: int = 0

    @staticmethod
    def in_range(x: int, start: int, end: int, inclusive_end: bool = False) -> bool:
        """Check if x is in the range (start, end] on the circular ring."""
        ring_size = 2 ** ChordNode.M
        x = x % ring_size
        start = start % ring_size
        end = end % ring_size

        if start < end:
            if inclusive_end:
                return start < x <= end
            else:
                return start < x < end
        else:  # Wrap around
            if inclusive_end:
                return x > start or x <= end
            else:
                return x > start or x < end

    def successor(self) -> Optional[int]:
        """Return the immediate successor."""
        return self.finger_table[0]

    def closest_preceding_finger(self, key_id: int) -> int:
        """
        Find the closest preceding finger for a key.

        This is the core of Chord's O(log N) lookup: each hop
        covers at least half the remaining distance to the target.
        """
        for i in range(self.M - 1, -1, -1):
            finger = self.finger_table[i]
            if finger is not None and self.in_range(finger, self.node_id, key_id):
                return finger
        return self.node_id

    def find_successor(self, key_id: int, network: dict) -> Tuple[int, int]:
        """
        Find the node responsible for a key.

        Returns (responsible_node, num_hops).
        """
        hops = 0
        current = self.node_id

        for _ in range(self.M + 5):  # Max iterations as safety
            node = network.get(current)
            if node is None:
                return current, hops

            successor = node.successor()
            if successor is None:
                return current, hops

            if self.in_range(key_id, current, successor, inclusive_end=True):
                return successor, hops

            next_node = node.closest_preceding_finger(key_id)
            if next_node == current:
                return successor, hops
            current = next_node
            hops += 1

        return current, hops


class ChordNetwork:
    """
    A simulated Chord network for testing.

    Handles node creation, finger table initialization,
    and key lookups.
    """

    def __init__(self, m: int = 8):
        ChordNode.M = m
        self.ring_size = 2 ** m
        self.nodes: dict[int, ChordNode] = {}

    def add_node(self, node_id: int) -> ChordNode:
        """Add a node to the Chord network."""
        node = ChordNode(node_id)
        self.nodes[node_id] = node
        return node

    def build_finger_tables(self):
        """
        Build finger tables for all nodes.

        finger[i] = successor of (node_id + 2^i) mod 2^M

        In a real Chord network, this is done incrementally via
        the stabilize() protocol. Here we build them all at once
        for simplicity.
        """
        sorted_ids = sorted(self.nodes.keys())
        if not sorted_ids:
            return

        for node_id, node in self.nodes.items():
            # Set predecessor
            idx = sorted_ids.index(node_id)
            node.predecessor = sorted_ids[(idx - 1) % len(sorted_ids)]

            # Build finger table
            for i in range(ChordNode.M):
                target = (node_id + 2 ** i) % self.ring_size

                # Find successor of target
                found = False
                for sid in sorted_ids:
                    if sid >= target:
                        node.finger_table[i] = sid
                        found = True
                        break
                if not found:
                    node.finger_table[i] = sorted_ids[0]  # Wrap around

            # Build successor list (next 3 successors)
            node.successor_list = []
            for j in range(1, min(4, len(sorted_ids))):
                succ = sorted_ids[(idx + j) % len(sorted_ids)]
                node.successor_list.append(succ)

    def lookup(self, origin: int, key: int) -> Tuple[int, int]:
        """Perform a key lookup starting from origin node."""
        if origin not in self.nodes:
            raise ValueError(f"Node {origin} not in network")
        return self.nodes[origin].find_successor(key, self.nodes)

    def store(self, key: int, value: str, origin: int):
        """Store a key-value pair via DHT routing."""
        responsible, hops = self.lookup(origin, key)
        if responsible in self.nodes:
            self.nodes[responsible].data[key] = value

    def analyze_lookups(self, num_lookups: int = 1000) -> dict:
        """Analyze lookup hop counts."""
        if not self.nodes:
            return {}

        node_ids = list(self.nodes.keys())
        hop_counts = []

        for _ in range(num_lookups):
            origin = random.choice(node_ids)
            key = random.randint(0, self.ring_size - 1)
            _, hops = self.lookup(origin, key)
            hop_counts.append(hops)

        return {
            "num_lookups": num_lookups,
            "num_nodes": len(self.nodes),
            "avg_hops": round(sum(hop_counts) / len(hop_counts), 2),
            "max_hops": max(hop_counts),
            "min_hops": min(hop_counts),
            "theoretical_max": ChordNode.M,  # O(log N)
        }


def demonstrate_chord():
    """Demonstrate the Chord DHT protocol."""
    print("=== Chord Protocol ===\n")

    network = ChordNetwork(m=8)

    # Add nodes at various positions
    node_ids = sorted(random.sample(range(256), 16))
    for nid in node_ids:
        network.add_node(nid)

    network.build_finger_tables()

    # Show a node's finger table
    sample_node = network.nodes[node_ids[0]]
    print(f"Node {sample_node.node_id} finger table:")
    for i, finger in enumerate(sample_node.finger_table):
        target = (sample_node.node_id + 2 ** i) % 256
        print(f"  finger[{i}]: start={target:3d}, successor={finger}")

    # Lookup analysis
    stats = network.analyze_lookups(1000)
    print(f"\nLookup analysis ({stats['num_nodes']} nodes, "
          f"{stats['num_lookups']} lookups):")
    print(f"  Average hops: {stats['avg_hops']}")
    print(f"  Max hops: {stats['max_hops']}")
    print(f"  Theoretical O(log N) = {stats['theoretical_max']}")
    print(f"  log2({stats['num_nodes']}) = {stats['num_nodes']:.0f} → "
          f"{len(bin(stats['num_nodes'])) - 2} bits")


demonstrate_chord()
```

### 4.2 Chord Stabilization

```python
def demonstrate_chord_stabilization():
    """Demonstrate how Chord handles node joins via stabilization."""
    print("=== Chord Stabilization ===\n")

    network = ChordNetwork(m=8)

    # Initial network: 4 nodes
    initial = [0, 64, 128, 192]
    for nid in initial:
        network.add_node(nid)
    network.build_finger_tables()

    stats_before = network.analyze_lookups(500)
    print(f"Before join: {len(initial)} nodes, avg hops={stats_before['avg_hops']}")

    # Join new nodes
    new_nodes = [32, 96, 160, 224]
    for nid in new_nodes:
        network.add_node(nid)
    network.build_finger_tables()

    stats_after = network.analyze_lookups(500)
    print(f"After join: {len(initial) + len(new_nodes)} nodes, "
          f"avg hops={stats_after['avg_hops']}")

    # Node departure
    network.nodes.pop(64)
    network.build_finger_tables()

    stats_depart = network.analyze_lookups(500)
    print(f"After departure: {len(network.nodes)} nodes, "
          f"avg hops={stats_depart['avg_hops']}")


demonstrate_chord_stabilization()
```

---

## 5. Kademlia Protocol

### 5.1 XOR Distance Metric

Kademlia's key innovation is using XOR as the distance metric, which is a valid metric (symmetric, satisfies triangle inequality) and enables efficient routing:

```python
class KademliaNode:
    """
    Implementation of the Kademlia DHT protocol.

    Key features:
    - XOR-based distance metric
    - k-buckets for routing (k contacts per distance range)
    - Parallel iterative lookups
    - Lazy routing table refresh
    """

    K = 20   # Replication parameter (bucket size)
    ALPHA = 3  # Parallelism parameter
    B = 160    # Key space bits

    def __init__(self, node_id: int):
        self.node_id = node_id
        # k-buckets: one per bit (bucket[i] holds nodes with distance 2^i to 2^(i+1))
        self.buckets: list[list[int]] = [[] for _ in range(self.B)]
        self.data: dict[int, str] = {}
        self.lookup_messages: int = 0

    @staticmethod
    def distance(a: int, b: int) -> int:
        """XOR distance between two node IDs."""
        return a ^ b

    @staticmethod
    def bucket_index(distance: int) -> int:
        """Determine which k-bucket a distance falls into."""
        if distance == 0:
            return 0
        return distance.bit_length() - 1

    def update_routing_table(self, other_id: int):
        """
        Update routing table with a newly discovered node.

        If the appropriate k-bucket is not full, add the node.
        If full, check if the least-recently-seen node is still
        alive; if not, replace it.
        """
        if other_id == self.node_id:
            return

        dist = self.distance(self.node_id, other_id)
        bucket_idx = self.bucket_index(dist)

        if bucket_idx >= len(self.buckets):
            return

        bucket = self.buckets[bucket_idx]

        if other_id in bucket:
            # Move to end (most recently seen)
            bucket.remove(other_id)
            bucket.append(other_id)
        elif len(bucket) < self.K:
            bucket.append(other_id)
        # else: bucket full, would ping least-recently-seen

    def find_closest(self, target_id: int, count: int = None) -> list[int]:
        """
        Find the closest nodes to a target in our routing table.
        """
        if count is None:
            count = self.K

        all_nodes = []
        for bucket in self.buckets:
            all_nodes.extend(bucket)

        all_nodes.sort(key=lambda n: self.distance(n, target_id))
        return all_nodes[:count]


class KademliaNetwork:
    """Simulated Kademlia network."""

    def __init__(self, key_bits: int = 16):
        KademliaNode.B = key_bits
        self.key_bits = key_bits
        self.key_space = 2 ** key_bits
        self.nodes: dict[int, KademliaNode] = {}

    def add_node(self, node_id: int) -> KademliaNode:
        """Add a node and update routing tables."""
        node = KademliaNode(node_id)
        self.nodes[node_id] = node

        # Bootstrap: update routing tables bidirectionally
        for existing_id, existing_node in self.nodes.items():
            if existing_id != node_id:
                node.update_routing_table(existing_id)
                existing_node.update_routing_table(node_id)

        return node

    def iterative_find_node(self, origin: int, target: int) -> Tuple[int, int]:
        """
        Perform an iterative FIND_NODE lookup.

        Returns (closest_node, messages_sent).
        """
        if origin not in self.nodes:
            return origin, 0

        messages = 0
        queried: set[int] = set()
        closest = self.nodes[origin].find_closest(target, KademliaNode.K)

        for _ in range(20):  # Max iterations
            # Select ALPHA unqueried nodes closest to target
            to_query = [
                n for n in closest if n not in queried
            ][:KademliaNode.ALPHA]

            if not to_query:
                break

            new_contacts = []
            for node_id in to_query:
                queried.add(node_id)
                messages += 1

                if node_id in self.nodes:
                    found = self.nodes[node_id].find_closest(target, KademliaNode.K)
                    new_contacts.extend(found)

            # Merge and keep K closest
            all_contacts = list(set(closest + new_contacts))
            all_contacts.sort(key=lambda n: KademliaNode.distance(n, target))
            new_closest = all_contacts[:KademliaNode.K]

            if new_closest == closest:
                break  # No improvement
            closest = new_closest

        result = closest[0] if closest else origin
        return result, messages

    def analyze_lookups(self, num_lookups: int = 500) -> dict:
        """Analyze Kademlia lookup performance."""
        node_ids = list(self.nodes.keys())
        if not node_ids:
            return {}

        hop_counts = []
        message_counts = []

        for _ in range(num_lookups):
            origin = random.choice(node_ids)
            target = random.randint(0, self.key_space - 1)
            _, messages = self.iterative_find_node(origin, target)
            message_counts.append(messages)

        return {
            "num_lookups": num_lookups,
            "num_nodes": len(self.nodes),
            "avg_messages": round(sum(message_counts) / len(message_counts), 2),
            "max_messages": max(message_counts),
            "theoretical": f"O(log({len(self.nodes)}))",
        }


def demonstrate_kademlia():
    """Demonstrate the Kademlia DHT protocol."""
    print("=== Kademlia Protocol ===\n")

    network = KademliaNetwork(key_bits=16)

    # Add nodes
    num_nodes = 100
    for _ in range(num_nodes):
        node_id = random.randint(0, 2 ** 16 - 1)
        while node_id in network.nodes:
            node_id = random.randint(0, 2 ** 16 - 1)
        network.add_node(node_id)

    # Show XOR distance properties
    ids = list(network.nodes.keys())[:3]
    print("XOR distance properties:")
    a, b, c = ids[0], ids[1], ids[2]
    print(f"  d({a}, {b}) = {KademliaNode.distance(a, b)}")
    print(f"  d({b}, {a}) = {KademliaNode.distance(b, a)} (symmetric)")
    print(f"  d({a}, {a}) = {KademliaNode.distance(a, a)} (identity)")
    d_ab = KademliaNode.distance(a, b)
    d_bc = KademliaNode.distance(b, c)
    d_ac = KademliaNode.distance(a, c)
    print(f"  d(a,b) + d(b,c) = {d_ab + d_bc} >= d(a,c) = {d_ac} "
          f"(triangle: {'✓' if d_ab + d_bc >= d_ac else '✗'})")

    # Lookup performance
    stats = network.analyze_lookups(500)
    print(f"\nLookup performance ({stats['num_nodes']} nodes):")
    print(f"  Avg messages per lookup: {stats['avg_messages']}")
    print(f"  Max messages: {stats['max_messages']}")
    print(f"  Theoretical: {stats['theoretical']}")


demonstrate_kademlia()
```

---

## 6. Replication on DHTs

### 6.1 Successor-Based Replication

```python
class ReplicatedDHT:
    """
    DHT with replication for fault tolerance.

    Each key is stored on N successor nodes on the ring.
    Reads and writes use quorum protocols.
    """

    def __init__(self, replication_factor: int = 3):
        self.N = replication_factor  # Total replicas
        self.W = 2  # Write quorum
        self.R = 2  # Read quorum
        self.ring = VirtualNodeRing(vnodes_per_node=50)
        self.node_data: dict[str, dict] = defaultdict(dict)  # node → {key: (value, version)}
        self.version_counter: int = 0

    def put(self, key: str, value: str) -> dict:
        """Write with quorum."""
        replicas = self.ring.get_node_and_replicas(key, self.N)
        if len(replicas) < self.W:
            return {"ok": False, "error": "Not enough replicas"}

        self.version_counter += 1
        version = self.version_counter

        acks = 0
        for node in replicas:
            self.node_data[node][key] = {"value": value, "version": version}
            acks += 1

        return {
            "ok": acks >= self.W,
            "replicas": replicas,
            "acks": acks,
            "version": version,
        }

    def get(self, key: str) -> dict:
        """Read with quorum and read-repair."""
        replicas = self.ring.get_node_and_replicas(key, self.N)

        responses = []
        for node in replicas:
            if key in self.node_data[node]:
                responses.append({
                    "node": node,
                    **self.node_data[node][key],
                })

        if len(responses) < self.R:
            return {"ok": False, "error": "Not enough responses"}

        # Return highest version
        best = max(responses, key=lambda r: r["version"])

        # Read-repair: update stale replicas
        for resp in responses:
            if resp["version"] < best["version"]:
                node = resp["node"]
                self.node_data[node][key] = {
                    "value": best["value"],
                    "version": best["version"],
                }

        return {"ok": True, "value": best["value"], "version": best["version"]}


def demonstrate_replicated_dht():
    """Demonstrate a replicated DHT with quorum reads/writes."""
    print("=== Replicated DHT ===\n")

    dht = ReplicatedDHT(replication_factor=3)

    # Add nodes
    for i in range(10):
        dht.ring.add_node(f"node-{i}")

    # Write and read
    result = dht.put("user:alice", "{'name': 'Alice', 'age': 30}")
    print(f"PUT user:alice → {result}")

    result = dht.get("user:alice")
    print(f"GET user:alice → {result}")

    # Quorum analysis
    print(f"\nQuorum configuration:")
    print(f"  N={dht.N}, W={dht.W}, R={dht.R}")
    print(f"  W + R = {dht.W + dht.R} > N = {dht.N}: "
          f"{'Strong consistency' if dht.W + dht.R > dht.N else 'Eventual consistency'}")


demonstrate_replicated_dht()
```

---

## 7. Churn Handling

### 7.1 Node Join/Leave Impact

```python
def analyze_churn_impact():
    """Analyze the impact of node churn on DHT performance."""
    print("=== Churn Impact Analysis ===\n")

    ring = VirtualNodeRing(vnodes_per_node=100)
    num_keys = 10000

    # Store initial key assignments
    initial_nodes = [f"node-{i}" for i in range(10)]
    for node in initial_nodes:
        ring.add_node(node)

    initial_assignment = {}
    for i in range(num_keys):
        key = f"key-{i}"
        initial_assignment[key] = ring.get_node(key)

    # Simulate churn: remove 2 nodes, add 3 new ones
    ring.remove_node("node-3")
    ring.remove_node("node-7")
    ring.add_node("node-10")
    ring.add_node("node-11")
    ring.add_node("node-12")

    # Count key movements
    moved = 0
    for i in range(num_keys):
        key = f"key-{i}"
        new_node = ring.get_node(key)
        if new_node != initial_assignment[key]:
            moved += 1

    pct_moved = moved / num_keys * 100
    print(f"Churn event: removed 2 nodes, added 3 nodes")
    print(f"  Keys moved: {moved}/{num_keys} ({pct_moved:.1f}%)")
    print(f"  Ideal (minimal disruption): ~{num_keys * 2 / 10:.0f} keys "
          f"({2 / 10 * 100:.0f}%)")
    print(f"  Overhead: {pct_moved - (2 / 10 * 100):.1f}% extra movement")


analyze_churn_impact()
```

---

## 8. Load Balancing

### 8.1 Bounded Load Consistent Hashing

Google's "bounded load" extension ensures no node gets more than (1 + epsilon) * average_load:

```python
class BoundedLoadHashRing:
    """
    Consistent hashing with bounded load (Google, 2017).

    Ensures no node receives more than (1 + epsilon) times the
    average load. When a node is overloaded, the key is assigned
    to the next node on the ring that is not overloaded.
    """

    def __init__(self, epsilon: float = 0.25, vnodes: int = 100):
        self.epsilon = epsilon
        self.ring = VirtualNodeRing(vnodes_per_node=vnodes)
        self.node_load: dict[str, int] = defaultdict(int)
        self.total_keys: int = 0

    def add_node(self, node_id: str):
        self.ring.add_node(node_id)
        self.node_load[node_id] = 0

    def _max_load(self) -> int:
        """Calculate the maximum allowed load per node."""
        num_nodes = len(self.node_load)
        if num_nodes == 0:
            return 0
        avg_load = max(1, self.total_keys / num_nodes)
        return int(avg_load * (1 + self.epsilon)) + 1

    def assign(self, key: str) -> str:
        """Assign a key to a node, respecting load bounds."""
        max_load = self._max_load()

        # Try nodes clockwise from the hash position
        candidates = self.ring.get_node_and_replicas(key, len(self.node_load))

        for node in candidates:
            if self.node_load[node] < max_load:
                self.node_load[node] += 1
                self.total_keys += 1
                return node

        # Fallback: all nodes overloaded (shouldn't happen with correct epsilon)
        first = candidates[0] if candidates else list(self.node_load.keys())[0]
        self.node_load[first] += 1
        self.total_keys += 1
        return first

    def stats(self) -> dict:
        loads = list(self.node_load.values())
        if not loads:
            return {}
        return {
            "max_load": max(loads),
            "min_load": min(loads),
            "avg_load": sum(loads) / len(loads),
            "max_allowed": self._max_load(),
            "imbalance": max(loads) / max(1, min(loads)),
        }


def demonstrate_bounded_load():
    """Compare standard vs bounded load consistent hashing."""
    print("=== Bounded Load Consistent Hashing ===\n")

    # Standard consistent hashing
    standard = VirtualNodeRing(vnodes_per_node=100)
    standard_counts: dict[str, int] = defaultdict(int)

    # Bounded load
    bounded = BoundedLoadHashRing(epsilon=0.25, vnodes=100)

    nodes = [f"node-{i}" for i in range(8)]
    for node in nodes:
        standard.add_node(node)
        bounded.add_node(node)

    num_keys = 10000
    for i in range(num_keys):
        key = f"key-{i}"
        standard_node = standard.get_node(key)
        standard_counts[standard_node] += 1
        bounded.assign(key)

    # Compare
    std_values = list(standard_counts.values())
    print(f"Standard consistent hashing:")
    print(f"  Max load: {max(std_values)}")
    print(f"  Min load: {min(std_values)}")
    print(f"  Imbalance: {max(std_values)/min(std_values):.2f}x")

    b_stats = bounded.stats()
    print(f"\nBounded load (epsilon={bounded.epsilon}):")
    print(f"  Max load: {b_stats['max_load']}")
    print(f"  Min load: {b_stats['min_load']}")
    print(f"  Max allowed: {b_stats['max_allowed']}")
    print(f"  Imbalance: {b_stats['imbalance']:.2f}x")


demonstrate_bounded_load()
```

---

## 9. Real-World DHT Systems

### 9.1 Comparison

| System | Protocol | Distance | Lookup | Used In |
|--------|----------|----------|--------|---------|
| **Chord** | Ring + finger table | Clockwise | O(log N) | Research |
| **Kademlia** | k-buckets, XOR | XOR | O(log N) | BitTorrent, Ethereum |
| **Pastry** | Prefix routing | Shared prefix | O(log N) | Microsoft (Halo) |
| **CAN** | d-dimensional space | Cartesian | O(d·N^(1/d)) | Research |
| **Dynamo** | Consistent hashing | Ring position | O(1)* | Amazon (DynamoDB) |

*Dynamo uses consistent hashing with full membership knowledge (all nodes know all others), so lookups are O(1) hops but require O(N) state.

### 9.2 Amazon Dynamo vs Academic DHTs

```python
def compare_dht_approaches():
    """Compare academic DHTs with production systems like Dynamo."""
    print("=== Academic DHTs vs Production Systems ===\n")

    comparisons = {
        "Membership Knowledge": {
            "Chord/Kademlia": "Partial (O(log N) state)",
            "Dynamo": "Full (O(N) state)",
        },
        "Lookup Hops": {
            "Chord/Kademlia": "O(log N) network hops",
            "Dynamo": "O(1) — direct routing",
        },
        "Consistency": {
            "Chord/Kademlia": "Eventual (basic)",
            "Dynamo": "Tunable (W + R > N for strong)",
        },
        "Failure Handling": {
            "Chord/Kademlia": "Successor lists, stabilization",
            "Dynamo": "Sloppy quorum, hinted handoff",
        },
        "Scale": {
            "Chord/Kademlia": "Millions of nodes (P2P)",
            "Dynamo": "Hundreds of nodes (datacenter)",
        },
    }

    for aspect, values in comparisons.items():
        print(f"{aspect}:")
        for system, desc in values.items():
            print(f"  {system:20s}: {desc}")
        print()


compare_dht_approaches()
```

---

## 10. Summary and Key Takeaways

### DHT Design Space

> **DHT DESIGN DIMENSIONS**
>
> Topology:     Ring (Chord) │ Tree (Kademlia) │ Hypercube (CAN)
> Distance:     Clockwise    │ XOR             │ Cartesian
> Routing:      Finger table │ k-buckets       │ Neighbor table
> State/node:   O(log N)     │ O(log N)        │ O(d)
> Lookup:       O(log N)     │ O(log N)        │ O(d·N^(1/d))
> Replication:  Successor    │ Closest nodes    │ Neighbors

### Key Principles

1. **Consistent hashing minimizes disruption**: Only K/N keys move when a node joins/leaves.
2. **Virtual nodes improve balance**: 100+ vnodes per physical node achieves <10% imbalance.
3. **Finger tables enable O(log N) routing**: Each hop covers half the remaining distance.
4. **XOR is an elegant distance metric**: Symmetric, satisfies triangle inequality, enables efficient k-bucket organization.
5. **Production systems trade generality for performance**: Dynamo uses O(N) state for O(1) lookup within a datacenter.

---

## 11. Practice Problems

### Problem 1: Consistent Hashing Analysis

With 10 physical nodes and 200 virtual nodes each, calculate the expected standard deviation of key distribution for 100,000 keys. How does this compare to 1 virtual node per physical node?

### Problem 2: Chord Finger Table

For a Chord ring with m=6 (64 positions) and nodes at positions {1, 8, 14, 21, 32, 38, 42, 51}, construct the complete finger table for node 14.

### Problem 3: Kademlia Routing

In a Kademlia network with B=8, ALPHA=3, K=4:
- What is the maximum number of messages needed for a single lookup?
- How many k-buckets can a single node have?
- If a bucket is full and a new contact is discovered, describe the eviction strategy.

### Problem 4: Implementation Challenge

Implement a `ChordNode.stabilize()` method that:
1. Asks its successor for its predecessor
2. If the predecessor is between the node and its successor, adopts it as new successor
3. Notifies the successor of its existence

### Problem 5: Bounded Load Analysis

Prove that with epsilon = 0.25 and consistent hashing, no node receives more than 1.25x the average load. What happens to lookup efficiency when keys are redirected due to load bounds?

---

## 12. References

1. Stoica, I. et al. (2001). "Chord: A Scalable Peer-to-peer Lookup Service for Internet Applications." *ACM SIGCOMM*.
2. Maymounkov, P. & Mazieres, D. (2002). "Kademlia: A Peer-to-peer Information System Based on the XOR Metric." *IPTPS*.
3. Rowstron, A. & Druschel, P. (2001). "Pastry: Scalable, Decentralized Object Location, and Routing for Large-Scale Peer-to-Peer Systems." *Middleware*.
4. DeCandia, G. et al. (2007). "Dynamo: Amazon's Highly Available Key-value Store." *SOSP*.
5. Karger, D. et al. (1997). "Consistent Hashing and Random Trees." *STOC*.
6. Mirrokni, V. et al. (2018). "Consistent Hashing with Bounded Loads." arXiv:1608.01350.
7. Kleppmann, M. (2017). *Designing Data-Intensive Applications*, Ch. 6. O'Reilly Media.

---

[Next: Lesson 21 — Gossip Protocols](./21_Gossip_Protocols.md)
