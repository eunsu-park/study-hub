"""
Exercises for Lesson 11: Partitioning and Sharding
Topic: Distributed_Systems

Solutions to practice problems from the lesson.
"""

import hashlib
import math
import random
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from bisect import bisect_right, insort


# === Exercise 1: Consistent Hashing with Bounded Load ===
# Problem: Implement consistent hashing with virtual nodes (vnodes)
# and the bounded-load extension (from Vimeo/Google research).
# The bounded-load variant caps the maximum load on any node to
# (1 + epsilon) * average_load.

class ConsistentHashRing:
    """
    Consistent hash ring with virtual nodes and bounded-load extension.
    """

    def __init__(self, epsilon: float = 0.25):
        self.epsilon = epsilon
        self.ring: List[int] = []  # sorted positions
        self.ring_map: Dict[int, str] = {}  # position -> node_id
        self.node_vnodes: Dict[str, List[int]] = defaultdict(list)
        self.node_load: Dict[str, int] = defaultdict(int)
        self.total_keys = 0

    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def add_node(self, node_id: str, num_vnodes: int = 150):
        """Add a node with virtual nodes to the ring."""
        for i in range(num_vnodes):
            vnode_key = f"{node_id}#vnode{i}"
            pos = self._hash(vnode_key)
            insort(self.ring, pos)
            self.ring_map[pos] = node_id
            self.node_vnodes[node_id].append(pos)

    def remove_node(self, node_id: str):
        """Remove a node and all its virtual nodes."""
        for pos in self.node_vnodes[node_id]:
            idx = self.ring.index(pos)
            self.ring.pop(idx)
            del self.ring_map[pos]
        del self.node_vnodes[node_id]
        self.node_load.pop(node_id, None)

    def get_node(self, key: str) -> str:
        """Get the node responsible for a key (standard consistent hashing)."""
        if not self.ring:
            raise ValueError("No nodes in the ring")
        h = self._hash(key)
        idx = bisect_right(self.ring, h)
        if idx == len(self.ring):
            idx = 0
        return self.ring_map[self.ring[idx]]

    def get_node_bounded(self, key: str) -> str:
        """
        Get the node for a key with bounded-load constraint.
        If the primary node is overloaded, skip to the next node.
        """
        if not self.ring:
            raise ValueError("No nodes in the ring")

        num_nodes = len(self.node_vnodes)
        avg_load = (self.total_keys + 1) / num_nodes if num_nodes > 0 else 1
        max_load = math.ceil((1 + self.epsilon) * avg_load)

        h = self._hash(key)
        idx = bisect_right(self.ring, h)

        visited = set()
        while len(visited) < num_nodes:
            if idx >= len(self.ring):
                idx = 0
            node = self.ring_map[self.ring[idx]]
            if node not in visited:
                visited.add(node)
                if self.node_load[node] < max_load:
                    self.node_load[node] += 1
                    self.total_keys += 1
                    return node
            idx += 1

        # Fallback: all nodes at capacity, pick the primary
        idx = bisect_right(self.ring, h) % len(self.ring)
        node = self.ring_map[self.ring[idx]]
        self.node_load[node] += 1
        self.total_keys += 1
        return node


def exercise_1():
    """
    Demonstrate consistent hashing with bounded-load extension.
    """
    print("=== Exercise 1: Consistent Hashing with Bounded Load ===\n")

    ring = ConsistentHashRing(epsilon=0.25)
    for i in range(4):
        ring.add_node(f"Node{i}", num_vnodes=100)

    # Assign 1000 keys with standard hashing
    standard_dist = defaultdict(int)
    for i in range(1000):
        node = ring.get_node(f"key_{i}")
        standard_dist[node] += 1

    print("Standard consistent hashing distribution (1000 keys):")
    for node in sorted(standard_dist):
        bar = "#" * (standard_dist[node] // 10)
        print(f"  {node}: {standard_dist[node]:4d} {bar}")

    max_std = max(standard_dist.values())
    min_std = min(standard_dist.values())
    print(f"  Max/Min ratio: {max_std/min_std:.2f}")

    # Bounded-load hashing
    ring2 = ConsistentHashRing(epsilon=0.25)
    for i in range(4):
        ring2.add_node(f"Node{i}", num_vnodes=100)

    for i in range(1000):
        ring2.get_node_bounded(f"key_{i}")

    print(f"\nBounded-load hashing (epsilon=0.25):")
    for node in sorted(ring2.node_load):
        bar = "#" * (ring2.node_load[node] // 10)
        print(f"  {node}: {ring2.node_load[node]:4d} {bar}")

    max_bl = max(ring2.node_load.values())
    min_bl = min(ring2.node_load.values())
    print(f"  Max/Min ratio: {max_bl/min_bl:.2f}")
    print(f"  Max load cap: {math.ceil(1.25 * 250)} (1.25 * avg)")
    print()


# === Exercise 2: Key Distribution Uniformity ===
# Problem: Measure how key distribution uniformity varies with
# different numbers of virtual nodes (vnodes). Show that more vnodes
# lead to more uniform distribution.

def measure_uniformity(
    num_nodes: int, num_vnodes: int, num_keys: int
) -> Tuple[float, float]:
    """
    Measure key distribution uniformity.
    Returns (std_dev, max_min_ratio).
    """
    ring = ConsistentHashRing()
    for i in range(num_nodes):
        ring.add_node(f"N{i}", num_vnodes=num_vnodes)

    dist = defaultdict(int)
    for i in range(num_keys):
        node = ring.get_node(f"key_{i}")
        dist[node] += 1

    counts = list(dist.values())
    avg = sum(counts) / len(counts)
    variance = sum((c - avg) ** 2 for c in counts) / len(counts)
    std_dev = math.sqrt(variance)
    ratio = max(counts) / min(counts) if min(counts) > 0 else float("inf")

    return (std_dev, ratio)


def exercise_2():
    """
    Compare key distribution uniformity across different vnode counts.
    """
    print("=== Exercise 2: Key Distribution vs VNode Count ===\n")

    num_nodes = 5
    num_keys = 10000
    vnode_counts = [1, 10, 50, 100, 200, 500]

    print(f"Nodes: {num_nodes}, Keys: {num_keys}")
    print(f"Expected per node: {num_keys // num_nodes}\n")
    print(f"{'VNodes':>8s} {'StdDev':>10s} {'Max/Min':>10s} {'Uniformity':>12s}")
    print("-" * 45)

    for vnodes in vnode_counts:
        std_dev, ratio = measure_uniformity(num_nodes, vnodes, num_keys)
        # Uniformity score: 1 / ratio (closer to 1.0 is better)
        uniformity = 1.0 / ratio
        print(f"{vnodes:8d} {std_dev:10.1f} {ratio:10.2f} {uniformity:12.3f}")

    print(
        "\nMore vnodes -> lower std_dev, lower max/min ratio -> "
        "better uniformity."
    )
    print("Typical recommendation: 100-200 vnodes per physical node.")
    print()


# === Exercise 3: Range-Based Partitioning with Auto-Split ===
# Problem: Implement range-based partitioning where each partition
# covers a key range. When a partition exceeds a threshold size,
# it automatically splits into two partitions.

class RangePartition:
    """A partition covering a key range [start, end)."""

    def __init__(self, start: str, end: str, partition_id: int):
        self.start = start
        self.end = end
        self.partition_id = partition_id
        self.data: Dict[str, int] = {}

    def contains(self, key: str) -> bool:
        """Check if a key falls in this partition's range."""
        return self.start <= key < self.end

    def size(self) -> int:
        return len(self.data)


class RangePartitionedStore:
    """
    Range-partitioned key-value store with automatic splitting.
    """

    def __init__(self, split_threshold: int = 10):
        self.split_threshold = split_threshold
        self.next_partition_id = 1
        # Start with a single partition covering all keys
        self.partitions = [
            RangePartition("\x00", "\xff" * 10, self._next_id())
        ]
        self.split_count = 0

    def _next_id(self) -> int:
        pid = self.next_partition_id
        self.next_partition_id += 1
        return pid

    def _find_partition(self, key: str) -> Optional[RangePartition]:
        for p in self.partitions:
            if p.contains(key):
                return p
        return None

    def put(self, key: str, value: int):
        """Write a key-value pair, auto-splitting if needed."""
        partition = self._find_partition(key)
        if partition is None:
            raise KeyError(f"No partition found for key '{key}'")

        partition.data[key] = value

        # Check if split is needed
        if partition.size() > self.split_threshold:
            self._split(partition)

    def get(self, key: str) -> Optional[int]:
        partition = self._find_partition(key)
        if partition:
            return partition.data.get(key)
        return None

    def _split(self, partition: RangePartition):
        """Split a partition into two at the median key."""
        sorted_keys = sorted(partition.data.keys())
        mid = len(sorted_keys) // 2
        split_key = sorted_keys[mid]

        # Create two new partitions
        left = RangePartition(partition.start, split_key, self._next_id())
        right = RangePartition(split_key, partition.end, self._next_id())

        # Redistribute data
        for key, value in partition.data.items():
            if key < split_key:
                left.data[key] = value
            else:
                right.data[key] = value

        # Replace old partition
        idx = self.partitions.index(partition)
        self.partitions[idx:idx + 1] = [left, right]
        self.split_count += 1


def exercise_3():
    """
    Demonstrate range-based partitioning with auto-split.
    """
    print("=== Exercise 3: Range-Based Partitioning with Auto-Split ===\n")

    store = RangePartitionedStore(split_threshold=5)

    print(f"Initial partitions: {len(store.partitions)}")
    print(f"Split threshold: {store.split_threshold} keys\n")

    # Insert keys
    keys = ["apple", "banana", "cherry", "date", "elderberry",
            "fig", "grape", "honeydew", "kiwi", "lemon",
            "mango", "nectarine", "orange", "papaya", "quince"]

    for key in keys:
        store.put(key, len(key))
        print(
            f"  Inserted '{key}': partitions={len(store.partitions)}, "
            f"splits={store.split_count}"
        )

    print(f"\nFinal partition layout:")
    for p in store.partitions:
        print(
            f"  Partition {p.partition_id}: [{p.start!r}..{p.end!r}) "
            f"size={p.size()} keys={sorted(p.data.keys())}"
        )

    # Verify all keys are retrievable
    for key in keys:
        assert store.get(key) == len(key), f"Key '{key}' not found"
    print(f"\nAll {len(keys)} keys verified successfully.")
    print(f"Total splits performed: {store.split_count}")
    print()


# === Main ===

if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
