"""
Exercises for Lesson 09: Replication Strategies
Topic: Distributed_Systems

Solutions to practice problems from the lesson.
"""

import hashlib
import random
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict


# === Exercise 1: Quorum-Based Read/Write System ===
# Problem: Implement a quorum-based read/write system with tunable
# W (write quorum), R (read quorum), and N (replication factor).
# Ensure R + W > N for strong consistency.

class QuorumReplica:
    """A single replica in the quorum system."""

    def __init__(self, replica_id: str):
        self.replica_id = replica_id
        self.store: Dict[str, Tuple[int, int]] = {}  # key -> (value, version)
        self.is_alive = True

    def write(self, key: str, value: int, version: int) -> bool:
        if not self.is_alive:
            return False
        current = self.store.get(key)
        if current is None or version > current[1]:
            self.store[key] = (value, version)
        return True

    def read(self, key: str) -> Optional[Tuple[int, int]]:
        if not self.is_alive:
            return None
        return self.store.get(key)


class QuorumSystem:
    """
    Quorum-based storage with tunable N, W, R.
    """

    def __init__(self, n: int, w: int, r: int):
        self.n = n
        self.w = w
        self.r = r
        self.replicas = [QuorumReplica(f"R{i}") for i in range(n)]
        self.version_counter = 0

        if w + r <= n:
            print(f"WARNING: W({w}) + R({r}) <= N({n}), may not guarantee consistency")

    def write(self, key: str, value: int) -> bool:
        """Write to W replicas. Returns True if write quorum met."""
        self.version_counter += 1
        version = self.version_counter

        acks = 0
        for replica in self.replicas:
            if replica.write(key, value, version):
                acks += 1

        return acks >= self.w

    def read(self, key: str) -> Optional[int]:
        """Read from R replicas and return the value with highest version."""
        results = []
        for replica in self.replicas:
            result = replica.read(key)
            if result is not None:
                results.append(result)

        if len(results) < self.r:
            return None  # read quorum not met

        # Return value with highest version
        best = max(results, key=lambda x: x[1])
        return best[0]


def exercise_1():
    """
    Demonstrate quorum reads/writes with different W, R, N settings.
    """
    print("=== Exercise 1: Quorum-Based Read/Write ===\n")

    # Strong consistency: R + W > N
    qs = QuorumSystem(n=5, w=3, r=3)
    print(f"Config: N=5, W=3, R=3 (R+W={6} > N=5)")

    qs.write("x", 42)
    result = qs.read("x")
    print(f"Write x=42, Read x={result}")
    assert result == 42

    # Update
    qs.write("x", 100)
    result = qs.read("x")
    print(f"Write x=100, Read x={result}")
    assert result == 100

    # Kill 2 replicas - still meets quorum
    qs.replicas[3].is_alive = False
    qs.replicas[4].is_alive = False
    print(f"\n2 replicas down (3 alive)")

    qs.write("y", 99)
    result = qs.read("y")
    print(f"Write y=99, Read y={result}")
    assert result == 99

    # Eventual consistency config: W=1, R=1
    print(f"\nConfig: N=3, W=1, R=1 (R+W=2 <= N=3, eventual consistency)")
    qs2 = QuorumSystem(n=3, w=1, r=1)
    qs2.write("z", 55)
    result = qs2.read("z")
    print(f"Write z=55, Read z={result}")
    print()


# === Exercise 2: Read Repair Protocol ===
# Problem: Implement read repair: when a read detects stale replicas
# (during a quorum read), update the stale replicas in the background.

class ReadRepairSystem:
    """Quorum system with read repair."""

    def __init__(self, n: int, w: int, r: int):
        self.n = n
        self.w = w
        self.r = r
        self.replicas = [QuorumReplica(f"R{i}") for i in range(n)]
        self.version_counter = 0
        self.repairs_performed = 0

    def write(self, key: str, value: int, target_replicas: Optional[List[int]] = None) -> bool:
        """Write to specific replicas (for testing partial writes)."""
        self.version_counter += 1
        version = self.version_counter

        targets = target_replicas or range(self.n)
        acks = 0
        for i in targets:
            if self.replicas[i].write(key, value, version):
                acks += 1
        return acks >= self.w

    def read_with_repair(self, key: str) -> Optional[int]:
        """
        Read from all replicas. Return the latest value and repair
        any stale replicas.
        """
        results = {}  # replica_index -> (value, version)
        for i, replica in enumerate(self.replicas):
            result = replica.read(key)
            if result is not None:
                results[i] = result

        if len(results) < self.r:
            return None

        # Find the latest version
        best_idx = max(results, key=lambda i: results[i][1])
        best_value, best_version = results[best_idx]

        # Read repair: update stale replicas
        for i, (value, version) in results.items():
            if version < best_version:
                self.replicas[i].write(key, best_value, best_version)
                self.repairs_performed += 1

        # Also write to replicas that had no data
        for i in range(self.n):
            if i not in results and self.replicas[i].is_alive:
                self.replicas[i].write(key, best_value, best_version)
                self.repairs_performed += 1

        return best_value


def exercise_2():
    """
    Demonstrate read repair fixing stale replicas.
    """
    print("=== Exercise 2: Read Repair Protocol ===\n")

    rr = ReadRepairSystem(n=3, w=2, r=2)

    # Write to only 2 of 3 replicas (simulating partial write)
    rr.write("x", 42, target_replicas=[0, 1])

    print("After partial write (replicas 0,1 only):")
    for i, rep in enumerate(rr.replicas):
        print(f"  R{i}: {rep.store.get('x', 'MISSING')}")

    # Read with repair
    result = rr.read_with_repair("x")
    print(f"\nRead with repair: x={result}")
    print(f"Repairs performed: {rr.repairs_performed}")

    print("\nAfter read repair:")
    for i, rep in enumerate(rr.replicas):
        print(f"  R{i}: {rep.store.get('x', 'MISSING')}")

    # Verify all replicas now have the data
    assert all(rep.store.get("x") is not None for rep in rr.replicas)
    print("\nAll replicas now consistent via read repair.")
    print()


# === Exercise 3: Anti-Entropy with Merkle Tree ===
# Problem: Implement anti-entropy using Merkle tree comparison to
# efficiently detect and synchronize differences between replicas.

class MerkleNode:
    """A node in a Merkle tree."""

    def __init__(self, hash_val: str, left=None, right=None, key: str = ""):
        self.hash_val = hash_val
        self.left = left
        self.right = right
        self.key = key  # leaf nodes store the key


def compute_hash(data: str) -> str:
    """Compute SHA-256 hash (truncated for display)."""
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def build_merkle_tree(data: Dict[str, int]) -> Optional[MerkleNode]:
    """
    Build a Merkle tree from key-value pairs.
    Leaves are hashes of individual key-value pairs.
    Internal nodes are hashes of their children.
    """
    if not data:
        return None

    sorted_keys = sorted(data.keys())
    leaves = []
    for key in sorted_keys:
        h = compute_hash(f"{key}:{data[key]}")
        leaves.append(MerkleNode(h, key=key))

    # Pad to power of 2
    while len(leaves) & (len(leaves) - 1) != 0:
        leaves.append(MerkleNode(compute_hash("EMPTY")))

    # Build tree bottom-up
    nodes = leaves
    while len(nodes) > 1:
        next_level = []
        for i in range(0, len(nodes), 2):
            combined = compute_hash(nodes[i].hash_val + nodes[i + 1].hash_val)
            next_level.append(MerkleNode(combined, nodes[i], nodes[i + 1]))
        nodes = next_level

    return nodes[0]


def find_differences(
    tree1: Optional[MerkleNode],
    tree2: Optional[MerkleNode],
    diffs: List[str],
):
    """
    Compare two Merkle trees and find differing leaf keys.
    """
    if tree1 is None and tree2 is None:
        return
    if tree1 is None or tree2 is None:
        if tree1 and tree1.key:
            diffs.append(tree1.key)
        if tree2 and tree2.key:
            diffs.append(tree2.key)
        return

    if tree1.hash_val == tree2.hash_val:
        return  # subtrees are identical

    # If leaf nodes, record the difference
    if tree1.key or tree2.key:
        if tree1.key:
            diffs.append(tree1.key)
        if tree2.key and tree2.key != tree1.key:
            diffs.append(tree2.key)
        return

    # Recurse into children
    find_differences(tree1.left, tree2.left, diffs)
    find_differences(tree1.right, tree2.right, diffs)


def exercise_3():
    """
    Demonstrate anti-entropy using Merkle tree comparison.
    """
    print("=== Exercise 3: Anti-Entropy with Merkle Trees ===\n")

    # Two replicas with mostly matching data
    replica1_data = {"a": 1, "b": 2, "c": 3, "d": 4}
    replica2_data = {"a": 1, "b": 2, "c": 99, "d": 4}  # 'c' differs

    tree1 = build_merkle_tree(replica1_data)
    tree2 = build_merkle_tree(replica2_data)

    print(f"Replica 1: {replica1_data}")
    print(f"Replica 2: {replica2_data}")
    print(f"\nRoot hashes: R1={tree1.hash_val}, R2={tree2.hash_val}")
    print(f"Roots match: {tree1.hash_val == tree2.hash_val}")

    diffs = []
    find_differences(tree1, tree2, diffs)
    print(f"\nDiffering keys found: {diffs}")
    assert "c" in diffs, "Should detect 'c' as different"

    # After syncing
    replica2_data["c"] = replica1_data["c"]
    tree2_fixed = build_merkle_tree(replica2_data)
    print(f"\nAfter syncing key 'c':")
    print(f"Roots match: {tree1.hash_val == tree2_fixed.hash_val}")

    diffs2 = []
    find_differences(tree1, tree2_fixed, diffs2)
    print(f"Remaining differences: {diffs2}")
    assert len(diffs2) == 0
    print("\nMerkle tree anti-entropy successfully detects and resolves differences.")
    print()


# === Main ===

if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
