"""
Consistent Hashing Ring with Virtual Nodes

Consistent hashing maps both keys and nodes onto a circular hash space (ring).
Each key is assigned to the nearest node clockwise on the ring. When nodes are
added or removed, only a fraction of keys need to be remapped — unlike modular
hashing where nearly all keys must move.

Virtual nodes improve distribution uniformity by mapping each physical node to
multiple positions on the ring.

Key concepts:
- Hash ring and clockwise assignment
- Virtual nodes (vnodes) for better load balancing
- Minimal key redistribution on topology changes
- O(log N) lookup via bisect on sorted ring positions

Usage:
    python 06_consistent_hashing.py
"""

from __future__ import annotations

import hashlib
from bisect import bisect_right
from collections import Counter


class ConsistentHashRing:
    """
    Consistent hashing ring with configurable virtual nodes.

    Each physical node is mapped to `num_vnodes` positions on the ring
    using SHA-1 hashing. Keys are assigned to the nearest node clockwise.
    """

    def __init__(self, num_vnodes: int = 150):
        self.num_vnodes = num_vnodes
        self._ring: dict[int, str] = {}      # hash_value -> node_name
        self._sorted_keys: list[int] = []     # Sorted ring positions
        self._nodes: set[str] = set()         # Physical node names

    @staticmethod
    def _hash(key: str) -> int:
        """Compute a consistent hash using SHA-1 (32-bit truncation)."""
        digest = hashlib.sha1(key.encode()).hexdigest()
        return int(digest, 16) % (2**32)

    def add_node(self, node: str) -> None:
        """Add a physical node with its virtual nodes to the ring."""
        if node in self._nodes:
            return
        self._nodes.add(node)
        for i in range(self.num_vnodes):
            vnode_key = f"{node}#vn{i}"
            h = self._hash(vnode_key)
            self._ring[h] = node
        self._sorted_keys = sorted(self._ring.keys())

    def remove_node(self, node: str) -> None:
        """Remove a physical node and all its virtual nodes from the ring."""
        if node not in self._nodes:
            return
        self._nodes.discard(node)
        for i in range(self.num_vnodes):
            vnode_key = f"{node}#vn{i}"
            h = self._hash(vnode_key)
            self._ring.pop(h, None)
        self._sorted_keys = sorted(self._ring.keys())

    def get_node(self, key: str) -> str | None:
        """
        Find the node responsible for the given key.
        Returns the nearest node clockwise on the ring.
        """
        if not self._ring:
            return None
        h = self._hash(key)
        # Find the first ring position >= h
        idx = bisect_right(self._sorted_keys, h)
        # Wrap around to the beginning of the ring
        if idx == len(self._sorted_keys):
            idx = 0
        return self._ring[self._sorted_keys[idx]]

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def ring_size(self) -> int:
        return len(self._sorted_keys)

    def __repr__(self) -> str:
        return (f"ConsistentHashRing(nodes={sorted(self._nodes)}, "
                f"vnodes={self.num_vnodes}, ring_size={self.ring_size})")


def generate_keys(n: int, prefix: str = "key") -> list[str]:
    """Generate n test keys."""
    return [f"{prefix}_{i:06d}" for i in range(n)]


def measure_distribution(ring: ConsistentHashRing,
                         keys: list[str]) -> dict[str, int]:
    """Count how many keys map to each node."""
    counter: Counter[str] = Counter()
    for key in keys:
        node = ring.get_node(key)
        if node:
            counter[node] += 1
    return dict(counter)


def print_distribution(dist: dict[str, int], total_keys: int) -> None:
    """Display key distribution across nodes."""
    for node in sorted(dist.keys()):
        count = dist[node]
        pct = 100.0 * count / total_keys
        bar = "#" * int(pct / 2)
        print(f"  {node:>10}: {count:>6} keys ({pct:5.1f}%) {bar}")


def demo_basic_operations() -> None:
    """Demonstrate basic consistent hashing operations."""
    print("=" * 65)
    print("Basic Consistent Hashing Operations")
    print("=" * 65)

    ring = ConsistentHashRing(num_vnodes=100)
    nodes = ["node-A", "node-B", "node-C"]
    for n in nodes:
        ring.add_node(n)

    print(f"\n{ring}\n")

    # Assign some keys
    test_keys = ["user:alice", "user:bob", "user:carol",
                 "session:123", "session:456", "cache:homepage"]

    print("Key assignments:")
    for key in test_keys:
        node = ring.get_node(key)
        print(f"  {key:>20} -> {node}")

    # Verify consistency: same key always maps to same node
    print("\nConsistency check (100 lookups of same key):")
    target = ring.get_node("user:alice")
    consistent = all(ring.get_node("user:alice") == target for _ in range(100))
    print(f"  user:alice -> {target} (consistent={consistent})")


def demo_node_changes() -> None:
    """Show minimal key redistribution when nodes are added/removed."""
    print("\n" + "=" * 65)
    print("Key Redistribution on Topology Changes")
    print("=" * 65)

    num_keys = 10000
    keys = generate_keys(num_keys)

    ring = ConsistentHashRing(num_vnodes=150)
    for n in ["node-A", "node-B", "node-C", "node-D"]:
        ring.add_node(n)

    # Record initial assignments
    initial = {key: ring.get_node(key) for key in keys}

    print(f"\nInitial distribution ({ring.node_count} nodes, {num_keys} keys):")
    print_distribution(measure_distribution(ring, keys), num_keys)

    # Add a node
    print("\n--- Adding node-E ---")
    ring.add_node("node-E")
    after_add = {key: ring.get_node(key) for key in keys}

    moved = sum(1 for k in keys if initial[k] != after_add[k])
    print(f"\nAfter adding node-E ({ring.node_count} nodes):")
    print_distribution(measure_distribution(ring, keys), num_keys)
    print(f"\n  Keys moved: {moved}/{num_keys} ({100.0 * moved / num_keys:.1f}%)")
    print(f"  Ideal redistribution: ~{num_keys / ring.node_count:.0f} "
          f"({100.0 / ring.node_count:.1f}%)")

    # Remove a node
    print("\n--- Removing node-B ---")
    before_remove = {key: ring.get_node(key) for key in keys}
    ring.remove_node("node-B")
    after_remove = {key: ring.get_node(key) for key in keys}

    moved = sum(1 for k in keys if before_remove[k] != after_remove[k])
    print(f"\nAfter removing node-B ({ring.node_count} nodes):")
    print_distribution(measure_distribution(ring, keys), num_keys)
    print(f"\n  Keys moved: {moved}/{num_keys} ({100.0 * moved / num_keys:.1f}%)")
    print(f"  (Only keys from node-B were reassigned)")


def demo_vnode_comparison() -> None:
    """Compare distribution uniformity with different numbers of vnodes."""
    print("\n" + "=" * 65)
    print("Virtual Node Count vs Distribution Uniformity")
    print("=" * 65)

    num_keys = 100000
    keys = generate_keys(num_keys)
    nodes = ["node-1", "node-2", "node-3", "node-4", "node-5"]
    ideal_per_node = num_keys / len(nodes)

    vnode_counts = [1, 10, 50, 150, 500]

    print(f"\n  {len(nodes)} nodes, {num_keys} keys, ideal = "
          f"{ideal_per_node:.0f} keys/node ({100/len(nodes):.1f}%)")
    print(f"\n  {'vnodes':>8}  {'std_dev':>8}  {'min%':>6}  {'max%':>6}  "
          f"{'spread':>8}  distribution")
    print("  " + "-" * 62)

    for vn in vnode_counts:
        ring = ConsistentHashRing(num_vnodes=vn)
        for n in nodes:
            ring.add_node(n)

        dist = measure_distribution(ring, keys)
        counts = [dist.get(n, 0) for n in nodes]
        mean = sum(counts) / len(counts)
        std_dev = (sum((c - mean) ** 2 for c in counts) / len(counts)) ** 0.5
        min_pct = 100.0 * min(counts) / num_keys
        max_pct = 100.0 * max(counts) / num_keys
        spread = max_pct - min_pct

        # Simple ASCII bar
        bar = "".join(f"{100*c/num_keys:.0f}/" for c in sorted(counts))

        print(f"  {vn:>8}  {std_dev:>8.0f}  {min_pct:>5.1f}%  {max_pct:>5.1f}%  "
              f"{spread:>7.1f}%  [{bar[:-1]}]")

    print("""
  Takeaway: More virtual nodes = more uniform distribution.
  - 1 vnode:    Extremely uneven (some nodes get 0-2x share)
  - 50 vnodes:  Reasonable (within ~5% of ideal)
  - 150 vnodes: Good for production (within ~2% of ideal)
  - 500 vnodes: Diminishing returns; more memory overhead
""")


def demo_key_movement_comparison() -> None:
    """Compare key movement: consistent hashing vs modular hashing."""
    print("=" * 65)
    print("Consistent Hashing vs Modular Hashing: Key Movement")
    print("=" * 65)

    num_keys = 10000
    keys = generate_keys(num_keys)

    # Modular hashing: key -> hash(key) % num_nodes
    def modular_assign(key: str, n: int) -> int:
        return ConsistentHashRing._hash(key) % n

    # With 4 nodes
    mod_before = {k: modular_assign(k, 4) for k in keys}
    # Add 1 node (now 5)
    mod_after = {k: modular_assign(k, 5) for k in keys}
    mod_moved = sum(1 for k in keys if mod_before[k] != mod_after[k])

    # Consistent hashing with 4 nodes
    ring = ConsistentHashRing(num_vnodes=150)
    for i in range(4):
        ring.add_node(f"node-{i}")
    ch_before = {k: ring.get_node(k) for k in keys}
    ring.add_node("node-4")
    ch_after = {k: ring.get_node(k) for k in keys}
    ch_moved = sum(1 for k in keys if ch_before[k] != ch_after[k])

    print(f"\n  Adding 1 node to a 4-node cluster ({num_keys} keys):")
    print(f"  Modular hashing:     {mod_moved:>5} keys moved "
          f"({100.0*mod_moved/num_keys:.1f}%)")
    print(f"  Consistent hashing:  {ch_moved:>5} keys moved "
          f"({100.0*ch_moved/num_keys:.1f}%)")
    print(f"  Improvement:         {mod_moved/max(ch_moved,1):.1f}x fewer moves\n")


if __name__ == "__main__":
    demo_basic_operations()
    demo_node_changes()
    demo_vnode_comparison()
    demo_key_movement_comparison()
    print("Done.")
