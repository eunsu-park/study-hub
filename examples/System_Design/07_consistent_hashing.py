"""
Consistent Hashing

Demonstrates:
- Consistent hashing ring
- Virtual nodes for load balance
- Node addition/removal with minimal key redistribution
- Comparison with modular hashing

Theory:
- Consistent hashing maps both keys and nodes onto a hash ring [0, 2^32).
- Each key is assigned to the first node found clockwise on the ring.
- Adding/removing a node only affects keys in adjacent ring segments.
- Virtual nodes: each physical node gets multiple positions on the ring,
  improving load distribution.

Adapted from System Design Lesson 07.
"""

import hashlib
import bisect
from collections import defaultdict


# Why: Using a ring of virtual nodes rather than mapping servers directly to
# hash positions. Virtual nodes ensure that when a server is added/removed,
# only ~1/N of the keys are redistributed — not all of them.
class ConsistentHashRing:
    """Consistent hashing ring with virtual nodes."""

    def __init__(self, virtual_nodes: int = 150):
        # Why: 150 virtual nodes per physical node provides a good balance of
        # load uniformity vs memory overhead. Too few vnodes (e.g., 1) causes
        # severe load imbalance; too many wastes memory on the ring.
        self.virtual_nodes = virtual_nodes
        self.ring: list[int] = []          # sorted hash positions
        self.ring_map: dict[int, str] = {} # hash → node name
        self.nodes: set[str] = set()

    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)

    def add_node(self, node: str) -> None:
        self.nodes.add(node)
        for i in range(self.virtual_nodes):
            vnode_key = f"{node}#v{i}"
            h = self._hash(vnode_key)
            self.ring_map[h] = node
            bisect.insort(self.ring, h)

    def remove_node(self, node: str) -> None:
        self.nodes.discard(node)
        to_remove = []
        for i in range(self.virtual_nodes):
            vnode_key = f"{node}#v{i}"
            h = self._hash(vnode_key)
            if h in self.ring_map:
                del self.ring_map[h]
                to_remove.append(h)
        self.ring = [h for h in self.ring if h not in set(to_remove)]

    def get_node(self, key: str) -> str | None:
        if not self.ring:
            return None
        h = self._hash(key)
        # Why: bisect_right finds the first ring position clockwise from the
        # key's hash. Wrapping to index 0 implements the ring topology — keys
        # past the last position wrap around to the first node on the ring.
        idx = bisect.bisect_right(self.ring, h)
        if idx == len(self.ring):
            idx = 0
        return self.ring_map[self.ring[idx]]

    def get_distribution(self, keys: list[str]) -> dict[str, int]:
        dist: dict[str, int] = defaultdict(int)
        for key in keys:
            node = self.get_node(key)
            if node:
                dist[node] += 1
        return dict(dist)


# ── Modular Hashing (for comparison) ────────────────────────────────────

# Why: Modular hashing is provided as a contrast to show its fatal weakness:
# when N changes (server added/removed), nearly every key remaps to a different
# server, causing a "thundering herd" of cache misses in production.
class ModularHash:
    """Simple modular hashing (key % N)."""

    def __init__(self, nodes: list[str]):
        self.nodes = nodes

    def get_node(self, key: str) -> str:
        h = int(hashlib.md5(key.encode()).hexdigest(), 16)
        return self.nodes[h % len(self.nodes)]

    def get_distribution(self, keys: list[str]) -> dict[str, int]:
        dist: dict[str, int] = defaultdict(int)
        for key in keys:
            dist[self.get_node(key)] += 1
        return dict(dist)


# ── Demos ───────────────────────────────────────────────────────────────

def print_dist(dist: dict[str, int], total: int) -> None:
    for node in sorted(dist):
        count = dist[node]
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"    {node:<10} {count:>6} ({pct:>5.1f}%) {bar}")


def demo_basic():
    print("=" * 60)
    print("CONSISTENT HASHING RING")
    print("=" * 60)

    ring = ConsistentHashRing(virtual_nodes=150)
    nodes = ["node-A", "node-B", "node-C"]
    for n in nodes:
        ring.add_node(n)

    keys = [f"key-{i}" for i in range(10000)]
    dist = ring.get_distribution(keys)

    print(f"\n  {len(nodes)} nodes, {ring.virtual_nodes} virtual nodes each")
    print(f"  {len(keys)} keys distribution:\n")
    print_dist(dist, len(keys))

    # Show specific key routing
    print(f"\n  Sample key routing:")
    for key in ["user:1001", "user:1002", "session:abc", "cache:page:1"]:
        node = ring.get_node(key)
        print(f"    {key} → {node}")


def demo_node_change():
    print("\n" + "=" * 60)
    print("NODE ADDITION/REMOVAL")
    print("=" * 60)

    ring = ConsistentHashRing(virtual_nodes=150)
    nodes = ["node-A", "node-B", "node-C"]
    for n in nodes:
        ring.add_node(n)

    keys = [f"key-{i}" for i in range(10000)]

    # Before
    before = {key: ring.get_node(key) for key in keys}

    # Add a node
    ring.add_node("node-D")
    after_add = {key: ring.get_node(key) for key in keys}
    moved = sum(1 for k in keys if before[k] != after_add[k])

    print(f"\n  After adding node-D:")
    print(f"    Keys moved: {moved}/{len(keys)} ({moved/len(keys)*100:.1f}%)")
    print(f"    Ideal (1/N): {len(keys)//4} ({100/4:.1f}%)")
    dist = ring.get_distribution(keys)
    print_dist(dist, len(keys))

    # Remove a node
    ring.remove_node("node-B")
    after_remove = {key: ring.get_node(key) for key in keys}
    moved2 = sum(1 for k in keys if after_add[k] != after_remove[k])

    print(f"\n  After removing node-B:")
    print(f"    Keys moved: {moved2}/{len(keys)} ({moved2/len(keys)*100:.1f}%)")
    dist2 = ring.get_distribution(keys)
    print_dist(dist2, len(keys))


def demo_vs_modular():
    print("\n" + "=" * 60)
    print("CONSISTENT vs MODULAR HASHING")
    print("=" * 60)

    keys = [f"key-{i}" for i in range(10000)]
    nodes_3 = ["node-A", "node-B", "node-C"]
    nodes_4 = nodes_3 + ["node-D"]

    # Modular hashing
    mod3 = ModularHash(nodes_3)
    mod4 = ModularHash(nodes_4)
    before_mod = {k: mod3.get_node(k) for k in keys}
    after_mod = {k: mod4.get_node(k) for k in keys}
    mod_moved = sum(1 for k in keys if before_mod[k] != after_mod.get(k))

    # Consistent hashing
    ch3 = ConsistentHashRing(virtual_nodes=150)
    for n in nodes_3:
        ch3.add_node(n)
    before_ch = {k: ch3.get_node(k) for k in keys}

    ch3.add_node("node-D")
    after_ch = {k: ch3.get_node(k) for k in keys}
    ch_moved = sum(1 for k in keys if before_ch[k] != after_ch[k])

    print(f"\n  Adding 1 node to 3-node cluster ({len(keys)} keys):")
    print(f"\n  {'Method':<25} {'Keys Moved':>11} {'Percentage':>11}")
    print(f"  {'-'*25} {'-'*11} {'-'*11}")
    print(f"  {'Modular (key % N)':<25} {mod_moved:>11} {mod_moved/len(keys)*100:>10.1f}%")
    print(f"  {'Consistent Hashing':<25} {ch_moved:>11} {ch_moved/len(keys)*100:>10.1f}%")
    print(f"\n  Modular hashing redistributes ~{mod_moved/len(keys)*100:.0f}% of keys!")
    print(f"  Consistent hashing only moves ~{ch_moved/len(keys)*100:.0f}% (close to ideal 25%).")


def demo_virtual_nodes():
    print("\n" + "=" * 60)
    print("VIRTUAL NODES IMPACT")
    print("=" * 60)

    keys = [f"key-{i}" for i in range(10000)]
    nodes = ["node-A", "node-B", "node-C", "node-D"]

    print(f"\n  {len(nodes)} nodes, {len(keys)} keys")
    print(f"\n  {'Vnodes':>8}  {'Std Dev':>8}  {'Min%':>6}  {'Max%':>6}  {'Balance'}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*10}")

    for vn in [1, 10, 50, 100, 200, 500]:
        ring = ConsistentHashRing(virtual_nodes=vn)
        for n in nodes:
            ring.add_node(n)
        dist = ring.get_distribution(keys)
        counts = [dist.get(n, 0) for n in nodes]
        avg = len(keys) / len(nodes)
        std = (sum((c - avg)**2 for c in counts) / len(counts)) ** 0.5
        min_pct = min(counts) / len(keys) * 100
        max_pct = max(counts) / len(keys) * 100
        balance = "Good" if max_pct - min_pct < 10 else "Fair" if max_pct - min_pct < 20 else "Poor"
        print(f"  {vn:>8}  {std:>7.0f}  {min_pct:>5.1f}  {max_pct:>5.1f}  {balance}")


if __name__ == "__main__":
    demo_basic()
    demo_node_change()
    demo_vs_modular()
    demo_virtual_nodes()
