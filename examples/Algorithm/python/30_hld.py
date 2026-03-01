"""
Example: Heavy-Light Decomposition
Topic: Algorithm – Lesson 30

Demonstrates:
  1. Tree decomposition into heavy/light chains
  2. Mapping tree to flat array for segment tree
  3. Path sum queries in O(log²N)
  4. Subtree queries in O(log N)

Run: python 30_hld.py
"""

import sys
from collections import defaultdict


# ============================================================
# Segment Tree (for HLD queries)
# ============================================================
class SegTree:
    """Sum segment tree for HLD path and subtree queries."""

    def __init__(self, n):
        self.n = n
        self.tree = [0] * (4 * n)

    def build(self, arr, node=1, lo=0, hi=None):
        if hi is None:
            hi = self.n - 1
        if lo == hi:
            if lo < len(arr):
                self.tree[node] = arr[lo]
            return
        mid = (lo + hi) // 2
        self.build(arr, 2 * node, lo, mid)
        self.build(arr, 2 * node + 1, mid + 1, hi)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def update(self, idx, val, node=1, lo=0, hi=None):
        if hi is None:
            hi = self.n - 1
        if lo == hi:
            self.tree[node] = val
            return
        mid = (lo + hi) // 2
        if idx <= mid:
            self.update(idx, val, 2 * node, lo, mid)
        else:
            self.update(idx, val, 2 * node + 1, mid + 1, hi)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def query(self, ql, qr, node=1, lo=0, hi=None):
        if hi is None:
            hi = self.n - 1
        if qr < lo or hi < ql:
            return 0
        if ql <= lo and hi <= qr:
            return self.tree[node]
        mid = (lo + hi) // 2
        return (self.query(ql, qr, 2 * node, lo, mid) +
                self.query(ql, qr, 2 * node + 1, mid + 1, hi))


# ============================================================
# Heavy-Light Decomposition
# ============================================================
class HLD:
    """Heavy-Light Decomposition for path and subtree queries."""

    def __init__(self, n, adj, values, root=0):
        self.n = n
        self.adj = adj
        self.values = values
        self.root = root

        self.parent = [-1] * n
        self.depth = [0] * n
        self.size = [1] * n
        self.heavy = [-1] * n  # heavy child
        self.head = [0] * n    # chain head
        self.pos = [0] * n     # position in flat array

        self._dfs_size()
        self._dfs_hld()
        self._build_segment_tree()

    def _dfs_size(self):
        """Compute subtree sizes and find heavy children (iterative)."""
        stack = [(self.root, -1, False)]
        order = []

        while stack:
            node, par, visited = stack.pop()
            if visited:
                self.size[node] = 1
                max_child_size = 0
                for child in self.adj[node]:
                    if child != par:
                        self.size[node] += self.size[child]
                        if self.size[child] > max_child_size:
                            max_child_size = self.size[child]
                            self.heavy[node] = child
                order.append(node)
                continue

            self.parent[node] = par
            stack.append((node, par, True))
            for child in self.adj[node]:
                if child != par:
                    self.depth[child] = self.depth[node] + 1
                    stack.append((child, node, False))

    def _dfs_hld(self):
        """Assign chain heads and positions (iterative DFS, heavy child first)."""
        timer = 0
        stack = [(self.root, self.root)]

        while stack:
            node, chain_head = stack.pop()
            self.head[node] = chain_head
            self.pos[node] = timer
            timer += 1

            # Push light children first (so they're processed later)
            light = []
            for child in self.adj[node]:
                if child != self.parent[node] and child != self.heavy[node]:
                    light.append(child)

            for child in reversed(light):
                stack.append((child, child))  # new chain

            # Push heavy child last (processed next)
            if self.heavy[node] != -1:
                stack.append((self.heavy[node], chain_head))

    def _build_segment_tree(self):
        """Build segment tree over the HLD-flattened array."""
        flat = [0] * self.n
        for v in range(self.n):
            flat[self.pos[v]] = self.values[v]
        self.seg = SegTree(self.n)
        self.seg.build(flat)

    def path_query(self, u, v):
        """Sum of values on the path from u to v. O(log²N)."""
        result = 0
        while self.head[u] != self.head[v]:
            # Move the deeper chain head up
            if self.depth[self.head[u]] < self.depth[self.head[v]]:
                u, v = v, u
            result += self.seg.query(self.pos[self.head[u]], self.pos[u])
            u = self.parent[self.head[u]]

        # Same chain: query between u and v
        if self.depth[u] > self.depth[v]:
            u, v = v, u
        result += self.seg.query(self.pos[u], self.pos[v])
        return result

    def subtree_query(self, v):
        """Sum of values in the subtree of v. O(log N)."""
        return self.seg.query(self.pos[v], self.pos[v] + self.size[v] - 1)

    def update(self, v, val):
        """Update the value of node v. O(log N)."""
        self.values[v] = val
        self.seg.update(self.pos[v], val)

    def lca(self, u, v):
        """Find LCA of u and v using HLD. O(log N)."""
        while self.head[u] != self.head[v]:
            if self.depth[self.head[u]] < self.depth[self.head[v]]:
                u, v = v, u
            u = self.parent[self.head[u]]
        return u if self.depth[u] <= self.depth[v] else v


# ============================================================
# Demo 1: Basic HLD on a Sample Tree
# ============================================================
def demo_basic():
    print("=" * 60)
    print("Demo 1: Basic HLD Construction")
    print("=" * 60)

    # Tree:
    #         0 (val=1)
    #        / \
    #       1   2 (val=5)
    #      /|   |
    #     3 4   5 (val=3)
    #    /
    #   6 (val=7)
    n = 7
    adj = defaultdict(list)
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (3, 6)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    values = [1, 2, 5, 4, 3, 3, 7]

    hld = HLD(n, adj, values, root=0)

    print(f"\n  Tree (7 nodes, values = {values}):")
    print(f"          0(1)")
    print(f"         / \\")
    print(f"       1(2)  2(5)")
    print(f"      / \\    |")
    print(f"    3(4) 4(3) 5(3)")
    print(f"    |")
    print(f"   6(7)")

    print(f"\n  HLD decomposition:")
    print(f"  Node:  {list(range(n))}")
    print(f"  Pos:   {hld.pos}")
    print(f"  Head:  {hld.head}")
    print(f"  Heavy: {hld.heavy}")
    print(f"  Size:  {hld.size}")

    # Show chains
    chains = defaultdict(list)
    for v in range(n):
        chains[hld.head[v]].append(v)
    print(f"\n  Chains:")
    for head, members in sorted(chains.items()):
        chain_str = " → ".join(str(m) for m in sorted(members, key=lambda x: hld.pos[x]))
        print(f"    Head {head}: [{chain_str}]")

    # Path queries
    print(f"\n  Path queries:")
    for u, v in [(6, 5), (4, 2), (0, 6), (3, 4)]:
        result = hld.path_query(u, v)
        lca = hld.lca(u, v)
        print(f"    path_sum({u}, {v}) = {result}, LCA = {lca}")

    # Subtree queries
    print(f"\n  Subtree queries:")
    for v in [0, 1, 3]:
        result = hld.subtree_query(v)
        print(f"    subtree_sum({v}) = {result}")
    print()


# ============================================================
# Demo 2: Path Queries on a Chain (Worst Case for Naive)
# ============================================================
def demo_chain():
    print("=" * 60)
    print("Demo 2: Path Queries on a Chain")
    print("=" * 60)

    # Linear chain: 0 - 1 - 2 - ... - 999
    n = 1000
    adj = defaultdict(list)
    for i in range(n - 1):
        adj[i].append(i + 1)
        adj[i + 1].append(i)
    values = list(range(1, n + 1))  # values 1 to n

    hld = HLD(n, adj, values, root=0)

    # The entire chain should be one heavy chain
    chains = defaultdict(list)
    for v in range(n):
        chains[hld.head[v]].append(v)

    print(f"\n  Chain of {n} nodes, values = [1, 2, ..., {n}]")
    print(f"  Number of chains: {len(chains)}")

    # Path query: sum from 0 to 999
    result = hld.path_query(0, n - 1)
    expected = n * (n + 1) // 2
    print(f"  path_sum(0, {n-1}) = {result} (expected {expected})")

    # Partial path
    result = hld.path_query(100, 200)
    expected = sum(range(101, 202))  # values 101 to 201
    print(f"  path_sum(100, 200) = {result} (expected {expected})")

    # Update and re-query
    hld.update(500, 0)
    result = hld.path_query(0, n - 1)
    print(f"  After setting node 500 to 0: path_sum(0, {n-1}) = {result}")
    print()


# ============================================================
# Demo 3: Balanced Binary Tree
# ============================================================
def demo_balanced():
    print("=" * 60)
    print("Demo 3: HLD on Balanced Binary Tree")
    print("=" * 60)

    # Build a complete binary tree of depth 4 (15 nodes)
    n = 15
    adj = defaultdict(list)
    for i in range(1, n):
        parent = (i - 1) // 2
        adj[parent].append(i)
        adj[i].append(parent)
    values = [1] * n

    hld = HLD(n, adj, values, root=0)

    # Count chains
    chains = defaultdict(list)
    for v in range(n):
        chains[hld.head[v]].append(v)

    print(f"\n  Complete binary tree, {n} nodes, depth = 3")
    print(f"  Number of chains: {len(chains)}")

    # Light edges on any root-to-leaf path
    leaf = n - 1  # rightmost leaf
    light_edges = 0
    v = leaf
    while v != 0:
        p = hld.parent[v]
        if hld.heavy[p] != v:
            light_edges += 1
        v = p

    import math
    print(f"  Light edges on path root→{leaf}: {light_edges}")
    print(f"  Theoretical max: log₂({n}) ≈ {math.log2(n):.1f}")

    # LCA queries
    print(f"\n  LCA queries:")
    for u, v in [(7, 8), (3, 4), (7, 14), (0, 14)]:
        lca = hld.lca(u, v)
        print(f"    LCA({u}, {v}) = {lca}")
    print()


if __name__ == "__main__":
    demo_basic()
    demo_chain()
    demo_balanced()
