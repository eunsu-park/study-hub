"""
Exercises for Lesson 30: Heavy-Light Decomposition
Topic: Algorithm

Solutions to practice problems covering HLD construction,
path queries, subtree queries, LCA, and path maximum.
"""

from collections import defaultdict


# ============================================================
# Shared: Segment Tree
# ============================================================
class SegTree:
    def __init__(self, n, op='sum'):
        self.n = n
        self.op = op
        self.identity = 0 if op == 'sum' else float('-inf')
        self.tree = [self.identity] * (4 * n)

    def _combine(self, a, b):
        return a + b if self.op == 'sum' else max(a, b)

    def build(self, arr, node=1, lo=0, hi=None):
        if hi is None: hi = self.n - 1
        if lo == hi:
            self.tree[node] = arr[lo] if lo < len(arr) else self.identity
            return
        mid = (lo + hi) // 2
        self.build(arr, 2*node, lo, mid)
        self.build(arr, 2*node+1, mid+1, hi)
        self.tree[node] = self._combine(self.tree[2*node], self.tree[2*node+1])

    def update(self, idx, val, node=1, lo=0, hi=None):
        if hi is None: hi = self.n - 1
        if lo == hi:
            self.tree[node] = val
            return
        mid = (lo + hi) // 2
        if idx <= mid: self.update(idx, val, 2*node, lo, mid)
        else: self.update(idx, val, 2*node+1, mid+1, hi)
        self.tree[node] = self._combine(self.tree[2*node], self.tree[2*node+1])

    def query(self, ql, qr, node=1, lo=0, hi=None):
        if hi is None: hi = self.n - 1
        if qr < lo or hi < ql: return self.identity
        if ql <= lo and hi <= qr: return self.tree[node]
        mid = (lo + hi) // 2
        return self._combine(
            self.query(ql, qr, 2*node, lo, mid),
            self.query(ql, qr, 2*node+1, mid+1, hi))


# ============================================================
# Shared: HLD
# ============================================================
class HLD:
    def __init__(self, n, adj, values, root=0, seg_op='sum'):
        self.n = n
        self.adj = adj
        self.values = values
        self.root = root
        self.parent = [-1] * n
        self.depth = [0] * n
        self.size = [1] * n
        self.heavy = [-1] * n
        self.head = [0] * n
        self.pos = [0] * n
        self._dfs_size()
        self._dfs_hld()
        flat = [0] * n
        for v in range(n):
            flat[self.pos[v]] = values[v]
        self.seg = SegTree(n, op=seg_op)
        self.seg.build(flat)

    def _dfs_size(self):
        stack = [(self.root, -1, False)]
        while stack:
            node, par, visited = stack.pop()
            if visited:
                self.size[node] = 1
                mx = 0
                for ch in self.adj[node]:
                    if ch != par:
                        self.size[node] += self.size[ch]
                        if self.size[ch] > mx:
                            mx = self.size[ch]
                            self.heavy[node] = ch
                continue
            self.parent[node] = par
            stack.append((node, par, True))
            for ch in self.adj[node]:
                if ch != par:
                    self.depth[ch] = self.depth[node] + 1
                    stack.append((ch, node, False))

    def _dfs_hld(self):
        timer = 0
        stack = [(self.root, self.root)]
        while stack:
            node, chain_head = stack.pop()
            self.head[node] = chain_head
            self.pos[node] = timer
            timer += 1
            light = []
            for ch in self.adj[node]:
                if ch != self.parent[node] and ch != self.heavy[node]:
                    light.append(ch)
            for ch in reversed(light):
                stack.append((ch, ch))
            if self.heavy[node] != -1:
                stack.append((self.heavy[node], chain_head))

    def path_query(self, u, v):
        result = self.seg.identity
        while self.head[u] != self.head[v]:
            if self.depth[self.head[u]] < self.depth[self.head[v]]:
                u, v = v, u
            result = self.seg._combine(
                result, self.seg.query(self.pos[self.head[u]], self.pos[u]))
            u = self.parent[self.head[u]]
        if self.depth[u] > self.depth[v]:
            u, v = v, u
        result = self.seg._combine(
            result, self.seg.query(self.pos[u], self.pos[v]))
        return result

    def subtree_query(self, v):
        return self.seg.query(self.pos[v], self.pos[v] + self.size[v] - 1)

    def update(self, v, val):
        self.values[v] = val
        self.seg.update(self.pos[v], val)

    def lca(self, u, v):
        while self.head[u] != self.head[v]:
            if self.depth[self.head[u]] < self.depth[self.head[v]]:
                u, v = v, u
            u = self.parent[self.head[u]]
        return u if self.depth[u] <= self.depth[v] else v


# ============================================================
# Exercise 1: HLD Construction Verification
# ============================================================
def exercise_1():
    """
    Build HLD and verify key properties:
    chains are contiguous, light edges ≤ log₂N, subtrees contiguous.
    """
    print("=== Exercise 1: HLD Construction Verification ===\n")

    import math

    n = 15
    adj = defaultdict(list)
    # Build complete binary tree
    for i in range(1, n):
        p = (i - 1) // 2
        adj[p].append(i)
        adj[i].append(p)
    values = list(range(1, n + 1))

    hld = HLD(n, adj, values, root=0)

    # Verify chains are contiguous
    chains = defaultdict(list)
    for v in range(n):
        chains[hld.head[v]].append((hld.pos[v], v))

    contiguous = True
    for head, members in chains.items():
        positions = sorted(p for p, v in members)
        for i in range(len(positions) - 1):
            if positions[i + 1] != positions[i] + 1:
                contiguous = False
                break

    print(f"  Binary tree, {n} nodes")
    print(f"  Chains contiguous: {contiguous}")

    # Verify light edge count
    max_light = 0
    for leaf in range(n):
        if hld.heavy[leaf] != -1:
            continue  # not a leaf
        light_count = 0
        v = leaf
        while v != 0:
            p = hld.parent[v]
            if hld.heavy[p] != v:
                light_count += 1
            v = p
        max_light = max(max_light, light_count)

    print(f"  Max light edges (root→leaf): {max_light} ≤ log₂({n}) = {math.log2(n):.1f}")

    # Verify subtree contiguity
    subtree_ok = True
    for v in range(n):
        # All descendants should have pos in [pos[v], pos[v]+size[v]-1]
        descendants = []
        stack = [v]
        while stack:
            node = stack.pop()
            descendants.append(node)
            for ch in adj[node]:
                if ch != hld.parent[node] and hld.parent[ch] == node:
                    stack.append(ch)
        positions = sorted(hld.pos[d] for d in descendants)
        expected = list(range(hld.pos[v], hld.pos[v] + hld.size[v]))
        if positions != expected:
            subtree_ok = False
            break

    print(f"  Subtrees contiguous: {subtree_ok}")
    print()


# ============================================================
# Exercise 2: Path Sum Queries
# ============================================================
def exercise_2():
    """
    Test path sum queries on different tree shapes.
    """
    print("=== Exercise 2: Path Sum Queries ===\n")

    # Tree 1: Linear chain
    n = 100
    adj = defaultdict(list)
    for i in range(n - 1):
        adj[i].append(i + 1)
        adj[i + 1].append(i)
    values = list(range(1, n + 1))
    hld = HLD(n, adj, values, root=0)

    # Path sum from 0 to 99
    result = hld.path_query(0, 99)
    expected = n * (n + 1) // 2
    print(f"  Chain (100 nodes): path_sum(0,99) = {result} (expected {expected})")

    # Tree 2: Star graph
    n = 50
    adj2 = defaultdict(list)
    for i in range(1, n):
        adj2[0].append(i)
        adj2[i].append(0)
    values2 = [10] * n
    hld2 = HLD(n, adj2, values2, root=0)

    result = hld2.path_query(1, 2)
    print(f"  Star (50 nodes, val=10): path_sum(1,2) = {result} (expected 30)")

    # Tree 3: Balanced binary
    n = 31
    adj3 = defaultdict(list)
    for i in range(1, n):
        p = (i - 1) // 2
        adj3[p].append(i)
        adj3[i].append(p)
    values3 = [1] * n
    hld3 = HLD(n, adj3, values3, root=0)

    # Path from leftmost leaf to rightmost leaf
    left_leaf = 15  # leftmost in complete binary tree
    right_leaf = 30  # rightmost
    result = hld3.path_query(left_leaf, right_leaf)
    print(f"  Binary tree (31 nodes, val=1): path_sum({left_leaf},{right_leaf}) = {result}")
    print()


# ============================================================
# Exercise 3: Path Maximum Query
# ============================================================
def exercise_3():
    """
    Path maximum query: find the maximum value on the path u→v.
    For edge weights, assign weight to the deeper endpoint.
    """
    print("=== Exercise 3: Path Maximum Query ===\n")

    # Tree with edge weights
    #     0
    #    / \
    #   1   2     edge weights: 0-1=5, 0-2=3, 1-3=7, 1-4=2, 2-5=8
    #  / \   \
    # 3   4   5
    n = 6
    adj = defaultdict(list)
    edges = [(0, 1, 5), (0, 2, 3), (1, 3, 7), (1, 4, 2), (2, 5, 8)]
    for u, v, w in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Assign edge weight to deeper node
    # First build tree to get parents
    parent = [-1] * n
    stack = [(0, -1)]
    visited = [False] * n
    visited[0] = True
    while stack:
        node, par = stack.pop()
        parent[node] = par
        for ch in adj[node]:
            if not visited[ch]:
                visited[ch] = True
                stack.append((ch, node))

    values = [0] * n
    for u, v, w in edges:
        if parent[v] == u:
            values[v] = w
        else:
            values[u] = w

    hld = HLD(n, adj, values, root=0, seg_op='max')

    print(f"  Edge weights: {[(u,v,w) for u,v,w in edges]}")
    print(f"  Node values (deeper endpoint): {values}")

    # Path max queries (skip LCA node to avoid root's value)
    queries = [(3, 5), (4, 5), (3, 4)]
    for u, v in queries:
        result = hld.path_query(u, v)
        print(f"  path_max({u}, {v}) = {result}")

    print(f"\n  Note: for edge-weighted path max, exclude LCA node's value")
    print(f"  if it represents the edge above the LCA (not on the path).")
    print()


# ============================================================
# Exercise 4: LCA via HLD
# ============================================================
def exercise_4():
    """
    Find LCA using HLD and verify correctness.
    """
    print("=== Exercise 4: LCA via HLD ===\n")

    # Build a tree
    n = 15
    adj = defaultdict(list)
    for i in range(1, n):
        p = (i - 1) // 2
        adj[p].append(i)
        adj[i].append(p)
    values = list(range(n))
    hld = HLD(n, adj, values, root=0)

    # LCA queries
    test_cases = [
        (7, 8, 3),    # siblings → parent
        (7, 9, 1),    # cousins → grandparent
        (7, 14, 0),   # different subtrees → root
        (3, 4, 1),    # siblings → parent
        (0, 7, 0),    # root and descendant → root
        (5, 6, 2),    # siblings
        (7, 10, 0),   # across subtrees
    ]

    print(f"  Complete binary tree, 15 nodes\n")
    print(f"  {'u':>4} | {'v':>4} | {'LCA':>4} | {'Expected':>9} | {'OK'}")
    print(f"  {'-'*32}")

    for u, v, expected in test_cases:
        result = hld.lca(u, v)
        ok = "✓" if result == expected else "✗"
        print(f"  {u:>4} | {v:>4} | {result:>4} | {expected:>9} | {ok}")

    print()


# ============================================================
# Exercise 5: Subtree Sum with Point Updates
# ============================================================
def exercise_5():
    """
    Support subtree sum queries and point updates using HLD.
    """
    print("=== Exercise 5: Subtree Queries with Updates ===\n")

    #     0(10)
    #    / \
    #   1(20) 2(30)
    #  / \     \
    # 3(5) 4(15) 5(25)
    n = 6
    adj = defaultdict(list)
    for u, v in [(0,1),(0,2),(1,3),(1,4),(2,5)]:
        adj[u].append(v)
        adj[v].append(u)
    values = [10, 20, 30, 5, 15, 25]

    hld = HLD(n, adj, values, root=0)

    # Initial subtree sums
    print(f"  Initial values: {values}\n")
    for v in range(n):
        s = hld.subtree_query(v)
        print(f"  subtree_sum({v}) = {s}")

    # Update node 3: 5 → 50
    print(f"\n  Update: node 3 value = 50")
    hld.update(3, 50)

    for v in [0, 1, 3]:
        s = hld.subtree_query(v)
        print(f"  subtree_sum({v}) = {s}")

    # Verify: subtree(0) should be 10+20+30+50+15+25 = 150
    total = sum([10, 20, 30, 50, 15, 25])
    actual = hld.subtree_query(0)
    print(f"\n  Total (subtree root): {actual} (expected {total})")
    print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
