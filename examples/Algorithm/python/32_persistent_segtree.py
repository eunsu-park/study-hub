"""
Example: Persistent Segment Tree
Topic: Algorithm – Lesson 32

Demonstrates:
  1. Path copying for persistent updates
  2. Version management (access any past version)
  3. K-th smallest in a range query
  4. Space analysis of persistent structures

Run: python 32_persistent_segtree.py
"""


# ============================================================
# Persistent Segment Tree
# ============================================================
class Node:
    """Persistent segment tree node (immutable after creation)."""
    __slots__ = ('val', 'left', 'right')

    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class PersistentSegTree:
    """Persistent segment tree supporting point updates and range queries.

    Each update creates O(log N) new nodes via path copying.
    All versions remain accessible.
    """

    def __init__(self, n):
        self.n = n
        self._node_count = 0

    def build(self, arr):
        """Build initial version from array. Returns root node."""
        return self._build(arr, 0, self.n - 1)

    def _build(self, arr, lo, hi):
        self._node_count += 1
        if lo == hi:
            return Node(val=arr[lo] if lo < len(arr) else 0)
        mid = (lo + hi) // 2
        left = self._build(arr, lo, mid)
        right = self._build(arr, mid + 1, hi)
        return Node(val=left.val + right.val, left=left, right=right)

    def build_empty(self):
        """Build an empty tree (all zeros)."""
        return self._build_empty(0, self.n - 1)

    def _build_empty(self, lo, hi):
        self._node_count += 1
        if lo == hi:
            return Node(val=0)
        mid = (lo + hi) // 2
        left = self._build_empty(lo, mid)
        right = self._build_empty(mid + 1, hi)
        return Node(val=0, left=left, right=right)

    def update(self, root, idx, val):
        """Create new version with arr[idx] set to val. Returns new root."""
        return self._update(root, 0, self.n - 1, idx, val)

    def _update(self, node, lo, hi, idx, val):
        self._node_count += 1
        if lo == hi:
            return Node(val=val)
        mid = (lo + hi) // 2
        if idx <= mid:
            new_left = self._update(node.left, lo, mid, idx, val)
            return Node(val=new_left.val + node.right.val,
                        left=new_left, right=node.right)  # share right
        else:
            new_right = self._update(node.right, mid + 1, hi, idx, val)
            return Node(val=node.left.val + new_right.val,
                        left=node.left, right=new_right)  # share left

    def insert(self, root, idx):
        """Increment count at idx. For frequency-based trees."""
        return self._insert(root, 0, self.n - 1, idx)

    def _insert(self, node, lo, hi, idx):
        self._node_count += 1
        if lo == hi:
            return Node(val=node.val + 1)
        mid = (lo + hi) // 2
        if idx <= mid:
            new_left = self._insert(node.left, lo, mid, idx)
            return Node(val=new_left.val + node.right.val,
                        left=new_left, right=node.right)
        else:
            new_right = self._insert(node.right, mid + 1, hi, idx)
            return Node(val=node.left.val + new_right.val,
                        left=node.left, right=new_right)

    def query(self, root, ql, qr):
        """Range sum query on [ql, qr] for given version."""
        return self._query(root, 0, self.n - 1, ql, qr)

    def _query(self, node, lo, hi, ql, qr):
        if node is None or qr < lo or hi < ql:
            return 0
        if ql <= lo and hi <= qr:
            return node.val
        mid = (lo + hi) // 2
        return (self._query(node.left, lo, mid, ql, qr) +
                self._query(node.right, mid + 1, hi, ql, qr))

    def kth_smallest(self, root_l, root_r, k):
        """Find k-th smallest in range [l, r] using prefix trees."""
        return self._kth(root_l, root_r, 0, self.n - 1, k)

    def _kth(self, nl, nr, lo, hi, k):
        if lo == hi:
            return lo
        mid = (lo + hi) // 2
        left_count = nr.left.val - nl.left.val
        if k <= left_count:
            return self._kth(nl.left, nr.left, lo, mid, k)
        else:
            return self._kth(nl.right, nr.right, mid + 1, hi, k - left_count)


# ============================================================
# Demo 1: Version Management
# ============================================================
def demo_versions():
    print("=" * 60)
    print("Demo 1: Persistent Array (Version Management)")
    print("=" * 60)

    arr = [1, 2, 3, 4, 5]
    n = len(arr)
    pst = PersistentSegTree(n)

    # Version 0: original
    roots = [pst.build(arr)]
    print(f"\n  v0: {arr}")

    # Version 1: set index 2 to 10
    roots.append(pst.update(roots[0], 2, 10))
    print(f"  v1: set [2]=10 → ", end="")
    print([pst.query(roots[1], i, i) for i in range(n)])

    # Version 2: set index 0 to 7 (based on v1)
    roots.append(pst.update(roots[1], 0, 7))
    print(f"  v2: set [0]=7  → ", end="")
    print([pst.query(roots[2], i, i) for i in range(n)])

    # Version 3: branch from v0, set index 4 to 9
    roots.append(pst.update(roots[0], 4, 9))
    print(f"  v3: (from v0) set [4]=9 → ", end="")
    print([pst.query(roots[3], i, i) for i in range(n)])

    # All versions are still accessible
    print(f"\n  Range sums [0..4] across versions:")
    for i, root in enumerate(roots):
        s = pst.query(root, 0, n - 1)
        print(f"    v{i}: sum = {s}")

    print(f"\n  Total nodes created: {pst._node_count}")
    print(f"  Without persistence: {4 * (2 * n - 1)} nodes (4 full copies)")
    print(f"  Savings: path copying shares ~{100 - 100 * pst._node_count / (4 * (2*n-1)):.0f}% of nodes")
    print()


# ============================================================
# Demo 2: K-th Smallest in a Range
# ============================================================
def demo_kth_range():
    print("=" * 60)
    print("Demo 2: K-th Smallest in a Range")
    print("=" * 60)

    arr = [5, 1, 3, 8, 2, 7, 4, 6]
    n = len(arr)

    print(f"\n  Array: {arr}")

    # Step 1: coordinate compression
    sorted_vals = sorted(set(arr))
    rank = {v: i for i, v in enumerate(sorted_vals)}
    M = len(sorted_vals)

    print(f"  Sorted unique: {sorted_vals}")
    print(f"  Rank mapping: {rank}")

    # Step 2: build persistent prefix trees
    pst = PersistentSegTree(M)
    roots = [pst.build_empty()]

    for i, val in enumerate(arr):
        roots.append(pst.insert(roots[-1], rank[val]))

    # Step 3: answer queries
    queries = [
        (0, 7, 1, "1st smallest in arr[0..7]"),
        (0, 7, 4, "4th smallest in arr[0..7]"),
        (0, 7, 8, "8th smallest in arr[0..7]"),
        (1, 5, 1, "1st smallest in arr[1..5]"),
        (1, 5, 3, "3rd smallest in arr[1..5]"),
        (2, 4, 2, "2nd smallest in arr[2..4]"),
    ]

    print(f"\n  Queries:")
    for l, r, k, desc in queries:
        rank_idx = pst.kth_smallest(roots[l], roots[r + 1], k)
        val = sorted_vals[rank_idx]
        subarray = arr[l:r + 1]
        sorted_sub = sorted(subarray)
        print(f"    {desc}: {val}")
        print(f"      arr[{l}..{r}] = {subarray}, sorted = {sorted_sub}")

    print()


# ============================================================
# Demo 3: Space Analysis
# ============================================================
def demo_space():
    print("=" * 60)
    print("Demo 3: Space Analysis")
    print("=" * 60)

    import math

    sizes = [100, 1000, 10000, 100000]

    print(f"\n  {'N':>8} | {'Build':>8} | {'Per Update':>11} | "
          f"{'100 Updates':>12} | {'Full Copy':>10} | {'Savings':>8}")
    print(f"  {'-'*62}")

    for n in sizes:
        build_nodes = 2 * n - 1  # full binary tree
        per_update = int(math.log2(n)) + 1  # path length
        total_100 = build_nodes + 100 * per_update
        full_copy = 101 * build_nodes  # 1 build + 100 copies
        savings = (1 - total_100 / full_copy) * 100

        print(f"  {n:>8,} | {build_nodes:>8,} | {per_update:>11} | "
              f"{total_100:>12,} | {full_copy:>10,} | {savings:>7.1f}%")

    print(f"\n  Path copying creates O(log N) nodes per update.")
    print(f"  Total space for K updates: O(N + K log N).")
    print(f"  Full copying would use O(NK) — exponentially worse.")
    print()


# ============================================================
# Demo 4: Range Count of Values
# ============================================================
def demo_range_count():
    print("=" * 60)
    print("Demo 4: Count Values in Range")
    print("=" * 60)

    arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    n = len(arr)

    print(f"\n  Array: {arr}")

    # Build persistent prefix trees
    sorted_vals = sorted(set(arr))
    rank = {v: i for i, v in enumerate(sorted_vals)}
    M = len(sorted_vals)

    pst = PersistentSegTree(M)
    roots = [pst.build_empty()]
    for val in arr:
        roots.append(pst.insert(roots[-1], rank[val]))

    # Count occurrences of a value in a range
    def count_in_range(l, r, val):
        if val not in rank:
            return 0
        idx = rank[val]
        return (pst.query(roots[r + 1], idx, idx) -
                pst.query(roots[l], idx, idx))

    # Count values in range [lo, hi] within arr[l..r]
    def count_between(l, r, lo, hi):
        lo_rank = 0
        hi_rank = M - 1
        for i, v in enumerate(sorted_vals):
            if v >= lo:
                lo_rank = i
                break
        for i in range(M - 1, -1, -1):
            if sorted_vals[i] <= hi:
                hi_rank = i
                break
        return (pst.query(roots[r + 1], lo_rank, hi_rank) -
                pst.query(roots[l], lo_rank, hi_rank))

    print(f"\n  Value frequency queries:")
    for l, r, val in [(0, 10, 5), (0, 4, 1), (5, 10, 3)]:
        cnt = count_in_range(l, r, val)
        print(f"    count({val}) in arr[{l}..{r}] = {cnt}")

    print(f"\n  Range count queries:")
    for l, r, lo, hi in [(0, 10, 3, 5), (0, 4, 1, 4), (3, 8, 2, 6)]:
        cnt = count_between(l, r, lo, hi)
        subarray = arr[l:r + 1]
        expected = sum(1 for x in subarray if lo <= x <= hi)
        print(f"    count({lo}≤v≤{hi}) in arr[{l}..{r}] = {cnt} "
              f"(verify: {expected})")

    print()


if __name__ == "__main__":
    demo_versions()
    demo_kth_range()
    demo_space()
    demo_range_count()
