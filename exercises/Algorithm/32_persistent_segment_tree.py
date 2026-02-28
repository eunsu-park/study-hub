"""
Exercises for Lesson 32: Persistent Segment Tree
Topic: Algorithm

Solutions to practice problems covering persistent arrays,
K-th smallest in range, version control simulation,
range frequency queries, and space analysis.
"""


# ============================================================
# Shared: Persistent Segment Tree
# ============================================================
class Node:
    __slots__ = ('val', 'left', 'right')
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class PST:
    """Persistent Segment Tree."""

    def __init__(self, n):
        self.n = n
        self.node_count = 0

    def build(self, arr):
        return self._build(arr, 0, self.n - 1)

    def _build(self, arr, lo, hi):
        self.node_count += 1
        if lo == hi:
            return Node(val=arr[lo] if lo < len(arr) else 0)
        mid = (lo + hi) // 2
        left = self._build(arr, lo, mid)
        right = self._build(arr, mid + 1, hi)
        return Node(val=left.val + right.val, left=left, right=right)

    def build_empty(self):
        return self._build([], 0, self.n - 1)

    def update(self, root, idx, val):
        return self._update(root, 0, self.n - 1, idx, val)

    def _update(self, node, lo, hi, idx, val):
        self.node_count += 1
        if lo == hi:
            return Node(val=val)
        mid = (lo + hi) // 2
        if idx <= mid:
            nl = self._update(node.left, lo, mid, idx, val)
            return Node(val=nl.val + node.right.val, left=nl, right=node.right)
        else:
            nr = self._update(node.right, mid + 1, hi, idx, val)
            return Node(val=node.left.val + nr.val, left=node.left, right=nr)

    def insert(self, root, idx):
        return self._insert(root, 0, self.n - 1, idx)

    def _insert(self, node, lo, hi, idx):
        self.node_count += 1
        if lo == hi:
            return Node(val=node.val + 1)
        mid = (lo + hi) // 2
        if idx <= mid:
            nl = self._insert(node.left, lo, mid, idx)
            return Node(val=nl.val + node.right.val, left=nl, right=node.right)
        else:
            nr = self._insert(node.right, mid + 1, hi, idx)
            return Node(val=node.left.val + nr.val, left=node.left, right=nr)

    def query(self, root, ql, qr):
        return self._query(root, 0, self.n - 1, ql, qr)

    def _query(self, node, lo, hi, ql, qr):
        if node is None or qr < lo or hi < ql: return 0
        if ql <= lo and hi <= qr: return node.val
        mid = (lo + hi) // 2
        return (self._query(node.left, lo, mid, ql, qr) +
                self._query(node.right, mid + 1, hi, ql, qr))

    def kth(self, nl, nr, k):
        return self._kth(nl, nr, 0, self.n - 1, k)

    def _kth(self, nl, nr, lo, hi, k):
        if lo == hi: return lo
        mid = (lo + hi) // 2
        left_count = nr.left.val - nl.left.val
        if k <= left_count:
            return self._kth(nl.left, nr.left, lo, mid, k)
        else:
            return self._kth(nl.right, nr.right, mid + 1, hi, k - left_count)


# ============================================================
# Exercise 1: Persistent Array
# ============================================================
def exercise_1():
    """
    Implement a persistent array with version branching.
    """
    print("=== Exercise 1: Persistent Array ===\n")

    arr = [10, 20, 30, 40, 50]
    n = len(arr)
    pst = PST(n)

    versions = {"v0": pst.build(arr)}
    print(f"  v0: {arr}")

    # v1: modify index 2 → 99
    versions["v1"] = pst.update(versions["v0"], 2, 99)
    v1_arr = [pst.query(versions["v1"], i, i) for i in range(n)]
    print(f"  v1 (from v0, set [2]=99): {v1_arr}")

    # v2: modify index 0 → 77 (based on v1)
    versions["v2"] = pst.update(versions["v1"], 0, 77)
    v2_arr = [pst.query(versions["v2"], i, i) for i in range(n)]
    print(f"  v2 (from v1, set [0]=77): {v2_arr}")

    # v3: branch from v0 (not v2!), modify index 4 → 88
    versions["v3"] = pst.update(versions["v0"], 4, 88)
    v3_arr = [pst.query(versions["v3"], i, i) for i in range(n)]
    print(f"  v3 (from v0, set [4]=88): {v3_arr}")

    # Verify all versions still accessible
    print(f"\n  All versions accessible:")
    for name, root in sorted(versions.items()):
        values = [pst.query(root, i, i) for i in range(n)]
        total = pst.query(root, 0, n - 1)
        print(f"    {name}: {values}, sum={total}")

    print(f"\n  Nodes created: {pst.node_count}")
    print()


# ============================================================
# Exercise 2: K-th Smallest in Range
# ============================================================
def exercise_2():
    """
    Find the K-th smallest element in arr[l..r].
    """
    print("=== Exercise 2: K-th Smallest in Range ===\n")

    arr = [7, 2, 5, 1, 8, 3, 6, 4, 9, 10]
    n = len(arr)
    print(f"  Array: {arr}\n")

    # Coordinate compression
    sorted_vals = sorted(set(arr))
    rank = {v: i for i, v in enumerate(sorted_vals)}
    M = len(sorted_vals)

    pst = PST(M)
    roots = [pst.build_empty()]
    for val in arr:
        roots.append(pst.insert(roots[-1], rank[val]))

    # Queries
    queries = [
        (0, 9, 1, "smallest in entire array"),
        (0, 9, 5, "5th smallest (median)"),
        (0, 9, 10, "largest in entire array"),
        (2, 6, 1, "smallest in arr[2..6]"),
        (2, 6, 3, "3rd smallest in arr[2..6]"),
        (0, 4, 2, "2nd smallest in arr[0..4]"),
    ]

    for l, r, k, desc in queries:
        rank_idx = pst.kth(roots[l], roots[r + 1], k)
        val = sorted_vals[rank_idx]
        subarray = sorted(arr[l:r + 1])
        expected = subarray[k - 1]
        ok = val == expected
        print(f"  {desc}:")
        print(f"    arr[{l}..{r}] sorted = {subarray}")
        print(f"    k={k}: got {val}, expected {expected} {'✓' if ok else '✗'}")
    print()


# ============================================================
# Exercise 3: Version Control Simulator
# ============================================================
def exercise_3():
    """
    Build a version control system with commit, checkout, branch, diff.
    """
    print("=== Exercise 3: Version Control Simulator ===\n")

    class VCS:
        def __init__(self, initial_data):
            self.n = len(initial_data)
            self.pst = PST(self.n)
            self.commits = [self.pst.build(initial_data)]
            self.messages = ["Initial commit"]
            self.branch_from = [None]

        def commit(self, base_version, changes, message):
            """Create new commit based on base_version with changes."""
            root = self.commits[base_version]
            for idx, val in changes:
                root = self.pst.update(root, idx, val)
            self.commits.append(root)
            self.messages.append(message)
            self.branch_from.append(base_version)
            return len(self.commits) - 1

        def checkout(self, version):
            """Read all values at a given version."""
            return [self.pst.query(self.commits[version], i, i)
                    for i in range(self.n)]

        def diff(self, v1, v2):
            """Find indices that differ between two versions."""
            diffs = []
            for i in range(self.n):
                a = self.pst.query(self.commits[v1], i, i)
                b = self.pst.query(self.commits[v2], i, i)
                if a != b:
                    diffs.append((i, a, b))
            return diffs

    vcs = VCS([1, 2, 3, 4, 5])

    v1 = vcs.commit(0, [(1, 20), (3, 40)], "Update indices 1 and 3")
    v2 = vcs.commit(v1, [(0, 10)], "Update index 0")
    v3 = vcs.commit(0, [(4, 50)], "Branch from v0: update index 4")

    print(f"  Commit history:")
    for i, msg in enumerate(vcs.messages):
        base = vcs.branch_from[i]
        base_str = f"(from v{base})" if base is not None else "(root)"
        print(f"    v{i}: {msg} {base_str}")

    print(f"\n  Checkout each version:")
    for i in range(len(vcs.commits)):
        data = vcs.checkout(i)
        print(f"    v{i}: {data}")

    print(f"\n  Diff v0 vs v2:")
    for idx, old, new in vcs.diff(0, v2):
        print(f"    index {idx}: {old} → {new}")

    print(f"\n  Diff v2 vs v3 (divergent branches):")
    for idx, a, b in vcs.diff(v2, v3):
        print(f"    index {idx}: v2={a}, v3={b}")
    print()


# ============================================================
# Exercise 4: Range Frequency Query
# ============================================================
def exercise_4():
    """
    Count occurrences of a value in arr[l..r] using persistent segment tree.
    """
    print("=== Exercise 4: Range Frequency Query ===\n")

    arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9]
    n = len(arr)
    print(f"  Array ({n} elements): {arr}\n")

    # Build persistent prefix trees
    sorted_vals = sorted(set(arr))
    rank = {v: i for i, v in enumerate(sorted_vals)}
    M = len(sorted_vals)

    pst = PST(M)
    roots = [pst.build_empty()]
    for val in arr:
        roots.append(pst.insert(roots[-1], rank[val]))

    def count_value(l, r, val):
        if val not in rank: return 0
        idx = rank[val]
        return pst.query(roots[r + 1], idx, idx) - pst.query(roots[l], idx, idx)

    # Queries
    queries = [
        (0, 14, 9, "count 9 in full array"),
        (0, 14, 5, "count 5 in full array"),
        (0, 5, 1, "count 1 in arr[0..5]"),
        (6, 14, 9, "count 9 in arr[6..14]"),
        (0, 14, 10, "count 10 (not present)"),
    ]

    for l, r, val, desc in queries:
        cnt = count_value(l, r, val)
        expected = arr[l:r+1].count(val)
        ok = cnt == expected
        print(f"  {desc}: {cnt} {'✓' if ok else '✗'}")

    # Mode query: most frequent element in arr[l..r]
    print(f"\n  Mode query (most frequent in arr[0..14]):")
    best_val, best_cnt = None, 0
    for val in sorted_vals:
        cnt = count_value(0, 14, val)
        if cnt > best_cnt:
            best_cnt = cnt
            best_val = val
    print(f"    Mode = {best_val} (appears {best_cnt} times)")
    print()


# ============================================================
# Exercise 5: Space Complexity Analysis
# ============================================================
def exercise_5():
    """
    Measure and verify the space complexity of persistent segment trees.
    """
    print("=== Exercise 5: Space Complexity Analysis ===\n")

    import math

    test_sizes = [100, 500, 1000, 5000, 10000]

    print(f"  {'N':>8} | {'Build':>8} | {'100 Updates':>12} | "
          f"{'Predicted':>10} | {'Match':>6}")
    print(f"  {'-'*50}")

    for n in test_sizes:
        pst = PST(n)
        arr = list(range(n))
        root = pst.build(arr)
        build_nodes = pst.node_count

        roots = [root]
        for i in range(100):
            idx = i % n
            roots.append(pst.update(roots[-1], idx, idx * 2))

        total_nodes = pst.node_count
        update_nodes = total_nodes - build_nodes

        # Predicted: build = 2N-1, per update = log2(N)+1
        pred_build = 2 * n - 1
        pred_per_update = int(math.log2(n)) + 1
        pred_total = pred_build + 100 * pred_per_update

        match = abs(total_nodes - pred_total) / pred_total < 0.1

        print(f"  {n:>8,} | {build_nodes:>8,} | {update_nodes:>12,} | "
              f"{pred_total:>10,} | {'~' if match else '!'}")

    # Compare with full copy
    print(f"\n  Full copy vs path copying (N=10000, 100 updates):")
    n = 10000
    full_copy = 101 * (2 * n - 1)
    path_copy = (2 * n - 1) + 100 * (int(math.log2(n)) + 1)
    ratio = full_copy / path_copy

    print(f"    Full copy: {full_copy:,} nodes")
    print(f"    Path copy: {path_copy:,} nodes")
    print(f"    Savings: {ratio:.0f}x fewer nodes ({(1 - path_copy/full_copy)*100:.1f}% reduction)")
    print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
