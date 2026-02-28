# Persistent Segment Tree

**Previous**: [Link-Cut Tree](./31_Link_Cut_Tree.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the concept of persistence in data structures and distinguish between partial and full persistence
2. Implement a persistent segment tree using path copying, creating new versions in O(log N) time and space
3. Solve the "K-th smallest element in a range" problem using persistent prefix segment trees
4. Apply persistent segment trees to versioned array problems and offline query processing
5. Analyze the space complexity of persistent structures and apply memory optimization techniques

---

## Table of Contents

1. [What Is Persistence?](#1-what-is-persistence)
2. [Path Copying Technique](#2-path-copying-technique)
3. [Persistent Segment Tree: Build and Update](#3-persistent-segment-tree-build-and-update)
4. [K-th Smallest in a Range](#4-k-th-smallest-in-a-range)
5. [Version Queries](#5-version-queries)
6. [Implementation](#6-implementation)
7. [Advanced Applications](#7-advanced-applications)
8. [Exercises](#8-exercises)

---

## 1. What Is Persistence?

A **persistent data structure** preserves all previous versions of itself when modified. Instead of overwriting data, it creates a new version while keeping old versions accessible.

| Type | Access Old | Modify Old |
|------|-----------|------------|
| Ephemeral (normal) | No | No |
| Partially persistent | Yes (read) | No |
| Fully persistent | Yes (read) | Yes (branch) |

**Key insight**: if updates touch O(log N) nodes (like in a segment tree), we only need to copy those O(log N) nodes — the rest is shared with the previous version.

### Motivation: Why Not Just Copy?

Copying the entire segment tree for each version costs O(N) per update. With Q updates, total space is O(NQ). With path copying, each update costs only O(log N), giving O(N + Q log N) total space.

---

## 2. Path Copying Technique

When updating a single leaf in a segment tree, only the path from root to that leaf changes (O(log N) nodes). We create new copies of only those nodes:

```
Version 0:          Version 1 (update leaf 3):

      [1,4]              [1,4]'
      /    \             /    \
   [1,2]  [3,4]     [1,2]  [3,4]'    ← shared [1,2]
   /  \   /  \      /  \   /  \
  1   2  3   4     1   2  3'  4      ← shared 1, 2, 4
                              ↑
                            new value
```

Nodes marked with `'` are newly created. Nodes without `'` are shared with version 0.

**Space per update**: O(log N) new nodes.
**Time per update**: O(log N) (same as normal segment tree).

---

## 3. Persistent Segment Tree: Build and Update

### Node Structure

```python
class Node:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

Unlike array-based segment trees, persistent segment trees use **pointer-based** representation (since nodes are shared across versions).

### Build (Version 0)

```python
def build(arr, lo, hi):
    if lo == hi:
        return Node(val=arr[lo])
    mid = (lo + hi) // 2
    left = build(arr, lo, mid)
    right = build(arr, mid + 1, hi)
    return Node(val=left.val + right.val, left=left, right=right)
```

### Update (Creates New Version)

```python
def update(node, lo, hi, idx, val):
    if lo == hi:
        return Node(val=val)
    mid = (lo + hi) // 2
    if idx <= mid:
        new_left = update(node.left, lo, mid, idx, val)
        return Node(val=new_left.val + node.right.val,
                    left=new_left, right=node.right)  # share right
    else:
        new_right = update(node.right, mid + 1, hi, idx, val)
        return Node(val=node.left.val + new_right.val,
                    left=node.left, right=new_right)  # share left
```

### Query (Same as Normal)

```python
def query(node, lo, hi, ql, qr):
    if node is None or qr < lo or hi < ql:
        return 0
    if ql <= lo and hi <= qr:
        return node.val
    mid = (lo + hi) // 2
    return (query(node.left, lo, mid, ql, qr) +
            query(node.right, mid + 1, hi, ql, qr))
```

### Version Management

```python
roots = []  # roots[i] = root of version i

# Build initial version
roots.append(build(arr, 0, N - 1))

# Create new version by updating index 3 to value 10
roots.append(update(roots[-1], 0, N - 1, 3, 10))

# Query version 0 (original) or version 1 (updated)
query(roots[0], 0, N - 1, 0, 4)  # uses old values
query(roots[1], 0, N - 1, 0, 4)  # uses new values
```

---

## 4. K-th Smallest in a Range

The classic application of persistent segment trees: find the K-th smallest element in a subarray `arr[l..r]`.

### Approach: Persistent Prefix Segment Trees

1. **Coordinate compression**: map values to ranks in [0, M-1]
2. **Build persistent prefix trees**: `roots[i]` represents a segment tree counting frequencies of ranks in `arr[0..i-1]`
3. **Range query**: the "difference" `roots[r+1] - roots[l]` gives frequencies in `arr[l..r]`
4. **Walk the tree**: find the K-th element by binary searching on accumulated counts

```python
def kth_smallest(root_l, root_r, lo, hi, k):
    """Find k-th smallest in the range represented by root_r - root_l."""
    if lo == hi:
        return lo  # this rank is the answer
    mid = (lo + hi) // 2
    left_count = root_r.left.val - root_l.left.val
    if k <= left_count:
        return kth_smallest(root_l.left, root_r.left, lo, mid, k)
    else:
        return kth_smallest(root_l.right, root_r.right,
                            mid + 1, hi, k - left_count)
```

### Complete Algorithm

```python
def solve_kth_range(arr, queries):
    N = len(arr)

    # Step 1: coordinate compression
    sorted_vals = sorted(set(arr))
    rank = {v: i for i, v in enumerate(sorted_vals)}
    M = len(sorted_vals)

    # Step 2: build persistent prefix trees
    roots = [build_empty(0, M - 1)]  # roots[0] = empty tree
    for i in range(N):
        # Insert rank of arr[i] into the next version
        roots.append(insert(roots[-1], 0, M - 1, rank[arr[i]]))

    # Step 3: answer queries
    results = []
    for l, r, k in queries:
        rank_idx = kth_smallest(roots[l], roots[r + 1], 0, M - 1, k)
        results.append(sorted_vals[rank_idx])

    return results
```

### Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Build all versions | O(N log M) | O(N log M) |
| Per query | O(log M) | — |
| Total | O((N + Q) log M) | O(N log M) |

---

## 5. Version Queries

Persistent segment trees naturally support versioned operations:

```python
# Version history:
# v0: initial array [1, 2, 3, 4, 5]
# v1: set index 2 to 10 → [1, 2, 10, 4, 5]
# v2: set index 0 to 7 → [7, 2, 10, 4, 5]
# v3: based on v1, set index 4 to 9 → [1, 2, 10, 4, 9] (branching!)

roots = [build([1, 2, 3, 4, 5], 0, 4)]
roots.append(update(roots[0], 0, 4, 2, 10))  # v1
roots.append(update(roots[1], 0, 4, 0, 7))   # v2
roots.append(update(roots[1], 0, 4, 4, 9))   # v3 (branches from v1!)

# All four versions are accessible simultaneously
for i, root in enumerate(roots):
    print(f"v{i}: sum = {query(root, 0, 4, 0, 4)}")
```

This is **full persistence** — we can branch from any version.

---

## 6. Implementation

### Complete Python Implementation

```python
class PersistentSegTree:
    """Persistent Segment Tree with point update and range query."""

    class Node:
        __slots__ = ('val', 'left', 'right')
        def __init__(self, val=0, left=None, right=None):
            self.val = val
            self.left = left
            self.right = right

    def __init__(self, n):
        self.n = n
        self.EMPTY = self.Node()  # sentinel

    def _build(self, arr, lo, hi):
        if lo == hi:
            return self.Node(val=arr[lo] if lo < len(arr) else 0)
        mid = (lo + hi) // 2
        left = self._build(arr, lo, mid)
        right = self._build(arr, mid + 1, hi)
        return self.Node(val=left.val + right.val, left=left, right=right)

    def build(self, arr):
        """Build initial version from array. Returns root."""
        return self._build(arr, 0, self.n - 1)

    def build_empty(self):
        """Build an empty tree (all zeros). Returns root."""
        return self._build_empty(0, self.n - 1)

    def _build_empty(self, lo, hi):
        if lo == hi:
            return self.Node(val=0)
        mid = (lo + hi) // 2
        left = self._build_empty(lo, mid)
        right = self._build_empty(mid + 1, hi)
        return self.Node(val=0, left=left, right=right)

    def update(self, root, idx, val):
        """Create a new version with arr[idx] = val. Returns new root."""
        return self._update(root, 0, self.n - 1, idx, val)

    def _update(self, node, lo, hi, idx, val):
        if lo == hi:
            return self.Node(val=val)
        mid = (lo + hi) // 2
        if idx <= mid:
            new_left = self._update(
                node.left or self.EMPTY, lo, mid, idx, val)
            right = node.right or self.EMPTY
            return self.Node(val=new_left.val + right.val,
                             left=new_left, right=right)
        else:
            left = node.left or self.EMPTY
            new_right = self._update(
                node.right or self.EMPTY, mid + 1, hi, idx, val)
            return self.Node(val=left.val + new_right.val,
                             left=left, right=new_right)

    def insert(self, root, idx):
        """Increment count at idx (for frequency trees). Returns new root."""
        return self._insert(root, 0, self.n - 1, idx)

    def _insert(self, node, lo, hi, idx):
        if lo == hi:
            return self.Node(val=(node.val if node else 0) + 1)
        mid = (lo + hi) // 2
        left = node.left if node else None
        right = node.right if node else None
        if idx <= mid:
            new_left = self._insert(left, lo, mid, idx)
            r_val = right.val if right else 0
            return self.Node(val=new_left.val + r_val,
                             left=new_left, right=right)
        else:
            new_right = self._insert(right, mid + 1, hi, idx)
            l_val = left.val if left else 0
            return self.Node(val=l_val + new_right.val,
                             left=left, right=new_right)

    def query(self, root, ql, qr):
        """Range sum query on [ql, qr]."""
        return self._query(root, 0, self.n - 1, ql, qr)

    def _query(self, node, lo, hi, ql, qr):
        if node is None or qr < lo or hi < ql:
            return 0
        if ql <= lo and hi <= qr:
            return node.val
        mid = (lo + hi) // 2
        return (self._query(node.left, lo, mid, ql, qr) +
                self._query(node.right, mid + 1, hi, ql, qr))

    def kth(self, root_l, root_r, k):
        """Find k-th smallest in range [l, r] using prefix trees."""
        return self._kth(root_l, root_r, 0, self.n - 1, k)

    def _kth(self, nl, nr, lo, hi, k):
        if lo == hi:
            return lo
        mid = (lo + hi) // 2
        left_count = ((nr.left.val if nr.left else 0) -
                      (nl.left.val if nl.left else 0))
        if k <= left_count:
            return self._kth(nl.left or self.EMPTY, nr.left or self.EMPTY,
                             lo, mid, k)
        else:
            return self._kth(nl.right or self.EMPTY, nr.right or self.EMPTY,
                             mid + 1, hi, k - left_count)


# Demo: K-th smallest in range
def demo_kth_range():
    arr = [5, 1, 3, 2, 4, 7, 6, 8]
    N = len(arr)

    # Coordinate compression
    sorted_vals = sorted(set(arr))
    rank = {v: i for i, v in enumerate(sorted_vals)}
    M = len(sorted_vals)

    pst = PersistentSegTree(M)

    # Build prefix trees
    roots = [pst.build_empty()]
    for val in arr:
        roots.append(pst.insert(roots[-1], rank[val]))

    # Query: 2nd smallest in arr[1..5] = [1,3,2,4,7]
    # Sorted: [1,2,3,4,7], 2nd = 2
    l, r, k = 1, 5, 2
    rank_idx = pst.kth(roots[l], roots[r + 1], k)
    print(f"  {k}-th smallest in arr[{l}..{r}] = {sorted_vals[rank_idx]}")

    # Query: 4th smallest in arr[0..7] = [5,1,3,2,4,7,6,8]
    # Sorted: [1,2,3,4,5,6,7,8], 4th = 4
    l, r, k = 0, 7, 4
    rank_idx = pst.kth(roots[l], roots[r + 1], k)
    print(f"  {k}-th smallest in arr[{l}..{r}] = {sorted_vals[rank_idx]}")
```

### Complexity Summary

| Operation | Time | Space |
|-----------|------|-------|
| Build initial version | O(N) | O(N) |
| Update (new version) | O(log N) | O(log N) new nodes |
| Query any version | O(log N) | — |
| K versions total | — | O(N + K log N) |
| K-th smallest setup | O(N log M) | O(N log M) |
| K-th smallest query | O(log M) | — |

---

## 7. Advanced Applications

### Persistent Array

Use a persistent segment tree as a persistent array:
- `get(version, index)`: query at a single point
- `set(version, index, value)`: update returns new version

Supports any number of branches and rollbacks.

### Count of Distinct Elements in Range

Combine persistent segment trees with offline sweepline:
1. Process array left to right
2. When encountering value v at position i, if v was last seen at position j, remove j and add i
3. `distinct[l..r] = query(roots[r+1], l, N-1)` on the persistent tree

### 2D Range Queries (Offline)

Persistent segment trees can simulate 2D queries:
- Sort events by one dimension
- Build persistent tree along the other dimension
- Binary search on versions for range queries

---

## 8. Exercises

### Exercise 1: Persistent Array

Implement a persistent array supporting:
1. `create(arr)`: initial version
2. `get(version, idx)`: read element
3. `set(version, idx, val)`: create new version with modified element
4. Demonstrate version branching: modify version 0 twice to create two independent branches

### Exercise 2: K-th Smallest in Range

Implement the full K-th smallest algorithm:
1. Coordinate compression
2. Build persistent prefix trees
3. Answer Q queries of the form "K-th smallest in arr[l..r]"
4. Test on arrays of size up to 10⁵

### Exercise 3: Version Control Simulator

Build a simple version control system:
1. `commit(changes)`: create new version with changes
2. `checkout(version)`: read any past version
3. `branch(version)`: create a new branch from a past version
4. `diff(v1, v2)`: find positions that differ between two versions

### Exercise 4: Range Frequency Query

Given an array, answer queries: "How many times does value x appear in arr[l..r]?"
1. Use coordinate compression
2. Build persistent prefix trees
3. Query the count at rank of x in `roots[r+1] - roots[l]`

### Exercise 5: Persistent Union-Find

Implement a partially persistent Union-Find:
1. After each union operation, save a version
2. Query connectivity at any past version
3. Use persistent segment tree to store parent array

*Hint*: this is a challenging problem. Use union by rank (without path compression) to keep the tree height O(log N).

---

## Navigation

**Previous**: [Link-Cut Tree](./31_Link_Cut_Tree.md)
