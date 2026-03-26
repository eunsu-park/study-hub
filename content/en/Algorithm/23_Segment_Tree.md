# Segment Tree

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the structure of a segment tree, including how nodes represent intervals and why the height is O(log n)
2. Build a segment tree in O(n) time and implement point updates and range sum/min/max queries in O(log n)
3. Implement lazy propagation to support range updates efficiently in O(log n) time
4. Apply segment trees to solve problems that require repeated range queries with interleaved updates
5. Analyze the trade-offs between segment trees and simpler structures like prefix sums or Fenwick trees

---

## Overview

A segment tree is a data structure that processes range queries and point updates in O(log n) time. It is used for various queries such as range sum, minimum, maximum, and more.

---

## Table of Contents

1. [Segment Tree Concept](#1-segment-tree-concept)
2. [Basic Implementation](#2-basic-implementation)
3. [Range Sum Query](#3-range-sum-query)
4. [Range Minimum/Maximum](#4-range-minimummaximum)
5. [Lazy Propagation](#5-lazy-propagation)
6. [Application Problems](#6-application-problems)
7. [Practice Problems](#7-practice-problems)

---

## 1. Segment Tree Concept

A segment tree is like a tournament bracket — each internal node summarizes the "competition result" (sum, min, or max) of the elements in its subtree. Just as a bracket tells you the winner of any sub-tournament in O(log n) steps, a segment tree answers any range query in O(log n) by combining the precomputed results of at most O(log n) nodes.

### 1.1 Basic Idea

```
Range sum segment tree for array [2, 4, 1, 3, 5, 2, 7, 6]

                    [30]              (0-7: total sum)
                  /      \
             [10]          [20]       (0-3, 4-7)
            /    \        /    \
         [6]     [4]   [7]     [13]   (0-1, 2-3, 4-5, 6-7)
        /  \    /  \   /  \    /  \
       [2] [4] [1] [3][5] [2] [7] [6] (each element)

Characteristics:
- Leaf nodes: each element of the original array
- Internal nodes: sum (or min/max) of child nodes
- Height: O(log n)
- Number of nodes: 2n - 1 (allocate up to 4n for safety)
```

### 1.2 Time Complexity

```
┌─────────────────┬─────────────┬──────────────┐
│ Operation       │ Time        │ Description  │
├─────────────────┼─────────────┼──────────────┤
│ Build tree      │ O(n)        │ One-time preprocessing │
│ Point update    │ O(log n)    │ Single value change │
│ Range query     │ O(log n)    │ Sum/min/max  │
│ Range update    │ O(log n)    │ Requires Lazy │
└─────────────────┴─────────────┴──────────────┘
```

### 1.3 Index Rules

```
1-indexed tree (recommended):
- Root: tree[1]
- Left child: tree[2*i]
- Right child: tree[2*i + 1]
- Parent: tree[i // 2]

0-indexed tree:
- Root: tree[0]
- Left child: tree[2*i + 1]
- Right child: tree[2*i + 2]
- Parent: tree[(i - 1) // 2]
```

---

## 2. Basic Implementation

### 2.1 Range Sum Segment Tree (Recursive)

```python
class SegmentTree:
    def __init__(self, arr):
        self.n = len(arr)
        # Allocate 4n nodes: a complete binary tree over n leaves has at most 4n nodes
        # when n is not a power of 2 — using 4n avoids out-of-bounds access without
        # computing the exact size needed
        self.tree = [0] * (4 * self.n)
        self._build(arr, 1, 0, self.n - 1)

    def _build(self, arr, node, start, end):
        """Build tree - O(n): each element is visited exactly once as a leaf"""
        if start == end:
            self.tree[node] = arr[start]  # Leaf node stores the element directly
        else:
            mid = (start + end) // 2
            # Build left and right subtrees first, then combine — post-order construction
            self._build(arr, 2 * node, start, mid)
            self._build(arr, 2 * node + 1, mid + 1, end)
            # Each internal node stores the aggregate of its children, so any range
            # query can be answered by combining O(log n) pre-computed node values
            self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def update(self, idx, val):
        """Point update - O(log n)"""
        self._update(1, 0, self.n - 1, idx, val)

    def _update(self, node, start, end, idx, val):
        if start == end:
            self.tree[node] = val  # Update the leaf — this is the single source of truth
        else:
            mid = (start + end) // 2
            if idx <= mid:
                self._update(2 * node, start, mid, idx, val)
            else:
                self._update(2 * node + 1, mid + 1, end, idx, val)
            # Re-aggregate on the way back up — only O(log n) ancestors need updating,
            # not the entire array like a naive approach would require
            self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def query(self, left, right):
        """Range sum query - O(log n)"""
        return self._query(1, 0, self.n - 1, left, right)

    def _query(self, node, start, end, left, right):
        # Node's interval is completely outside the query range — contribute nothing
        if right < start or end < left:
            return 0

        # Node's interval is completely inside the query range — use the precomputed value
        # directly without descending further; this is what gives O(log n) query time
        if left <= start and end <= right:
            return self.tree[node]

        # Partial overlap: split and combine both halves
        mid = (start + end) // 2
        left_sum = self._query(2 * node, start, mid, left, right)
        right_sum = self._query(2 * node + 1, mid + 1, end, left, right)
        return left_sum + right_sum


# Usage example
arr = [2, 4, 1, 3, 5, 2, 7, 6]
st = SegmentTree(arr)

print(st.query(0, 7))  # 30 (total sum)
print(st.query(2, 5))  # 11 (1+3+5+2)

st.update(3, 10)  # Change arr[3] to 10
print(st.query(2, 5))  # 18 (1+10+5+2)
```

### 2.2 Iterative Implementation (Bottom-up)

```python
class SegmentTreeIterative:
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0] * (2 * self.n)

        # Leaves occupy indices [n, 2n-1]; this avoids storing a root at index 0
        # and makes parent/child arithmetic clean: parent(i) = i//2, children = 2i, 2i+1
        for i in range(self.n):
            self.tree[self.n + i] = arr[i]

        # Build internal nodes bottom-up: each node is the sum of its two children.
        # Processing right-to-left ensures children are always filled before their parent.
        for i in range(self.n - 1, 0, -1):
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]

    def update(self, idx, val):
        """Point update — walk from leaf to root, re-aggregating each ancestor"""
        idx += self.n  # Translate array index to tree index
        self.tree[idx] = val

        while idx > 1:
            idx //= 2  # Move to parent
            # Parent always equals the sum of its two children — recompute directly
            self.tree[idx] = self.tree[2 * idx] + self.tree[2 * idx + 1]

    def query(self, left, right):
        """Range sum [left, right] — process boundaries inward until they meet"""
        left += self.n
        right += self.n
        result = 0

        while left <= right:
            # If left is a right child, it is not fully covered by its parent —
            # include it directly and advance past it
            if left % 2 == 1:
                result += self.tree[left]
                left += 1
            # Symmetrically, if right is a left child, include it and retreat
            if right % 2 == 0:
                result += self.tree[right]
                right -= 1
            left //= 2
            right //= 2

        return result
```

### 2.3 C++ Implementation

```cpp
#include <vector>
using namespace std;

class SegmentTree {
private:
    vector<long long> tree;
    int n;

    void build(const vector<int>& arr, int node, int start, int end) {
        if (start == end) {
            tree[node] = arr[start];
        } else {
            int mid = (start + end) / 2;
            build(arr, 2 * node, start, mid);
            build(arr, 2 * node + 1, mid + 1, end);
            tree[node] = tree[2 * node] + tree[2 * node + 1];
        }
    }

    void update(int node, int start, int end, int idx, long long val) {
        if (start == end) {
            tree[node] = val;
        } else {
            int mid = (start + end) / 2;
            if (idx <= mid) {
                update(2 * node, start, mid, idx, val);
            } else {
                update(2 * node + 1, mid + 1, end, idx, val);
            }
            tree[node] = tree[2 * node] + tree[2 * node + 1];
        }
    }

    long long query(int node, int start, int end, int left, int right) {
        if (right < start || end < left) return 0;
        if (left <= start && end <= right) return tree[node];

        int mid = (start + end) / 2;
        return query(2 * node, start, mid, left, right) +
               query(2 * node + 1, mid + 1, end, left, right);
    }

public:
    SegmentTree(const vector<int>& arr) {
        n = arr.size();
        tree.resize(4 * n);
        build(arr, 1, 0, n - 1);
    }

    void update(int idx, long long val) {
        update(1, 0, n - 1, idx, val);
    }

    long long query(int left, int right) {
        return query(1, 0, n - 1, left, right);
    }
};
```

---

## 3. Range Sum Query

### 3.1 Query Process Visualization

```
Array: [2, 4, 1, 3, 5, 2, 7, 6]
Query: query(2, 5) = 1 + 3 + 5 + 2 = 11

                    [30]
                  /      \
             [10]          [20]
            /    \        /    \
         [6]     [4]   [7]     [13]
        /  \    /  \   /  \    /  \
       [2] [4] [1] [3][5] [2] [7] [6]
            ↑    ↑   ↑   ↑
           Range: 2 ~ 5

Query decomposition:
[2-3]: Completely included → tree = 4 (1+3)
[4-5]: Completely included → tree = 7 (5+2)
Result: 4 + 7 = 11
```

### 3.2 Difference Update

```python
class SegmentTreeDiff:
    """Update by difference rather than value"""

    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self._build(arr, 1, 0, self.n - 1)

    def _build(self, arr, node, start, end):
        if start == end:
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            self._build(arr, 2 * node, start, mid)
            self._build(arr, 2 * node + 1, mid + 1, end)
            self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def add(self, idx, diff):
        """arr[idx] += diff"""
        self._add(1, 0, self.n - 1, idx, diff)

    def _add(self, node, start, end, idx, diff):
        if start == end:
            self.tree[node] += diff
        else:
            mid = (start + end) // 2
            if idx <= mid:
                self._add(2 * node, start, mid, idx, diff)
            else:
                self._add(2 * node + 1, mid + 1, end, idx, diff)
            self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def query(self, left, right):
        return self._query(1, 0, self.n - 1, left, right)

    def _query(self, node, start, end, left, right):
        if right < start or end < left:
            return 0
        if left <= start and end <= right:
            return self.tree[node]
        mid = (start + end) // 2
        return (self._query(2 * node, start, mid, left, right) +
                self._query(2 * node + 1, mid + 1, end, left, right))
```

---

## 4. Range Minimum/Maximum

### 4.1 Minimum Segment Tree

```python
class MinSegmentTree:
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [float('inf')] * (4 * self.n)
        self._build(arr, 1, 0, self.n - 1)

    def _build(self, arr, node, start, end):
        if start == end:
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            self._build(arr, 2 * node, start, mid)
            self._build(arr, 2 * node + 1, mid + 1, end)
            self.tree[node] = min(self.tree[2 * node], self.tree[2 * node + 1])

    def update(self, idx, val):
        self._update(1, 0, self.n - 1, idx, val)

    def _update(self, node, start, end, idx, val):
        if start == end:
            self.tree[node] = val
        else:
            mid = (start + end) // 2
            if idx <= mid:
                self._update(2 * node, start, mid, idx, val)
            else:
                self._update(2 * node + 1, mid + 1, end, idx, val)
            self.tree[node] = min(self.tree[2 * node], self.tree[2 * node + 1])

    def query(self, left, right):
        """Minimum value in range [left, right]"""
        return self._query(1, 0, self.n - 1, left, right)

    def _query(self, node, start, end, left, right):
        if right < start or end < left:
            return float('inf')
        if left <= start and end <= right:
            return self.tree[node]
        mid = (start + end) // 2
        return min(self._query(2 * node, start, mid, left, right),
                   self._query(2 * node + 1, mid + 1, end, left, right))


# Example
arr = [5, 2, 8, 1, 9, 3, 7, 4]
st = MinSegmentTree(arr)
print(st.query(0, 7))  # 1 (overall minimum)
print(st.query(2, 5))  # 1 (minimum among 8, 1, 9, 3)
print(st.query(4, 7))  # 3 (minimum among 9, 3, 7, 4)
```

### 4.2 Minimum + Index

```python
class MinIndexSegmentTree:
    """Returns minimum value and its index"""

    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [(float('inf'), -1)] * (4 * self.n)
        self._build(arr, 1, 0, self.n - 1)

    def _build(self, arr, node, start, end):
        if start == end:
            self.tree[node] = (arr[start], start)
        else:
            mid = (start + end) // 2
            self._build(arr, 2 * node, start, mid)
            self._build(arr, 2 * node + 1, mid + 1, end)
            self.tree[node] = min(self.tree[2 * node], self.tree[2 * node + 1])

    def query(self, left, right):
        """Returns (minimum value, index)"""
        return self._query(1, 0, self.n - 1, left, right)

    def _query(self, node, start, end, left, right):
        if right < start or end < left:
            return (float('inf'), -1)
        if left <= start and end <= right:
            return self.tree[node]
        mid = (start + end) // 2
        return min(self._query(2 * node, start, mid, left, right),
                   self._query(2 * node + 1, mid + 1, end, left, right))
```

---

## 5. Lazy Propagation

### 5.1 Necessity

```
Problem: Add v to all elements in range [l, r]

Regular segment tree: O(n) per update (visits all elements)
Lazy Propagation: O(log n) per update

Idea:
- Don't apply updates immediately, process "later"
- Store pending updates in lazy[node]
- Propagate to children only when needed
```

### 5.2 Range Addition + Range Sum

```python
class LazySegmentTree:
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        # lazy[node] stores the "pending addition" not yet pushed to children.
        # This defers work: instead of updating all n leaves in a range immediately,
        # we record the intent at a high-level node and push down only when needed.
        self.lazy = [0] * (4 * self.n)
        self._build(arr, 1, 0, self.n - 1)

    def _build(self, arr, node, start, end):
        if start == end:
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            self._build(arr, 2 * node, start, mid)
            self._build(arr, 2 * node + 1, mid + 1, end)
            self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def _propagate(self, node, start, end):
        """Push pending lazy update to this node and schedule it for children"""
        if self.lazy[node] != 0:
            # Applying lazy[node] to a range of (end - start + 1) elements means
            # the range sum increases by lazy[node] * count — no need to visit each leaf
            self.tree[node] += self.lazy[node] * (end - start + 1)

            # Pass the deferred update down to children — they will apply it
            # only when they are actually visited, preserving O(log n) per operation
            if start != end:
                self.lazy[2 * node] += self.lazy[node]
                self.lazy[2 * node + 1] += self.lazy[node]

            self.lazy[node] = 0  # This node's pending update is now applied

    def update_range(self, left, right, val):
        """Add val to range [left, right]"""
        self._update_range(1, 0, self.n - 1, left, right, val)

    def _update_range(self, node, start, end, left, right, val):
        # Always propagate before reading or modifying this node, so tree[node]
        # reflects the true value and children receive accurate lazy deltas
        self._propagate(node, start, end)

        if right < start or end < left:
            return  # Completely outside the update range — nothing to do

        if left <= start and end <= right:
            # Completely inside — store the update in lazy instead of descending further.
            # _propagate immediately applies it to tree[node] so the node is up-to-date,
            # while children remain untouched until they are queried or updated.
            self.lazy[node] += val
            self._propagate(node, start, end)
            return

        # Partial overlap: recurse into both children and re-aggregate afterward
        mid = (start + end) // 2
        self._update_range(2 * node, start, mid, left, right, val)
        self._update_range(2 * node + 1, mid + 1, end, left, right, val)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def query(self, left, right):
        """Range sum query"""
        return self._query(1, 0, self.n - 1, left, right)

    def _query(self, node, start, end, left, right):
        # Flush any deferred lazy value before reading this node's sum
        self._propagate(node, start, end)

        if right < start or end < left:
            return 0

        if left <= start and end <= right:
            return self.tree[node]  # Already up-to-date after propagation above

        mid = (start + end) // 2
        return (self._query(2 * node, start, mid, left, right) +
                self._query(2 * node + 1, mid + 1, end, left, right))


# Usage example
arr = [1, 2, 3, 4, 5]
st = LazySegmentTree(arr)

print(st.query(0, 4))  # 15

st.update_range(1, 3, 10)  # [1, 12, 13, 14, 5]
print(st.query(0, 4))  # 45
print(st.query(1, 3))  # 39
```

### 5.3 Range Addition + Range Minimum (Lazy)

```python
class LazyMinSegmentTree:
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [float('inf')] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        self._build(arr, 1, 0, self.n - 1)

    def _build(self, arr, node, start, end):
        if start == end:
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            self._build(arr, 2 * node, start, mid)
            self._build(arr, 2 * node + 1, mid + 1, end)
            self.tree[node] = min(self.tree[2 * node], self.tree[2 * node + 1])

    def _propagate(self, node):
        if self.lazy[node] != 0:
            self.tree[node] += self.lazy[node]
            if 2 * node < len(self.lazy):
                self.lazy[2 * node] += self.lazy[node]
                self.lazy[2 * node + 1] += self.lazy[node]
            self.lazy[node] = 0

    def update_range(self, left, right, val):
        self._update_range(1, 0, self.n - 1, left, right, val)

    def _update_range(self, node, start, end, left, right, val):
        self._propagate(node)

        if right < start or end < left:
            return

        if left <= start and end <= right:
            self.lazy[node] += val
            self._propagate(node)
            return

        mid = (start + end) // 2
        self._update_range(2 * node, start, mid, left, right, val)
        self._update_range(2 * node + 1, mid + 1, end, left, right, val)
        self.tree[node] = min(self.tree[2 * node], self.tree[2 * node + 1])

    def query(self, left, right):
        return self._query(1, 0, self.n - 1, left, right)

    def _query(self, node, start, end, left, right):
        self._propagate(node)

        if right < start or end < left:
            return float('inf')

        if left <= start and end <= right:
            return self.tree[node]

        mid = (start + end) // 2
        return min(self._query(2 * node, start, mid, left, right),
                   self._query(2 * node + 1, mid + 1, end, left, right))
```

---

## 6. Application Problems

### 6.1 Inversion Count

```python
def count_inversions(arr):
    """
    Inversion pairs: number of pairs (i, j) where i < j and arr[i] > arr[j]
    Using segment tree
    """
    # Coordinate compression
    sorted_arr = sorted(set(arr))
    rank = {v: i for i, v in enumerate(sorted_arr)}
    n = len(sorted_arr)

    st = SegmentTree([0] * n)
    count = 0

    for val in arr:
        r = rank[val]
        # Count of values greater than r (already processed)
        count += st.query(r + 1, n - 1)
        # Add current value
        st.add(r, 1)

    return count
```

### 6.2 Finding K-th Element

```python
def find_kth(st, k):
    """
    Find index of k-th smallest element in segment tree
    st[i] = 1 if element i exists, 0 otherwise
    """
    node = 1
    start, end = 0, st.n - 1

    while start != end:
        mid = (start + end) // 2
        left_count = st.tree[2 * node]

        if k <= left_count:
            node = 2 * node
            end = mid
        else:
            k -= left_count
            node = 2 * node + 1
            start = mid + 1

    return start
```

### 6.3 2D Segment Tree

```python
class SegmentTree2D:
    """2D range sum segment tree"""

    def __init__(self, matrix):
        self.n = len(matrix)
        self.m = len(matrix[0]) if self.n > 0 else 0
        self.tree = [[0] * (4 * self.m) for _ in range(4 * self.n)]
        if self.n > 0 and self.m > 0:
            self._build_x(matrix, 1, 0, self.n - 1)

    def _build_x(self, matrix, node_x, start_x, end_x):
        if start_x == end_x:
            self._build_y(matrix, node_x, start_x, end_x, 1, 0, self.m - 1, True)
        else:
            mid_x = (start_x + end_x) // 2
            self._build_x(matrix, 2 * node_x, start_x, mid_x)
            self._build_x(matrix, 2 * node_x + 1, mid_x + 1, end_x)
            self._build_y(matrix, node_x, start_x, end_x, 1, 0, self.m - 1, False)

    def _build_y(self, matrix, node_x, start_x, end_x, node_y, start_y, end_y, leaf_x):
        if start_y == end_y:
            if leaf_x:
                self.tree[node_x][node_y] = matrix[start_x][start_y]
            else:
                self.tree[node_x][node_y] = (self.tree[2 * node_x][node_y] +
                                              self.tree[2 * node_x + 1][node_y])
        else:
            mid_y = (start_y + end_y) // 2
            self._build_y(matrix, node_x, start_x, end_x, 2 * node_y, start_y, mid_y, leaf_x)
            self._build_y(matrix, node_x, start_x, end_x, 2 * node_y + 1, mid_y + 1, end_y, leaf_x)
            self.tree[node_x][node_y] = (self.tree[node_x][2 * node_y] +
                                          self.tree[node_x][2 * node_y + 1])

    def query(self, x1, y1, x2, y2):
        """Rectangle range sum [(x1,y1), (x2,y2)]"""
        return self._query_x(1, 0, self.n - 1, x1, x2, y1, y2)

    def _query_x(self, node_x, start_x, end_x, x1, x2, y1, y2):
        if x2 < start_x or end_x < x1:
            return 0
        if x1 <= start_x and end_x <= x2:
            return self._query_y(node_x, 1, 0, self.m - 1, y1, y2)

        mid_x = (start_x + end_x) // 2
        return (self._query_x(2 * node_x, start_x, mid_x, x1, x2, y1, y2) +
                self._query_x(2 * node_x + 1, mid_x + 1, end_x, x1, x2, y1, y2))

    def _query_y(self, node_x, node_y, start_y, end_y, y1, y2):
        if y2 < start_y or end_y < y1:
            return 0
        if y1 <= start_y and end_y <= y2:
            return self.tree[node_x][node_y]

        mid_y = (start_y + end_y) // 2
        return (self._query_y(node_x, 2 * node_y, start_y, mid_y, y1, y2) +
                self._query_y(node_x, 2 * node_y + 1, mid_y + 1, end_y, y1, y2))
```

---

## 7. Practice Problems

### Recommended Problems

| Difficulty | Problem | Platform | Type |
|--------|------|--------|------|
| ⭐⭐⭐ | [Range Sum Query](https://www.acmicpc.net/problem/2042) | BOJ | Basic |
| ⭐⭐⭐ | [Minimum Value](https://www.acmicpc.net/problem/10868) | BOJ | Min Query |
| ⭐⭐⭐ | [Range Sum Query - Mutable](https://leetcode.com/problems/range-sum-query-mutable/) | LeetCode | Basic |
| ⭐⭐⭐⭐ | [Range Sum Query 2](https://www.acmicpc.net/problem/10999) | BOJ | Lazy |
| ⭐⭐⭐⭐ | [Sequence and Query 17](https://www.acmicpc.net/problem/14438) | BOJ | Min Query |
| ⭐⭐⭐⭐⭐ | [Count of Smaller Numbers After Self](https://leetcode.com/problems/count-of-smaller-numbers-after-self/) | LeetCode | Application |

---

## Next Steps

- [24_Fenwick_Tree.md](./24_Fenwick_Tree.md) - Fenwick Tree

---

## References

- [Segment Tree](https://cp-algorithms.com/data_structures/segment_tree.html)
- [Lazy Propagation](https://cp-algorithms.com/data_structures/segment_tree.html#lazy-propagation)
