# Fenwick Tree (Binary Indexed Tree)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the lowbit trick (`i & -i`) and describe how it determines the range each Fenwick tree node is responsible for
2. Implement prefix sum queries and point updates on a Fenwick tree in O(log n) time
3. Extend the Fenwick tree to support range update and range query operations using a difference array technique
4. Apply the Fenwick tree to solve inversion count and order-statistics problems efficiently
5. Implement a 2D Fenwick tree to handle 2D range sum queries with point updates
6. Compare Fenwick trees and segment trees by implementation complexity, memory usage, and capability

---

## Overview

Fenwick Tree (BIT: Binary Indexed Tree) is a data structure that processes range sum queries and point updates in O(log n) time. It is simpler to implement and more memory efficient than a segment tree.

---

## Table of Contents

1. [Fenwick Tree Concept](#1-fenwick-tree-concept)
2. [Basic Implementation](#2-basic-implementation)
3. [Operations](#3-operations)
4. [Applications](#4-applications)
5. [2D Fenwick Tree](#5-2d-fenwick-tree)
6. [Practice Problems](#6-practice-problems)

---

## 1. Fenwick Tree Concept

### 1.1 Structure

```
Fenwick Tree: 1-indexed array based

Key idea: The range that element i is responsible for
          is determined by the lowest bit (lowbit) of i

lowbit(i) = i & (-i)

Example (n=8):
Index:     1    2    3    4    5    6    7    8
lowbit:    1    2    1    4    1    2    1    8
Range:    [1,1][1,2][3,3][1,4][5,5][5,6][7,7][1,8]

tree[1] = arr[1]
tree[2] = arr[1] + arr[2]
tree[3] = arr[3]
tree[4] = arr[1] + arr[2] + arr[3] + arr[4]
tree[5] = arr[5]
tree[6] = arr[5] + arr[6]
tree[7] = arr[7]
tree[8] = arr[1] + ... + arr[8]
```

### 1.2 Lowbit Visualization

```
When representing index in binary:

1  = 0001 → lowbit = 1   → tree[1] = A[1]
2  = 0010 → lowbit = 2   → tree[2] = A[1..2]
3  = 0011 → lowbit = 1   → tree[3] = A[3]
4  = 0100 → lowbit = 4   → tree[4] = A[1..4]
5  = 0101 → lowbit = 1   → tree[5] = A[5]
6  = 0110 → lowbit = 2   → tree[6] = A[5..6]
7  = 0111 → lowbit = 1   → tree[7] = A[7]
8  = 1000 → lowbit = 8   → tree[8] = A[1..8]

Pattern:
- Odd indices: lowbit = 1 (only itself)
- Powers of 2: lowbit = index (from 1 to itself)
```

### 1.3 Time/Space Complexity

```
┌─────────────────┬─────────────┬─────────────┐
│ Operation       │ Time        │ Description │
├─────────────────┼─────────────┼─────────────┤
│ Build           │ O(n)        │ or O(nlogn) │
│ Point update    │ O(log n)    │ Value change│
│ Prefix sum      │ O(log n)    │ [1, i] sum  │
│ Range sum       │ O(log n)    │ [l, r] sum  │
└─────────────────┴─────────────┴─────────────┘

Space: O(n) - 1/2 to 1/4 of segment tree
```

---

## 2. Basic Implementation

### 2.1 Python Implementation

```python
class FenwickTree:
    def __init__(self, n):
        """Fenwick tree of size n (1-indexed)"""
        self.n = n
        self.tree = [0] * (n + 1)  # Index 0 is unused; tree is 1-indexed

    def update(self, i, delta):
        """arr[i] += delta — propagate the change to all ancestor nodes"""
        while i <= self.n:
            self.tree[i] += delta
            # i & (-i) strips all bits except the lowest set bit (the "lowbit").
            # Adding lowbit(i) jumps to the next node that is responsible for a
            # range that includes i — visiting exactly O(log n) ancestors.
            i += i & (-i)

    def prefix_sum(self, i):
        """arr[1] + arr[2] + ... + arr[i] — decompose prefix into O(log n) disjoint ranges"""
        total = 0
        while i > 0:
            total += self.tree[i]
            # Subtracting lowbit(i) moves to the previous node whose range is disjoint
            # from and to the left of tree[i]'s range — together they tile [1, i] exactly.
            i -= i & (-i)
        return total

    def range_sum(self, l, r):
        """arr[l] + arr[l+1] + ... + arr[r]
        Works by inclusion-exclusion: prefix[r] - prefix[l-1] cancels out [1, l-1]
        """
        return self.prefix_sum(r) - self.prefix_sum(l - 1)


# Usage example
n = 8
bit = FenwickTree(n)

# Build array [0, 1, 2, 3, 4, 5, 6, 7, 8] (1-indexed)
for i in range(1, n + 1):
    bit.update(i, i)

print(bit.prefix_sum(4))   # 10 (1+2+3+4)
print(bit.range_sum(2, 5)) # 14 (2+3+4+5)

bit.update(3, 5)  # arr[3] += 5
print(bit.range_sum(2, 5)) # 19 (2+8+4+5)
```

### 2.2 Initialize from Array

```python
class FenwickTreeFromArray:
    def __init__(self, arr):
        """Build Fenwick tree from array - O(n)
        This is faster than calling update() n times (which would be O(n log n)).
        Each node accumulates contributions from its children in a single forward pass.
        """
        self.n = len(arr)
        self.tree = [0] * (self.n + 1)

        # O(n) initialization: propagate each node's value to its immediate parent.
        # This bottom-up construction ensures every ancestor node receives the correct
        # partial sum without redundant computation.
        for i in range(1, self.n + 1):
            self.tree[i] += arr[i - 1]  # arr is 0-indexed
            j = i + (i & (-i))          # j is the direct parent of i in the BIT structure
            if j <= self.n:
                self.tree[j] += self.tree[i]  # Pass current node's total to its parent

    def update(self, i, delta):
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)

    def prefix_sum(self, i):
        total = 0
        while i > 0:
            total += self.tree[i]
            i -= i & (-i)
        return total

    def range_sum(self, l, r):
        return self.prefix_sum(r) - self.prefix_sum(l - 1)


# Usage example
arr = [1, 2, 3, 4, 5, 6, 7, 8]
bit = FenwickTreeFromArray(arr)
print(bit.range_sum(1, 4))  # 10
```

### 2.3 C++ Implementation

```cpp
#include <vector>
using namespace std;

class FenwickTree {
private:
    vector<long long> tree;
    int n;

public:
    FenwickTree(int n) : n(n), tree(n + 1, 0) {}

    FenwickTree(const vector<int>& arr) : n(arr.size()), tree(arr.size() + 1, 0) {
        for (int i = 1; i <= n; i++) {
            tree[i] += arr[i - 1];
            int j = i + (i & (-i));
            if (j <= n) tree[j] += tree[i];
        }
    }

    void update(int i, long long delta) {
        while (i <= n) {
            tree[i] += delta;
            i += i & (-i);
        }
    }

    long long prefixSum(int i) {
        long long total = 0;
        while (i > 0) {
            total += tree[i];
            i -= i & (-i);
        }
        return total;
    }

    long long rangeSum(int l, int r) {
        return prefixSum(r) - prefixSum(l - 1);
    }
};
```

---

## 3. Operations

### 3.1 Update Process Visualization

```
update(3, 5): arr[3] += 5

Index movement: 3 → 4 → 8 (→ 16 exceeds n)

3  = 0011 → tree[3] += 5
     0011 + 0001 = 0100
4  = 0100 → tree[4] += 5
     0100 + 0100 = 1000
8  = 1000 → tree[8] += 5
     1000 + 1000 = 10000 (> n, stop)

Affected nodes: tree[3], tree[4], tree[8]
```

### 3.2 Query Process Visualization

```
prefix_sum(7): arr[1] + ... + arr[7]

Index movement: 7 → 6 → 4 → 0

7  = 0111 → total += tree[7]   (arr[7])
     0111 - 0001 = 0110
6  = 0110 → total += tree[6]   (arr[5..6])
     0110 - 0010 = 0100
4  = 0100 → total += tree[4]   (arr[1..4])
     0100 - 0100 = 0000
0  = 0000 (stop)

Result: tree[7] + tree[6] + tree[4] = arr[1..7]
```

### 3.3 Point Query

```python
class FenwickTreePointQuery:
    """Range update + point query"""

    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 1)

    def range_update(self, l, r, delta):
        """arr[l..r] += delta"""
        self._update(l, delta)
        if r + 1 <= self.n:
            self._update(r + 1, -delta)

    def _update(self, i, delta):
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)

    def point_query(self, i):
        """arr[i] value"""
        total = 0
        while i > 0:
            total += self.tree[i]
            i -= i & (-i)
        return total


# Example
bit = FenwickTreePointQuery(8)
bit.range_update(2, 5, 3)  # arr[2..5] += 3
print(bit.point_query(3))  # 3
print(bit.point_query(6))  # 0
```

### 3.4 Finding K-th Element

```python
def find_kth(bit, k):
    """
    Find minimum index where prefix sum >= k
    (bit[i] = 1 if element exists)
    Time: O(log n) — binary lifting on the BIT structure
    This is faster than binary searching over prefix_sum() calls (which would be O(log²n))
    because we descend the BIT directly, one bit at a time.
    """
    n = bit.n
    pos = 0
    total = 0

    # Try to extend pos by adding each power of 2 from high to low.
    # If adding 2^i doesn't overshoot the k-th element, we commit to that step.
    # This greedy bit-by-bit descent reconstructs the answer in O(log n) steps.
    log_n = n.bit_length()
    for i in range(log_n - 1, -1, -1):
        next_pos = pos + (1 << i)
        if next_pos <= n and total + bit.tree[next_pos] < k:
            total += bit.tree[next_pos]
            pos = next_pos  # Safe to extend — the k-th element is further right

    return pos + 1  # pos is the last "safe" position; pos+1 is the k-th element


# Example: dynamic k-th element
class DynamicKth:
    def __init__(self, max_val):
        self.bit = FenwickTree(max_val)

    def add(self, x):
        """Add element x"""
        self.bit.update(x, 1)

    def remove(self, x):
        """Remove element x"""
        self.bit.update(x, -1)

    def kth(self, k):
        """K-th smallest element"""
        return find_kth(self.bit, k)

    def count_less(self, x):
        """Count of elements less than x"""
        return self.bit.prefix_sum(x - 1)


# Usage
dk = DynamicKth(100)
dk.add(5)
dk.add(10)
dk.add(3)
dk.add(7)
print(dk.kth(2))  # 5 (sorted: 3, 5, 7, 10)
print(dk.count_less(7))  # 2 (3, 5)
```

---

## 4. Applications

### 4.1 Inversion Count

```python
def count_inversions(arr):
    """
    Inversion pairs: count of (i, j) where i < j and arr[i] > arr[j]
    Time: O(n log n)
    """
    # Coordinate compression
    sorted_arr = sorted(set(arr))
    rank = {v: i + 1 for i, v in enumerate(sorted_arr)}
    max_rank = len(sorted_arr)

    bit = FenwickTree(max_rank)
    count = 0

    # Process from right to left
    for val in reversed(arr):
        r = rank[val]
        # Count of values smaller than r (already processed = was on the right)
        count += bit.prefix_sum(r - 1)
        bit.update(r, 1)

    return count


# Example
arr = [7, 5, 6, 4]
print(count_inversions(arr))  # 5: (7,5), (7,6), (7,4), (5,4), (6,4)
```

### 4.2 Range Update + Range Query

```python
class FenwickTreeRURQ:
    """
    Range Update, Range Query using two BITs.
    The key insight: prefix_sum(i) = bit1[i] * i - bit2[i].
    This algebraic identity lets us encode a range [l, r] addition of delta as
    point updates to two BITs, enabling O(log n) for both update and query.
    """
    def __init__(self, n):
        self.n = n
        self.bit1 = [0] * (n + 2)
        self.bit2 = [0] * (n + 2)

    def _update(self, bit, i, delta):
        while i <= self.n:
            bit[i] += delta
            i += i & (-i)

    def _query(self, bit, i):
        total = 0
        while i > 0:
            total += bit[i]
            i -= i & (-i)
        return total

    def range_update(self, l, r, delta):
        """arr[l..r] += delta — encoded as two difference-array tricks simultaneously"""
        # bit1 is a standard difference array BIT: adding delta at l and -delta at r+1
        # means query(bit1, i) = delta for i in [l, r] and 0 otherwise
        self._update(self.bit1, l, delta)
        self._update(self.bit1, r + 1, -delta)
        # bit2 adjustment ensures the prefix_sum formula prefix_sum(i) = bit1*i - bit2
        # holds correctly — the l-1 and r terms cancel out contributions outside [l, r]
        self._update(self.bit2, l, delta * (l - 1))
        self._update(self.bit2, r + 1, -delta * r)

    def prefix_sum(self, i):
        """arr[1] + ... + arr[i] using the two-BIT identity"""
        return self._query(self.bit1, i) * i - self._query(self.bit2, i)

    def range_sum(self, l, r):
        """arr[l] + ... + arr[r]"""
        return self.prefix_sum(r) - self.prefix_sum(l - 1)


# Example
bit = FenwickTreeRURQ(8)
bit.range_update(2, 5, 3)  # arr[2..5] += 3
print(bit.range_sum(1, 4))  # 9 (0+3+3+3)
print(bit.range_sum(3, 6))  # 12 (3+3+3+3)
```

### 4.3 Offline Query Processing

```python
def offline_range_sum(arr, queries):
    """
    Query: (l, r, type)
    type 1: arr[l] += r
    type 2: return sum of arr[l..r]
    """
    n = len(arr)
    bit = FenwickTreeFromArray(arr)
    results = []

    for query in queries:
        if query[0] == 1:
            _, idx, val = query
            bit.update(idx, val)
        else:
            _, l, r = query
            results.append(bit.range_sum(l, r))

    return results
```

---

## 5. 2D Fenwick Tree

### 5.1 Implementation

```python
class FenwickTree2D:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.tree = [[0] * (cols + 1) for _ in range(rows + 1)]

    def update(self, x, y, delta):
        """arr[x][y] += delta"""
        i = x
        while i <= self.rows:
            j = y
            while j <= self.cols:
                self.tree[i][j] += delta
                j += j & (-j)
            i += i & (-i)

    def prefix_sum(self, x, y):
        """Sum of arr[1..x][1..y]"""
        total = 0
        i = x
        while i > 0:
            j = y
            while j > 0:
                total += self.tree[i][j]
                j -= j & (-j)
            i -= i & (-i)
        return total

    def range_sum(self, x1, y1, x2, y2):
        """Sum of arr[x1..x2][y1..y2]"""
        return (self.prefix_sum(x2, y2)
                - self.prefix_sum(x1 - 1, y2)
                - self.prefix_sum(x2, y1 - 1)
                + self.prefix_sum(x1 - 1, y1 - 1))


# Example
bit2d = FenwickTree2D(4, 4)
matrix = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
]

# Initialize
for i in range(4):
    for j in range(4):
        bit2d.update(i + 1, j + 1, matrix[i][j])

print(bit2d.range_sum(2, 2, 3, 3))  # 34 (6+7+10+11)
```

### 5.2 C++ 2D Implementation

```cpp
class FenwickTree2D {
private:
    vector<vector<long long>> tree;
    int rows, cols;

public:
    FenwickTree2D(int r, int c) : rows(r), cols(c) {
        tree.assign(r + 1, vector<long long>(c + 1, 0));
    }

    void update(int x, int y, long long delta) {
        for (int i = x; i <= rows; i += i & (-i)) {
            for (int j = y; j <= cols; j += j & (-j)) {
                tree[i][j] += delta;
            }
        }
    }

    long long prefixSum(int x, int y) {
        long long total = 0;
        for (int i = x; i > 0; i -= i & (-i)) {
            for (int j = y; j > 0; j -= j & (-j)) {
                total += tree[i][j];
            }
        }
        return total;
    }

    long long rangeSum(int x1, int y1, int x2, int y2) {
        return prefixSum(x2, y2) - prefixSum(x1 - 1, y2)
               - prefixSum(x2, y1 - 1) + prefixSum(x1 - 1, y1 - 1);
    }
};
```

---

## 6. Practice Problems

### Recommended Problems

| Difficulty | Problem | Platform | Type |
|--------|------|--------|------|
| ⭐⭐⭐ | [Range Sum Query](https://www.acmicpc.net/problem/2042) | BOJ | Basic |
| ⭐⭐⭐ | [Range Sum Query - Mutable](https://leetcode.com/problems/range-sum-query-mutable/) | LeetCode | Basic |
| ⭐⭐⭐ | [Bubble Sort](https://www.acmicpc.net/problem/1517) | BOJ | Inversion |
| ⭐⭐⭐⭐ | [Count of Smaller Numbers](https://leetcode.com/problems/count-of-smaller-numbers-after-self/) | LeetCode | Inversion |
| ⭐⭐⭐⭐ | [Rectangle Sum](https://www.acmicpc.net/problem/11658) | BOJ | 2D BIT |

---

## Segment Tree vs Fenwick Tree

```
┌────────────────┬──────────────┬──────────────┐
│ Criterion      │ Segment Tree │ Fenwick Tree │
├────────────────┼──────────────┼──────────────┤
│ Space          │ O(4n)        │ O(n)         │
│ Implementation │ Medium       │ Simple ✓     │
│ Constant factor│ Large        │ Small ✓      │
│ Point query    │ ✓            │ ✓            │
│ Range query    │ ✓            │ ✓            │
│ Range update   │ Needs Lazy   │ Needs 2 BITs │
│ Versatility    │ High ✓       │ Low (sum only)│
│ Min/Max        │ ✓            │ ✗            │
└────────────────┴──────────────┴──────────────┘

Conclusion:
- Only need range sum → Fenwick Tree
- Min/Max/Complex operations → Segment Tree
```

---

## Next Steps

- [25_Network_Flow.md](./25_Network_Flow.md) - Network Flow

---

## References

- [Fenwick Tree](https://cp-algorithms.com/data_structures/fenwick.html)
- [Binary Indexed Trees - TopCoder](https://www.topcoder.com/thrive/articles/Binary%20Indexed%20Trees)
