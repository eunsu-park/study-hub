"""
Segment Tree
Segment Tree for Range Queries

A data structure for efficiently processing range queries and point updates.
"""

from typing import List, Callable, Optional


# =============================================================================
# 1. Basic Segment Tree (Range Sum)
# =============================================================================

class SegmentTree:
    """
    Segment Tree (Range Sum)
    - Point update: O(log n)
    - Range query: O(log n)
    - Space: O(n)
    """

    def __init__(self, arr: List[int]):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        if self.n > 0:
            self._build(arr, 1, 0, self.n - 1)

    def _build(self, arr: List[int], node: int, start: int, end: int):
        """Build tree - O(n)"""
        if start == end:
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            self._build(arr, 2 * node, start, mid)
            self._build(arr, 2 * node + 1, mid + 1, end)
            self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def update(self, idx: int, val: int):
        """Point update - O(log n)"""
        self._update(1, 0, self.n - 1, idx, val)

    def _update(self, node: int, start: int, end: int, idx: int, val: int):
        if start == end:
            self.tree[node] = val
        else:
            mid = (start + end) // 2
            if idx <= mid:
                self._update(2 * node, start, mid, idx, val)
            else:
                self._update(2 * node + 1, mid + 1, end, idx, val)
            self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def query(self, left: int, right: int) -> int:
        """Range sum query - O(log n)"""
        return self._query(1, 0, self.n - 1, left, right)

    def _query(self, node: int, start: int, end: int, left: int, right: int) -> int:
        if right < start or end < left:
            return 0  # Out of range
        if left <= start and end <= right:
            return self.tree[node]  # Fully contained

        mid = (start + end) // 2
        left_sum = self._query(2 * node, start, mid, left, right)
        right_sum = self._query(2 * node + 1, mid + 1, end, left, right)
        return left_sum + right_sum


# =============================================================================
# 2. Generic Segment Tree (Arbitrary Operation)
# =============================================================================

class GenericSegmentTree:
    """
    Generic Segment Tree (arbitrary associative operation)
    - Works with any operation satisfying the associative property
    """

    def __init__(self, arr: List[int], func: Callable[[int, int], int], identity: int):
        """
        func: associative operation (e.g., min, max, gcd, +, *)
        identity: identity element (e.g., inf for min, 0 for +, 1 for *)
        """
        self.n = len(arr)
        self.func = func
        self.identity = identity
        self.tree = [identity] * (4 * self.n)
        if self.n > 0:
            self._build(arr, 1, 0, self.n - 1)

    def _build(self, arr: List[int], node: int, start: int, end: int):
        if start == end:
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            self._build(arr, 2 * node, start, mid)
            self._build(arr, 2 * node + 1, mid + 1, end)
            self.tree[node] = self.func(self.tree[2 * node], self.tree[2 * node + 1])

    def update(self, idx: int, val: int):
        self._update(1, 0, self.n - 1, idx, val)

    def _update(self, node: int, start: int, end: int, idx: int, val: int):
        if start == end:
            self.tree[node] = val
        else:
            mid = (start + end) // 2
            if idx <= mid:
                self._update(2 * node, start, mid, idx, val)
            else:
                self._update(2 * node + 1, mid + 1, end, idx, val)
            self.tree[node] = self.func(self.tree[2 * node], self.tree[2 * node + 1])

    def query(self, left: int, right: int) -> int:
        return self._query(1, 0, self.n - 1, left, right)

    def _query(self, node: int, start: int, end: int, left: int, right: int) -> int:
        if right < start or end < left:
            return self.identity
        if left <= start and end <= right:
            return self.tree[node]

        mid = (start + end) // 2
        left_val = self._query(2 * node, start, mid, left, right)
        right_val = self._query(2 * node + 1, mid + 1, end, left, right)
        return self.func(left_val, right_val)


# =============================================================================
# 3. Lazy Propagation (Range Update)
# =============================================================================

class LazySegmentTree:
    """
    Segment Tree with Lazy Propagation
    - Range update: O(log n)
    - Range query: O(log n)
    """

    def __init__(self, arr: List[int]):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        if self.n > 0:
            self._build(arr, 1, 0, self.n - 1)

    def _build(self, arr: List[int], node: int, start: int, end: int):
        if start == end:
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            self._build(arr, 2 * node, start, mid)
            self._build(arr, 2 * node + 1, mid + 1, end)
            self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def _push_down(self, node: int, start: int, end: int):
        """Propagate lazy value"""
        if self.lazy[node] != 0:
            mid = (start + end) // 2

            # Left child
            self.tree[2 * node] += self.lazy[node] * (mid - start + 1)
            self.lazy[2 * node] += self.lazy[node]

            # Right child
            self.tree[2 * node + 1] += self.lazy[node] * (end - mid)
            self.lazy[2 * node + 1] += self.lazy[node]

            self.lazy[node] = 0

    def update_range(self, left: int, right: int, val: int):
        """Add val to range [left, right]"""
        self._update_range(1, 0, self.n - 1, left, right, val)

    def _update_range(self, node: int, start: int, end: int, left: int, right: int, val: int):
        if right < start or end < left:
            return

        if left <= start and end <= right:
            self.tree[node] += val * (end - start + 1)
            self.lazy[node] += val
            return

        self._push_down(node, start, end)

        mid = (start + end) // 2
        self._update_range(2 * node, start, mid, left, right, val)
        self._update_range(2 * node + 1, mid + 1, end, left, right, val)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def query(self, left: int, right: int) -> int:
        """Range sum query"""
        return self._query(1, 0, self.n - 1, left, right)

    def _query(self, node: int, start: int, end: int, left: int, right: int) -> int:
        if right < start or end < left:
            return 0

        if left <= start and end <= right:
            return self.tree[node]

        self._push_down(node, start, end)

        mid = (start + end) // 2
        left_sum = self._query(2 * node, start, mid, left, right)
        right_sum = self._query(2 * node + 1, mid + 1, end, left, right)
        return left_sum + right_sum


# =============================================================================
# 4. Iterative Segment Tree
# =============================================================================

class IterativeSegmentTree:
    """
    Iterative Segment Tree (non-recursive)
    Memory efficient, cache friendly
    """

    def __init__(self, arr: List[int]):
        self.n = len(arr)
        self.tree = [0] * (2 * self.n)

        # Fill leaf nodes
        for i in range(self.n):
            self.tree[self.n + i] = arr[i]

        # Build internal nodes
        for i in range(self.n - 1, 0, -1):
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]

    def update(self, idx: int, val: int):
        """Point update"""
        idx += self.n
        self.tree[idx] = val

        while idx > 1:
            idx //= 2
            self.tree[idx] = self.tree[2 * idx] + self.tree[2 * idx + 1]

    def query(self, left: int, right: int) -> int:
        """Range [left, right] sum"""
        left += self.n
        right += self.n + 1
        result = 0

        while left < right:
            if left % 2 == 1:
                result += self.tree[left]
                left += 1
            if right % 2 == 1:
                right -= 1
                result += self.tree[right]
            left //= 2
            right //= 2

        return result


# =============================================================================
# 5. 2D Segment Tree
# =============================================================================

class SegmentTree2D:
    """
    2D Segment Tree (Range Sum)
    - Query/Update: O(log n * log m)
    """

    def __init__(self, matrix: List[List[int]]):
        if not matrix or not matrix[0]:
            self.n = self.m = 0
            return

        self.n = len(matrix)
        self.m = len(matrix[0])
        self.tree = [[0] * (4 * self.m) for _ in range(4 * self.n)]
        self._build_x(matrix, 1, 0, self.n - 1)

    def _build_x(self, matrix: List[List[int]], node_x: int, lx: int, rx: int):
        if lx == rx:
            self._build_y(matrix, node_x, lx, rx, 1, 0, self.m - 1, lx)
        else:
            mid = (lx + rx) // 2
            self._build_x(matrix, 2 * node_x, lx, mid)
            self._build_x(matrix, 2 * node_x + 1, mid + 1, rx)
            self._merge_y(node_x, 1, 0, self.m - 1)

    def _build_y(self, matrix, node_x, lx, rx, node_y, ly, ry, row):
        if ly == ry:
            self.tree[node_x][node_y] = matrix[row][ly]
        else:
            mid = (ly + ry) // 2
            self._build_y(matrix, node_x, lx, rx, 2 * node_y, ly, mid, row)
            self._build_y(matrix, node_x, lx, rx, 2 * node_y + 1, mid + 1, ry, row)
            self.tree[node_x][node_y] = self.tree[node_x][2 * node_y] + self.tree[node_x][2 * node_y + 1]

    def _merge_y(self, node_x, node_y, ly, ry):
        if ly == ry:
            self.tree[node_x][node_y] = self.tree[2 * node_x][node_y] + self.tree[2 * node_x + 1][node_y]
        else:
            mid = (ly + ry) // 2
            self._merge_y(node_x, 2 * node_y, ly, mid)
            self._merge_y(node_x, 2 * node_y + 1, mid + 1, ry)
            self.tree[node_x][node_y] = self.tree[node_x][2 * node_y] + self.tree[node_x][2 * node_y + 1]

    def query(self, x1: int, y1: int, x2: int, y2: int) -> int:
        """Rectangle range sum from (x1,y1) to (x2,y2)"""
        return self._query_x(1, 0, self.n - 1, x1, x2, y1, y2)

    def _query_x(self, node_x, lx, rx, x1, x2, y1, y2):
        if x2 < lx or rx < x1:
            return 0
        if x1 <= lx and rx <= x2:
            return self._query_y(node_x, 1, 0, self.m - 1, y1, y2)

        mid = (lx + rx) // 2
        left = self._query_x(2 * node_x, lx, mid, x1, x2, y1, y2)
        right = self._query_x(2 * node_x + 1, mid + 1, rx, x1, x2, y1, y2)
        return left + right

    def _query_y(self, node_x, node_y, ly, ry, y1, y2):
        if y2 < ly or ry < y1:
            return 0
        if y1 <= ly and ry <= y2:
            return self.tree[node_x][node_y]

        mid = (ly + ry) // 2
        left = self._query_y(node_x, 2 * node_y, ly, mid, y1, y2)
        right = self._query_y(node_x, 2 * node_y + 1, mid + 1, ry, y1, y2)
        return left + right


# =============================================================================
# 6. Application: Inversion Count
# =============================================================================

def count_inversions_segtree(arr: List[int]) -> int:
    """
    Inversion count using segment tree
    Time Complexity: O(n log n)
    """
    if not arr:
        return 0

    # Coordinate compression
    sorted_arr = sorted(set(arr))
    rank = {v: i for i, v in enumerate(sorted_arr)}
    n = len(sorted_arr)

    # Segment tree (stores frequencies)
    tree = [0] * (4 * n)

    def update(node, start, end, idx):
        if start == end:
            tree[node] += 1
        else:
            mid = (start + end) // 2
            if idx <= mid:
                update(2 * node, start, mid, idx)
            else:
                update(2 * node + 1, mid + 1, end, idx)
            tree[node] = tree[2 * node] + tree[2 * node + 1]

    def query(node, start, end, left, right):
        if right < start or end < left:
            return 0
        if left <= start and end <= right:
            return tree[node]
        mid = (start + end) // 2
        return query(2 * node, start, mid, left, right) + \
               query(2 * node + 1, mid + 1, end, left, right)

    inversions = 0
    for val in arr:
        r = rank[val]
        # Count values greater than r (among already inserted)
        inversions += query(1, 0, n - 1, r + 1, n - 1)
        update(1, 0, n - 1, r)

    return inversions


# =============================================================================
# Tests
# =============================================================================

def main():
    print("=" * 60)
    print("Segment Tree Examples")
    print("=" * 60)

    # 1. Basic Segment Tree
    print("\n[1] Basic Segment Tree (Range Sum)")
    arr = [1, 3, 5, 7, 9, 11]
    st = SegmentTree(arr)
    print(f"    Array: {arr}")
    print(f"    query(1, 3): {st.query(1, 3)}")  # 3+5+7=15
    st.update(2, 6)  # 5 -> 6
    print(f"    After update(2, 6) query(1, 3): {st.query(1, 3)}")  # 3+6+7=16

    # 2. Generic Segment Tree (Minimum)
    print("\n[2] Generic Segment Tree (Range Minimum)")
    arr = [5, 2, 8, 1, 9, 3]
    min_st = GenericSegmentTree(arr, min, float('inf'))
    print(f"    Array: {arr}")
    print(f"    min(1, 4): {min_st.query(1, 4)}")  # min(2,8,1,9)=1
    min_st.update(3, 10)  # 1 -> 10
    print(f"    After update(3, 10) min(1, 4): {min_st.query(1, 4)}")  # min(2,8,10,9)=2

    # 3. Lazy Propagation
    print("\n[3] Lazy Propagation (Range Update)")
    arr = [1, 2, 3, 4, 5]
    lazy_st = LazySegmentTree(arr)
    print(f"    Array: {arr}")
    print(f"    query(0, 4): {lazy_st.query(0, 4)}")  # 15
    lazy_st.update_range(1, 3, 10)  # Add 10 to range [1,3]
    print(f"    After update_range(1, 3, +10) query(0, 4): {lazy_st.query(0, 4)}")  # 45

    # 4. Iterative Segment Tree
    print("\n[4] Iterative Segment Tree")
    arr = [1, 3, 5, 7, 9, 11]
    iter_st = IterativeSegmentTree(arr)
    print(f"    Array: {arr}")
    print(f"    query(1, 4): {iter_st.query(1, 4)}")  # 3+5+7+9=24

    # 5. 2D Segment Tree
    print("\n[5] 2D Segment Tree")
    matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    st2d = SegmentTree2D(matrix)
    print(f"    Matrix: {matrix}")
    print(f"    query(0,0,1,1): {st2d.query(0, 0, 1, 1)}")  # 1+2+4+5=12
    print(f"    query(1,1,2,2): {st2d.query(1, 1, 2, 2)}")  # 5+6+8+9=28

    # 6. Inversion Count
    print("\n[6] Inversion Count")
    arr = [2, 4, 1, 3, 5]
    inv = count_inversions_segtree(arr)
    print(f"    Array: {arr}")
    print(f"    Inversion count: {inv}")  # (2,1), (4,1), (4,3) = 3

    # 7. Complexity Comparison
    print("\n[7] Complexity Comparison")
    print("    | Operation     | Array   | Segment Tree  | Lazy      |")
    print("    |---------------|---------|---------------|-----------|")
    print("    | Point update  | O(1)    | O(log n)      | O(log n)  |")
    print("    | Range update  | O(n)    | O(n)          | O(log n)  |")
    print("    | Range query   | O(n)    | O(log n)      | O(log n)  |")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
