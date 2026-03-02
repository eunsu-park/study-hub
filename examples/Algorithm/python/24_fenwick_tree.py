"""
Fenwick Tree (Binary Indexed Tree)
Fenwick Tree (BIT)

A data structure for efficiently processing range sums and point updates.
"""

from typing import List


# =============================================================================
# 1. Basic Fenwick Tree (Range Sum)
# =============================================================================

class FenwickTree:
    """
    Fenwick Tree (Binary Indexed Tree)
    - Point update: O(log n)
    - Prefix sum: O(log n)
    - Range sum: O(log n)
    - Space: O(n)
    """

    def __init__(self, n: int):
        """Create an empty Fenwick tree of size n"""
        self.n = n
        self.tree = [0] * (n + 1)  # 1-indexed

    @classmethod
    def from_array(cls, arr: List[int]) -> 'FenwickTree':
        """Build Fenwick tree from array - O(n)"""
        ft = cls(len(arr))

        # Efficient construction (O(n))
        for i, val in enumerate(arr):
            ft.tree[i + 1] += val
            parent = i + 1 + (ft._lowbit(i + 1))
            if parent <= ft.n:
                ft.tree[parent] += ft.tree[i + 1]

        return ft

    def _lowbit(self, x: int) -> int:
        """Lowest set bit (x & -x)"""
        return x & (-x)

    def update(self, idx: int, delta: int):
        """Add delta at position idx (0-indexed) - O(log n)"""
        idx += 1  # Convert to 1-indexed

        while idx <= self.n:
            self.tree[idx] += delta
            idx += self._lowbit(idx)

    def prefix_sum(self, idx: int) -> int:
        """Sum of [0, idx] (0-indexed) - O(log n)"""
        idx += 1
        result = 0

        while idx > 0:
            result += self.tree[idx]
            idx -= self._lowbit(idx)

        return result

    def range_sum(self, left: int, right: int) -> int:
        """Sum of [left, right] (0-indexed) - O(log n)"""
        if left == 0:
            return self.prefix_sum(right)
        return self.prefix_sum(right) - self.prefix_sum(left - 1)

    def get(self, idx: int) -> int:
        """Value at position idx (0-indexed)"""
        return self.range_sum(idx, idx)

    def set(self, idx: int, val: int):
        """Set value at position idx to val"""
        current = self.get(idx)
        self.update(idx, val - current)


# =============================================================================
# 2. Range Update + Point Query Fenwick Tree
# =============================================================================

class FenwickTreeRangeUpdate:
    """
    Range Update + Point Query
    Uses difference array technique
    """

    def __init__(self, n: int):
        self.n = n
        self.tree = [0] * (n + 1)

    def _lowbit(self, x: int) -> int:
        return x & (-x)

    def _update(self, idx: int, delta: int):
        idx += 1
        while idx <= self.n:
            self.tree[idx] += delta
            idx += self._lowbit(idx)

    def update_range(self, left: int, right: int, delta: int):
        """Add delta to range [left, right]"""
        self._update(left, delta)
        if right + 1 < self.n:
            self._update(right + 1, -delta)

    def query(self, idx: int) -> int:
        """Value at position idx"""
        idx += 1
        result = 0
        while idx > 0:
            result += self.tree[idx]
            idx -= self._lowbit(idx)
        return result


# =============================================================================
# 3. Range Update + Range Query Fenwick Tree
# =============================================================================

class FenwickTreeRangeUpdateRangeQuery:
    """
    Range Update + Range Query
    Uses two BITs
    """

    def __init__(self, n: int):
        self.n = n
        self.tree1 = [0] * (n + 1)  # B[i]
        self.tree2 = [0] * (n + 1)  # B[i] * i

    def _lowbit(self, x: int) -> int:
        return x & (-x)

    def _update(self, tree: List[int], idx: int, delta: int):
        while idx <= self.n:
            tree[idx] += delta
            idx += self._lowbit(idx)

    def _prefix_sum(self, tree: List[int], idx: int) -> int:
        result = 0
        while idx > 0:
            result += tree[idx]
            idx -= self._lowbit(idx)
        return result

    def update_range(self, left: int, right: int, delta: int):
        """Add delta to range [left, right] (1-indexed)"""
        self._update(self.tree1, left, delta)
        self._update(self.tree1, right + 1, -delta)
        self._update(self.tree2, left, delta * (left - 1))
        self._update(self.tree2, right + 1, -delta * right)

    def prefix_sum(self, idx: int) -> int:
        """Sum of [1, idx] (1-indexed)"""
        return self._prefix_sum(self.tree1, idx) * idx - self._prefix_sum(self.tree2, idx)

    def range_sum(self, left: int, right: int) -> int:
        """Sum of [left, right] (1-indexed)"""
        return self.prefix_sum(right) - self.prefix_sum(left - 1)


# =============================================================================
# 4. 2D Fenwick Tree
# =============================================================================

class FenwickTree2D:
    """
    2D Fenwick Tree
    - Point update: O(log n * log m)
    - Rectangle sum: O(log n * log m)
    """

    def __init__(self, n: int, m: int):
        self.n = n
        self.m = m
        self.tree = [[0] * (m + 1) for _ in range(n + 1)]

    def _lowbit(self, x: int) -> int:
        return x & (-x)

    def update(self, x: int, y: int, delta: int):
        """Add delta at (x, y) (0-indexed)"""
        x += 1
        while x <= self.n:
            y_idx = y + 1
            while y_idx <= self.m:
                self.tree[x][y_idx] += delta
                y_idx += self._lowbit(y_idx)
            x += self._lowbit(x)

    def prefix_sum(self, x: int, y: int) -> int:
        """Sum of (0,0) to (x,y) (0-indexed)"""
        x += 1
        result = 0
        while x > 0:
            y_idx = y + 1
            while y_idx > 0:
                result += self.tree[x][y_idx]
                y_idx -= self._lowbit(y_idx)
            x -= self._lowbit(x)
        return result

    def range_sum(self, x1: int, y1: int, x2: int, y2: int) -> int:
        """Rectangle sum from (x1,y1) to (x2,y2) (0-indexed)"""
        result = self.prefix_sum(x2, y2)
        if x1 > 0:
            result -= self.prefix_sum(x1 - 1, y2)
        if y1 > 0:
            result -= self.prefix_sum(x2, y1 - 1)
        if x1 > 0 and y1 > 0:
            result += self.prefix_sum(x1 - 1, y1 - 1)
        return result


# =============================================================================
# 5. Inversion Count
# =============================================================================

def count_inversions(arr: List[int]) -> int:
    """
    Inversion count using Fenwick tree
    Time Complexity: O(n log n)
    """
    if not arr:
        return 0

    # Coordinate compression
    sorted_arr = sorted(set(arr))
    rank = {v: i for i, v in enumerate(sorted_arr)}
    n = len(sorted_arr)

    ft = FenwickTree(n)
    inversions = 0

    for val in arr:
        r = rank[val]
        # Count of indices greater than r (among already inserted)
        inversions += ft.prefix_sum(n - 1) - ft.prefix_sum(r)
        ft.update(r, 1)

    return inversions


# =============================================================================
# 6. Finding the K-th Element
# =============================================================================

class FenwickTreeKth:
    """Fenwick tree supporting k-th element queries"""

    def __init__(self, n: int):
        self.n = n
        self.tree = [0] * (n + 1)

    def _lowbit(self, x: int) -> int:
        return x & (-x)

    def update(self, idx: int, delta: int):
        """Add delta at idx (1-indexed)"""
        while idx <= self.n:
            self.tree[idx] += delta
            idx += self._lowbit(idx)

    def prefix_sum(self, idx: int) -> int:
        """Sum of [1, idx]"""
        result = 0
        while idx > 0:
            result += self.tree[idx]
            idx -= self._lowbit(idx)
        return result

    def find_kth(self, k: int) -> int:
        """
        Find index of k-th element (1-indexed)
        Smallest idx such that prefix_sum(idx) >= k
        Time Complexity: O(log n)
        """
        idx = 0
        bit_mask = 1

        while bit_mask <= self.n:
            bit_mask <<= 1
        bit_mask >>= 1

        while bit_mask > 0:
            next_idx = idx + bit_mask
            if next_idx <= self.n and self.tree[next_idx] < k:
                idx = next_idx
                k -= self.tree[idx]
            bit_mask >>= 1

        return idx + 1


# =============================================================================
# 7. Application: Count Elements Smaller Than K in Range
# =============================================================================

def count_smaller_in_range(arr: List[int], queries: List[tuple]) -> List[int]:
    """
    Query: (left, right, k) - count of elements < k in arr[left:right+1]
    Offline queries + Fenwick tree
    Time Complexity: O((n + q) log n)
    """
    n = len(arr)
    q = len(queries)

    # Coordinate compression
    all_vals = sorted(set(arr) | set(k for _, _, k in queries))
    val_to_idx = {v: i + 1 for i, v in enumerate(all_vals)}
    m = len(all_vals)

    # Generate (value, index, type) events
    events = []
    for i, val in enumerate(arr):
        events.append((val_to_idx[val], i, 'arr', None))

    for qi, (left, right, k) in enumerate(queries):
        k_idx = val_to_idx.get(k, m + 1)
        events.append((k_idx, right, 'query_end', (qi, left, right)))

    events.sort()

    # Results
    results = [0] * q
    ft = FenwickTree(n)

    for val_idx, pos, event_type, data in events:
        if event_type == 'arr':
            ft.update(pos, 1)
        else:
            qi, left, right = data
            results[qi] = ft.range_sum(left, right)

    return results


# =============================================================================
# Tests
# =============================================================================

def main():
    print("=" * 60)
    print("Fenwick Tree (BIT) Examples")
    print("=" * 60)

    # 1. Basic Fenwick Tree
    print("\n[1] Basic Fenwick Tree")
    arr = [1, 3, 5, 7, 9, 11]
    ft = FenwickTree.from_array(arr)
    print(f"    Array: {arr}")
    print(f"    prefix_sum(3): {ft.prefix_sum(3)}")  # 1+3+5+7=16
    print(f"    range_sum(1, 4): {ft.range_sum(1, 4)}")  # 3+5+7+9=24
    ft.update(2, 5)  # 5 -> 10
    print(f"    After update(2, +5) range_sum(1, 4): {ft.range_sum(1, 4)}")  # 3+10+7+9=29

    # 2. Range Update + Point Query
    print("\n[2] Range Update + Point Query")
    ft_ru = FenwickTreeRangeUpdate(6)
    ft_ru.update_range(1, 3, 5)  # Add 5 to [1,3]
    ft_ru.update_range(2, 4, 3)  # Add 3 to [2,4]
    print(f"    update_range(1, 3, +5), update_range(2, 4, +3)")
    for i in range(6):
        print(f"    query({i}): {ft_ru.query(i)}")

    # 3. 2D Fenwick Tree
    print("\n[3] 2D Fenwick Tree")
    ft2d = FenwickTree2D(3, 3)
    # Fill matrix
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    for i in range(3):
        for j in range(3):
            ft2d.update(i, j, matrix[i][j])
    print(f"    Matrix: {matrix}")
    print(f"    range_sum(0,0,1,1): {ft2d.range_sum(0, 0, 1, 1)}")  # 1+2+4+5=12
    print(f"    range_sum(1,1,2,2): {ft2d.range_sum(1, 1, 2, 2)}")  # 5+6+8+9=28

    # 4. Inversion Count
    print("\n[4] Inversion Count")
    arr = [2, 4, 1, 3, 5]
    inv = count_inversions(arr)
    print(f"    Array: {arr}")
    print(f"    Inversions: {inv}")  # (2,1), (4,1), (4,3) = 3

    # 5. K-th Element
    print("\n[5] K-th Element")
    ft_kth = FenwickTreeKth(10)
    for val in [3, 5, 7, 1, 9]:
        ft_kth.update(val, 1)
    print(f"    Inserted elements: [3, 5, 7, 1, 9]")
    for k in [1, 2, 3, 4, 5]:
        print(f"    {k}-th element: {ft_kth.find_kth(k)}")

    # 6. Fenwick Tree vs Segment Tree Comparison
    print("\n[6] Fenwick Tree vs Segment Tree")
    print("    | Property      | Fenwick Tree | Segment Tree  |")
    print("    |---------------|-------------|---------------|")
    print("    | Space         | O(n)        | O(4n)         |")
    print("    | Implementation| Easy        | Moderate      |")
    print("    | Point update  | O(log n)    | O(log n)      |")
    print("    | Range query   | O(log n)    | O(log n)      |")
    print("    | Range update  | 2 BITs      | Lazy          |")
    print("    | Operations    | Invertible  | Arbitrary     |")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
