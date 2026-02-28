"""
Exercises for Lesson 24: Fenwick Tree (Binary Indexed Tree)
Topic: Algorithm

Solutions to practice problems from the lesson.
Problems: Range Sum Query, Inversion Count, 2D BIT.
"""


# === Exercise 1: Fenwick Tree for Range Sum Query ===
# Problem: Support point updates and prefix/range sum queries.
# Approach: BIT with O(log n) update and O(log n) query.

def exercise_1():
    """Solution: Basic Fenwick Tree (1-indexed)."""
    class FenwickTree:
        def __init__(self, n):
            self.n = n
            self.tree = [0] * (n + 1)  # 1-indexed

        def update(self, i, delta):
            """Add delta to index i (1-indexed)."""
            while i <= self.n:
                self.tree[i] += delta
                i += i & (-i)  # Move to parent in update tree

        def prefix_sum(self, i):
            """Return sum of elements [1..i]."""
            s = 0
            while i > 0:
                s += self.tree[i]
                i -= i & (-i)  # Move to parent in query tree
            return s

        def range_sum(self, l, r):
            """Return sum of elements [l..r] (1-indexed)."""
            return self.prefix_sum(r) - self.prefix_sum(l - 1)

    # Build from array
    arr = [0, 1, 3, 5, 7, 9, 11]  # 1-indexed (index 0 unused)
    n = len(arr) - 1
    bit = FenwickTree(n)
    for i in range(1, n + 1):
        bit.update(i, arr[i])

    # Queries
    print(f"prefix_sum(3) = {bit.prefix_sum(3)}")  # 1+3+5 = 9
    assert bit.prefix_sum(3) == 9

    print(f"range_sum(2, 5) = {bit.range_sum(2, 5)}")  # 3+5+7+9 = 24
    assert bit.range_sum(2, 5) == 24

    # Point update: add 3 to index 2
    bit.update(2, 3)
    print(f"After +3 at index 2, range_sum(1, 3) = {bit.range_sum(1, 3)}")  # 1+6+5 = 12
    assert bit.range_sum(1, 3) == 12

    print("All Fenwick Tree Range Sum tests passed!")


# === Exercise 2: Count Inversions Using Fenwick Tree ===
# Problem: Count pairs (i, j) where i < j and arr[i] > arr[j].
# Approach: Process elements left to right. For each element, count how many
#   previously inserted elements are larger (inversions). Use coordinate compression.

def exercise_2():
    """Solution: Inversion count with BIT and coordinate compression."""
    class FenwickTree:
        def __init__(self, n):
            self.n = n
            self.tree = [0] * (n + 1)

        def update(self, i, delta=1):
            while i <= self.n:
                self.tree[i] += delta
                i += i & (-i)

        def prefix_sum(self, i):
            s = 0
            while i > 0:
                s += self.tree[i]
                i -= i & (-i)
            return s

    def count_inversions(arr):
        if not arr:
            return 0

        # Coordinate compression: map values to ranks 1..n
        sorted_unique = sorted(set(arr))
        rank = {v: i + 1 for i, v in enumerate(sorted_unique)}
        max_rank = len(sorted_unique)

        bit = FenwickTree(max_rank)
        inversions = 0

        for val in arr:
            r = rank[val]
            # Count elements already inserted with rank > r
            # This equals (total inserted so far) - (count of elements <= r)
            inversions += bit.prefix_sum(max_rank) - bit.prefix_sum(r)
            bit.update(r)

        return inversions

    # Brute force verifier
    def brute_inversions(arr):
        count = 0
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                if arr[i] > arr[j]:
                    count += 1
        return count

    tests = [
        [2, 4, 1, 3, 5],
        [5, 4, 3, 2, 1],
        [1, 2, 3, 4, 5],
        [3, 1, 2],
        [1],
    ]

    for arr in tests:
        result = count_inversions(arr)
        expected = brute_inversions(arr)
        print(f"inversions({arr}) = {result}")
        assert result == expected, f"Expected {expected}, got {result}"

    print("All Inversion Count tests passed!")


# === Exercise 3: 2D Fenwick Tree ===
# Problem: Support point updates and 2D prefix sum queries on a matrix.
# This is a more advanced extension from the recommended problems.

def exercise_3():
    """Solution: 2D Binary Indexed Tree."""
    class FenwickTree2D:
        def __init__(self, rows, cols):
            self.rows = rows
            self.cols = cols
            self.tree = [[0] * (cols + 1) for _ in range(rows + 1)]

        def update(self, r, c, delta):
            """Add delta to cell (r, c) (1-indexed)."""
            i = r
            while i <= self.rows:
                j = c
                while j <= self.cols:
                    self.tree[i][j] += delta
                    j += j & (-j)
                i += i & (-i)

        def prefix_sum(self, r, c):
            """Return sum of submatrix (1,1) to (r, c)."""
            s = 0
            i = r
            while i > 0:
                j = c
                while j > 0:
                    s += self.tree[i][j]
                    j -= j & (-j)
                i -= i & (-i)
            return s

        def range_sum(self, r1, c1, r2, c2):
            """Return sum of submatrix (r1, c1) to (r2, c2) (1-indexed)."""
            return (self.prefix_sum(r2, c2)
                    - self.prefix_sum(r1 - 1, c2)
                    - self.prefix_sum(r2, c1 - 1)
                    + self.prefix_sum(r1 - 1, c1 - 1))

    # 3x3 matrix (1-indexed):
    # 1 2 3
    # 4 5 6
    # 7 8 9
    matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]

    rows, cols = 3, 3
    bit2d = FenwickTree2D(rows, cols)
    for i in range(rows):
        for j in range(cols):
            bit2d.update(i + 1, j + 1, matrix[i][j])

    # Query: sum of entire matrix
    total = bit2d.range_sum(1, 1, 3, 3)
    print(f"Total sum: {total}")
    assert total == 45  # 1+2+...+9 = 45

    # Query: sum of submatrix (2,2) to (3,3)
    sub = bit2d.range_sum(2, 2, 3, 3)
    print(f"Sum (2,2)-(3,3): {sub}")
    assert sub == 28  # 5+6+8+9 = 28

    # Point update: add 10 to (1,1)
    bit2d.update(1, 1, 10)
    total = bit2d.range_sum(1, 1, 3, 3)
    print(f"After +10 at (1,1), total: {total}")
    assert total == 55

    print("All 2D Fenwick Tree tests passed!")


if __name__ == "__main__":
    print("=== Exercise 1: Fenwick Tree Range Sum ===")
    exercise_1()
    print("\n=== Exercise 2: Count Inversions ===")
    exercise_2()
    print("\n=== Exercise 3: 2D Fenwick Tree ===")
    exercise_3()
    print("\nAll exercises completed!")
