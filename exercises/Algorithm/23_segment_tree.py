"""
Exercises for Lesson 23: Segment Tree
Topic: Algorithm

Solutions to practice problems from the lesson.
Problems: Range Sum Query (mutable), Minimum Query, Lazy Propagation.
"""


# === Exercise 1: Segment Tree for Range Sum Query ===
# Problem: Support point updates and range sum queries on an array.
# Approach: Build a segment tree with O(n) build, O(log n) update, O(log n) query.

def exercise_1():
    """Solution: Basic Segment Tree for range sum queries."""
    class SegmentTree:
        def __init__(self, arr):
            self.n = len(arr)
            self.tree = [0] * (4 * self.n)
            if self.n > 0:
                self._build(arr, 1, 0, self.n - 1)

        def _build(self, arr, node, start, end):
            if start == end:
                self.tree[node] = arr[start]
            else:
                mid = (start + end) // 2
                self._build(arr, 2 * node, start, mid)
                self._build(arr, 2 * node + 1, mid + 1, end)
                self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

        def update(self, idx, val):
            """Set arr[idx] = val."""
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
                self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

        def query(self, l, r):
            """Return sum of arr[l..r]."""
            return self._query(1, 0, self.n - 1, l, r)

        def _query(self, node, start, end, l, r):
            if r < start or end < l:
                return 0  # Out of range
            if l <= start and end <= r:
                return self.tree[node]  # Fully in range
            mid = (start + end) // 2
            left_sum = self._query(2 * node, start, mid, l, r)
            right_sum = self._query(2 * node + 1, mid + 1, end, l, r)
            return left_sum + right_sum

    arr = [1, 3, 5, 7, 9, 11]
    st = SegmentTree(arr)

    # Range queries
    print(f"sum(1, 3) = {st.query(1, 3)}")  # 3+5+7 = 15
    assert st.query(1, 3) == 15

    print(f"sum(0, 5) = {st.query(0, 5)}")  # 1+3+5+7+9+11 = 36
    assert st.query(0, 5) == 36

    # Point update
    st.update(1, 10)  # arr[1] = 10
    print(f"After update arr[1]=10, sum(1, 3) = {st.query(1, 3)}")  # 10+5+7 = 22
    assert st.query(1, 3) == 22

    st.update(3, 0)   # arr[3] = 0
    print(f"After update arr[3]=0, sum(0, 5) = {st.query(0, 5)}")  # 1+10+5+0+9+11 = 36
    assert st.query(0, 5) == 36

    print("All Range Sum Query tests passed!")


# === Exercise 2: Segment Tree for Range Minimum Query ===
# Problem: Support point updates and range minimum queries.

def exercise_2():
    """Solution: Segment Tree for range minimum queries."""
    class MinSegmentTree:
        def __init__(self, arr):
            self.n = len(arr)
            self.tree = [float('inf')] * (4 * self.n)
            if self.n > 0:
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

        def query(self, l, r):
            """Return minimum of arr[l..r]."""
            return self._query(1, 0, self.n - 1, l, r)

        def _query(self, node, start, end, l, r):
            if r < start or end < l:
                return float('inf')
            if l <= start and end <= r:
                return self.tree[node]
            mid = (start + end) // 2
            return min(
                self._query(2 * node, start, mid, l, r),
                self._query(2 * node + 1, mid + 1, end, l, r)
            )

    arr = [5, 1, 4, 2, 8, 3, 7, 6]
    st = MinSegmentTree(arr)

    print(f"min(0, 7) = {st.query(0, 7)}")  # 1
    assert st.query(0, 7) == 1

    print(f"min(2, 5) = {st.query(2, 5)}")  # min(4,2,8,3) = 2
    assert st.query(2, 5) == 2

    print(f"min(4, 7) = {st.query(4, 7)}")  # min(8,3,7,6) = 3
    assert st.query(4, 7) == 3

    st.update(1, 10)  # arr[1] = 10
    print(f"After arr[1]=10, min(0, 3) = {st.query(0, 3)}")  # min(5,10,4,2) = 2
    assert st.query(0, 3) == 2

    st.update(3, 0)
    print(f"After arr[3]=0, min(0, 3) = {st.query(0, 3)}")  # min(5,10,4,0) = 0
    assert st.query(0, 3) == 0

    print("All Range Minimum Query tests passed!")


# === Exercise 3: Segment Tree with Lazy Propagation ===
# Problem: Support range updates (add val to all elements in [l, r])
#   and range sum queries, both in O(log n).

def exercise_3():
    """Solution: Segment Tree with lazy propagation for range updates."""
    class LazySegmentTree:
        def __init__(self, arr):
            self.n = len(arr)
            self.tree = [0] * (4 * self.n)
            self.lazy = [0] * (4 * self.n)
            if self.n > 0:
                self._build(arr, 1, 0, self.n - 1)

        def _build(self, arr, node, start, end):
            if start == end:
                self.tree[node] = arr[start]
            else:
                mid = (start + end) // 2
                self._build(arr, 2 * node, start, mid)
                self._build(arr, 2 * node + 1, mid + 1, end)
                self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

        def _push_down(self, node, start, end):
            """Propagate pending lazy update to children."""
            if self.lazy[node] != 0:
                mid = (start + end) // 2
                # Apply to left child
                self.tree[2 * node] += self.lazy[node] * (mid - start + 1)
                self.lazy[2 * node] += self.lazy[node]
                # Apply to right child
                self.tree[2 * node + 1] += self.lazy[node] * (end - mid)
                self.lazy[2 * node + 1] += self.lazy[node]
                # Clear lazy
                self.lazy[node] = 0

        def range_update(self, l, r, val):
            """Add val to all elements in [l, r]."""
            self._range_update(1, 0, self.n - 1, l, r, val)

        def _range_update(self, node, start, end, l, r, val):
            if r < start or end < l:
                return
            if l <= start and end <= r:
                self.tree[node] += val * (end - start + 1)
                self.lazy[node] += val
                return
            self._push_down(node, start, end)
            mid = (start + end) // 2
            self._range_update(2 * node, start, mid, l, r, val)
            self._range_update(2 * node + 1, mid + 1, end, l, r, val)
            self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

        def query(self, l, r):
            """Return sum of arr[l..r]."""
            return self._query(1, 0, self.n - 1, l, r)

        def _query(self, node, start, end, l, r):
            if r < start or end < l:
                return 0
            if l <= start and end <= r:
                return self.tree[node]
            self._push_down(node, start, end)
            mid = (start + end) // 2
            return (self._query(2 * node, start, mid, l, r) +
                    self._query(2 * node + 1, mid + 1, end, l, r))

    arr = [1, 2, 3, 4, 5]
    st = LazySegmentTree(arr)

    print(f"Initial sum(0, 4) = {st.query(0, 4)}")  # 15
    assert st.query(0, 4) == 15

    st.range_update(1, 3, 10)  # Add 10 to indices 1,2,3
    # arr becomes [1, 12, 13, 14, 5]
    print(f"After +10 on [1,3], sum(0, 4) = {st.query(0, 4)}")  # 45
    assert st.query(0, 4) == 45

    print(f"sum(1, 3) = {st.query(1, 3)}")  # 12+13+14 = 39
    assert st.query(1, 3) == 39

    st.range_update(0, 4, -5)  # Subtract 5 from all
    # arr becomes [-4, 7, 8, 9, 0]
    print(f"After -5 on [0,4], sum(0, 4) = {st.query(0, 4)}")  # 20
    assert st.query(0, 4) == 20

    print(f"sum(2, 4) = {st.query(2, 4)}")  # 8+9+0 = 17
    assert st.query(2, 4) == 17

    print("All Lazy Propagation tests passed!")


if __name__ == "__main__":
    print("=== Exercise 1: Range Sum Query (Segment Tree) ===")
    exercise_1()
    print("\n=== Exercise 2: Range Minimum Query ===")
    exercise_2()
    print("\n=== Exercise 3: Lazy Propagation ===")
    exercise_3()
    print("\nAll exercises completed!")
