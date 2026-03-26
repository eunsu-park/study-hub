"""
Advanced DP Optimization
Advanced Dynamic Programming Optimization Techniques

Optimization techniques for improving the time complexity of DP solutions.
"""

from typing import List, Tuple, Callable
from collections import deque
from math import inf


# =============================================================================
# 1. Convex Hull Trick (CHT)
# =============================================================================

class ConvexHullTrick:
    """
    Convex Hull Trick
    Minimum query: min(a[i] * x + b[i]) for all i
    Condition: a[i] must be monotonically decreasing (or increasing)

    Time Complexity: insertion O(1) amortized, query O(log n) or O(1)
    """

    def __init__(self):
        self.lines = deque()  # (slope, y-intercept)

    def is_bad(self, l1: Tuple[int, int], l2: Tuple[int, int], l3: Tuple[int, int]) -> bool:
        """Check if l2 is unnecessary (between l1 and l3)"""
        # Intersection comparison: if (l1, l2) intersection >= (l2, l3) intersection, l2 is unnecessary
        # (b2 - b1) / (a1 - a2) >= (b3 - b2) / (a2 - a3)
        # (b2 - b1) * (a2 - a3) >= (b3 - b2) * (a1 - a2)
        a1, b1 = l1
        a2, b2 = l2
        a3, b3 = l3
        return (b2 - b1) * (a2 - a3) >= (b3 - b2) * (a1 - a2)

    def add_line(self, a: int, b: int):
        """
        Add line y = ax + b
        a must be monotonically decreasing
        """
        line = (a, b)

        while len(self.lines) >= 2 and self.is_bad(self.lines[-2], self.lines[-1], line):
            self.lines.pop()

        self.lines.append(line)

    def query_min(self, x: int) -> int:
        """
        Minimum value query at x
        O(1) when x is monotonically increasing
        """
        while len(self.lines) >= 2:
            a1, b1 = self.lines[0]
            a2, b2 = self.lines[1]
            if a1 * x + b1 >= a2 * x + b2:
                self.lines.popleft()
            else:
                break

        a, b = self.lines[0]
        return a * x + b


class LiChaoTree:
    """
    Li Chao Tree (Segment tree-based CHT)
    Supports arbitrary line insertion and query at any x

    Time Complexity: insertion O(log C), query O(log C)
    C = coordinate range
    """

    def __init__(self, lo: int, hi: int):
        self.lo = lo
        self.hi = hi
        self.tree = {}  # Dominant line stored per node

    def _eval(self, line: Tuple[int, int], x: int) -> int:
        """Evaluate line value"""
        if line is None:
            return inf
        a, b = line
        return a * x + b

    def add_line(self, a: int, b: int):
        """Add a line"""
        self._add_line_impl((a, b), self.lo, self.hi, 1)

    def _add_line_impl(self, new_line: Tuple[int, int], lo: int, hi: int, node: int):
        if lo > hi:
            return

        mid = (lo + hi) // 2
        cur_line = self.tree.get(node)

        # Compare at midpoint
        new_better_at_mid = self._eval(new_line, mid) < self._eval(cur_line, mid)

        if cur_line is None or new_better_at_mid:
            self.tree[node], new_line = new_line, cur_line

        if lo == hi or new_line is None:
            return

        # Propagate to left/right children
        new_better_at_lo = self._eval(new_line, lo) < self._eval(self.tree.get(node), lo)

        if new_better_at_lo:
            self._add_line_impl(new_line, lo, mid - 1, 2 * node)
        else:
            self._add_line_impl(new_line, mid + 1, hi, 2 * node + 1)

    def query(self, x: int) -> int:
        """Minimum value at x"""
        return self._query_impl(x, self.lo, self.hi, 1)

    def _query_impl(self, x: int, lo: int, hi: int, node: int) -> int:
        if lo > hi:
            return inf

        result = self._eval(self.tree.get(node), x)

        if lo == hi:
            return result

        mid = (lo + hi) // 2
        if x <= mid:
            return min(result, self._query_impl(x, lo, mid - 1, 2 * node))
        else:
            return min(result, self._query_impl(x, mid + 1, hi, 2 * node + 1))


# =============================================================================
# 2. Divide and Conquer Optimization
# =============================================================================

def dc_optimization(n: int, m: int, cost: Callable[[int, int], int]) -> List[List[int]]:
    """
    Divide and Conquer Optimization
    Condition: opt[k][i] <= opt[k][i+1] (monotonicity)
    Recurrence: dp[k][j] = min(dp[k-1][i] + cost(i, j)) for i < j

    Time Complexity: O(k * n log n) (improved from O(k * n^2))

    n: number of elements
    m: number of partition groups
    cost(i, j): cost of the interval from i+1 to j
    """
    INF = float('inf')
    dp = [[INF] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 0

    def compute(k: int, lo: int, hi: int, opt_lo: int, opt_hi: int):
        """Compute dp[k][lo:hi+1], optimal split point is in range opt_lo to opt_hi"""
        if lo > hi:
            return

        mid = (lo + hi) // 2
        best_cost = INF
        best_opt = opt_lo

        for i in range(opt_lo, min(opt_hi, mid) + 1):
            curr_cost = dp[k - 1][i] + cost(i, mid)
            if curr_cost < best_cost:
                best_cost = curr_cost
                best_opt = i

        dp[k][mid] = best_cost

        # Divide and conquer
        compute(k, lo, mid - 1, opt_lo, best_opt)
        compute(k, mid + 1, hi, best_opt, opt_hi)

    for k in range(1, m + 1):
        compute(k, k, n, k - 1, n - 1)

    return dp


def dc_optimization_example():
    """
    Example: partition an array into k groups
    Cost of each group = sum of squared differences of elements in the interval
    """
    arr = [1, 5, 2, 8, 3, 7, 4, 6]
    n = len(arr)
    k = 3

    # Preprocessing: prefix sum for cost calculation
    prefix = [0] * (n + 1)
    prefix_sq = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + arr[i]
        prefix_sq[i + 1] = prefix_sq[i] + arr[i] * arr[i]

    def cost(l: int, r: int) -> int:
        """Variance of interval [l+1, r] (sum of squares - mean * sum)"""
        if l >= r:
            return 0
        length = r - l
        s = prefix[r] - prefix[l]
        sq = prefix_sq[r] - prefix_sq[l]
        # Variance = E[X^2] - E[X]^2, here: sum_of_squares - sum^2/n
        return sq * length - s * s

    dp = dc_optimization(n, k, cost)
    return dp[k][n]


# =============================================================================
# 3. Knuth Optimization
# =============================================================================

def knuth_optimization(n: int, cost: List[List[int]]) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Knuth Optimization
    Condition: cost satisfies the Quadrangle Inequality
              cost[a][c] + cost[b][d] <= cost[a][d] + cost[b][c] (a <= b <= c <= d)
    Recurrence: dp[i][j] = min(dp[i][k] + dp[k][j]) + cost[i][j] for i < k < j

    Time Complexity: O(n^2) (improved from O(n^3))

    Examples: optimal binary search tree, matrix chain multiplication
    """
    INF = float('inf')
    dp = [[0] * n for _ in range(n)]
    opt = [[0] * n for _ in range(n)]

    # Base case: length 1
    for i in range(n):
        opt[i][i] = i

    # Length 2 and above
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = INF

            # opt[i][j-1] <= opt[i][j] <= opt[i+1][j]
            lo = opt[i][j - 1] if j > 0 else i
            hi = opt[i + 1][j] if i + 1 < n else j

            for k in range(lo, min(hi, j) + 1):
                curr = dp[i][k] + dp[k + 1][j] + cost[i][j]
                if curr < dp[i][j]:
                    dp[i][j] = curr
                    opt[i][j] = k

    return dp, opt


def optimal_bst(keys: List[int], freq: List[int]) -> int:
    """
    Optimal Binary Search Tree
    freq[i] = access frequency of keys[i]
    """
    n = len(keys)

    # cost[i][j] = freq[i] + ... + freq[j]
    cost = [[0] * n for _ in range(n)]
    for i in range(n):
        total = 0
        for j in range(i, n):
            total += freq[j]
            cost[i][j] = total

    dp, opt = knuth_optimization(n, cost)
    return dp[0][n - 1]


# =============================================================================
# 4. 1D/1D DP Optimization (Monotone Queue)
# =============================================================================

def sliding_window_max(arr: List[int], k: int) -> List[int]:
    """
    Sliding Window Maximum
    Time Complexity: O(n)
    """
    result = []
    dq = deque()  # (index, value)

    for i, val in enumerate(arr):
        # Remove elements outside window
        while dq and dq[0][0] <= i - k:
            dq.popleft()

        # Remove elements smaller than current value
        while dq and dq[-1][1] <= val:
            dq.pop()

        dq.append((i, val))

        if i >= k - 1:
            result.append(dq[0][1])

    return result


def dp_with_monotone_queue(arr: List[int], k: int) -> List[int]:
    """
    DP with Monotone Queue
    dp[i] = max(dp[j] + arr[i]) for i-k <= j < i
    Time Complexity: O(n)
    """
    n = len(arr)
    dp = [0] * n
    dq = deque()  # (index, dp value)

    for i in range(n):
        # Remove elements outside range
        while dq and dq[0][0] < i - k:
            dq.popleft()

        # Compute dp using maximum
        if dq:
            dp[i] = dq[0][1] + arr[i]
        else:
            dp[i] = arr[i]

        # Insert current dp value
        while dq and dq[-1][1] <= dp[i]:
            dq.pop()
        dq.append((i, dp[i]))

    return dp


# =============================================================================
# 5. Slope Trick
# =============================================================================

class SlopeTrick:
    """
    Slope Trick
    Efficient management of convex functions
    Useful for optimizing sums of absolute value functions
    """

    def __init__(self):
        import heapq
        self.left = []   # Max heap (stored as negatives)
        self.right = []  # Min heap
        self.min_f = 0
        self.add_l = 0   # Value to add to left
        self.add_r = 0   # Value to add to right

    def add_abs(self, a: int):
        """
        f(x) += |x - a|
        """
        import heapq

        # Slope change at position a: left +1, right -1
        l = -self.left[0] + self.add_l if self.left else -inf
        r = self.right[0] + self.add_r if self.right else inf

        if a <= l:
            # a is on the left side
            self.min_f += l - a
            heapq.heappush(self.left, -(a - self.add_l))
            # Move left max to right
            val = -heapq.heappop(self.left) + self.add_l
            heapq.heappush(self.right, val - self.add_r)
        elif a >= r:
            # a is on the right side
            self.min_f += a - r
            heapq.heappush(self.right, a - self.add_r)
            # Move right min to left
            val = heapq.heappop(self.right) + self.add_r
            heapq.heappush(self.left, -(val - self.add_l))
        else:
            # a is in the flat region
            heapq.heappush(self.left, -(a - self.add_l))
            heapq.heappush(self.right, a - self.add_r)

    def shift(self, a: int, b: int):
        """
        f(x) -> f(x-a) (left shift), f(x) -> f(x-b) (right shift)
        Extends the flat region
        """
        self.add_l += a
        self.add_r += b

    def get_min(self) -> int:
        """Return minimum value"""
        return self.min_f


# =============================================================================
# 6. Alien Trick (WQS Binary Search)
# =============================================================================

def alien_trick_example(arr: List[int], k: int) -> int:
    """
    Alien Trick (WQS Binary Search / Lagrange Relaxation)
    Relaxes the problem of selecting exactly k elements

    Example: select exactly k elements from array, no two adjacent, maximize sum
    """

    def check(penalty: float) -> Tuple[float, int]:
        """
        Solve the relaxed problem with penalty
        Returns: (optimal value, number of selected elements)
        """
        n = len(arr)
        # dp[i][0]: up to i, arr[i] not selected
        # dp[i][1]: up to i, arr[i] selected

        dp = [[-inf, -inf] for _ in range(n)]
        cnt = [[0, 0] for _ in range(n)]

        dp[0][0] = 0
        dp[0][1] = arr[0] - penalty
        cnt[0][1] = 1

        for i in range(1, n):
            # Not selected
            if dp[i - 1][0] > dp[i - 1][1]:
                dp[i][0] = dp[i - 1][0]
                cnt[i][0] = cnt[i - 1][0]
            else:
                dp[i][0] = dp[i - 1][1]
                cnt[i][0] = cnt[i - 1][1]

            # Selected (only from previous not-selected state)
            dp[i][1] = dp[i - 1][0] + arr[i] - penalty
            cnt[i][1] = cnt[i - 1][0] + 1

        if dp[n - 1][0] > dp[n - 1][1]:
            return dp[n - 1][0], cnt[n - 1][0]
        return dp[n - 1][1], cnt[n - 1][1]

    # Binary search
    lo, hi = -10**9, 10**9

    while hi - lo > 1e-6:
        mid = (lo + hi) / 2
        _, count = check(mid)
        if count >= k:
            lo = mid
        else:
            hi = mid

    result, _ = check(lo)
    return int(result + lo * k)


# =============================================================================
# Tests
# =============================================================================

def main():
    print("=" * 60)
    print("Advanced DP Optimization Examples")
    print("=" * 60)

    # 1. Convex Hull Trick
    print("\n[1] Convex Hull Trick (CHT)")
    cht = ConvexHullTrick()
    # Lines: y = -3x + 10, y = -2x + 5, y = -1x + 3
    cht.add_line(-3, 10)
    cht.add_line(-2, 5)
    cht.add_line(-1, 3)
    print("    Lines: y=-3x+10, y=-2x+5, y=-x+3")
    for x in [0, 1, 2, 3, 4, 5]:
        print(f"    min at x={x}: {cht.query_min(x)}")

    # 2. Li Chao Tree
    print("\n[2] Li Chao Tree")
    lct = LiChaoTree(-100, 100)
    lct.add_line(2, 5)   # y = 2x + 5
    lct.add_line(-1, 10) # y = -x + 10
    lct.add_line(1, 0)   # y = x
    print("    Lines: y=2x+5, y=-x+10, y=x")
    for x in [-5, 0, 3, 7]:
        print(f"    min at x={x}: {lct.query(x)}")

    # 3. Divide and Conquer Optimization
    print("\n[3] Divide and Conquer Optimization")
    result = dc_optimization_example()
    print(f"    Partition array [1,5,2,8,3,7,4,6] into 3 groups")
    print(f"    Minimum cost: {result}")

    # 4. Knuth Optimization
    print("\n[4] Knuth Optimization (Optimal BST)")
    keys = [10, 20, 30, 40]
    freq = [4, 2, 6, 3]
    cost = optimal_bst(keys, freq)
    print(f"    Keys: {keys}")
    print(f"    Frequencies: {freq}")
    print(f"    Minimum search cost: {cost}")

    # 5. Monotone Queue
    print("\n[5] Monotone Queue Optimization")
    arr = [1, 3, -1, -3, 5, 3, 6, 7]
    k = 3
    result = sliding_window_max(arr, k)
    print(f"    Array: {arr}")
    print(f"    Window size: {k}")
    print(f"    Sliding window max: {result}")

    # 6. DP with Monotone Queue
    print("\n[6] Monotone Queue DP")
    arr = [2, 1, 5, 1, 3, 2]
    k = 2
    dp = dp_with_monotone_queue(arr, k)
    print(f"    Array: {arr}, k={k}")
    print(f"    dp[i] = max(dp[j] + arr[i]) for i-k <= j < i")
    print(f"    DP: {dp}")

    # 7. Slope Trick
    print("\n[7] Slope Trick")
    st = SlopeTrick()
    points = [1, 5, 2, 8]
    for p in points:
        st.add_abs(p)
    print(f"    Points: {points}")
    print(f"    f(x) = sum(|x - p|) minimum: {st.get_min()}")

    # 8. Complexity Comparison
    print("\n[8] Optimization Technique Comparison")
    print("    | Technique         | Original      | Optimized     | Condition               |")
    print("    |-------------------|---------------|---------------|-------------------------|")
    print("    | CHT               | O(n^2)        | O(n)          | Monotone slope          |")
    print("    | Li Chao Tree      | O(n^2)        | O(n log C)    | None                    |")
    print("    | D&C Optimization  | O(kn^2)       | O(kn log n)   | Monotone opt            |")
    print("    | Knuth Optimization| O(n^3)        | O(n^2)        | Quadrangle inequality   |")
    print("    | Monotone Queue    | O(nk)         | O(n)          | Window optimization     |")
    print("    | Alien Trick       | Constrained   | Relaxed       | Convexity               |")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
