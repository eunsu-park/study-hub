"""
Exercises for Lesson 28: Advanced DP Optimization
Topic: Algorithm

Solutions to practice problems from the lesson.
Problems: Convex Hull Trick (CHT), D&C Optimization, Knuth Optimization.
"""


# === Exercise 1: Convex Hull Trick (CHT) ===
# Problem: Given lines y = m*x + b, answer queries: for a given x, what is
#   the minimum y across all lines?
# Approach: Maintain a set of "useful" lines. Remove lines that are never optimal.
# Application: DP optimization where dp[i] = min(dp[j] + cost(j,i)) and cost
#   can be decomposed into a linear function.

def exercise_1():
    """Solution: Convex Hull Trick for minimum query over lines."""
    class ConvexHullTrick:
        """Maintains lines y = mx + b for minimum y queries.
        Assumes queries come in increasing x order (monotone CHT)."""

        def __init__(self):
            self.lines = []  # (slope, intercept)
            self.ptr = 0

        def _bad(self, l1, l2, l3):
            """Check if l2 is never the best between l1 and l3."""
            # l2 is bad if the intersection of l1 and l3 is to the left of
            # the intersection of l1 and l2
            return ((l3[1] - l1[1]) * (l1[0] - l2[0]) <=
                    (l2[1] - l1[1]) * (l1[0] - l3[0]))

        def add_line(self, m, b):
            """Add line y = mx + b. Lines must be added in decreasing slope order."""
            line = (m, b)
            while len(self.lines) >= 2 and self._bad(self.lines[-2], self.lines[-1], line):
                self.lines.pop()
            self.lines.append(line)

        def query(self, x):
            """Get minimum y for given x. Queries must come in increasing x order."""
            if not self.lines:
                return float('inf')
            # Pointer-based: amortized O(1) per query for sorted x
            while (self.ptr < len(self.lines) - 1 and
                   self.lines[self.ptr][0] * x + self.lines[self.ptr][1] >=
                   self.lines[self.ptr + 1][0] * x + self.lines[self.ptr + 1][1]):
                self.ptr += 1
            m, b = self.lines[self.ptr]
            return m * x + b

    # Test: add lines and query
    cht = ConvexHullTrick()
    # Lines (decreasing slope order): y = 3x+1, y = 2x+2, y = 1x+5, y = 0x+10
    cht.add_line(3, 1)
    cht.add_line(2, 2)
    cht.add_line(1, 5)
    cht.add_line(0, 10)

    # Queries in increasing x order
    queries = [0, 1, 2, 3, 5, 10]
    print("CHT minimum queries:")
    for x in queries:
        result = cht.query(x)
        # Verify against brute force
        brute = min(3 * x + 1, 2 * x + 2, 1 * x + 5, 0 * x + 10)
        print(f"  x={x}: CHT={result}, brute={brute}")
        assert result == brute

    print("All CHT tests passed!")


# === Exercise 2: DP with CHT Application ===
# Problem: Partition array into groups with cost. Minimize total cost.
#   dp[i] = min over j < i of (dp[j] + (prefix[i] - prefix[j])^2)
#   This can be decomposed for CHT optimization.

def exercise_2():
    """Solution: DP optimization using CHT."""
    def min_partition_cost(arr):
        """
        dp[i] = min cost to partition arr[0..i-1]
        dp[i] = min(dp[j] + (S[i] - S[j])^2) for j < i
        where S[i] = prefix sum up to i

        Expanding: dp[j] + S[i]^2 - 2*S[i]*S[j] + S[j]^2
        = (dp[j] + S[j]^2) + S[i]^2 - 2*S[i]*S[j]
        = (-2*S[j])*S[i] + (dp[j] + S[j]^2) + S[i]^2

        This is a line: y = m*x + b where:
          m = -2*S[j], b = dp[j] + S[j]^2, x = S[i]
        And we want to minimize y + S[i]^2, but S[i]^2 is constant for given i.
        So we minimize y = m*x + b using CHT.
        """
        n = len(arr)
        # Prefix sums
        S = [0] * (n + 1)
        for i in range(n):
            S[i + 1] = S[i] + arr[i]

        dp = [float('inf')] * (n + 1)
        dp[0] = 0

        # Use CHT with lines: slope = -2*S[j], intercept = dp[j] + S[j]^2
        # S[j] increases, so slopes decrease (good for monotone CHT)
        lines = []  # (slope, intercept)
        ptr = [0]

        def add_line(m, b):
            line = (m, b)
            while len(lines) >= 2:
                l1, l2 = lines[-2], lines[-1]
                if ((line[1] - l1[1]) * (l1[0] - l2[0]) <=
                        (l2[1] - l1[1]) * (l1[0] - line[0])):
                    lines.pop()
                else:
                    break
            lines.append(line)

        def query(x):
            while (ptr[0] < len(lines) - 1 and
                   lines[ptr[0]][0] * x + lines[ptr[0]][1] >=
                   lines[ptr[0] + 1][0] * x + lines[ptr[0] + 1][1]):
                ptr[0] += 1
            m, b = lines[ptr[0]]
            return m * x + b

        # Add line for j=0
        add_line(-2 * S[0], dp[0] + S[0] ** 2)

        for i in range(1, n + 1):
            # Query at x = S[i]
            dp[i] = query(S[i]) + S[i] ** 2
            # Add line for j=i
            add_line(-2 * S[i], dp[i] + S[i] ** 2)

        return dp[n]

    # Brute force DP for verification
    def min_partition_cost_brute(arr):
        n = len(arr)
        S = [0] * (n + 1)
        for i in range(n):
            S[i + 1] = S[i] + arr[i]
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        for i in range(1, n + 1):
            for j in range(i):
                dp[i] = min(dp[i], dp[j] + (S[i] - S[j]) ** 2)
        return dp[n]

    tests = [
        [1, 2, 3],
        [1, 1, 1, 1],
        [5],
        [1, 2, 3, 4, 5],
        [10, 1, 10, 1, 10],
    ]

    for arr in tests:
        result = min_partition_cost(arr)
        expected = min_partition_cost_brute(arr)
        print(f"arr={arr}: CHT={result}, brute={expected}")
        assert result == expected

    print("All DP with CHT tests passed!")


# === Exercise 3: Knuth Optimization ===
# Problem: Optimal binary search tree / file merging problem.
#   dp[i][j] = min cost to merge files i..j
#   dp[i][j] = min over k in [i, j-1] of (dp[i][k] + dp[k+1][j]) + sum(i, j)
#   Knuth: if C[i][j] satisfies quadrangle inequality, then opt[i][j-1] <= opt[i][j] <= opt[i+1][j]
#   This reduces O(n^3) to O(n^2).

def exercise_3():
    """Solution: File merging with Knuth optimization in O(n^2)."""
    def file_merge_knuth(sizes):
        """
        Minimize total cost of merging consecutive files.
        dp[i][j] = min cost to merge files i..j (0-indexed)
        """
        n = len(sizes)
        if n <= 1:
            return 0

        # Prefix sums for range sum queries
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + sizes[i]

        def range_sum(i, j):
            return prefix[j + 1] - prefix[i]

        INF = float('inf')
        dp = [[0] * n for _ in range(n)]
        opt = [[0] * n for _ in range(n)]  # optimal split point

        # Base case: single files have no merge cost
        for i in range(n):
            opt[i][i] = i

        # Fill by increasing interval length
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                dp[i][j] = INF
                # Knuth optimization: search range restricted by opt[i][j-1]..opt[i+1][j]
                lo = opt[i][j - 1]
                hi = opt[i + 1][j] if i + 1 <= j else j

                for k in range(lo, min(hi, j) + 1):
                    cost = dp[i][k] + (dp[k + 1][j] if k + 1 <= j else 0) + range_sum(i, j)
                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        opt[i][j] = k

        return dp[0][n - 1]

    # Brute force O(n^3) DP for verification
    def file_merge_brute(sizes):
        n = len(sizes)
        if n <= 1:
            return 0
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + sizes[i]
        INF = float('inf')
        dp = [[INF] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = 0
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                for k in range(i, j):
                    cost = dp[i][k] + dp[k + 1][j] + prefix[j + 1] - prefix[i]
                    dp[i][j] = min(dp[i][j], cost)
        return dp[0][n - 1]

    tests = [
        [40, 30, 30],
        [10, 20, 30],
        [10, 20, 30, 40],
        [1, 2, 3, 4, 5],
        [5],
        [3, 3],
    ]

    for sizes in tests:
        result = file_merge_knuth(sizes)
        expected = file_merge_brute(sizes)
        print(f"sizes={sizes}: Knuth={result}, brute={expected}")
        assert result == expected

    print("All File Merge (Knuth) tests passed!")


if __name__ == "__main__":
    print("=== Exercise 1: Convex Hull Trick ===")
    exercise_1()
    print("\n=== Exercise 2: DP with CHT Application ===")
    exercise_2()
    print("\n=== Exercise 3: Knuth Optimization (File Merge) ===")
    exercise_3()
    print("\nAll exercises completed!")
