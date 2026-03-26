"""
Practice Problems
Comprehensive Practice Problems

Problems that require combining various algorithms to solve.
"""

from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict, deque, Counter
from heapq import heappush, heappop
from functools import lru_cache
from bisect import bisect_left, bisect_right
import sys


# =============================================================================
# 1. Two Pointers + Binary Search: Subarray Sum
# =============================================================================

def subarray_sum_k(arr: List[int], k: int) -> int:
    """
    Count of subarrays with sum equal to k
    Time Complexity: O(n)
    """
    count = 0
    prefix_sum = 0
    sum_count = defaultdict(int)
    sum_count[0] = 1

    for num in arr:
        prefix_sum += num
        # If prefix_sum - k appeared before
        count += sum_count[prefix_sum - k]
        sum_count[prefix_sum] += 1

    return count


def longest_subarray_sum_at_most_k(arr: List[int], k: int) -> int:
    """
    Longest subarray with sum at most k
    (for arrays with positive values only)
    Time Complexity: O(n)
    """
    n = len(arr)
    max_len = 0
    current_sum = 0
    left = 0

    for right in range(n):
        current_sum += arr[right]

        while current_sum > k and left <= right:
            current_sum -= arr[left]
            left += 1

        max_len = max(max_len, right - left + 1)

    return max_len


# =============================================================================
# 2. BFS + DP: Shortest Path with Cost
# =============================================================================

def shortest_path_with_fuel(n: int, edges: List[Tuple[int, int, int]],
                            start: int, end: int, fuel: int) -> int:
    """
    Shortest path with fuel constraint
    edges: (u, v, fuel_cost)
    Returns: minimum distance (-1 if impossible)
    Time Complexity: O(V * F + E)
    """
    graph = defaultdict(list)
    for u, v, cost in edges:
        graph[u].append((v, cost))
        graph[v].append((u, cost))

    # BFS with state: (node, remaining_fuel)
    INF = float('inf')
    dist = [[INF] * (fuel + 1) for _ in range(n)]
    dist[start][fuel] = 0

    queue = deque([(start, fuel, 0)])  # (node, remaining_fuel, distance)

    while queue:
        node, remaining, d = queue.popleft()

        if node == end:
            return d

        if d > dist[node][remaining]:
            continue

        for neighbor, cost in graph[node]:
            if remaining >= cost:
                new_fuel = remaining - cost
                if d + 1 < dist[neighbor][new_fuel]:
                    dist[neighbor][new_fuel] = d + 1
                    queue.append((neighbor, new_fuel, d + 1))

    return -1


# =============================================================================
# 3. Greedy + Priority Queue: Job Scheduling
# =============================================================================

def max_profit_scheduling(jobs: List[Tuple[int, int, int]]) -> int:
    """
    Job scheduling: select non-overlapping jobs for maximum profit
    jobs: (start, end, profit)
    Time Complexity: O(n log n)
    """
    n = len(jobs)
    if n == 0:
        return 0

    # Sort by end time
    jobs = sorted(jobs, key=lambda x: x[1])
    end_times = [job[1] for job in jobs]

    dp = [0] * (n + 1)

    for i in range(n):
        start, end, profit = jobs[i]

        # Last job that ends before current job starts
        j = bisect_right(end_times, start) - 1

        # Select vs skip
        dp[i + 1] = max(dp[i], (dp[j + 1] if j >= 0 else 0) + profit)

    return dp[n]


# =============================================================================
# 4. String + DP: Edit Distance Application
# =============================================================================

def min_operations_to_palindrome(s: str) -> int:
    """
    Minimum insertions/deletions to make a string a palindrome
    = length - LCS(s, reverse(s))
    """
    n = len(s)
    t = s[::-1]

    # LCS
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if s[i - 1] == t[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return n - dp[n][n]


def longest_palindromic_subsequence(s: str) -> int:
    """
    Longest palindromic subsequence
    = LCS(s, reverse(s))
    """
    n = len(s)
    t = s[::-1]

    dp = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if s[i - 1] == t[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[n][n]


# =============================================================================
# 5. Graph + Union-Find: Connected Components
# =============================================================================

class DSU:
    """Disjoint Set Union"""
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        self.size[px] += self.size[py]
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True


def min_cost_to_connect_all(n: int, edges: List[Tuple[int, int, int]]) -> int:
    """
    Minimum cost to connect all nodes (MST)
    edges: (u, v, cost)
    """
    edges = sorted(edges, key=lambda x: x[2])
    dsu = DSU(n)
    total_cost = 0
    edges_used = 0

    for u, v, cost in edges:
        if dsu.union(u, v):
            total_cost += cost
            edges_used += 1
            if edges_used == n - 1:
                break

    return total_cost if edges_used == n - 1 else -1


def count_islands(grid: List[List[int]]) -> int:
    """
    Number of islands (Union-Find version)
    grid[i][j] = 1: land, 0: water
    """
    if not grid or not grid[0]:
        return 0

    rows, cols = len(grid), len(grid[0])
    dsu = DSU(rows * cols)
    count = 0

    def idx(r, c):
        return r * cols + c

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                count += 1

    directions = [(0, 1), (1, 0)]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0:
                continue
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                    if dsu.union(idx(r, c), idx(nr, nc)):
                        count -= 1

    return count


# =============================================================================
# 6. Segment Tree + Coordinate Compression: Range Queries
# =============================================================================

class SegmentTree:
    """Range sum segment tree"""
    def __init__(self, n: int):
        self.n = n
        self.tree = [0] * (4 * n)

    def update(self, node: int, start: int, end: int, idx: int, delta: int):
        if idx < start or idx > end:
            return
        if start == end:
            self.tree[node] += delta
            return
        mid = (start + end) // 2
        self.update(2 * node, start, mid, idx, delta)
        self.update(2 * node + 1, mid + 1, end, idx, delta)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def query(self, node: int, start: int, end: int, l: int, r: int) -> int:
        if r < start or end < l:
            return 0
        if l <= start and end <= r:
            return self.tree[node]
        mid = (start + end) // 2
        return (self.query(2 * node, start, mid, l, r) +
                self.query(2 * node + 1, mid + 1, end, l, r))


def count_smaller_after_self(nums: List[int]) -> List[int]:
    """
    Count of smaller elements after each element
    Segment tree + coordinate compression
    Time Complexity: O(n log n)
    """
    # Coordinate compression
    sorted_nums = sorted(set(nums))
    rank = {v: i for i, v in enumerate(sorted_nums)}
    n = len(sorted_nums)

    result = []
    st = SegmentTree(n)

    # Process in reverse order
    for num in reversed(nums):
        r = rank[num]
        # Sum of indices less than r (= count of smaller elements)
        count = st.query(1, 0, n - 1, 0, r - 1) if r > 0 else 0
        result.append(count)
        st.update(1, 0, n - 1, r, 1)

    return result[::-1]


# =============================================================================
# 7. Backtracking: Combinatorial Optimization
# =============================================================================

def solve_sudoku(board: List[List[str]]) -> bool:
    """
    Sudoku Solver
    board: 9x9, '1'-'9' or '.'
    """

    def is_valid(row: int, col: int, num: str) -> bool:
        # Row check
        if num in board[row]:
            return False
        # Column check
        for r in range(9):
            if board[r][col] == num:
                return False
        # 3x3 box check
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if board[r][c] == num:
                    return False
        return True

    def solve() -> bool:
        for row in range(9):
            for col in range(9):
                if board[row][col] == '.':
                    for num in '123456789':
                        if is_valid(row, col, num):
                            board[row][col] = num
                            if solve():
                                return True
                            board[row][col] = '.'
                    return False
        return True

    return solve()


# =============================================================================
# 8. Bitmask DP: Traveling Salesman Problem (TSP)
# =============================================================================

def tsp(dist: List[List[int]]) -> int:
    """
    Traveling Salesman Problem (TSP)
    Minimum cost to visit all cities and return to the starting point
    Time Complexity: O(n^2 * 2^n)
    """
    n = len(dist)
    INF = float('inf')

    @lru_cache(maxsize=None)
    def dp(mask: int, last: int) -> int:
        if mask == (1 << n) - 1:  # All cities visited
            return dist[last][0] if dist[last][0] > 0 else INF

        result = INF
        for next_city in range(n):
            if mask & (1 << next_city):
                continue
            if dist[last][next_city] == 0:
                continue
            result = min(result, dist[last][next_city] + dp(mask | (1 << next_city), next_city))

        return result

    return dp(1, 0)  # Start from city 0


# =============================================================================
# 9. Binary Search + Greedy: Parametric Search
# =============================================================================

def min_max_distance(houses: List[int], k: int) -> int:
    """
    Minimize the maximum distance to the farthest house by placing k post offices
    Parametric search: "Can the maximum distance be at most d?"
    Time Complexity: O(n log D)
    """
    houses = sorted(houses)
    n = len(houses)

    def can_cover(max_dist: int) -> bool:
        """Check if all houses can be covered with maximum distance max_dist"""
        count = 1
        last_post = houses[0]

        for house in houses:
            if house - last_post > 2 * max_dist:
                count += 1
                last_post = house

        return count <= k

    lo, hi = 0, houses[-1] - houses[0]

    while lo < hi:
        mid = (lo + hi) // 2
        if can_cover(mid):
            hi = mid
        else:
            lo = mid + 1

    return lo


def min_pages_per_student(pages: List[int], students: int) -> int:
    """
    Minimize the maximum pages when distributing books to students
    (only consecutive books allowed)
    """
    def is_feasible(max_pages: int) -> bool:
        count = 1
        current = 0
        for p in pages:
            if p > max_pages:
                return False
            if current + p > max_pages:
                count += 1
                current = p
            else:
                current += p
        return count <= students

    lo, hi = max(pages), sum(pages)

    while lo < hi:
        mid = (lo + hi) // 2
        if is_feasible(mid):
            hi = mid
        else:
            lo = mid + 1

    return lo


# =============================================================================
# 10. Comprehensive: LIS + Binary Search
# =============================================================================

def longest_increasing_subsequence(nums: List[int]) -> Tuple[int, List[int]]:
    """
    Longest Increasing Subsequence (length + actual sequence)
    Time Complexity: O(n log n)
    """
    n = len(nums)
    if n == 0:
        return 0, []

    # tails[i] = minimum last element among LIS of length i+1
    tails = []
    # LIS length at each position
    lengths = [0] * n
    # Previous element tracking
    prev = [-1] * n
    # Which original index each tails entry came from
    tail_idx = []

    for i, num in enumerate(nums):
        pos = bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
            tail_idx.append(i)
        else:
            tails[pos] = num
            tail_idx[pos] = i

        lengths[i] = pos + 1
        if pos > 0:
            # Previous element: last j where lengths[j] == pos
            for j in range(i - 1, -1, -1):
                if lengths[j] == pos and nums[j] < num:
                    prev[i] = j
                    break

    # Reconstruct LIS
    max_len = max(lengths)
    result = []
    idx = lengths.index(max_len)

    # Find optimal starting point
    for i in range(n - 1, -1, -1):
        if lengths[i] == max_len:
            idx = i
            break

    while idx != -1:
        result.append(nums[idx])
        # Find previous element
        target_len = lengths[idx] - 1
        if target_len == 0:
            break
        for j in range(idx - 1, -1, -1):
            if lengths[j] == target_len and nums[j] < nums[idx]:
                idx = j
                break
        else:
            break

    return max_len, result[::-1]


# =============================================================================
# Tests
# =============================================================================

def main():
    print("=" * 60)
    print("Practice Problems Examples")
    print("=" * 60)

    # 1. Subarray Sum
    print("\n[1] Subarray Sum")
    arr = [1, 2, 3, -3, 1, 2]
    k = 3
    count = subarray_sum_k(arr, k)
    print(f"    Array: {arr}, k={k}")
    print(f"    Count of subarrays with sum {k}: {count}")

    # 2. Job Scheduling
    print("\n[2] Job Scheduling (Maximum Profit)")
    jobs = [(1, 3, 50), (2, 5, 20), (4, 6, 70), (6, 7, 60)]
    profit = max_profit_scheduling(jobs)
    print(f"    Jobs (start, end, profit): {jobs}")
    print(f"    Maximum profit: {profit}")

    # 3. Palindrome
    print("\n[3] String to Palindrome")
    s = "aebcbda"
    ops = min_operations_to_palindrome(s)
    lps = longest_palindromic_subsequence(s)
    print(f"    String: '{s}'")
    print(f"    Minimum operations to palindrome: {ops}")
    print(f"    Longest palindromic subsequence length: {lps}")

    # 4. MST
    print("\n[4] Minimum Spanning Tree")
    edges = [(0, 1, 4), (0, 2, 3), (1, 2, 1), (1, 3, 2), (2, 3, 4)]
    cost = min_cost_to_connect_all(4, edges)
    print(f"    Edges: {edges}")
    print(f"    MST cost: {cost}")

    # 5. Number of Islands
    print("\n[5] Number of Islands")
    grid = [
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 1]
    ]
    islands = count_islands([row[:] for row in grid])
    print(f"    Grid:")
    for row in grid:
        print(f"      {row}")
    print(f"    Number of islands: {islands}")

    # 6. Count Smaller After Self
    print("\n[6] Count Smaller Elements After Self")
    nums = [5, 2, 6, 1]
    result = count_smaller_after_self(nums)
    print(f"    Array: {nums}")
    print(f"    Result: {result}")

    # 7. TSP
    print("\n[7] Traveling Salesman Problem (TSP)")
    dist = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    min_cost = tsp(dist)
    print(f"    Distance matrix:")
    for row in dist:
        print(f"      {row}")
    print(f"    Minimum cost: {min_cost}")

    # 8. Parametric Search
    print("\n[8] Book Allocation Problem")
    pages = [12, 34, 67, 90]
    students = 2
    min_pages = min_pages_per_student(pages, students)
    print(f"    Pages: {pages}, students: {students}")
    print(f"    Minimize maximum pages: {min_pages}")

    # 9. LIS
    print("\n[9] Longest Increasing Subsequence (LIS)")
    nums = [10, 9, 2, 5, 3, 7, 101, 18]
    length, seq = longest_increasing_subsequence(nums)
    print(f"    Array: {nums}")
    print(f"    LIS length: {length}")
    print(f"    LIS example: {seq}")

    # 10. Algorithm Selection Guide
    print("\n[10] Algorithm Selection Guide")
    print("    | Problem Type           | Key Algorithm                |")
    print("    |------------------------|------------------------------|")
    print("    | Subarray sum           | HashMap + prefix sum         |")
    print("    | Interval scheduling    | Sorting + greedy/DP          |")
    print("    | String transformation  | DP (LCS, edit distance)      |")
    print("    | Graph connectivity     | Union-Find, BFS/DFS          |")
    print("    | Range queries          | Segment tree, BIT            |")
    print("    | Combinatorial optim.   | Backtracking + pruning       |")
    print("    | Exhaustive (small n)   | Bitmask DP                   |")
    print("    | Minimize maximum       | Parametric search            |")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
