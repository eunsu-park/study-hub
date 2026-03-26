"""
Bitmask DP (Bitmask Dynamic Programming)
Bitmask DP

A DP technique that uses bit operations to represent set states.
"""

from typing import List, Tuple
from functools import lru_cache


# =============================================================================
# 1. Bit Operation Basics
# =============================================================================

def bit_operations_demo():
    """Basic bit operations"""
    n = 5  # Set size

    # Empty set
    empty = 0

    # Full set {0, 1, 2, 3, 4}
    full = (1 << n) - 1  # 11111 (binary)

    # Add element i
    def add(mask: int, i: int) -> int:
        return mask | (1 << i)

    # Remove element i
    def remove(mask: int, i: int) -> int:
        return mask & ~(1 << i)

    # Toggle element i
    def toggle(mask: int, i: int) -> int:
        return mask ^ (1 << i)

    # Check if element i is included
    def contains(mask: int, i: int) -> bool:
        return bool(mask & (1 << i))

    # Count elements
    def count(mask: int) -> int:
        return bin(mask).count('1')

    # Lowest bit (smallest element)
    def lowest_bit(mask: int) -> int:
        return mask & (-mask)

    # Iterate over subsets
    def subsets(mask: int):
        """Iterate over all subsets of mask"""
        subset = mask
        while True:
            yield subset
            if subset == 0:
                break
            subset = (subset - 1) & mask

    return {
        'empty': empty,
        'full': full,
        'add': add,
        'remove': remove,
        'toggle': toggle,
        'contains': contains,
        'count': count,
        'lowest_bit': lowest_bit,
        'subsets': subsets
    }


# =============================================================================
# 2. Traveling Salesman Problem (TSP)
# =============================================================================

def tsp(dist: List[List[int]]) -> int:
    """
    Traveling Salesman Problem (TSP)
    Minimum cost to visit all cities and return to the starting point

    Time Complexity: O(n^2 * 2^n)
    Space Complexity: O(n * 2^n)
    """
    n = len(dist)
    INF = float('inf')

    # dp[mask][i] = min cost when visited cities are in mask and currently at city i
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1][0] = 0  # Start at city 0

    for mask in range(1 << n):
        for last in range(n):
            if dp[mask][last] == INF:
                continue
            if not (mask & (1 << last)):
                continue

            for next_city in range(n):
                if mask & (1 << next_city):
                    continue

                new_mask = mask | (1 << next_city)
                dp[new_mask][next_city] = min(
                    dp[new_mask][next_city],
                    dp[mask][last] + dist[last][next_city]
                )

    # Return to start after visiting all cities
    full_mask = (1 << n) - 1
    result = min(dp[full_mask][i] + dist[i][0] for i in range(n))

    return result if result != INF else -1


def tsp_path(dist: List[List[int]]) -> Tuple[int, List[int]]:
    """TSP returning minimum cost and path"""
    n = len(dist)
    INF = float('inf')

    dp = [[INF] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]

    dp[1][0] = 0

    for mask in range(1 << n):
        for last in range(n):
            if dp[mask][last] == INF:
                continue

            for next_city in range(n):
                if mask & (1 << next_city):
                    continue

                new_mask = mask | (1 << next_city)
                new_cost = dp[mask][last] + dist[last][next_city]

                if new_cost < dp[new_mask][next_city]:
                    dp[new_mask][next_city] = new_cost
                    parent[new_mask][next_city] = last

    full_mask = (1 << n) - 1
    min_cost = INF
    last_city = -1

    for i in range(n):
        cost = dp[full_mask][i] + dist[i][0]
        if cost < min_cost:
            min_cost = cost
            last_city = i

    # Reconstruct path
    path = []
    mask = full_mask
    city = last_city

    while city != -1:
        path.append(city)
        prev_city = parent[mask][city]
        mask ^= (1 << city)
        city = prev_city

    path.reverse()
    path.append(0)  # Return to start

    return min_cost, path


# =============================================================================
# 3. Set Partition Problem
# =============================================================================

def can_partition_k_subsets(nums: List[int], k: int) -> bool:
    """
    Whether the array can be partitioned into k subsets with equal sums
    Time Complexity: O(n * 2^n)
    """
    total = sum(nums)
    if total % k != 0:
        return False

    target = total // k
    n = len(nums)

    # dp[mask] = current bucket sum (mod target) when using elements in mask
    dp = [-1] * (1 << n)
    dp[0] = 0

    for mask in range(1 << n):
        if dp[mask] == -1:
            continue

        for i in range(n):
            if mask & (1 << i):
                continue

            if dp[mask] + nums[i] <= target:
                new_mask = mask | (1 << i)
                dp[new_mask] = (dp[mask] + nums[i]) % target

    return dp[(1 << n) - 1] == 0


# =============================================================================
# 4. Minimum Cost Assignment Problem
# =============================================================================

def min_cost_assignment(cost: List[List[int]]) -> int:
    """
    Minimum cost to assign n jobs to n people (one-to-one)
    cost[i][j] = cost for person i to perform job j

    Time Complexity: O(n * 2^n)
    """
    n = len(cost)

    @lru_cache(maxsize=None)
    def dp(mask: int) -> int:
        person = bin(mask).count('1')

        if person == n:
            return 0

        min_cost = float('inf')
        for job in range(n):
            if mask & (1 << job):
                continue

            min_cost = min(min_cost, cost[person][job] + dp(mask | (1 << job)))

        return min_cost

    return dp(0)


# =============================================================================
# 5. Hamiltonian Path
# =============================================================================

def hamiltonian_path_count(adj: List[List[int]]) -> int:
    """
    Count the number of Hamiltonian paths (paths visiting every vertex exactly once)
    adj: adjacency matrix (adj[i][j] = 1 if edge i->j exists)

    Time Complexity: O(n^2 * 2^n)
    """
    n = len(adj)

    # dp[mask][i] = number of paths visiting vertices in mask, ending at i
    dp = [[0] * n for _ in range(1 << n)]

    # Initialize: start from each vertex
    for i in range(n):
        dp[1 << i][i] = 1

    for mask in range(1 << n):
        for last in range(n):
            if dp[mask][last] == 0:
                continue
            if not (mask & (1 << last)):
                continue

            for next_v in range(n):
                if mask & (1 << next_v):
                    continue
                if not adj[last][next_v]:
                    continue

                new_mask = mask | (1 << next_v)
                dp[new_mask][next_v] += dp[mask][last]

    # Sum of paths visiting all vertices
    full_mask = (1 << n) - 1
    return sum(dp[full_mask])


# =============================================================================
# 6. SOS DP (Sum over Subsets)
# =============================================================================

def sos_dp(arr: List[int]) -> List[int]:
    """
    Sum over Subsets DP
    Compute the sum of values for all subsets of each mask

    result[mask] = sum(arr[subset]) for all subset of mask

    Time Complexity: O(n * 2^n)
    """
    n = len(arr).bit_length()
    dp = arr.copy()

    # Extend to cover 0~(len(arr)-1)
    while len(dp) < (1 << n):
        dp.append(0)

    for i in range(n):
        for mask in range(1 << n):
            if mask & (1 << i):
                dp[mask] += dp[mask ^ (1 << i)]

    return dp


# =============================================================================
# 7. Maximum Independent Set (Bitmask Brute Force)
# =============================================================================

def max_independent_set(adj: List[List[int]]) -> int:
    """
    Maximum independent set size in a graph (set of non-adjacent vertices)
    Bitmask brute force for small graphs

    Time Complexity: O(2^n * n^2)
    """
    n = len(adj)
    max_size = 0

    for mask in range(1 << n):
        # Check if mask is an independent set
        valid = True
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            for j in range(i + 1, n):
                if not (mask & (1 << j)):
                    continue
                if adj[i][j]:
                    valid = False
                    break
            if not valid:
                break

        if valid:
            max_size = max(max_size, bin(mask).count('1'))

    return max_size


# =============================================================================
# 8. Domino Tiling (Broken Profile DP)
# =============================================================================

def domino_tiling(m: int, n: int) -> int:
    """
    Number of ways to tile an m x n grid with 1x2 dominoes
    Bitmask DP (profile method)

    Time Complexity: O(n * 2^m * 2^m)
    """
    if m > n:
        m, n = n, m

    # dp[col][profile] = number of ways to fill up to current column with given profile
    dp = {0: 1}

    for col in range(n):
        for row in range(m):
            new_dp = {}

            for profile, count in dp.items():
                # Current cell is already filled
                if profile & (1 << row):
                    new_profile = profile ^ (1 << row)
                    new_dp[new_profile] = new_dp.get(new_profile, 0) + count
                else:
                    # Horizontal domino (extends to next column)
                    new_profile = profile | (1 << row)
                    new_dp[new_profile] = new_dp.get(new_profile, 0) + count

                    # Vertical domino (together with cell below)
                    if row + 1 < m and not (profile & (1 << (row + 1))):
                        new_dp[profile] = new_dp.get(profile, 0) + count

            dp = new_dp

    return dp.get(0, 0)


# =============================================================================
# Tests
# =============================================================================

def main():
    print("=" * 60)
    print("Bitmask DP Examples")
    print("=" * 60)

    # 1. Bit Operation Basics
    print("\n[1] Bit Operation Basics")
    ops = bit_operations_demo()
    mask = 0b10110  # {1, 2, 4}
    print(f"    mask = {bin(mask)} ({mask})")
    print(f"    Element count: {ops['count'](mask)}")
    print(f"    Contains 3: {ops['contains'](mask, 3)}")
    print(f"    Contains 2: {ops['contains'](mask, 2)}")
    print(f"    Add 3: {bin(ops['add'](mask, 3))}")
    print(f"    Subsets: ", end="")
    for s in ops['subsets'](mask):
        print(f"{bin(s)} ", end="")
    print()

    # 2. TSP
    print("\n[2] Traveling Salesman Problem (TSP)")
    dist = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    min_cost, path = tsp_path(dist)
    print(f"    Distance matrix: 4x4")
    print(f"    Minimum cost: {min_cost}")
    print(f"    Path: {path}")

    # 3. Set Partition
    print("\n[3] K Subset Partition")
    nums = [4, 3, 2, 3, 5, 2, 1]
    k = 4
    result = can_partition_k_subsets(nums, k)
    print(f"    Array: {nums}, k={k}")
    print(f"    Partitionable: {result}")

    # 4. Assignment Problem
    print("\n[4] Minimum Cost Assignment")
    cost = [
        [9, 2, 7, 8],
        [6, 4, 3, 7],
        [5, 8, 1, 8],
        [7, 6, 9, 4]
    ]
    min_assign = min_cost_assignment(cost)
    print(f"    Cost matrix: 4x4")
    print(f"    Minimum cost: {min_assign}")

    # 5. Hamiltonian Path
    print("\n[5] Hamiltonian Path Count")
    adj = [
        [0, 1, 1, 1],
        [1, 0, 1, 0],
        [1, 1, 0, 1],
        [1, 0, 1, 0]
    ]
    count = hamiltonian_path_count(adj)
    print(f"    Adjacency matrix: 4x4")
    print(f"    Hamiltonian path count: {count}")

    # 6. Maximum Independent Set
    print("\n[6] Maximum Independent Set")
    adj2 = [
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ]
    mis = max_independent_set(adj2)
    print(f"    Graph: 4-cycle")
    print(f"    Maximum independent set size: {mis}")

    # 7. Domino Tiling
    print("\n[7] Domino Tiling")
    for m, n in [(2, 3), (2, 4), (3, 4)]:
        count = domino_tiling(m, n)
        print(f"    {m}x{n} grid: {count} ways")

    # 8. SOS DP
    print("\n[8] SOS DP (Sum over Subsets)")
    arr = [1, 2, 4, 8]  # Each element is the value of that bit
    result = sos_dp(arr)
    print(f"    Array: {arr}")
    print(f"    result[0b0111] = result[7] = {result[7]}")
    print(f"    (Subsets: {{0}},{{1}},{{0,1}},{{2}},{{0,2}},{{1,2}},{{0,1,2}} sum)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
