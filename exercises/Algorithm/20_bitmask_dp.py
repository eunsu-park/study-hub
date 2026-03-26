"""
Exercises for Lesson 20: Bitmask DP
Topic: Algorithm

Solutions to practice problems from the lesson.
Problems: TSP, Bitmask DP, Partition Equal Subset Sum, Can I Win.
"""


# === Exercise 1: Traveling Salesman Problem (TSP) ===
# Problem: Find the minimum cost to visit all cities exactly once and return to start.
# Approach: dp[mask][i] = minimum cost to visit cities in mask, ending at city i.
#   mask is a bitmask where bit j means city j has been visited.

def exercise_1():
    """Solution: Bitmask DP for TSP in O(n^2 * 2^n)."""
    def tsp(dist):
        """
        dist: n x n distance matrix
        Returns: minimum tour cost starting and ending at city 0
        """
        n = len(dist)
        INF = float('inf')

        # dp[mask][i] = min cost to reach city i having visited cities in mask
        dp = [[INF] * n for _ in range(1 << n)]
        dp[1][0] = 0  # Start at city 0, only city 0 visited

        for mask in range(1 << n):
            for u in range(n):
                if dp[mask][u] == INF:
                    continue
                if not (mask & (1 << u)):
                    continue  # u must be visited in mask

                for v in range(n):
                    if mask & (1 << v):
                        continue  # v already visited
                    new_mask = mask | (1 << v)
                    cost = dp[mask][u] + dist[u][v]
                    if cost < dp[new_mask][v]:
                        dp[new_mask][v] = cost

        # All cities visited: mask = (1 << n) - 1
        full_mask = (1 << n) - 1
        result = INF
        for u in range(n):
            if dp[full_mask][u] + dist[u][0] < result:
                result = dp[full_mask][u] + dist[u][0]

        return result

    # Test case 1: 4 cities
    dist = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0],
    ]
    result = tsp(dist)
    print(f"TSP cost (4 cities): {result}")
    assert result == 80  # 0->1->3->2->0 = 10+25+30+15 = 80

    # Test case 2: 3 cities
    dist2 = [
        [0, 1, 15],
        [2, 0, 7],
        [9, 8, 0],
    ]
    result2 = tsp(dist2)
    print(f"TSP cost (3 cities): {result2}")
    # Possible tours: 0->1->2->0 = 1+7+9=17, 0->2->1->0 = 15+8+2=25
    assert result2 == 17

    print("All TSP tests passed!")


# === Exercise 2: Partition Equal Subset Sum ===
# Problem: Determine if an array can be partitioned into two subsets with equal sum.
#   Input: [1, 5, 11, 5]
#   Output: True (subsets [1, 5, 5] and [11])
# Approach: This is equivalent to finding a subset with sum = total/2.
#   Use bitmask or standard subset sum DP.

def exercise_2():
    """Solution: Subset sum DP for partition problem."""
    def can_partition(nums):
        total = sum(nums)
        if total % 2 != 0:
            return False

        target = total // 2
        # dp[i] = True if we can make sum i from some subset
        dp = [False] * (target + 1)
        dp[0] = True

        for num in nums:
            # Traverse backwards to avoid counting same element twice
            for j in range(target, num - 1, -1):
                if dp[j - num]:
                    dp[j] = True

        return dp[target]

    tests = [
        ([1, 5, 11, 5], True),    # [1,5,5] and [11]
        ([1, 2, 3, 5], False),
        ([1, 1], True),
        ([1, 2, 3, 4, 5, 6, 7], True),  # sum=28, target=14: [1,6,7] or [2,5,7] etc.
        ([100], False),
    ]

    for nums, expected in tests:
        result = can_partition(nums)
        print(f"can_partition({nums}) = {result}")
        assert result == expected

    print("All Partition Equal Subset Sum tests passed!")


# === Exercise 3: Can I Win (Game Theory + Bitmask) ===
# Problem: Two players take turns choosing numbers 1..maxChoosable (each used once).
#   The first player to reach or exceed desiredTotal wins.
#   Determine if the first player can always win.

def exercise_3():
    """Solution: Bitmask DP with memoization for game theory."""
    def can_i_win(max_choosable, desired_total):
        # If sum of all numbers < desired_total, nobody can win
        if max_choosable * (max_choosable + 1) // 2 < desired_total:
            return False

        if desired_total <= 0:
            return True

        memo = {}

        def can_win(mask, remaining):
            if mask in memo:
                return memo[mask]

            for i in range(1, max_choosable + 1):
                bit = 1 << i
                if mask & bit:
                    continue  # number i already used

                # If choosing i wins immediately, or the opponent cannot win
                # after we choose i
                if i >= remaining or not can_win(mask | bit, remaining - i):
                    memo[mask] = True
                    return True

            memo[mask] = False
            return False

        return can_win(0, desired_total)

    tests = [
        (10, 11, False),
        (10, 0, True),
        (10, 1, True),
        (4, 6, True),     # Player 1 picks 4, then any move player 2 makes, player 1 can win
        (5, 50, False),   # 1+2+3+4+5=15 < 50
    ]

    for mc, dt, expected in tests:
        result = can_i_win(mc, dt)
        print(f"can_i_win(max={mc}, total={dt}) = {result}")
        assert result == expected

    print("All Can I Win tests passed!")


if __name__ == "__main__":
    print("=== Exercise 1: Traveling Salesman Problem ===")
    exercise_1()
    print("\n=== Exercise 2: Partition Equal Subset Sum ===")
    exercise_2()
    print("\n=== Exercise 3: Can I Win ===")
    exercise_3()
    print("\nAll exercises completed!")
