"""
Dynamic Programming (DP) Basics
Basic Dynamic Programming

An algorithmic technique that solves complex problems by breaking them into smaller subproblems.
"""

from typing import List
from functools import lru_cache


# =============================================================================
# 1. Fibonacci Sequence (Three Methods)
# =============================================================================

# Method 1: Recursion (inefficient - O(2^n))
def fibonacci_recursive(n: int) -> int:
    """Recursion: exponential time complexity"""
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)


# Method 2: Memoization (Top-down DP)
def fibonacci_memo(n: int, memo: dict = None) -> int:
    """Memoization: O(n)"""
    if memo is None:
        memo = {}
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo)
    return memo[n]


# Method 2b: Using lru_cache decorator
@lru_cache(maxsize=None)
def fibonacci_cached(n: int) -> int:
    """Using lru_cache: O(n)"""
    if n <= 1:
        return n
    return fibonacci_cached(n - 1) + fibonacci_cached(n - 2)


# Method 3: Tabulation (Bottom-up DP)
def fibonacci_tabulation(n: int) -> int:
    """Tabulation: O(n) time, O(n) space"""
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]


# Method 4: Space Optimization
def fibonacci_optimized(n: int) -> int:
    """Space optimized: O(n) time, O(1) space"""
    if n <= 1:
        return n
    prev2, prev1 = 0, 1
    for _ in range(2, n + 1):
        curr = prev1 + prev2
        prev2, prev1 = prev1, curr
    return prev1


# =============================================================================
# 2. Climbing Stairs
# =============================================================================
def climb_stairs(n: int) -> int:
    """
    Number of ways to climb n stairs
    when you can take 1 or 2 steps at a time
    """
    if n <= 2:
        return n
    prev2, prev1 = 1, 2
    for _ in range(3, n + 1):
        curr = prev1 + prev2
        prev2, prev1 = prev1, curr
    return prev1


# =============================================================================
# 3. Coin Change
# =============================================================================
def coin_change(coins: List[int], amount: int) -> int:
    """
    Minimum number of coins to make the given amount
    Returns -1 if impossible
    """
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i and dp[i - coin] != float('inf'):
                dp[i] = min(dp[i], dp[i - coin] + 1)

    return dp[amount] if dp[amount] != float('inf') else -1


def coin_change_ways(coins: List[int], amount: int) -> int:
    """
    Number of ways to make the given amount using the given coins
    """
    dp = [0] * (amount + 1)
    dp[0] = 1

    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]

    return dp[amount]


# =============================================================================
# 4. 0/1 Knapsack Problem
# =============================================================================
def knapsack_01(weights: List[int], values: List[int], capacity: int) -> int:
    """
    0/1 Knapsack: each item is either included or excluded
    Returns maximum value
    """
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # Case: don't include current item
            dp[i][w] = dp[i - 1][w]
            # Case: include current item if it fits
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])

    return dp[n][capacity]


def knapsack_01_optimized(weights: List[int], values: List[int], capacity: int) -> int:
    """0/1 Knapsack (space optimized - 1D array)"""
    dp = [0] * (capacity + 1)

    for i in range(len(weights)):
        # Iterate in reverse to prevent reusing the same item
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

    return dp[capacity]


# =============================================================================
# 5. Longest Common Subsequence (LCS)
# =============================================================================
def longest_common_subsequence(text1: str, text2: str) -> int:
    """
    Length of the longest common subsequence of two strings
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def get_lcs_string(text1: str, text2: str) -> str:
    """Reconstruct the LCS string"""
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if text1[i - 1] == text2[j - 1]:
            lcs.append(text1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return ''.join(reversed(lcs))


# =============================================================================
# 6. Longest Increasing Subsequence (LIS)
# =============================================================================
def longest_increasing_subsequence(nums: List[int]) -> int:
    """
    Length of the longest increasing subsequence (O(n^2))
    """
    if not nums:
        return 0

    n = len(nums)
    dp = [1] * n  # dp[i] = LIS length ending at nums[i]

    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)


def lis_binary_search(nums: List[int]) -> int:
    """
    Length of the longest increasing subsequence (O(n log n))
    Using binary search
    """
    from bisect import bisect_left

    if not nums:
        return 0

    tails = []  # tails[i] = smallest tail element for LIS of length i+1

    for num in nums:
        pos = bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num

    return len(tails)


# =============================================================================
# 7. Maximum Subarray Sum (Kadane's Algorithm)
# =============================================================================
def max_subarray_sum(nums: List[int]) -> int:
    """
    Maximum sum of a contiguous subarray (Kadane's Algorithm)
    O(n) time, O(1) space
    """
    if not nums:
        return 0

    max_sum = curr_sum = nums[0]

    for num in nums[1:]:
        curr_sum = max(num, curr_sum + num)
        max_sum = max(max_sum, curr_sum)

    return max_sum


# =============================================================================
# 8. House Robber
# =============================================================================
def rob(nums: List[int]) -> int:
    """
    Maximum amount when adjacent houses cannot be robbed
    """
    if not nums:
        return 0
    if len(nums) <= 2:
        return max(nums)

    prev2, prev1 = nums[0], max(nums[0], nums[1])

    for i in range(2, len(nums)):
        curr = max(prev1, prev2 + nums[i])
        prev2, prev1 = prev1, curr

    return prev1


# =============================================================================
# Tests
# =============================================================================
def main():
    print("=" * 60)
    print("Dynamic Programming (DP) Basic Examples")
    print("=" * 60)

    # 1. Fibonacci
    print("\n[1] Fibonacci Sequence")
    n = 10
    print(f"    fib({n}) = {fibonacci_tabulation(n)}")
    print(f"    First 10: {[fibonacci_optimized(i) for i in range(10)]}")

    # 2. Climbing Stairs
    print("\n[2] Climbing Stairs")
    for n in [2, 3, 5]:
        ways = climb_stairs(n)
        print(f"    {n} stairs: {ways} ways")

    # 3. Coin Change
    print("\n[3] Coin Change")
    coins = [1, 2, 5]
    amount = 11
    min_coins = coin_change(coins, amount)
    ways = coin_change_ways(coins, amount)
    print(f"    Coins: {coins}, Amount: {amount}")
    print(f"    Minimum coins: {min_coins}")
    print(f"    Number of ways: {ways}")

    # 4. 0/1 Knapsack
    print("\n[4] 0/1 Knapsack Problem")
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    capacity = 5
    max_value = knapsack_01(weights, values, capacity)
    print(f"    Weights: {weights}, Values: {values}")
    print(f"    Capacity: {capacity}, Max value: {max_value}")

    # 5. LCS
    print("\n[5] Longest Common Subsequence (LCS)")
    text1, text2 = "ABCDGH", "AEDFHR"
    length = longest_common_subsequence(text1, text2)
    lcs_str = get_lcs_string(text1, text2)
    print(f"    String 1: {text1}")
    print(f"    String 2: {text2}")
    print(f"    LCS length: {length}, LCS: '{lcs_str}'")

    # 6. LIS
    print("\n[6] Longest Increasing Subsequence (LIS)")
    nums = [10, 9, 2, 5, 3, 7, 101, 18]
    length = longest_increasing_subsequence(nums)
    length_fast = lis_binary_search(nums)
    print(f"    Array: {nums}")
    print(f"    LIS length: {length} (O(n^2)), {length_fast} (O(n log n))")

    # 7. Maximum Subarray Sum
    print("\n[7] Maximum Subarray Sum (Kadane)")
    nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    max_sum = max_subarray_sum(nums)
    print(f"    Array: {nums}")
    print(f"    Maximum sum: {max_sum}")

    # 8. House Robber
    print("\n[8] House Robber")
    houses = [2, 7, 9, 3, 1]
    max_money = rob(houses)
    print(f"    House values: {houses}")
    print(f"    Maximum amount: {max_money}")

    print("\n" + "=" * 60)
    print("DP Approach Comparison")
    print("=" * 60)
    print("""
    | Approach       | Direction | Implementation | Advantage                 |
    |----------------|-----------|----------------|---------------------------|
    | Memoization    | Top-down  | Recursion+Cache| Only computes needed parts|
    | Tabulation     | Bottom-up | Loop + Array   | No stack overflow         |

    DP Problem-Solving Steps:
    1. Define state: what dp[i] represents
    2. Derive recurrence: express dp[i] in terms of previous states
    3. Set base cases: initial values
    4. Determine computation order: based on dependencies
    5. Extract answer: from the dp table
    """)


if __name__ == "__main__":
    main()
