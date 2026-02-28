"""
Exercises for Lesson 18: Dynamic Programming
Topic: Algorithm

Solutions to practice problems from the lesson.
Problems: Climbing Stairs, Coin Change, LCS, 0/1 Knapsack, LIS, Edit Distance.
"""


# === Exercise 1: Climbing Stairs ===
# Problem: You can climb 1 or 2 steps at a time. How many ways to reach step n?
#   Input: n = 5
#   Output: 8

def exercise_1():
    """Solution: dp[i] = dp[i-1] + dp[i-2] (Fibonacci variant)."""
    def climb_stairs(n):
        if n <= 2:
            return n
        dp = [0] * (n + 1)
        dp[1] = 1
        dp[2] = 2
        for i in range(3, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[n]

    tests = [(1, 1), (2, 2), (3, 3), (4, 5), (5, 8), (10, 89)]
    for n, expected in tests:
        result = climb_stairs(n)
        print(f"climb_stairs({n}) = {result}")
        assert result == expected

    print("All Climbing Stairs tests passed!")


# === Exercise 2: Coin Change ===
# Problem: Find the minimum number of coins to make up amount.
#   Input: coins = [1, 5, 10, 25], amount = 30
#   Output: 2 (25 + 5)

def exercise_2():
    """Solution: dp[i] = min coins needed for amount i."""
    def coin_change(coins, amount):
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0

        for i in range(1, amount + 1):
            for coin in coins:
                if coin <= i and dp[i - coin] + 1 < dp[i]:
                    dp[i] = dp[i - coin] + 1

        return dp[amount] if dp[amount] != float('inf') else -1

    tests = [
        ([1, 5, 10, 25], 30, 2),    # 25 + 5
        ([1, 5, 10], 11, 2),         # 10 + 1
        ([2], 3, -1),                # impossible
        ([1], 0, 0),                 # 0 coins needed
        ([1, 3, 4], 6, 2),           # 3 + 3
    ]

    for coins, amount, expected in tests:
        result = coin_change(coins, amount)
        print(f"coin_change({coins}, {amount}) = {result}")
        assert result == expected

    print("All Coin Change tests passed!")


# === Exercise 3: Longest Common Subsequence (LCS) ===
# Problem: Find the length of the longest common subsequence of two strings.
#   Input: "ABCBDAB", "BDCAB"
#   Output: 4 ("BCAB")

def exercise_3():
    """Solution: 2D DP where dp[i][j] = LCS length of s1[:i] and s2[:j]."""
    def lcs(s1, s2):
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    # Also reconstruct the LCS string
    def lcs_with_string(s1, s2):
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        # Backtrack to find the LCS string
        result = []
        i, j = m, n
        while i > 0 and j > 0:
            if s1[i - 1] == s2[j - 1]:
                result.append(s1[i - 1])
                i -= 1
                j -= 1
            elif dp[i - 1][j] > dp[i][j - 1]:
                i -= 1
            else:
                j -= 1

        return dp[m][n], ''.join(reversed(result))

    tests = [
        ("ABCBDAB", "BDCAB", 4),
        ("ABC", "DEF", 0),
        ("ABC", "ABC", 3),
        ("", "ABC", 0),
    ]

    for s1, s2, expected in tests:
        result = lcs(s1, s2)
        print(f'LCS("{s1}", "{s2}") = {result}')
        assert result == expected

    length, string = lcs_with_string("ABCBDAB", "BDCAB")
    print(f'LCS string: "{string}" (length {length})')
    assert length == 4

    print("All LCS tests passed!")


# === Exercise 4: 0/1 Knapsack ===
# Problem: Given items with weights and values, maximize value within weight capacity.
#   Input: weights = [2, 3, 4, 5], values = [3, 4, 5, 6], capacity = 5
#   Output: 7 (items with weight 2 and 3, values 3+4=7)

def exercise_4():
    """Solution: 2D DP, space-optimized to 1D."""
    def knapsack(weights, values, capacity):
        n = len(weights)
        # 1D DP: dp[w] = max value achievable with capacity w
        dp = [0] * (capacity + 1)

        for i in range(n):
            # Traverse backwards to avoid using same item twice
            for w in range(capacity, weights[i] - 1, -1):
                dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

        return dp[capacity]

    tests = [
        ([2, 3, 4, 5], [3, 4, 5, 6], 5, 7),
        ([1, 2, 3], [6, 10, 12], 5, 22),    # items 1+2 (w=3) or 2+3 (w=5, v=22)
        ([10], [100], 5, 0),                  # item too heavy
        ([1, 1, 1], [1, 1, 1], 2, 2),        # take any 2
    ]

    for weights, values, cap, expected in tests:
        result = knapsack(weights, values, cap)
        print(f"knapsack(w={weights}, v={values}, cap={cap}) = {result}")
        assert result == expected

    print("All Knapsack tests passed!")


# === Exercise 5: Longest Increasing Subsequence (LIS) ===
# Problem: Find the length of the longest strictly increasing subsequence.
#   Input: [10, 9, 2, 5, 3, 7, 101, 18]
#   Output: 4 (e.g., [2, 3, 7, 101])
# Approach: O(n log n) using binary search with patience sorting.

def exercise_5():
    """Solution: O(n log n) LIS using binary search."""
    from bisect import bisect_left

    def lis_length(nums):
        if not nums:
            return 0

        # tails[i] = smallest tail element for IS of length i+1
        tails = []

        for num in nums:
            pos = bisect_left(tails, num)
            if pos == len(tails):
                tails.append(num)
            else:
                tails[pos] = num

        return len(tails)

    tests = [
        ([10, 9, 2, 5, 3, 7, 101, 18], 4),
        ([0, 1, 0, 3, 2, 3], 4),
        ([7, 7, 7, 7], 1),
        ([1, 2, 3, 4, 5], 5),
        ([5, 4, 3, 2, 1], 1),
    ]

    for nums, expected in tests:
        result = lis_length(nums)
        print(f"LIS({nums}) = {result}")
        assert result == expected

    print("All LIS tests passed!")


# === Exercise 6: Edit Distance ===
# Problem: Find the minimum number of operations (insert, delete, replace)
#   to convert word1 to word2.
#   Input: "horse", "ros"
#   Output: 3

def exercise_6():
    """Solution: 2D DP for edit distance."""
    def edit_distance(word1, word2):
        m, n = len(word1), len(word2)
        # dp[i][j] = edit distance between word1[:i] and word2[:j]
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Base cases: converting from/to empty string
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],      # delete from word1
                        dp[i][j - 1],      # insert into word1
                        dp[i - 1][j - 1]   # replace
                    )

        return dp[m][n]

    tests = [
        ("horse", "ros", 3),
        ("intention", "execution", 5),
        ("", "abc", 3),
        ("abc", "", 3),
        ("abc", "abc", 0),
    ]

    for w1, w2, expected in tests:
        result = edit_distance(w1, w2)
        print(f'edit_distance("{w1}", "{w2}") = {result}')
        assert result == expected

    print("All Edit Distance tests passed!")


if __name__ == "__main__":
    print("=== Exercise 1: Climbing Stairs ===")
    exercise_1()
    print("\n=== Exercise 2: Coin Change ===")
    exercise_2()
    print("\n=== Exercise 3: Longest Common Subsequence ===")
    exercise_3()
    print("\n=== Exercise 4: 0/1 Knapsack ===")
    exercise_4()
    print("\n=== Exercise 5: Longest Increasing Subsequence ===")
    exercise_5()
    print("\n=== Exercise 6: Edit Distance ===")
    exercise_6()
    print("\nAll exercises completed!")
