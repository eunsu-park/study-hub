"""
Exercises for Lesson 19: Greedy Algorithms
Topic: Algorithm

Solutions to practice problems from the lesson.
"""


# === Exercise 1: Gas Station Circuit ===
# Problem: Find starting point for a circular gas station route.
#   gas[i] = fuel gained at station i
#   cost[i] = fuel needed to travel from station i to i+1
#   Return the starting station index, or -1 if no solution.

def exercise_1():
    """Solution: Greedy - track total and current tank."""
    def can_complete_circuit(gas, cost):
        total_tank = 0
        current_tank = 0
        start = 0

        for i in range(len(gas)):
            diff = gas[i] - cost[i]
            total_tank += diff
            current_tank += diff

            # If current tank goes negative, we cannot start from any station
            # in [start, i]. Reset start to i+1 and reset current tank.
            if current_tank < 0:
                start = i + 1
                current_tank = 0

        # If total gas >= total cost, a solution exists and start is correct.
        return start if total_tank >= 0 else -1

    tests = [
        ([1, 2, 3, 4, 5], [3, 4, 5, 1, 2], 3),
        ([2, 3, 4], [3, 4, 3], -1),
        ([5, 1, 2, 3, 4], [4, 4, 1, 5, 1], 4),
        ([3, 1, 1], [1, 2, 2], 0),
    ]

    for gas, cost, expected in tests:
        result = can_complete_circuit(gas, cost)
        print(f"gas={gas}, cost={cost} -> start={result}")
        assert result == expected

    print("All Gas Station tests passed!")


# === Exercise 2: Activity Selection (Meeting Room) ===
# Problem: Given meetings with start and end times, find the maximum number
#   of non-overlapping meetings.
# Approach: Sort by end time, greedily select meetings that start after
#   the previous one ends.

def exercise_2():
    """Solution: Greedy activity selection in O(n log n)."""
    def max_meetings(intervals):
        if not intervals:
            return 0

        # Sort by end time
        intervals.sort(key=lambda x: x[1])

        count = 1
        last_end = intervals[0][1]

        for i in range(1, len(intervals)):
            if intervals[i][0] >= last_end:
                count += 1
                last_end = intervals[i][1]

        return count

    tests = [
        ([(1, 4), (3, 5), (0, 6), (5, 7), (3, 8), (5, 9), (6, 10), (8, 11), (8, 12), (2, 13), (12, 14)], 4),
        ([(1, 2), (3, 4), (5, 6)], 3),    # no overlap
        ([(1, 10), (2, 3), (4, 5)], 2),    # skip the long one
        ([(1, 2)], 1),
        ([], 0),
    ]

    for intervals, expected in tests:
        result = max_meetings(intervals)
        print(f"max_meetings({intervals}) = {result}")
        assert result == expected

    print("All Activity Selection tests passed!")


# === Exercise 3: Jump Game ===
# Problem: Given an array of non-negative integers where each element represents
#   the maximum jump length from that position, determine if you can reach the last index.
#   Input: [2, 3, 1, 1, 4]
#   Output: True

def exercise_3():
    """Solution: Greedy - track the farthest reachable index."""
    def can_jump(nums):
        farthest = 0
        for i in range(len(nums)):
            if i > farthest:
                return False  # cannot reach position i
            farthest = max(farthest, i + nums[i])
            if farthest >= len(nums) - 1:
                return True
        return True

    tests = [
        ([2, 3, 1, 1, 4], True),
        ([3, 2, 1, 0, 4], False),
        ([0], True),
        ([2, 0, 0], True),
        ([1, 0, 1, 0], False),
    ]

    for nums, expected in tests:
        result = can_jump(nums)
        print(f"can_jump({nums}) = {result}")
        assert result == expected

    print("All Jump Game tests passed!")


# === Exercise 4: Minimum Jump Game (Jump Game II) ===
# Problem: Return the minimum number of jumps to reach the last index.
#   Input: [2, 3, 1, 1, 4]
#   Output: 2 (jump from 0->1, then 1->4)

def exercise_4():
    """Solution: Greedy BFS-like approach."""
    def min_jumps(nums):
        if len(nums) <= 1:
            return 0

        jumps = 0
        current_end = 0   # farthest reachable with current number of jumps
        farthest = 0       # farthest reachable overall

        for i in range(len(nums) - 1):
            farthest = max(farthest, i + nums[i])

            if i == current_end:
                jumps += 1
                current_end = farthest

                if current_end >= len(nums) - 1:
                    break

        return jumps

    tests = [
        ([2, 3, 1, 1, 4], 2),
        ([2, 3, 0, 1, 4], 2),
        ([1, 1, 1, 1], 3),
        ([1], 0),
        ([5, 4, 3, 2, 1], 1),
    ]

    for nums, expected in tests:
        result = min_jumps(nums)
        print(f"min_jumps({nums}) = {result}")
        assert result == expected

    print("All Minimum Jump Game tests passed!")


if __name__ == "__main__":
    print("=== Exercise 1: Gas Station Circuit ===")
    exercise_1()
    print("\n=== Exercise 2: Activity Selection ===")
    exercise_2()
    print("\n=== Exercise 3: Jump Game ===")
    exercise_3()
    print("\n=== Exercise 4: Minimum Jump Game ===")
    exercise_4()
    print("\nAll exercises completed!")
