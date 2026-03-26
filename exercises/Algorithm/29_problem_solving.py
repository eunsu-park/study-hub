"""
Exercises for Lesson 29: Problem Solving Strategies
Topic: Algorithm

Solutions to comprehensive practice problems from the lesson.
These problems integrate multiple algorithm techniques.
"""

import heapq
from collections import deque


# === Exercise 1: Kth Largest Number in Array ===
# Problem: Find the k-th largest number in an unsorted array.
# Hint from lesson: Use heap or quickselect.

def exercise_1():
    """Solution: Min-heap approach for O(n log k)."""
    def kth_largest(nums, k):
        # Maintain a min-heap of size k
        heap = []
        for num in nums:
            heapq.heappush(heap, num)
            if len(heap) > k:
                heapq.heappop(heap)
        return heap[0]

    tests = [
        ([3, 2, 1, 5, 6, 4], 2, 5),
        ([3, 2, 3, 1, 2, 4, 5, 5, 6], 4, 4),
        ([1, 2, 3, 4, 5], 1, 5),
        ([7, 10, 4, 3, 20, 15], 3, 10),
    ]

    for nums, k, expected in tests:
        result = kth_largest(nums, k)
        print(f"kth_largest({nums}, k={k}) = {result}")
        assert result == expected

    print("All Kth Largest tests passed!")


# === Exercise 2: Maze Shortest Distance (BFS) ===
# Problem: Find the shortest path in a maze from start to end.
#   '0' is passable, '1' is a wall.

def exercise_2():
    """Solution: BFS for shortest path in unweighted grid."""
    def shortest_path(maze, start, end):
        """
        maze: 2D grid ('0' passable, '1' wall)
        start: (row, col) start position
        end: (row, col) end position
        Returns: shortest distance, or -1 if unreachable
        """
        m, n = len(maze), len(maze[0])
        if maze[start[0]][start[1]] == '1' or maze[end[0]][end[1]] == '1':
            return -1

        dx = [0, 0, 1, -1]
        dy = [1, -1, 0, 0]

        dist = [[-1] * n for _ in range(m)]
        dist[start[0]][start[1]] = 0
        queue = deque([start])

        while queue:
            x, y = queue.popleft()

            if (x, y) == end:
                return dist[x][y]

            for d in range(4):
                nx, ny = x + dx[d], y + dy[d]
                if (0 <= nx < m and 0 <= ny < n and
                        maze[nx][ny] == '0' and dist[nx][ny] == -1):
                    dist[nx][ny] = dist[x][y] + 1
                    queue.append((nx, ny))

        return -1

    # Test case 1
    maze = [
        ['0', '0', '1', '0'],
        ['0', '0', '0', '0'],
        ['1', '0', '1', '0'],
        ['0', '0', '0', '0'],
    ]
    result = shortest_path(maze, (0, 0), (3, 3))
    print(f"Maze 1 shortest path: {result}")
    assert result == 6

    # Test case 2: unreachable
    maze2 = [
        ['0', '1'],
        ['1', '0'],
    ]
    result = shortest_path(maze2, (0, 0), (1, 1))
    print(f"Maze 2 (blocked) shortest path: {result}")
    assert result == -1

    # Test case 3: start == end
    result = shortest_path(maze, (0, 0), (0, 0))
    print(f"Maze start==end: {result}")
    assert result == 0

    print("All Maze Shortest Distance tests passed!")


# === Exercise 3: Maximum Consecutive Subarray Sum (Kadane's Algorithm) ===
# Problem: Find the maximum sum of a contiguous subarray.
#   Input: [-2, 1, -3, 4, -1, 2, 1, -5, 4]
#   Output: 6 (subarray [4, -1, 2, 1])

def exercise_3():
    """Solution: Kadane's algorithm in O(n)."""
    def max_subarray_sum(nums):
        if not nums:
            return 0

        max_sum = nums[0]
        current_sum = nums[0]

        for i in range(1, len(nums)):
            # Either extend the current subarray or start a new one
            current_sum = max(nums[i], current_sum + nums[i])
            max_sum = max(max_sum, current_sum)

        return max_sum

    # Also return the subarray itself
    def max_subarray_with_indices(nums):
        if not nums:
            return 0, 0, 0

        max_sum = nums[0]
        current_sum = nums[0]
        start = end = 0
        temp_start = 0

        for i in range(1, len(nums)):
            if nums[i] > current_sum + nums[i]:
                current_sum = nums[i]
                temp_start = i
            else:
                current_sum += nums[i]

            if current_sum > max_sum:
                max_sum = current_sum
                start = temp_start
                end = i

        return max_sum, start, end

    tests = [
        ([-2, 1, -3, 4, -1, 2, 1, -5, 4], 6),
        ([1], 1),
        ([5, 4, -1, 7, 8], 23),
        ([-1, -2, -3], -1),
    ]

    for nums, expected in tests:
        result = max_subarray_sum(nums)
        print(f"max_subarray({nums}) = {result}")
        assert result == expected

    # Show subarray
    nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    max_val, s, e = max_subarray_with_indices(nums)
    print(f"\nSubarray [{s}:{e+1}] = {nums[s:e+1]}, sum = {max_val}")
    assert max_val == 6

    print("All Kadane's Algorithm tests passed!")


# === Exercise 4: Longest Increasing Subsequence (DP + Binary Search) ===
# Problem: Find the length of the longest strictly increasing subsequence.
#   Input: [10, 9, 2, 5, 3, 7, 101, 18]
#   Output: 4

def exercise_4():
    """Solution: O(n log n) LIS using patience sorting."""
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

    # Reconstruct the actual LIS
    def lis_reconstruct(nums):
        if not nums:
            return []

        n = len(nums)
        tails = []
        indices = []       # indices[i] = index of tails[i] in nums
        predecessors = [-1] * n  # for backtracking

        for i, num in enumerate(nums):
            pos = bisect_left(tails, num)
            if pos == len(tails):
                tails.append(num)
                indices.append(i)
            else:
                tails[pos] = num
                indices[pos] = i

            if pos > 0:
                predecessors[i] = indices[pos - 1]

        # Backtrack
        result = []
        idx = indices[-1]
        while idx != -1:
            result.append(nums[idx])
            idx = predecessors[idx]

        return list(reversed(result))

    tests = [
        ([10, 9, 2, 5, 3, 7, 101, 18], 4),
        ([0, 1, 0, 3, 2, 3], 4),
        ([7, 7, 7, 7], 1),
        ([1, 2, 3, 4, 5], 5),
    ]

    for nums, expected in tests:
        result = lis_length(nums)
        print(f"LIS length of {nums} = {result}")
        assert result == expected

    # Reconstruct example
    nums = [10, 9, 2, 5, 3, 7, 101, 18]
    subsequence = lis_reconstruct(nums)
    print(f"One LIS of {nums}: {subsequence} (length {len(subsequence)})")
    assert len(subsequence) == 4
    # Verify it's strictly increasing
    for i in range(len(subsequence) - 1):
        assert subsequence[i] < subsequence[i + 1]

    print("All LIS tests passed!")


# === Exercise 5: Word Transformation (BFS) ===
# Problem: Transform start_word to end_word by changing one letter at a time.
#   Each intermediate word must be in the dictionary.
#   Return the length of the shortest transformation sequence.

def exercise_5():
    """Solution: BFS on the word graph."""
    def word_ladder(begin_word, end_word, word_list):
        word_set = set(word_list)
        if end_word not in word_set:
            return 0

        queue = deque([(begin_word, 1)])
        visited = {begin_word}

        while queue:
            word, length = queue.popleft()

            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    if c == word[i]:
                        continue
                    new_word = word[:i] + c + word[i + 1:]

                    if new_word == end_word:
                        return length + 1

                    if new_word in word_set and new_word not in visited:
                        visited.add(new_word)
                        queue.append((new_word, length + 1))

        return 0

    # Test case 1
    result = word_ladder("hit", "cog",
                         ["hot", "dot", "dog", "lot", "log", "cog"])
    print(f"hit -> cog: {result} steps")
    assert result == 5  # hit -> hot -> dot -> dog -> cog

    # Test case 2: impossible
    result = word_ladder("hit", "cog",
                         ["hot", "dot", "dog", "lot", "log"])
    print(f"hit -> cog (no cog): {result} steps")
    assert result == 0

    # Test case 3
    result = word_ladder("a", "c", ["a", "b", "c"])
    print(f"a -> c: {result} steps")
    assert result == 2

    print("All Word Transformation tests passed!")


if __name__ == "__main__":
    print("=== Exercise 1: Kth Largest Number ===")
    exercise_1()
    print("\n=== Exercise 2: Maze Shortest Distance ===")
    exercise_2()
    print("\n=== Exercise 3: Maximum Subarray Sum ===")
    exercise_3()
    print("\n=== Exercise 4: Longest Increasing Subsequence ===")
    exercise_4()
    print("\n=== Exercise 5: Word Transformation ===")
    exercise_5()
    print("\nAll exercises completed!")
