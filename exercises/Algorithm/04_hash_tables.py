"""
Exercises for Lesson 04: Hash Tables
Topic: Algorithm

Solutions to practice problems from the lesson.
The lesson provides recommended LeetCode/BOJ problems. We implement the
key problems directly: Two Sum, Valid Anagram, Group Anagrams,
Longest Consecutive Sequence, and LRU Cache.
"""

from collections import defaultdict, OrderedDict


# === Exercise 1: Two Sum ===
# Problem: Given an array and a target, return indices of two numbers that sum to target.
#   Input: nums = [2, 7, 11, 15], target = 9
#   Output: [0, 1]

def exercise_1():
    """Solution: Hash map for O(n) lookup."""
    def two_sum(nums, target):
        # Map each value to its index. For each number, check if
        # (target - number) already exists in the map.
        seen = {}
        for i, num in enumerate(nums):
            complement = target - num
            if complement in seen:
                return [seen[complement], i]
            seen[num] = i
        return []

    # Test cases
    result = two_sum([2, 7, 11, 15], 9)
    print(f"nums=[2,7,11,15], target=9 -> {result}")
    assert result == [0, 1]

    result = two_sum([3, 2, 4], 6)
    print(f"nums=[3,2,4], target=6 -> {result}")
    assert result == [1, 2]

    result = two_sum([3, 3], 6)
    print(f"nums=[3,3], target=6 -> {result}")
    assert result == [0, 1]

    print("All Two Sum tests passed!")


# === Exercise 2: Valid Anagram ===
# Problem: Determine if two strings are anagrams.
#   Input: s = "anagram", t = "nagaram"
#   Output: True

def exercise_2():
    """Solution: Frequency counting with a dictionary."""
    def is_anagram(s, t):
        if len(s) != len(t):
            return False
        freq = {}
        for c in s:
            freq[c] = freq.get(c, 0) + 1
        for c in t:
            freq[c] = freq.get(c, 0) - 1
            if freq[c] < 0:
                return False
        return True

    tests = [
        ("anagram", "nagaram", True),
        ("rat", "car", False),
        ("listen", "silent", True),
        ("hello", "world", False),
        ("", "", True),
    ]

    for s, t, expected in tests:
        result = is_anagram(s, t)
        print(f'is_anagram("{s}", "{t}") = {result}')
        assert result == expected

    print("All Valid Anagram tests passed!")


# === Exercise 3: Group Anagrams ===
# Problem: Group strings that are anagrams of each other.
#   Input: ["eat", "tea", "tan", "ate", "nat", "bat"]
#   Output: [["eat","tea","ate"], ["tan","nat"], ["bat"]]

def exercise_3():
    """Solution: Sort each word to create a canonical key."""
    def group_anagrams(strs):
        # Words that are anagrams will have the same sorted characters.
        groups = defaultdict(list)
        for s in strs:
            key = tuple(sorted(s))
            groups[key].append(s)
        return list(groups.values())

    strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
    result = group_anagrams(strs)
    print(f"Input: {strs}")
    print(f"Groups: {result}")

    # Verify: sort each group and the list of groups for consistent comparison
    result_sorted = sorted([sorted(g) for g in result])
    expected_sorted = sorted([
        sorted(["eat", "tea", "ate"]),
        sorted(["tan", "nat"]),
        sorted(["bat"]),
    ])
    assert result_sorted == expected_sorted

    print("All Group Anagrams tests passed!")


# === Exercise 4: Longest Consecutive Sequence ===
# Problem: Find the length of the longest consecutive elements sequence in O(n).
#   Input: [100, 4, 200, 1, 3, 2]
#   Output: 4 (sequence [1, 2, 3, 4])

def exercise_4():
    """Solution: HashSet approach - only start counting from sequence starts."""
    def longest_consecutive(nums):
        if not nums:
            return 0

        num_set = set(nums)
        max_len = 0

        for num in num_set:
            # Only start counting if num is the beginning of a sequence
            # (i.e., num-1 is not in the set). This ensures O(n) total work.
            if num - 1 not in num_set:
                current = num
                length = 1
                while current + 1 in num_set:
                    current += 1
                    length += 1
                max_len = max(max_len, length)

        return max_len

    tests = [
        ([100, 4, 200, 1, 3, 2], 4),
        ([0, 3, 7, 2, 5, 8, 4, 6, 0, 1], 9),
        ([], 0),
        ([1], 1),
        ([1, 2, 0, 1], 3),
    ]

    for nums, expected in tests:
        result = longest_consecutive(nums)
        print(f"nums={nums} -> longest={result}")
        assert result == expected

    print("All Longest Consecutive Sequence tests passed!")


# === Exercise 5: LRU Cache ===
# Problem: Implement an LRU (Least Recently Used) cache with get and put in O(1).

def exercise_5():
    """Solution: OrderedDict maintains insertion order with O(1) move_to_end."""
    class LRUCache:
        def __init__(self, capacity):
            self.capacity = capacity
            self.cache = OrderedDict()

        def get(self, key):
            if key not in self.cache:
                return -1
            # Move to end to mark as recently used
            self.cache.move_to_end(key)
            return self.cache[key]

        def put(self, key, value):
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                # Pop the least recently used (first item)
                self.cache.popitem(last=False)

    cache = LRUCache(2)

    cache.put(1, 1)
    cache.put(2, 2)
    print(f"get(1) = {cache.get(1)}")  # 1
    assert cache.get(1) == 1

    cache.put(3, 3)  # evicts key 2
    print(f"get(2) = {cache.get(2)}")  # -1 (evicted)
    assert cache.get(2) == -1

    cache.put(4, 4)  # evicts key 1
    print(f"get(1) = {cache.get(1)}")  # -1 (evicted)
    assert cache.get(1) == -1

    print(f"get(3) = {cache.get(3)}")  # 3
    assert cache.get(3) == 3

    print(f"get(4) = {cache.get(4)}")  # 4
    assert cache.get(4) == 4

    print("All LRU Cache tests passed!")


if __name__ == "__main__":
    print("=== Exercise 1: Two Sum ===")
    exercise_1()
    print("\n=== Exercise 2: Valid Anagram ===")
    exercise_2()
    print("\n=== Exercise 3: Group Anagrams ===")
    exercise_3()
    print("\n=== Exercise 4: Longest Consecutive Sequence ===")
    exercise_4()
    print("\n=== Exercise 5: LRU Cache ===")
    exercise_5()
    print("\nAll exercises completed!")
