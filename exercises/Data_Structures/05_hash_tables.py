"""
Exercise 05: Hash Tables

Practice hash table implementation and hash-based problems.
"""


def two_sum(nums, target):
    """Find two indices whose values sum to target.

    Use a hash map for O(n) time.

    >>> two_sum([2, 7, 11, 15], 9)
    [0, 1]
    >>> two_sum([3, 2, 4], 6)
    [1, 2]
    """
    # TODO: Implement this
    pass


def group_anagrams(words):
    """Group words that are anagrams of each other.

    >>> sorted([sorted(g) for g in group_anagrams(["eat","tea","tan","ate","nat","bat"])])
    [['ate', 'eat', 'tea'], ['bat'], ['nat', 'tan']]
    """
    # TODO: Implement this
    pass


def first_non_repeating(s):
    """Find the index of the first non-repeating character.

    >>> first_non_repeating("leetcode")
    0
    >>> first_non_repeating("aabb")
    -1
    """
    # TODO: Implement this
    pass


def longest_consecutive(nums):
    """Find the length of the longest consecutive sequence.

    Must run in O(n) time using a set.

    >>> longest_consecutive([100, 4, 200, 1, 3, 2])
    4
    >>> longest_consecutive([0, 3, 7, 2, 5, 8, 4, 6, 0, 1])
    9
    """
    # TODO: Implement this
    pass


class HashSet:
    """Implement a simple hash set using chaining.

    >>> hs = HashSet()
    >>> hs.add(1); hs.add(2); hs.add(1)
    >>> len(hs)
    2
    >>> 1 in hs
    True
    >>> hs.remove(1)
    >>> 1 in hs
    False
    """

    def __init__(self, capacity=8):
        # TODO: Initialize
        pass

    def add(self, item):
        # TODO: Implement this
        pass

    def remove(self, item):
        # TODO: Implement this
        pass

    def __contains__(self, item):
        # TODO: Implement this
        pass

    def __len__(self):
        # TODO: Implement this
        pass


if __name__ == "__main__":
    assert two_sum([2, 7, 11, 15], 9) == [0, 1]
    assert two_sum([3, 2, 4], 6) == [1, 2]
    print("two_sum: PASSED")

    result = group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
    result_sorted = sorted([sorted(g) for g in result])
    assert result_sorted == [['ate', 'eat', 'tea'], ['bat'], ['nat', 'tan']]
    print("group_anagrams: PASSED")

    assert first_non_repeating("leetcode") == 0
    assert first_non_repeating("aabb") == -1
    print("first_non_repeating: PASSED")

    assert longest_consecutive([100, 4, 200, 1, 3, 2]) == 4
    assert longest_consecutive([0, 3, 7, 2, 5, 8, 4, 6, 0, 1]) == 9
    print("longest_consecutive: PASSED")

    hs = HashSet()
    hs.add(1); hs.add(2); hs.add(1)
    assert len(hs) == 2
    assert 1 in hs
    hs.remove(1)
    assert 1 not in hs
    print("HashSet: PASSED")

    print("\nAll tests passed!")
