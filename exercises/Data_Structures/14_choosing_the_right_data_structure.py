"""
Exercise 14: Choosing the Right Data Structure

Practice choosing and combining data structures for real problems.
"""


def most_frequent_k(nums, k):
    """Return the k most frequent elements.

    Choose the optimal data structure(s).

    >>> sorted(most_frequent_k([1,1,1,2,2,3], 2))
    [1, 2]
    >>> most_frequent_k([1], 1)
    [1]
    """
    # TODO: Implement this
    pass


def first_unique_char(s):
    """Return index of first non-repeating character, or -1.

    >>> first_unique_char("leetcode")
    0
    >>> first_unique_char("aabb")
    -1
    """
    # TODO: Implement this
    pass


class MinMaxStack:
    """Stack that supports O(1) push, pop, get_min, and get_max.

    >>> s = MinMaxStack()
    >>> s.push(5); s.push(3); s.push(7)
    >>> s.get_min()
    3
    >>> s.get_max()
    7
    >>> s.pop()
    7
    >>> s.get_max()
    5
    """

    def __init__(self):
        # TODO: Initialize
        pass

    def push(self, val):
        # TODO: Implement this
        pass

    def pop(self):
        # TODO: Implement this
        pass

    def get_min(self):
        # TODO: Implement this
        pass

    def get_max(self):
        # TODO: Implement this
        pass


def task_scheduler(tasks, cooldown):
    """Find minimum intervals to complete all tasks with cooldown.

    Same tasks must have at least `cooldown` intervals between them.

    >>> task_scheduler(["A","A","A","B","B","B"], 2)
    8
    >>> task_scheduler(["A","A","A","B","B","B"], 0)
    6
    """
    # TODO: Implement this
    pass


def longest_substring_no_repeat(s):
    """Find length of longest substring without repeating chars.

    Choose the right data structure for O(n) time.

    >>> longest_substring_no_repeat("abcabcbb")
    3
    >>> longest_substring_no_repeat("bbbbb")
    1
    >>> longest_substring_no_repeat("pwwkew")
    3
    """
    # TODO: Implement this
    pass


if __name__ == "__main__":
    assert sorted(most_frequent_k([1,1,1,2,2,3], 2)) == [1, 2]
    assert most_frequent_k([1], 1) == [1]
    print("most_frequent_k: PASSED")

    assert first_unique_char("leetcode") == 0
    assert first_unique_char("aabb") == -1
    print("first_unique_char: PASSED")

    s = MinMaxStack()
    s.push(5); s.push(3); s.push(7)
    assert s.get_min() == 3
    assert s.get_max() == 7
    assert s.pop() == 7
    assert s.get_max() == 5
    assert s.pop() == 3
    assert s.get_min() == 5
    print("MinMaxStack: PASSED")

    assert task_scheduler(["A","A","A","B","B","B"], 2) == 8
    assert task_scheduler(["A","A","A","B","B","B"], 0) == 6
    print("task_scheduler: PASSED")

    assert longest_substring_no_repeat("abcabcbb") == 3
    assert longest_substring_no_repeat("bbbbb") == 1
    assert longest_substring_no_repeat("pwwkew") == 3
    print("longest_substring_no_repeat: PASSED")

    print("\nAll tests passed!")
