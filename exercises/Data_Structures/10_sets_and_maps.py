"""
Exercise 10: Sets and Maps

Practice set operations, dict patterns, and hash-based problems.
"""


def intersection_of_lists(lst1, lst2):
    """Find common elements preserving order from lst1.

    >>> intersection_of_lists([1, 2, 3, 4], [3, 4, 5, 6])
    [3, 4]
    >>> intersection_of_lists([1, 2], [3, 4])
    []
    """
    # TODO: Implement this using a set
    pass


def find_missing_number(nums):
    """Find the missing number in [0, n].

    nums contains n distinct numbers from range [0, n].

    >>> find_missing_number([3, 0, 1])
    2
    >>> find_missing_number([0, 1])
    2
    """
    # TODO: Implement this
    pass


def word_frequency(text):
    """Count word frequencies (case-insensitive).

    >>> result = word_frequency("the cat sat on the mat the cat")
    >>> result['the']
    3
    >>> result['cat']
    2
    """
    # TODO: Implement this
    pass


def is_isomorphic(s, t):
    """Check if two strings are isomorphic.

    Two strings are isomorphic if characters in s can be replaced
    to get t (one-to-one mapping).

    >>> is_isomorphic("egg", "add")
    True
    >>> is_isomorphic("foo", "bar")
    False
    >>> is_isomorphic("paper", "title")
    True
    """
    # TODO: Implement this
    pass


def subdomain_visits(domains):
    """Count visits to each subdomain.

    >>> result = subdomain_visits(["9001 discuss.leetcode.com", "50 leetcode.com"])
    >>> result["leetcode.com"]
    9051
    >>> result["discuss.leetcode.com"]
    9001
    >>> result["com"]
    9051
    """
    # TODO: Implement this
    pass


if __name__ == "__main__":
    assert intersection_of_lists([1, 2, 3, 4], [3, 4, 5, 6]) == [3, 4]
    assert intersection_of_lists([1, 2], [3, 4]) == []
    print("intersection_of_lists: PASSED")

    assert find_missing_number([3, 0, 1]) == 2
    assert find_missing_number([0, 1]) == 2
    assert find_missing_number([0]) == 1
    print("find_missing_number: PASSED")

    freq = word_frequency("the cat sat on the mat the cat")
    assert freq['the'] == 3
    assert freq['cat'] == 2
    assert freq['sat'] == 1
    print("word_frequency: PASSED")

    assert is_isomorphic("egg", "add") is True
    assert is_isomorphic("foo", "bar") is False
    assert is_isomorphic("paper", "title") is True
    print("is_isomorphic: PASSED")

    visits = subdomain_visits(["9001 discuss.leetcode.com", "50 leetcode.com"])
    assert visits["leetcode.com"] == 9051
    assert visits["discuss.leetcode.com"] == 9001
    assert visits["com"] == 9051
    print("subdomain_visits: PASSED")

    print("\nAll tests passed!")
