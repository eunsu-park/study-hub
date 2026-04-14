"""
Exercise 11: Strings as Data Structures

Practice trie implementation, pattern matching, and string problems.
"""


class Trie:
    """Implement a trie with insert, search, and starts_with.

    >>> t = Trie()
    >>> t.insert("apple")
    >>> t.search("apple")
    True
    >>> t.search("app")
    False
    >>> t.starts_with("app")
    True
    """

    def __init__(self):
        # TODO: Initialize
        pass

    def insert(self, word):
        # TODO: Implement this
        pass

    def search(self, word):
        # TODO: Implement this
        pass

    def starts_with(self, prefix):
        # TODO: Implement this
        pass


def longest_common_prefix(words):
    """Find the longest common prefix among a list of strings.

    >>> longest_common_prefix(["flower", "flow", "flight"])
    'fl'
    >>> longest_common_prefix(["dog", "racecar", "car"])
    ''
    >>> longest_common_prefix([])
    ''
    """
    # TODO: Implement this
    pass


def is_anagram(s1, s2):
    """Check if two strings are anagrams.

    >>> is_anagram("listen", "silent")
    True
    >>> is_anagram("hello", "world")
    False
    """
    # TODO: Implement this
    pass


def pattern_match_count(text, pattern):
    """Count non-overlapping occurrences of pattern in text.

    >>> pattern_match_count("ababababab", "aba")
    2
    >>> pattern_match_count("hello world", "xyz")
    0
    """
    # TODO: Implement this
    pass


def word_break(s, word_dict):
    """Check if string can be segmented into dictionary words.

    >>> word_break("leetcode", ["leet", "code"])
    True
    >>> word_break("applepenapple", ["apple", "pen"])
    True
    >>> word_break("catsandog", ["cats", "dog", "sand", "and", "cat"])
    False
    """
    # TODO: Implement this (use dynamic programming)
    pass


if __name__ == "__main__":
    t = Trie()
    t.insert("apple"); t.insert("app"); t.insert("banana")
    assert t.search("apple") is True
    assert t.search("app") is True
    assert t.search("ap") is False
    assert t.starts_with("ap") is True
    assert t.starts_with("ban") is True
    assert t.starts_with("cat") is False
    print("Trie: PASSED")

    assert longest_common_prefix(["flower", "flow", "flight"]) == "fl"
    assert longest_common_prefix(["dog", "racecar", "car"]) == ""
    assert longest_common_prefix([]) == ""
    print("longest_common_prefix: PASSED")

    assert is_anagram("listen", "silent") is True
    assert is_anagram("hello", "world") is False
    print("is_anagram: PASSED")

    assert pattern_match_count("ababababab", "aba") == 2
    assert pattern_match_count("hello world", "xyz") == 0
    print("pattern_match_count: PASSED")

    assert word_break("leetcode", ["leet", "code"]) is True
    assert word_break("applepenapple", ["apple", "pen"]) is True
    assert word_break("catsandog", ["cats", "dog", "sand", "and", "cat"]) is False
    print("word_break: PASSED")

    print("\nAll tests passed!")
