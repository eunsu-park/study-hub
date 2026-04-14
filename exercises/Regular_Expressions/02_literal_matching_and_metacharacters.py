"""
Exercise 02: Literal Matching and Metacharacters

Practice literal matching, dot, anchors, alternation, and escaping.
"""

import re


def count_substring(text, substring):
    """Count occurrences of a literal substring using regex.

    Must handle metacharacters in the substring properly.

    Args:
        text: Input string.
        substring: Literal string to search for.

    Returns:
        Integer count of occurrences.
    """
    # TODO: Implement this (hint: use re.escape)
    pass


def match_three_letter_words(text):
    """Find all three-letter words that start and end with the same letter.

    E.g., "aba", "cdc", "pop" but not "abc" or "ab".

    Args:
        text: Input string.

    Returns:
        List of matching words.
    """
    # TODO: Implement this
    pass


def starts_and_ends_with(text, start, end):
    """Check if text starts with 'start' and ends with 'end'.

    Args:
        text: Input string.
        start: Literal string the text should start with.
        end: Literal string the text should end with.

    Returns:
        True if both conditions are met.
    """
    # TODO: Implement this using ^ and $ anchors
    pass


def find_either(text, word1, word2):
    """Find all occurrences of word1 or word2 in text.

    Args:
        text: Input string.
        word1: First word to find.
        word2: Second word to find.

    Returns:
        List of all found words.
    """
    # TODO: Implement this using alternation
    pass


def escape_and_find(text, literal):
    """Find a literal string that may contain metacharacters.

    Args:
        text: Input string.
        literal: String to find (may contain regex metacharacters).

    Returns:
        True if found, False otherwise.
    """
    # TODO: Implement this using re.escape
    pass


# === Tests ===

assert count_substring("hello world hello", "hello") == 2
assert count_substring("a.b.c.d", ".") == 3
assert count_substring("(yes) (no)", "(") == 2

assert sorted(match_three_letter_words("aba cdc pop abc ab deed")) == ['aba', 'cdc', 'pop']

assert starts_and_ends_with("Hello World", "Hello", "World") is True
assert starts_and_ends_with("Hello World", "Hello", "Hello") is False

assert find_either("I have a cat and a dog", "cat", "dog") == ["cat", "dog"]
assert find_either("no matches here", "cat", "dog") == []

assert escape_and_find("Is this real? (yes/no)", "real? (yes/no)") is True
assert escape_and_find("hello world", "hello.world") is False

print("All tests passed!")
