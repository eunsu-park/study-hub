"""
Exercise 01: What Are Regular Expressions

Practice using re.search, re.match, re.findall, re.finditer,
re.fullmatch, and match objects.
"""

import re


def find_first_number(text):
    """Find the first number (sequence of digits) in text.

    Args:
        text: Input string.

    Returns:
        The first number as a string, or None if no number found.
    """
    # TODO: Implement this using re.search
    pass


def find_all_words(text):
    """Find all words (sequences of word characters) in text.

    Args:
        text: Input string.

    Returns:
        List of all words found.
    """
    # TODO: Implement this using re.findall
    pass


def starts_with_digit(text):
    """Check if the text starts with a digit.

    Args:
        text: Input string.

    Returns:
        True if text starts with a digit, False otherwise.
    """
    # TODO: Implement this using re.match
    pass


def is_valid_date_format(text):
    """Check if the entire string matches YYYY-MM-DD format.

    Args:
        text: Input string.

    Returns:
        True if the entire string matches, False otherwise.
    """
    # TODO: Implement this using re.fullmatch
    pass


def extract_positions(text, pattern):
    """Find all matches of pattern and return their (start, end) positions.

    Args:
        text: Input string.
        pattern: Regex pattern string.

    Returns:
        List of (start, end) tuples for each match.
    """
    # TODO: Implement this using re.finditer
    pass


def count_matches(text, pattern):
    """Count the number of non-overlapping matches of pattern in text.

    Args:
        text: Input string.
        pattern: Regex pattern string.

    Returns:
        Integer count of matches.
    """
    # TODO: Implement this using re.findall
    pass


# === Tests ===

assert find_first_number("abc 42 def 99") == "42"
assert find_first_number("no numbers") is None
assert find_first_number("123abc") == "123"

assert find_all_words("Hello, World! 123") == ["Hello", "World", "123"]
assert find_all_words("   ") == []
assert find_all_words("one") == ["one"]

assert starts_with_digit("3 cats") is True
assert starts_with_digit("cats 3") is False
assert starts_with_digit("") is False

assert is_valid_date_format("2024-01-15") is True
assert is_valid_date_format("Date: 2024-01-15") is False
assert is_valid_date_format("2024-1-5") is False

assert extract_positions("abc 123 def 456", r'\d+') == [(4, 7), (12, 15)]
assert extract_positions("no match", r'\d+') == []

assert count_matches("cat bat rat mat", r'[a-z]at') == 4
assert count_matches("hello", r'\d') == 0

print("All tests passed!")
