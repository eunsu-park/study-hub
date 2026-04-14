"""
Exercise 04: Quantifiers

Practice *, +, ?, {n,m}, greedy vs lazy matching.
"""

import re


def find_optional_plural(text, word):
    """Find a word with an optional trailing 's' (simple plural).

    Args:
        text: Input string.
        word: Base word (e.g., "cat").

    Returns:
        List of matches (e.g., ["cat", "cats"]).
    """
    # TODO: Implement this using ?
    pass


def extract_quoted_strings(text):
    """Extract all double-quoted strings from text.

    Use lazy matching to handle multiple quoted strings correctly.

    Args:
        text: Input string.

    Returns:
        List of strings (without the quotes).
    """
    # TODO: Implement this using lazy quantifier
    pass


def validate_password_length(password, min_len, max_len):
    """Check if password length is between min_len and max_len.

    Args:
        password: Password string.
        min_len: Minimum length.
        max_len: Maximum length.

    Returns:
        True if password length is valid.
    """
    # TODO: Implement this using {n,m}
    pass


def find_repeated_chars(text, n):
    """Find all sequences of exactly n repeated identical characters.

    Args:
        text: Input string.
        n: Number of repetitions.

    Returns:
        List of matched sequences.
    """
    # TODO: Implement this using backreference and {n}
    pass


def extract_html_tags(html):
    """Extract all HTML tag names from an HTML string.

    Use a negated character class for efficiency.

    Args:
        html: HTML string.

    Returns:
        List of tag names (e.g., ["div", "p", "span"]).
    """
    # TODO: Implement this
    pass


# === Tests ===

assert find_optional_plural("I have 3 cats and 1 cat", "cat") == ["cats", "cat"]
assert find_optional_plural("no dogs here", "cat") == []

assert extract_quoted_strings('He said "hello" and "world"') == ["hello", "world"]
assert extract_quoted_strings('no quotes') == []
assert extract_quoted_strings('"single"') == ["single"]

assert validate_password_length("short", 8, 20) is False
assert validate_password_length("longenough", 8, 20) is True
assert validate_password_length("a" * 21, 8, 20) is False

assert find_repeated_chars("aabbbcccc", 2) == ["aa", "cc"]
assert find_repeated_chars("aabbbcccc", 3) == ["bbb", "ccc"]

assert extract_html_tags("<div><p>text</p><span>more</span></div>") == ["div", "p", "p", "span", "span", "div"]

print("All tests passed!")
