"""
Exercise 03: Character Classes

Practice custom character classes, ranges, negation, and shorthands.
"""

import re


def extract_vowels(text):
    """Extract all vowels (a, e, i, o, u) from text, case-insensitive.

    Args:
        text: Input string.

    Returns:
        List of vowels found (preserving original case).
    """
    # TODO: Implement this
    pass


def is_hex_color(text):
    """Check if text is a valid hex color code (#RGB or #RRGGBB).

    Must start with # followed by exactly 3 or 6 hex digits.

    Args:
        text: Input string.

    Returns:
        True if valid hex color.
    """
    # TODO: Implement this
    pass


def extract_non_digits(text):
    """Extract all non-digit characters from text.

    Args:
        text: Input string.

    Returns:
        String of non-digit characters joined together.
    """
    # TODO: Implement this
    pass


def find_consonant_words(text):
    """Find words that contain only consonants (no vowels).

    Words are sequences of alphabetic characters.

    Args:
        text: Input string.

    Returns:
        List of consonant-only words.
    """
    # TODO: Implement this
    pass


def extract_special_chars(text):
    """Extract all characters that are NOT alphanumeric or whitespace.

    Args:
        text: Input string.

    Returns:
        List of special characters found.
    """
    # TODO: Implement this
    pass


# === Tests ===

assert extract_vowels("Hello World") == ['e', 'o', 'o']
assert extract_vowels("AEIOU") == ['A', 'E', 'I', 'O', 'U']
assert extract_vowels("rhythm") == []

assert is_hex_color("#FF0000") is True
assert is_hex_color("#abc") is True
assert is_hex_color("#GGHHII") is False
assert is_hex_color("FF0000") is False
assert is_hex_color("#FF00") is False

assert extract_non_digits("abc123def456") == "abcdef"
assert extract_non_digits("12345") == ""

assert find_consonant_words("my gym fly cry hello") == ["my", "gym", "fly", "cry"]

assert extract_special_chars("Hello, World! #1") == [',', '!', '#']
assert extract_special_chars("abc123") == []

print("All tests passed!")
