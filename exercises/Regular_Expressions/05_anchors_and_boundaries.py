"""
Exercise 05: Anchors and Boundaries

Practice ^, $, \\b, \\B, and re.MULTILINE.
"""

import re


def find_lines_starting_with(text, prefix):
    """Find all lines that start with a given prefix.

    Args:
        text: Multi-line input string.
        prefix: The prefix to match at line start.

    Returns:
        List of full lines starting with the prefix.
    """
    # TODO: Implement this using ^ and re.MULTILINE
    pass


def find_whole_word(text, word):
    """Find all occurrences of a word as a complete word (not part of another).

    Args:
        text: Input string.
        word: The word to find.

    Returns:
        List of matches.
    """
    # TODO: Implement this using \b
    pass


def validate_identifier(name):
    """Check if name is a valid Python identifier.

    Must start with a letter or underscore, followed by letters,
    digits, or underscores. Length 1-30.

    Args:
        name: String to validate.

    Returns:
        True if valid identifier.
    """
    # TODO: Implement this using ^ and $
    pass


def count_blank_lines(text):
    """Count the number of blank lines in text.

    A blank line is a line with no content (just a newline).

    Args:
        text: Multi-line input string.

    Returns:
        Integer count of blank lines.
    """
    # TODO: Implement this using ^$ and re.MULTILINE
    pass


def strip_trailing_whitespace(text):
    """Remove trailing spaces and tabs from each line.

    Args:
        text: Multi-line input string.

    Returns:
        Text with trailing whitespace removed from each line.
    """
    # TODO: Implement this using $ anchor
    pass


# === Tests ===

text = "ERROR: failed\nINFO: ok\nERROR: timeout\nDEBUG: trace"
assert find_lines_starting_with(text, "ERROR") == ["ERROR: failed", "ERROR: timeout"]
assert find_lines_starting_with(text, "DEBUG") == ["DEBUG: trace"]
assert find_lines_starting_with(text, "WARN") == []

assert find_whole_word("cat scatter category", "cat") == ["cat"]
assert find_whole_word("Java JavaScript", "Java") == ["Java"]
assert find_whole_word("hello", "world") == []

assert validate_identifier("my_var") is True
assert validate_identifier("_private") is True
assert validate_identifier("1bad") is False
assert validate_identifier("") is False
assert validate_identifier("a" * 31) is False

assert count_blank_lines("line1\n\nline3\n\nline5") == 2
assert count_blank_lines("no blanks") == 0

assert strip_trailing_whitespace("hello   \nworld  ") == "hello\nworld"
assert strip_trailing_whitespace("no trailing") == "no trailing"

print("All tests passed!")
