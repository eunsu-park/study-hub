"""
Exercise 11: Performance and Pitfalls

Practice writing safe patterns, avoiding common pitfalls,
and choosing between regex and string methods.
"""

import re


def safe_tag_extract(html):
    """Extract HTML tag names using a safe (non-backtracking) pattern.

    Use a negated character class instead of .* to avoid backtracking.

    Args:
        html: HTML string.

    Returns:
        List of tag names (opening tags only, no closing tags).
    """
    # TODO: Implement this using [^>]* instead of .*
    pass


def safe_quoted_extract(text):
    """Extract double-quoted strings using a safe pattern.

    Use [^"]* instead of .*? for better performance.

    Args:
        text: Input string.

    Returns:
        List of quoted contents (without quotes).
    """
    # TODO: Implement this with [^"]*
    pass


def fix_pattern(dangerous_pattern):
    """Convert a potentially dangerous pattern to a safe equivalent.

    Handle these cases:
    - (a+)+ -> a+
    - (.*)*  -> .*
    - (\\w+)+ -> \\w+

    Args:
        dangerous_pattern: A regex pattern string.

    Returns:
        A safer equivalent pattern string.
    """
    # TODO: Implement this
    pass


def should_use_regex(task):
    """Determine if regex is appropriate for a given task.

    Args:
        task: One of "fixed_replace", "pattern_match",
              "simple_split", "html_parse", "format_validate".

    Returns:
        True if regex is recommended, False if string methods or
        a parser would be better.
    """
    # TODO: Implement this
    pass


def efficient_search(text, word):
    """Perform a whole-word search using the most efficient approach.

    Use string method first for a quick check, then regex for
    exact word boundary matching only if needed.

    Args:
        text: Input string.
        word: Word to find.

    Returns:
        True if the word appears as a whole word.
    """
    # TODO: Implement this (fast path + regex fallback)
    pass


# === Tests ===

assert safe_tag_extract("<div><p>text</p></div>") == ["div", "p"]
assert safe_tag_extract('<a href="url">link</a>') == ["a"]

assert safe_quoted_extract('He said "hello" and "world"') == ["hello", "world"]
assert safe_quoted_extract("no quotes") == []

assert fix_pattern(r"(a+)+") == r"a+"
assert fix_pattern(r"(.*)*") == r".*"
assert fix_pattern(r"(\w+)+") == r"\w+"

assert should_use_regex("fixed_replace") is False
assert should_use_regex("pattern_match") is True
assert should_use_regex("simple_split") is False
assert should_use_regex("html_parse") is False
assert should_use_regex("format_validate") is True

assert efficient_search("The cat sat", "cat") is True
assert efficient_search("category", "cat") is False
assert efficient_search("no match", "cat") is False

print("All tests passed!")
