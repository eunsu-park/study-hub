"""
Exercise 06: Groups and Capturing

Practice capture groups, named groups, backreferences,
and non-capturing groups.
"""

import re


def parse_date(date_str):
    """Parse a date string in YYYY-MM-DD format into components.

    Args:
        date_str: Date string (e.g., "2024-01-15").

    Returns:
        Dict with keys 'year', 'month', 'day' (as strings),
        or None if format doesn't match.
    """
    # TODO: Implement this using named groups
    pass


def swap_first_last_name(text):
    """Convert "Last, First" to "First Last".

    Args:
        text: String in "Last, First" format.

    Returns:
        String in "First Last" format.
    """
    # TODO: Implement this using groups and re.sub
    pass


def find_repeated_words(text):
    """Find words that appear twice in a row (case-insensitive).

    Args:
        text: Input string.

    Returns:
        List of repeated words (lowercase).
    """
    # TODO: Implement this using backreferences
    pass


def extract_markdown_links(text):
    """Extract all Markdown links [text](url) from text.

    Args:
        text: String containing Markdown.

    Returns:
        List of (text, url) tuples.
    """
    # TODO: Implement this using capture groups
    pass


def reformat_phone(phone):
    """Reformat a 10-digit phone to (XXX) XXX-XXXX.

    Accept formats: XXXXXXXXXX, XXX-XXX-XXXX, XXX.XXX.XXXX

    Args:
        phone: Phone number string.

    Returns:
        Formatted phone string, or None if invalid.
    """
    # TODO: Implement this using groups
    pass


# === Tests ===

assert parse_date("2024-01-15") == {"year": "2024", "month": "01", "day": "15"}
assert parse_date("bad-date") is None

assert swap_first_last_name("Smith, John") == "John Smith"
assert swap_first_last_name("Doe, Jane") == "Jane Doe"

assert find_repeated_words("the the cat sat sat") == ["the", "sat"]
assert find_repeated_words("The the") == ["the"]
assert find_repeated_words("no repeats here") == []

links = extract_markdown_links("Click [here](https://a.com) and [there](https://b.com)")
assert links == [("here", "https://a.com"), ("there", "https://b.com")]

assert reformat_phone("5558675309") == "(555) 867-5309"
assert reformat_phone("555-867-5309") == "(555) 867-5309"
assert reformat_phone("555.867.5309") == "(555) 867-5309"
assert reformat_phone("123") is None

print("All tests passed!")
