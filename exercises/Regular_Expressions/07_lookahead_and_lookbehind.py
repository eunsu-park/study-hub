"""
Exercise 07: Lookahead and Lookbehind

Practice positive/negative lookahead/lookbehind assertions.
"""

import re


def find_words_before_comma(text):
    """Find words that are immediately followed by a comma.

    The comma should NOT be included in the match.

    Args:
        text: Input string.

    Returns:
        List of words before commas.
    """
    # TODO: Implement this using positive lookahead
    pass


def find_numbers_not_in_dollars(text):
    """Find numbers that are NOT preceded by a dollar sign.

    Args:
        text: Input string.

    Returns:
        List of number strings.
    """
    # TODO: Implement this using negative lookbehind
    pass


def extract_price_values(text):
    """Extract numeric values from prices (after $ sign).

    Do NOT include the $ sign in the result.

    Args:
        text: Input string.

    Returns:
        List of price value strings.
    """
    # TODO: Implement this using positive lookbehind
    pass


def add_thousand_separators(number):
    """Add comma separators to a number string.

    Args:
        number: Integer as int or string (e.g., 1234567).

    Returns:
        String with commas (e.g., "1,234,567").
    """
    # TODO: Implement this using lookahead/lookbehind in re.sub
    pass


def validate_password_rules(password):
    """Check that password has uppercase, lowercase, digit, and 8+ chars.

    Args:
        password: Password string.

    Returns:
        True if all rules pass, False otherwise.
    """
    # TODO: Implement this using multiple lookaheads
    pass


# === Tests ===

assert find_words_before_comma("apple, banana, cherry and grape") == ["apple", "banana"]
assert find_words_before_comma("no commas") == []

assert find_numbers_not_in_dollars("$50 and 10 items at $25") == ["10"]

assert extract_price_values("$19.99 and $5.00 and 42") == ["19.99", "5.00"]
assert extract_price_values("no prices") == []

assert add_thousand_separators(1234567) == "1,234,567"
assert add_thousand_separators(1000) == "1,000"
assert add_thousand_separators(42) == "42"
assert add_thousand_separators(1000000000) == "1,000,000,000"

assert validate_password_rules("P@ssw0rd") is True
assert validate_password_rules("password") is False
assert validate_password_rules("SHORT1") is False
assert validate_password_rules("alllowercase1") is False

print("All tests passed!")
