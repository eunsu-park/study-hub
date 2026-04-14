"""
Exercise 10: Common Patterns

Practice building email, URL, IP, date, and phone patterns.
"""

import re


def validate_email(email):
    """Validate a basic email format.

    Must have: local part @ domain . TLD (2+ letters).

    Args:
        email: Email string to validate.

    Returns:
        True if valid format.
    """
    # TODO: Implement this
    pass


def extract_urls(text):
    """Extract all http/https URLs from text.

    Remove trailing punctuation (. , ; ! ?) from URLs.

    Args:
        text: Input text.

    Returns:
        List of clean URL strings.
    """
    # TODO: Implement this
    pass


def validate_ipv4(ip):
    """Validate an IPv4 address (each octet 0-255).

    Args:
        ip: IP address string.

    Returns:
        True if valid IPv4.
    """
    # TODO: Implement this
    pass


def validate_iso_date(date_str):
    """Validate a date in YYYY-MM-DD format.

    Month must be 01-12, day must be 01-31.

    Args:
        date_str: Date string.

    Returns:
        True if valid format.
    """
    # TODO: Implement this
    pass


def normalize_phone(phone):
    """Normalize a phone number to (XXX) XXX-XXXX format.

    Accept: digits, spaces, hyphens, dots, parens.
    Optional leading +1 or 1 country code.

    Args:
        phone: Phone string in any common format.

    Returns:
        Normalized string, or None if not 10 digits.
    """
    # TODO: Implement this
    pass


# === Tests ===

assert validate_email("user@example.com") is True
assert validate_email("a.b+c@sub.domain.org") is True
assert validate_email("@bad.com") is False
assert validate_email("user@") is False

urls = extract_urls("Visit https://google.com. Also http://test.com/path?q=1.")
assert urls == ["https://google.com", "http://test.com/path?q=1"]

assert validate_ipv4("192.168.1.1") is True
assert validate_ipv4("0.0.0.0") is True
assert validate_ipv4("255.255.255.255") is True
assert validate_ipv4("256.0.0.1") is False
assert validate_ipv4("1.2.3") is False

assert validate_iso_date("2024-01-15") is True
assert validate_iso_date("2024-13-01") is False
assert validate_iso_date("2024-01-32") is False

assert normalize_phone("555-867-5309") == "(555) 867-5309"
assert normalize_phone("+1 555 867 5309") == "(555) 867-5309"
assert normalize_phone("123") is None

print("All tests passed!")
