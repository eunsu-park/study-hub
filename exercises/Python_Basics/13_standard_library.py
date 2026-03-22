"""
Exercise 13: Standard Library

Practice using Counter, datetime, and other standard library modules.
"""

from collections import Counter, defaultdict
from datetime import datetime, timedelta
from itertools import groupby
from functools import reduce


def top_n_words(text, n):
    """Return the n most common words in text (case-insensitive).

    Words are split by whitespace. Use collections.Counter.

    Args:
        text: Input text string.
        n: Number of top words to return.

    Returns:
        List of (word, count) tuples, most common first.
    """
    # TODO: Implement using Counter
    pass


def most_common_char(s):
    """Return the most common character in a string (excluding spaces).

    If tie, return any one of the tied characters.

    Args:
        s: Input string.

    Returns:
        The most common character.
    """
    # TODO: Implement using Counter
    pass


def days_between(date_str1, date_str2):
    """Calculate the absolute number of days between two dates.

    Date format: "YYYY-MM-DD"

    Args:
        date_str1: First date string.
        date_str2: Second date string.

    Returns:
        Absolute number of days between the two dates.
    """
    # TODO: Implement using datetime.strptime and timedelta
    pass


def add_business_days(start_date_str, num_days):
    """Add business days (Mon-Fri) to a start date.

    Args:
        start_date_str: Start date as "YYYY-MM-DD".
        num_days: Number of business days to add (positive).

    Returns:
        Result date as "YYYY-MM-DD" string.
    """
    # TODO: Implement by skipping weekends
    pass


def consecutive_groups(numbers):
    """Find groups of consecutive integers in a sorted list.

    Example: [1,2,3,5,6,8,10,11,12] -> [[1,2,3],[5,6],[8],[10,11,12]]

    Args:
        numbers: Sorted list of integers.

    Returns:
        List of lists, each containing consecutive numbers.
    """
    # TODO: Implement using itertools.groupby or manual approach
    pass


def running_total(numbers):
    """Return a list of running totals.

    Example: [1, 2, 3, 4] -> [1, 3, 6, 10]

    Use itertools.accumulate or functools.reduce approach.

    Args:
        numbers: List of numbers.

    Returns:
        List of cumulative sums.
    """
    # TODO: Implement this
    pass


# === Tests ===

# Top N words
text = "the cat sat on the mat the cat likes the mat"
top = top_n_words(text, 2)
assert top[0] == ("the", 4), "Most common word"
assert top[1] == ("mat", 2) or top[1] == ("cat", 2), "Second most common"

# Most common char
assert most_common_char("aabbbcc") == "b", "Most common b"
assert most_common_char("hello world") in ("l", "o"), "Most common in hello world"

# Days between
assert days_between("2024-01-01", "2024-01-31") == 30, "30 days"
assert days_between("2024-03-01", "2024-01-01") == 60, "60 days (abs)"
assert days_between("2024-01-01", "2024-01-01") == 0, "Same day"

# Business days
assert add_business_days("2024-01-08", 5) == "2024-01-15", "Mon + 5 bdays = next Mon"
assert add_business_days("2024-01-05", 1) == "2024-01-08", "Fri + 1 bday = Mon"
assert add_business_days("2024-01-12", 3) == "2024-01-17", "Fri + 3 bdays = Wed"

# Consecutive groups
assert consecutive_groups([1, 2, 3, 5, 6, 8, 10, 11, 12]) == [
    [1, 2, 3], [5, 6], [8], [10, 11, 12]
], "Consecutive groups"
assert consecutive_groups([1]) == [[1]], "Single element"
assert consecutive_groups([1, 3, 5]) == [[1], [3], [5]], "No consecutive"

# Running total
assert running_total([1, 2, 3, 4]) == [1, 3, 6, 10], "Running total"
assert running_total([5]) == [5], "Single element"
assert running_total([]) == [], "Empty list"

print("All tests passed!")
