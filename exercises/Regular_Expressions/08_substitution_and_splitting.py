"""
Exercise 08: Substitution and Splitting

Practice re.sub, re.subn, re.split, callbacks, and group references.
"""

import re


def censor_words(text, words):
    """Replace specified words with '***' (case-insensitive).

    Args:
        text: Input string.
        words: Set of words to censor.

    Returns:
        Censored text.
    """
    # TODO: Implement this using re.sub with callback
    pass


def reformat_dates(text):
    """Convert all dates from MM/DD/YYYY to YYYY-MM-DD format.

    Args:
        text: Input string containing dates.

    Returns:
        Text with reformatted dates.
    """
    # TODO: Implement this using re.sub with group references
    pass


def normalize_whitespace(text):
    """Replace all consecutive whitespace with a single space and strip.

    Args:
        text: Input string.

    Returns:
        Normalized string.
    """
    # TODO: Implement this
    pass


def split_sentences(text):
    """Split text into sentences (ending with . ! or ?).

    Keep the punctuation attached to the sentence.

    Args:
        text: Input string.

    Returns:
        List of sentences (stripped of leading/trailing whitespace).
    """
    # TODO: Implement this using re.split with groups
    pass


def replace_template_vars(template, variables):
    """Replace {{var}} placeholders with values from variables dict.

    Args:
        template: Template string with {{var}} placeholders.
        variables: Dict mapping variable names to values.

    Returns:
        String with placeholders replaced.
    """
    # TODO: Implement this using re.sub with callback
    pass


# === Tests ===

assert censor_words("This is damn bad", {"damn", "bad"}) == "This is *** ***"
assert censor_words("Hello World", {"test"}) == "Hello World"

assert reformat_dates("Born on 01/15/2024") == "Born on 2024-01-15"
assert reformat_dates("12/31/2024 and 06/15/2025") == "2024-12-31 and 2025-06-15"

assert normalize_whitespace("  Hello   World  \n test  ") == "Hello World test"

sentences = split_sentences("Hello! How are you? I am fine.")
assert sentences == ["Hello!", "How are you?", "I am fine."]

result = replace_template_vars("Hello, {{name}}! Age: {{age}}", {"name": "Alice", "age": "30"})
assert result == "Hello, Alice! Age: 30"

print("All tests passed!")
