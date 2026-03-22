"""
Exercise 07: Strings

Practice string manipulation, word counting, pattern matching, and encoding.
"""

import re


def reverse_words(sentence):
    """Reverse the order of words in a sentence.

    Preserve single spaces between words. Strip leading/trailing whitespace.
    Example: "hello world" -> "world hello"

    Args:
        sentence: Input string.

    Returns:
        String with words in reverse order.
    """
    # TODO: Implement this
    pass


def word_frequency(text):
    """Count the frequency of each word in text (case-insensitive).

    Words are split by whitespace. Strip punctuation from each word using
    str.strip() with '.,!?;:"\\'()'.

    Args:
        text: Input text string.

    Returns:
        Dict mapping lowercase words to their counts.
    """
    # TODO: Implement this
    pass


def is_valid_email(email):
    """Check if a string is a valid email address using regex.

    A valid email has:
    - One or more word characters, dots, or hyphens before @
    - One or more word characters or hyphens after @
    - A dot followed by 2-4 alphabetic characters at the end

    Pattern: r'^[\\w.-]+@[\\w-]+\\.[a-zA-Z]{2,4}$'

    Args:
        email: String to validate.

    Returns:
        True if valid email, False otherwise.
    """
    # TODO: Implement this using re.match
    pass


def caesar_cipher(text, shift):
    """Encrypt text using a Caesar cipher.

    Shift only alphabetic characters. Preserve case and non-alpha chars.
    Wrap around (e.g., 'z' shifted by 1 becomes 'a').

    Args:
        text: Input string.
        shift: Integer shift value (can be negative for decryption).

    Returns:
        Encrypted string.
    """
    # TODO: Implement this
    pass


def title_case(sentence):
    """Convert a sentence to title case.

    Capitalize the first letter of each word, lowercase the rest.
    Do NOT use str.title() -- implement manually.

    Args:
        sentence: Input string.

    Returns:
        Title-cased string.
    """
    # TODO: Implement this without using str.title()
    pass


def compress_string(s):
    """Compress a string using run-length encoding.

    Example: "aaabbc" -> "a3b2c1"
    If compressed is not shorter than original, return original.

    Args:
        s: Input string (only lowercase letters).

    Returns:
        Compressed string, or original if compression doesn't help.
    """
    # TODO: Implement this
    pass


# === Tests ===

assert reverse_words("hello world") == "world hello", "Reverse 2 words"
assert reverse_words("  the quick brown fox  ") == "fox brown quick the", "Strip + reverse"
assert reverse_words("single") == "single", "Single word"

freq = word_frequency("the cat sat on the mat. The cat!")
assert freq["the"] == 3, "Frequency: the"
assert freq["cat"] == 2, "Frequency: cat"
assert freq["mat"] == 1, "Frequency: mat"

assert is_valid_email("user@example.com") is True, "Valid email"
assert is_valid_email("user.name@domain.org") is True, "Valid with dot"
assert is_valid_email("invalid@") is False, "No domain"
assert is_valid_email("@domain.com") is False, "No user"
assert is_valid_email("user@domain") is False, "No TLD"

assert caesar_cipher("abc", 1) == "bcd", "Shift +1"
assert caesar_cipher("xyz", 3) == "abc", "Wrap around"
assert caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!", "ROT13"
assert caesar_cipher("bcd", -1) == "abc", "Shift -1 (decrypt)"

assert title_case("hello world") == "Hello World", "Basic title case"
assert title_case("HELLO WORLD") == "Hello World", "Uppercase input"

assert compress_string("aaabbc") == "a3b2c1", "Compress aaabbc"
assert compress_string("abc") == "abc", "No compression benefit"
assert compress_string("aaaaaa") == "a6", "All same"

print("All tests passed!")
