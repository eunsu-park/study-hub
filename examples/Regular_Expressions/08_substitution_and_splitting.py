"""
08 Substitution and Splitting
==============================
Demonstrates re.sub, re.subn, re.split, callback functions,
group references, and edge cases.
"""

import re


def basic_substitution():
    """Replace patterns with re.sub."""
    text = "I have 3 cats and 12 dogs"
    print(f"Digits to #: {re.sub(r'\\d', '#', text)}")
    print(f"Numbers to N: {re.sub(r'\\d+', 'N', text)}")
    print(f"First only:   {re.sub(r'\\d+', 'N', text, count=1)}")


def group_references():
    """Use group references in replacement strings."""
    # Swap names
    text = "Smith, John"
    print(f"Swap: {re.sub(r'(\\w+), (\\w+)', r'\\2 \\1', text)}")

    # Reformat dates
    text2 = "01/15/2024"
    print(f"Date: {re.sub(r'(\\d{2})/(\\d{2})/(\\d{4})', r'\\3-\\1-\\2', text2)}")

    # Named groups
    text3 = "2024-01-15"
    result = re.sub(
        r'(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})',
        r'\g<d>/\g<m>/\g<y>',
        text3
    )
    print(f"Named: {result}")


def callback_functions():
    """Use callback functions for dynamic replacements."""
    text = "I have 3 cats and 12 dogs"

    # Double all numbers
    result = re.sub(r'\d+', lambda m: str(int(m.group()) * 2), text)
    print(f"Doubled: {result}")

    # Temperature conversion
    text2 = "Today: 72F, Tomorrow: 85F"
    def f_to_c(m):
        c = (int(m.group(1)) - 32) * 5 / 9
        return f"{c:.1f}C"
    print(f"Celsius: {re.sub(r'(\\d+)F', f_to_c, text2)}")

    # Uppercase words starting with 'p'
    text3 = "python programming is powerful"
    result = re.sub(r'\bp\w+', lambda m: m.group().upper(), text3)
    print(f"Uppercased: {result}")


def subn_demo():
    """Count replacements with re.subn."""
    text = "cat bat rat cat mat"
    result, count = re.subn(r'[cbr]at', 'dog', text)
    print(f"Result: {result}")
    print(f"Count:  {count}")


def basic_splitting():
    """Split strings on patterns."""
    text = "Hello   World\tPython\nRegex"
    print(f"\\s+: {re.split(r'\\s+', text)}")

    text2 = "apple, banana,cherry ,  grape"
    print(f"comma: {re.split(r'\\s*,\\s*', text2)}")

    text3 = "one;two,three:four|five"
    print(f"multi: {re.split(r'[;,:|]', text3)}")


def split_with_groups():
    """Keep delimiters by using capture groups."""
    text = "one1two2three3four"
    print(f"Without group: {re.split(r'\\d', text)}")
    print(f"With group:    {re.split(r'(\\d)', text)}")

    # Split sentences, keep punctuation
    text2 = "Hello! How are you? Fine."
    parts = re.split(r'([.!?])\\s*', text2)
    print(f"Sentences: {parts}")


def practical_examples():
    """Real-world substitution and splitting."""
    # Template variables
    template = "Hello, {{name}}! You have {{count}} messages."
    variables = {"name": "Alice", "count": "5"}
    result = re.sub(r'\{\{(\w+)\}\}', lambda m: variables.get(m.group(1), m.group()), template)
    print(f"Template: {result}")

    # Strip HTML tags
    html = "<p>Hello <b>World</b>!</p>"
    clean = re.sub(r'<[^>]+>', '', html)
    print(f"No HTML: {clean}")

    # Normalize whitespace
    text = "  Hello   World  \n  How  "
    clean = re.sub(r'\s+', ' ', text).strip()
    print(f"Normalized: '{clean}'")


if __name__ == "__main__":
    sections = [
        ("Basic Substitution", basic_substitution),
        ("Group References", group_references),
        ("Callback Functions", callback_functions),
        ("Subn Demo", subn_demo),
        ("Basic Splitting", basic_splitting),
        ("Split with Groups", split_with_groups),
        ("Practical Examples", practical_examples),
    ]

    for title, func in sections:
        print(f"\n{'=' * 50}")
        print(f" {title}")
        print('=' * 50)
        func()
