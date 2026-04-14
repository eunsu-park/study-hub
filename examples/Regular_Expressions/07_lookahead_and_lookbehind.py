"""
07 Lookahead and Lookbehind
============================
Demonstrates positive/negative lookahead and lookbehind,
combined lookarounds, and practical applications.
"""

import re


def positive_lookahead():
    """Match only when followed by a specific pattern."""
    text = "foobar foobaz foo"
    print(f"foo(?=bar): {re.findall(r'foo(?=bar)', text)}")

    text2 = "100px, 50em, 200px, 75%"
    print(f"\\d+(?=px):  {re.findall(r'\\d+(?=px)', text2)}")

    text3 = "apple, banana, cherry and grape"
    print(f"\\w+(?=,):   {re.findall(r'\\w+(?=,)', text3)}")


def negative_lookahead():
    """Match only when NOT followed by a pattern."""
    text = "foobar foobaz foo"
    print(f"foo(?!bar): {re.findall(r'foo(?!bar)', text)}")

    # Non-comment lines
    code = "# comment\nprint('hi')\n# comment\nx = 42"
    lines = re.findall(r'^(?!#).*$', code, re.M)
    print(f"Non-comment: {lines}")


def positive_lookbehind():
    """Match only when preceded by a specific pattern."""
    text = "Price: $50, Quantity: 10, Total: $500"
    print(f"(?<=\\$)\\d+: {re.findall(r'(?<=\\$)\\d+', text)}")

    text2 = "name=John age=30 city=NYC"
    print(f"(?<=\\=)\\w+: {re.findall(r'(?<=\\=)\\w+', text2)}")

    # Inside parentheses
    text3 = "Hello (World) and (Python)"
    print(f"Inside parens: {re.findall(r'(?<=\\()\\w+(?=\\))', text3)}")


def negative_lookbehind():
    """Match only when NOT preceded by a pattern."""
    text = "Price: $50, Quantity: 10, Total: $500"
    print(f"Not after $: {re.findall(r'(?<!\\$)\\b\\d+', text)}")


def combined_lookarounds():
    """Chain multiple lookarounds for complex validation."""
    # Password validation
    pattern = r'^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[!@#$%]).{8,}$'
    passwords = ["P@ssw0rd", "password", "SHORT1!", "Valid1!abc"]
    for pwd in passwords:
        valid = bool(re.match(pattern, pwd))
        print(f"  '{pwd}': {'Strong' if valid else 'Weak'}")


def lookaround_substitution():
    """Use lookarounds in re.sub for zero-width insertions."""
    # Add commas to numbers
    def add_commas(n):
        return re.sub(r'(?<=\d)(?=(\d{3})+(?!\d))', ',', str(n))

    for n in [1234567, 1000000000, 42, 1000]:
        print(f"  {n} -> {add_commas(n)}")

    # camelCase to words
    text = "camelCaseToSeparateWords"
    result = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    print(f"\ncamelCase: {result}")


if __name__ == "__main__":
    sections = [
        ("Positive Lookahead", positive_lookahead),
        ("Negative Lookahead", negative_lookahead),
        ("Positive Lookbehind", positive_lookbehind),
        ("Negative Lookbehind", negative_lookbehind),
        ("Combined Lookarounds", combined_lookarounds),
        ("Lookaround Substitution", lookaround_substitution),
    ]

    for title, func in sections:
        print(f"\n{'=' * 50}")
        print(f" {title}")
        print('=' * 50)
        func()
