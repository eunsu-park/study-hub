"""
02 Literal Matching and Metacharacters
=======================================
Demonstrates literal matching, the dot metacharacter, anchors,
alternation, escaping, and re.escape().
"""

import re


def literal_matching():
    """Match literal text in strings."""
    text = "The cat sat on the mat"
    print(f"Find 'cat': {re.findall(r'cat', text)}")
    print(f"Find 'at':  {re.findall(r'at', text)}")
    print(f"Case sensitive: {re.findall(r'CAT', text)}")


def dot_metacharacter():
    """The dot matches any single character except newline."""
    print("Pattern c.t:")
    for word in ["cat", "cot", "cut", "c1t", "c_t", "ct", "cart"]:
        match = re.fullmatch(r'c.t', word)
        print(f"  '{word}' -> {'Match' if match else 'No match'}")

    text = "a.b acb a1b a\nb"
    print(f"\nPattern a.b in '{repr(text)}':")
    print(f"  Matches: {re.findall(r'a.b', text)}")


def anchors_basic():
    """Demonstrate ^ and $ anchors."""
    text = "Python is great"
    print(f"^Python: {bool(re.search(r'^Python', text))}")
    print(f"^is:     {bool(re.search(r'^is', text))}")
    print(f"great$:  {bool(re.search(r'great$', text))}")
    print(f"Python$: {bool(re.search(r'Python$', text))}")

    # Full string validation
    for code in ["12345", "1234", "123456", "abcde"]:
        valid = bool(re.fullmatch(r'\d{5}', code))
        print(f"  ZIP '{code}': {'Valid' if valid else 'Invalid'}")


def alternation():
    """The pipe operator for OR matching."""
    text = "I have a cat and a dog and a bird"
    print(f"cat|dog:      {re.findall(r'cat|dog', text)}")
    print(f"cat|dog|bird: {re.findall(r'cat|dog|bird', text)}")

    # Alternation scope
    print(f"\ngray|grey:   {re.findall(r'gray|grey', 'gray and grey')}")
    print(f"gr(a|e)y:    {re.findall(r'gr(a|e)y', 'gray and grey')}")
    print(f"gr(?:a|e)y:  {re.findall(r'gr(?:a|e)y', 'gray and grey')}")


def escaping():
    """Escape metacharacters to match them literally."""
    # Without escaping
    print(f"Unescaped '.': {re.findall(r'.', 'a.b')}")
    print(f"Escaped '\\.': {re.findall(r'\.', 'a.b')}")

    # Match a price
    price = "The price is $9.99"
    match = re.search(r'\$\d+\.\d{2}', price)
    print(f"Price: {match.group() if match else 'Not found'}")

    # re.escape for user input
    user_input = "Is this real? (yes/no)"
    escaped = re.escape(user_input)
    print(f"\nOriginal: {user_input}")
    print(f"Escaped:  {escaped}")
    text = "Question: Is this real? (yes/no)"
    match = re.search(escaped, text)
    print(f"Found:    {match.group() if match else 'Not found'}")


def combining_patterns():
    """Build patterns combining literals and metacharacters."""
    # Date-like pattern
    text = "Events on 2024-01-15 and 2024-12-31"
    dates = re.findall(r'\d{4}-\d{2}-\d{2}', text)
    print(f"Dates: {dates}")

    # File extensions
    files = "report.pdf, data.csv, image.png, script.py"
    matches = re.findall(r'\w+\.(?:pdf|csv)', files)
    print(f"PDF/CSV files: {matches}")

    # Lines starting with ERROR
    log = "INFO: OK\nERROR: fail\nINFO: retry\nERROR: timeout"
    errors = re.findall(r'^ERROR.*', log, re.MULTILINE)
    print(f"Error lines: {errors}")


if __name__ == "__main__":
    sections = [
        ("Literal Matching", literal_matching),
        ("Dot Metacharacter", dot_metacharacter),
        ("Anchors", anchors_basic),
        ("Alternation", alternation),
        ("Escaping", escaping),
        ("Combining Patterns", combining_patterns),
    ]

    for title, func in sections:
        print(f"\n{'=' * 50}")
        print(f" {title}")
        print('=' * 50)
        func()
