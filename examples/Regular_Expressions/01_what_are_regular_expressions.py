"""
01 What Are Regular Expressions
================================
Demonstrates the basics of Python's re module: search, match,
findall, finditer, fullmatch, and compiled patterns.
"""

import re


def basic_search():
    """Demonstrate re.search() for finding the first match."""
    text = "The year is 2024 and the month is 12."

    match = re.search(r'\d{4}', text)
    if match:
        print(f"Found: {match.group()}")
        print(f"Position: {match.start()}-{match.end()}")
        print(f"Span: {match.span()}")


def match_vs_search():
    """Show the difference between re.match() and re.search()."""
    text = "Error 404: Page not found"

    print("re.match() checks start of string only:")
    print(f"  match(r'\\d+', text)  -> {re.match(r'd+', text)}")
    print(f"  match(r'Error', text) -> {re.match(r'Error', text)}")

    print("\nre.search() scans the entire string:")
    print(f"  search(r'\\d+', text) -> {re.search(r'd+', text)}")


def findall_demo():
    """Demonstrate re.findall() to get all matches."""
    text = "Prices: $10.50, $23.99, $5.00, and $199.99"

    prices = re.findall(r'\$\d+\.\d{2}', text)
    print(f"All prices: {prices}")

    # findall with no matches
    missing = re.findall(r'EUR\d+', text)
    print(f"EUR prices: {missing}")


def finditer_demo():
    """Demonstrate re.finditer() for detailed match info."""
    text = "2024-01-15 Error: failed\n2024-01-16 Info: retry OK"

    print("Dates found:")
    for match in re.finditer(r'\d{4}-\d{2}-\d{2}', text):
        print(f"  {match.group()} at position {match.start()}")


def fullmatch_demo():
    """Demonstrate re.fullmatch() for validation."""
    pattern = r'\d{4}-\d{2}-\d{2}'

    test_cases = [
        "2024-01-15",
        "Date: 2024-01-15",
        "2024-1-5",
    ]

    print("Full string validation:")
    for test in test_cases:
        result = re.fullmatch(pattern, test)
        print(f"  '{test}' -> {'Match' if result else 'No match'}")


def compiled_pattern():
    """Demonstrate re.compile() for pattern reuse."""
    phone_pattern = re.compile(r'(\d{3})-(\d{3})-(\d{4})')

    texts = [
        "Call 555-867-5309 for info",
        "Fax: 555-123-4567",
        "No phone here",
        "Both 111-222-3333 and 444-555-6666",
    ]

    for text in texts:
        matches = phone_pattern.findall(text)
        if matches:
            for area, exchange, number in matches:
                print(f"  ({area}) {exchange}-{number}")
        else:
            print(f"  No phone in: '{text}'")


def match_object_demo():
    """Explore Match object methods."""
    text = "My phone number is 555-867-5309."
    match = re.search(r'(\d{3})-(\d{3})-(\d{4})', text)

    if match:
        print(f"Full match:  {match.group()}")
        print(f"Group 0:     {match.group(0)}")
        print(f"Group 1:     {match.group(1)}")
        print(f"Group 2:     {match.group(2)}")
        print(f"Group 3:     {match.group(3)}")
        print(f"All groups:  {match.groups()}")
        print(f"Start:       {match.start()}")
        print(f"End:         {match.end()}")
        print(f"Span:        {match.span()}")


def raw_string_importance():
    """Show why raw strings matter for regex."""
    # Without raw string: \b is Python's backspace
    result_bad = re.findall('\bword\b', 'a word here')
    print(f"Without raw string: {result_bad}")

    # With raw string: \b is word boundary
    result_good = re.findall(r'\bword\b', 'a word here')
    print(f"With raw string:    {result_good}")


if __name__ == "__main__":
    sections = [
        ("Basic Search", basic_search),
        ("Match vs Search", match_vs_search),
        ("Find All", findall_demo),
        ("Find Iterator", finditer_demo),
        ("Full Match", fullmatch_demo),
        ("Compiled Pattern", compiled_pattern),
        ("Match Object", match_object_demo),
        ("Raw Strings", raw_string_importance),
    ]

    for title, func in sections:
        print(f"\n{'=' * 50}")
        print(f" {title}")
        print('=' * 50)
        func()
