"""
04 Quantifiers
===============
Demonstrates *, +, ?, {n,m}, greedy vs lazy matching,
and backtracking behavior.
"""

import re


def basic_quantifiers():
    """Demonstrate *, +, and ? quantifiers."""
    text = "ac abc abbc abbbc"

    print(f"ab*c:  {re.findall(r'ab*c', text)}")
    print(f"ab+c:  {re.findall(r'ab+c', text)}")

    text2 = "color colour colouur"
    print(f"colou?r: {re.findall(r'colou?r', text2)}")

    text3 = "http://a.com https://b.com"
    print(f"https?://: {re.findall(r'https?://', text3)}")


def exact_repetition():
    """Use {n}, {n,}, and {n,m} for precise control."""
    text = "1 12 123 1234 12345"
    print(f"\\d{{4}}:    {re.findall(r'd{4}', text)}")
    print(f"\\d{{3,}}:   {re.findall(r'd{3,}', text)}")
    print(f"\\d{{2,4}}:  {re.findall(r'd{2,4}', text)}")

    # ZIP code validation
    for code in ["12345", "1234", "12345-6789"]:
        valid = bool(re.fullmatch(r'\d{5}(-\d{4})?', code))
        print(f"  ZIP '{code}': {'Valid' if valid else 'Invalid'}")


def greedy_vs_lazy():
    """Compare greedy and lazy matching."""
    text = "<b>bold</b> and <i>italic</i>"

    greedy = re.search(r'<.*>', text).group()
    lazy = re.findall(r'<.*?>', text)

    print(f"Greedy <.*>:  {greedy}")
    print(f"Lazy <.*?>:   {lazy}")

    text2 = "aabab"
    print(f"\nGreedy a.*b: {re.search(r'a.*b', text2).group()}")
    print(f"Lazy a.*?b:  {re.search(r'a.*?b', text2).group()}")


def negated_class_alternative():
    """Use negated character class instead of lazy quantifier."""
    html = '<span class="name">John</span>'

    lazy = re.search(r'<.*?>', html).group()
    negated = re.search(r'<[^>]+>', html).group()

    print(f"Lazy .*?:     {lazy}")
    print(f"Negated [^>]: {negated}")
    print("(Both work, but negated class is more efficient)")


def quantifier_pitfalls():
    """Common quantifier mistakes."""
    # * matches empty strings
    print(f"\\d* on 'abc': {re.findall(r'd*', 'abc')}")
    print(f"\\d+ on 'abc': {re.findall(r'd+', 'abc')}")

    # Quantifier scope
    text = "ababab cd ab"
    print(f"\n(?:ab)+: {re.findall(r'(?:ab)+', text)}")
    print(f"(ab)+:   {re.findall(r'(ab)+', text)}")


if __name__ == "__main__":
    sections = [
        ("Basic Quantifiers", basic_quantifiers),
        ("Exact Repetition", exact_repetition),
        ("Greedy vs Lazy", greedy_vs_lazy),
        ("Negated Class Alternative", negated_class_alternative),
        ("Quantifier Pitfalls", quantifier_pitfalls),
    ]

    for title, func in sections:
        print(f"\n{'=' * 50}")
        print(f" {title}")
        print('=' * 50)
        func()
