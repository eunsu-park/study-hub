"""
03 Character Classes
=====================
Demonstrates character classes, ranges, negation, shorthand
notations (\\d, \\w, \\s), and Unicode awareness.
"""

import re


def basic_character_class():
    """Match specific sets of characters with [...]."""
    text = "bag big bog bug byg"
    print(f"b[aeiou]g: {re.findall(r'b[aeiou]g', text)}")
    print(f"b[aio]g:   {re.findall(r'b[aio]g', text)}")


def character_ranges():
    """Use ranges inside character classes."""
    text = "Hello World 123 test"
    print(f"[a-z]+:      {re.findall(r'[a-z]+', text)}")
    print(f"[A-Z]+:      {re.findall(r'[A-Z]+', text)}")
    print(f"[0-9]+:      {re.findall(r'[0-9]+', text)}")
    print(f"[a-zA-Z]+:   {re.findall(r'[a-zA-Z]+', text)}")

    # Hex digits
    hex_text = "Color: #FF8C00, value=0x1A3F"
    print(f"\nHex: {re.findall(r'[0-9a-fA-F]+', hex_text)}")


def negated_classes():
    """Use [^...] to match anything NOT in the set."""
    text = "Hello World"
    print(f"[^aeiou]:  {re.findall(r'[^aeiou]', text)}")

    text2 = "abc123def456"
    print(f"[^0-9]+:   {re.findall(r'[^0-9]+', text2)}")

    # ^ only means negation as first char
    text3 = "a^bc"
    print(f"[^abc]:    {re.findall(r'[^abc]', text3)}")
    print(f"[a^bc]:    {re.findall(r'[a^bc]', text3)}")


def shorthand_classes():
    """Use \\d, \\w, \\s and their negations."""
    text = "User: alice_99, Age: 25, Email: alice@test.com"
    print(f"\\d+: {re.findall(r'd+', text)}")
    print(f"\\w+: {re.findall(r'w+', text)}")
    print(f"\\D+: {re.findall(r'D+', text)}")
    print(f"\\W+: {re.findall(r'W+', text)}")
    print(f"\\S+: {re.findall(r'S+', text)}")


def combining_shorthands():
    """Mix shorthands and literal characters in brackets."""
    # Digits or hyphens
    text = "Call 555-867-5309"
    print(f"[\\d-]+:  {re.findall(r'[d-]+', text)}")

    # Word chars or dots
    text2 = "file.txt image.png README"
    print(f"[\\w.]+:  {re.findall(r'[w.]+', text2)}")


def dot_vs_class():
    """Compare . with character classes."""
    text = "a.b acb"
    print(f"a.b:   {re.findall(r'a.b', text)}")
    print(f"a[.]b: {re.findall(r'a[.]b', text)}")


def practical_examples():
    """Real-world character class usage."""
    # Hex color codes
    css = "colors: #FF0000, #00ff00, #abc, not #xyz"
    colors = re.findall(r'#[0-9a-fA-F]{3,6}', css)
    print(f"Hex colors: {colors}")

    # Clean whitespace
    text = "  Hello   World   !  "
    clean = re.sub(r'\s+', ' ', text).strip()
    print(f"Cleaned: '{clean}'")

    # Extract initials
    name = "John Michael Smith"
    initials = "".join(re.findall(r'[A-Z]', name))
    print(f"Initials: {initials}")


if __name__ == "__main__":
    sections = [
        ("Basic Character Class", basic_character_class),
        ("Character Ranges", character_ranges),
        ("Negated Classes", negated_classes),
        ("Shorthand Classes", shorthand_classes),
        ("Combining Shorthands", combining_shorthands),
        ("Dot vs Class", dot_vs_class),
        ("Practical Examples", practical_examples),
    ]

    for title, func in sections:
        print(f"\n{'=' * 50}")
        print(f" {title}")
        print('=' * 50)
        func()
