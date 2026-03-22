"""
07 Strings
==========
Demonstrates string methods, f-string formatting, regular expressions,
encoding/decoding, and common string manipulation patterns.
"""

import re
import textwrap


def string_methods():
    """Common string methods for searching, splitting, and transforming."""
    s = "  Hello, World! Welcome to Python.  "

    # Whitespace handling
    print(f"Original: {s!r}")
    print(f"strip():  {s.strip()!r}")
    print(f"lstrip(): {s.lstrip()!r}")
    print(f"rstrip(): {s.rstrip()!r}")

    # Case methods
    text = "hello world"
    print(f"\nupper():      {text.upper()!r}")
    print(f"capitalize(): {text.capitalize()!r}")
    print(f"title():      {text.title()!r}")
    print(f"swapcase():   {'Hello'.swapcase()!r}")

    # Search methods
    sentence = "the quick brown fox jumps over the lazy dog"
    print(f"\nfind('fox'):     {sentence.find('fox')}")
    print(f"count('the'):    {sentence.count('the')}")
    print(f"startswith:      {sentence.startswith('the')}")
    print(f"endswith:        {sentence.endswith('dog')}")

    # Split and join
    csv_line = "apple,banana,cherry,date"
    parts = csv_line.split(",")
    print(f"\nSplit: {parts}")
    print(f"Join:  {' | '.join(parts)}")

    words = sentence.split()
    print(f"Words: {words[:5]}...")
    print(f"Word count: {len(words)}")

    # Replace
    print(f"\nReplace: {sentence.replace('fox', 'cat')}")

    # Partition
    key, sep, value = "name=Alice".partition("=")
    print(f"Partition: key={key!r}, value={value!r}")


def fstring_features():
    """Advanced f-string formatting (Python 3.6+)."""
    # Number formatting
    n = 1234567.89
    print(f"Default:    {n}")
    print(f"Comma:      {n:,.2f}")
    print(f"Width 15:   {n:>15,.2f}")
    print(f"Percentage: {0.856:.1%}")
    print(f"Scientific: {n:.3e}")

    # Integer formatting
    x = 255
    print(f"\nDecimal:  {x:d}")
    print(f"Binary:   {x:08b}")
    print(f"Octal:    {x:o}")
    print(f"Hex:      {x:02x}")
    print(f"Hex(cap): {x:02X}")

    # Alignment
    for name, score in [("Alice", 95), ("Bob", 82), ("Charlie", 91)]:
        print(f"  {name:<10} | {'*' * (score // 10):>10} | {score:>3}")

    # Fill character
    title = "Report"
    print(f"\n{title:=^40}")
    print(f"{title:.<40}")
    print(f"{title:_>40}")

    # Nested f-strings and expressions
    items = ["apple", "banana", "cherry"]
    print(f"\nItems: {', '.join(f'{i+1}.{item}' for i, item in enumerate(items))}")

    # Date formatting
    from datetime import datetime
    now = datetime(2025, 3, 15, 14, 30, 0)
    print(f"Date: {now:%Y-%m-%d %H:%M}")
    print(f"Day:  {now:%A, %B %d}")


def regex_examples():
    """Regular expressions for pattern matching."""
    # Basic matching
    text = "Call us at 555-1234 or 555-5678 for more info."

    # findall: get all matches
    phones = re.findall(r"\d{3}-\d{4}", text)
    print(f"Phone numbers: {phones}")

    # search: first match
    match = re.search(r"(\d{3})-(\d{4})", text)
    if match:
        print(f"First match: {match.group()}")
        print(f"Area: {match.group(1)}, Number: {match.group(2)}")

    # sub: replace matches
    censored = re.sub(r"\d", "X", text)
    print(f"Censored: {censored}")

    # Compiled pattern for reuse
    email_pattern = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
    emails_text = "Contact alice@example.com or bob.smith@company.co.uk"
    found = email_pattern.findall(emails_text)
    print(f"\nEmails found: {found}")

    # Named groups
    log = "2025-03-15 14:30:00 ERROR Database connection failed"
    pattern = r"(?P<date>\d{4}-\d{2}-\d{2}) (?P<time>\d{2}:\d{2}:\d{2}) (?P<level>\w+) (?P<msg>.+)"
    m = re.match(pattern, log)
    if m:
        print(f"\nLog parsing:")
        print(f"  Date:    {m.group('date')}")
        print(f"  Level:   {m.group('level')}")
        print(f"  Message: {m.group('msg')}")

    # Split with regex
    text = "one, two;  three   four\tfive"
    tokens = re.split(r"[,;\s]+", text)
    print(f"\nRegex split: {tokens}")

    # Lookahead and lookbehind
    prices = "apple $3.50 banana $1.25 cherry $4.00"
    amounts = re.findall(r"(?<=\$)\d+\.\d+", prices)
    print(f"Prices (lookbehind): {amounts}")


def encoding_demo():
    """String encoding, bytes, and Unicode handling."""
    # Strings are Unicode in Python 3
    text = "Hello, \u4e16\u754c!"  # "Hello, World!" in Chinese
    print(f"Unicode: {text}")
    print(f"Length:  {len(text)} characters")

    # Encoding to bytes
    utf8 = text.encode("utf-8")
    print(f"\nUTF-8 bytes:  {utf8}")
    print(f"Byte length:  {len(utf8)}")

    latin = "Hello".encode("ascii")
    print(f"ASCII bytes:  {latin}")

    # Decoding bytes to string
    decoded = utf8.decode("utf-8")
    print(f"Decoded: {decoded}")

    # Byte operations
    data = b"Hello, World!"
    print(f"\nByte literal:  {data}")
    print(f"Type:          {type(data).__name__}")
    print(f"Hex:           {data.hex()}")
    print(f"From hex:      {bytes.fromhex('48656c6c6f')}")

    # ord() and chr()
    print(f"\nord('A') = {ord('A')}")
    print(f"chr(65)  = {chr(65)}")
    print(f"ord('\u00e9') = {ord('\u00e9')}")  # e with accent

    # Unicode normalization
    import unicodedata
    s1 = "\u00e9"        # e-acute (single codepoint)
    s2 = "e\u0301"       # e + combining acute accent
    print(f"\n{s1!r} == {s2!r}: {s1 == s2}")
    s1n = unicodedata.normalize("NFC", s1)
    s2n = unicodedata.normalize("NFC", s2)
    print(f"After NFC normalization: {s1n == s2n}")


def practical_string_ops():
    """Real-world string manipulation patterns."""
    # Text wrapping
    long_text = (
        "Python is a versatile programming language that emphasizes "
        "readability and simplicity. It supports multiple programming "
        "paradigms including procedural, object-oriented, and functional."
    )
    wrapped = textwrap.fill(long_text, width=50)
    print(f"Wrapped text:\n{wrapped}")

    # Template-like string building
    template = "Dear {name},\nYour order #{order_id} is {status}."
    for data in [
        {"name": "Alice", "order_id": "A123", "status": "shipped"},
        {"name": "Bob", "order_id": "B456", "status": "processing"},
    ]:
        print(f"\n{template.format(**data)}")

    # String validation
    samples = ["hello123", "12345", "HELLO", "  ", "hello world", ""]
    print(f"\n{'String':>12} | alpha | digit | alnum | space | empty")
    print("-" * 60)
    for s in samples:
        if s:
            print(
                f"{s!r:>12} | "
                f"{str(s.isalpha()):>5} | "
                f"{str(s.isdigit()):>5} | "
                f"{str(s.isalnum()):>5} | "
                f"{str(s.isspace()):>5} |"
            )

    # Multiline string alignment
    sql = textwrap.dedent("""\
        SELECT name, age
        FROM users
        WHERE age > 18
        ORDER BY name
    """)
    print(f"\nDedented SQL:\n{sql}")


if __name__ == "__main__":
    sections = [
        ("String Methods", string_methods),
        ("f-string Features", fstring_features),
        ("Regular Expressions", regex_examples),
        ("Encoding Demo", encoding_demo),
        ("Practical String Ops", practical_string_ops),
    ]

    for title, func in sections:
        print(f"\n{'=' * 50}")
        print(f"  {title}")
        print('=' * 50)
        func()
