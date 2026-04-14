"""
09 Flags and Options
=====================
Demonstrates re.IGNORECASE, re.MULTILINE, re.DOTALL, re.VERBOSE,
re.ASCII, combining flags, and inline flags.
"""

import re


def ignorecase_flag():
    """Case-insensitive matching with re.I."""
    text = "Python PYTHON python PyThOn"
    print(f"Default:    {re.findall(r'python', text)}")
    print(f"IGNORECASE: {re.findall(r'python', text, re.I)}")


def multiline_flag():
    """Line-by-line anchors with re.M."""
    text = "Line 1\nLine 2\nLine 3"
    print(f"Default:    {re.findall(r'^Line', text)}")
    print(f"MULTILINE:  {re.findall(r'^Line', text, re.M)}")


def dotall_flag():
    """Make . match newlines with re.S."""
    text = "<div>\nHello\n</div>"
    match_default = re.search(r'<div>(.+)</div>', text)
    match_dotall = re.search(r'<div>(.+)</div>', text, re.S)
    print(f"Default: {match_default}")
    print(f"DOTALL:  {match_dotall.group(1) if match_dotall else None}")


def verbose_flag():
    """Write readable patterns with re.X."""
    pattern = re.compile(r"""
        ^                       # Start
        [\w.+-]+                # Local part
        @                       # @ symbol
        [\w-]+                  # Domain
        (?:\.[\w-]+)*           # Subdomains
        \.[a-zA-Z]{2,}          # TLD
        $                       # End
    """, re.VERBOSE)

    for email in ["user@example.com", "bad@", "@missing.com"]:
        valid = bool(pattern.match(email))
        print(f"  '{email}': {'Valid' if valid else 'Invalid'}")


def combining_flags():
    """Combine flags with the | operator."""
    text = "Hello World\nhello python\nHELLO REGEX"
    matches = re.findall(r'^hello\b', text, re.I | re.M)
    print(f"I + M: {matches}")


def inline_flags():
    """Use inline flags within patterns."""
    print(f"(?i)python: {re.findall(r'(?i)python', 'Python PYTHON')}")
    print(f"(?m)^Line:  {re.findall(r'(?m)^Line', 'Line 1\nLine 2')}")

    # Scoped inline flag
    pattern = r'Hello (?i:world)'
    print(f"\nScoped (?i:world):")
    print(f"  'Hello WORLD': {bool(re.search(pattern, 'Hello WORLD'))}")
    print(f"  'HELLO WORLD': {bool(re.search(pattern, 'HELLO WORLD'))}")


def practical_verbose_pattern():
    """Real-world VERBOSE pattern for log parsing."""
    log = "2024-01-15 08:30:45 ERROR Connection failed\n2024-01-15 08:30:46 INFO Retrying"

    pattern = re.compile(r"""
        ^(\d{4}-\d{2}-\d{2})   # Date
        \s+(\d{2}:\d{2}:\d{2}) # Time
        \s+(ERROR|WARN)         # Level
        \s+(.+)$               # Message
    """, re.X | re.M)

    for match in pattern.finditer(log):
        date, time, level, msg = match.groups()
        print(f"  [{level}] {date} {time}: {msg}")


if __name__ == "__main__":
    sections = [
        ("IGNORECASE", ignorecase_flag),
        ("MULTILINE", multiline_flag),
        ("DOTALL", dotall_flag),
        ("VERBOSE", verbose_flag),
        ("Combining Flags", combining_flags),
        ("Inline Flags", inline_flags),
        ("Practical VERBOSE", practical_verbose_pattern),
    ]

    for title, func in sections:
        print(f"\n{'=' * 50}")
        print(f" {title}")
        print('=' * 50)
        func()
