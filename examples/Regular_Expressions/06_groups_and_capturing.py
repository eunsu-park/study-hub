"""
06 Groups and Capturing
========================
Demonstrates capture groups, named groups, non-capturing groups,
backreferences, and group numbering.
"""

import re


def capture_groups():
    """Extract parts of a match using capture groups."""
    text = "Date: 2024-01-15"
    match = re.search(r'(\d{4})-(\d{2})-(\d{2})', text)
    if match:
        print(f"Full:  {match.group()}")
        print(f"Year:  {match.group(1)}")
        print(f"Month: {match.group(2)}")
        print(f"Day:   {match.group(3)}")
        print(f"All:   {match.groups()}")


def named_groups():
    """Use named groups for readable pattern extraction."""
    text = "2024-01-15 08:30:45"
    pattern = r'(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})\s+(?P<hour>\d{2}):(?P<min>\d{2}):(?P<sec>\d{2})'
    match = re.search(pattern, text)
    if match:
        print(f"Year:  {match.group('year')}")
        print(f"Dict:  {match.groupdict()}")

    # Named groups in substitution
    result = re.sub(
        r'(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})',
        r'\g<d>/\g<m>/\g<y>',
        "2024-01-15"
    )
    print(f"Reformatted: {result}")


def non_capturing_groups():
    """Use (?:...) for grouping without capture."""
    text = "gray grey"
    print(f"Capturing:     {re.findall(r'gr(a|e)y', text)}")
    print(f"Non-capturing: {re.findall(r'gr(?:a|e)y', text)}")

    text2 = "http://a.com https://b.com"
    print(f"\nCapturing:     {re.findall(r'(https?)://(\\S+)', text2)}")
    print(f"Non-capturing: {re.findall(r'(?:https?)://\\S+', text2)}")


def backreferences():
    """Match repeated text with backreferences."""
    # Find doubled words
    text = "the the cat sat sat on the mat"
    dupes = re.findall(r'\b(\w+)\s+\1\b', text)
    print(f"Doubled words: {dupes}")

    # Match HTML tags
    html = "<b>bold</b> <i>italic</i> <b>broken</i>"
    tags = re.findall(r'<(\w+)>.*?</\1>', html)
    print(f"Matched tags: {tags}")

    # Repeated characters
    text2 = "aardvark bookkeeper"
    print(f"Repeated chars: {re.findall(r'(.)\\1', text2)}")


def findall_with_groups():
    """Understand findall behavior with groups."""
    text = "2024-01-15 and 2024-12-31"

    print(f"No groups:  {re.findall(r'\\d{4}-\\d{2}-\\d{2}', text)}")
    print(f"One group:  {re.findall(r'(\\d{4})-\\d{2}-\\d{2}', text)}")
    print(f"Two groups: {re.findall(r'(\\d{4})-(\\d{2})-(\\d{2})', text)}")


def practical_examples():
    """Real-world group usage."""
    # Parse key-value pairs
    config = "host=localhost\nport=5432\ndb=myapp"
    pairs = dict(re.findall(r'^(\w+)=(.+)$', config, re.M))
    print(f"Config: {pairs}")

    # Swap names
    names = "Smith, John\nDoe, Jane"
    swapped = re.sub(r'(\w+),\s*(\w+)', r'\2 \1', names)
    print(f"Swapped:\n{swapped}")


if __name__ == "__main__":
    sections = [
        ("Capture Groups", capture_groups),
        ("Named Groups", named_groups),
        ("Non-Capturing Groups", non_capturing_groups),
        ("Backreferences", backreferences),
        ("Findall with Groups", findall_with_groups),
        ("Practical Examples", practical_examples),
    ]

    for title, func in sections:
        print(f"\n{'=' * 50}")
        print(f" {title}")
        print('=' * 50)
        func()
