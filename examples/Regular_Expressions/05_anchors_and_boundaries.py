"""
05 Anchors and Boundaries
==========================
Demonstrates ^, $, \\b, \\B, \\A, \\Z, and re.MULTILINE.
"""

import re


def start_end_anchors():
    """Use ^ and $ for position matching."""
    text = "Python is great"
    print(f"^Python: {bool(re.search(r'^Python', text))}")
    print(f"^is:     {bool(re.search(r'^is', text))}")
    print(f"great$:  {bool(re.search(r'great$', text))}")


def multiline_mode():
    """Demonstrate re.MULTILINE for line-by-line anchors."""
    text = "Line 1: Hello\nLine 2: World\nLine 3: Python"

    print(f"Without MULTILINE: {re.findall(r'^Line', text)}")
    print(f"With MULTILINE:    {re.findall(r'^Line', text, re.M)}")
    print(f"Last words:        {re.findall(r'w+$', text, re.M)}")


def word_boundaries():
    """Use \\b to match whole words."""
    text = "cat scatter category caterpillar"

    print(f"'cat' anywhere: {re.findall(r'cat', text)}")
    print(f"'cat' as word:  {re.findall(r'\\bcat\\b', text)}")

    text2 = "I love JavaScript, not just Java"
    print(f"\n'Java' as word: {re.findall(r'\\bJava\\b', text2)}")

    text3 = "preview, preprocess, present, compress"
    print(f"Words with 'pre': {re.findall(r'\\bpre\\w+', text3)}")


def non_word_boundaries():
    """Use \\B to match inside words."""
    text = "cat scatter category"
    print(f"\\Bcat\\B: {re.findall(r'\\Bcat\\B', text)}")
    print(f"\\Bcat:   {re.findall(r'\\Bcat', text)}")


def absolute_anchors():
    """Demonstrate \\A and \\Z vs ^ and $."""
    text = "First line\nSecond line\nThird line"

    print(f"^\\w+ (MULTILINE): {re.findall(r'^\\w+', text, re.M)}")
    print(f"\\A\\w+ (MULTILINE): {re.findall(r'\\A\\w+', text, re.M)}")
    print(f"\\w+\\Z (MULTILINE): {re.findall(r'\\w+\\Z', text, re.M)}")


def practical_examples():
    """Real-world anchor usage."""
    # Find blank lines
    text = "Line 1\n\nLine 3\n\nLine 5"
    blanks = len(re.findall(r'^$', text, re.M))
    print(f"Blank lines: {blanks}")

    # Strip trailing whitespace
    code = "def hello():   \n    pass  \n"
    clean = re.sub(r'[ \t]+$', '', code, flags=re.M)
    print(f"Cleaned: {repr(clean)}")

    # Validate username
    for name in ["alice", "Bob_99", "ab", "1alice"]:
        valid = bool(re.fullmatch(r'[a-zA-Z]\w{2,15}', name))
        print(f"  Username '{name}': {'Valid' if valid else 'Invalid'}")


if __name__ == "__main__":
    sections = [
        ("Start/End Anchors", start_end_anchors),
        ("Multiline Mode", multiline_mode),
        ("Word Boundaries", word_boundaries),
        ("Non-Word Boundaries", non_word_boundaries),
        ("Absolute Anchors", absolute_anchors),
        ("Practical Examples", practical_examples),
    ]

    for title, func in sections:
        print(f"\n{'=' * 50}")
        print(f" {title}")
        print('=' * 50)
        func()
