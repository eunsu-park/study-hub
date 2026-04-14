"""
Exercise 09: Flags and Options

Practice re.IGNORECASE, re.MULTILINE, re.DOTALL, re.VERBOSE,
and combining flags.
"""

import re


def case_insensitive_count(text, word):
    """Count occurrences of a word regardless of case.

    Args:
        text: Input string.
        word: Word to count.

    Returns:
        Integer count.
    """
    # TODO: Implement this using re.IGNORECASE
    pass


def extract_error_lines(log_text):
    """Extract all lines that contain 'ERROR' (case-sensitive).

    Args:
        log_text: Multi-line log string.

    Returns:
        List of full error lines.
    """
    # TODO: Implement this using re.MULTILINE
    pass


def extract_multiline_blocks(text, open_tag, close_tag):
    """Extract content between open_tag and close_tag, even across lines.

    Args:
        text: Input string (may contain newlines).
        open_tag: Opening tag string.
        close_tag: Closing tag string.

    Returns:
        List of content strings between tags.
    """
    # TODO: Implement this using re.DOTALL
    pass


def remove_comments(config_text):
    """Remove lines that start with # (comment lines).

    Args:
        config_text: Multi-line configuration text.

    Returns:
        Text with comment lines removed.
    """
    # TODO: Implement this using re.MULTILINE
    pass


def create_verbose_email_pattern():
    """Create a verbose email validation pattern.

    Returns:
        Compiled regex pattern that validates basic email format.
    """
    # TODO: Implement this using re.VERBOSE
    # Pattern should match: local@domain.tld
    pass


# === Tests ===

assert case_insensitive_count("Python PYTHON python", "python") == 3
assert case_insensitive_count("Hello World", "missing") == 0

log = "INFO: ok\nERROR: fail\nINFO: retry\nERROR: timeout"
assert extract_error_lines(log) == ["ERROR: fail", "ERROR: timeout"]

html = "<div>\nHello\nWorld\n</div>\n<div>\nTest\n</div>"
blocks = extract_multiline_blocks(html, "<div>", "</div>")
assert len(blocks) == 2
assert "Hello" in blocks[0]

config = "# comment\nhost=localhost\n# another\nport=5432"
clean = remove_comments(config)
assert "# comment" not in clean
assert "host=localhost" in clean

email_pat = create_verbose_email_pattern()
assert email_pat.match("user@example.com")
assert not email_pat.match("@bad.com")

print("All tests passed!")
