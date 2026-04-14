"""
Exercise 12: Real-World Applications

Practice log parsing, data cleaning, config parsing, and tokenization.
"""

import re


def parse_log_entries(log_text):
    """Parse log entries in format: [YYYY-MM-DD HH:MM:SS] LEVEL message

    Args:
        log_text: Multi-line log string.

    Returns:
        List of dicts with keys: 'datetime', 'level', 'message'.
    """
    # TODO: Implement this
    pass


def clean_and_normalize(text):
    """Clean text by:
    1. Removing HTML tags
    2. Collapsing multiple whitespace to single space
    3. Stripping leading/trailing whitespace

    Args:
        text: Dirty text string.

    Returns:
        Clean text string.
    """
    # TODO: Implement this
    pass


def parse_ini_config(ini_text):
    """Parse INI-style configuration into a nested dict.

    Handle [section] headers and key=value pairs.
    Ignore comment lines (starting with # or ;).
    Ignore empty lines.

    Args:
        ini_text: INI format configuration string.

    Returns:
        Dict of {section: {key: value}}.
    """
    # TODO: Implement this
    pass


def tokenize_expression(expr):
    """Tokenize a simple arithmetic expression.

    Token types: NUMBER (int/float), OP (+,-,*,/), LPAREN, RPAREN.
    Skip whitespace.

    Args:
        expr: Expression string.

    Returns:
        List of (type, value) tuples.
    """
    # TODO: Implement this
    pass


def batch_rename_files(filenames, pattern, replacement):
    """Apply regex substitution to a list of filenames.

    Args:
        filenames: List of filename strings.
        pattern: Regex pattern to match.
        replacement: Replacement string (may use group references).

    Returns:
        List of (old_name, new_name) tuples for changed files only.
    """
    # TODO: Implement this
    pass


# === Tests ===

log = "[2024-01-15 08:30:45] ERROR Connection failed\n[2024-01-15 08:30:46] INFO Retrying"
entries = parse_log_entries(log)
assert len(entries) == 2
assert entries[0]['level'] == 'ERROR'
assert entries[0]['message'] == 'Connection failed'
assert entries[1]['datetime'] == '2024-01-15 08:30:46'

assert clean_and_normalize("  <p>Hello   World</p>  ") == "Hello World"
assert clean_and_normalize("<b>bold</b>  text") == "bold text"

ini = "[db]\nhost = localhost\nport = 5432\n# comment\n[server]\nport = 8080"
config = parse_ini_config(ini)
assert config['db']['host'] == 'localhost'
assert config['db']['port'] == '5432'
assert config['server']['port'] == '8080'

tokens = tokenize_expression("3 + 4.5 * (2 - 1)")
assert ('NUMBER', '3') in tokens or ('NUMBER', 3) in tokens
assert ('OP', '+') in tokens
assert ('LPAREN', '(') in tokens

renames = batch_rename_files(
    ["IMG_20240115.jpg", "IMG_20240116.jpg", "README.md"],
    r'IMG_(\d{4})(\d{2})(\d{2})',
    r'\1-\2-\3'
)
assert len(renames) == 2
assert renames[0] == ("IMG_20240115.jpg", "2024-01-15.jpg")

print("All tests passed!")
