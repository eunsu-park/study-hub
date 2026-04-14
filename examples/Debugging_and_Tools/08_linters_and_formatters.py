"""
08 Linters and Formatters
=========================
Demonstrates code that would trigger linter warnings and shows
the correct patterns. Run with: ruff check 08_linters_and_formatters.py
"""


# --- Good patterns (no warnings) ---

def clean_code_examples():
    """Examples of clean code that passes all linters."""
    # Proper None comparison (not E711)
    value = None
    if value is None:
        print("Value is None")

    # Proper exception handling (not E722)
    try:
        result = int("42")
    except ValueError:
        print("Invalid number")

    # No unused variables (not F841)
    result = compute_something(10)
    print(f"Result: {result}")

    # No unused imports (F401 would flag unused ones)
    import math
    print(f"Pi: {math.pi}")


def compute_something(x):
    """A simple function with proper docstring."""
    return x * 2 + 1


# --- Common warnings demo ---

def linter_warning_catalog():
    """Catalog of common linter warnings with examples."""
    print("=== Common Linter Warnings ===")

    warnings = [
        ("F401", "Unused import",
         "import os  # but os is never used"),
        ("F841", "Unused variable",
         "result = expensive()  # but result is never read"),
        ("E711", "Comparison to None",
         "if x == None:  # should be: if x is None:"),
        ("E722", "Bare except",
         "except:  # should be: except Exception:"),
        ("E501", "Line too long",
         "x = very_long_function(arg1, arg2, arg3, arg4, arg5, arg6)"),
        ("W291", "Trailing whitespace",
         "x = 42   # trailing spaces after this"),
        ("E302", "Expected 2 blank lines",
         "# Missing blank line before function definition"),
        ("E401", "Multiple imports on one line",
         "import os, sys  # should be separate lines"),
    ]

    for code, name, example in warnings:
        print(f"  {code} - {name}")
        print(f"    Example: {example}")
        print()


def formatting_before_after():
    """Show before/after formatting examples."""
    print("=== Before/After Formatting ===")

    examples = [
        (
            "Inconsistent quotes",
            """x = 'hello'  ; y = "world" """,
            """x = "hello"\ny = "world" """,
        ),
        (
            "Missing spaces",
            "x=1+2*3",
            "x = 1 + 2 * 3",
        ),
        (
            "Trailing comma",
            "items = [1, 2, 3]  # short list",
            "items = [\n    1,\n    2,\n    3,\n]  # long list with trailing comma",
        ),
    ]

    for name, before, after in examples:
        print(f"  {name}:")
        print(f"    Before: {before}")
        print(f"    After:  {after}")
        print()


def tool_comparison():
    """Show a comparison of linting tools."""
    print("=== Tool Comparison ===")
    print(f"  {'Tool':<12} {'Speed':<10} {'Auto-fix':<10} {'Format':<10}")
    print(f"  {'-'*42}")
    tools = [
        ("pylint", "Slow", "No", "No"),
        ("flake8", "Medium", "No", "No"),
        ("ruff", "Very fast", "Yes", "Yes"),
        ("black", "Fast", "N/A", "Yes"),
    ]
    for name, speed, fix, fmt in tools:
        print(f"  {name:<12} {speed:<10} {fix:<10} {fmt:<10}")
    print()
    print("  Recommendation: Start with ruff (replaces all others)")
    print()


def pyproject_toml_example():
    """Show example pyproject.toml configuration."""
    print("=== pyproject.toml Example ===")
    config = '''
[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "W", "F", "I", "N", "UP"]
ignore = ["E501"]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]

[tool.black]
line-length = 100
'''
    print(config)


if __name__ == "__main__":
    clean_code_examples()
    linter_warning_catalog()
    formatting_before_after()
    tool_comparison()
    pyproject_toml_example()
