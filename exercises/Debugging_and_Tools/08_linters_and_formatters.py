"""
Exercise 08: Linters and Formatters

Practice fixing linter warnings and configuring tools.
"""
import os
import sys
import json


# --- Exercise 1: Fix all linter warnings in this code ---

def fix_linter_warnings():
    """Fix all the linter warnings in the code below.

    Common warnings to fix:
    - Unused imports (F401)
    - Unused variables (F841)
    - Comparison to None (E711)
    - Bare except (E722)
    - Missing function calls

    Returns:
        dict: Results of the fixed operations.
    """
    # TODO: Fix all linter warnings below

    # Warning: unused variable (F841)
    unused_result = 42

    # Warning: comparison to None (E711)
    value = None
    if value == None:
        value = "default"

    # Warning: bare except (E722)
    try:
        number = int("123")
    except:
        number = 0

    # Warning: statement with no effect (W0104)
    data = [1, 2, 3]
    data.sort
    # Should be: data.sort()

    return {"value": value, "number": number, "data": data}


# --- Exercise 2: Write a pyproject.toml configuration ---

def generate_ruff_config():
    """Generate a ruff configuration as a string.

    Create a pyproject.toml configuration for ruff with:
    - Line length: 100
    - Target Python version: 3.12
    - Selected rules: E, W, F, I, N, UP
    - Ignored rules: E501
    - Per-file ignore: F401 for __init__.py

    Returns:
        str: The pyproject.toml content for ruff configuration.
    """
    # TODO: Return a valid pyproject.toml string for ruff
    pass


# --- Exercise 3: Identify linter violations ---

def identify_violations(code_lines):
    """Identify the type of linter violation in each line.

    Args:
        code_lines: A list of code strings, each with one violation.

    Returns:
        list: A list of violation codes (e.g., "F401", "E711").
    """
    # TODO: For each line, return the likely linter violation code
    # Possible codes: F401 (unused import), F841 (unused variable),
    # E711 (compare to None), E722 (bare except), E501 (line too long)
    pass


# --- Exercise 4: Create a pre-commit config ---

def generate_precommit_config():
    """Generate a .pre-commit-config.yaml as a string.

    Include:
    - ruff (lint + format)
    - trailing-whitespace hook
    - end-of-file-fixer hook

    Returns:
        str: The YAML content for .pre-commit-config.yaml.
    """
    # TODO: Return a valid pre-commit config YAML string
    pass


if __name__ == "__main__":
    # Test fix_linter_warnings
    result = fix_linter_warnings()
    assert result["value"] == "default"
    assert result["number"] == 123
    assert result["data"] == [1, 2, 3]  # sorted
    print("fix_linter_warnings: PASSED")

    # Test generate_ruff_config
    config = generate_ruff_config()
    assert config is not None, "Should return config string"
    assert "ruff" in config
    assert "100" in config
    assert "py312" in config or "3.12" in config
    print("generate_ruff_config: PASSED")

    # Test identify_violations
    code_lines = [
        'import os  # but os is never used',
        'result = expensive()  # result never read',
        'if x == None:',
        'except:',
        'x = "a" * 200  # very long line',
    ]
    violations = identify_violations(code_lines)
    assert violations is not None
    assert len(violations) == 5
    assert violations[0] == "F401"
    assert violations[2] == "E711"
    print("identify_violations: PASSED")

    # Test generate_precommit_config
    config = generate_precommit_config()
    assert config is not None
    assert "ruff" in config
    assert "trailing-whitespace" in config
    print("generate_precommit_config: PASSED")
