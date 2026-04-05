# Unit Testing with pytest

**Previous**: [Testing Fundamentals](./01_Testing_Fundamentals.md) | **Next**: [Test Fixtures and Parameterization](./03_Test_Fixtures_and_Parameterization.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Write and run test functions using pytest conventions
2. Leverage pytest's assertion introspection for clear failure messages
3. Test for expected exceptions with `pytest.raises`
4. Control test execution with marks, skip, and xfail
5. Capture and assert on printed output and logged messages

---

## Why pytest?

Python ships with `unittest` in the standard library, but pytest has become the de facto standard for Python testing. Here is why:

- **Plain functions** — No need to subclass `TestCase` or remember `self.assertEqual` methods
- **Assertion introspection** — Write plain `assert` statements; pytest rewrites them to show detailed failure info
- **Rich plugin ecosystem** — Over 800 plugins for everything from parallel execution to snapshot testing
- **Fixture system** — Powerful, composable dependency injection (covered in Lesson 03)
- **Better output** — Color-coded, concise, configurable

```python
# unittest style — verbose
import unittest

class TestMath(unittest.TestCase):
    def test_add(self):
        self.assertEqual(1 + 1, 2)

# pytest style — clean
def test_add():
    assert 1 + 1 == 2
```

Install pytest:

```bash
pip install pytest
```

---

## Writing Your First Tests

### Test Discovery Rules

pytest discovers tests automatically using these conventions:

1. Files named `test_*.py` or `*_test.py`
2. Functions prefixed with `test_`
3. Classes prefixed with `Test` (no `__init__` method)
4. Methods in test classes prefixed with `test_`

```python
# test_strings.py

def test_upper():
    assert "hello".upper() == "HELLO"

def test_strip():
    assert "  spaces  ".strip() == "spaces"

class TestSplit:
    def test_simple_split(self):
        assert "a,b,c".split(",") == ["a", "b", "c"]

    def test_split_with_limit(self):
        assert "a,b,c".split(",", 1) == ["a", "b,c"]
```

Run tests:

```bash
# Run all tests in current directory (recursive)
pytest

# Run a specific file
pytest test_strings.py

# Run a specific test function
pytest test_strings.py::test_upper

# Run a specific test class method
pytest test_strings.py::TestSplit::test_simple_split

# Verbose output
pytest -v

# Short traceback
pytest --tb=short

# Stop on first failure
pytest -x

# Run last failed tests only
pytest --lf
```

---

## Assertion Introspection

pytest rewrites `assert` statements at import time to provide rich failure messages. You do not need special assertion methods.

```python
def test_list_equality():
    expected = [1, 2, 3, 4, 5]
    actual = [1, 2, 3, 4, 6]
    assert actual == expected
```

Failure output:

```
FAILED test_example.py::test_list_equality
    assert actual == expected
E    AssertionError: assert [1, 2, 3, 4, 6] == [1, 2, 3, 4, 5]
E      At index 4 diff: 6 != 5
```

### Common Assertion Patterns

```python
def test_equality():
    assert calculate_total([10, 20, 30]) == 60

def test_truthiness():
    assert is_valid_email("user@example.com")

def test_membership():
    result = get_supported_formats()
    assert "pdf" in result

def test_identity():
    singleton_a = get_config()
    singleton_b = get_config()
    assert singleton_a is singleton_b

def test_approximate_equality():
    # For floating-point comparisons
    assert 0.1 + 0.2 == pytest.approx(0.3)
    assert calculate_pi() == pytest.approx(3.14159, rel=1e-3)

def test_string_contains():
    error_msg = get_error_message(404)
    assert "not found" in error_msg.lower()

def test_length():
    items = fetch_page(page=1, size=10)
    assert len(items) == 10

def test_type():
    result = parse_config("app.yaml")
    assert isinstance(result, dict)
```

---

## Testing Exceptions with pytest.raises

Use `pytest.raises` as a context manager to verify that code raises the expected exception.

```python
import pytest

def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Divisor cannot be zero")
    return a / b


def test_divide_by_zero_raises_value_error():
    with pytest.raises(ValueError):
        divide(10, 0)

def test_divide_by_zero_error_message():
    with pytest.raises(ValueError, match="Divisor cannot be zero"):
        divide(10, 0)

def test_divide_by_zero_inspect_exception():
    with pytest.raises(ValueError) as exc_info:
        divide(10, 0)
    assert "zero" in str(exc_info.value)
    assert exc_info.type is ValueError
```

### Testing Multiple Exception Types

```python
def parse_age(value: str) -> int:
    if not value:
        raise ValueError("Age cannot be empty")
    age = int(value)  # May raise ValueError for non-numeric
    if age < 0:
        raise ValueError("Age cannot be negative")
    if age > 150:
        raise ValueError("Age seems unrealistic")
    return age


def test_empty_age():
    with pytest.raises(ValueError, match="empty"):
        parse_age("")

def test_negative_age():
    with pytest.raises(ValueError, match="negative"):
        parse_age("-5")

def test_non_numeric_age():
    with pytest.raises(ValueError):
        parse_age("abc")

def test_unrealistic_age():
    with pytest.raises(ValueError, match="unrealistic"):
        parse_age("200")
```

---

## Organizing Tests with Classes

While pytest works great with plain functions, classes help group related tests:

```python
class TestShoppingCart:
    """Tests for the ShoppingCart class."""

    def test_new_cart_is_empty(self):
        cart = ShoppingCart()
        assert len(cart.items) == 0

    def test_add_item(self):
        cart = ShoppingCart()
        cart.add("apple", quantity=3, price=1.50)
        assert len(cart.items) == 1

    def test_total_with_multiple_items(self):
        cart = ShoppingCart()
        cart.add("apple", quantity=2, price=1.50)
        cart.add("bread", quantity=1, price=3.00)
        assert cart.total() == 6.00

    def test_remove_item(self):
        cart = ShoppingCart()
        cart.add("apple", quantity=1, price=1.50)
        cart.remove("apple")
        assert len(cart.items) == 0

    def test_remove_nonexistent_item_raises(self):
        cart = ShoppingCart()
        with pytest.raises(KeyError):
            cart.remove("banana")
```

Note: Unlike `unittest.TestCase`, pytest test classes should **not** have an `__init__` method. Use fixtures (Lesson 03) instead of `setUp`/`tearDown`.

---

## Marks: Categorizing and Controlling Tests

Marks are decorators that attach metadata to tests. pytest uses marks to filter, skip, or modify test behavior.

### Built-in Marks

```python
import pytest
import sys

# Skip unconditionally
@pytest.mark.skip(reason="Feature not implemented yet")
def test_future_feature():
    assert fancy_new_thing() == 42

# Skip conditionally
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Unix-only test"
)
def test_unix_permissions():
    import os
    assert os.access("/tmp", os.W_OK)

# Expected failure — test is known to fail
@pytest.mark.xfail(reason="Bug #1234 not yet fixed")
def test_known_bug():
    assert broken_function() == "correct"

# Strict xfail — fail the test if it unexpectedly passes
@pytest.mark.xfail(strict=True, reason="Should fail until fix is merged")
def test_strict_known_bug():
    assert broken_function() == "correct"
```

### Custom Marks

Define your own marks to categorize tests:

```python
# conftest.py or pyproject.toml to register marks
# [tool.pytest.ini_options]
# markers = [
#     "slow: marks tests as slow (deselect with '-m \"not slow\"')",
#     "integration: marks tests requiring external services",
# ]

@pytest.mark.slow
def test_large_dataset_processing():
    data = generate_large_dataset(size=1_000_000)
    result = process(data)
    assert len(result) > 0

@pytest.mark.integration
def test_database_connection():
    conn = connect_to_db()
    assert conn.is_alive()
```

Run with mark filters:

```bash
# Run only slow tests
pytest -m slow

# Run everything EXCEPT slow tests
pytest -m "not slow"

# Run tests marked slow OR integration
pytest -m "slow or integration"
```

---

## Output Capture

pytest captures `stdout` and `stderr` by default. It only shows output for failing tests. You can explicitly test printed output using the `capsys` fixture.

```python
def greet(name: str) -> None:
    """Print a greeting message."""
    print(f"Hello, {name}!")


def test_greet_output(capsys):
    greet("World")
    captured = capsys.readouterr()
    assert captured.out == "Hello, World!\n"
    assert captured.err == ""


def warn_user(message: str) -> None:
    """Print a warning to stderr."""
    import sys
    print(f"WARNING: {message}", file=sys.stderr)


def test_warn_output(capsys):
    warn_user("disk almost full")
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "disk almost full" in captured.err
```

### Controlling Capture Behavior

```bash
# Disable capture (see all print output in real time)
pytest -s

# Equivalent
pytest --capture=no

# Default (capture both stdout and stderr)
pytest --capture=fd
```

---

## Testing Logging Output

Use the `caplog` fixture to test log messages:

```python
import logging

logger = logging.getLogger(__name__)

def process_order(order_id: int) -> str:
    logger.info(f"Processing order {order_id}")
    if order_id <= 0:
        logger.error(f"Invalid order ID: {order_id}")
        raise ValueError("Order ID must be positive")
    logger.info(f"Order {order_id} processed successfully")
    return "success"


def test_process_order_logs(caplog):
    with caplog.at_level(logging.INFO):
        result = process_order(42)

    assert result == "success"
    assert "Processing order 42" in caplog.text
    assert "processed successfully" in caplog.text


def test_invalid_order_logs_error(caplog):
    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError):
            process_order(-1)

    assert "Invalid order ID: -1" in caplog.text
```

---

## Practical Example: Testing a Password Validator

Bringing together multiple pytest features in a realistic scenario:

```python
# password_validator.py
import re
from dataclasses import dataclass


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str]


def validate_password(password: str) -> ValidationResult:
    """Validate a password against security rules."""
    errors = []

    if len(password) < 8:
        errors.append("Password must be at least 8 characters")
    if len(password) > 128:
        errors.append("Password must be at most 128 characters")
    if not re.search(r"[A-Z]", password):
        errors.append("Password must contain an uppercase letter")
    if not re.search(r"[a-z]", password):
        errors.append("Password must contain a lowercase letter")
    if not re.search(r"[0-9]", password):
        errors.append("Password must contain a digit")
    if not re.search(r"[!@#$%^&*]", password):
        errors.append("Password must contain a special character")

    return ValidationResult(is_valid=len(errors) == 0, errors=errors)
```

```python
# test_password_validator.py
import pytest
from password_validator import validate_password


class TestValidPassword:
    def test_strong_password_is_valid(self):
        result = validate_password("MyStr0ng!Pass")
        assert result.is_valid is True
        assert result.errors == []

    def test_minimum_valid_password(self):
        result = validate_password("Aa1!xxxx")
        assert result.is_valid is True


class TestPasswordLength:
    def test_too_short(self):
        result = validate_password("Aa1!")
        assert result.is_valid is False
        assert "at least 8" in result.errors[0]

    def test_too_long(self):
        result = validate_password("A" * 129 + "a1!")
        assert result.is_valid is False
        assert any("at most 128" in e for e in result.errors)

    def test_exactly_eight_characters(self):
        result = validate_password("Aa1!xxxx")
        assert result.is_valid is True


class TestPasswordCharacterRequirements:
    def test_missing_uppercase(self):
        result = validate_password("mystr0ng!pass")
        assert not result.is_valid
        assert any("uppercase" in e for e in result.errors)

    def test_missing_lowercase(self):
        result = validate_password("MYSTR0NG!PASS")
        assert not result.is_valid
        assert any("lowercase" in e for e in result.errors)

    def test_missing_digit(self):
        result = validate_password("MyStrong!Pass")
        assert not result.is_valid
        assert any("digit" in e for e in result.errors)

    def test_missing_special_character(self):
        result = validate_password("MyStr0ngPass1")
        assert not result.is_valid
        assert any("special" in e for e in result.errors)


class TestMultipleErrors:
    def test_empty_password_reports_all_errors(self):
        result = validate_password("")
        assert not result.is_valid
        assert len(result.errors) >= 4  # length + missing char types
```

---

## Useful Command-Line Options

```bash
# Show slowest N tests
pytest --durations=10

# Run tests matching a keyword expression
pytest -k "password and not slow"

# Run tests in parallel (requires pytest-xdist)
pytest -n auto

# Generate JUnit XML report (for CI)
pytest --junitxml=results.xml

# Show local variables in tracebacks
pytest -l

# Drop into debugger on failure
pytest --pdb

# Collect tests without running them
pytest --collect-only

# Show print output for all tests (not just failures)
pytest -s -v
```

---

## Exercises

1. **Write unit tests**: Create a `temperature.py` module with functions `celsius_to_fahrenheit(c)` and `fahrenheit_to_celsius(f)`. Write at least 6 tests covering normal values, freezing point, boiling point, and negative temperatures. Use `pytest.approx` for floating-point comparisons.

2. **Test exceptions**: Write a function `parse_rgb(color_string)` that parses strings like `"rgb(255, 128, 0)"` and returns a tuple `(255, 128, 0)`. It should raise `ValueError` for invalid formats and values outside 0-255. Write tests using `pytest.raises` with `match` patterns.

3. **Mark and filter**: Create a test file with at least 8 tests. Mark some as `@pytest.mark.slow` and some as `@pytest.mark.smoke`. Practice running different subsets using `-m` expressions.

---

**License**: CC BY-NC 4.0
