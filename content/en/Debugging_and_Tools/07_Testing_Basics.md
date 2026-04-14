# Testing Basics

**Previous**: [Logging](./06_Logging.md) | **Next**: [Linters and Formatters](./08_Linters_and_Formatters.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why testing is a debugging tool, not just a quality gate
2. Write effective `assert` statements to validate assumptions
3. Create test functions using `pytest` conventions
4. Organize tests with test files, test classes, and fixtures
5. Use `unittest.TestCase` with assertion methods
6. Apply test-driven debugging: write a failing test first, then fix the bug
7. Run tests from the command line and interpret the output
8. Understand test coverage and its relationship to bug detection

---

Testing is one of the most powerful debugging tools available. When you encounter a bug, the first step should be to write a test that demonstrates the bug -- a test that fails. Then fix the code until the test passes. This approach, called **test-driven debugging**, ensures you truly understand the bug, confirms you've fixed it, and prevents it from ever returning.

> **Key Insight:** A bug without a test is a bug waiting to return. Every bug you fix is an opportunity to add a test that makes your codebase permanently stronger.

---

## 1. The `assert` Statement

### 1.1 Basic Assert

The simplest form of testing:

```python
def add(a, b):
    return a + b

# Test with assert
assert add(2, 3) == 5
assert add(-1, 1) == 0
assert add(0, 0) == 0
print("All tests passed!")
```

If any assertion fails, Python raises `AssertionError`:

```python
assert add(2, 3) == 6
# AssertionError
```

### 1.2 Assert with Messages

Always include a descriptive message:

```python
assert add(2, 3) == 5, f"Expected 5, got {add(2, 3)}"
assert len(result) > 0, "Result should not be empty"
assert isinstance(value, int), f"Expected int, got {type(value).__name__}"
```

### 1.3 When to Use Raw Assert

```python
# Inline sanity checks during development
def process(data):
    assert data is not None, "data must not be None"
    assert len(data) > 0, "data must not be empty"
    
    result = transform(data)
    assert isinstance(result, list), f"transform() should return list, got {type(result)}"
    
    return result
```

**Warning**: `assert` statements are removed when Python runs with `-O` (optimize) flag. Never use `assert` for input validation in production code -- use `if`/`raise` instead.

---

## 2. pytest: The Standard Test Framework

### 2.1 Writing Your First Test

```python
# file: test_calculator.py
def add(a, b):
    return a + b

def test_add_positive_numbers():
    assert add(2, 3) == 5

def test_add_negative_numbers():
    assert add(-1, -2) == -3

def test_add_zero():
    assert add(5, 0) == 5
    assert add(0, 5) == 5
```

### 2.2 Running Tests

```bash
# Install pytest
pip install pytest

# Run all tests in current directory
pytest

# Run specific file
pytest test_calculator.py

# Run with verbose output
pytest -v

# Run specific test function
pytest test_calculator.py::test_add_positive_numbers

# Stop at first failure
pytest -x
```

### 2.3 Test Output

```
$ pytest -v
========================= test session starts ==========================
test_calculator.py::test_add_positive_numbers PASSED
test_calculator.py::test_add_negative_numbers PASSED
test_calculator.py::test_add_zero PASSED
========================== 3 passed in 0.01s ===========================
```

When a test fails:

```
FAILED test_calculator.py::test_add_positive_numbers - AssertionError:
    assert 4 == 5
    +  where 4 = add(2, 2)
```

pytest shows the exact values that were compared, which is extremely helpful for debugging.

---

## 3. Organizing Tests

### 3.1 File Structure

```
project/
├── calculator.py          # Source code
├── validator.py
├── tests/
│   ├── test_calculator.py # Tests for calculator.py
│   └── test_validator.py  # Tests for validator.py
```

### 3.2 Test Classes

```python
# test_calculator.py
class TestCalculator:
    def test_add(self):
        assert add(2, 3) == 5

    def test_subtract(self):
        assert subtract(5, 3) == 2

    def test_divide(self):
        assert divide(10, 2) == 5

    def test_divide_by_zero(self):
        import pytest
        with pytest.raises(ZeroDivisionError):
            divide(10, 0)
```

### 3.3 Testing Exceptions

```python
import pytest

def test_invalid_input():
    with pytest.raises(ValueError):
        int("not_a_number")

def test_exception_message():
    with pytest.raises(ValueError, match="invalid literal"):
        int("abc")
```

---

## 4. pytest Fixtures

Fixtures provide setup and teardown for tests:

```python
import pytest

@pytest.fixture
def sample_data():
    """Provide test data."""
    return [1, 2, 3, 4, 5]

@pytest.fixture
def empty_list():
    return []

def test_sum(sample_data):
    assert sum(sample_data) == 15

def test_length(sample_data):
    assert len(sample_data) == 5

def test_empty_sum(empty_list):
    assert sum(empty_list) == 0
```

### Setup and Teardown with Fixtures

```python
import pytest

@pytest.fixture
def temp_file(tmp_path):
    """Create a temporary file for testing."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("hello world")
    yield file_path
    # Cleanup happens automatically (tmp_path is cleaned up by pytest)

def test_read_file(temp_file):
    content = temp_file.read_text()
    assert content == "hello world"
```

---

## 5. unittest: The Standard Library Alternative

### 5.1 Basic unittest Test

```python
import unittest

class TestCalculator(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)
    
    def test_add_negative(self):
        self.assertEqual(add(-1, -2), -3)
    
    def test_add_returns_number(self):
        result = add(2, 3)
        self.assertIsInstance(result, (int, float))

if __name__ == "__main__":
    unittest.main()
```

### 5.2 Common unittest Assertions

| Method | Checks |
|--------|--------|
| `assertEqual(a, b)` | `a == b` |
| `assertNotEqual(a, b)` | `a != b` |
| `assertTrue(x)` | `bool(x) is True` |
| `assertFalse(x)` | `bool(x) is False` |
| `assertIs(a, b)` | `a is b` |
| `assertIsNone(x)` | `x is None` |
| `assertIn(a, b)` | `a in b` |
| `assertIsInstance(a, b)` | `isinstance(a, b)` |
| `assertRaises(Exc)` | Exception is raised |
| `assertAlmostEqual(a, b)` | `round(a-b, 7) == 0` |

### 5.3 Setup and Teardown

```python
class TestDatabase(unittest.TestCase):
    def setUp(self):
        """Called before each test method."""
        self.db = create_test_database()
        self.db.insert({"name": "Alice"})
    
    def tearDown(self):
        """Called after each test method."""
        self.db.close()
    
    def test_query(self):
        result = self.db.query("Alice")
        self.assertIsNotNone(result)
```

---

## 6. Test-Driven Debugging

### 6.1 The Process

```
Bug reported: "average() returns wrong value for single-element lists"
          │
          ▼
Step 1: Write a test that FAILS
          │    def test_average_single_element():
          │        assert average([42]) == 42.0  # This fails!
          ▼
Step 2: Run the test to confirm it fails
          │    FAILED: assert 0.0 == 42.0
          ▼
Step 3: Debug and fix the code
          │    Found: dividing by len(data) + 1 instead of len(data)
          ▼
Step 4: Run the test to confirm it PASSES
          │    PASSED
          ▼
Step 5: Run ALL tests to ensure nothing else broke
          │    5 passed, 0 failed
          ▼
Step 6: Commit the fix AND the test
```

### 6.2 Example

```python
# The buggy function
def average(numbers):
    return sum(numbers) / (len(numbers) + 1)  # BUG: +1

# Step 1: Write the failing test
def test_average_single():
    assert average([42]) == 42.0

def test_average_multiple():
    assert average([10, 20, 30]) == 20.0

# Step 2: Run tests -- they fail
# Step 3: Fix the function
def average(numbers):
    return sum(numbers) / len(numbers)  # Fixed: removed +1

# Step 4: Tests pass
# Step 5: Add edge case tests
def test_average_negative():
    assert average([-10, 10]) == 0.0

def test_average_empty():
    import pytest
    with pytest.raises(ZeroDivisionError):
        average([])
```

---

## 7. Parametrized Tests

Test many inputs with one function:

```python
import pytest

@pytest.mark.parametrize("input_val, expected", [
    ([1, 2, 3], 6),
    ([0], 0),
    ([-1, 1], 0),
    ([10, 20, 30, 40], 100),
])
def test_sum(input_val, expected):
    assert sum(input_val) == expected

@pytest.mark.parametrize("a, b, expected", [
    (2, 3, 5),
    (-1, 1, 0),
    (0, 0, 0),
    (100, 200, 300),
    (-5, -3, -8),
])
def test_add(a, b, expected):
    assert add(a, b) == expected
```

---

## 8. What to Test

### 8.1 Testing Checklist

```
For each function, consider testing:
□ Normal case (happy path)
□ Edge cases (empty input, single element, zero, negative)
□ Boundary values (first/last valid value)
□ Error cases (invalid input, should raise exception)
□ Return type (correct type?)
□ Side effects (did it modify something it shouldn't?)
```

### 8.2 Example: Comprehensive Tests for a Function

```python
def clamp(value, minimum, maximum):
    """Restrict value to [minimum, maximum] range."""
    if minimum > maximum:
        raise ValueError("minimum must be <= maximum")
    return max(minimum, min(value, maximum))

# Tests
class TestClamp:
    def test_value_within_range(self):
        assert clamp(5, 0, 10) == 5
    
    def test_value_below_minimum(self):
        assert clamp(-5, 0, 10) == 0
    
    def test_value_above_maximum(self):
        assert clamp(15, 0, 10) == 10
    
    def test_value_at_minimum(self):
        assert clamp(0, 0, 10) == 0
    
    def test_value_at_maximum(self):
        assert clamp(10, 0, 10) == 10
    
    def test_equal_min_max(self):
        assert clamp(5, 7, 7) == 7
    
    def test_invalid_range(self):
        with pytest.raises(ValueError, match="minimum must be <= maximum"):
            clamp(5, 10, 0)
    
    def test_negative_range(self):
        assert clamp(-5, -10, -1) == -5
    
    def test_float_values(self):
        assert clamp(3.14, 0.0, 10.0) == 3.14
```

---

## 9. Test Coverage

### 9.1 What Is Coverage?

Test coverage measures what percentage of your code is executed during tests:

```bash
pip install pytest-cov
pytest --cov=mymodule --cov-report=term-missing
```

```
Name            Stmts   Miss  Cover   Missing
----------------------------------------------
calculator.py      20      4    80%   15-18
validator.py       35     10    71%   22-31
----------------------------------------------
TOTAL              55     14    75%
```

### 9.2 Coverage Guidelines

- **80%+** is a good target for most projects
- **100%** coverage does NOT mean zero bugs (you can cover every line but miss edge cases)
- Focus on testing **critical paths** and **edge cases**, not arbitrary coverage numbers

---

## 10. Testing Tips

### 10.1 Keep Tests Simple

Each test should test **one thing**:

```python
# BAD: Tests too many things
def test_everything():
    result = process([1, 2, 3])
    assert len(result) == 3
    assert result[0] > 0
    assert sum(result) == 12
    assert isinstance(result, list)

# GOOD: One assertion per concept
def test_process_preserves_length():
    assert len(process([1, 2, 3])) == 3

def test_process_returns_positive():
    assert all(x > 0 for x in process([1, 2, 3]))
```

### 10.2 Test Names Should Be Descriptive

```python
# BAD
def test_1():
    ...

# GOOD
def test_average_returns_zero_for_equal_positive_and_negative():
    ...
```

### 10.3 Tests Should Be Independent

Tests should not depend on each other or on execution order.

---

## Summary

- Testing is a debugging tool: write a failing test first, then fix the bug
- `assert` provides inline checks; `pytest` provides a full testing framework
- Use `pytest.raises()` to test that exceptions are raised correctly
- Organize tests in `test_` files with `test_` function names
- Fixtures provide reusable setup/teardown for tests
- Parametrized tests let you test many inputs with one function
- Test coverage helps identify untested code but 100% coverage does not guarantee correctness
- Every bug fix should be accompanied by a test that prevents regression

---

## Exercises

1. Write pytest tests for a given function, covering normal and edge cases
2. Use test-driven debugging to find and fix a bug
3. Write parametrized tests for a math function
4. Use `pytest.raises()` to test error handling

**Previous**: [Logging](./06_Logging.md) | **Next**: [Linters and Formatters](./08_Linters_and_Formatters.md)
