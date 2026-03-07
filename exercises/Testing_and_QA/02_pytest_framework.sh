#!/bin/bash
# Exercises for Lesson 02: Pytest Framework
# Topic: Testing_and_QA
# Solutions to practice problems from the lesson.

# === Exercise 1: Parametrize and Markers ===
# Problem: Write parametrized tests for a FizzBuzz function.
# Use markers to separate fast and slow tests.
exercise_1() {
    echo "=== Exercise 1: Parametrize and Markers ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import pytest

def fizzbuzz(n: int) -> str:
    """Return FizzBuzz result for a given number."""
    if n % 15 == 0:
        return "FizzBuzz"
    if n % 3 == 0:
        return "Fizz"
    if n % 5 == 0:
        return "Buzz"
    return str(n)

# Parametrize: one test function covers many cases
@pytest.mark.parametrize("input_val, expected", [
    (1, "1"),
    (2, "2"),
    (3, "Fizz"),
    (5, "Buzz"),
    (6, "Fizz"),
    (10, "Buzz"),
    (15, "FizzBuzz"),
    (30, "FizzBuzz"),
    (7, "7"),
])
def test_fizzbuzz(input_val, expected):
    assert fizzbuzz(input_val) == expected

# Marker: tag tests that verify edge cases
@pytest.mark.edge_case
@pytest.mark.parametrize("n", [0, -3, -5, -15])
def test_fizzbuzz_negative_and_zero(n):
    """Negative multiples still follow FizzBuzz rules."""
    result = fizzbuzz(n)
    if n % 15 == 0:
        assert result == "FizzBuzz"
    elif n % 3 == 0:
        assert result == "Fizz"
    elif n % 5 == 0:
        assert result == "Buzz"

# Run only edge case tests: pytest -m edge_case
# Run everything except: pytest -m "not edge_case"
SOLUTION
}

# === Exercise 2: Custom Markers and Configuration ===
# Problem: Create a pyproject.toml with custom marker registration
# and demonstrate marker-based test selection.
exercise_2() {
    echo "=== Exercise 2: Custom Markers and Configuration ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# pyproject.toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests requiring external services",
    "smoke: marks critical path tests for quick validation",
    "edge_case: marks boundary and edge case tests",
]
testpaths = ["tests"]
addopts = "-v --strict-markers"
# --strict-markers: fail on unregistered markers (catches typos)

# test_with_markers.py
import pytest

@pytest.mark.smoke
def test_app_starts():
    """Critical: app must start. Run with: pytest -m smoke"""
    assert True  # replace with real startup check

@pytest.mark.slow
def test_full_data_import():
    """Takes 30+ seconds. Exclude with: pytest -m 'not slow'"""
    import time
    # time.sleep(30)  # Simulated slow operation
    assert True

@pytest.mark.integration
def test_database_connection():
    """Requires running PostgreSQL. Run with: pytest -m integration"""
    assert True

# Combine markers for fine-grained selection:
# pytest -m "smoke and not slow"     -> fast critical tests
# pytest -m "smoke or integration"   -> all important tests
# pytest -m "not (slow or integration)" -> only fast unit tests
SOLUTION
}

# === Exercise 3: Test Discovery and Organization ===
# Problem: Organize tests for a web application following pytest conventions.
exercise_3() {
    echo "=== Exercise 3: Test Discovery and Organization ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Recommended project structure:
#
# myapp/
# ├── src/
# │   └── myapp/
# │       ├── __init__.py
# │       ├── models.py
# │       ├── services.py
# │       └── api.py
# ├── tests/
# │   ├── conftest.py          # Shared fixtures for ALL tests
# │   ├── unit/
# │   │   ├── conftest.py      # Unit-specific fixtures (e.g., mocked DB)
# │   │   ├── test_models.py
# │   │   └── test_services.py
# │   ├── integration/
# │   │   ├── conftest.py      # Integration fixtures (e.g., test DB)
# │   │   ├── test_api.py
# │   │   └── test_database.py
# │   └── e2e/
# │       ├── conftest.py      # E2E fixtures (e.g., browser, server)
# │       └── test_workflows.py
# └── pyproject.toml

# tests/conftest.py — shared by all test types
import pytest

@pytest.fixture(scope="session")
def app():
    """Create application for the entire test session."""
    from myapp import create_app
    return create_app(testing=True)

# tests/unit/conftest.py — unit tests get mocked dependencies
@pytest.fixture
def mock_db(mocker):
    """Replace real DB with a mock for unit tests."""
    return mocker.patch("myapp.services.get_db")

# tests/integration/conftest.py — integration tests get real dependencies
@pytest.fixture(scope="module")
def test_db():
    """Create a real test database for integration tests."""
    # Setup: create tables
    yield db_connection
    # Teardown: drop tables

# pytest discovers tests by default rules:
# - Files matching test_*.py or *_test.py
# - Functions starting with test_
# - Classes starting with Test (no __init__)
SOLUTION
}

# === Exercise 4: Assertion Introspection ===
# Problem: Write tests that take advantage of pytest's assertion introspection
# for clear failure messages. Include custom assertion helpers.
exercise_4() {
    echo "=== Exercise 4: Assertion Introspection ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import pytest

# pytest rewrites assert statements to show detailed failure info.
# No need for assertEqual, assertTrue, etc.

def test_string_comparison():
    """On failure, pytest shows the exact diff between strings."""
    expected = "Hello, World!"
    actual = "Hello, World!"  # Change this to see the diff
    assert actual == expected

def test_list_comparison():
    """On failure, pytest shows which elements differ."""
    expected = [1, 2, 3, 4, 5]
    actual = [1, 2, 3, 4, 5]
    assert actual == expected

def test_dict_comparison():
    """On failure, pytest shows added, removed, and changed keys."""
    expected = {"name": "Alice", "age": 30, "role": "admin"}
    actual = {"name": "Alice", "age": 30, "role": "admin"}
    assert actual == expected

# Custom assertion helper for domain-specific checks
def assert_valid_email(email: str):
    """Reusable assertion with clear error message."""
    assert isinstance(email, str), f"Email must be a string, got {type(email)}"
    assert "@" in email, f"Email missing '@': {email}"
    parts = email.split("@")
    assert len(parts) == 2, f"Email has multiple '@': {email}"
    assert "." in parts[1], f"Email domain missing '.': {email}"
    assert len(parts[0]) > 0, f"Email has empty local part: {email}"

def test_valid_email():
    assert_valid_email("user@example.com")

def test_invalid_email_missing_at():
    with pytest.raises(AssertionError, match="missing '@'"):
        assert_valid_email("userexample.com")

# Using pytest.approx for floating-point comparisons
def test_float_precision():
    """Never use == for floats. pytest.approx handles tolerance."""
    assert 0.1 + 0.2 == pytest.approx(0.3)
    assert 3.14 == pytest.approx(3.14159, abs=0.01)
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 02: Pytest Framework"
echo "===================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
