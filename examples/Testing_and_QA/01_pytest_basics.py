#!/usr/bin/env python3
"""Example: Pytest Basics

Demonstrates basic test functions, assertion introspection, markers,
parametrize, and common pytest idioms.
Related lesson: 01_Testing_Fundamentals.md, 02_Pytest_Framework.md
"""

# =============================================================================
# WHY PYTEST?
# pytest is the de facto standard for Python testing because:
#   1. No boilerplate — plain functions + assert, no class inheritance needed
#   2. Rich assertion introspection — shows exactly what differed on failure
#   3. Powerful fixture system — dependency injection for test setup
#   4. Plugin ecosystem — 1000+ plugins for every need
# =============================================================================

import pytest
import math


# =============================================================================
# BASIC TEST FUNCTIONS
# =============================================================================
# pytest discovers any function starting with "test_" in files starting with
# "test_" or ending with "_test.py". No class needed — keep it simple.

def test_addition():
    """The simplest possible test — a single assertion."""
    assert 1 + 1 == 2


def test_string_methods():
    """pytest's assertion introspection shows the actual vs expected string
    on failure, so you never need assertEqual or custom messages for basics."""
    greeting = "Hello, World!"
    assert greeting.lower() == "hello, world!"
    assert greeting.split(", ") == ["Hello", "World!"]


def test_list_operations():
    """Demonstrate that pytest shows the diff for collection comparisons,
    which makes debugging list mismatches trivial."""
    fruits = ["apple", "banana", "cherry"]
    fruits.append("date")
    assert len(fruits) == 4
    assert "banana" in fruits
    assert fruits[-1] == "date"


# =============================================================================
# TESTING EXCEPTIONS
# =============================================================================
# Use pytest.raises as a context manager. This is better than try/except
# because the test FAILS if the exception is NOT raised — which is the
# behavior you want when testing error paths.

def test_division_by_zero():
    """Verify that the correct exception type is raised."""
    with pytest.raises(ZeroDivisionError):
        _ = 1 / 0


def test_exception_message():
    """Use the match parameter (regex) to also verify the error message.
    This guards against the right exception type with the wrong reason."""
    with pytest.raises(ValueError, match=r"invalid literal"):
        int("not_a_number")


def test_exception_attributes():
    """Access the exception info object for detailed assertions."""
    with pytest.raises(KeyError) as exc_info:
        d = {"a": 1}
        _ = d["missing_key"]

    # exc_info.value is the actual exception instance
    assert exc_info.value.args[0] == "missing_key"


# =============================================================================
# APPROXIMATE COMPARISONS
# =============================================================================
# Floating point math introduces rounding errors. pytest.approx handles
# this cleanly without manually computing epsilon tolerances.

def test_floating_point():
    """Never use == for floats. pytest.approx uses a relative tolerance
    of 1e-6 by default, which covers most practical cases."""
    assert 0.1 + 0.2 == pytest.approx(0.3)
    assert math.sin(math.pi) == pytest.approx(0, abs=1e-10)


def test_approx_sequences():
    """pytest.approx also works on lists and dicts of floats."""
    computed = [0.1 + 0.2, 0.3 + 0.4]
    expected = [0.3, 0.7]
    assert computed == pytest.approx(expected)


# =============================================================================
# PARAMETRIZE — DATA-DRIVEN TESTS
# =============================================================================
# Why parametrize? Writing a separate test function for every input is tedious
# and obscures the pattern. @pytest.mark.parametrize generates one test case
# per parameter set, each reported independently.

@pytest.mark.parametrize("input_val, expected", [
    (1, 1),
    (2, 4),
    (3, 9),
    (4, 16),
    (-3, 9),
    (0, 0),
])
def test_square(input_val, expected):
    """Each tuple becomes a separate test case (6 tests total).
    If one fails, the others still run — no early abort."""
    assert input_val ** 2 == expected


@pytest.mark.parametrize("word, expected_palindrome", [
    ("racecar", True),
    ("hello", False),
    ("madam", True),
    ("", True),           # Edge case: empty string
    ("a", True),          # Edge case: single character
    ("Aba", True),        # Case-insensitive check
])
def test_is_palindrome(word, expected_palindrome):
    """Parametrize shines for edge-case coverage — list all interesting
    inputs in one place instead of scattering them across functions."""
    result = word.lower() == word.lower()[::-1]
    assert result == expected_palindrome


# Multiple parametrize decorators create the CARTESIAN PRODUCT of inputs.
# This is powerful for testing combinations without nested loops.

@pytest.mark.parametrize("base", [2, 10])
@pytest.mark.parametrize("exponent", [0, 1, 2, 3])
def test_power(base, exponent):
    """2 bases x 4 exponents = 8 test cases, all generated automatically."""
    result = base ** exponent
    assert result == pow(base, exponent)


# =============================================================================
# MARKERS — CATEGORIZING TESTS
# =============================================================================
# Markers let you tag tests and run subsets: pytest -m slow, pytest -m "not slow"
# Register custom markers in pytest.ini or pyproject.toml to avoid warnings.

@pytest.mark.slow
def test_large_computation():
    """Tagged as 'slow' so CI can skip it with: pytest -m 'not slow'"""
    total = sum(range(10_000_000))
    assert total == 49_999_995_000_000


@pytest.mark.skip(reason="Feature not yet implemented")
def test_future_feature():
    """skip: unconditionally skip this test. The reason appears in output."""
    assert False, "This should never run"


@pytest.mark.skipif(
    not hasattr(math, "isqrt"),
    reason="math.isqrt requires Python 3.8+"
)
def test_integer_sqrt():
    """skipif: skip only when a condition is true.
    Great for platform-specific or version-specific tests."""
    assert math.isqrt(16) == 4
    assert math.isqrt(17) == 4  # floor of sqrt


@pytest.mark.xfail(reason="Known bug in upstream library")
def test_known_bug():
    """xfail: expect this test to fail. If it unexpectedly PASSES,
    pytest reports it as XPASS so you know the bug was fixed."""
    # Simulating a known issue
    result = round(2.675, 2)
    # Due to floating-point representation, this rounds to 2.67 not 2.68
    assert result == 2.68


# =============================================================================
# ORGANIZING TESTS IN CLASSES
# =============================================================================
# Classes are optional but useful for grouping related tests.
# No inheritance from unittest.TestCase needed — plain classes work.

class TestStringOperations:
    """Group related tests. Each method is still discovered by its test_ prefix."""

    def test_upper(self):
        assert "hello".upper() == "HELLO"

    def test_strip(self):
        assert "  hello  ".strip() == "hello"

    def test_replace(self):
        assert "hello world".replace("world", "pytest") == "hello pytest"

    def test_split_with_limit(self):
        """Demonstrate testing with a maxsplit argument."""
        result = "a.b.c.d".split(".", maxsplit=2)
        assert result == ["a", "b", "c.d"]


# =============================================================================
# RUNNING THIS FILE
# =============================================================================
# From the command line:
#   pytest 01_pytest_basics.py -v          # verbose output
#   pytest 01_pytest_basics.py -v -k "palindrome"  # run only matching tests
#   pytest 01_pytest_basics.py -v -m slow  # run only @slow tests
#   pytest 01_pytest_basics.py --tb=short  # shorter traceback on failure

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
