#!/bin/bash
# Exercises for Lesson 01: Testing Fundamentals
# Topic: Testing_and_QA
# Solutions to practice problems from the lesson.

# === Exercise 1: Writing Your First Test ===
# Problem: Write a test for a function that checks if a number is prime.
# Include tests for edge cases (0, 1, 2, negative numbers).
exercise_1() {
    echo "=== Exercise 1: Writing Your First Test ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# is_prime.py
def is_prime(n: int) -> bool:
    """Check if a number is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

# test_is_prime.py
import pytest
from is_prime import is_prime

# Test basic primes
def test_small_primes():
    assert is_prime(2) is True
    assert is_prime(3) is True
    assert is_prime(5) is True
    assert is_prime(7) is True

# Test non-primes
def test_non_primes():
    assert is_prime(4) is False
    assert is_prime(6) is False
    assert is_prime(9) is False
    assert is_prime(15) is False

# Test edge cases — these are where most bugs hide
def test_edge_cases():
    assert is_prime(0) is False   # 0 is not prime
    assert is_prime(1) is False   # 1 is not prime by convention
    assert is_prime(-5) is False  # Negatives are not prime
    assert is_prime(2) is True    # Smallest prime

# Test larger prime
def test_large_prime():
    assert is_prime(97) is True
    assert is_prime(100) is False
SOLUTION
}

# === Exercise 2: Understanding Test Levels ===
# Problem: Classify the following test descriptions into the correct level:
# Unit, Integration, System, or Acceptance.
exercise_2() {
    echo "=== Exercise 2: Understanding Test Levels ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Classification of test scenarios by testing level:

# 1. "Test that the calculate_discount() function returns 10% off for orders over $100"
#    Level: UNIT TEST
#    Why: Tests a single function in isolation with specific input/output

# 2. "Test that the checkout service saves the order to the database and sends
#     a confirmation email"
#    Level: INTEGRATION TEST
#    Why: Tests the interaction between multiple components (service, DB, email)

# 3. "Test that a user can browse products, add them to cart, and complete purchase"
#    Level: SYSTEM TEST (End-to-End)
#    Why: Tests the full user workflow across the entire application

# 4. "Test that the sorting algorithm correctly handles an empty list"
#    Level: UNIT TEST
#    Why: Tests a single algorithm with a specific edge-case input

# 5. "Test that the REST API returns proper error codes when the database is down"
#    Level: INTEGRATION TEST
#    Why: Tests how the API layer handles database failure (cross-component)

# 6. "The customer verifies that they can reset their password via email"
#    Level: ACCEPTANCE TEST
#    Why: Written from the customer's perspective to validate business requirements

# Key principle: The Testing Pyramid
#       /\        Fewer, slower, more expensive
#      /  \       System/E2E Tests
#     /----\
#    / Integ \    Integration Tests
#   /--------\
#  / Unit     \   Many, fast, cheap
# /____________\
SOLUTION
}

# === Exercise 3: Test Case Design ===
# Problem: Design test cases for a password validation function.
# Requirements: min 8 chars, 1 uppercase, 1 lowercase, 1 digit, 1 special char.
exercise_3() {
    echo "=== Exercise 3: Test Case Design ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import pytest

def validate_password(password: str) -> tuple[bool, list[str]]:
    """Validate password strength. Returns (is_valid, list_of_errors)."""
    errors = []
    if len(password) < 8:
        errors.append("Must be at least 8 characters")
    if not any(c.isupper() for c in password):
        errors.append("Must contain an uppercase letter")
    if not any(c.islower() for c in password):
        errors.append("Must contain a lowercase letter")
    if not any(c.isdigit() for c in password):
        errors.append("Must contain a digit")
    if not any(c in "!@#$%^&*()-_=+[]{}|;:',.<>?/" for c in password):
        errors.append("Must contain a special character")
    return len(errors) == 0, errors

# --- Valid passwords ---
def test_valid_password():
    is_valid, errors = validate_password("Str0ng!Pass")
    assert is_valid is True
    assert errors == []

def test_minimum_valid():
    """Boundary: exactly 8 characters meeting all criteria"""
    is_valid, errors = validate_password("Ab1!xxxx")
    assert is_valid is True

# --- Invalid: too short ---
def test_too_short():
    is_valid, errors = validate_password("Ab1!")
    assert is_valid is False
    assert "Must be at least 8 characters" in errors

# --- Invalid: missing uppercase ---
def test_no_uppercase():
    is_valid, errors = validate_password("abcdefg1!")
    assert is_valid is False
    assert "Must contain an uppercase letter" in errors

# --- Invalid: missing lowercase ---
def test_no_lowercase():
    is_valid, errors = validate_password("ABCDEFG1!")
    assert is_valid is False
    assert "Must contain a lowercase letter" in errors

# --- Invalid: missing digit ---
def test_no_digit():
    is_valid, errors = validate_password("Abcdefgh!")
    assert is_valid is False
    assert "Must contain a digit" in errors

# --- Invalid: missing special character ---
def test_no_special():
    is_valid, errors = validate_password("Abcdefg1")
    assert is_valid is False
    assert "Must contain a special character" in errors

# --- Multiple errors ---
def test_empty_password():
    is_valid, errors = validate_password("")
    assert is_valid is False
    assert len(errors) == 5  # All rules violated

# --- Edge cases ---
def test_exactly_7_chars():
    """Boundary: one less than minimum"""
    is_valid, errors = validate_password("Ab1!xxx")
    assert is_valid is False

def test_exactly_8_chars_invalid():
    is_valid, errors = validate_password("aaaaaaaa")
    assert is_valid is False
SOLUTION
}

# === Exercise 4: Test Quality Characteristics ===
# Problem: Identify and fix problems in the following poorly written tests.
exercise_4() {
    echo "=== Exercise 4: Test Quality Characteristics ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# BAD TEST (multiple problems):
def test_everything():
    # Problem 1: Tests multiple unrelated things (not focused)
    # Problem 2: Shared mutable state between assertions
    # Problem 3: No clear failure message
    calc = Calculator()
    assert calc.add(1, 2) == 3
    assert calc.subtract(5, 3) == 2
    assert calc.multiply(2, 4) == 8
    result = calc.divide(10, 0)  # If this throws, we don't know which failed

# GOOD TESTS (one concept per test, clear names):
def test_add_returns_sum_of_two_numbers():
    """Focused: tests ONE behavior with a descriptive name."""
    calc = Calculator()
    assert calc.add(1, 2) == 3

def test_subtract_returns_difference():
    calc = Calculator()
    assert calc.subtract(5, 3) == 2

def test_multiply_returns_product():
    calc = Calculator()
    assert calc.multiply(2, 4) == 8

def test_divide_by_zero_raises_error():
    """Isolated: tests the error path separately."""
    calc = Calculator()
    with pytest.raises(ZeroDivisionError):
        calc.divide(10, 0)

# FIRST Principles for Good Tests:
# F - Fast: Tests should run quickly (milliseconds, not seconds)
# I - Independent: No test depends on another test's outcome
# R - Repeatable: Same result every time (no random failures)
# S - Self-validating: Pass or fail, no manual inspection needed
# T - Timely: Written close in time to the production code
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 01: Testing Fundamentals"
echo "======================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
