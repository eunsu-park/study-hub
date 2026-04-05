#!/bin/bash
# Exercises for Lesson 06: Unit Testing Best Practices
# Topic: Testing_and_QA
# Solutions to practice problems from the lesson.

# === Exercise 1: Arrange-Act-Assert Pattern ===
# Problem: Refactor messy tests into clean AAA pattern.
exercise_1() {
    echo "=== Exercise 1: Arrange-Act-Assert Pattern ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import pytest

class ShoppingCart:
    def __init__(self):
        self.items = []

    def add_item(self, name: str, price: float, qty: int = 1):
        self.items.append({"name": name, "price": price, "qty": qty})

    def total(self) -> float:
        return sum(i["price"] * i["qty"] for i in self.items)

    def remove_item(self, name: str):
        self.items = [i for i in self.items if i["name"] != name]

    def item_count(self) -> int:
        return sum(i["qty"] for i in self.items)

# BAD: Everything mixed together
def test_cart_bad():
    cart = ShoppingCart()
    cart.add_item("Apple", 1.50, 3)
    assert cart.total() == 4.50
    cart.add_item("Banana", 0.75, 2)
    assert cart.total() == 6.00
    assert cart.item_count() == 5
    cart.remove_item("Apple")
    assert cart.total() == 1.50

# GOOD: Clear AAA structure, one concept per test

def test_total_single_item():
    # Arrange
    cart = ShoppingCart()
    cart.add_item("Apple", 1.50, 3)

    # Act
    total = cart.total()

    # Assert
    assert total == 4.50

def test_total_multiple_items():
    # Arrange
    cart = ShoppingCart()
    cart.add_item("Apple", 1.50, 3)
    cart.add_item("Banana", 0.75, 2)

    # Act
    total = cart.total()

    # Assert
    assert total == 6.00

def test_item_count():
    # Arrange
    cart = ShoppingCart()
    cart.add_item("Apple", 1.50, 3)
    cart.add_item("Banana", 0.75, 2)

    # Act
    count = cart.item_count()

    # Assert
    assert count == 5

def test_remove_item_updates_total():
    # Arrange
    cart = ShoppingCart()
    cart.add_item("Apple", 1.50, 3)
    cart.add_item("Banana", 0.75, 2)

    # Act
    cart.remove_item("Apple")

    # Assert
    assert cart.total() == 1.50
SOLUTION
}

# === Exercise 2: Test Naming Conventions ===
# Problem: Improve test names to be descriptive and follow conventions.
exercise_2() {
    echo "=== Exercise 2: Test Naming Conventions ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import pytest

# BAD names:
# def test_1():
# def test_it():
# def test_calc():
# def test_function():

# GOOD: method_scenario_expectedBehavior pattern

def test_divide_by_nonzero_returns_quotient():
    assert 10 / 2 == 5.0

def test_divide_by_zero_raises_zerodivisionerror():
    with pytest.raises(ZeroDivisionError):
        _ = 10 / 0

# GOOD: Given-When-Then naming for complex scenarios

def test_given_empty_cart_when_checkout_then_raises_error():
    cart = ShoppingCart()
    with pytest.raises(ValueError):
        cart.checkout()

def test_given_items_in_cart_when_apply_coupon_then_total_reduced():
    cart = ShoppingCart()
    cart.add_item("Book", 20.00)
    cart.apply_coupon("SAVE10")
    assert cart.total() == 18.00  # 10% off

# GOOD: Class grouping for related tests

class TestUserRegistration:
    """Group all registration-related tests."""

    def test_valid_email_succeeds(self):
        pass

    def test_duplicate_email_raises_conflict(self):
        pass

    def test_weak_password_returns_validation_error(self):
        pass

class TestUserLogin:
    """Group all login-related tests."""

    def test_correct_credentials_returns_token(self):
        pass

    def test_wrong_password_returns_unauthorized(self):
        pass

    def test_locked_account_returns_forbidden(self):
        pass

# Naming guidelines:
# 1. Start with the method/function being tested
# 2. Include the scenario or condition
# 3. End with the expected outcome
# 4. Use underscores for readability (Python convention)
# 5. Be specific enough that the test name explains the failure
SOLUTION
}

# === Exercise 3: Test Isolation ===
# Problem: Fix tests that share mutable state and demonstrate
# proper isolation techniques.
exercise_3() {
    echo "=== Exercise 3: Test Isolation ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import pytest

# BAD: Shared mutable state between tests
_shared_list = []  # Global state — tests affect each other!

def test_bad_add_item():
    _shared_list.append("item1")
    assert len(_shared_list) == 1  # Passes first time

def test_bad_list_empty():
    assert len(_shared_list) == 0  # FAILS if test_bad_add_item runs first!

# GOOD FIX 1: Use fixtures for fresh state
@pytest.fixture
def item_list():
    """Each test gets its own list — no leaking."""
    return []

def test_good_add_item(item_list):
    item_list.append("item1")
    assert len(item_list) == 1

def test_good_list_empty(item_list):
    assert len(item_list) == 0  # Always passes — fresh list

# GOOD FIX 2: Use monkeypatch for environment variables
def get_debug_mode():
    import os
    return os.environ.get("DEBUG", "false") == "true"

def test_debug_mode_on(monkeypatch):
    monkeypatch.setenv("DEBUG", "true")
    assert get_debug_mode() is True

def test_debug_mode_off(monkeypatch):
    monkeypatch.delenv("DEBUG", raising=False)
    assert get_debug_mode() is False
    # monkeypatch automatically restores the original env after test

# GOOD FIX 3: Use tmp_path for file system isolation
def test_write_file(tmp_path):
    filepath = tmp_path / "output.txt"
    filepath.write_text("test data")
    assert filepath.read_text() == "test data"
    # tmp_path is unique per test and cleaned up after session

# Key isolation principles:
# 1. No global/module-level mutable state
# 2. Each test creates its own data (via fixtures)
# 3. Tests must pass in any order (use: pytest --randomly)
# 4. Clean up side effects (files, env vars, DB records)
SOLUTION
}

# === Exercise 4: Testing Edge Cases Systematically ===
# Problem: Write comprehensive edge case tests for a string truncation function.
exercise_4() {
    echo "=== Exercise 4: Testing Edge Cases Systematically ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import pytest

def truncate(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to max_length, appending suffix if truncated."""
    if max_length < len(suffix):
        raise ValueError(f"max_length must be >= {len(suffix)}")
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

# --- Normal cases ---
def test_short_text_unchanged():
    assert truncate("Hi", 10) == "Hi"

def test_long_text_truncated():
    assert truncate("Hello World", 8) == "Hello..."

# --- Boundary cases ---
def test_exact_length_unchanged():
    assert truncate("Hello", 5) == "Hello"

def test_one_over_length():
    assert truncate("Hello!", 5) == "He..."

# --- Edge cases ---
def test_empty_string():
    assert truncate("", 5) == ""

def test_max_length_equals_suffix_length():
    assert truncate("Hello", 3) == "..."

def test_max_length_less_than_suffix_raises():
    with pytest.raises(ValueError):
        truncate("Hello", 2)

def test_custom_suffix():
    assert truncate("Hello World", 9, suffix=">>") == "Hello W>>"

def test_empty_suffix():
    assert truncate("Hello World", 5, suffix="") == "Hello"

# --- Systematic edge case checklist ---
# For any function, always test:
# 1. Empty input (empty string, empty list, None)
# 2. Single element (one character, one item)
# 3. Boundary values (exactly at limit, one above, one below)
# 4. Large input (very long strings, huge numbers)
# 5. Special characters (unicode, emoji, newlines)
# 6. Type errors (wrong type input)
# 7. Negative numbers / zero (for numeric inputs)
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 06: Unit Testing Best Practices"
echo "==============================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
