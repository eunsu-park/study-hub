#!/bin/bash
# Exercises for Lesson 17: Testing Legacy Code
# Topic: Testing_and_QA
# Solutions to practice problems from the lesson.

# === Exercise 1: Characterization Test ===
# Problem: Take a function and write characterization tests for it.
# Run it with at least 10 different inputs and record the outputs.
# Include at least one case where the behavior seems wrong.
exercise_1() {
    echo "=== Exercise 1: Characterization Test ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import pytest

# --- Legacy function (do NOT modify) ---

def calculate_shipping(weight_kg, destination, is_express=False):
    """Legacy shipping calculator. Behavior captured, not specified."""
    base = 5.00
    if destination == "domestic":
        rate = 1.50
    elif destination == "international":
        rate = 4.00
    else:
        rate = 2.50  # Unknown destination gets a default

    cost = base + (weight_kg * rate)

    if is_express:
        cost *= 2
    if weight_kg > 30:
        cost += 15.00  # Heavy surcharge

    # BUG: Negative weight is not rejected — returns nonsensical result
    return round(cost, 2)

# --- Characterization tests ---
# These values were obtained by RUNNING the function, not from a spec.

class TestCharacterizeShipping:
    """Golden master: lock in current behavior before refactoring."""

    def test_domestic_light(self):
        assert calculate_shipping(1.0, "domestic") == 6.50

    def test_domestic_medium(self):
        assert calculate_shipping(10.0, "domestic") == 20.00

    def test_domestic_heavy(self):
        assert calculate_shipping(35.0, "domestic") == 72.50  # Includes surcharge

    def test_international_light(self):
        assert calculate_shipping(2.0, "international") == 13.00

    def test_international_heavy(self):
        assert calculate_shipping(50.0, "international") == 220.00

    def test_express_doubles_cost(self):
        assert calculate_shipping(5.0, "domestic", is_express=True) == 25.00

    def test_express_international_heavy(self):
        assert calculate_shipping(40.0, "international", is_express=True) == 360.00

    def test_unknown_destination(self):
        assert calculate_shipping(3.0, "mars") == 12.50

    def test_zero_weight(self):
        assert calculate_shipping(0.0, "domestic") == 5.00

    # Characterizing a BUG: negative weight yields a negative-ish cost.
    # This is current behavior, not correct behavior.
    def test_negative_weight_bug(self):
        result = calculate_shipping(-5.0, "domestic")
        assert result == -2.50  # base(5) + (-5 * 1.50) = -2.50
        # This is clearly wrong but it IS the current behavior.

    def test_fractional_weight(self):
        assert calculate_shipping(0.5, "domestic") == 5.75
SOLUTION
}

# === Exercise 2: Seam Identification ===
# Problem: Given a legacy function with three external dependencies
# (database, HTTP API, file system), identify all seams and propose
# the least invasive way to make it testable.
exercise_2() {
    echo "=== Exercise 2: Seam Identification ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from unittest.mock import Mock, patch
import pytest

# --- Legacy function (original, untestable) ---

def generate_invoice_legacy(order_id):
    """
    Original legacy function with 3 embedded dependencies:
    1. Database (psycopg2)
    2. HTTP API (requests)
    3. File system (open)
    """
    import psycopg2
    import requests

    # Dependency 1: Database
    conn = psycopg2.connect("postgresql://prod@db/orders")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM orders WHERE id = %s", (order_id,))
    order = cursor.fetchone()
    conn.close()

    # Dependency 2: HTTP API
    response = requests.get(f"https://tax-api.com/rate?state={order[4]}")
    tax_rate = response.json()["rate"]

    subtotal = order[3]
    tax = subtotal * tax_rate
    total = subtotal + tax

    # Dependency 3: File system
    with open(f"/invoices/{order_id}.txt", "w") as f:
        f.write(f"Order {order_id}: ${total:.2f}")

    return total

# --- Seam analysis ---
#
# SEAM 1 (Link seam): psycopg2.connect  -> patch at module import
# SEAM 2 (Link seam): requests.get      -> patch at module import
# SEAM 3 (Link seam): builtins.open     -> patch at module import
#
# LEAST INVASIVE approach: use unittest.mock.patch (link seams)
# to substitute all three dependencies WITHOUT changing the legacy code.

# --- Test using link seams (no code changes required) ---

@patch("builtins.open", create=True)
@patch("requests.get")
@patch("psycopg2.connect")
def test_generate_invoice_via_link_seams(mock_connect, mock_get, mock_open):
    """Test legacy function using link seams only — zero code changes."""

    # Mock database
    mock_cursor = Mock()
    mock_cursor.fetchone.return_value = (1, "Alice", "2024-01-15", 100.00, "CA")
    mock_conn = Mock()
    mock_conn.cursor.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    # Mock HTTP API
    mock_response = Mock()
    mock_response.json.return_value = {"rate": 0.08}
    mock_get.return_value = mock_response

    # Mock file system
    mock_file = Mock()
    mock_open.return_value.__enter__ = Mock(return_value=mock_file)
    mock_open.return_value.__exit__ = Mock(return_value=False)

    total = generate_invoice_legacy(1)

    assert total == 108.00
    mock_cursor.execute.assert_called_once()
    mock_get.assert_called_once_with("https://tax-api.com/rate?state=CA")

# --- Better long-term approach: Extract to Object Seam ---

class InvoiceGenerator:
    """Refactored version using dependency injection (object seam)."""

    def __init__(self, db=None, tax_client=None, file_writer=None):
        self.db = db
        self.tax_client = tax_client
        self.file_writer = file_writer

    def generate(self, order_id):
        order = self.db.get_order(order_id)
        tax_rate = self.tax_client.get_rate(order["state"])
        total = order["subtotal"] * (1 + tax_rate)
        self.file_writer.write(order_id, f"Order {order_id}: ${total:.2f}")
        return total

def test_invoice_generator_with_object_seams():
    mock_db = Mock()
    mock_db.get_order.return_value = {"state": "CA", "subtotal": 100.00}

    mock_tax = Mock()
    mock_tax.get_rate.return_value = 0.08

    mock_writer = Mock()

    gen = InvoiceGenerator(db=mock_db, tax_client=mock_tax, file_writer=mock_writer)
    total = gen.generate(1)

    assert total == 108.00
    mock_writer.write.assert_called_once()
SOLUTION
}

# === Exercise 3: Extract and Override ===
# Problem: Take a class with a hardcoded dependency on datetime.now()
# and apply the Extract and Override technique. Write tests for
# time-dependent behavior.
exercise_3() {
    echo "=== Exercise 3: Extract and Override ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import pytest
from datetime import datetime

# --- Legacy class with hardcoded datetime.now() ---

class SubscriptionManager:
    """Manages subscription validity checks."""

    def _get_current_time(self):
        """Extracted seam: override this in tests."""
        return datetime.now()

    def is_active(self, subscription):
        now = self._get_current_time()
        return subscription["start"] <= now <= subscription["end"]

    def days_remaining(self, subscription):
        now = self._get_current_time()
        if now > subscription["end"]:
            return 0
        return (subscription["end"] - now).days

    def status(self, subscription):
        now = self._get_current_time()
        if now < subscription["start"]:
            return "pending"
        elif now > subscription["end"]:
            return "expired"
        else:
            return "active"

# --- Test subclass overrides the time method ---

class TestableSubscriptionManager(SubscriptionManager):
    """Override the time seam for deterministic testing."""

    def __init__(self, fixed_time: datetime):
        self._fixed_time = fixed_time

    def _get_current_time(self):
        return self._fixed_time

# --- Tests ---

@pytest.fixture
def subscription():
    return {
        "start": datetime(2024, 1, 1),
        "end": datetime(2024, 12, 31),
    }

def test_active_during_subscription(subscription):
    manager = TestableSubscriptionManager(datetime(2024, 6, 15))
    assert manager.is_active(subscription) is True
    assert manager.status(subscription) == "active"

def test_expired_after_end(subscription):
    manager = TestableSubscriptionManager(datetime(2025, 3, 1))
    assert manager.is_active(subscription) is False
    assert manager.status(subscription) == "expired"
    assert manager.days_remaining(subscription) == 0

def test_pending_before_start(subscription):
    manager = TestableSubscriptionManager(datetime(2023, 11, 1))
    assert manager.is_active(subscription) is False
    assert manager.status(subscription) == "pending"

def test_days_remaining_mid_year(subscription):
    manager = TestableSubscriptionManager(datetime(2024, 7, 1))
    remaining = manager.days_remaining(subscription)
    assert remaining == 183  # Jul 1 to Dec 31

def test_last_day_is_still_active(subscription):
    manager = TestableSubscriptionManager(datetime(2024, 12, 31))
    assert manager.is_active(subscription) is True
    assert manager.days_remaining(subscription) == 0
SOLUTION
}

# === Exercise 4: Sprout Method ===
# Problem: Given a legacy function that needs new functionality
# (adding input validation), implement it as a sprout method.
# Write full tests for the new method.
exercise_4() {
    echo "=== Exercise 4: Sprout Method ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import pytest

# --- Legacy function (untested, fragile, do not modify logic) ---

def process_payment_legacy(amount, card_number, currency="USD"):
    """
    Legacy payment processor — 500+ lines in real life.
    We need to add input validation WITHOUT touching this code.
    Strategy: sprout a validate_payment_input() method.
    """
    # ... imagine 500 lines of payment logic ...
    # We add a call to the sprouted method at the entry point:
    validate_payment_input(amount, card_number, currency)
    # ... rest of legacy logic continues unchanged ...
    return {"status": "charged", "amount": amount, "currency": currency}

# --- Sprouted method (fully testable, isolated from legacy) ---

SUPPORTED_CURRENCIES = {"USD", "EUR", "GBP", "JPY", "CAD"}

def validate_payment_input(amount, card_number, currency):
    """
    Sprout method: new validation logic, separate from legacy code.
    Raises ValueError for invalid inputs.
    """
    if not isinstance(amount, (int, float)):
        raise ValueError(f"Amount must be numeric, got {type(amount).__name__}")

    if amount <= 0:
        raise ValueError(f"Amount must be positive, got {amount}")

    if amount > 999_999.99:
        raise ValueError(f"Amount exceeds maximum: {amount}")

    if not isinstance(card_number, str) or not card_number.isdigit():
        raise ValueError("Card number must be a string of digits")

    if len(card_number) not in (13, 14, 15, 16):
        raise ValueError(f"Invalid card number length: {len(card_number)}")

    if currency not in SUPPORTED_CURRENCIES:
        raise ValueError(f"Unsupported currency: {currency}")

# --- Tests for the sprouted method ---

class TestValidatePaymentInput:
    def test_valid_input(self):
        # Should not raise
        validate_payment_input(50.00, "4111111111111111", "USD")

    def test_negative_amount(self):
        with pytest.raises(ValueError, match="must be positive"):
            validate_payment_input(-10, "4111111111111111", "USD")

    def test_zero_amount(self):
        with pytest.raises(ValueError, match="must be positive"):
            validate_payment_input(0, "4111111111111111", "USD")

    def test_excessive_amount(self):
        with pytest.raises(ValueError, match="exceeds maximum"):
            validate_payment_input(1_000_000, "4111111111111111", "USD")

    def test_non_numeric_amount(self):
        with pytest.raises(ValueError, match="must be numeric"):
            validate_payment_input("fifty", "4111111111111111", "USD")

    def test_card_number_with_letters(self):
        with pytest.raises(ValueError, match="string of digits"):
            validate_payment_input(50, "4111-1111-1111", "USD")

    def test_card_number_too_short(self):
        with pytest.raises(ValueError, match="Invalid card number length"):
            validate_payment_input(50, "12345", "USD")

    def test_unsupported_currency(self):
        with pytest.raises(ValueError, match="Unsupported currency"):
            validate_payment_input(50, "4111111111111111", "BTC")

    def test_all_supported_currencies(self):
        for cur in SUPPORTED_CURRENCIES:
            validate_payment_input(10.00, "4111111111111111", cur)
SOLUTION
}

# === Exercise 5: Strangler Fig Plan ===
# Problem: For a module with 500 lines of untested code, create a
# written plan for introducing tests over 4 sprints.
exercise_5() {
    echo "=== Exercise 5: Strangler Fig Plan ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
"""
Strangler Fig Test Introduction Plan
=====================================
Module: order_processing.py (500 lines, 0% coverage)
Timeline: 4 sprints (2 weeks each)

Risk Assessment (from code analysis):
  HIGH   - calculate_totals() : Complex discount logic, 8 branches
  HIGH   - charge_payment()   : External payment gateway integration
  MEDIUM - validate_order()   : Input validation, 5 edge cases
  MEDIUM - apply_tax()        : Tax lookup by region, 50 states
  LOW    - format_receipt()   : String formatting, pure function
  LOW    - send_confirmation(): Email dispatch, easily mockable

Sprint 1: Characterize & Quick Wins (16 hours)
-----------------------------------------------
  [8h] Write characterization tests for calculate_totals()
       - Record outputs for 20+ input combinations
       - Identify edge cases (negative qty, zero price, bulk discount)
       - Expected coverage gain: 0% -> 25%

  [4h] Write unit tests for format_receipt() (pure function)
       - 5 test cases covering all receipt formats
       - No mocking needed — easy win for team morale

  [4h] Write characterization tests for validate_order()
       - Cover all validation branches
       - Document which validations exist vs. which are missing

Sprint 2: Extract Seams (12 hours)
-----------------------------------
  [6h] Refactor charge_payment() to use dependency injection
       - Extract StripeGateway to a parameter
       - Add characterization tests BEFORE refactoring
       - Write unit tests with mock gateway AFTER refactoring
       - Expected coverage gain: 25% -> 45%

  [4h] Sprout method: add missing input validations
       - validate_line_items() — new, fully tested
       - validate_shipping_address() — new, fully tested

  [2h] Add integration test for the happy path
       - order creation -> payment -> receipt
       - Uses test doubles for all external services

Sprint 3: Deepen Coverage (8 hours)
-------------------------------------
  [4h] Unit tests for apply_tax() with mocked tax service
       - Cover all 50 states + international
       - Property test: tax is always non-negative
       - Expected coverage gain: 45% -> 70%

  [2h] Test send_confirmation() with mock email client
       - Verify email content and recipients

  [2h] Add edge case tests for calculate_totals()
       - Empty cart, single item, max items
       - Coupon codes, expired coupons
       - Expected coverage gain: 70% -> 80%

Sprint 4: Harden & Sustain (4 hours)
--------------------------------------
  [2h] Add regression tests for 3 known bugs (from issue tracker)
       - Each bug gets a failing test FIRST, then verify fix

  [2h] Set up CI enforcement
       - coverage >= 80% for order_processing module
       - All new code requires tests (pre-commit hook)

  Final coverage target: >= 80%

Metrics to Track:
  - Line coverage (per sprint)
  - Number of characterization tests vs specification tests
  - Defect escape rate (before vs after testing)
  - Team velocity impact (should stabilize by Sprint 3)
"""

# Executable version: priority scorer for untested functions

def prioritize_functions(functions: list[dict]) -> list[dict]:
    """Score and rank functions by testing priority."""
    for fn in functions:
        score = (
            fn["risk"] * 2 +
            fn["change_frequency"] * 1.5 +
            fn["bug_history"] * 1.5 +
            fn["complexity"]
        )
        fn["priority_score"] = score
    return sorted(functions, key=lambda f: f["priority_score"], reverse=True)

# Example usage:
functions = [
    {"name": "calculate_totals", "risk": 9, "change_frequency": 8,
     "bug_history": 7, "complexity": 8},
    {"name": "charge_payment", "risk": 10, "change_frequency": 5,
     "bug_history": 6, "complexity": 7},
    {"name": "format_receipt", "risk": 2, "change_frequency": 3,
     "bug_history": 1, "complexity": 2},
    {"name": "validate_order", "risk": 6, "change_frequency": 7,
     "bug_history": 5, "complexity": 5},
    {"name": "apply_tax", "risk": 5, "change_frequency": 4,
     "bug_history": 3, "complexity": 6},
    {"name": "send_confirmation", "risk": 3, "change_frequency": 2,
     "bug_history": 2, "complexity": 3},
]

ranked = prioritize_functions(functions)
for fn in ranked:
    print(f"  {fn['priority_score']:5.1f}  {fn['name']}")

# Expected output (highest priority first):
#   40.5  calculate_totals
#   37.5  charge_payment
#   28.5  validate_order
#   20.5  apply_tax
#   12.5  send_confirmation
#   10.0  format_receipt
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 17: Testing Legacy Code"
echo "======================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
echo ""
exercise_5
