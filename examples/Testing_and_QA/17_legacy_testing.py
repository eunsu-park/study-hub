#!/usr/bin/env python3
"""Example: Testing Legacy Code

Demonstrates characterization tests, finding seams for testability,
the strangler fig pattern, and techniques for safely adding tests
to untested code.
Related lesson: 17_Legacy_Testing.md
"""

# =============================================================================
# WHY LEGACY TESTING?
#
# Legacy code = code without tests (Michael Feathers' definition).
# The dilemma: you need tests to refactor safely, but the code isn't
# testable without refactoring. Break the cycle with:
#
#   1. Characterization tests — document current behavior (bugs included)
#   2. Seams — find points where you can inject test doubles
#   3. Strangler fig — gradually replace legacy code with tested code
#   4. Sprout method/class — add new behavior in tested code, call from legacy
# =============================================================================

import pytest
from datetime import datetime, timedelta
from typing import Optional
from unittest.mock import patch, MagicMock


# =============================================================================
# LEGACY CODE — HARD TO TEST (BEFORE REFACTORING)
# =============================================================================

# This simulates typical legacy code problems:
# - Global state, hidden dependencies, no injection points
# - Mixed concerns (business logic + I/O + formatting)
# - No interfaces or abstractions

_PRICING_DB = {
    "WIDGET_A": 29.99,
    "WIDGET_B": 49.99,
    "WIDGET_C": 99.99,
}

_DISCOUNT_RULES = {
    "LOYAL": 0.10,
    "BULK": 0.15,
    "EMPLOYEE": 0.25,
}


def get_current_time() -> datetime:
    """Hidden dependency — makes testing time-dependent logic hard."""
    return datetime.now()


def calculate_order_total(items: list[dict], customer_type: str = "REGULAR") -> dict:
    """Legacy function with multiple problems:
    - Reads from global _PRICING_DB (hidden dependency)
    - Calls get_current_time() (hidden dependency)
    - Mixes calculation with formatting
    - Has a subtle bug in the weekend surcharge logic
    """
    subtotal = 0.0
    line_items = []

    for item in items:
        sku = item["sku"]
        qty = item["qty"]

        if sku not in _PRICING_DB:
            # Legacy behavior: silently skip unknown SKUs (arguably a bug,
            # but characterization tests document this AS-IS behavior)
            continue

        price = _PRICING_DB[sku]
        line_total = price * qty
        subtotal += line_total
        line_items.append({"sku": sku, "qty": qty, "price": price, "total": line_total})

    # Apply customer discount
    discount_rate = _DISCOUNT_RULES.get(customer_type, 0.0)
    discount = round(subtotal * discount_rate, 2)

    # Weekend surcharge (the "bug" — Saturday is 5, Sunday is 6,
    # but this checks weekday() < 5 which is weekdays, not weekends)
    now = get_current_time()
    surcharge = 0.0
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        surcharge = round(subtotal * 0.05, 2)

    total = round(subtotal - discount + surcharge, 2)

    return {
        "line_items": line_items,
        "subtotal": subtotal,
        "discount": discount,
        "surcharge": surcharge,
        "total": total,
        "generated_at": now.isoformat(),
    }


# =============================================================================
# CHARACTERIZATION TESTS
# =============================================================================
# These tests document CURRENT behavior, not desired behavior.
# They are a safety net for refactoring — if behavior changes, tests break.

class TestCharacterization:
    """Characterization tests: pin down existing behavior before refactoring."""

    def test_basic_order(self):
        """Document the happy path behavior."""
        items = [{"sku": "WIDGET_A", "qty": 2}]
        with patch("__main__.get_current_time", return_value=datetime(2024, 1, 15, 10, 0)):
            result = calculate_order_total(items)

        assert result["subtotal"] == 59.98
        assert result["discount"] == 0.0
        assert result["total"] == 59.98

    def test_unknown_sku_silently_skipped(self):
        """Document the legacy behavior: unknown SKUs are ignored.
        This might be a bug, but we document it first, fix later."""
        items = [
            {"sku": "WIDGET_A", "qty": 1},
            {"sku": "NONEXISTENT", "qty": 5},
        ]
        with patch("__main__.get_current_time", return_value=datetime(2024, 1, 15, 10, 0)):
            result = calculate_order_total(items)

        # Only WIDGET_A is counted — NONEXISTENT is silently dropped
        assert len(result["line_items"]) == 1
        assert result["subtotal"] == 29.99

    def test_loyal_customer_discount(self):
        """Pin the 10% loyalty discount behavior."""
        items = [{"sku": "WIDGET_C", "qty": 1}]
        with patch("__main__.get_current_time", return_value=datetime(2024, 1, 15, 10, 0)):
            result = calculate_order_total(items, customer_type="LOYAL")

        assert result["discount"] == 10.0  # 10% of 99.99
        assert result["total"] == 89.99

    def test_weekend_surcharge(self):
        """Document the weekend surcharge behavior."""
        items = [{"sku": "WIDGET_B", "qty": 1}]
        # Saturday
        saturday = datetime(2024, 1, 13, 10, 0)
        with patch("__main__.get_current_time", return_value=saturday):
            result = calculate_order_total(items)

        assert result["surcharge"] == 2.5  # 5% of 49.99 rounded
        assert result["total"] == 52.49

    def test_weekday_no_surcharge(self):
        """Verify no surcharge on weekdays."""
        items = [{"sku": "WIDGET_B", "qty": 1}]
        monday = datetime(2024, 1, 15, 10, 0)
        with patch("__main__.get_current_time", return_value=monday):
            result = calculate_order_total(items)

        assert result["surcharge"] == 0.0


# =============================================================================
# SEAM TECHNIQUE — MAKING LEGACY CODE TESTABLE
# =============================================================================
# A "seam" is a place where you can change behavior without editing the code.
# Common seams: subclass-and-override, dependency injection, monkey-patching.

class OrderCalculator:
    """Refactored version using dependency injection (seams).
    The legacy function's hidden dependencies become constructor parameters."""

    def __init__(self, pricing: dict = None, discounts: dict = None, clock=None):
        self.pricing = pricing or _PRICING_DB
        self.discounts = discounts or _DISCOUNT_RULES
        self.clock = clock or get_current_time

    def calculate(self, items: list[dict], customer_type: str = "REGULAR") -> dict:
        """Same logic as legacy function, but dependencies are injectable."""
        subtotal = 0.0
        line_items = []

        for item in items:
            sku = item["sku"]
            qty = item["qty"]
            if sku not in self.pricing:
                continue
            price = self.pricing[sku]
            line_total = price * qty
            subtotal += line_total
            line_items.append({"sku": sku, "qty": qty, "price": price, "total": line_total})

        discount_rate = self.discounts.get(customer_type, 0.0)
        discount = round(subtotal * discount_rate, 2)

        now = self.clock()
        surcharge = round(subtotal * 0.05, 2) if now.weekday() >= 5 else 0.0

        total = round(subtotal - discount + surcharge, 2)
        return {
            "line_items": line_items,
            "subtotal": subtotal,
            "discount": discount,
            "surcharge": surcharge,
            "total": total,
        }


class TestRefactoredWithSeams:
    """Tests using injected dependencies — no patching needed."""

    def test_custom_pricing(self):
        """Inject custom pricing data — test is fully self-contained."""
        calc = OrderCalculator(
            pricing={"TEST_SKU": 10.0},
            clock=lambda: datetime(2024, 1, 15),  # Monday
        )
        result = calc.calculate([{"sku": "TEST_SKU", "qty": 3}])
        assert result["subtotal"] == 30.0

    def test_weekend_surcharge_via_injection(self):
        """Control the clock via injection — deterministic test."""
        calc = OrderCalculator(
            pricing={"X": 100.0},
            clock=lambda: datetime(2024, 1, 13),  # Saturday
        )
        result = calc.calculate([{"sku": "X", "qty": 1}])
        assert result["surcharge"] == 5.0

    def test_unknown_discount_type_returns_zero(self):
        calc = OrderCalculator(
            pricing={"X": 100.0},
            discounts={},
            clock=lambda: datetime(2024, 1, 15),
        )
        result = calc.calculate([{"sku": "X", "qty": 1}], customer_type="VIP")
        assert result["discount"] == 0.0


# =============================================================================
# STRANGLER FIG PATTERN
# =============================================================================
# Gradually replace legacy code by routing calls through a new implementation
# while keeping the old one as a fallback.

class StranglerOrderService:
    """Facade that delegates to new or legacy implementation.
    New features go to the new calculator; legacy paths stay until migrated."""

    def __init__(self, new_calc: OrderCalculator, use_new: bool = False):
        self.new_calc = new_calc
        self.use_new = use_new

    def calculate(self, items: list[dict], customer_type: str = "REGULAR") -> dict:
        if self.use_new:
            return self.new_calc.calculate(items, customer_type)
        else:
            return calculate_order_total(items, customer_type)


class TestStranglerPattern:
    """Verify that old and new paths produce equivalent results."""

    def test_new_matches_legacy(self):
        """The strangler produces the same output as the legacy code."""
        items = [{"sku": "WIDGET_A", "qty": 2}, {"sku": "WIDGET_B", "qty": 1}]
        fixed_time = datetime(2024, 1, 15, 10, 0)

        with patch("__main__.get_current_time", return_value=fixed_time):
            legacy_result = calculate_order_total(items, "LOYAL")

        new_calc = OrderCalculator(clock=lambda: fixed_time)
        strangler = StranglerOrderService(new_calc, use_new=True)
        new_result = strangler.calculate(items, "LOYAL")

        assert new_result["subtotal"] == legacy_result["subtotal"]
        assert new_result["discount"] == legacy_result["discount"]
        assert new_result["total"] == legacy_result["total"]


# =============================================================================
# SPROUT METHOD PATTERN
# =============================================================================
# Add new functionality in a tested method, call it from legacy code.

def validate_order_items(items: list[dict], valid_skus: set) -> list[str]:
    """SPROUT: New validation logic, fully tested, called from legacy path.
    Returns list of error messages (empty = valid)."""
    errors = []
    for item in items:
        if "sku" not in item or "qty" not in item:
            errors.append(f"Missing required fields in item: {item}")
        elif item["sku"] not in valid_skus:
            errors.append(f"Unknown SKU: {item['sku']}")
        elif item["qty"] <= 0:
            errors.append(f"Invalid quantity for {item['sku']}: {item['qty']}")
    return errors


class TestSproutMethod:
    """The sprout method is tested independently of the legacy code."""

    def test_valid_items(self):
        errors = validate_order_items(
            [{"sku": "A", "qty": 1}], valid_skus={"A", "B"}
        )
        assert errors == []

    def test_unknown_sku(self):
        errors = validate_order_items(
            [{"sku": "UNKNOWN", "qty": 1}], valid_skus={"A"}
        )
        assert len(errors) == 1
        assert "Unknown SKU" in errors[0]

    def test_invalid_quantity(self):
        errors = validate_order_items(
            [{"sku": "A", "qty": 0}], valid_skus={"A"}
        )
        assert "Invalid quantity" in errors[0]

    def test_missing_fields(self):
        errors = validate_order_items([{"sku": "A"}], valid_skus={"A"})
        assert "Missing required fields" in errors[0]


# =============================================================================
# RUNNING THIS FILE
# =============================================================================
# pytest 17_legacy_testing.py -v

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
