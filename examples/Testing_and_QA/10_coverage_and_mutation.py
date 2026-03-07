#!/usr/bin/env python3
"""Example: Coverage Analysis and Mutation Testing

Demonstrates code coverage measurement with coverage.py/pytest-cov,
coverage interpretation, mutation testing with mutmut, and strategies
for meaningful coverage targets.
Related lesson: 13_Code_Coverage.md, 14_Mutation_Testing.md
"""

# =============================================================================
# WHY COVERAGE AND MUTATION TESTING?
#
# Coverage answers: "Which lines did my tests execute?"
# Mutation testing answers: "Would my tests catch real bugs?"
#
# Coverage alone is misleading — 100% line coverage does NOT mean
# your tests are good. You can cover every line without asserting anything.
# Mutation testing validates that your tests actually detect changes
# (mutations) to your code.
#
# Together, they give a complete picture of test suite quality:
#   1. Coverage: identifies UNTESTED code
#   2. Mutation: identifies WEAK tests (tests that pass even when code is wrong)
# =============================================================================

import pytest


# =============================================================================
# PRODUCTION CODE
# =============================================================================
# This module has various code paths to demonstrate coverage analysis.

class PriceCalculator:
    """Calculate prices with discounts, taxes, and promotions."""

    TAX_RATES = {
        "US": 0.08,
        "UK": 0.20,
        "DE": 0.19,
        "JP": 0.10,
        "KR": 0.10,
    }

    def __init__(self, country: str = "US"):
        if country not in self.TAX_RATES:
            raise ValueError(f"Unsupported country: {country}")
        self.country = country
        self.tax_rate = self.TAX_RATES[country]

    def calculate_subtotal(self, items: list[dict]) -> float:
        """Calculate subtotal from a list of items.
        Each item has 'price' and 'quantity'."""
        if not items:
            return 0.0

        subtotal = 0.0
        for item in items:
            if item["price"] < 0:
                raise ValueError("Price cannot be negative")
            if item["quantity"] < 0:
                raise ValueError("Quantity cannot be negative")
            subtotal += item["price"] * item["quantity"]
        return round(subtotal, 2)

    def apply_discount(self, subtotal: float, discount_code: str = None) -> float:
        """Apply discount based on code or subtotal threshold."""
        if discount_code == "SAVE10":
            return round(subtotal * 0.90, 2)
        elif discount_code == "SAVE20":
            return round(subtotal * 0.80, 2)
        elif discount_code == "HALF":
            return round(subtotal * 0.50, 2)
        elif discount_code is not None:
            raise ValueError(f"Invalid discount code: {discount_code}")

        # Automatic volume discount (no code needed)
        if subtotal > 500:
            return round(subtotal * 0.95, 2)  # 5% off for orders > $500
        elif subtotal > 1000:
            # BUG: This branch is unreachable! subtotal > 500 catches it first.
            # Mutation testing would catch this — the branch never executes.
            return round(subtotal * 0.90, 2)

        return subtotal

    def calculate_tax(self, amount: float) -> float:
        """Calculate tax for the given amount."""
        return round(amount * self.tax_rate, 2)

    def calculate_shipping(self, subtotal: float, express: bool = False) -> float:
        """Calculate shipping cost based on order value."""
        if subtotal >= 100:
            base_shipping = 0.0  # Free shipping over $100
        elif subtotal >= 50:
            base_shipping = 5.99
        else:
            base_shipping = 9.99

        if express:
            base_shipping += 15.00

        return round(base_shipping, 2)

    def calculate_total(
        self,
        items: list[dict],
        discount_code: str = None,
        express_shipping: bool = False,
    ) -> dict:
        """Calculate the complete order total."""
        subtotal = self.calculate_subtotal(items)
        discounted = self.apply_discount(subtotal, discount_code)
        tax = self.calculate_tax(discounted)
        shipping = self.calculate_shipping(discounted, express_shipping)

        return {
            "subtotal": subtotal,
            "discount": round(subtotal - discounted, 2),
            "tax": tax,
            "shipping": shipping,
            "total": round(discounted + tax + shipping, 2),
        }


def categorize_score(score: int) -> str:
    """Categorize a test score into a grade.
    This function has clear branch points for coverage analysis."""
    if not isinstance(score, (int, float)):
        raise TypeError("Score must be a number")
    if score < 0 or score > 100:
        raise ValueError("Score must be between 0 and 100")

    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"


def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number iteratively."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return n

    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


# =============================================================================
# TESTS WITH COVERAGE FOCUS
# =============================================================================
# These tests are designed to demonstrate coverage concepts.
# Run with: pytest --cov=. --cov-report=term-missing 10_coverage_and_mutation.py

class TestPriceCalculator:
    """Tests for PriceCalculator — designed to show coverage analysis."""

    @pytest.fixture
    def calc(self):
        return PriceCalculator("US")

    # --- Subtotal tests ---

    def test_subtotal_empty(self, calc):
        """Covers the early return for empty items list."""
        assert calc.calculate_subtotal([]) == 0.0

    def test_subtotal_single_item(self, calc):
        items = [{"price": 10.00, "quantity": 2}]
        assert calc.calculate_subtotal(items) == 20.00

    def test_subtotal_multiple_items(self, calc):
        items = [
            {"price": 10.00, "quantity": 2},
            {"price": 5.50, "quantity": 3},
        ]
        assert calc.calculate_subtotal(items) == 36.50

    def test_subtotal_negative_price(self, calc):
        """Covers the negative price validation branch."""
        with pytest.raises(ValueError, match="Price cannot be negative"):
            calc.calculate_subtotal([{"price": -5.00, "quantity": 1}])

    def test_subtotal_negative_quantity(self, calc):
        """Covers the negative quantity validation branch."""
        with pytest.raises(ValueError, match="Quantity cannot be negative"):
            calc.calculate_subtotal([{"price": 5.00, "quantity": -1}])

    # --- Discount tests ---

    def test_discount_save10(self, calc):
        """Covers the SAVE10 branch."""
        assert calc.apply_discount(100.0, "SAVE10") == 90.0

    def test_discount_save20(self, calc):
        """Covers the SAVE20 branch."""
        assert calc.apply_discount(100.0, "SAVE20") == 80.0

    def test_discount_half(self, calc):
        """Covers the HALF branch."""
        assert calc.apply_discount(100.0, "HALF") == 50.0

    def test_discount_invalid_code(self, calc):
        """Covers the invalid code branch."""
        with pytest.raises(ValueError, match="Invalid discount code"):
            calc.apply_discount(100.0, "BADCODE")

    def test_discount_volume_over_500(self, calc):
        """Covers the automatic 5% discount for orders > $500."""
        result = calc.apply_discount(600.0)
        assert result == 570.0

    def test_discount_no_discount(self, calc):
        """Covers the default return (no discount applied)."""
        assert calc.apply_discount(100.0) == 100.0

    # NOTE: The subtotal > 1000 branch is UNREACHABLE because the > 500
    # check comes first. Coverage report will show this as uncovered.
    # This is exactly the kind of bug that coverage analysis reveals!

    # --- Tax tests ---

    def test_tax_us(self, calc):
        assert calc.calculate_tax(100.0) == 8.0

    def test_tax_uk(self):
        calc = PriceCalculator("UK")
        assert calc.calculate_tax(100.0) == 20.0

    # --- Shipping tests ---

    def test_shipping_free(self, calc):
        """Covers subtotal >= 100 branch (free shipping)."""
        assert calc.calculate_shipping(150.0) == 0.0

    def test_shipping_medium(self, calc):
        """Covers subtotal >= 50 branch."""
        assert calc.calculate_shipping(75.0) == 5.99

    def test_shipping_standard(self, calc):
        """Covers subtotal < 50 branch."""
        assert calc.calculate_shipping(30.0) == 9.99

    def test_shipping_express(self, calc):
        """Covers the express shipping branch."""
        assert calc.calculate_shipping(30.0, express=True) == 24.99

    # --- Integration test ---

    def test_calculate_total(self, calc):
        """End-to-end test covering the full calculation pipeline."""
        items = [
            {"price": 25.00, "quantity": 2},
            {"price": 15.00, "quantity": 1},
        ]
        result = calc.calculate_total(items, discount_code="SAVE10")

        assert result["subtotal"] == 65.0
        assert result["discount"] == 6.50
        assert result["tax"] == 4.68  # 58.50 * 0.08
        assert result["shipping"] == 5.99  # 50 <= 58.50 < 100
        assert result["total"] == 69.17

    # --- Constructor test ---

    def test_unsupported_country(self):
        """Covers the ValueError branch in __init__."""
        with pytest.raises(ValueError, match="Unsupported country"):
            PriceCalculator("XX")


class TestCategorizeScore:
    """100% branch coverage for categorize_score."""

    @pytest.mark.parametrize("score, expected", [
        (95, "A"),   # score >= 90
        (90, "A"),   # boundary: exactly 90
        (85, "B"),   # 80 <= score < 90
        (80, "B"),   # boundary: exactly 80
        (75, "C"),   # 70 <= score < 80
        (65, "D"),   # 60 <= score < 70
        (55, "F"),   # score < 60
        (0, "F"),    # boundary: zero
        (100, "A"),  # boundary: max
    ])
    def test_grade_categories(self, score, expected):
        assert categorize_score(score) == expected

    def test_invalid_type(self):
        with pytest.raises(TypeError):
            categorize_score("ninety")

    def test_out_of_range_high(self):
        with pytest.raises(ValueError):
            categorize_score(101)

    def test_out_of_range_low(self):
        with pytest.raises(ValueError):
            categorize_score(-1)


class TestFibonacci:
    """Tests demonstrating boundary value coverage."""

    @pytest.mark.parametrize("n, expected", [
        (0, 0),   # base case: n == 0
        (1, 1),   # base case: n == 1
        (2, 1),   # first computed value
        (5, 5),
        (10, 55),
        (20, 6765),
    ])
    def test_fibonacci_values(self, n, expected):
        assert fibonacci(n) == expected

    def test_fibonacci_negative(self):
        with pytest.raises(ValueError):
            fibonacci(-1)


# =============================================================================
# COVERAGE AND MUTATION TESTING CONFIGURATION
# =============================================================================
# Below are configuration examples. In a real project, put these in
# pyproject.toml or setup.cfg.

PYPROJECT_TOML_EXAMPLE = """
# === pyproject.toml ===
# Coverage configuration
[tool.pytest.ini_options]
addopts = "--cov=src --cov-report=term-missing --cov-report=html"
# Run with: pytest
# This automatically generates coverage reports

[tool.coverage.run]
source = ["src"]
branch = true              # Enable BRANCH coverage (not just line)
omit = [
    "*/tests/*",
    "*/migrations/*",
    "*/__pycache__/*",
]

[tool.coverage.report]
# Fail the build if coverage drops below threshold
fail_under = 80
show_missing = true
exclude_lines = [
    "pragma: no cover",    # Explicit exclusion marker
    "def __repr__",        # Repr methods rarely need testing
    "if TYPE_CHECKING:",   # Import-only blocks
    "raise NotImplementedError",
]

[tool.coverage.html]
directory = "htmlcov"

# === Mutation Testing with mutmut ===
[tool.mutmut]
paths_to_mutate = "src/"
tests_dir = "tests/"
runner = "python -m pytest"
# Run with: mutmut run
# View results: mutmut results
# Inspect survivor: mutmut show <id>
"""

# =============================================================================
# INTERPRETING COVERAGE REPORTS
# =============================================================================
# Run: pytest --cov=. --cov-report=term-missing 10_coverage_and_mutation.py
#
# Sample output:
# Name                          Stmts   Miss Branch BrPart  Cover   Missing
# -----------------------------------------------------------------------
# 10_coverage_and_mutation.py     105      2     30      2    97%   82, 84
#
# Key metrics:
#   Stmts:   Total executable statements
#   Miss:    Statements not executed by any test
#   Branch:  Total branches (if/else)
#   BrPart:  Partially covered branches (only true OR false was tested)
#   Cover:   Overall coverage percentage
#   Missing: Line numbers not covered
#
# The missing lines 82, 84 correspond to the unreachable > 1000 branch
# in apply_discount — this reveals a real bug!

# =============================================================================
# MUTATION TESTING WORKFLOW
# =============================================================================
# 1. Install: pip install mutmut
# 2. Run:     mutmut run --paths-to-mutate=mymodule.py
# 3. Review:  mutmut results
#
# Mutmut modifies your code (mutations) and runs your tests:
#   - KILLED:    Test caught the mutation (GOOD)
#   - SURVIVED:  Test did NOT catch the mutation (BAD — test is weak)
#   - TIMEOUT:   Mutation caused infinite loop (usually OK)
#
# Example mutations mutmut applies:
#   - Replace > with >=, <, <=
#   - Replace + with -, * with /
#   - Replace True with False
#   - Remove function calls
#   - Replace return values
#
# Example: If mutmut changes `score >= 90` to `score >= 91` and ALL tests
# still pass, your boundary test is missing — you don't test score=90.

# =============================================================================
# BEST PRACTICES
# =============================================================================
#
# 1. Aim for high BRANCH coverage, not just line coverage
#    - Branch coverage counts True AND False paths of each conditional
#    - Line coverage might show 100% even if only one branch was taken
#
# 2. Don't chase 100% coverage blindly
#    - Focus on critical paths and business logic
#    - Some code (error handlers, defensive programming) is hard to test
#    - Mark intentionally untested code with "# pragma: no cover"
#
# 3. Use coverage to FIND untested code, not as a quality metric
#    - High coverage + weak assertions = false confidence
#    - Mutation testing reveals assertion quality
#
# 4. Set a coverage FLOOR in CI to prevent regression
#    - Start with your current coverage, increase gradually
#    - Use fail_under in coverage config
#
# 5. Review coverage reports regularly
#    - New features should have tests (coverage shows gaps)
#    - Refactoring should not drop coverage
#    - HTML reports make it easy to spot patterns

# =============================================================================
# RUNNING THIS FILE
# =============================================================================
# Basic test run:
#   pytest 10_coverage_and_mutation.py -v
#
# With coverage:
#   pytest --cov=. --cov-report=term-missing --cov-branch 10_coverage_and_mutation.py
#
# HTML coverage report:
#   pytest --cov=. --cov-report=html 10_coverage_and_mutation.py
#   open htmlcov/index.html
#
# Mutation testing:
#   pip install mutmut
#   mutmut run --paths-to-mutate=10_coverage_and_mutation.py

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
