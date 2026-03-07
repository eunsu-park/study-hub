#!/usr/bin/env python3
"""Example: TDD Calculator

Demonstrates Test-Driven Development by building a calculator step-by-step.
Each section shows the RED-GREEN-REFACTOR cycle:
  1. RED:    Write a failing test
  2. GREEN:  Write the minimum code to pass
  3. REFACTOR: Improve the code while keeping tests green
Related lesson: 05_Test_Driven_Development.md
"""

# =============================================================================
# WHY TDD?
# TDD inverts the usual workflow: tests come FIRST, then implementation.
# Benefits:
#   - Forces you to think about the interface before implementation
#   - Every line of production code is backed by a test
#   - Gives you confidence to refactor aggressively
#   - Produces better-designed, more modular code
#   - Catches regressions immediately
# =============================================================================

import pytest
from typing import List


# =============================================================================
# THE CALCULATOR — Built incrementally via TDD
# =============================================================================
# In a real TDD session, you would write ONE test, then ONE implementation,
# then refactor. Here we show the final result with comments explaining
# the TDD journey.

class CalculatorError(Exception):
    """Custom exception for calculator errors.
    TDD iteration 4: We needed this when testing division by zero."""
    pass


class Calculator:
    """A calculator built via TDD.

    TDD iteration history:
      1. Basic addition
      2. Subtraction, multiplication
      3. Division with error handling
      4. Expression history
      5. Chaining operations
      6. Memory operations
    """

    def __init__(self):
        # Iteration 4: Added history tracking after writing test_history
        self._history: List[str] = []
        # Iteration 6: Added memory after writing test_memory_store
        self._memory: float = 0.0

    def add(self, a: float, b: float) -> float:
        """Iteration 1: First feature — driven by test_add.
        Started with just 'return a + b', then added history in iteration 4."""
        result = a + b
        self._history.append(f"{a} + {b} = {result}")
        return result

    def subtract(self, a: float, b: float) -> float:
        """Iteration 2: test_subtract drove this implementation."""
        result = a - b
        self._history.append(f"{a} - {b} = {result}")
        return result

    def multiply(self, a: float, b: float) -> float:
        """Iteration 2: test_multiply drove this implementation."""
        result = a * b
        self._history.append(f"{a} * {b} = {result}")
        return result

    def divide(self, a: float, b: float) -> float:
        """Iteration 3: test_divide_by_zero drove the error handling.
        Without TDD, we might have forgotten the zero check."""
        if b == 0:
            raise CalculatorError("Cannot divide by zero")
        result = a / b
        self._history.append(f"{a} / {b} = {result}")
        return result

    def power(self, base: float, exponent: float) -> float:
        """Iteration 5: test_power and test_negative_exponent drove this."""
        result = base ** exponent
        self._history.append(f"{base} ^ {exponent} = {result}")
        return result

    def sqrt(self, value: float) -> float:
        """Iteration 5: test_sqrt_negative drove the validation."""
        if value < 0:
            raise CalculatorError("Cannot compute square root of negative number")
        result = value ** 0.5
        self._history.append(f"sqrt({value}) = {result}")
        return result

    @property
    def history(self) -> List[str]:
        """Iteration 4: test_history showed we needed operation tracking."""
        return self._history.copy()  # Return copy to prevent external mutation

    def clear_history(self) -> None:
        """Iteration 4: test_clear_history drove this."""
        self._history.clear()

    def memory_store(self, value: float) -> None:
        """Iteration 6: Memory operations — common calculator feature."""
        self._memory = value

    def memory_recall(self) -> float:
        """Iteration 6: Returns stored memory value."""
        return self._memory

    def memory_clear(self) -> None:
        """Iteration 6: Reset memory to zero."""
        self._memory = 0.0

    def memory_add(self, value: float) -> None:
        """Iteration 6: Add to memory (M+)."""
        self._memory += value


# =============================================================================
# FIXTURE
# =============================================================================

@pytest.fixture
def calc():
    """Fresh calculator for each test — ensures isolation."""
    return Calculator()


# =============================================================================
# ITERATION 1: BASIC ADDITION (RED -> GREEN -> REFACTOR)
# =============================================================================
# RED: Wrote this test first, before Calculator.add existed -> NameError
# GREEN: Implemented Calculator with add() returning a + b
# REFACTOR: Nothing to refactor yet

class TestAddition:
    """TDD started here: the very first test."""

    def test_add_positive_numbers(self, calc):
        assert calc.add(2, 3) == 5

    def test_add_negative_numbers(self, calc):
        """After GREEN, we add more cases to build confidence."""
        assert calc.add(-1, -1) == -2

    def test_add_zero(self, calc):
        """Edge case discovered during TDD: identity element."""
        assert calc.add(5, 0) == 5

    def test_add_floats(self, calc):
        """Float support — came naturally since Python handles it."""
        assert calc.add(1.5, 2.5) == pytest.approx(4.0)


# =============================================================================
# ITERATION 2: SUBTRACTION AND MULTIPLICATION
# =============================================================================
# RED: test_subtract failed -> no subtract method
# GREEN: Added subtract returning a - b
# REFACTOR: Noticed add/subtract/multiply have identical structure
#           but kept them separate for clarity (YAGNI)

class TestSubtraction:
    def test_subtract(self, calc):
        assert calc.subtract(10, 4) == 6

    def test_subtract_resulting_negative(self, calc):
        assert calc.subtract(3, 7) == -4


class TestMultiplication:
    def test_multiply(self, calc):
        assert calc.multiply(3, 4) == 12

    def test_multiply_by_zero(self, calc):
        """Edge case: anything times zero is zero."""
        assert calc.multiply(100, 0) == 0

    def test_multiply_negatives(self, calc):
        """Negative times negative is positive."""
        assert calc.multiply(-3, -4) == 12


# =============================================================================
# ITERATION 3: DIVISION WITH ERROR HANDLING
# =============================================================================
# RED: test_divide_by_zero failed -> no error raised
# GREEN: Added zero check raising CalculatorError
# REFACTOR: Created CalculatorError instead of using generic ValueError

class TestDivision:
    def test_divide(self, calc):
        assert calc.divide(10, 2) == 5.0

    def test_divide_float_result(self, calc):
        """Division should always return float in Python 3."""
        assert calc.divide(7, 2) == 3.5

    def test_divide_by_zero(self, calc):
        """THIS TEST drove the creation of CalculatorError.
        Without TDD, we might have forgotten error handling or used
        a generic exception type."""
        with pytest.raises(CalculatorError, match="Cannot divide by zero"):
            calc.divide(10, 0)


# =============================================================================
# ITERATION 4: HISTORY TRACKING
# =============================================================================
# RED: test_history failed -> no history attribute
# GREEN: Added self._history list and history property
# REFACTOR: Moved history.append into each method (DRY would suggest
#           a decorator, but that is premature optimization here)

class TestHistory:
    def test_history_starts_empty(self, calc):
        assert calc.history == []

    def test_history_records_operations(self, calc):
        """History was a new REQUIREMENT discovered during design.
        TDD forced us to define the exact format upfront."""
        calc.add(1, 2)
        calc.multiply(3, 4)
        assert len(calc.history) == 2
        assert calc.history[0] == "1 + 2 = 3"
        assert calc.history[1] == "3 * 4 = 12"

    def test_clear_history(self, calc):
        calc.add(1, 1)
        calc.clear_history()
        assert calc.history == []

    def test_history_is_immutable_copy(self, calc):
        """Test drove us to return a copy, preventing external mutation."""
        calc.add(1, 1)
        history = calc.history
        history.append("hacked!")
        assert len(calc.history) == 1  # Original unchanged


# =============================================================================
# ITERATION 5: POWER AND SQUARE ROOT
# =============================================================================
# RED: test_sqrt_negative failed -> no validation
# GREEN: Added validation with CalculatorError
# REFACTOR: Reused CalculatorError from division

class TestPower:
    def test_power(self, calc):
        assert calc.power(2, 3) == 8

    def test_power_zero_exponent(self, calc):
        assert calc.power(5, 0) == 1

    def test_negative_exponent(self, calc):
        assert calc.power(2, -1) == pytest.approx(0.5)


class TestSquareRoot:
    def test_sqrt(self, calc):
        assert calc.sqrt(16) == 4.0

    def test_sqrt_non_perfect(self, calc):
        assert calc.sqrt(2) == pytest.approx(1.41421356)

    def test_sqrt_zero(self, calc):
        assert calc.sqrt(0) == 0.0

    def test_sqrt_negative(self, calc):
        """This test DROVE the negative validation in sqrt().
        Without TDD, we might have let it return a complex number."""
        with pytest.raises(CalculatorError, match="negative"):
            calc.sqrt(-4)


# =============================================================================
# ITERATION 6: MEMORY OPERATIONS
# =============================================================================
# RED: test_memory_store failed -> no memory attributes
# GREEN: Added _memory and memory methods
# REFACTOR: Simple enough — no refactoring needed

class TestMemory:
    def test_memory_starts_at_zero(self, calc):
        assert calc.memory_recall() == 0.0

    def test_memory_store_and_recall(self, calc):
        calc.memory_store(42.0)
        assert calc.memory_recall() == 42.0

    def test_memory_clear(self, calc):
        calc.memory_store(100.0)
        calc.memory_clear()
        assert calc.memory_recall() == 0.0

    def test_memory_add(self, calc):
        """M+ button: accumulate values in memory."""
        calc.memory_add(10.0)
        calc.memory_add(5.0)
        assert calc.memory_recall() == 15.0

    def test_memory_workflow(self, calc):
        """Integration test: realistic usage pattern."""
        result = calc.add(10, 20)       # 30
        calc.memory_store(result)       # M = 30
        result2 = calc.multiply(5, 6)   # 30
        calc.memory_add(result2)        # M = 60
        assert calc.memory_recall() == 60.0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================
# After unit tests for each feature, write integration tests that
# exercise realistic workflows spanning multiple features.

class TestIntegration:
    def test_complex_calculation_with_history(self, calc):
        """Realistic multi-step calculation with history verification."""
        a = calc.add(100, 50)       # 150
        b = calc.multiply(a, 2)     # 300
        c = calc.divide(b, 3)       # 100
        d = calc.subtract(c, 25)    # 75

        assert d == 75.0
        assert len(calc.history) == 4

    def test_error_does_not_corrupt_history(self, calc):
        """Failed operations should NOT appear in history."""
        calc.add(1, 1)

        with pytest.raises(CalculatorError):
            calc.divide(1, 0)

        # Only the successful operation is in history
        assert len(calc.history) == 1


# =============================================================================
# TDD SUMMARY
# =============================================================================
# This calculator was built through 6 TDD iterations:
#   1. Addition          -> basic structure
#   2. Sub/Mul           -> expanded arithmetic
#   3. Division          -> error handling pattern established
#   4. History           -> new feature requirement
#   5. Power/Sqrt        -> reused error pattern from step 3
#   6. Memory            -> independent feature, cleanly added
#
# Each iteration followed RED -> GREEN -> REFACTOR:
#   RED:     Write a test that fails
#   GREEN:   Write minimal code to pass
#   REFACTOR: Improve without changing behavior (tests prove it)
#
# Key TDD insight: The tests DROVE the design. We didn't plan
# CalculatorError or history immutability upfront — the tests
# showed us we needed them.

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
