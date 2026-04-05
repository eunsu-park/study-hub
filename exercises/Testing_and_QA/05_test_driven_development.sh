#!/bin/bash
# Exercises for Lesson 05: Test-Driven Development
# Topic: Testing_and_QA
# Solutions to practice problems from the lesson.

# === Exercise 1: TDD Red-Green-Refactor for a Stack ===
# Problem: Build a stack data structure using TDD. Show each iteration
# of the Red-Green-Refactor cycle.
exercise_1() {
    echo "=== Exercise 1: TDD Red-Green-Refactor for a Stack ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import pytest

# ITERATION 1 — RED: test_new_stack_is_empty fails (Stack doesn't exist)
# GREEN: Create Stack class with is_empty returning True
# REFACTOR: Nothing to refactor yet

# ITERATION 2 — RED: test_push_makes_non_empty fails
# GREEN: Add push method that sets a flag
# REFACTOR: Use a list instead of a flag

# FINAL RESULT after all TDD iterations:

class Stack:
    def __init__(self):
        self._items = []

    def is_empty(self) -> bool:
        return len(self._items) == 0

    def push(self, item):
        self._items.append(item)

    def pop(self):
        if self.is_empty():
            raise IndexError("Cannot pop from empty stack")
        return self._items.pop()

    def peek(self):
        if self.is_empty():
            raise IndexError("Cannot peek at empty stack")
        return self._items[-1]

    def size(self) -> int:
        return len(self._items)

# Tests written in TDD order (each drove a specific implementation):

def test_new_stack_is_empty():
    """Iteration 1: The very first test."""
    s = Stack()
    assert s.is_empty() is True

def test_push_makes_non_empty():
    """Iteration 2: Push was added to pass this."""
    s = Stack()
    s.push(42)
    assert s.is_empty() is False

def test_pop_returns_last_pushed():
    """Iteration 3: Pop was added."""
    s = Stack()
    s.push(42)
    assert s.pop() == 42

def test_pop_removes_element():
    """Iteration 3b: Verify pop actually removes."""
    s = Stack()
    s.push(42)
    s.pop()
    assert s.is_empty() is True

def test_pop_empty_raises():
    """Iteration 4: Error handling was added."""
    s = Stack()
    with pytest.raises(IndexError, match="Cannot pop"):
        s.pop()

def test_peek_returns_without_removing():
    """Iteration 5: Peek was added."""
    s = Stack()
    s.push(99)
    assert s.peek() == 99
    assert s.is_empty() is False  # Still there

def test_lifo_order():
    """Iteration 6: Verify LIFO behavior."""
    s = Stack()
    s.push(1)
    s.push(2)
    s.push(3)
    assert s.pop() == 3
    assert s.pop() == 2
    assert s.pop() == 1

def test_size():
    """Iteration 7: size method was added."""
    s = Stack()
    assert s.size() == 0
    s.push("a")
    s.push("b")
    assert s.size() == 2
SOLUTION
}

# === Exercise 2: TDD for String Calculator ===
# Problem: Implement a string calculator using TDD.
# The function takes a string of comma-separated numbers and returns their sum.
exercise_2() {
    echo "=== Exercise 2: TDD for String Calculator ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import pytest

def string_calculator(numbers: str) -> int:
    """Add numbers given as a string.
    Rules (each discovered through TDD):
    - Empty string returns 0
    - Single number returns that number
    - Multiple comma-separated numbers are summed
    - Newlines work as delimiters too
    - Negative numbers raise ValueError
    """
    if not numbers:
        return 0

    # Support both comma and newline delimiters
    numbers = numbers.replace("\n", ",")
    parts = numbers.split(",")
    values = [int(p.strip()) for p in parts if p.strip()]

    # Check for negatives (discovered via test_negative_raises)
    negatives = [v for v in values if v < 0]
    if negatives:
        raise ValueError(f"Negatives not allowed: {negatives}")

    return sum(values)

# Tests in TDD order:

# Iteration 1: Empty string
def test_empty_string():
    assert string_calculator("") == 0

# Iteration 2: Single number
def test_single_number():
    assert string_calculator("5") == 5

# Iteration 3: Two numbers
def test_two_numbers():
    assert string_calculator("1,2") == 3

# Iteration 4: Multiple numbers
def test_multiple_numbers():
    assert string_calculator("1,2,3,4,5") == 15

# Iteration 5: Newline delimiter
def test_newline_delimiter():
    assert string_calculator("1\n2,3") == 6

# Iteration 6: Negative number handling
def test_negative_raises():
    with pytest.raises(ValueError, match="Negatives not allowed"):
        string_calculator("1,-2,3")

def test_multiple_negatives_listed():
    with pytest.raises(ValueError, match=r"\[-2, -4\]"):
        string_calculator("1,-2,3,-4")

# Iteration 7: Whitespace handling
def test_whitespace_around_numbers():
    assert string_calculator(" 1 , 2 , 3 ") == 6
SOLUTION
}

# === Exercise 3: TDD Anti-Patterns ===
# Problem: Identify TDD anti-patterns in given code and explain
# how to fix them.
exercise_3() {
    echo "=== Exercise 3: TDD Anti-Patterns ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# ANTI-PATTERN 1: Writing all tests first, then implementing
# Problem: You lose the Red-Green feedback loop
# Fix: Write ONE test, make it pass, then write the next test

# ANTI-PATTERN 2: Writing too much code to make a test pass
# BAD:
def test_sort_list():
    assert sort([3,1,2]) == [1,2,3]
# Then implementing quicksort with 50 lines
# FIX: Start with the simplest implementation (even if naive):
#   return sorted(input_list)  # Refactor later if needed

# ANTI-PATTERN 3: Testing implementation details
# BAD:
def test_cache_uses_dict():
    cache = Cache()
    assert isinstance(cache._storage, dict)  # Private attribute!
# FIX: Test behavior, not structure:
def test_cache_stores_and_retrieves():
    cache = Cache()
    cache.set("key", "value")
    assert cache.get("key") == "value"

# ANTI-PATTERN 4: Skipping the Refactor step
# Problem: Code works but accumulates tech debt
# Fix: After GREEN, always ask:
#   - Can I remove duplication?
#   - Can I improve naming?
#   - Can I simplify the logic?
#   The tests protect you during refactoring!

# ANTI-PATTERN 5: Gold plating (adding untested features)
# BAD: Adding a "clear_all" method without a test driving it
# FIX: Every line of production code should be driven by a failing test
# If no test needs it, you don't need it (YAGNI)

# ANTI-PATTERN 6: Not running tests after refactoring
# FIX: The refactor step MUST end with all tests green
# Run: pytest --tb=short after every change
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 05: Test-Driven Development"
echo "==========================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
